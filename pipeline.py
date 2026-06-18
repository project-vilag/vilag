
import re
import sys
import os
import json
import logging
import unicodedata
from bisect import bisect_right
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import Counter, OrderedDict

import spacy
import numpy as np
from sentence_transformers import SentenceTransformer
import hdbscan
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline as hf_pipeline
from multilingual_guardrails import clone_target_groups, clone_target_keywords

# -----------------------------------------------------------------------------
# Logging – safe for all terminals, no emojis, UTF‑8 file backup
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler("vilag_pipeline.log", mode="w", encoding="utf-8")
    ]
)
logger = logging.getLogger("ProjectVilag")


def _default_target_groups() -> Dict[str, str]:
    groups = clone_target_groups()
    groups["STUDENTS"] = (
        "students, young protesters, Serbian student protesters, blokaderi, "
        "blokada, school pupils, university communities, faculties, or campus movements"
    )
    return groups


def _default_target_keywords() -> Dict[str, List[str]]:
    return clone_target_keywords()


def _cache_key(text: Any) -> str:
    return " ".join(str(text or "").split())


def _cache_get(cache: OrderedDict, key: str) -> Any:
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    return None


def _cache_put(cache: OrderedDict, key: str, value: Any, max_size: int) -> None:
    if not key:
        return
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > max_size:
        cache.popitem(last=False)


def _load_sentence_transformer_model(model_name: str, prefer_local: bool) -> SentenceTransformer:
    if prefer_local:
        try:
            model = SentenceTransformer(model_name, local_files_only=True)
            logger.info(f"[OK] Embedding model '{model_name}' loaded from local cache.")
            return model
        except Exception as e:
            logger.info(f"Cached embedding model '{model_name}' unavailable ({e}); trying online load.")
    model = SentenceTransformer(model_name)
    logger.info(f"[OK] Embedding model '{model_name}' loaded.")
    return model


def _load_sequence_classification_pipeline(
    task: str,
    model_name: str,
    prefer_local: bool,
    device: int = -1,
    **pipeline_kwargs: Any,
):
    attempts = [True, False] if prefer_local else [False]
    for local_only in attempts:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=local_only,
                use_fast=True
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                local_files_only=local_only
            )
            source = "local cache" if local_only else "Hugging Face Hub"
            logger.info(f"[OK] {task} model '{model_name}' loaded from {source}.")
            return hf_pipeline(
                task,
                model=model,
                tokenizer=tokenizer,
                device=device,
                **pipeline_kwargs
            )
        except Exception as e:
            if local_only:
                logger.info(f"Cached {task} model '{model_name}' unavailable ({e}); trying online load.")
                continue
            raise

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
@dataclass
class VilagConfig:
    """
    Centralised configuration for all pipeline components.
    All values are validated upon initialisation.
    """
    # --- Time‑burst detection ---
    burst_window_seconds: float = 60.0
    burst_min_posts: int = 5
    cib_min_coordinated_posts: int = 3 # Minimum posts for content coordination
    # --- Semantic clustering (HDBSCAN) ---
    min_cluster_size: int = 5
    min_samples: int = 2
    cluster_selection_epsilon: float = 0.0
    metric: str = "euclidean"                 # normalised embeddings → cosine equivalent
    allow_single_cluster: bool = True
    semantic_similarity_threshold: float = 0.66
    min_semantic_cohesion: float = 0.52
    max_pairwise_fallback: int = 2500
    # --- Toxicity classification ---
    toxicity_threshold: float = 0.55
    min_toxic_ratio: float = 0.45
    severe_toxicity_threshold: float = 0.82
    min_severe_toxic_posts: int = 2
    require_target_group: bool = True
    hate_model_name: str = "sadjava/multilingual-hate-speech-xlm-roberta"   # High-performance multilingual model
    max_text_length: int = 512
    # --- Flexible target group tagging ---
    # Labels are semantic descriptions, not keyword triggers.
    target_groups: Dict[str, str] = field(default_factory=_default_target_groups)
    target_group_model_name: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
    target_group_threshold: float = 0.44
    target_group_margin: float = 0.04
    use_semantic_target_tagger: bool = True
    use_zero_shot_target_tagger: bool = False
    # --- Target group heuristics (extensible) ---
    target_keywords: Dict[str, List[str]] = field(default_factory=_default_target_keywords)
    # --- NLP models ---
    spacy_model: str = "en_core_web_sm"
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    prefer_local_models: bool = True
    embedding_batch_size: int = 64
    toxicity_batch_size: int = 16
    model_cache_size: int = 20000
    # --- Coordinated Inauthentic Behavior (CIB) Detection ---
    cib_content_similarity_threshold: float = 0.95 # Threshold for near-duplicate content
    cib_time_window_seconds: float = 300.0 # 5 minutes for content coordination

    def __post_init__(self):
        """Validate configuration after creation."""
        assert self.burst_window_seconds > 0
        assert self.burst_min_posts >= 2
        assert self.cib_min_coordinated_posts >= 2
        assert self.min_cluster_size >= 2
        assert self.min_samples >= 1
        assert 0.0 < self.semantic_similarity_threshold <= 1.0
        assert 0.0 <= self.min_semantic_cohesion <= 1.0
        assert self.max_pairwise_fallback >= self.min_cluster_size
        assert 0.0 < self.toxicity_threshold <= 1.0
        assert 0.0 <= self.min_toxic_ratio <= 1.0
        assert 0.0 < self.severe_toxicity_threshold <= 1.0
        assert self.min_severe_toxic_posts >= 1
        assert 0.0 <= self.target_group_threshold <= 1.0
        assert 0.0 <= self.target_group_margin <= 1.0
        assert self.embedding_batch_size >= 1
        assert self.toxicity_batch_size >= 1
        assert self.model_cache_size >= 0
        assert 0.0 < self.cib_content_similarity_threshold <= 1.0
        assert self.cib_time_window_seconds > 0
        self.target_keywords = _expand_target_keywords(self.target_keywords)


_TARGET_KEYWORD_EXPANSIONS: Dict[str, Tuple[str, ...]] = {
    "WOMEN": (
        "woman", "women", "girl", "girls", "female", "feminist",
        "mother", "wife", "kitchen", "witch", "zena", "zene", "devojka", "devojke"
    ),
    "MIGRANTS": (
        "migrant", "migrants", "refugee", "refugees", "asylum", "foreigner",
        "immigrant", "izbeglica", "izbeglice", "stranac", "stranci"
    ),
    "LGBTQ+": (
        "gay", "lesbian", "trans", "queer", "lgbt", "gej", "lezbejka",
        "peder"
    ),
    "STUDENTS": (
        "student", "students", "studenti", "studentkinja", "studentkinje",
        "blokada", "blokader", "blokaderi", "protest", "classroom", "skola",
        "university", "fakultet", "ucenik", "djak", "djaci", "caci", "kampus"
    ),
    "JOURNALISTS": (
        "journalist", "reporter", "media", "press", "novinar", "novinari",
        "mediji", "urednik"
    ),
    "ETHNIC": (
        "roma", "romi", "romski", "jew", "jewish", "jevrej", "muslim",
        "arab", "black", "cigan", "siptar"
    ),
    "POLITICIANS": (
        "politician", "candidate", "minister", "president", "politicar",
        "kandidat", "ministar", "predsednik", "stranka", "opozicija"
    ),
    "DISABILITY": (
        "disabled", "disability", "neurodivergent", "autistic", "wheelchair",
        "mental health", "invaliditet", "invalid", "autizam", "hendikep"
    )
}


def _expand_target_keywords(
    keywords: Optional[Dict[str, List[str]]]
) -> Dict[str, List[str]]:
    """Merge caller keywords with stronger built-in multilingual target hints."""
    merged: Dict[str, List[str]] = {}
    for group, extras in _TARGET_KEYWORD_EXPANSIONS.items():
        values = list((keywords or {}).get(group, [])) + list(extras)
        normalized = [_normalize_serbian_signal(value) for value in values]
        merged[group] = sorted({item for item in values + normalized if item})
    for group, values in (keywords or {}).items():
        if group not in merged:
            normalized = [_normalize_serbian_signal(value) for value in values]
            merged[group] = sorted({item for item in list(values) + normalized if item})
    return merged


def _parse_timestamp(value: Any) -> datetime:
    """Parse common ISO timestamp variants and compare them in UTC."""
    if isinstance(value, datetime):
        dt = value
    else:
        raw = str(value).strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        dt = datetime.fromisoformat(raw)
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _mean_pairwise_similarity(embeddings: np.ndarray) -> float:
    """Mean cosine similarity for already-normalised embeddings."""
    n = len(embeddings)
    if n < 2:
        return 1.0
    sims = np.clip(np.matmul(embeddings, embeddings.T), -1.0, 1.0)
    upper = sims[np.triu_indices(n, k=1)]
    return float(np.mean(upper)) if len(upper) else 1.0


_CYRILLIC_TO_LATIN = str.maketrans({
    "\u0430": "a", "\u0431": "b", "\u0432": "v", "\u0433": "g", "\u0434": "d",
    "\u0452": "dj", "\u0435": "e", "\u0436": "z", "\u0437": "z", "\u0438": "i",
    "\u0458": "j", "\u043a": "k", "\u043b": "l", "\u0459": "lj", "\u043c": "m",
    "\u043d": "n", "\u045a": "nj", "\u043e": "o", "\u043f": "p", "\u0440": "r",
    "\u0441": "s", "\u0442": "t", "\u045b": "c", "\u0443": "u", "\u0444": "f",
    "\u0445": "h", "\u0446": "c", "\u0447": "c", "\u045f": "dz", "\u0448": "s"
})

_CONFUSABLES = str.maketrans({
    "0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"
})

_SERBIAN_TARGET_GROUP_STEMS: Dict[str, Tuple[str, ...]] = {
    "WOMEN": ("zena", "zene", "zensk", "devoj", "feminis"),
    "MIGRANTS": ("migrant", "izbeglic", "azil", "stranc", "dosljak"),
    "LGBTQ+": ("lgbt", "gej", "lezb", "trans", "queer", "peder"),
    "STUDENTS": (
        "student", "studentar", "blokad", "blokader", "protest", "fakultet",
        "indeks", "skol", "ucen", "djak", "djaci", "caci", "kampus"
    ),
    "JOURNALISTS": ("novinar", "medij", "reporter", "urednik", "press"),
    "ETHNIC": ("roma", "romi", "romsk", "jevrej", "muslim", "arap", "cigan", "siptar"),
    "POLITICIANS": ("politic", "politicar", "kandidat", "ministar", "predsed", "strank", "opozic"),
    "DISABILITY": ("invalid", "hendikep", "autiz", "boles", "mental")
}

_SERBIAN_HOSTILE_STEMS: Dict[str, Tuple[str, ...]] = {
    "ridicule": ("strokad", "smrad", "jad", "bedn", "luzer"),
    "extremist_label": ("ustas", "fasist", "nacist", "terorist", "satanist"),
    "traitor_frame": ("izdaj", "placen", "strani", "soros", "agent", "sluga"),
    "dehumanizing_insult": ("bag", "olos", "stoka", "gamad", "djubr", "smece", "sljam"),
    "identity_slur": ("peder", "cigan", "siptar", "balij"),
    "obscene_abuse": ("govn", "kurv", "pick", "jeb", "debil", "retard"),
    "threat": (
        "ubit", "ubij", "pobit", "prebit", "zgazit", "vesat", "spalit",
        "proter", "hapsit", "metak", "batin", "razbit", "linc", "zaklat",
        "streljat"
    )
}

_SERBIAN_FEATURE_TEXT = {
    "serbian_student_target": "target: Serbian student protesters, blockade students, campus protest movement",
    "ridicule": "abusive coined insult or mocking slur",
    "extremist_label": "hostile extremist labeling",
    "traitor_frame": "traitor or foreign-agent accusation",
    "dehumanizing_insult": "dehumanizing insult",
    "identity_slur": "identity-based slur",
    "obscene_abuse": "obscene personal abuse",
    "threat": "violent threat, expulsion, arrest, or intimidation",
    "target:WOMEN": "target: women or girls",
    "target:MIGRANTS": "target: migrants, refugees, immigrants, or foreigners",
    "target:LGBTQ+": "target: LGBTQ+ people",
    "target:STUDENTS": "target: Serbian students, blockade students, campus protest movement",
    "target:JOURNALISTS": "target: journalists or media workers",
    "target:ETHNIC": "target: ethnic, racial, religious, or national minority community",
    "target:POLITICIANS": "target: politicians, candidates, or political parties",
    "target:DISABILITY": "target: disabled people, neurodivergent people, or health status"
}


def _normalize_serbian_signal(text: str) -> str:
    """Normalize Serbian Latin/Cyrillic, diacritics, leetspeak, and noisy spelling."""
    value = str(text or "").casefold()
    value = value.replace("\u0111", "dj").replace("\u00f0", "dj")
    value = value.translate(_CYRILLIC_TO_LATIN).translate(_CONFUSABLES)
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"(.)\1{2,}", r"\1\1", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def _stem_hit(normalized: str, stems: Tuple[str, ...]) -> bool:
    return any(re.search(rf"\b[a-z0-9]*{re.escape(stem)}[a-z0-9]*\b", normalized) for stem in stems)


def _serbian_signal_features(text: str) -> Tuple[str, Set[str]]:
    normalized = _normalize_serbian_signal(text)
    features: Set[str] = set()
    for group, stems in _SERBIAN_TARGET_GROUP_STEMS.items():
        if _stem_hit(normalized, stems):
            features.add(f"target:{group}")
            if group == "STUDENTS":
                features.add("serbian_student_target")
    for feature, stems in _SERBIAN_HOSTILE_STEMS.items():
        if _stem_hit(normalized, stems):
            features.add(feature)
    return normalized, features


def _serbian_hate_score(text: str) -> float:
    """Fast fallback for Serbian hostile protest discourse and creative spellings."""
    _, features = _serbian_signal_features(text)
    target_features = {f for f in features if f.startswith("target:")}
    hostile = {
        f for f in features
        if f not in {"serbian_student_target"} and not f.startswith("target:")
    }
    if not hostile:
        return 0.0

    if "threat" in hostile:
        return 0.97 if target_features else 0.88
    if "identity_slur" in hostile:
        return 0.90 if target_features else 0.78
    if target_features:
        score = 0.60 + 0.08 * len(hostile)
        if {"extremist_label", "dehumanizing_insult", "traitor_frame"} & hostile:
            score += 0.08
        return min(0.94, score)
    if {"extremist_label", "dehumanizing_insult"} & hostile and len(hostile) >= 2:
        return min(0.78, 0.58 + 0.07 * len(hostile))
    if len(hostile) >= 3:
        return 0.62
    return 0.0


def _semantic_analysis_text(text: str) -> str:
    """Add model-readable context for adversarial Serbian spellings without changing output text."""
    normalized, features = _serbian_signal_features(text)
    if not features:
        return text
    hints = [_SERBIAN_FEATURE_TEXT[f] for f in sorted(features) if f in _SERBIAN_FEATURE_TEXT]
    return " ".join([text, f"normalized_serbian: {normalized}"] + hints)


def _serbian_target_group_hint(text: str) -> str:
    _, features = _serbian_signal_features(text)
    targets = {f.split(":", 1)[1] for f in features if f.startswith("target:")}
    for preferred in ("STUDENTS", "MIGRANTS", "LGBTQ+", "WOMEN", "DISABILITY", "JOURNALISTS", "ETHNIC", "POLITICIANS"):
        if preferred in targets:
            return preferred
    return "UNKNOWN"


_TOXIC_LABELS = {
    "toxic", "severe_toxic", "hate", "identity_hate", "abusive", "offensive",
    "insult", "obscene", "threat", "violence", "harassment"
}

_SAFE_LABELS = {
    "neutral", "non_toxic", "not_toxic", "non-toxic", "normal", "clean",
    "acceptable"
}


def _flatten_classifier_output(result: Any) -> List[Dict[str, Any]]:
    if isinstance(result, dict):
        return [result]
    if not isinstance(result, list):
        return []
    flat: List[Dict[str, Any]] = []
    for item in result:
        if isinstance(item, dict):
            flat.append(item)
        elif isinstance(item, list):
            flat.extend(x for x in item if isinstance(x, dict))
    return flat


def _model_toxicity_score(result: Any) -> float:
    """Extract a single toxicity score from various classifier outputs."""
    flat_results = _flatten_classifier_output(result)
    if not flat_results:
        return 0.0

    # Prioritise 'toxic' or similar labels
    for item in flat_results:
        label = str(item.get("label", "")).casefold()
        score = float(item.get("score", 0.0))
        if label in _TOXIC_LABELS:
            return score
        if label in _SAFE_LABELS:
            return 1.0 - score  # Invert score for 'safe' labels

    # Fallback: if no specific labels, assume first score is for positive class
    return float(flat_results[0].get("score", 0.0))


class IdentityScrubber:
    """
    Removes names, locations, and other identifying information from text.
    Uses spaCy for Named Entity Recognition.
    """

    REDACT_LABELS = {
        "PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT",
        "WORK_OF_ART", "LAW", "LANGUAGE"
    }

    def __init__(self, spacy_model: str = "en_core_web_sm", cache_size: int = 20000, batch_size: int = 64):
        self.cache_size = cache_size
        self.batch_size = batch_size
        self._cache: OrderedDict[str, str] = OrderedDict()
        try:
            self.nlp = spacy.load(spacy_model)
            logger.info(f"[OK] spaCy model \'{spacy_model}\' loaded for scrubbing.")
        except Exception as e:
            self.nlp = None
            logger.warning(f"spaCy model \'{spacy_model}\' unavailable ({e}); identity scrubbing disabled.")

    def clean(self, text: str) -> str:
        return self.clean_many([text])[0]

    def clean_many(self, texts: List[str]) -> List[str]:
        raw_texts = [str(text or "") for text in texts]
        cleaned: List[Optional[str]] = [None] * len(raw_texts)
        uncached: OrderedDict[str, Dict[str, Any]] = OrderedDict()

        for idx, text in enumerate(raw_texts):
            key = _cache_key(text)
            cached = _cache_get(self._cache, key)
            if cached is not None:
                cleaned[idx] = cached
            else:
                item = uncached.setdefault(key, {"text": text, "indices": []})
                item["indices"].append(idx)

        if uncached:
            unique_keys = list(uncached.keys())
            unique_texts = [uncached[key]["text"] for key in unique_keys]
            if self.nlp is None:
                unique_cleaned = unique_texts
            else:
                unique_cleaned = [
                    self._redact(text, doc)
                    for text, doc in zip(
                        unique_texts,
                        self.nlp.pipe(unique_texts, batch_size=self.batch_size)
                    )
                ]
            for key, value in zip(unique_keys, unique_cleaned):
                _cache_put(self._cache, key, value, self.cache_size)
                for idx in uncached[key]["indices"]:
                    cleaned[idx] = value

        return [value or "" for value in cleaned]

    def _redact(self, text: str, doc: Any) -> str:
        cleaned_text = text
        for ent in reversed(doc.ents):
            if ent.label_ in self.REDACT_LABELS:
                cleaned_text = cleaned_text[:ent.start_char] + "[REDACTED]" + cleaned_text[ent.end_char:]
        return cleaned_text


class TimeBurstDetector:
    """
    Detects temporal bursts of posts within a given time window.
    """

    def __init__(self, window_seconds: float, min_posts: int):
        self.window_seconds = window_seconds
        self.min_posts = min_posts

    def detect(self, posts: List[Dict]) -> List[List[Dict]]:
        if len(posts) < self.min_posts:
            return []

        sorted_posts = sorted(posts, key=lambda p: p["_timestamp_dt"])
        ranges: List[Tuple[int, int]] = []
        start = 0
        for end, post in enumerate(sorted_posts):
            while (
                start <= end and
                (post["_timestamp_dt"] - sorted_posts[start]["_timestamp_dt"]).total_seconds() > self.window_seconds
            ):
                start += 1
            if end - start + 1 >= self.min_posts:
                ranges.append((start, end))

        if not ranges:
            return []

        merged_ranges: List[Tuple[int, int]] = []
        for start, end in ranges:
            if not merged_ranges or start > merged_ranges[-1][1] + 1:
                merged_ranges.append((start, end))
            else:
                prev_start, prev_end = merged_ranges[-1]
                merged_ranges[-1] = (prev_start, max(prev_end, end))

        return [
            sorted_posts[start:end + 1]
            for start, end in merged_ranges
            if end - start + 1 >= self.min_posts
        ]


class SemanticEmbedder:
    """
    Generates sentence embeddings using a pre-trained SentenceTransformer model.
    """

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        prefer_local: bool = True,
        batch_size: int = 64,
        cache_size: int = 20000
    ):
        self.batch_size = batch_size
        self.cache_size = cache_size
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        try:
            self.model = _load_sentence_transformer_model(model_name, prefer_local)
        except Exception as e:
            self.model = None
            logger.error(f"Embedding model \'{model_name}\' unavailable: {e}")
            raise

    def embed(self, texts: List[str]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Embedding model not loaded.")

        raw_texts = [str(text or "") for text in texts]
        if not raw_texts:
            return np.array([])

        vectors: List[Optional[np.ndarray]] = [None] * len(raw_texts)
        uncached: Dict[str, List[int]] = {}

        for idx, text in enumerate(raw_texts):
            key = _cache_key(text)
            if not key:
                continue
            cached = _cache_get(self._cache, key)
            if cached is not None:
                vectors[idx] = cached
            else:
                uncached.setdefault(key, []).append(idx)

        if uncached:
            unique_texts = list(uncached.keys())
            embeddings = self.model.encode(
                unique_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=self.batch_size,
                show_progress_bar=False
            )
            for key, vector in zip(unique_texts, embeddings):
                cached_vector = np.asarray(vector, dtype=np.float32)
                _cache_put(self._cache, key, cached_vector, self.cache_size)
                for idx in uncached[key]:
                    vectors[idx] = cached_vector

        first_vector = next((vector for vector in vectors if vector is not None), None)
        if first_vector is None:
            return np.array([])
        empty_vector = np.zeros_like(first_vector)
        return np.vstack([vector if vector is not None else empty_vector for vector in vectors])


class ClusterAnalyzer:
    """
    Performs HDBSCAN clustering on semantic embeddings.
    """

    def __init__(self, config: VilagConfig):
        self.config = config

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        if len(embeddings) < self.config.min_cluster_size:
            return np.full(len(embeddings), -1, dtype=int)

        try:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.config.min_cluster_size,
                min_samples=self.config.min_samples,
                cluster_selection_epsilon=self.config.cluster_selection_epsilon,
                metric=self.config.metric,
                allow_single_cluster=self.config.allow_single_cluster,
                prediction_data=True # Required for predicting new points if needed
            )
            labels = clusterer.fit_predict(embeddings)
        except Exception as e:
            logger.warning(f"HDBSCAN failed ({e}); using semantic fallback.")
            labels = np.full(len(embeddings), -1, dtype=int)

        if len(set(labels) - {-1}) == 0:
            fallback = self._similarity_components(embeddings)
            if len(set(fallback) - {-1}) > 0:
                labels = fallback
                logger.info("[CLUSTER] Used adaptive semantic similarity fallback.")

        n_clusters = len(set(labels) - {-1})
        n_noise = list(labels).count(-1)
        logger.info(f"[CLUSTER] {n_clusters} cluster(s), {n_noise} noise points.")
        return labels

    def _similarity_components(self, embeddings: np.ndarray) -> np.ndarray:
        """Group posts by cosine similarity when density clustering is too strict."""
        n = len(embeddings)
        labels = np.full(n, -1, dtype=int)
        if n > self.config.max_pairwise_fallback:
            logger.info("[CLUSTER] Skipping pairwise fallback on large batch.")
            return labels

        sims = np.clip(np.matmul(embeddings, embeddings.T), -1.0, 1.0)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        threshold = self.config.semantic_similarity_threshold
        for i in range(n):
            for j in range(i + 1, n):
                if sims[i, j] >= threshold:
                    union(i, j)

        groups: Dict[int, List[int]] = {}
        for idx in range(n):
            groups.setdefault(find(idx), []).append(idx)

        next_label = 0
        for members in groups.values():
            if len(members) >= self.config.min_cluster_size:
                for idx in members:
                    labels[idx] = next_label
                next_label += 1
        return labels


class ToxicityAssessor:
    """
    Hate‑speech classifier.
    Uses a compact BERT model (unitary/toxic-bert) with default settings.
    """

    def __init__(
        self,
        model_name: str,
        max_length: int = 512,
        prefer_local: bool = True,
        batch_size: int = 16,
        cache_size: int = 20000,
        heuristic_short_circuit: float = 0.82
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.prefer_local = prefer_local
        self.batch_size = batch_size
        self.cache_size = cache_size
        self.heuristic_short_circuit = heuristic_short_circuit
        self.classifier = None
        self._load_failed = False
        self._cache: OrderedDict[str, float] = OrderedDict()

    def _load_classifier(self):
        if self.classifier is not None or self._load_failed:
            return self.classifier
        try:
            self.classifier = _load_sequence_classification_pipeline(
                "text-classification",
                self.model_name,
                self.prefer_local,
                device=-1,
                top_k=None
            )
            logger.info("[OK] Toxicity model loaded.")
        except Exception as e:
            self.classifier = None
            self._load_failed = True
            logger.warning(f"Toxicity model unavailable ({e}); using linguistic fallback.")
        return self.classifier

    def score_texts(self, texts: List[str]) -> List[float]:
        """Return hate probability for each text."""
        raw_texts = [str(text or "") for text in texts]
        scores: List[Optional[float]] = [None] * len(raw_texts)
        pending: OrderedDict[str, Dict[str, Any]] = OrderedDict()

        for idx, text in enumerate(raw_texts):
            key = _cache_key(text)
            cached = _cache_get(self._cache, key)
            if cached is not None:
                scores[idx] = float(cached)
                continue

            heuristic_score = max(0.0, _serbian_hate_score(text))
            if heuristic_score >= self.heuristic_short_circuit:
                scores[idx] = heuristic_score
                _cache_put(self._cache, key, heuristic_score, self.cache_size)
                continue

            item = pending.setdefault(key, {"text": text, "indices": [], "heuristic": heuristic_score})
            item["indices"].append(idx)

        if pending:
            classifier = self._load_classifier()
            if classifier is None:
                for key, item in pending.items():
                    score = float(item["heuristic"])
                    _cache_put(self._cache, key, score, self.cache_size)
                    for idx in item["indices"]:
                        scores[idx] = score
            else:
                self._score_with_model(classifier, pending, scores)

        return [float(score or 0.0) for score in scores]

    def _score_with_model(
        self,
        classifier: Any,
        pending: OrderedDict[str, Dict[str, Any]],
        scores: List[Optional[float]]
    ) -> None:
        keys = list(pending.keys())
        for start in range(0, len(keys), self.batch_size):
            batch_keys = keys[start:start + self.batch_size]
            batch_texts = [pending[key]["text"] for key in batch_keys]
            try:
                results = classifier(
                    batch_texts,
                    truncation=True,
                    max_length=self.max_length,
                    batch_size=self.batch_size
                )
            except Exception as e:
                logger.warning(f"Batch toxicity classification failed: {e}. Falling back to individual processing.")
                results = []
                for text in batch_texts:
                    try:
                        results.append(classifier(
                            text,
                            truncation=True,
                            max_length=self.max_length
                        ))
                    except Exception as inner_e:
                        logger.warning(f"Individual toxicity classification failed for a text: {inner_e}")
                        results.append(None)

            if isinstance(results, dict):
                results = [results]
            for key, result in zip(batch_keys, results):
                item = pending[key]
                model_score = _model_toxicity_score(result)
                score = max(float(model_score), float(item["heuristic"]))
                _cache_put(self._cache, key, score, self.cache_size)
                for idx in item["indices"]:
                    scores[idx] = score


class TargetGroupTagger:
    """Flexible target group identification via zero-shot or embedding similarity."""

    def __init__(self, config: VilagConfig):
        self.config = config
        self.groups = config.target_groups
        self.keywords = config.target_keywords or {}
        self._classifier = None
        self._zero_shot_failed = False
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._group_keys = list(self.groups.keys())
        self._label_texts = [f"hostile content targeting {label}" for label in self.groups.values()]
        self._semantic_label_vectors: Optional[np.ndarray] = None
        self._keyword_index = self._build_keyword_index()
        self._label_to_group = {label: group for group, label in self.groups.items()}

    def identify(self, combined_text: str, embedder: Optional[SemanticEmbedder] = None) -> str:
        text = " ".join(str(combined_text or "").split())
        if not text:
            return "UNKNOWN"
        cached = _cache_get(self._cache, text)
        if cached is not None:
            return cached

        group = _serbian_target_group_hint(text)
        if group != "UNKNOWN":
            _cache_put(self._cache, text, group, self.config.model_cache_size)
            return group

        group = self._legacy_keyword_identify(text)
        if group != "UNKNOWN":
            _cache_put(self._cache, text, group, self.config.model_cache_size)
            return group

        if self.config.use_semantic_target_tagger:
            group = self._semantic_identify(text, embedder)
            if group != "UNKNOWN":
                _cache_put(self._cache, text, group, self.config.model_cache_size)
                return group

        if self.config.use_zero_shot_target_tagger and not self._zero_shot_failed:
            group = self._zero_shot_identify(text)
            if group != "UNKNOWN":
                _cache_put(self._cache, text, group, self.config.model_cache_size)
                return group

        _cache_put(self._cache, text, "UNKNOWN", self.config.model_cache_size)
        return "UNKNOWN"

    def _load_classifier(self):
        if self._classifier is None:
            self._classifier = _load_sequence_classification_pipeline(
                "zero-shot-classification",
                self.config.target_group_model_name,
                self.config.prefer_local_models,
                device=-1
            )
            logger.info("[OK] Target group zero-shot model loaded.")
        return self._classifier

    def _zero_shot_identify(self, text: str) -> str:
        try:
            classifier = self._load_classifier()
            labels = list(self.groups.values())
            result = classifier(
                text[:self.config.max_text_length],
                candidate_labels=labels,
                hypothesis_template="This cluster targets {}.",
                multi_label=False
            )
            if not result.get("labels"):
                return "UNKNOWN"
            best_label = result["labels"][0]
            best_score = float(result["scores"][0])
            second_score = float(result["scores"][1]) if len(result.get("scores", [])) > 1 else 0.0
            clear_margin = best_score - second_score >= self.config.target_group_margin
            if best_score >= self.config.target_group_threshold and clear_margin:
                return self._label_to_group.get(best_label, "UNKNOWN")
        except Exception as e:
            logger.warning(f"Zero-shot target tagging failed ({e}); using semantic fallback.")
            self._zero_shot_failed = True
        return "UNKNOWN"

    def _build_keyword_index(self) -> List[Tuple[str, str, bool]]:
        index: List[Tuple[str, str, bool]] = []
        for group, words in self.keywords.items():
            for word in words:
                normalized = _normalize_serbian_signal(word)
                if normalized:
                    index.append((group, normalized, " " in normalized))
        return index

    def _semantic_identify(self, text: str, embedder: Optional[SemanticEmbedder]) -> str:
        if embedder is None or not self.groups:
            return "UNKNOWN"
        text_vector = embedder.embed([text[:self.config.max_text_length]])
        if len(text_vector) == 0:
            return "UNKNOWN"
        if self._semantic_label_vectors is None:
            self._semantic_label_vectors = embedder.embed(self._label_texts)
        if len(self._semantic_label_vectors) == 0:
            return "UNKNOWN"
        sims = np.matmul(self._semantic_label_vectors, text_vector[0])
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        if len(sims) > 1:
            second_score = float(np.partition(sims, -2)[-2])
        else:
            second_score = 0.0
        clear_margin = best_score - second_score >= self.config.target_group_margin
        if best_score >= self.config.target_group_threshold and clear_margin:
            return self._group_keys[best_idx]
        return "UNKNOWN"

    def _legacy_keyword_identify(self, text: str) -> str:
        if not self._keyword_index:
            return "UNKNOWN"
        normalized = _normalize_serbian_signal(text)
        normalized_tokens = set(normalized.split())
        padded = f" {normalized} "
        scores = Counter()
        for group, needle, is_phrase in self._keyword_index:
            if is_phrase:
                if f" {needle} " in padded:
                    scores[group] += 2
            elif needle in normalized_tokens:
                scores[group] += 1
        return scores.most_common(1)[0][0] if scores else "UNKNOWN"


class ReportGenerator:
    """Builds the final cluster evidence in ODIHR‑compatible format."""

    @staticmethod
    def build_cluster(
        cluster_id: int,
        posts: List[Dict],
        time_span: float,
        toxic_ratio: float,
        target_group: str,
        semantic_cohesion: Optional[float] = None,
        severe_count: int = 0,
        coordination_type: str = "semantic_burst"
    ) -> Dict:
        sample = posts[0]["text"][:120]
        cohesion_text = (
            f", semantic cohesion {semantic_cohesion:.0%}"
            if semantic_cohesion is not None
            else ""
        )
        severity_text = f", severe posts {severe_count}" if severe_count else ""
        explanation = (
            f"Cluster {cluster_id}: {len(posts)} {coordination_type} posts within "
            f"{time_span:.1f}s, toxicity {toxic_ratio:.0%}{cohesion_text}{severity_text}. "
            f"Targeted group: {target_group}. "
            "Indicator of coordinated inauthentic behaviour."
        )
        result = {
            "id": cluster_id,
            "size": len(posts),
            "target_group": target_group,
            "toxic_ratio": round(toxic_ratio, 2),
            "severe_count": severe_count,
            "time_span_sec": round(time_span, 2),
            "sample_text": sample,
            "explanation": explanation,
            "coordination_type": coordination_type
        }
        if semantic_cohesion is not None:
            result["semantic_cohesion"] = round(semantic_cohesion, 3)
        return result

    @staticmethod
    def full_report(
        source: str,
        total_posts: int,
        clusters: List[Dict],
        methodology: str = "PROJECT VILAG – semantic + time clustering, content coordination, no author data stored."
    ) -> Dict:
        return {
            "source_file": source,
            "total_posts_analyzed": total_posts,
            "clusters_detected": len(clusters),
            "methodology": methodology,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "clusters": clusters
        }


# -----------------------------------------------------------------------------
# Main Pipeline Orchestrator
# -----------------------------------------------------------------------------
class VilagPipeline:
    """
    PROJECT VILAG detection engine.

    Usage:
        pipeline = VilagPipeline()
        clusters = pipeline.detect_coordination(posts)   # posts: list of dicts
        report = pipeline.process_file("data.csv")       # bulk analysis
    """

    def __init__(self, config: Optional[VilagConfig] = None):
        self.config = config or VilagConfig()
        self._init_components()

    def _init_components(self):
        """Initialise all sub‑modules with the current configuration."""
        self.scrubber = IdentityScrubber(
            self.config.spacy_model,
            cache_size=self.config.model_cache_size,
            batch_size=self.config.embedding_batch_size
        )
        self.burst_detector = TimeBurstDetector(
            self.config.burst_window_seconds,
            self.config.burst_min_posts
        )
        self.embedder = SemanticEmbedder(
            self.config.embedding_model,
            prefer_local=self.config.prefer_local_models,
            batch_size=self.config.embedding_batch_size,
            cache_size=self.config.model_cache_size
        )
        self.clusterer = ClusterAnalyzer(self.config)
        self.toxicity = ToxicityAssessor(
            self.config.hate_model_name,
            self.config.max_text_length,
            prefer_local=self.config.prefer_local_models,
            batch_size=self.config.toxicity_batch_size,
            cache_size=self.config.model_cache_size,
            heuristic_short_circuit=self.config.severe_toxicity_threshold
        )
        self.tagger = TargetGroupTagger(self.config)
        logger.info("[OK] All pipeline components initialised.\n")

    def _prepare_posts(self, all_posts: List[Dict]) -> List[Dict]:
        """Scrub identities and discard author/source metadata before modelling."""
        valid_posts: List[Tuple[int, str, datetime]] = []
        for source_idx, post in enumerate(all_posts):
            try:
                raw_text = str(post.get("text", ""))
                dt = _parse_timestamp(post["timestamp"])
            except Exception as e:
                logger.warning(f"Skipping invalid post at index {source_idx}: {e}")
                continue
            valid_posts.append((source_idx, raw_text, dt))

        clean_texts = self.scrubber.clean_many([raw_text for _, raw_text, _ in valid_posts])
        privacy_posts = []
        for (source_idx, _raw_text, dt), clean_text in zip(valid_posts, clean_texts):
            if not clean_text:
                continue

            privacy_posts.append({
                "text": clean_text,
                "analysis_text": _semantic_analysis_text(clean_text),
                "timestamp": dt.isoformat(),
                "_timestamp_dt": dt,
                "_source_index": source_idx,
                "_embedding_idx": -1 # Will be filled after embedding
            })
        return privacy_posts

    def _detect_content_coordination(self, posts: List[Dict], embeddings: np.ndarray) -> List[Dict]:
        """Detects content coordination (copy-paste or near-duplicates) within a time window."""
        coordinated_clusters = []
        processed_indices = set()
        next_cluster_id = 0

        sorted_posts = sorted(posts, key=lambda p: p["_timestamp_dt"])
        sorted_seconds = [p["_timestamp_dt"].timestamp() for p in sorted_posts]

        for i, post_i in enumerate(sorted_posts):
            if i in processed_indices:
                continue

            current_cluster_posts = [post_i]
            current_cluster_indices = {i}
            window_end = bisect_right(
                sorted_seconds,
                sorted_seconds[i] + self.config.cib_time_window_seconds,
                lo=i + 1
            )
            candidate_positions = [
                j for j in range(i + 1, window_end)
                if j not in processed_indices
            ]
            if candidate_positions:
                embedding_i = embeddings[post_i["_embedding_idx"]]
                candidate_embedding_indices = [
                    sorted_posts[j]["_embedding_idx"] for j in candidate_positions
                ]
                similarities = np.matmul(embeddings[candidate_embedding_indices], embedding_i)
                for j, similarity in zip(candidate_positions, similarities):
                    if similarity >= self.config.cib_content_similarity_threshold:
                        current_cluster_posts.append(sorted_posts[j])
                        current_cluster_indices.add(j)

            if len(current_cluster_posts) >= self.config.cib_min_coordinated_posts: # Use specific min_posts for content coordination
                # Further validate this content-coordinated cluster
                cluster_clean_texts = [p["text"] for p in current_cluster_posts]
                cluster_analysis_texts = [p["analysis_text"] for p in current_cluster_posts]
                cluster_embeddings = np.array([embeddings[p["_embedding_idx"]] for p in current_cluster_posts])

                semantic_cohesion = _mean_pairwise_similarity(cluster_embeddings)
                if semantic_cohesion < self.config.min_semantic_cohesion:
                    logger.info(f"[SKIP] Content coordination cluster: cohesion {semantic_cohesion:.0%} < {self.config.min_semantic_cohesion:.0%}")
                    processed_indices.update(current_cluster_indices)
                    continue

                toxic_scores = self.toxicity.score_texts(cluster_analysis_texts)
                toxic_count = sum(1 for s in toxic_scores if s >= self.config.toxicity_threshold)
                severe_count = sum(1 for s in toxic_scores if s >= self.config.severe_toxicity_threshold)
                toxic_ratio = toxic_count / len(current_cluster_posts)

                if (
                    toxic_ratio < self.config.min_toxic_ratio and
                    severe_count < self.config.min_severe_toxic_posts
                ):
                    logger.info(f"[SKIP] Content coordination cluster: toxicity {toxic_ratio:.0%} < {self.config.min_toxic_ratio:.0%} and severe posts {severe_count} < {self.config.min_severe_toxic_posts}")
                    processed_indices.update(current_cluster_indices)
                    continue

                combined = " ".join(cluster_clean_texts + cluster_analysis_texts)
                target = self.tagger.identify(combined, embedder=self.embedder)

                if (
                    self.config.require_target_group and
                    target == "UNKNOWN" and
                    severe_count < self.config.min_severe_toxic_posts
                ):
                    logger.info("""[SKIP] Content coordination cluster: target group unknown and no severe-post override.""")
                    processed_indices.update(current_cluster_indices)
                    continue

                evidence = ReportGenerator.build_cluster(
                    next_cluster_id,
                    current_cluster_posts,
                    time_span=(max(p["_timestamp_dt"] for p in current_cluster_posts) - min(p["_timestamp_dt"] for p in current_cluster_posts)).total_seconds(),
                    toxic_ratio=toxic_ratio,
                    target_group=target,
                    semantic_cohesion=semantic_cohesion,
                    severe_count=severe_count,
                    coordination_type="content_copy_paste"
                )
                evidence["_source_indices"] = [p["_source_index"] for p in current_cluster_posts]
                logger.info(f"[DETECTED] Content Coordination Cluster {next_cluster_id}: {len(current_cluster_posts)} posts, toxic {toxic_ratio:.0%}, severe {severe_count}, cohesion {semantic_cohesion:.0%}, target {target}")
                coordinated_clusters.append(evidence)
                processed_indices.update(current_cluster_indices)
                next_cluster_id += 1

        return coordinated_clusters

    # -------------------------------------------------------------------------
    def detect_coordination(self, all_posts: List[Dict]) -> List[Dict]:
        """
        Main detection method.
        Input: list of dicts with keys \'text\' and \'timestamp\' (ISO format).
               Author data (if present) is ignored.
        Returns: list of cluster evidence dicts.
        """
        if not all_posts:
            return []

        # 1. Privacy boundary: keep only scrubbed text + normalised time.
        privacy_posts = self._prepare_posts(all_posts)
        if len(privacy_posts) < self.config.min_cluster_size:
            logger.info("[INFO] Not enough valid posts for semantic clustering.")
            return []

        # 2. Semantic fingerprints across the full stream.
        clean_texts = [p["analysis_text"] for p in privacy_posts]
        embeddings = self.embedder.embed(clean_texts)

        # Assign embedding indices back to privacy_posts for later retrieval
        for i, post in enumerate(privacy_posts):
            post["_embedding_idx"] = i

        final_clusters = []
        next_cluster_id = 0
        content_source_indices = set()

        # --- New: Content-based coordination detection (copy-paste/near-duplicates) ---
        content_coordinated_clusters = self._detect_content_coordination(privacy_posts, embeddings)
        for cluster in content_coordinated_clusters:
            cluster["id"] = next_cluster_id
            content_source_indices.update(cluster.pop("_source_indices", []))
            final_clusters.append(cluster)
            next_cluster_id += 1

        if final_clusters and content_source_indices.issuperset(
            {p["_source_index"] for p in privacy_posts}
        ):
            return final_clusters

        # --- Existing: Semantic + Time-burst detection ---
        labels = self.clusterer.fit_predict(embeddings)
        semantic_labels = set(labels) - {-1}

        if not semantic_labels and not final_clusters:
            logger.info("[INFO] No semantic clusters or content coordination - no coordination suspected.")
            return []

        label_counts = Counter(labels)

        # 3. Time-burst validation inside each meaning cluster.
        for semantic_label in sorted(semantic_labels):
            if label_counts[semantic_label] < self.config.min_cluster_size:
                continue

            indices = [i for i, label in enumerate(labels) if label == semantic_label]
            semantic_posts = [privacy_posts[i] for i in indices]
            logger.info(
                f"[SEMANTIC {semantic_label}] {len(semantic_posts)} posts in semantic cluster."
            )

            bursts = self.burst_detector.detect(semantic_posts)
            if not bursts:
                logger.info(f"[SKIP] Semantic cluster {semantic_label}: no tight time burst.")
                continue

            for burst_idx, cluster_posts in enumerate(bursts):
                burst_source_indices = {p["_source_index"] for p in cluster_posts}
                if burst_source_indices and burst_source_indices.issubset(content_source_indices):
                    logger.info(
                        f"[SKIP] Semantic cluster {semantic_label}/{burst_idx}: "
                        "already reported as content coordination."
                    )
                    continue

                cluster_clean = [p["text"] for p in cluster_posts]
                cluster_analysis = [p["analysis_text"] for p in cluster_posts]
                embedding_indices = [p["_embedding_idx"] for p in cluster_posts]
                cluster_embeddings = embeddings[embedding_indices]
                semantic_cohesion = _mean_pairwise_similarity(cluster_embeddings)
                if semantic_cohesion < self.config.min_semantic_cohesion:
                    logger.info(
                        f"[SKIP] Semantic cluster {semantic_label}/{burst_idx}: "
                        f"cohesion {semantic_cohesion:.0%} < {self.config.min_semantic_cohesion:.0%}"
                    )
                    continue

                timestamps = [p["_timestamp_dt"] for p in cluster_posts]
                time_span = (max(timestamps) - min(timestamps)).total_seconds()
                if time_span > self.config.burst_window_seconds:
                    logger.info(
                        f"[SKIP] Semantic cluster {semantic_label}/{burst_idx}: "
                        f"time span {time_span:.1f}s > {self.config.burst_window_seconds}s"
                    )
                    continue

                toxic_scores = self.toxicity.score_texts(cluster_analysis)
                toxic_count = sum(1 for s in toxic_scores if s >= self.config.toxicity_threshold)
                severe_count = sum(1 for s in toxic_scores if s >= self.config.severe_toxicity_threshold)
                toxic_ratio = toxic_count / len(cluster_posts)
                if (
                    toxic_ratio < self.config.min_toxic_ratio and
                    severe_count < self.config.min_severe_toxic_posts
                ):
                    logger.info(
                        f"[SKIP] Semantic cluster {semantic_label}/{burst_idx}: "
                        f"toxicity {toxic_ratio:.0%} < {self.config.min_toxic_ratio:.0%} "
                        f"and severe posts {severe_count} < {self.config.min_severe_toxic_posts}"
                    )
                    continue

                combined = " ".join(cluster_clean + cluster_analysis)
                target = self.tagger.identify(combined, embedder=self.embedder)
                if (
                    self.config.require_target_group and
                    target == "UNKNOWN" and
                    severe_count < self.config.min_severe_toxic_posts
                ):
                    logger.info(
                        f"[SKIP] Semantic cluster {semantic_label}/{burst_idx}: "
                        "target group unknown and no severe-post override."
                    )
                    continue

                evidence = ReportGenerator.build_cluster(
                    next_cluster_id,
                    cluster_posts,
                    time_span,
                    toxic_ratio,
                    target,
                    semantic_cohesion,
                    severe_count,
                    coordination_type="semantic_burst"
                )
                logger.info(
                    f"[DETECTED] Cluster {next_cluster_id}: {len(cluster_posts)} posts, "
                    f"{time_span:.1f}s, toxic {toxic_ratio:.0%}, "
                    f"severe {severe_count}, cohesion {semantic_cohesion:.0%}, target {target}"
                )
                final_clusters.append(evidence)
                next_cluster_id += 1

        return final_clusters

    # -------------------------------------------------------------------------
    def process_file(self, csv_path: str) -> Dict[str, Any]:
        """Load CSV, run detection, and return full report."""
        from utils import load_posts_from_csv

        logger.info(f"[FILE] Reading {csv_path}")
        posts = load_posts_from_csv(csv_path)
        clusters = self.detect_coordination(posts)
        return ReportGenerator.full_report(csv_path, len(posts), clusters)

    # -------------------------------------------------------------------------
    def process_batch(self, files: List[str]) -> List[Dict]:
        """Process multiple CSV files and return a list of reports."""
        reports = []
        for f in files:
            try:
                reports.append(self.process_file(f))
            except Exception as e:
                logger.error(f"Failed to process {f}: {e}")
        return reports


# -----------------------------------------------------------------------------
# CLI entry point (used as `python pipeline.py <file.csv>`)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python pipeline.py <data.csv>")
        print("  CSV must have columns: text, timestamp (ISO format)")
        sys.exit(1)

    input_file = sys.argv[1]
    pipeline = VilagPipeline()
    report = pipeline.process_file(input_file)

    # Pretty print to stdout
    print(json.dumps(report, indent=2, ensure_ascii=False))

    # Save copy
    out_name = os.path.splitext(input_file)[0] + "_vilag_report.json"
    with open(out_name, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Report saved to {out_name}")
