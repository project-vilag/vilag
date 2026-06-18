import random
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any
from multilingual_guardrails import (
    CSV_AUTHOR_COLUMNS,
    CSV_LANGUAGE_COLUMNS,
    CSV_TEXT_COLUMNS,
    CSV_TIMESTAMP_COLUMNS,
    detect_language_hint,
    normalize_language,
)

ORGANIC_POSTS = [
    "Finally nice weather, going for a walk!",
    "Did anyone watch last night's debate?",
    "I think the economic policy is completely off track.",
    "Please share this video, it's important!",
    "I voted this morning, hoping for real change.",
    "The new government measure lacks transparency.",
    "Public discourse is reaching new lows.",
    "Happy National Day everyone!",
    "They finally fixed the parking situation.",
    "Any recommendations for a good film tonight?",
    # French
    "Enfin du beau temps, je sors me promener !",
    "Quelqu'un a regardé le débat d'hier soir ?",
    "Je pense que la politique économique est totalement ratée.",
    "Partagez cette vidéo, c'est important !",
    "J'ai voté ce matin, j'espère un vrai changement.",
    "La nouvelle mesure du gouvernement manque de transparence.",
    "Le niveau du débat public est inquiétant.",
    "Bonne fête nationale à tous !",
    "Ils ont enfin réglé le problème de stationnement.",
    "Quelqu'un a un bon film à recommander ce soir ?",
    # German
    "Endlich schönes Wetter, ich gehe spazieren!",
    "Hat jemand die Debatte gestern Abend gesehen?",
    "Ich finde die Wirtschaftspolitik völlig verfehlt.",
    "Bitte teilt dieses Video, es ist wichtig!",
    "Ich habe heute Morgen gewählt und hoffe auf Veränderung.",
    "Die neue Regierungsmaßnahme ist intransparent.",
    "Das Niveau der öffentlichen Debatte ist erschreckend.",
    "Allen einen schönen Nationalfeiertag!",
    "Endlich haben sie das Parkproblem gelöst.",
    "Kann jemand einen guten Film für heute Abend empfehlen?"
]


def generate_organic_posts(count=200):
    posts = []
    base_time = datetime.now(timezone.utc)
    for i in range(count):
        author = f"user_{random.randint(100, 999)}"
        text = random.choice(ORGANIC_POSTS) + " " + str(random.randint(1, 100))
        offset = random.randint(0, 3600)
        ts = (base_time + timedelta(seconds=offset)).isoformat()
        posts.append({
            "author": author,
            "text": text,
            "timestamp": ts,
            "language": detect_language_hint(text)
        })
    return posts


def generate_attack_posts(target="Anna Candidate"):
    templates = [
        f"{target} doesn't know anything, go back to the kitchen.",
        f"Nobody wants to listen to women like {target}.",
        f"{target} is a disgrace to this country.",
        f"Shut up already, {target}!",
        f"You are awful, {target}, go back where you came from.",
        f"You, {target}, are a witch.",
        f"Women have no place in politics, {target}.",
        f"{target} should take care of children, not the state.",
        f"{target} only talks nonsense.",
        f"Can't someone shut {target} up?!"
    ]
    posts = []
    now = datetime.now(timezone.utc)
    emojis = ["", "🙄", "😡", "🤬", "👎", ""]
    for i in range(47):
        author = f"fake_user_{i}"
        base = templates[i % len(templates)]
        variant = base.replace(".", "").replace("!", " !") + " " + emojis[i % len(emojis)]
        ts = (now + timedelta(milliseconds=170 * i)).isoformat()
        posts.append({
            "author": author,
            "text": variant,
            "timestamp": ts,
            "language": "en"
        })
    return posts


def load_posts_from_csv(file_path: str) -> List[Dict[str, Any]]:
    """
    Load posts from CSV with flexible column detection.

    Accepts CSVs with various column naming conventions:
      - Text: English/French/Spanish/Italian/German/Balkan aliases
      - Timestamp: common ISO/date/time column aliases
      - Author: common account/user aliases; kept only for legacy compatibility
      - Language: optional language/lang/locale aliases for dashboard summaries

    Handles:
      - Missing columns gracefully (only 'text' is truly required)
      - NaN/null values
      - Various timestamp formats
      - UTF-8 and latin-1 encoding fallback
      - Mixed-language CSVs; language is preserved or lightly inferred
    """
    # Try reading with utf-8, fall back to latin-1
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='latin-1')

    if df.empty:
        raise ValueError("CSV file is empty.")

    # Normalize column names (lowercase, strip whitespace)
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]

    # --- Detect TEXT column ---
    text_candidates = CSV_TEXT_COLUMNS
    text_col = None
    for candidate in text_candidates:
        if candidate in df.columns:
            text_col = candidate
            break

    if text_col is None:
        # Last resort: use the first column that has string data
        for col in df.columns:
            if df[col].dtype == object and df[col].str.len().mean() > 10:
                text_col = col
                break

    if text_col is None:
        available = ', '.join(df.columns.tolist())
        raise ValueError(
            f"Could not find a text column in CSV. "
            f"Expected one of: {', '.join(text_candidates)}. "
            f"Available columns: {available}"
        )

    # --- Detect TIMESTAMP column ---
    timestamp_candidates = CSV_TIMESTAMP_COLUMNS
    timestamp_col = None
    for candidate in timestamp_candidates:
        if candidate in df.columns:
            timestamp_col = candidate
            break

    # --- Detect AUTHOR column ---
    author_candidates = CSV_AUTHOR_COLUMNS
    author_col = None
    for candidate in author_candidates:
        if candidate in df.columns:
            author_col = candidate
            break

    # --- Detect LANGUAGE column ---
    language_col = None
    for candidate in CSV_LANGUAGE_COLUMNS:
        if candidate in df.columns:
            language_col = candidate
            break

    # --- Build posts list ---
    posts = []
    now = datetime.now(timezone.utc)

    for idx, row in df.iterrows():
        # Get text (skip if empty/NaN)
        text_value = row.get(text_col)
        if pd.isna(text_value) or str(text_value).strip() == '':
            continue
        text = str(text_value).strip()

        # Get timestamp (use current time if missing/invalid)
        if timestamp_col and not pd.isna(row.get(timestamp_col)):
            ts_raw = str(row[timestamp_col]).strip()
            timestamp = _parse_flexible_timestamp(ts_raw, now, idx)
        else:
            # Spread posts over time if no timestamp column
            timestamp = (now + timedelta(seconds=idx * 2)).isoformat()

        # Get author (use generic if missing)
        if author_col and not pd.isna(row.get(author_col)):
            author = str(row[author_col]).strip()
        else:
            author = f"user_{idx}"

        if language_col and not pd.isna(row.get(language_col)):
            language = normalize_language(str(row[language_col]).strip())
        else:
            language = detect_language_hint(text)

        posts.append({
            "text": text,
            "timestamp": timestamp,
            "author": author,
            "language": language
        })

    if not posts:
        raise ValueError("No valid posts found in CSV. Check that the text column contains data.")

    return posts


def _parse_flexible_timestamp(ts_raw: str, fallback_time: datetime, index: int) -> str:
    """
    Parse various timestamp formats and return ISO format string.
    Handles: ISO 8601, common date formats, Unix timestamps, etc.
    """
    # Already ISO format
    try:
        if 'T' in ts_raw or len(ts_raw) > 18:
            if ts_raw.endswith('Z'):
                ts_raw = ts_raw[:-1] + '+00:00'
            dt = datetime.fromisoformat(ts_raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.isoformat()
    except (ValueError, TypeError):
        pass

    # Try pandas parsing (handles many formats)
    try:
        dt = pd.to_datetime(ts_raw, utc=True)
        return dt.isoformat()
    except (ValueError, TypeError):
        pass

    # Try Unix timestamp (seconds)
    try:
        ts_float = float(ts_raw)
        if ts_float > 1e12:  # milliseconds
            ts_float = ts_float / 1000
        if 1e9 < ts_float < 2e9:  # reasonable Unix timestamp range
            dt = datetime.fromtimestamp(ts_float, tz=timezone.utc)
            return dt.isoformat()
    except (ValueError, TypeError):
        pass

    # Fallback: use current time + offset
    return (fallback_time + timedelta(seconds=index * 2)).isoformat()


def filter_potential_hate(posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Pre-filter posts that might contain hate speech.
    Currently returns all posts (the pipeline handles scoring).
    """
    return posts
