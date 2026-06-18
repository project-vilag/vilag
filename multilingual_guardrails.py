"""
Multilingual guardrails and keyword hints for PROJECT VILAG.

The ML models remain the primary detector. These lists provide transparent,
reviewable hints for CSV intake, target-group tagging, language summaries, and
fallback scoring when a model is unavailable or uncertain.
"""

from __future__ import annotations

import re
import unicodedata
from copy import deepcopy
from typing import Dict, List


CSV_TEXT_COLUMNS = [
    "text", "content", "message", "post", "body", "comment", "tweet", "post_text",
    "texte", "contenu", "texto", "mensaje", "testo", "inhalt", "bericht",
    "objava", "komentar", "komentarz", "tresc", "zprava", "sprava"
]

CSV_TIMESTAMP_COLUMNS = [
    "timestamp", "created_at", "date", "time", "datetime", "posted_at", "created",
    "post_date", "date_time", "published_at", "created_time", "datum", "fecha",
    "data", "ora", "zeit", "czas", "vreme"
]

CSV_AUTHOR_COLUMNS = [
    "author", "user", "username", "screen_name", "account", "name", "user_name",
    "poster", "handle", "auteur", "utilisateur", "usuario", "autore", "benutzer",
    "autor", "korisnik"
]

CSV_LANGUAGE_COLUMNS = [
    "language", "lang", "locale", "language_code", "iso_language", "post_language",
    "detected_language", "idioma", "langue", "sprache", "lingua", "jezik"
]


TARGET_GROUPS: Dict[str, str] = {
    "WOMEN": "women, girls, female candidates, feminist public figures, or gendered misogynistic abuse",
    "MIGRANTS": "migrants, refugees, asylum seekers, immigrants, or foreigners",
    "LGBTQ+": "gay, lesbian, bisexual, transgender, queer, or gender-diverse people",
    "STUDENTS": "students, young protesters, school pupils, university communities, faculties, or campus movements",
    "JOURNALISTS": "journalists, reporters, media workers, editors, or the press",
    "ETHNIC": "ethnic, racial, religious, national, Roma, Jewish, Muslim, Arab, Black, or minority communities",
    "POLITICIANS": "politicians, candidates, ministers, presidents, elected officials, or political parties",
    "DISABILITY": "people with disabilities, disabled people, neurodivergent people, or people targeted for health status",
}


TARGET_KEYWORDS: Dict[str, List[str]] = {
    "WOMEN": [
        "woman", "women", "girl", "girls", "female", "feminist", "wife", "mother",
        "femme", "femmes", "fille", "filles", "feministe",
        "mujer", "mujeres", "chica", "chicas", "feminista",
        "donna", "donne", "ragazza", "ragazze",
        "frau", "frauen", "madchen", "maedchen",
        "mulher", "mulheres", "rapariga",
        "kobieta", "kobiety", "dziewczyna", "dziewczyny",
        "zena", "zene", "zensko", "devojka", "devojke",
        "kadin", "kadinlar", "meisje", "vrouwen", "kvinna", "kvinner",
        "kvinde", "naine", "sieviete", "moteris", "femeie", "femei"
    ],
    "MIGRANTS": [
        "migrant", "migrants", "refugee", "refugees", "asylum", "foreigner",
        "foreigners", "immigrant", "immigrants", "border", "arrival",
        "migrant", "migrants", "refugie", "refugies", "asile", "etranger",
        "extranjero", "extranjeros", "refugiado", "refugiados", "inmigrante",
        "migrante", "migranti", "rifugiato", "rifugiati", "straniero",
        "fluechtling", "fluchtling", "auslaender", "auslander", "zuwanderer",
        "uchodzca", "uchodzcy", "cudzoziemiec", "imigranci",
        "izbeglica", "izbjeglica", "izbeglice", "migranti", "stranac", "stranci",
        "siginmaci", "gocmen", "gocmenler", "vluchteling", "migranten",
        "flyktning", "flykting", "pakolainen", "bivsi", "azil"
    ],
    "LGBTQ+": [
        "lgbt", "lgbtq", "lgbtqia", "gay", "lesbian", "bisexual", "trans",
        "transgender", "queer", "nonbinary", "homosexual",
        "lesbienne", "homosexuel", "transgenre", "queer",
        "lesbiana", "homosexual", "transgenero",
        "lesbica", "omosessuale", "transgender",
        "schwul", "lesbisch", "trans", "queer",
        "gej", "lezbejka", "lezbejke", "transrodna", "homoseksual",
        "gey", "lezbiyen", "trans", "queer"
    ],
    "STUDENTS": [
        "student", "students", "pupil", "pupils", "school", "university",
        "campus", "faculty", "classroom", "protester", "protesters", "blockade",
        "etudiant", "etudiants", "ecole", "universite", "faculte",
        "estudiante", "estudiantes", "universidad", "facultad",
        "studente", "studenti", "universita", "facolta",
        "studenten", "universitaet", "universitat", "schule",
        "uczen", "studenci", "studentka", "uniwersytet",
        "studenti", "studentkinja", "studentkinje", "ucenik", "djak", "djaci",
        "skola", "fakultet", "blokada", "blokader", "blokaderi", "kampus",
        "ogrenci", "universite", "okul", "studenten", "studenter", "opiskelija"
    ],
    "JOURNALISTS": [
        "journalist", "journalists", "reporter", "reporters", "media", "press",
        "editor", "newsroom", "newspaper", "broadcaster",
        "journaliste", "journalistes", "presse", "media", "medias",
        "periodista", "periodistas", "prensa", "medios",
        "giornalista", "giornalisti", "stampa",
        "journalist", "journalisten", "presse", "medien",
        "dziennikarz", "dziennikarze", "media", "prasa",
        "novinar", "novinari", "novinarka", "mediji", "urednik",
        "gazeteci", "basin", "pers", "journalistiek"
    ],
    "ETHNIC": [
        "minority", "minorities", "ethnic", "race", "racial", "religion",
        "religious", "roma", "romani", "jew", "jewish", "muslim", "islam",
        "arab", "black", "nationality", "nation", "diaspora",
        "minorite", "ethnique", "religieux", "juif", "musulman", "noir",
        "minoria", "etnico", "religioso", "judio", "musulman", "negro",
        "minoranza", "etnico", "religioso", "ebreo", "musulmano", "nero",
        "minderheit", "ethnisch", "religion", "juedisch", "muslimisch",
        "mniejszosc", "etniczny", "zyd", "muzu\u0142manin",
        "romi", "romski", "jevrej", "musliman", "arap", "crnac",
        "azinlik", "yahudi", "musluman", "arap", "siyah"
    ],
    "POLITICIANS": [
        "politician", "politicians", "candidate", "candidates", "minister",
        "president", "mayor", "party", "parliament", "government", "opposition",
        "politique", "candidat", "candidate", "ministre", "president", "parti",
        "politico", "politicos", "candidato", "ministro", "presidente", "partido",
        "politico", "candidata", "ministro", "presidente", "partito",
        "politiker", "kandidat", "minister", "praesident", "partei",
        "polityk", "kandydat", "minister", "prezydent", "partia",
        "politicar", "politicari", "kandidat", "ministar", "predsednik",
        "predsjednik", "stranka", "opozicija", "siyasetci", "aday", "bakan"
    ],
    "DISABILITY": [
        "disabled", "disability", "neurodivergent", "autistic", "wheelchair",
        "mental health", "illness", "health condition",
        "handicape", "handicapes", "handicap", "autiste", "sante mentale",
        "discapacitado", "discapacidad", "autista", "salud mental",
        "disabile", "disabilita", "autistico", "salute mentale",
        "behindert", "behinderung", "autistisch", "psychische gesundheit",
        "niepelnosprawnosc", "niepelnosprawny", "autystyczny",
        "invaliditet", "invalid", "autizam", "mentalno zdravlje",
        "engelli", "otizm", "ruh sagligi"
    ],
}


HOSTILE_KEYWORDS: Dict[str, List[str]] = {
    "threat": [
        "kill", "attack", "beat", "shoot", "burn", "hang", "destroy", "eliminate",
        "wipe out", "hunt down", "drive out", "kick out", "deport", "expel",
        "tuer", "attaquer", "frapper", "bruler", "expulser", "eliminer",
        "matar", "atacar", "golpear", "quemar", "expulsar", "eliminar",
        "uccidere", "attaccare", "bruciare", "espellere", "eliminare",
        "toeten", "toten", "angreifen", "verbrennen", "vertreiben",
        "zabic", "atakowac", "spalic", "wyrzucic",
        "ubit", "ubij", "pobit", "prebit", "spalit", "proter", "hapsit",
        "metak", "batin", "razbit", "linc", "streljat"
    ],
    "dehumanizing_insult": [
        "filth", "trash", "garbage", "parasite", "scum", "subhuman",
        "salete", "ordure", "dechet", "parasite",
        "basura", "escoria", "parasito",
        "spazzatura", "feccia", "parassita",
        "abschaum", "muell", "parasit",
        "smiec", "odpady", "pasozyt",
        "stoka", "bagra", "olos", "gamad", "djubre", "smece", "sljam",
        "cop", "pislik", "parazit"
    ],
    "identity_slur": [
        "slur", "hate name", "identity insult", "peder", "cigan", "siptar",
        "balij", "homophobic insult", "racist insult", "antisemitic insult"
    ],
    "traitor_frame": [
        "traitor", "foreign agent", "enemy within", "paid by", "sellout",
        "traitre", "agent etranger", "vendu",
        "traidor", "agente extranjero", "vendido",
        "traditore", "agente straniero",
        "verraeter", "auslaendischer agent",
        "zdrajca", "obcy agent",
        "izdaj", "placen", "strani agent", "sluga", "soros",
        "hain", "yabanci ajan"
    ],
    "obscene_abuse": [
        "shut up", "idiot", "moron", "degenerate", "disgrace",
        "tais toi", "idiot", "debile", "honte",
        "callate", "idiota", "imbecil", "verg\u00fcenza",
        "stai zitto", "idiota", "vergogna",
        "halt den mund", "idiot", "schande",
        "zamknij sie", "idiota", "hanba",
        "cuti", "debil", "retard", "sramota", "govn", "kurv", "jeb",
        "sus", "aptal", "rezalet"
    ],
    "coordinated_call": [
        "share this", "copy this", "send this to everyone", "post this everywhere",
        "same message", "use this text", "make it trend",
        "partagez", "copiez", "publiez partout",
        "comparte", "copia", "publica en todas partes",
        "teilen", "kopieren", "poste das",
        "podeli", "kopiraj", "salji svima", "objavi svuda"
    ],
}


HOSTILE_FEATURE_TEXT: Dict[str, str] = {
    "threat": "violent threat, expulsion, or intimidation",
    "dehumanizing_insult": "dehumanizing insult",
    "identity_slur": "identity-based slur or identity insult",
    "traitor_frame": "traitor or foreign-agent accusation",
    "obscene_abuse": "obscene personal abuse",
    "coordinated_call": "explicit call to copy, share, or coordinate posting"
}


LANGUAGE_NAMES = {
    "und": "Unspecified",
    "cyrillic": "Cyrillic-script text",
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "pl": "Polish",
    "sr": "Serbian",
    "bs": "Bosnian",
    "hr": "Croatian",
    "sl": "Slovenian",
    "mk": "Macedonian",
    "sq": "Albanian",
    "el": "Greek",
    "tr": "Turkish",
    "ro": "Romanian",
    "bg": "Bulgarian",
    "cs": "Czech",
    "sk": "Slovak",
    "hu": "Hungarian",
    "sv": "Swedish",
    "no": "Norwegian",
    "da": "Danish",
    "fi": "Finnish",
    "et": "Estonian",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "uk": "Ukrainian",
    "ru": "Russian",
    "hy": "Armenian",
    "ka": "Georgian",
}

LANGUAGE_ALIASES = {
    "english": "en", "eng": "en",
    "french": "fr", "francais": "fr", "fran\u00e7ais": "fr", "fra": "fr",
    "german": "de", "deutsch": "de", "ger": "de",
    "spanish": "es", "espanol": "es", "espa\u00f1ol": "es",
    "italian": "it", "italiano": "it",
    "portuguese": "pt", "portugues": "pt", "portugu\u00eas": "pt",
    "dutch": "nl", "nederlands": "nl",
    "polish": "pl", "polski": "pl",
    "serbian": "sr", "srpski": "sr",
    "bosnian": "bs", "bosanski": "bs",
    "croatian": "hr", "hrvatski": "hr",
    "slovenian": "sl", "slovene": "sl",
    "macedonian": "mk",
    "albanian": "sq", "shqip": "sq",
    "greek": "el", "ellinika": "el",
    "turkish": "tr", "turkce": "tr",
    "romanian": "ro", "romana": "ro",
    "bulgarian": "bg",
    "czech": "cs", "cesky": "cs",
    "slovak": "sk", "slovensky": "sk",
    "hungarian": "hu", "magyar": "hu",
    "swedish": "sv", "svenska": "sv",
    "norwegian": "no", "norsk": "no",
    "danish": "da", "dansk": "da",
    "finnish": "fi", "suomi": "fi",
    "estonian": "et", "eesti": "et",
    "latvian": "lv", "latviesu": "lv",
    "lithuanian": "lt", "lietuviu": "lt",
    "ukrainian": "uk", "ukrainian": "uk",
    "russian": "ru", "russkiy": "ru",
    "armenian": "hy",
    "georgian": "ka",
}

LANGUAGE_HINTS = {
    "en": ["the", "and", "with", "from", "this", "that", "people"],
    "fr": ["le", "la", "les", "des", "pour", "avec", "dans", "est", "nous"],
    "de": ["der", "die", "das", "und", "mit", "nicht", "ist", "sind"],
    "es": ["el", "la", "los", "las", "que", "con", "para", "esta", "son"],
    "it": ["il", "la", "gli", "che", "con", "per", "sono", "questa"],
    "pt": ["o", "a", "os", "as", "que", "com", "para", "esta", "sao"],
    "nl": ["de", "het", "een", "met", "voor", "niet", "zijn"],
    "pl": ["nie", "jest", "dla", "oraz", "przez", "ludzie", "ten"],
    "sr": ["nije", "ovo", "smo", "ste", "studenti", "mediji", "stranka"],
    "bs": ["nije", "ovo", "smo", "ste", "ljudi", "mediji"],
    "hr": ["nije", "ovo", "smo", "ste", "ljudi", "mediji"],
    "tr": ["bir", "ve", "icin", "degil", "olan", "bu", "insan"],
    "ro": ["si", "este", "pentru", "cu", "nu", "oamenii"],
    "cs": ["neni", "pro", "lidi", "tento", "jsou", "jako"],
    "sk": ["nie", "pre", "ludia", "tento", "su", "ako"],
    "hu": ["nem", "hogy", "egy", "emberek", "vagy", "mint"],
    "sv": ["och", "inte", "for", "med", "det", "som"],
    "no": ["og", "ikke", "for", "med", "det", "som"],
    "da": ["og", "ikke", "for", "med", "det", "som"],
    "fi": ["ja", "ei", "on", "ovat", "kanssa", "ihmiset"],
}


GUARDRAILS = {
    "privacy": [
        "Ignore author identifiers in modelling and output.",
        "Display only scrubbed text snippets in reports.",
        "Use language hints for triage, not as evidence of identity."
    ],
    "operator_review": [
        "Treat every cluster as a lead requiring human review.",
        "Do not use automated output as a final enforcement decision.",
        "Record methodology and thresholds with exported evidence."
    ],
    "csv_contract": [
        "A text-like column is required.",
        "Timestamp is recommended for burst detection.",
        "Language columns are optional; language is inferred only for summaries."
    ]
}


def clone_target_groups() -> Dict[str, str]:
    return dict(TARGET_GROUPS)


def clone_target_keywords() -> Dict[str, List[str]]:
    return deepcopy(TARGET_KEYWORDS)


def clone_hostile_keywords() -> Dict[str, List[str]]:
    return deepcopy(HOSTILE_KEYWORDS)


def normalize_for_matching(value: str) -> str:
    text = str(value or "").casefold()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def normalize_language(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "und"
    lowered = raw.casefold().replace("_", "-")
    short = lowered.split("-", 1)[0]
    if short in LANGUAGE_NAMES:
        return short
    normalized = normalize_for_matching(raw)
    return LANGUAGE_ALIASES.get(lowered) or LANGUAGE_ALIASES.get(normalized) or short or "und"


def language_display_name(value: str) -> str:
    code = normalize_language(value)
    return LANGUAGE_NAMES.get(code, code.upper() if code else LANGUAGE_NAMES["und"])


def detect_language_hint(text: str) -> str:
    raw = str(text or "")
    if not raw.strip():
        return "und"

    if re.search(r"[\u0370-\u03FF]", raw):
        return "el"
    if re.search(r"[\u0530-\u058F]", raw):
        return "hy"
    if re.search(r"[\u10A0-\u10FF]", raw):
        return "ka"
    if re.search(r"[\u0400-\u04FF]", raw):
        return "cyrillic"

    normalized = normalize_for_matching(raw)
    tokens = set(normalized.split())
    if not tokens:
        return "und"

    scores = {
        code: sum(1 for hint in hints if hint in tokens)
        for code, hints in LANGUAGE_HINTS.items()
    }
    best_code, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score >= 2:
        return best_code
    return "und"
