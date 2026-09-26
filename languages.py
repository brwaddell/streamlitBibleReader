"""Story languages stored on story_content_flat.language_code.

English is the source text. Every other code is a translation target.
ElevenLabs multilingual v2 accepts the ISO 639-1 code in elevenlabs_code.
"""

from typing import Dict, List, NamedTuple


class Language(NamedTuple):
    code: str
    name: str
    prompt_name: str
    elevenlabs_code: str


LANGUAGES: List[Language] = [
    Language("en", "English", "English", "en"),
    Language("es", "Spanish", "Spanish", "es"),
    Language("fr", "French", "French", "fr"),
    Language("de", "German", "German", "de"),
    Language("pt", "Portuguese", "Brazilian Portuguese", "pt"),
    Language("it", "Italian", "Italian", "it"),
    Language("ar", "Arabic", "Modern Standard Arabic", "ar"),
    Language("zh", "Mandarin", "Simplified Chinese (Mandarin)", "zh"),
    Language("ja", "Japanese", "Japanese", "ja"),
]

LANGUAGE_CODES: List[str] = [lang.code for lang in LANGUAGES]
LANGUAGE_BY_CODE: Dict[str, Language] = {lang.code: lang for lang in LANGUAGES}
TRANSLATION_TARGET_CODES: List[str] = [code for code in LANGUAGE_CODES if code != "en"]

READING_LEVEL_LABELS = {
    "grade_1": "Pre-K / Grade 1 (ages 3–6)",
    "grade_2": "Grade 2 (ages 5–7)",
    "grade_3": "Grade 3 (ages 7–8)",
    "grade_4": "Grade 4 (ages 9–10)",
    "grade_5": "Grade 5 (ages 11+)",
}


def language_label(code: str) -> str:
    lang = LANGUAGE_BY_CODE.get((code or "").strip().lower())
    if not lang:
        return code or ""
    return f"{lang.name} ({lang.code})"


def elevenlabs_language_code(code: str) -> str:
    lang = LANGUAGE_BY_CODE.get((code or "en").strip().lower())
    if not lang:
        return (code or "en").strip().lower()
    return lang.elevenlabs_code


def translation_system_prompt(target_code: str, reading_level_label: str) -> str:
    lang = LANGUAGE_BY_CODE.get((target_code or "").strip().lower())
    target_name = lang.prompt_name if lang else target_code
    return (
        "You are a professional children's Bible translator. "
        f"Translate the following text into {target_name} for a {reading_level_label} audience. "
        "Maintain the tone, rhythm, and simplicity of the original. "
        "Use the standard children's Bible form of proper names in the target language. "
        "Do not add titles, notes, or commentary. Return only the translated text."
    )
