"""OpenAI structured scene prompts for storybook illustration (no Streamlit)."""
import json
from typing import Any, Dict, Optional


SCENE_HARD_RULES = """HARD RULES (always obey):
1) Every human character is Middle Eastern in appearance and dressed in ancient Near East clothing (draped robes, tunics, simple sandals, period-appropriate head coverings when relevant)—never default to European or modern dress.
2) Never use a human figure to represent God or any deity (no visible divine body or face; no person standing in for God). Show the divine through light, weather, nature, implied voice, empty space, symbolic objects, or similar—not a human form as God."""

# Baked into every Leonardo positive; Gemini uses SCENE_HARD_RULES via lib.SYSTEM_INSTRUCTIONS.
LEONARDO_POSITIVE_IMAGE_RULES = (
    "Mandatory: all humans Middle Eastern, ancient Near East dress. "
    "Never depict God or any deity as a human—use light, nature, symbols, implied presence only."
)

# Mirrors the intent of lib.SYSTEM_INSTRUCTIONS for Leonardo (diffusion-friendly, compact).
LEONARDO_ILLUSTRATOR_LOCK = (
    "CRITICAL character consistency: same face, hair, and clothing as the character description—fixed cast, no redesign. "
    "Friendly reassuring storybook mood. No text, letters, or typography in the image. "
)

LEONARDO_NEGATIVE_IMAGE_HARD_RULES = (
    "European default ethnicity, modern clothing, wrong-period Western dress, "
    "God as human, deity as person, anthropomorphic divine figure, visible divine face or body"
)

LEONARDO_NEGATIVE_KID_SOFT = (
    "graphic gore, horror aesthetic, sexualized, disturbing mood, scary-for-toddlers imagery, "
    "sinister or cruel expressions on sympathetic characters"
)


SCENE_JSON_INSTRUCTION = f"""Think like a film director blocking a shot: terse notes only, no prose flourishes.

{SCENE_HARD_RULES}

Return a single JSON object with these string fields (keep each value short—phrases or one tight sentence, not paragraphs):
- scene_visual: What the single frame shows (action + key props + place). Max ~20 words. No art-style adjectives.
- composition: Camera and framing in a few words (e.g. "wide, horizon low", "OTS on mother", "two-shot center").
- environment: Time, weather, architecture in one short phrase.
- characters_visible: Who is on camera and where they stand or move—minimal words.
- negative_additions: Comma-separated scene-specific avoids. If the page implies divine presence, include explicit avoids such as: no humanoid deity, no anthropomorphic God, no visible face or body for the divine. Otherwise still note any other local avoids.

Do not include markdown or code fences."""


def suggest_scene_prompt(
    openai_client,
    page_text: str,
    character_ref: str = "",
    global_style: str = "",
    story_title: str = "",
    model: str = "gpt-4o-mini",
) -> Optional[Dict[str, Any]]:
    """
    Call OpenAI for structured scene breakdown. Returns dict with scene_visual, composition, etc., or None on error.
    """
    if not (page_text or "").strip():
        return None
    title_line = f'Story title: "{story_title}". ' if story_title else ""
    style_line = f"Style context only—do not echo style jargon in scene fields: {global_style[:500]}\n" if global_style else ""
    user = (
        f"{title_line}{style_line}"
        f"Cast/continuity (director notes): {character_ref or 'main story characters'}.\n"
        f'Page text: """{(page_text or "").strip()[:4000]}"""\n\n'
        f"{SCENE_JSON_INSTRUCTION}"
    )
    try:
        resp = openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a storyboard director: turn page text into minimal shot notes—who stands where, "
                        "what we see, how the frame is cut. Be concise. "
                        + SCENE_HARD_RULES
                    ),
                },
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            max_tokens=380,
        )
        text = resp.choices[0].message.content
        if not text:
            return None
        return json.loads(text)
    except Exception:
        return None


def merge_scene_to_extra_details(data: Dict[str, Any]) -> str:
    """Flatten structured scene JSON into one paragraph for build_prompt extra_details."""
    if not data:
        return ""
    parts = []
    for key in ("scene_visual", "composition", "environment", "characters_visible"):
        v = (data.get(key) or "").strip()
        if v:
            parts.append(v)
    return " ".join(parts).strip()


def merge_negative_additions(data: Dict[str, Any]) -> str:
    return (data.get("negative_additions") or "").strip()
