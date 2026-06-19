"""OpenAI structured scene prompts for storybook illustration (no Streamlit)."""
import json
from typing import Any, Dict, List, Optional, Sequence


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
    page_range_label: str = "",
) -> Optional[Dict[str, Any]]:
    """
    Call OpenAI for structured scene breakdown. Returns dict with scene_visual, composition, etc., or None on error.
    """
    if not (page_text or "").strip():
        return None
    title_line = f'Story title: "{story_title}". ' if story_title else ""
    style_line = f"Style context only—do not echo style jargon in scene fields: {global_style[:500]}\n" if global_style else ""
    range_line = (
        f"This single illustration covers story pages {page_range_label}. "
        if page_range_label
        else ""
    )
    user = (
        f"{title_line}{style_line}"
        f"{range_line}"
        f"Cast/continuity (director notes): {character_ref or 'main story characters'}.\n"
        f'Combined page text: """{(page_text or "").strip()[:4000]}"""\n\n'
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


SCENE_PARAGRAPH_INSTRUCTION = f"""Turn the combined page text into ONE visual scene description for an image generator.

{SCENE_HARD_RULES}

Write a single paragraph (2–4 sentences, ~40–80 words) that describes what the illustration should show:
- Who is visible, what they are doing, where they stand, and key props
- Camera/framing in plain words (wide shot, close-up, etc.)
- Time of day, weather, and setting details when relevant
- Do NOT include art-style words (no "watercolor", "storybook style", etc.) — style is handled separately
- Do NOT quote or repeat the page text verbatim — translate narrative into a visible moment
- If multiple pages are combined, pick ONE cohesive moment that best represents all of them

Return only the paragraph, no labels or markdown."""


def generate_scene_description_paragraph(
    openai_client,
    page_text: str,
    *,
    story_summary: str = "",
    story_title: str = "",
    character_ref: str = "",
    location_ref: str = "",
    page_range_label: str = "",
    model: str = "gpt-4o-mini",
) -> Optional[str]:
    """
    Call OpenAI to turn page text into a Leonardo-ready scene paragraph.
    Returns the paragraph string, or None on error.
    """
    if not (page_text or "").strip():
        return None
    title_line = f'Story title: "{story_title}".\n' if story_title else ""
    summary_line = (
        f'Story summary (context only): """{(story_summary or "").strip()[:1500]}"""\n'
        if (story_summary or "").strip()
        else ""
    )
    range_line = (
        f"This illustration covers pages {page_range_label}.\n"
        if page_range_label
        else ""
    )
    char_line = (
        f"Known characters (keep consistent): {(character_ref or '').strip()[:800]}\n"
        if (character_ref or "").strip()
        else ""
    )
    loc_line = (
        f"Known location (if relevant): {(location_ref or '').strip()[:400]}\n"
        if (location_ref or "").strip()
        else ""
    )
    user = (
        f"{title_line}{summary_line}{range_line}{char_line}{loc_line}"
        f'Combined page text: """{(page_text or "").strip()[:4000]}"""\n\n'
        f"{SCENE_PARAGRAPH_INSTRUCTION}"
    )
    try:
        resp = openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a children's book art director. Convert story page text into a concise "
                        "visual scene description for an image generator. "
                        + SCENE_HARD_RULES
                    ),
                },
                {"role": "user", "content": user},
            ],
            max_tokens=220,
            temperature=0.4,
        )
        text = (resp.choices[0].message.content or "").strip()
        return text or None
    except Exception:
        return None


def _format_story_blocks_for_prompt(blocks: Sequence[Dict[str, Any]]) -> str:
    """Format all image blocks as labeled story sections for a single ChatGPT call."""
    parts: List[str] = []
    for block in blocks:
        block_start = block.get("block_start")
        page_range = (block.get("page_range_label") or "").strip()
        header = f"--- Block pages {page_range} (block_start={block_start}) ---"
        page_lines: List[str] = []
        for page in block.get("pages") or []:
            pn = page.get("page_number")
            text = (page.get("text") or "").strip()
            if text:
                page_lines.append(f"Page {pn}: {text}")
        if not page_lines:
            combined = (block.get("combined_text") or "").strip()
            if combined:
                page_lines.append(combined)
        if page_lines:
            parts.append(header + "\n" + "\n".join(page_lines))
    return "\n\n".join(parts)


def _full_story_scenes_instruction(pages_per_image: int) -> str:
    return f"""You are planning ALL illustrations for a children's book in one pass for visual consistency.

{SCENE_HARD_RULES}

The story is split into blocks below. Each block becomes ONE illustration covering those pages ({pages_per_image} pages share each image).

For EACH block, write one scene description paragraph (2–4 sentences, ~40–80 words):
- Who is visible, what they are doing, where they stand, and key props
- Camera/framing in plain words (wide shot, close-up, etc.)
- Time of day, weather, and setting details when relevant
- Do NOT include art-style words (no "watercolor", "storybook style", etc.)
- Do NOT quote or repeat page text verbatim — translate narrative into a visible moment
- If a block has multiple pages, pick ONE cohesive moment that best represents them
- Use the FULL story context: keep each named character's appearance and clothing consistent across blocks
- Vary compositions and settings — avoid repeating the same framing in consecutive blocks

Return JSON only with this exact shape (one entry per block, every block_start included):
{{"scenes": [{{"block_start": <int>, "scene_description": "<paragraph>"}}, ...]}}
No markdown or code fences."""


def generate_all_scene_descriptions(
    openai_client,
    blocks: Sequence[Dict[str, Any]],
    *,
    story_summary: str = "",
    story_title: str = "",
    character_refs: str = "",
    location_refs: str = "",
    pages_per_image: int = 3,
    model: str = "gpt-4o-mini",
) -> Optional[Dict[int, str]]:
    """
    Send the entire story to ChatGPT and get scene descriptions for every block at once.
    Returns block_start -> scene paragraph, or None on error.
    """
    if not blocks:
        return None
    story_body = _format_story_blocks_for_prompt(blocks)
    if not story_body.strip():
        return None

    title_line = f'Story title: "{story_title}".\n' if story_title else ""
    summary_line = (
        f'Story summary: """{(story_summary or "").strip()[:2000]}"""\n\n'
        if (story_summary or "").strip()
        else ""
    )
    char_line = (
        f"Known characters (keep consistent in every illustration):\n{(character_refs or '').strip()[:2000]}\n\n"
        if (character_refs or "").strip()
        else ""
    )
    loc_line = (
        f"Known locations (use when relevant):\n{(location_refs or '').strip()[:1000]}\n\n"
        if (location_refs or "").strip()
        else ""
    )
    user = (
        f"{title_line}{summary_line}{char_line}{loc_line}"
        f"FULL STORY BY ILLUSTRATION BLOCK:\n\n{story_body}\n\n"
        f"{_full_story_scenes_instruction(pages_per_image)}"
    )
    n_blocks = len(blocks)
    try:
        resp = openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a children's book art director planning every illustration in a book "
                        "as one cohesive visual story. Maintain character continuity across all scenes. "
                        + SCENE_HARD_RULES
                    ),
                },
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            max_tokens=min(16000, 180 * n_blocks + 400),
            temperature=0.35,
        )
        text = resp.choices[0].message.content
        if not text:
            return None
        data = json.loads(text)
        scenes = data.get("scenes") or []
        result: Dict[int, str] = {}
        for entry in scenes:
            if not isinstance(entry, dict):
                continue
            bs = entry.get("block_start")
            desc = (entry.get("scene_description") or "").strip()
            if bs is None or not desc:
                continue
            try:
                result[int(bs)] = desc
            except (TypeError, ValueError):
                continue
        return result or None
    except Exception:
        return None
