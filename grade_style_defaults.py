"""Default image style presets per reading level (Image Processor Settings tab)."""

DEFAULT_CHARACTER_REF = (
    "Middle eastern characters, period appropriate clothes and tools and personal items. "
    "No duplicate people, no human figures to represent God"
)

GRADE_STYLE_DEFAULTS = {
    "grade_1": {
        "age_appropriateness": "Pre-school (ages 3–5). Simple, friendly, reassuring visuals.",
        "global_style": "Bright, simple flat-color illustrations, bold outlines, thick brushstrokes, very clear subjects, minimal background detail. High contrast and simplicity.",
        "character_ref": DEFAULT_CHARACTER_REF,
        "color_palette": "Earth tones, high contrast",
        "lighting": "Soft morning light",
        "framing": "Medium shot, centered subject, warm composition",
        "default_image_provider": "leonardo",
    },
    "grade_2": {
        "age_appropriateness": "Early reader (5–6 yrs). Clear focal points, engaging and easy to process.",
        "global_style": "Soft watercolor textures, hand-drawn charcoal outlines, gentle gradients, whimsical and warm atmosphere. Storybook feel.",
        "character_ref": DEFAULT_CHARACTER_REF,
        "color_palette": "Earth tones, warm browns and greens",
        "lighting": "Soft morning light",
        "framing": "Medium shot, centered subject, warm composition",
        "default_image_provider": "leonardo",
    },
    "grade_3": {
        "age_appropriateness": "Developing reader (7–8 yrs). More narrative detail while remaining approachable.",
        "global_style": "Richer watercolor textures, expressive linework, gentle gradients, warm storybook atmosphere. More environmental context.",
        "character_ref": DEFAULT_CHARACTER_REF,
        "color_palette": "Earth tones, warm browns and greens",
        "lighting": "Soft morning light",
        "framing": "Medium shot, balanced composition, warm storybook framing",
        "default_image_provider": "leonardo",
    },
    "grade_4": {
        "age_appropriateness": "Fluent reader (9–10 yrs). Sophisticated, more complex visuals.",
        "global_style": "Cinematic digital art, rich textures, detailed environmental storytelling, dramatic lighting (Chiaroscuro). Reverent tone.",
        "character_ref": DEFAULT_CHARACTER_REF,
        "color_palette": "Earth tones, rich and deep",
        "lighting": "Dramatic Chiaroscuro, reverent and epic",
        "framing": "Cinematic framing, rule of thirds, reverent composition",
        "default_image_provider": "leonardo",
    },
    "grade_5": {
        "age_appropriateness": "Independent reader (11+ yrs). Mature, nuanced visuals for older readers.",
        "global_style": "Cinematic digital art, rich textures, detailed environmental storytelling, dramatic lighting (Chiaroscuro). Reverent and epic tone.",
        "character_ref": DEFAULT_CHARACTER_REF,
        "color_palette": "Earth tones, rich and deep",
        "lighting": "Dramatic Chiaroscuro, reverent and epic",
        "framing": "Cinematic framing, rule of thirds, epic composition",
        "default_image_provider": "leonardo",
    },
}
