-- Extend story_grade_styles for Leonardo + canonical character reference URL. Run in Supabase SQL Editor.

ALTER TABLE story_grade_styles
  ADD COLUMN IF NOT EXISTS character_reference_image_url TEXT,
  ADD COLUMN IF NOT EXISTS leonardo_seed BIGINT,
  ADD COLUMN IF NOT EXISTS default_image_provider TEXT;

COMMENT ON COLUMN story_grade_styles.character_reference_image_url IS 'Public URL (e.g. R2) for canonical character reference image';
COMMENT ON COLUMN story_grade_styles.leonardo_seed IS 'Optional fixed seed for Leonardo consistency per story+grade';
COMMENT ON COLUMN story_grade_styles.default_image_provider IS 'gemini or leonardo';
