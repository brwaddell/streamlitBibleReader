-- Labeled reference image URL(s) per story + reading level (JSON array; first entry used for generation).
-- Run in Supabase SQL Editor after story_grade_styles exists.

ALTER TABLE story_grade_styles
  ADD COLUMN IF NOT EXISTS reference_images jsonb NOT NULL DEFAULT '[]'::jsonb;

COMMENT ON COLUMN story_grade_styles.reference_images IS
  'Array of {"label","url"} for visual consistency; only the first entry is used for generation';
