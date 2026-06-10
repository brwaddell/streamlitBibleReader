-- Storybook reference library (style / characters / locations) per story + reading level.
-- Run in Supabase SQL Editor after story_grade_styles exists.

ALTER TABLE story_grade_styles
  ADD COLUMN IF NOT EXISTS storybook_references jsonb NOT NULL DEFAULT '{}'::jsonb;

COMMENT ON COLUMN story_grade_styles.storybook_references IS
  'Optional Leonardo reference library: {"style": {...}, "characters": [...], "locations": [...]} with url + leonardo_init_image_id per approved entry';
