-- Persist ChatGPT-generated style scene per story + grade.
-- Run in Supabase SQL Editor after story_grade_styles exists.

ALTER TABLE story_grade_styles
  ADD COLUMN IF NOT EXISTS style_scene_text text;

COMMENT ON COLUMN story_grade_styles.style_scene_text IS
  'ChatGPT-generated story-specific style scene description for Leonardo';
