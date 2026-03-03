-- Run this in Supabase SQL Editor to enable per-story, per-grade image style persistence.
-- Table: story_grade_styles (one row per story_id + reading_level).

CREATE TABLE IF NOT EXISTS story_grade_styles (
  story_id bigint NOT NULL REFERENCES stories(id) ON DELETE CASCADE,
  reading_level text NOT NULL,
  age_appropriateness text,
  global_style text,
  character_ref text,
  color_palette text,
  lighting text,
  framing text,
  reference_page_index integer,
  PRIMARY KEY (story_id, reading_level)
);

-- Optional: enable RLS and add policies if your project uses Row Level Security.
-- ALTER TABLE story_grade_styles ENABLE ROW LEVEL SECURITY;

-- If the table already exists, add the reference page column with:
-- ALTER TABLE story_grade_styles ADD COLUMN IF NOT EXISTS reference_page_index integer;
