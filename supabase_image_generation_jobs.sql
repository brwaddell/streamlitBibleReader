-- Per-page async image jobs (e.g. Leonardo). Run in Supabase SQL Editor.
CREATE TABLE IF NOT EXISTS image_generation_jobs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  story_id BIGINT NOT NULL,
  reading_level TEXT NOT NULL,
  story_content_flat_id TEXT NOT NULL,
  provider TEXT NOT NULL DEFAULT 'leonardo',
  external_generation_id TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',
  error_message TEXT,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_image_gen_jobs_story_level_status
  ON image_generation_jobs (story_id, reading_level, status);
