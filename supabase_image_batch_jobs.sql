-- Batch jobs for Gemini Batch API image generation.
-- Run in Supabase SQL Editor or migration.
CREATE TABLE image_batch_jobs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  batch_name TEXT NOT NULL,
  story_id INT NOT NULL,
  reading_level TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',
  page_count INT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now(),
  completed_at TIMESTAMPTZ,
  error_message TEXT
);
