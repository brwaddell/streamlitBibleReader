-- Batch image pipeline: runs + per-block items. Run in Supabase SQL Editor.

CREATE TABLE IF NOT EXISTS image_pipeline_runs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  status TEXT NOT NULL DEFAULT 'running',
  language_code TEXT NOT NULL DEFAULT 'en',
  items_total INTEGER NOT NULL DEFAULT 0,
  items_done INTEGER NOT NULL DEFAULT 0,
  items_failed INTEGER NOT NULL DEFAULT 0,
  leonardo_spend_note TEXT,
  last_error TEXT,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_image_pipeline_runs_status
  ON image_pipeline_runs (status, created_at DESC);

CREATE TABLE IF NOT EXISTS image_pipeline_items (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  run_id UUID NOT NULL REFERENCES image_pipeline_runs(id) ON DELETE CASCADE,
  story_id BIGINT NOT NULL,
  reading_level TEXT NOT NULL,
  language_code TEXT NOT NULL DEFAULT 'en',
  block_start INTEGER NOT NULL,
  anchor_row_id TEXT NOT NULL,
  page_range_label TEXT,
  status TEXT NOT NULL DEFAULT 'queued',
  style_scene TEXT,
  illustration_moment TEXT,
  character_ref_id TEXT,
  pending_image_url TEXT,
  leonardo_cost TEXT,
  error_message TEXT,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_image_pipeline_items_run_status
  ON image_pipeline_items (run_id, status);

CREATE INDEX IF NOT EXISTS idx_image_pipeline_items_review
  ON image_pipeline_items (status, story_id, reading_level);

-- One active item per block (queued, generating, or pending review).
CREATE UNIQUE INDEX IF NOT EXISTS idx_image_pipeline_items_active_block
  ON image_pipeline_items (story_id, reading_level, language_code, block_start)
  WHERE status IN ('queued', 'generating', 'pending_review');
