# Storybook Image Processor

Bulk image generation for storybook apps. Paste story text, split into pages, generate images with **Gemini** (e.g. `gemini-3-pro-image-preview`) or **Leonardo.ai**, use **OpenAI** for structured scene prompts, review/approve or regenerate, then export to Cloudflare R2 and insert URLs into `story_content_flat`.

## Setup

1. **Create a virtual environment** (recommended):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**:

   Copy `.env.example` to `.env` and fill in your keys:

   - `SUPABASE_URL` – your Supabase project URL
   - `SUPABASE_ANON_KEY` – Supabase anon/public key (for auth login)
   - `SUPABASE_SERVICE_KEY` – Supabase service_role key (for database)
   - `OPENAI_API_KEY` – from [OpenAI](https://platform.openai.com/api-keys) (scene prompt suggestions)
   - `GEMINI_API_KEY` – from [Google AI Studio](https://aistudio.google.com/apikey) (Gemini image generation)
   - `LEONARDO_API_KEY`, `LEONARDO_MODEL_ID` – from [Leonardo.ai API](https://docs.leonardo.ai/) (optional; set Image Processor provider to Leonardo)
   - `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY` – from [Cloudflare R2](https://developers.cloudflare.com/r2/) (images + audio storage)
   - `R2_PUBLIC_URL_IMAGES`, `R2_PUBLIC_URL_AUDIO` – public URLs for your R2 buckets (e.g. `https://pub-xxx.r2.dev` or custom domain)

4. **Supabase Auth**:

   - In Supabase Dashboard → Authentication → Providers: enable **Email**
   - Go to Authentication → Users → Add user (email + password) for each person who should access the app

5. **Create Cloudflare R2 buckets**:

   In Cloudflare Dashboard → R2 → Create bucket:

   - Create `storybook-images` for WebP images
   - Create `storybook-audio` for MP3 audio
   - Enable **Public access** on each bucket (or use custom domains)
   - Copy the public URL (e.g. `https://pub-xxxxx.r2.dev`) into `R2_PUBLIC_URL_IMAGES` and `R2_PUBLIC_URL_AUDIO`

6. **Supabase SQL (Leonardo job queue + style columns)**:

   Run in the SQL Editor (once per project):

   - [`supabase_image_generation_jobs.sql`](supabase_image_generation_jobs.sql) – per-page Leonardo async jobs
   - [`supabase_story_grade_styles_leonardo.sql`](supabase_story_grade_styles_leonardo.sql) – `character_reference_image_url`, `leonardo_seed`, `default_image_provider` on `story_grade_styles`

7. **Optional: Per-story character/style** (Image Processor → **Settings** tab):

   To use different characters or styles per story (instead of the default Noah style), add optional columns to your `stories` table in Supabase. The app will use them when you select that story:

   - `character_ref` (text) – e.g. "David: young shepherd, dark hair, simple tunic..."
   - Optional: `global_style`, `age_appropriateness`, `color_palette`, `lighting`, `framing` (text)

   If these columns are missing, the app uses the reading-level defaults (e.g. Noah for Bible stories).

## Run

```bash
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Push your repo to GitHub (ensure `.env` and `.streamlit/secrets.toml` are in `.gitignore`).

2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.

3. Click **New app**:
   - Repository: your repo
   - Branch: main
   - Main file path: `app.py`

4. Open **Advanced settings** and add these secrets (as key-value pairs):
   - `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_KEY`
   - `OPENAI_API_KEY`, `GEMINI_API_KEY`, and if using Leonardo: `LEONARDO_API_KEY`, `LEONARDO_MODEL_ID`
   - `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`
   - `R2_PUBLIC_URL_IMAGES`, `R2_PUBLIC_URL_AUDIO`

5. Click **Deploy**. The app will build and run. Only users you create in Supabase Auth can sign in.

6. (Optional) Enable **Sign in required** in app settings to add another layer of access control via Streamlit accounts.

## Flow

1. **Select story & reading level** – Choose from `stories` table and a reading level (grade_1–grade_5).
2. **Paste story text** – Enter raw text and set a delimiter (default `#`) to split into pages.
3. **Image Processor → Settings (optional)** – Reference image URL, Leonardo seed, default provider (Gemini vs Leonardo), and style fields saved to `story_grade_styles`.
4. **Image Processor → Workflow** – Scene prompts (OpenAI suggestions + edits), then **Gemini** (including batch API) or **Leonardo** (submit queue + check jobs).
5. **Review** – Approve to R2 + `story_content_flat`, regenerate, bulk approve, or clear images.

## Storage Layout (Cloudflare R2)

Images are stored as WebP (max 800×800, ~100KB) at:

```
{story_id}/{reading_level}/page_{page_index}.webp
```

Audio is stored at:

```
stories/{story_id}/{language_code}/{reading_level}/{gender}/page_{page_index}.mp3
```

## Custom domains (image.aptreelearning.com / audio.aptreelearning.com)

Set `R2_PUBLIC_URL_IMAGES` and `R2_PUBLIC_URL_AUDIO` in `.env` to your custom domains. To update existing DB URLs to use them:

```bash
python3 scripts/update_urls_to_custom_domain.py
```

## Migrating from Supabase Storage

If you have existing images/audio in Supabase Storage, run the migration script:

```bash
python3 scripts/migrate_supabase_to_r2.py
```

This downloads each file from Supabase, uploads to R2, and updates `story_content_flat` URLs.
