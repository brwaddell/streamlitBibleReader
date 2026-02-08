# Storybook Image Processor

Bulk image generation for storybook apps. Paste story text, split into pages, generate images via Nano Banana Pro (Gemini 3 Pro Image), use OpenAI GPT for extra-details hints, review/approve or regenerate with corrections, then export to Supabase Storage and insert URLs into `book_pages`.

## Setup

1. **Create a virtual environment** (recommended):

   ```bash
   python -m venv .venv
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
   - `SUPABASE_SERVICE_KEY` – Supabase service_role key (for storage + DB)
   - `OPENAI_API_KEY` – from [OpenAI](https://platform.openai.com/api-keys) (extra-details hints)
   - `GEMINI_API_KEY` – from [Google AI Studio](https://aistudio.google.com/apikey) (Nano Banana Pro)

4. **Supabase Auth**:

   - In Supabase Dashboard → Authentication → Providers: enable **Email**
   - Go to Authentication → Users → Add user (email + password) for each person who should access the app

5. **Create the Supabase storage bucket**:

   In Supabase Dashboard → Storage → New bucket:

   - Name: `storybook-images`
   - Public: Yes (for public image URLs)
   - File size limit: as needed (e.g., 5MB)

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
   - `SUPABASE_URL`
   - `SUPABASE_ANON_KEY`
   - `SUPABASE_SERVICE_KEY`
   - `OPENAI_API_KEY`
   - `GEMINI_API_KEY`

5. Click **Deploy**. The app will build and run. Only users you create in Supabase Auth can sign in.

6. (Optional) Enable **Sign in required** in app settings to add another layer of access control via Streamlit accounts.

## Flow

1. **Select story & reading level** – Choose from `stories` table and a reading level (grade_1–grade_5).
2. **Paste story text** – Enter raw text and set a delimiter (default `---`) to split into pages.
3. **Set image controls** – Global style, character reference, color palette, negative prompt.
4. **Generate** – Generate images per page or in bulk. Approve or regenerate with correction instructions.
5. **Export** – Upload approved images to `story-images` bucket and insert rows into `book_pages`.

## Storage Layout

Images are stored as WebP (max 800×800, ~100KB) at:

```
storybook-images/{story_id}/{reading_level}/page_{page_index}.webp
```
