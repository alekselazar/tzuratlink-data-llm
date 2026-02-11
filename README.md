# tzuratlink-data-llm

**Automated tagging workflow for the [tzuratlink](https://github.com/your-org/tzuratlink) project.** This app runs a page-tagging pipeline: PDF → layout (Tesseract) → margin filtering → block font classification (Hebrew / Rashi) → Rashi line splitting and Tesseract OCR → line text (Tesseract) → Sefaria streams (filtered by commentary config) → block/segment alignment (fuzzy + embeddings) → commentary span matching → boundary cuts → review UI → persist to Mongo (tzuratlink-data schema).

**Status: still being tested.** The pipeline and UI are in active use; behavior and APIs may change.

## Quick start

1. **Create `.env` and add your OpenAI API key** (used for block font classification and embeddings)

   ```bash
   cp .env.example .env
   ```

   Edit `.env` and set:

   ```env
   OPENAI_API_KEY=sk-your-key-here
   ```

2. **Optional: commentary filter**  
   For Gemara-style pages the app uses commentary titles starting with "Rashi on" / "Tosafot on". To change this (e.g. for Shulchan Arukh), copy and edit the config:

   ```bash
   cp commentary_config.example.json commentary_config.json
   ```

3. **Build and run**

   ```bash
   docker compose up --build
   ```

4. **Open**

   - **UI:** http://localhost:5173  
   - **API health:** http://localhost:8080/health  

5. **Test flow**

   - Enter a **PDF URL** (e.g. a public PDF link) and a **page ref** (e.g. `Berakhot 2a`).
   - Click **Load Page Data**. The pipeline runs (PDF → layout → margin filter → font classification → Rashi split/Tesseract → line text → Sefaria → alignment → commentary matching → boundary cuts).
   - On the review page, click a reference to see its bboxes and ref text.
   - Use **Finalize** to write the page to MongoDB (tzuratlink-data schema).

**PDF input:** Use an HTTP(S) URL in the form, or put a PDF in `./data/` and use path `/data/your.pdf`. Put `rashi.tessdata` (or `rashi.traineddata`) in `./data/` for Rashi line OCR; in Docker the app uses `RASHI_TESSDATA_DIR=/data`.

## API endpoints

- POST `/api/sessions/start`
- GET  `/api/sessions/<sid>`
- POST `/api/sessions/<sid>/apply_fixes`
- POST `/api/sessions/<sid>/finalize`
- GET  `/api/pages/<page_id>`

## What to improve next

1. Add tests (pytest for backend pipeline nodes, API contract tests).
2. Harden security: validate/sanitize `pdf_url` (e.g. block `file://`, restrict HTTP redirects), rate-limit session start.
3. Build out the annotation UI: block→stream assignment, segment boundaries and cut handles, preview overlays.
