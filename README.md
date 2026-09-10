# agent_backend

FastAPI backend for the PPTist-first conversational PPT agent.

## Current Model

- Upload accepts only `.ppt` and `.pptx`.
- LibreOffice converts the upload to a temporary PDF only to render
  `baseline/pages_png/page_NNN.png`.
- The original PPT/PPTX is kept only until the frontend PPTist parser initializes
  the project.
- Each page's authoritative state is `pages/page_NNN/state/pptist_slide.json`.
- Read-only inspection and Patch operate from PPTist JSON directly.
- Full Pipeline uses HTML only inside a single turn directory, then converts the
  final HTML back into PPTist JSON and `state/reread_page.png`.
- Backend PDF upload, persistent HTML preview/chunks, HTML manual sync, backend
  PDF export, and PDF source serving have been removed.

## Run

```bash
pip install -r agent_backend/requirements.txt
python -m playwright install chromium

python -m uvicorn agent_backend.server.app:app \
  --host 127.0.0.1 --port 8765
```

## Main HTTP Surface

| Method & path | Purpose |
| --- | --- |
| `POST /api/decks` | Upload `.ppt` / `.pptx`; creates baseline PNGs and waits for frontend PPTist initialization. |
| `POST /api/projects/{pid}/initialize-pptist` | Store all per-page PPTist slide JSON from the frontend parser. |
| `GET /api/projects/{pid}/deck.json` | Return the whole deck as PPTist JSON in display order. |
| `GET /api/projects/{pid}/pages/{slot}/slide.json` | Return one authoritative PPTist slide by stable slot. |
| `POST /api/projects/{pid}/deck/stage` | Stage manual-edit slide JSON candidates by content hash. |
| `POST /api/projects/{pid}/deck/reread-images` | Upload frontend-rendered page PNGs paired with staged JSON. |
| `POST /api/projects/{pid}/deck/reread-ready` | Promote staged JSON/PNG and mark Full Pipeline understanding stale. |
| `GET /api/projects/{pid}/outline` | Deterministic outline derived from PPTist JSON. |
| `GET /api/projects/{pid}/events` | Project SSE events. |
| `POST /api/sessions/{sid}/chat` | Agent chat entrypoint. |

Export is handled by the frontend PPTist editor.
