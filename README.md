# 📄 Intelligent PDF Question Answering Chatbot
**LLM + RAG + NLP + Groq API**

Upload a PDF, ask questions, and get answers based **only** on the document, with source file and page numbers.

## How it works
```
PDF Upload -> Text Extraction (pypdf) -> Chunking -> Embeddings (all-MiniLM-L6-v2)
   -> FAISS index -> User Question -> Semantic Search (top 3)
   -> Groq LLM (answers only from context) -> Answer + Source Pages
```

## Project structure
```
Intelligent_PDF_Chatbot/
├── app.py              # Streamlit interface
├── config.py           # All settings (model, chunk size, top-k)
├── pdf_processor.py    # PDF reading + chunking
├── vector_database.py  # Embeddings + FAISS search
├── rag_pipeline.py     # Retrieval + Groq answer generation
├── requirements.txt
├── .env                # Your secret API key
└── README.md
```

## Setup (Windows + VS Code)

1. Install **Python 3.12** (tick "Add Python to PATH").
2. Get a free API key: https://console.groq.com/keys
3. Open the project folder in VS Code, open a terminal (Terminal -> New Terminal) and run:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

4. Open the `.env` file and replace `your_api_key` with your real key:

```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxx
GROQ_MODEL=llama-3.3-70b-versatile
```
(No quotes, no spaces around `=`.)

5. Run the app:

```bash
streamlit run app.py
```

Your browser opens at http://localhost:8501.

> First run only: the small embedding model (~90 MB) is downloaded once and then reused.

## Using the app
1. Upload a PDF in the sidebar and wait for the green **Ready** box.
2. Type a question in the chat box.
3. Read the answer; the **Sources** chips show the file and page numbers.
4. Click **Clear Chat** to reset the conversation.

## Settings you can change (config.py)
| Setting | Default | Meaning |
|---|---|---|
| `CHUNK_SIZE` | 800 | Characters per chunk |
| `CHUNK_OVERLAP` | 150 | Overlap between chunks |
| `TOP_K` | 3 | Chunks sent to the LLM |
| `GROQ_MODEL` (.env) | llama-3.3-70b-versatile | Faster option: `llama-3.1-8b-instant` |

## Troubleshooting
| Problem | Fix |
|---|---|
| `running scripts is disabled` when activating venv | Run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`, or use Command Prompt |
| "GROQ_API_KEY is not set" | Check `.env` is named exactly `.env` (not `.env.txt`), then restart Streamlit (Ctrl+C, run again) |
| "Invalid Groq API key" | Re-copy the key from console.groq.com/keys |
| "model_decommissioned" / model error | Change `GROQ_MODEL` in `.env` to a model listed at https://console.groq.com/docs/models |
| "No readable text found" | The PDF is scanned images; OCR is not included |
| Slow first start | The embedding model is downloading once; later runs are fast |
| Torch/"__path__" warnings in terminal | Harmless; they do not affect the app |

## Notes
- No local LLM, no Ollama, no GPU needed; answers come from the Groq API.
- Embeddings are cached per PDF, so re-running with the same file is instant.
- If the PDF does not contain the answer, the bot replies:
  *"I could not find this information in the uploaded PDF."*
