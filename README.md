# Art Mash (Streamlit)

A FaceMash-style voting app for historical art from Gallerix storeroom.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes
- Uses `artmash.sqlite3` in the project folder by default. On Streamlit Cloud, it will fall back to `/tmp/artmash.sqlite3` if the filesystem is read-only.
- No server-side image downloading: uses iframe mode (most reliable) or browser-side `<img>` tag when metadata is available.
- Bulk loading is resumable (run small batches to avoid timeouts).
