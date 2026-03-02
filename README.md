# Art Mash (Flat Folder)

This version runs entirely from a **single folder** (no subpackages).

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Database location
By default the app uses `artmash.sqlite3` in the working directory (if writable).
Override with:

```bash
ARTMASH_DB_PATH=/absolute/path/to/artmash.sqlite3 streamlit run app.py
```

## Notes
- Crawling fetches **HTML only** and caches it under `.cache_artmash/`.
- Leaderboards are computed live from SQLite after every vote.
