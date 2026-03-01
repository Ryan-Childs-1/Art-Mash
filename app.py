# app.py
# Art Mash — Storeroom Randomized + Resumable Bulk Loader + Live Leaderboards (Stable + Cloud-safe)
#
# ✅ Working baseline + fixes:
# - Voting is pair-locked in session_state (prevents wrong-winner rerun mixups)
# - Bulk loader is resumable WITHOUT storing giant JSON blobs in ingest_state (uses artists table + last_artist_id cursor)
# - Leaderboards are live DB queries and robust to NULLs
# - Cloud-safe filesystem: DB + cache fall back to /tmp when working dir is read-only
# - SQLite busy_timeout + small retries to reduce "database is locked" issues
#
# Dependencies: streamlit only (stdlib otherwise)

import os
import re
import time
import json
import math
import hashlib
import sqlite3
import random
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from urllib.parse import urljoin
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from html.parser import HTMLParser

import streamlit as st
import streamlit.components.v1 as components

# ----------------------------
# Constants
# ----------------------------
APP_NAME = "Art Mash"
BASE = "https://gallerix.org"
STOREROOM_ROOT = "https://gallerix.org/storeroom/"
LETTER_URL = "https://gallerix.org/storeroom/letter/{L}/"
DEFAULT_UA = "ArtMash/2.8 (Streamlit; respectful crawler)"

# Network defaults (kept simple)
DEFAULT_TIMEOUT = 10.0
DEFAULT_DELAY = 0.2

# TrueSkill-lite defaults
TS_MU0 = 25.0
TS_SIGMA0 = 8.333
TS_BETA = 4.1667
TS_TAU = 0.08

LATIN_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

STYLE_KEYWORDS = {
    "Renaissance": ["renaissance"],
    "Baroque": ["baroque"],
    "Rococo": ["rococo"],
    "Romanticism": ["romantic", "romanticism"],
    "Realism": ["realism", "realist"],
    "Impressionism": ["impressionism", "impressionist"],
    "Post-Impressionism": ["post-impression", "postimpression"],
    "Expressionism": ["expressionism", "expressionist"],
    "Symbolism": ["symbolism", "symbolist"],
    "Cubism": ["cubism", "cubist"],
    "Surrealism": ["surrealism", "surrealist"],
    "Abstract": ["abstract", "abstraction"],
    "Neoclassicism": ["neoclassic", "neoclassicism"],
}

ARTIST_URL_RE = re.compile(r"^https://gallerix\.org/storeroom/(\d+)/?$")
PAINTING_URL_RE = re.compile(r"^https://gallerix\.org/storeroom/\d+/N/\d+/?$")

# ----------------------------
# Paths (cloud-safe)
# ----------------------------
def _is_writable_dir(path: str) -> bool:
    try:
        os.makedirs(path, exist_ok=True)
        test_path = os.path.join(path, ".write_test")
        with open(test_path, "w") as f:
            f.write("ok")
        os.remove(test_path)
        return True
    except Exception:
        return False

def resolve_db_path(default_name: str = "artmash.sqlite3") -> str:
    env = os.getenv("ARTMASH_DB_PATH")
    if env:
        return env
    cwd = os.getcwd()
    if _is_writable_dir(cwd):
        return os.path.join(cwd, default_name)
    return os.path.join("/tmp", default_name)

def resolve_cache_dir(default_name: str = ".cache_artmash") -> str:
    env = os.getenv("ARTMASH_CACHE_DIR")
    if env:
        return env
    cwd = os.getcwd()
    if _is_writable_dir(cwd):
        return os.path.join(cwd, default_name)
    return os.path.join("/tmp", default_name)

DB_PATH = resolve_db_path()
CACHE_DIR = resolve_cache_dir()

# ----------------------------
# SQLite (robust)
# ----------------------------
def db() -> sqlite3.Connection:
    # timeout helps with brief write contention on Streamlit reruns
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn

def table_columns(conn: sqlite3.Connection, table: str) -> set:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cur.fetchall()}

def ensure_column(conn: sqlite3.Connection, table: str, col: str, decl: str):
    cols = table_columns(conn, table)
    if col in cols:
        return
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")

def migrate_schema(conn: sqlite3.Connection):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS paintings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE,
            img_url TEXT,
            title TEXT,
            artist TEXT,
            artist_url TEXT,
            meta TEXT,
            tags TEXT,
            elo REAL DEFAULT 1500.0,
            mu REAL DEFAULT 25.0,
            sigma REAL DEFAULT 8.333,
            games INTEGER DEFAULT 0,
            wins INTEGER DEFAULT 0,
            losses INTEGER DEFAULT 0,
            last_seen INTEGER DEFAULT 0,
            last_vote INTEGER DEFAULT 0
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS artists (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE,
            name TEXT,
            last_seen INTEGER DEFAULT 0
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS votes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts INTEGER,
            left_url TEXT,
            right_url TEXT,
            winner_url TEXT,
            mode TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ingest_state (
            key TEXT PRIMARY KEY,
            val TEXT
        )
        """
    )

    # Backfill for older DBs
    ensure_column(conn, "paintings", "artist_url", "TEXT")
    ensure_column(conn, "paintings", "tags", "TEXT")
    ensure_column(conn, "paintings", "mu", "REAL DEFAULT 25.0")
    ensure_column(conn, "paintings", "sigma", "REAL DEFAULT 8.333")
    ensure_column(conn, "paintings", "last_vote", "INTEGER DEFAULT 0")
    try:
        ensure_column(conn, "votes", "mode", "TEXT")
    except Exception:
        pass

    conn.execute("UPDATE paintings SET mu=? WHERE mu IS NULL", (TS_MU0,))
    conn.execute("UPDATE paintings SET sigma=? WHERE sigma IS NULL", (TS_SIGMA0,))
    conn.execute("UPDATE paintings SET last_vote=0 WHERE last_vote IS NULL")

    # Indexes
    conn.execute("CREATE INDEX IF NOT EXISTS idx_paintings_last_seen ON paintings(last_seen DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_paintings_last_vote ON paintings(last_vote DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_paintings_artist_url ON paintings(artist_url)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_votes_ts ON votes(ts DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_artists_url ON artists(url)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_artists_name ON artists(name)")
    conn.commit()

def init_db():
    conn = db()
    try:
        migrate_schema(conn)
    finally:
        conn.close()

# ----------------------------
# Small helpers
# ----------------------------
def now_ts() -> int:
    return int(time.time())

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def parse_artist_url_from_painting(painting_url: str) -> str:
    m = re.search(r"^https://gallerix\.org/storeroom/(\d+)/N/\d+/?$", painting_url)
    if not m:
        return ""
    return f"https://gallerix.org/storeroom/{m.group(1)}/"

def ensure_dirs():
    # cache is optional; if it fails, we simply skip caching
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        os.makedirs(os.path.join(CACHE_DIR, "html"), exist_ok=True)
    except Exception:
        pass

# ----------------------------
# Fetcher (HTML only, optional disk cache)
# ----------------------------
@dataclass
class FetchResult:
    ok: bool
    status: str
    content_type: str
    data: Optional[bytes]
    url: str
    from_cache: bool

def cache_path(kind: str, key: str) -> str:
    return os.path.join(CACHE_DIR, kind, key)

def fetch_bytes(
    url: str,
    *,
    user_agent: str,
    timeout: float,
    max_bytes: int,
    referer: Optional[str] = None,
    cache_kind: str = "html",
    cache_ttl_sec: int = 7 * 24 * 3600,
) -> FetchResult:
    ensure_dirs()
    key = sha1(url)
    path = cache_path(cache_kind, key)

    # read cache
    try:
        if os.path.exists(path):
            age = now_ts() - int(os.path.getmtime(path))
            if age <= cache_ttl_sec:
                data = open(path, "rb").read()
                return FetchResult(True, "cache", "", data, url, True)
    except Exception:
        pass

    headers = {"User-Agent": user_agent, "Accept": "text/html,*/*;q=0.8"}
    if referer:
        headers["Referer"] = referer

    try:
        req = Request(url, headers=headers)
        with urlopen(req, timeout=timeout) as resp:
            ctype = (resp.headers.get("Content-Type") or "")
            data = resp.read(max_bytes)

        # write cache (best effort)
        try:
            ensure_dirs()
            with open(path, "wb") as f:
                f.write(data)
        except Exception:
            pass

        return FetchResult(True, "ok", ctype, data, url, False)

    except HTTPError as e:
        return FetchResult(False, f"HTTP {e.code}", "", None, url, False)
    except URLError as e:
        return FetchResult(False, f"URL error: {getattr(e, 'reason', e)}", "", None, url, False)
    except Exception as e:
        return FetchResult(False, f"Error: {e}", "", None, url, False)

def fetch_text(
    url: str,
    *,
    user_agent: str,
    timeout: float,
    max_bytes: int,
    referer: Optional[str] = None,
) -> Tuple[Optional[str], FetchResult]:
    fr = fetch_bytes(
        url,
        user_agent=user_agent,
        timeout=timeout,
        max_bytes=max_bytes,
        referer=referer,
        cache_kind="html",
    )
    if not fr.ok or fr.data is None:
        return None, fr
    return fr.data.decode("utf-8", errors="replace"), fr

# ----------------------------
# HTML parsers
# ----------------------------
class HrefImgParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs: List[str] = []
        self.img_srcs: List[str] = []

    def handle_starttag(self, tag, attrs):
        t = tag.lower()
        a = {k.lower(): (v if v is not None else "") for k, v in attrs}
        if t == "a" and "href" in a:
            self.hrefs.append(a["href"])
        if t == "img" and "src" in a:
            self.img_srcs.append(a["src"])

class LinkTextParser(HTMLParser):
    """Captures <a href="...">TEXT</a> reliably (for artist names on letter pages)."""
    def __init__(self):
        super().__init__()
        self.links: List[Tuple[str, str]] = []
        self._in_a = False
        self._a_href = ""
        self._buf: List[str] = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            a = {k.lower(): (v if v is not None else "") for k, v in attrs}
            self._in_a = True
            self._a_href = a.get("href", "")
            self._buf = []

    def handle_data(self, data):
        if self._in_a and data:
            self._buf.append(data)

    def handle_endtag(self, tag):
        if tag.lower() == "a" and self._in_a:
            text = re.sub(r"\s+", " ", "".join(self._buf)).strip()
            href = self._a_href.strip()
            if href:
                self.links.append((href, text))
            self._in_a = False
            self._a_href = ""
            self._buf = []

def extract_links(html: str, base_url: str) -> Tuple[List[str], List[str]]:
    p = HrefImgParser()
    p.feed(html)

    def dedupe(xs):
        seen = set()
        out = []
        for x in xs:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    hrefs, imgs = [], []
    for h in p.hrefs:
        try:
            hrefs.append(urljoin(base_url, h))
        except Exception:
            pass
    for s in p.img_srcs:
        try:
            imgs.append(urljoin(base_url, s))
        except Exception:
            pass
    return dedupe(hrefs), dedupe(imgs)

def extract_link_texts(html: str, base_url: str) -> List[Tuple[str, str]]:
    p = LinkTextParser()
    p.feed(html)
    out: List[Tuple[str, str]] = []
    seen = set()
    for href, text in p.links:
        try:
            u = urljoin(base_url, href)
        except Exception:
            continue
        key = (u, text)
        if key in seen:
            continue
        seen.add(key)
        out.append((u, text))
    return out

# ----------------------------
# Painting page extraction (optional / lazy)
# ----------------------------
def extract_title_artist_meta_and_text(html: str) -> Tuple[str, str, str, str]:
    title, artist, meta = "", "", ""

    m = re.search(r"(?is)<h1[^>]*>(.*?)</h1>", html)
    if m:
        h1 = re.sub(r"(?is)<.*?>", " ", m.group(1))
        title = re.sub(r"\s+", " ", h1).strip()

    m = re.search(r"(?is)Painter:\s*<a[^>]*>(.*?)</a>", html)
    if m:
        artist = re.sub(r"(?is)<.*?>", " ", m.group(1))
        artist = re.sub(r"\s+", " ", artist).strip()

    m = re.search(r"(?m)^\s*(\d{3,4}\.\s*[^<]{0,140}cm\.)\s*$", html)
    if m:
        meta = m.group(1).strip()

    raw_text = re.sub(r"(?is)<script.*?</script>", " ", html)
    raw_text = re.sub(r"(?is)<style.*?</style>", " ", raw_text)
    raw_text = re.sub(r"(?is)<.*?>", " ", raw_text)
    raw_text = re.sub(r"\s+", " ", raw_text).strip()
    return title, artist, meta, raw_text

def extract_primary_image_url(html: str, page_url: str) -> Optional[str]:
    _, imgs = extract_links(html, page_url)
    for u in imgs:
        if u.lower().endswith((".webp", ".jpg", ".jpeg", ".png")):
            return u
    m = re.search(r"(https?://[^\"'\s>]+?\.(?:webp|jpg|jpeg|png))", html, re.I)
    return m.group(1) if m else None

def infer_style_tags(text: str) -> List[str]:
    t = (text or "").lower()
    tags = []
    for label, kws in STYLE_KEYWORDS.items():
        if any(k in t for k in kws):
            tags.append(label)
    return tags[:5]

# ----------------------------
# ingest_state helpers
# ----------------------------
def ingest_state_get(key: str, default: str = "") -> str:
    conn = db()
    try:
        migrate_schema(conn)
        cur = conn.execute("SELECT val FROM ingest_state WHERE key=?", (key,))
        r = cur.fetchone()
        return r[0] if r and r[0] is not None else default
    finally:
        conn.close()

def ingest_state_set(key: str, val: str):
    conn = db()
    try:
        migrate_schema(conn)
        conn.execute(
            """
            INSERT INTO ingest_state (key, val) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET val=excluded.val
            """,
            (key, val),
        )
        conn.commit()
    finally:
        conn.close()

# ----------------------------
# SQLite ops
# ----------------------------
def upsert_artist(url: str, name: str):
    """Upsert artist and backfill paintings.artist when missing."""
    url = (url or "").strip()
    name = (name or "").strip()
    if not url:
        return

    conn = db()
    try:
        migrate_schema(conn)
        conn.execute(
            """
            INSERT INTO artists (url, name, last_seen)
            VALUES (?, ?, ?)
            ON CONFLICT(url) DO UPDATE SET
                name=CASE
                    WHEN excluded.name IS NULL OR excluded.name=''
                    THEN artists.name
                    ELSE excluded.name
                END,
                last_seen=excluded.last_seen
            """,
            (url, name or None, now_ts()),
        )
        if name:
            conn.execute(
                """
                UPDATE paintings
                SET artist=?
                WHERE (artist IS NULL OR artist='')
                  AND artist_url=?
                """,
                (name, url),
            )
        conn.commit()
    finally:
        conn.close()

def get_artist_name(url: str) -> str:
    if not url:
        return ""
    conn = db()
    try:
        migrate_schema(conn)
        cur = conn.execute("SELECT name FROM artists WHERE url=?", (url,))
        r = cur.fetchone()
        return (r[0] or "").strip() if r else ""
    finally:
        conn.close()

def upsert_minimal_painting(url: str, artist_name: str = "", artist_url: str = "") -> bool:
    """
    Insert if missing; update artist fields if provided.
    Ensures artist_url is populated if parseable.
    """
    url = (url or "").strip()
    if not url:
        return False
    if not artist_url:
        artist_url = parse_artist_url_from_painting(url)

    conn = db()
    inserted = False
    try:
        migrate_schema(conn)
        cur = conn.execute(
            "INSERT OR IGNORE INTO paintings (url, artist, artist_url, last_seen) VALUES (?, ?, ?, ?)",
            (url, artist_name or None, artist_url or None, now_ts()),
        )
        inserted = (cur.rowcount == 1)

        conn.execute(
            """
            UPDATE paintings
            SET last_seen=?,
                artist=CASE
                    WHEN (artist IS NULL OR artist='') AND (? IS NOT NULL AND ?!='') THEN ?
                    ELSE artist
                END,
                artist_url=CASE
                    WHEN (artist_url IS NULL OR artist_url='') AND (? IS NOT NULL AND ?!='') THEN ?
                    ELSE artist_url
                END
            WHERE url=?
            """,
            (now_ts(), artist_name, artist_name, artist_name, artist_url, artist_url, artist_url, url),
        )
        conn.commit()
    finally:
        conn.close()
    return inserted

def upsert_painting_full(url: str, img_url: str, title: str, artist: str, artist_url: str, meta: str, tags: List[str]):
    conn = db()
    try:
        migrate_schema(conn)
        conn.execute(
            """
            INSERT INTO paintings (url, img_url, title, artist, artist_url, meta, tags, elo, mu, sigma, last_seen, last_vote)
            VALUES (?, ?, ?, ?, ?, ?, ?, 1500.0, ?, ?, ?, COALESCE((SELECT last_vote FROM paintings WHERE url=?), 0))
            ON CONFLICT(url) DO UPDATE SET
                img_url=CASE WHEN paintings.img_url IS NULL OR paintings.img_url='' THEN excluded.img_url ELSE paintings.img_url END,
                title=CASE WHEN paintings.title IS NULL OR paintings.title='' THEN excluded.title ELSE paintings.title END,
                artist=CASE WHEN paintings.artist IS NULL OR paintings.artist='' THEN excluded.artist ELSE paintings.artist END,
                artist_url=CASE WHEN paintings.artist_url IS NULL OR paintings.artist_url='' THEN excluded.artist_url ELSE paintings.artist_url END,
                meta=CASE WHEN paintings.meta IS NULL OR paintings.meta='' THEN excluded.meta ELSE paintings.meta END,
                tags=CASE WHEN paintings.tags IS NULL OR paintings.tags='' THEN excluded.tags ELSE paintings.tags END,
                last_seen=excluded.last_seen
            """,
            (
                url,
                img_url or None,
                title or None,
                artist or None,
                artist_url or None,
                meta or None,
                json.dumps(tags) if tags else None,
                TS_MU0,
                TS_SIGMA0,
                now_ts(),
                url,
            ),
        )
        conn.commit()
    finally:
        conn.close()

def get_pool(limit: int = 4000) -> List[Dict]:
    conn = db()
    try:
        migrate_schema(conn)
        cur = conn.execute(
            """
            SELECT url, img_url, title, artist, artist_url, meta, tags,
                   elo, mu, sigma, games, wins, losses, last_seen, last_vote
            FROM paintings
            ORDER BY last_seen DESC
            LIMIT ?
            """,
            (int(limit),),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    out = []
    for r in rows:
        out.append(
            dict(
                url=r[0],
                img_url=r[1] or "",
                title=(r[2] or "").strip(),
                artist=(r[3] or "").strip(),
                artist_url=(r[4] or "").strip(),
                meta=r[5] or "",
                tags=json.loads(r[6]) if r[6] else [],
                elo=float(r[7] or 1500.0),
                mu=float(r[8] or TS_MU0),
                sigma=float(r[9] or TS_SIGMA0),
                games=int(r[10] or 0),
                wins=int(r[11] or 0),
                losses=int(r[12] or 0),
                last_seen=int(r[13] or 0),
                last_vote=int(r[14] or 0),
            )
        )
    return out

def get_painting(url: str) -> Optional[Dict]:
    conn = db()
    try:
        migrate_schema(conn)
        cur = conn.execute(
            """
            SELECT url, img_url, title, artist, artist_url, meta, tags,
                   elo, mu, sigma, games, wins, losses, last_seen, last_vote
            FROM paintings WHERE url=?
            """,
            (url,),
        )
        r = cur.fetchone()
    finally:
        conn.close()
    if not r:
        return None
    return dict(
        url=r[0],
        img_url=r[1] or "",
        title=(r[2] or "").strip(),
        artist=(r[3] or "").strip(),
        artist_url=(r[4] or "").strip(),
        meta=r[5] or "",
        tags=json.loads(r[6]) if r[6] else [],
        elo=float(r[7] or 1500.0),
        mu=float(r[8] or TS_MU0),
        sigma=float(r[9] or TS_SIGMA0),
        games=int(r[10] or 0),
        wins=int(r[11] or 0),
        losses=int(r[12] or 0),
        last_seen=int(r[13] or 0),
        last_vote=int(r[14] or 0),
    )

def record_vote(left_url: str, right_url: str, winner_url: str, mode: str):
    conn = db()
    try:
        migrate_schema(conn)
        conn.execute(
            "INSERT INTO votes (ts, left_url, right_url, winner_url, mode) VALUES (?, ?, ?, ?, ?)",
            (now_ts(), left_url, right_url, winner_url, mode),
        )
        conn.commit()
    finally:
        conn.close()

# ----------------------------
# Crawling helpers
# ----------------------------
def extract_artist_pairs_from_letter_page(html: str, letter_page_url: str) -> List[Tuple[str, str]]:
    """Returns (artist_url, artist_name) using anchor text. Filters only storeroom artist URLs."""
    pairs = extract_link_texts(html, letter_page_url)
    out: List[Tuple[str, str]] = []
    seen = set()
    for u, txt in pairs:
        if ARTIST_URL_RE.match(u):
            au = u.rstrip("/") + "/"
            name = (txt or "").strip()
            # skip letter navigation etc
            if name and len(name) <= 2 and name.upper() in LATIN_LETTERS:
                continue
            key = (au, name.lower())
            if key in seen:
                continue
            seen.add(key)
            out.append((au, name))
    return out

def extract_painting_urls_from_artist_page(html: str, artist_url: str) -> List[str]:
    links, _ = extract_links(html, artist_url)
    pics = []
    for u in links:
        uu = u.rstrip("/") + "/"
        if PAINTING_URL_RE.match(uu):
            pics.append(uu)
    # stable dedupe
    seen, out = set(), []
    for p in pics:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out

def load_all_artists_batch(user_agent: str, timeout: float, delay: float, letters_per_run: int) -> Tuple[int, int, str]:
    """
    Loads artists from letter pages A..Z.
    Progress stored in ingest_state key 'artist_letter_pos' (0..26).
    """
    pos = int(ingest_state_get("artist_letter_pos", "0") or "0")
    start = max(0, min(pos, len(LATIN_LETTERS)))
    end = min(len(LATIN_LETTERS), start + max(1, int(letters_per_run)))

    added, updated = 0, 0
    dbg = []

    for i in range(start, end):
        L = LATIN_LETTERS[i]
        url = LETTER_URL.format(L=L)
        html, fr = fetch_text(url, user_agent=user_agent, timeout=timeout, max_bytes=3_000_000, referer=STOREROOM_ROOT)
        if not html:
            dbg.append(f"{L}:fail({fr.status})")
            if delay > 0:
                time.sleep(delay)
            continue

        pairs = extract_artist_pairs_from_letter_page(html, url)
        dbg.append(f"{L}:{len(pairs)}")

        for au, name in pairs:
            # detect insert vs update count cheaply by checking existence
            conn = db()
            try:
                migrate_schema(conn)
                cur = conn.execute("SELECT 1 FROM artists WHERE url=? LIMIT 1", (au,))
                exists = cur.fetchone() is not None
            finally:
                conn.close()

            upsert_artist(au, name)
            if exists:
                updated += 1
            else:
                added += 1

        if delay > 0:
            time.sleep(delay)

    ingest_state_set("artist_letter_pos", str(end))
    return added, updated, " | ".join(dbg) if dbg else "no-op"

def load_all_paintings_batch(
    user_agent: str,
    timeout: float,
    delay: float,
    artists_per_run: int,
    paintings_cap_per_artist: int,
) -> Tuple[int, int, int, str]:
    """
    Crawls artists from the artists table (no giant JSON stored).
    Progress stored in ingest_state key 'last_artist_id' (cursor by artists.id).
    """
    last_id = int(ingest_state_get("last_artist_id", "0") or "0")

    conn = db()
    try:
        migrate_schema(conn)
        total_artists = conn.execute("SELECT COUNT(*) FROM artists").fetchone()[0]
        cur = conn.execute(
            "SELECT id, url, COALESCE(name,'') FROM artists WHERE id > ? ORDER BY id ASC LIMIT ?",
            (last_id, int(artists_per_run)),
        )
        artist_rows = cur.fetchall()
    finally:
        conn.close()

    if total_artists == 0:
        return 0, 0, 0, "No artists loaded yet. Run step 1 (Load artists) first."
    if not artist_rows:
        return 0, 0, 0, "No remaining artists in this crawl. (You're done!)"

    unique_inserts = 0
    artists_done = 0
    missing_names = 0
    dbg = []

    max_seen_id = last_id
    for artist_id, artist_url, name in artist_rows:
        max_seen_id = max(max_seen_id, int(artist_id))
        html, fr = fetch_text(artist_url, user_agent=user_agent, timeout=timeout, max_bytes=4_000_000, referer=STOREROOM_ROOT)
        if not html:
            dbg.append(f"artist:fail({fr.status})")
            if delay > 0:
                time.sleep(delay)
            continue

        name = (name or "").strip()
        if not name:
            m = re.search(r"(?is)<h1[^>]*>(.*?)</h1>", html)
            if m:
                h1 = re.sub(r"(?is)<.*?>", " ", m.group(1))
                name = re.sub(r"\s+", " ", h1).strip()
                if name:
                    upsert_artist(artist_url, name)

        if not name:
            missing_names += 1

        pics = extract_painting_urls_from_artist_page(html, artist_url)
        if paintings_cap_per_artist > 0:
            pics = pics[: int(paintings_cap_per_artist)]

        for pu in pics:
            if upsert_minimal_painting(pu, artist_name=name, artist_url=artist_url):
                unique_inserts += 1

        artists_done += 1
        dbg.append(f"{(name[:18]+'…') if name else 'Unknown'}:{len(pics)}")

        if delay > 0:
            time.sleep(delay)

    ingest_state_set("last_artist_id", str(max_seen_id))
    status = " | ".join(dbg[:8]) + (" | …" if len(dbg) > 8 else "")
    return unique_inserts, artists_done, missing_names, status

# ----------------------------
# Vote / rating system
# ----------------------------
def normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def trueskill_lite_update(
    mu_a: float,
    sig_a: float,
    mu_b: float,
    sig_b: float,
    a_wins: bool,
    beta: float = TS_BETA,
    tau: float = TS_TAU,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    # dynamics
    sig_a = math.sqrt(sig_a * sig_a + tau * tau)
    sig_b = math.sqrt(sig_b * sig_b + tau * tau)

    c2 = 2 * beta * beta + sig_a * sig_a + sig_b * sig_b
    c = math.sqrt(c2) + 1e-12

    t = (mu_a - mu_b) / c
    if not a_wins:
        t = -t

    Phi = normal_cdf(t)
    phi = normal_pdf(t)
    v = phi / max(Phi, 1e-12)
    w = v * (v + t)

    delta_mu_a = (sig_a * sig_a / c) * v
    delta_mu_b = (sig_b * sig_b / c) * v

    if a_wins:
        mu_a_new = mu_a + delta_mu_a
        mu_b_new = mu_b - delta_mu_b
    else:
        mu_a_new = mu_a - delta_mu_a
        mu_b_new = mu_b + delta_mu_b

    sig_a2 = sig_a * sig_a * (1.0 - (sig_a * sig_a / c2) * w)
    sig_b2 = sig_b * sig_b * (1.0 - (sig_b * sig_b / c2) * w)

    return (mu_a_new, math.sqrt(max(sig_a2, 1e-6))), (mu_b_new, math.sqrt(max(sig_b2, 1e-6)))

def k_factor(games: int) -> float:
    if games < 10:
        return 48.0
    if games < 50:
        return 28.0
    return 16.0

def elo_expected(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))

def elo_update(r_a: float, r_b: float, score_a: float, k: float) -> Tuple[float, float]:
    ea = elo_expected(r_a, r_b)
    eb = 1.0 - ea
    return (r_a + k * (score_a - ea), r_b + k * ((1.0 - score_a) - eb))

def mu_sigma_to_value(mu: float, sigma: float) -> float:
    return float(mu) - 3.0 * float(sigma)

def value_score_0_100(v: float) -> float:
    return clamp((v / 40.0) * 100.0, 0.0, 100.0)

def apply_vote(winner_url: str, loser_url: str):
    # Always updates winner positively, loser negatively
    w = get_painting(winner_url)
    l = get_painting(loser_url)
    if not w or not l:
        return

    k = (k_factor(int(w["games"])) + k_factor(int(l["games"]))) / 2.0
    new_rw, new_rl = elo_update(float(w["elo"]), float(l["elo"]), 1.0, k)
    (mu_w, sig_w), (mu_l, sig_l) = trueskill_lite_update(float(w["mu"]), float(w["sigma"]), float(l["mu"]), float(l["sigma"]), True)

    ts = now_ts()

    # retry on transient lock
    for attempt in range(3):
        try:
            conn = db()
            try:
                migrate_schema(conn)
                conn.execute(
                    """
                    UPDATE paintings
                    SET elo=?, mu=?, sigma=?, games=games+1, wins=wins+1, last_vote=?
                    WHERE url=?
                    """,
                    (new_rw, mu_w, sig_w, ts, winner_url),
                )
                conn.execute(
                    """
                    UPDATE paintings
                    SET elo=?, mu=?, sigma=?, games=games+1, losses=losses+1, last_vote=?
                    WHERE url=?
                    """,
                    (new_rl, mu_l, sig_l, ts, loser_url),
                )
                conn.commit()
            finally:
                conn.close()
            break
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() and attempt < 2:
                time.sleep(0.15 * (attempt + 1))
                continue
            raise

# ----------------------------
# Leaderboards (live queries)
# ----------------------------
def paintings_leaderboard_live(limit: int, min_games: int) -> List[Dict]:
    conn = db()
    try:
        migrate_schema(conn)
        cur = conn.execute(
            """
            SELECT url,
                   COALESCE(title,'') AS title,
                   COALESCE(artist,'') AS artist,
                   COALESCE(mu, ?) AS mu,
                   COALESCE(sigma, ?) AS sigma,
                   COALESCE(games,0) AS games,
                   COALESCE(wins,0) AS wins,
                   COALESCE(losses,0) AS losses,
                   COALESCE(last_vote,0) AS last_vote
            FROM paintings
            WHERE COALESCE(games,0) >= ?
            ORDER BY (COALESCE(mu, ?) - 3*COALESCE(sigma, ?)) DESC,
                     COALESCE(games,0) DESC,
                     COALESCE(last_vote,0) DESC
            LIMIT ?
            """,
            (TS_MU0, TS_SIGMA0, int(min_games), TS_MU0, TS_SIGMA0, int(limit)),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    out = []
    for i, r in enumerate(rows, 1):
        v = float(r[3]) - 3.0 * float(r[4])
        out.append(
            {
                "rank": i,
                "score_0_100": round(value_score_0_100(v), 1),
                "value": round(v, 4),
                "mu": round(float(r[3]), 4),
                "sigma": round(float(r[4]), 4),
                "games": int(r[5]),
                "wins": int(r[6]),
                "losses": int(r[7]),
                "artist": (r[2] or "").strip(),
                "title": (r[1] or "").strip(),
                "url": r[0],
            }
        )
    return out

def artists_leaderboard_live(
    limit: int,
    topk: int,
    min_artist_games: int,
    min_painting_games: int,
    include_unknown: bool,
) -> List[Dict]:
    """
    Canonical grouping by artist_url (stable).
    Display name uses paintings.artist, else artists.name, else URL.
    Artist score = mean of top-k painting conservative values.
    """
    conn = db()
    try:
        migrate_schema(conn)
        cur = conn.execute(
            """
            SELECT
                COALESCE(p.artist_url,'') AS artist_url,
                COALESCE(NULLIF(TRIM(p.artist),''), a.name, '') AS artist_name,
                p.url,
                COALESCE(NULLIF(TRIM(p.title),''),'') AS title,
                COALESCE(p.mu, ?) AS mu,
                COALESCE(p.sigma, ?) AS sigma,
                COALESCE(p.games,0) AS games
            FROM paintings p
            LEFT JOIN artists a ON a.url = p.artist_url
            WHERE COALESCE(p.games,0) >= ?
            """,
            (TS_MU0, TS_SIGMA0, int(min_painting_games)),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    buckets: Dict[str, Dict] = {}
    for artist_url, artist_name, url, title, mu, sigma, games in rows:
        artist_url = (artist_url or "").strip()
        if not artist_url:
            if not include_unknown:
                continue
            bucket_key = "unknown"
            display_name = (artist_name or "").strip() or "Unknown artist"
        else:
            bucket_key = artist_url
            display_name = (artist_name or "").strip() or get_artist_name(artist_url) or artist_url

        if bucket_key not in buckets:
            buckets[bucket_key] = {"artist": display_name, "artist_url": artist_url, "items": [], "games_total": 0}

        v = float(mu) - 3.0 * float(sigma)
        buckets[bucket_key]["items"].append((v, int(games), url, title))
        buckets[bucket_key]["games_total"] += int(games)

    agg_rows = []
    for b in buckets.values():
        if int(b["games_total"]) < int(min_artist_games):
            continue
        items = sorted(b["items"], key=lambda t: (t[0], t[1]), reverse=True)
        top = items[: max(1, int(topk))]
        vals = [t[0] for t in top]
        score = sum(vals) / max(1, len(vals))
        best = top[0] if top else None
        agg_rows.append(
            {
                "artist": b["artist"],
                "artist_url": b["artist_url"],
                "value": float(score),
                "score_0_100": float(value_score_0_100(score)),
                "games": int(b["games_total"]),
                "paintings": len(b["items"]),
                "best_url": best[2] if best else "",
                "best_title": best[3] if best else "",
            }
        )

    agg_rows.sort(key=lambda r: (float(r["value"]), int(r["games"]), int(r["paintings"])), reverse=True)
    out = []
    for i, r in enumerate(agg_rows[: int(limit)], 1):
        out.append(
            {
                "rank": i,
                "artist": r["artist"],
                "score_0_100": round(float(r["score_0_100"]), 1),
                "value": round(float(r["value"]), 4),
                "games": int(r["games"]),
                "paintings": int(r["paintings"]),
                "best_title": r["best_title"],
                "best_url": r["best_url"],
                "artist_url": r["artist_url"],
            }
        )
    return out

# ----------------------------
# Rendering
# ----------------------------
def render_painting_display(p: Dict, display_mode: str, height: int = 620):
    url = p["url"]
    img_url = p.get("img_url") or ""
    if display_mode == "iframe" or not img_url:
        components.iframe(url, height=height, scrolling=True)
        return
    html = f"""
    <div style="width:100%; height:{height}px; display:flex; align-items:center; justify-content:center; background:#111; border-radius:12px; overflow:hidden;">
      <img src="{img_url}" style="max-width:100%; max-height:100%; object-fit:contain;" />
    </div>
    """
    components.html(html, height=height, scrolling=False)

# ----------------------------
# Optional enrichment
# ----------------------------
def ingest_painting_page_full(painting_url: str, user_agent: str, timeout: float, delay: float) -> bool:
    html, _ = fetch_text(painting_url, user_agent=user_agent, timeout=timeout, max_bytes=2_500_000, referer=painting_url)
    if not html:
        return False

    title, artist_from_page, meta, text = extract_title_artist_meta_and_text(html)
    img_url = extract_primary_image_url(html, painting_url) or ""
    tags = infer_style_tags(text)

    artist_url = parse_artist_url_from_painting(painting_url)
    artist_name = (artist_from_page or "").strip()
    known = get_artist_name(artist_url) if artist_url else ""
    if known:
        artist_name = known
    if artist_url and artist_name:
        upsert_artist(artist_url, artist_name)

    upsert_painting_full(painting_url, img_url, title, artist_name, artist_url, meta, tags)

    if delay > 0:
        time.sleep(delay)
    return True

# ----------------------------
# Queue + locked matchup logic (session)
# ----------------------------
def build_session_queue(limit: int, seed: int) -> List[str]:
    pool = get_pool(limit=limit)
    urls = [p["url"] for p in pool]
    rng = random.Random(seed)
    rng.shuffle(urls)
    return urls

def next_from_queue() -> Optional[str]:
    q = st.session_state.get("queue_urls", [])
    idx = st.session_state.get("queue_idx", 0)
    if not q:
        return None
    url = q[idx % len(q)]
    st.session_state["queue_idx"] = (idx + 1) % len(q)
    return url

def pick_pair_from_queue() -> Optional[Tuple[Dict, Dict]]:
    a_url = next_from_queue()
    b_url = next_from_queue()
    if not a_url or not b_url or a_url == b_url:
        return None
    A = get_painting(a_url)
    B = get_painting(b_url)
    if not A or not B:
        return None
    return A, B

def set_current_pair(left_url: str, right_url: str):
    st.session_state["current_left_url"] = left_url
    st.session_state["current_right_url"] = right_url

def get_current_pair_urls() -> Tuple[Optional[str], Optional[str]]:
    return st.session_state.get("current_left_url"), st.session_state.get("current_right_url")

def clear_current_pair():
    st.session_state.pop("current_left_url", None)
    st.session_state.pop("current_right_url", None)

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title=APP_NAME, page_icon="🖼️", layout="wide")
init_db()

st.title("🖼️ Art Mash")
st.caption("Votes persist to SQLite and leaderboards update live after every vote. Voting is pair-locked to prevent rerun mixups.")

with st.sidebar:
    st.header("Queue")
    queue_limit = st.slider("Queue size (from DB)", 200, 20000, 4000, 200, key="sb_queue_limit")
    queue_seed = st.number_input("Queue seed", 0, 10_000_000, 1337, 1, key="sb_queue_seed")
    rebuild_queue = st.button("🔀 Rebuild session queue", use_container_width=True, key="sb_rebuild_queue")

    st.divider()
    st.header("Display")
    display_mode = st.selectbox("Display mode", ["iframe (reliable)", "img tag (fast)"], index=0, key="sb_disp_mode")
    disp_mode_key = "iframe" if display_mode.startswith("iframe") else "img"
    iframe_height = st.slider("Display height", 360, 900, 620, 10, key="sb_iframe_h")
    show_meta = st.checkbox("Show metadata", value=True, key="sb_show_meta")

    with st.expander("Diagnostics", expanded=False):
        st.caption(f"DB: {DB_PATH}")
        st.caption(f"Cache: {CACHE_DIR}")

tabs = st.tabs(["Vote", "Bulk Load", "Leaderboards"])

# Session init
if rebuild_queue or "queue_urls" not in st.session_state:
    st.session_state["queue_urls"] = build_session_queue(limit=int(queue_limit), seed=int(queue_seed))
    st.session_state["queue_idx"] = 0
    clear_current_pair()

if "seen_urls" not in st.session_state:
    st.session_state["seen_urls"] = set()

# ----------------------------
# Vote tab (pair-locked)
# ----------------------------
with tabs[0]:
    st.write(
        f"Session queue size: **{len(st.session_state['queue_urls'])}**  |  "
        f"Queue position: **{st.session_state.get('queue_idx',0)}**"
    )

    left_url, right_url = get_current_pair_urls()

    if not left_url or not right_url or left_url == right_url:
        pair = pick_pair_from_queue()
        if not pair:
            st.warning("Not enough paintings in the queue. Use Bulk Load to ingest more paintings, or increase queue size.")
            st.stop()
        left, right = pair
        set_current_pair(left["url"], right["url"])
    else:
        left = get_painting(left_url)
        right = get_painting(right_url)
        if not left or not right:
            pair = pick_pair_from_queue()
            if not pair:
                st.warning("Not enough paintings in the queue. Use Bulk Load to ingest more paintings, or increase queue size.")
                st.stop()
            left, right = pair
            set_current_pair(left["url"], right["url"])

    st.session_state["seen_urls"].add(left["url"])
    st.session_state["seen_urls"].add(right["url"])

    # Optional lazy enrich for img-tag mode
    if disp_mode_key == "img":
        if (not left.get("img_url")) or (not left.get("title")) or (not left.get("artist")):
            ingest_painting_page_full(left["url"], DEFAULT_UA, float(DEFAULT_TIMEOUT), 0.0)
            left = get_painting(left["url"]) or left
        if (not right.get("img_url")) or (not right.get("title")) or (not right.get("artist")):
            ingest_painting_page_full(right["url"], DEFAULT_UA, float(DEFAULT_TIMEOUT), 0.0)
            right = get_painting(right["url"]) or right

    colL, colR = st.columns(2, gap="large")

    def card(col, p: Dict, label: str):
        with col:
            st.subheader(label)
            render_painting_display(p, disp_mode_key, height=int(iframe_height))
            if show_meta:
                v = mu_sigma_to_value(p["mu"], p["sigma"])
                st.markdown(
                    f"**Score:** `{value_score_0_100(v):.1f}/100`  |  "
                    f"**μ/σ:** `{p['mu']:.2f}/{p['sigma']:.2f}`  |  "
                    f"**Games:** `{p['games']}`"
                )
                if p.get("title"):
                    st.markdown(f"**Title:** {p['title']}")
                if p.get("artist"):
                    st.markdown(f"**Artist:** {p['artist']}")
                st.markdown(f"[Open]({p['url']})")

    card(colL, left, "A")
    card(colR, right, "B")

    b1, b2, b3 = st.columns([1, 1, 1])
    vote_a = b1.button("✅ Vote A", use_container_width=True, key="vote_a_btn")
    vote_b = b2.button("✅ Vote B", use_container_width=True, key="vote_b_btn")
    skip = b3.button("↩ Skip", use_container_width=True, key="vote_skip_btn")

    locked_left_url, locked_right_url = get_current_pair_urls()

    if vote_a and locked_left_url and locked_right_url:
        record_vote(locked_left_url, locked_right_url, locked_left_url, mode="vote")
        apply_vote(locked_left_url, locked_right_url)
        clear_current_pair()
        st.rerun()

    if vote_b and locked_left_url and locked_right_url:
        record_vote(locked_left_url, locked_right_url, locked_right_url, mode="vote")
        apply_vote(locked_right_url, locked_left_url)
        clear_current_pair()
        st.rerun()

    if skip:
        clear_current_pair()
        st.rerun()

# ----------------------------
# Bulk Load tab
# ----------------------------
with tabs[1]:
    st.subheader("Bulk Loader (Resumable)")
    st.write(
        "Step 1 loads artists from A..Z letter pages and stores artist **names**. "
        "Step 2 crawls each artist page to store painting URLs and attaches those names."
    )

    with st.expander("Loader settings", expanded=True):
        user_agent = st.text_input("User-Agent", value=DEFAULT_UA, key="bl_ua")
        timeout = st.slider("Timeout (sec)", 3, 30, int(DEFAULT_TIMEOUT), 1, key="bl_timeout")
        crawl_delay = st.slider("Crawl delay (sec)", 0.0, 2.0, float(DEFAULT_DELAY), 0.05, key="bl_delay")

        cA, cB, cC = st.columns(3)
        with cA:
            letters_per_run = st.slider("Letters per run (artists)", 1, 26, 6, 1, key="bl_letters")
        with cB:
            artists_per_run = st.slider("Artists per run (paintings)", 1, 500, 50, 1, key="bl_artists_per")
        with cC:
            cap_per_artist = st.slider("Cap paintings per artist (0 = no cap)", 0, 5000, 0, 50, key="bl_cap")

        st.caption(f"DB path: {DB_PATH}")
        st.caption(f"Cache dir: {CACHE_DIR}")

    # progress + stats (from DB, not ingest_state JSON)
    conn = db()
    try:
        migrate_schema(conn)
        artist_count = conn.execute("SELECT COUNT(*) FROM artists").fetchone()[0]
        painting_count = conn.execute("SELECT COUNT(*) FROM paintings").fetchone()[0]
    finally:
        conn.close()

    letter_pos = int(ingest_state_get("artist_letter_pos", "0") or "0")
    last_artist_id = int(ingest_state_get("last_artist_id", "0") or "0")

    st.info(
        f"Artists in DB: **{artist_count}** | Paintings in DB: **{painting_count}** | "
        f"Letter progress: **{letter_pos}/26** | Artist crawl cursor id: **{last_artist_id}**"
    )

    c1, c2 = st.columns(2, gap="large")

    with c1:
        if st.button("1) Load artists (batch)", type="primary", use_container_width=True, key="bl_load_artists"):
            added, updated, status = load_all_artists_batch(
                user_agent=user_agent,
                timeout=float(timeout),
                delay=float(crawl_delay),
                letters_per_run=int(letters_per_run),
            )
            st.success(f"Added {added} artists; updated {updated} rows. {status}")
            st.rerun()

        if st.button("Reset letter progress", use_container_width=True, key="bl_reset_letters"):
            ingest_state_set("artist_letter_pos", "0")
            st.warning("Reset letter progress to 0.")
            st.rerun()

    with c2:
        if st.button("2) Load paintings (batch)", type="primary", use_container_width=True, key="bl_load_paintings"):
            unique_inserts, artists_done, missing_names, status = load_all_paintings_batch(
                user_agent=user_agent,
                timeout=float(timeout),
                delay=float(crawl_delay),
                artists_per_run=int(artists_per_run),
                paintings_cap_per_artist=int(cap_per_artist),
            )
            st.success(
                f"Inserted {unique_inserts} new painting URLs from {artists_done} artists. "
                f"Missing names: {missing_names}. {status}"
            )
            st.session_state["queue_urls"] = build_session_queue(limit=int(queue_limit), seed=int(queue_seed))
            st.session_state["queue_idx"] = 0
            clear_current_pair()
            st.rerun()

        if st.button("Reset artist crawl cursor", use_container_width=True, key="bl_reset_artist_cursor"):
            ingest_state_set("last_artist_id", "0")
            st.warning("Reset artist crawl cursor to 0.")
            st.rerun()

    st.divider()
    st.subheader("Optional: enrich a random sample (titles / img URLs)")
    st.write("Fetches ONLY HTML for a sample of painting pages to improve metadata. Still no image downloads server-side.")

    enrich_n = st.slider("Enrich N paintings now", 0, 200, 20, 5, key="bl_enrich_n")
    if st.button("Enrich sample", use_container_width=True, key="bl_enrich_btn") and enrich_n > 0:
        pool = get_pool(limit=max(500, enrich_n * 10))
        rng = random.Random(int(queue_seed) ^ (now_ts() // 10))
        rng.shuffle(pool)
        sample = pool[: int(enrich_n)]
        prog = st.progress(0)
        ok = 0
        for i, p in enumerate(sample):
            if ingest_painting_page_full(p["url"], user_agent, float(timeout), float(crawl_delay)):
                ok += 1
            prog.progress(int(100 * (i + 1) / max(1, len(sample))))
        st.success(f"Enriched {ok}/{len(sample)} paintings.")
        st.rerun()

# ----------------------------
# Leaderboards tab
# ----------------------------
with tabs[2]:
    st.subheader("Leaderboards (Live)")

    top_cols = st.columns(2)
    with top_cols[0]:
        top_p = st.slider("Top N paintings", 25, 1000, 200, 25, key="lb_top_paintings")
    with top_cols[1]:
        min_pg = st.slider("Min games per painting", 0, 50, 1, 1, key="lb_min_games_painting")

    prow = paintings_leaderboard_live(limit=int(top_p), min_games=int(min_pg))
    st.markdown("### 🏆 Top Paintings")
    st.dataframe(prow, use_container_width=True, height=520)

    st.download_button(
        "Download paintings leaderboard JSON",
        data=json.dumps(prow, indent=2).encode("utf-8"),
        file_name="artmash_paintings_leaderboard.json",
        mime="application/json",
        use_container_width=True,
        key="lb_dl_paintings",
    )

    st.divider()
    st.markdown("### 🎨 Top Artists (Canonical by artist URL)")

    a1, a2, a3, a4 = st.columns(4)
    with a1:
        top_a = st.slider("Top N artists", 25, 1000, 200, 25, key="lb_top_artists")
    with a2:
        topk = st.slider("Aggregate top-k paintings", 1, 20, 5, 1, key="lb_topk")
    with a3:
        min_ag = st.slider("Min total games per artist", 0, 500, 5, 1, key="lb_min_artist_games")
    with a4:
        min_paint_games = st.slider("Min games per painting (artist agg)", 0, 50, 1, 1, key="lb_min_paint_games")

    include_unknown = st.checkbox("Include Unknown artist bucket", value=False, key="lb_include_unknown")

    arow = artists_leaderboard_live(
        limit=int(top_a),
        topk=int(topk),
        min_artist_games=int(min_ag),
        min_painting_games=int(min_paint_games),
        include_unknown=bool(include_unknown),
    )
    st.dataframe(arow, use_container_width=True, height=520)

    st.download_button(
        "Download artist leaderboard JSON",
        data=json.dumps(arow, indent=2).encode("utf-8"),
        file_name="artmash_artist_leaderboard.json",
        mime="application/json",
        use_container_width=True,
        key="lb_dl_artists",
    )

st.caption("Tip: If images do not show in img-tag mode (hotlink restrictions), use iframe mode.")
