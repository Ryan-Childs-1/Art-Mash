# app.py
# Art Mash — Storeroom Randomized + Full Index Loader + Leaderboards (Artist NAME-based)
#
# ✅ FIXED per request:
# - Leaderboards + UI now key artists by *artist name* (not artist_id)
# - Bulk loader now actually *captures artist names* from letter pages (anchor text)
# - Adds an `artists` table to persist (artist_url -> artist_name) mapping
# - When loading paintings, we attach artist_name from the artists table
#
# ✅ Improvements:
# - Robust anchor-text parsing for letter pages (previous code only captured hrefs)
# - Artist leaderboard excludes unknown/blank artists by default (toggle to include)
# - Safer ingestion: better diagnostics + optional per-run caps
# - Vote view shows artist name cleanly (no IDs)
# - Leaderboards sort in real-time by conservative TrueSkill: mu - 3*sigma (and tie-breaks)
#
# Still:
# - No server-side image downloading (iframe or browser-side <img>)
# - Resumable batch ingest for Streamlit Cloud
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

DEFAULT_UA = "ArtMash/2.4 (Streamlit; respectful crawler)"
CACHE_DIR = ".cache_artmash"

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

# Storeroom URL patterns
ARTIST_URL_RE = re.compile(r"^https://gallerix\.org/storeroom/(\d+)/?$")
PAINTING_URL_RE = re.compile(r"^https://gallerix\.org/storeroom/\d+/N/\d+/?$")


# ----------------------------
# DB path + migrations
# ----------------------------
def resolve_db_path(default_name: str = "artmash.sqlite3") -> str:
    env = os.getenv("ARTMASH_DB_PATH")
    if env:
        return env
    try:
        test_path = os.path.join(os.getcwd(), ".write_test")
        with open(test_path, "w") as f:
            f.write("ok")
        os.remove(test_path)
        return default_name
    except Exception:
        return os.path.join("/tmp", default_name)


DB_PATH = resolve_db_path()


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
    # NEW: artists table for mapping url -> name
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

    ensure_column(conn, "paintings", "artist_url", "TEXT")
    ensure_column(conn, "paintings", "tags", "TEXT")
    ensure_column(conn, "paintings", "mu", "REAL DEFAULT 25.0")
    ensure_column(conn, "paintings", "sigma", "REAL DEFAULT 8.333")
    ensure_column(conn, "paintings", "last_vote", "INTEGER DEFAULT 0")

    try:
        ensure_column(conn, "votes", "mode", "TEXT")
    except Exception:
        pass

    conn.execute("UPDATE paintings SET mu=25.0 WHERE mu IS NULL")
    conn.execute("UPDATE paintings SET sigma=8.333 WHERE sigma IS NULL")
    conn.execute("UPDATE paintings SET last_vote=0 WHERE last_vote IS NULL")
    conn.commit()


def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db():
    conn = db()
    try:
        migrate_schema(conn)
    finally:
        conn.close()


# ----------------------------
# Utilities
# ----------------------------
def ensure_dirs():
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(os.path.join(CACHE_DIR, "html"), exist_ok=True)


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def now_ts() -> int:
    return int(time.time())


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def parse_artist_url_from_painting(painting_url: str) -> str:
    # https://gallerix.org/storeroom/{artist_id}/N/{painting_id}/ -> artist url:
    m = re.search(r"^https://gallerix\.org/storeroom/(\d+)/N/\d+/?$", painting_url)
    if not m:
        return ""
    return f"https://gallerix.org/storeroom/{m.group(1)}/"


# ----------------------------
# Disk cache fetcher (HTML only)
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

    if os.path.exists(path):
        age = now_ts() - int(os.path.getmtime(path))
        if age <= cache_ttl_sec:
            try:
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

        try:
            with open(path, "wb") as f:
                f.write(data)
        except Exception:
            pass

        return FetchResult(True, "ok", ctype, data, url, False)

    except HTTPError as e:
        return FetchResult(False, f"HTTP {e.code}", "", None, url, False)
    except URLError as e:
        return FetchResult(False, f"URL error: {e.reason}", "", None, url, False)
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
    """
    Captures anchor href + visible text.
    Used to get artist names from letter pages.
    """
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
    out = []
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
# Gallerix painting page extraction (optional / lazy)
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
# DB ops (artists + paintings)
# ----------------------------
def upsert_artist(url: str, name: str):
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
                name=COALESCE(excluded.name, artists.name),
                last_seen=excluded.last_seen
            """,
            (url, name or None, now_ts()),
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


def upsert_minimal_painting(url: str, artist_name: str = "", artist_url: str = ""):
    conn = db()
    try:
        migrate_schema(conn)
        conn.execute(
            """
            INSERT INTO paintings (url, artist, artist_url, last_seen)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(url) DO UPDATE SET
                artist=COALESCE(excluded.artist, paintings.artist),
                artist_url=COALESCE(excluded.artist_url, paintings.artist_url),
                last_seen=excluded.last_seen
            """,
            (url, artist_name or None, artist_url or None, now_ts()),
        )
        conn.commit()
    finally:
        conn.close()


def upsert_painting_full(url: str, img_url: str, title: str, artist: str, artist_url: str, meta: str, tags: List[str]):
    conn = db()
    try:
        migrate_schema(conn)
        conn.execute(
            """
            INSERT INTO paintings (url, img_url, title, artist, artist_url, meta, tags, elo, mu, sigma, last_seen, last_vote)
            VALUES (?, ?, ?, ?, ?, ?, ?, 1500.0, ?, ?, ?, 0)
            ON CONFLICT(url) DO UPDATE SET
                img_url=COALESCE(excluded.img_url, paintings.img_url),
                title=COALESCE(excluded.title, paintings.title),
                artist=COALESCE(excluded.artist, paintings.artist),
                artist_url=COALESCE(excluded.artist_url, paintings.artist_url),
                meta=COALESCE(excluded.meta, paintings.meta),
                tags=COALESCE(excluded.tags, paintings.tags),
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
            (limit,),
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
# Storeroom crawling
# ----------------------------
def extract_artist_urls_from_letter_page(html: str, letter_page_url: str) -> List[Tuple[str, str]]:
    """
    Returns list of (artist_url, artist_name) from letter pages.
    This is the critical fix to use artist *names*.
    """
    pairs = extract_link_texts(html, letter_page_url)
    out: List[Tuple[str, str]] = []
    seen = set()
    for u, txt in pairs:
        if ARTIST_URL_RE.match(u):
            au = u.rstrip("/") + "/"
            name = (txt or "").strip()
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
        if PAINTING_URL_RE.match(u.rstrip("/") + "/"):
            pics.append(u.rstrip("/") + "/")
    seen, out = set(), []
    for p in pics:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


# ----------------------------
# Ingest painting page (optional / lazy enrichment)
# ----------------------------
def ingest_painting_page_full(painting_url: str, user_agent: str, timeout: float, delay: float) -> bool:
    html, _ = fetch_text(painting_url, user_agent=user_agent, timeout=timeout, max_bytes=2_500_000, referer=painting_url)
    if not html:
        return False

    title, artist_name_from_page, meta, text = extract_title_artist_meta_and_text(html)
    img_url = extract_primary_image_url(html, painting_url) or ""
    tags = infer_style_tags(text)

    artist_url = parse_artist_url_from_painting(painting_url)
    artist_name = (artist_name_from_page or "").strip()

    # If we already know the artist name from letter crawl, prefer that (more consistent)
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
# Display helpers (no server-side image downloading)
# ----------------------------
def render_painting_display(p: Dict, display_mode: str, height: int = 620):
    url = p["url"]
    img_url = p.get("img_url") or ""

    if display_mode == "iframe":
        components.iframe(url, height=height, scrolling=True)
        return

    if not img_url:
        components.iframe(url, height=height, scrolling=True)
        return

    html = f"""
    <div style="width:100%; height:{height}px; display:flex; align-items:center; justify-content:center; background:#111; border-radius:12px; overflow:hidden;">
      <img src="{img_url}" style="max-width:100%; max-height:100%; object-fit:contain;" />
    </div>
    """
    components.html(html, height=height, scrolling=False)


# ----------------------------
# Queue-based “see almost all paintings”
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


# ----------------------------
# “Load ALL artists + ALL paintings” (resumable batch)
# ----------------------------
def load_all_artists_batch(user_agent: str, timeout: float, delay: float, letters_per_run: int) -> Tuple[int, int, str]:
    """
    Loads artists by crawling letter pages A..Z. Resumable.
    Stores:
      - ingest_state: artists_json (list of artist URLs)
      - artists table: url -> name
    """
    pos = int(ingest_state_get("artist_letter_pos", "0") or "0")
    start = pos
    end = min(len(LATIN_LETTERS), start + letters_per_run)

    artists_json = ingest_state_get("artists_json", "[]")
    try:
        artists_set = set(json.loads(artists_json))
    except Exception:
        artists_set = set()

    added_urls = 0
    added_names = 0
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

        pairs = extract_artist_urls_from_letter_page(html, url)
        dbg.append(f"{L}:{len(pairs)}")

        for au, name in pairs:
            if au not in artists_set:
                artists_set.add(au)
                added_urls += 1
            if name:
                upsert_artist(au, name)
                added_names += 1

        if delay > 0:
            time.sleep(delay)

    ingest_state_set("artists_json", json.dumps(sorted(list(artists_set))))
    ingest_state_set("artist_letter_pos", str(end if end < len(LATIN_LETTERS) else len(LATIN_LETTERS)))

    status = " | ".join(dbg) if dbg else "no-op"
    return added_urls, added_names, status


def load_all_paintings_batch(
    user_agent: str,
    timeout: float,
    delay: float,
    artists_per_run: int,
    paintings_cap_per_artist: int,
) -> Tuple[int, int, int, str]:
    """
    Uses stored artist list to crawl artist pages and store ALL painting URLs (minimal inserts),
    attaching artist *name* from artists table.
    """
    artists_json = ingest_state_get("artists_json", "[]")
    try:
        artists = json.loads(artists_json)
    except Exception:
        artists = []

    if not artists:
        return 0, 0, 0, "No artists loaded yet. Run 'Load artists' first."

    idx = int(ingest_state_get("artist_idx", "0") or "0")
    start = idx
    end = min(len(artists), start + artists_per_run)

    paintings_added = 0
    artists_done = 0
    artists_missing_name = 0
    dbg = []

    for i in range(start, end):
        artist_url = artists[i]
        html, fr = fetch_text(artist_url, user_agent=user_agent, timeout=timeout, max_bytes=4_000_000, referer=STOREROOM_ROOT)
        if not html:
            dbg.append(f"artist:fail({fr.status})")
            if delay > 0:
                time.sleep(delay)
            continue

        # best-effort: ensure we have an artist name
        name = get_artist_name(artist_url)
        if not name:
            # sometimes artist page has a header; try to extract something human-readable
            m = re.search(r"(?is)<h1[^>]*>(.*?)</h1>", html)
            if m:
                h1 = re.sub(r"(?is)<.*?>", " ", m.group(1))
                name = re.sub(r"\s+", " ", h1).strip()
                if name:
                    upsert_artist(artist_url, name)
        if not name:
            artists_missing_name += 1

        pics = extract_painting_urls_from_artist_page(html, artist_url)
        if paintings_cap_per_artist > 0:
            pics = pics[:paintings_cap_per_artist]

        for pu in pics:
            upsert_minimal_painting(pu, artist_name=name, artist_url=artist_url)
            paintings_added += 1

        artists_done += 1
        dbg.append(f"{name[:18]+'…' if name else 'Unknown'}:{len(pics)}")

        if delay > 0:
            time.sleep(delay)

    ingest_state_set("artist_idx", str(end if end < len(artists) else len(artists)))
    status = " | ".join(dbg[:8]) + (" | …" if len(dbg) > 8 else "")
    return paintings_added, artists_done, artists_missing_name, status


# ----------------------------
# Rating system (TrueSkill-lite)
# ----------------------------
def normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def trueskill_lite_update(
    mu_a: float, sig_a: float,
    mu_b: float, sig_b: float,
    a_wins: bool,
    beta: float = TS_BETA,
    tau: float = TS_TAU,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
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
    return mu - 3.0 * sigma


def value_score_0_100(v: float) -> float:
    return clamp((v / 40.0) * 100.0, 0.0, 100.0)


def conservative_value(p: Dict) -> float:
    return float(p["mu"]) - 3.0 * float(p["sigma"])


def painting_rank_key(p: Dict) -> Tuple[float, int, int]:
    return (conservative_value(p), int(p.get("games", 0)), int(p.get("last_vote", 0)))


def apply_vote(winner_url: str, loser_url: str, mode: str = "vote"):
    w = get_painting(winner_url)
    l = get_painting(loser_url)
    if not w or not l:
        return

    k = (k_factor(w["games"]) + k_factor(l["games"])) / 2.0
    new_rw, new_rl = elo_update(w["elo"], l["elo"], 1.0, k)

    (mu_w, sig_w), (mu_l, sig_l) = trueskill_lite_update(w["mu"], w["sigma"], l["mu"], l["sigma"], True)

    conn = db()
    try:
        migrate_schema(conn)
        conn.execute(
            """
            UPDATE paintings
            SET elo=?, mu=?, sigma=?, games=games+1, wins=wins+1, last_vote=?
            WHERE url=?
            """,
            (new_rw, mu_w, sig_w, now_ts(), winner_url),
        )
        conn.execute(
            """
            UPDATE paintings
            SET elo=?, mu=?, sigma=?, games=games+1, losses=losses+1, last_vote=?
            WHERE url=?
            """,
            (new_rl, mu_l, sig_l, now_ts(), loser_url),
        )
        conn.commit()
    finally:
        conn.close()


# ----------------------------
# Leaderboards (NAME-based)
# ----------------------------
def paintings_leaderboard_live(pool: List[Dict], n: int, min_games: int = 0) -> List[Dict]:
    filtered = [p for p in pool if int(p.get("games", 0)) >= int(min_games)]
    filtered.sort(key=painting_rank_key, reverse=True)

    rows = []
    for i, p in enumerate(filtered[:n], 1):
        v = conservative_value(p)
        rows.append({
            "rank": i,
            "score_0_100": round(value_score_0_100(v), 1),
            "value": round(v, 4),
            "mu": round(float(p["mu"]), 4),
            "sigma": round(float(p["sigma"]), 4),
            "games": int(p["games"]),
            "wins": int(p["wins"]),
            "losses": int(p["losses"]),
            "artist": (p.get("artist") or ""),
            "title": (p.get("title") or ""),
            "url": p["url"],
        })
    return rows


def artists_leaderboard_live(
    pool: List[Dict],
    n: int,
    topk: int = 5,
    min_artist_games: int = 0,
    min_painting_games: int = 0,
    include_unknown: bool = False,
) -> List[Dict]:
    buckets: Dict[str, Dict] = {}

    for p in pool:
        if int(p.get("games", 0)) < int(min_painting_games):
            continue
        name = (p.get("artist") or "").strip()

        if not name and not include_unknown:
            continue
        if not name:
            name = "Unknown artist"

        key = name.lower()
        if key not in buckets:
            buckets[key] = {"display": name, "paintings": [], "games_total": 0}
        buckets[key]["paintings"].append(p)
        buckets[key]["games_total"] += int(p.get("games", 0))

    rows = []
    for _, b in buckets.items():
        if int(b["games_total"]) < int(min_artist_games):
            continue

        paintings_sorted = sorted(b["paintings"], key=painting_rank_key, reverse=True)
        top = paintings_sorted[:max(1, int(topk))]
        vals = [conservative_value(p) for p in top]
        artist_value = sum(vals) / max(1, len(vals))
        best = top[0] if top else None

        rows.append({
            "artist": b["display"],
            "value": artist_value,
            "score_0_100": value_score_0_100(artist_value),
            "games": int(b["games_total"]),
            "paintings": len(b["paintings"]),
            "best_title": (best.get("title") if best else ""),
            "best_url": (best.get("url") if best else ""),
        })

    rows.sort(key=lambda r: (float(r["value"]), int(r["games"]), int(r["paintings"])), reverse=True)

    out = []
    for i, r in enumerate(rows[:n], 1):
        out.append({
            "rank": i,
            "artist": r["artist"],
            "score_0_100": round(float(r["score_0_100"]), 1),
            "value": round(float(r["value"]), 4),
            "games": int(r["games"]),
            "paintings": int(r["paintings"]),
            "best_title": r["best_title"] or "",
            "best_url": r["best_url"] or "",
        })
    return out


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title=APP_NAME, page_icon="🖼️", layout="wide")
init_db()

st.title("🖼️ Art Mash (Artist Name-based)")
st.caption(
    "Artists are now keyed and displayed by *name* (not ID). Bulk loader captures names from storeroom letter pages."
)

with st.sidebar:
    st.header("Networking")
    user_agent = st.text_input("User-Agent", value=DEFAULT_UA)
    timeout = st.slider("Timeout (sec)", 3, 30, 10, 1)
    crawl_delay = st.slider("Crawl delay (sec)", 0.0, 2.0, 0.2, 0.05)

    st.divider()
    st.header("Initial randomization / queue")
    queue_limit = st.slider("Queue size (from DB)", 200, 20000, 4000, 200)
    queue_seed = st.number_input("Queue seed", 0, 10_000_000, 1337, 1)
    rebuild_queue = st.button("🔀 Rebuild session queue", use_container_width=True)

    st.divider()
    st.header("Display")
    display_mode = st.selectbox("Display mode", ["iframe (reliable)", "img tag (fast)"], index=0)
    disp_mode_key = "iframe" if display_mode.startswith("iframe") else "img"
    iframe_height = st.slider("Display height", 360, 900, 620, 10)
    show_meta = st.checkbox("Show metadata", value=True)

    st.divider()
    st.header("Bulk loader (resumable)")
    letters_per_run = st.slider("Letters per run (artists)", 1, 26, 6, 1)
    artists_per_run = st.slider("Artists per run (paintings)", 1, 500, 50, 1)
    cap_per_artist = st.slider("Cap paintings per artist (0 = no cap)", 0, 5000, 0, 50)

    st.caption(f"DB path: {DB_PATH}")

tabs = st.tabs(["Vote", "Bulk Load", "Leaderboards", "Admin"])

# Session queue init
if rebuild_queue or "queue_urls" not in st.session_state:
    st.session_state["queue_urls"] = build_session_queue(limit=int(queue_limit), seed=int(queue_seed))
    st.session_state["queue_idx"] = 0

if "seen_urls" not in st.session_state:
    st.session_state["seen_urls"] = set()

# ----------------------------
# Vote tab
# ----------------------------
with tabs[0]:
    pool_count = len(get_pool(limit=4000))
    st.write(f"DB pool (recent sample): **{pool_count}**  |  Session queue size: **{len(st.session_state['queue_urls'])}**")

    pair = pick_pair_from_queue()
    if not pair:
        st.warning("Not enough paintings in the queue. Load more via Bulk Load, or increase queue size.")
    else:
        left, right = pair
        st.session_state["seen_urls"].add(left["url"])
        st.session_state["seen_urls"].add(right["url"])

        # Lazy enrich when needed:
        # - For img-tag mode: need img_url
        # - For nicer metadata: title/artist
        if disp_mode_key == "img":
            if (not left["img_url"]) or (not left["title"]) or (not left["artist"]):
                ingest_painting_page_full(left["url"], user_agent, float(timeout), 0.0)
                left = get_painting(left["url"]) or left
            if (not right["img_url"]) or (not right["title"]) or (not right["artist"]):
                ingest_painting_page_full(right["url"], user_agent, float(timeout), 0.0)
                right = get_painting(right["url"]) or right

        colL, colR = st.columns(2, gap="large")

        def card(col, p: Dict, label: str):
            with col:
                st.subheader(label)
                render_painting_display(p, disp_mode_key, height=int(iframe_height))

                if show_meta:
                    v = mu_sigma_to_value(p["mu"], p["sigma"])
                    st.markdown(
                        f"**Score:** `{value_score_0_100(v):.1f}/100`  |  **μ/σ:** `{p['mu']:.2f}/{p['sigma']:.2f}`  |  **Games:** `{p['games']}`"
                    )
                    if p.get("title"):
                        st.markdown(f"**Title:** {p['title']}")
                    if p.get("artist"):
                        st.markdown(f"**Artist:** {p['artist']}")
                    if p.get("meta"):
                        st.markdown(f"**Info:** {p['meta']}")
                    st.markdown(f"[Open]({p['url']})")

        card(colL, left, "A")
        card(colR, right, "B")

        b1, b2, b3 = st.columns([1, 1, 1])
        vote_a = b1.button("✅ Vote A (left)", use_container_width=True)
        vote_b = b2.button("✅ Vote B (right)", use_container_width=True)
        skip = b3.button("↩ Skip", use_container_width=True)

        if vote_a:
            record_vote(left["url"], right["url"], left["url"], mode="vote")
            apply_vote(left["url"], right["url"], mode="vote")
            st.rerun()
        if vote_b:
            record_vote(left["url"], right["url"], right["url"], mode="vote")
            apply_vote(right["url"], left["url"], mode="vote")
            st.rerun()
        if skip:
            st.rerun()

# ----------------------------
# Bulk Load tab
# ----------------------------
with tabs[1]:
    st.subheader("Bulk Loader (Resumable)")
    st.write(
        "Step 1 loads ALL artists from storeroom letter pages (and captures names). "
        "Step 2 loads ALL painting URLs from each artist page and attaches artist names."
    )

    artists_json = ingest_state_get("artists_json", "[]")
    try:
        artists_list = json.loads(artists_json)
    except Exception:
        artists_list = []
    letter_pos = int(ingest_state_get("artist_letter_pos", "0") or "0")
    artist_idx = int(ingest_state_get("artist_idx", "0") or "0")

    st.info(
        f"Artists loaded: **{len(artists_list)}** | Letter progress: **{letter_pos}/26** | Artist progress: **{artist_idx}/{len(artists_list) or 0}**"
    )

    c1, c2 = st.columns(2, gap="large")

    with c1:
        if st.button("1) Load artists (batch)", type="primary", use_container_width=True):
            added_urls, added_names, status = load_all_artists_batch(
                user_agent=user_agent,
                timeout=float(timeout),
                delay=float(crawl_delay),
                letters_per_run=int(letters_per_run),
            )
            st.success(f"Added {added_urls} artist URLs and stored {added_names} names. {status}")
            st.rerun()

        if st.button("Reset artist letter progress", use_container_width=True):
            ingest_state_set("artist_letter_pos", "0")
            st.warning("Reset letter progress to 0. Artist URL list remains.")
            st.rerun()

    with c2:
        if st.button("2) Load paintings (batch)", type="primary", use_container_width=True):
            paintings_added, artists_done, missing_names, status = load_all_paintings_batch(
                user_agent=user_agent,
                timeout=float(timeout),
                delay=float(crawl_delay),
                artists_per_run=int(artists_per_run),
                paintings_cap_per_artist=int(cap_per_artist),
            )
            st.success(f"Stored {paintings_added} painting URLs from {artists_done} artists. Missing names: {missing_names}. {status}")
            st.session_state["queue_urls"] = build_session_queue(limit=int(queue_limit), seed=int(queue_seed))
            st.session_state["queue_idx"] = 0
            st.rerun()

        if st.button("Reset artist index progress", use_container_width=True):
            ingest_state_set("artist_idx", "0")
            st.warning("Reset artist crawl index to 0.")
            st.rerun()

    st.divider()
    st.subheader("Optional: enrich a random sample (titles/img URLs)")
    st.write("Fetches ONLY HTML for a small sample of painting pages to improve metadata. Still no image downloads server-side.")
    enrich_n = st.slider("Enrich N paintings now", 0, 200, 20, 5)
    if st.button("Enrich sample", use_container_width=True) and enrich_n > 0:
        pool = get_pool(limit=max(500, enrich_n * 10))
        rng = random.Random(int(queue_seed) ^ (now_ts() // 10))
        rng.shuffle(pool)
        sample = pool[:enrich_n]
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
    st.subheader("Leaderboards (Live, artist NAME-based)")

    cA, cB, cC = st.columns([1, 1, 2])
    with cB:
        if st.button("🔄 Refresh now", use_container_width=True):
            st.rerun()

    pool_limit = st.slider("Pool sample size for leaderboards", 1000, 50000, 20000, 1000)
    pool = get_pool(limit=int(pool_limit))
    st.caption(f"Using {len(pool)} paintings from DB for leaderboard computation.")

    st.markdown("### 🏆 Top Paintings")
    pcol1, pcol2 = st.columns(2)
    with pcol1:
        top_p = st.slider("Top N paintings", 25, 1000, 200, 25)
    with pcol2:
        min_pg = st.slider("Min games per painting", 0, 50, 1, 1)

    prow = paintings_leaderboard_live(pool, n=int(top_p), min_games=int(min_pg))
    st.dataframe(prow, use_container_width=True, height=520)
    st.download_button(
        "Download paintings leaderboard JSON",
        data=json.dumps(prow, indent=2).encode("utf-8"),
        file_name="artmash_paintings_leaderboard.json",
        mime="application/json",
        use_container_width=True,
    )

    st.divider()
    st.markdown("### 🎨 Top Artists")
    acol1, acol2, acol3 = st.columns(3)
    with acol1:
        top_a = st.slider("Top N artists", 25, 1000, 200, 25)
    with acol2:
        topk = st.slider("Aggregate top-k paintings per artist", 1, 20, 5, 1)
    with acol3:
        min_ag = st.slider("Min total games per artist", 0, 500, 5, 1)

    min_paint_games = st.slider("Min games per painting (for artist agg)", 0, 50, 1, 1)
    include_unknown = st.checkbox("Include Unknown artist bucket", value=False)

    arow = artists_leaderboard_live(
        pool,
        n=int(top_a),
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
    )

# ----------------------------
# Admin tab
# ----------------------------
with tabs[3]:
    st.subheader("Admin")
    st.caption(f"DB path: {DB_PATH}")

    if st.button("Rebuild queue now", use_container_width=True):
        st.session_state["queue_urls"] = build_session_queue(limit=int(queue_limit), seed=int(queue_seed))
        st.session_state["queue_idx"] = 0
        st.success("Queue rebuilt.")
        st.rerun()

    if st.button("Clear session seen set", use_container_width=True):
        st.session_state["seen_urls"] = set()
        st.success("Cleared.")
        st.rerun()

    if st.button("Clear HTML cache (local)", use_container_width=True):
        try:
            import shutil
            shutil.rmtree(CACHE_DIR, ignore_errors=True)
            st.success("Cache cleared.")
        except Exception as e:
            st.error(f"Failed: {e}")

st.caption(
    "Tip: Use iframe display mode if images aren’t showing. It avoids hotlink/CDN restrictions entirely. "
    "Bulk loading is resumable; run in small batches to avoid timeouts."
)
