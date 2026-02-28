# app.py
# Art Mash — historical art FaceMash using Gallerix (NO server-side image downloading)
#
# ✅ CHANGE REQUEST IMPLEMENTED:
# - Crawl source switched to https://gallerix.org/storeroom/
# - Random letter page: https://gallerix.org/storeroom/letter/H/
# - Random artist page: https://gallerix.org/storeroom/<artist_id>/
# - Random paintings pulled from that artist page (painting pages look like: /storeroom/<id>/N/<id>/)
# - “Surprise me” ingest: add a few new paintings so voting shows new works almost every time.
#
# ✅ FIXES:
# - Robust SQLite schema migrations (handles old DBs on Streamlit Cloud)
# - DB path auto-resolves to writable location (/tmp) when needed
# - Safe retry on queries after migration
#
# DISPLAY (no downloading art):
# - iframe embedding painting page (most reliable)
# - or browser-side <img src="..."> (fast; may fail if CDN blocks hotlinking)
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

DEFAULT_UA = "ArtMash/2.2 (Streamlit; respectful crawler)"
CACHE_DIR = ".cache_artmash"

# TrueSkill-lite defaults (similar spirit to TrueSkill)
TS_MU0 = 25.0
TS_SIGMA0 = 8.333  # ~ mu/3
TS_BETA = 4.1667   # skill variance
TS_TAU = 0.08      # small dynamics (drift)

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

LATIN_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


# ----------------------------
# DB path + migrations (CRITICAL FIX)
# ----------------------------
def resolve_db_path(default_name: str = "artmash.sqlite3") -> str:
    """
    Streamlit Cloud sometimes has repo paths that behave oddly with write permissions.
    Prefer a writable location. /tmp is always writable in Streamlit Cloud containers.
    Override via env ARTMASH_DB_PATH if desired.
    """
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
            meta TEXT,
            elo REAL DEFAULT 1500.0,
            games INTEGER DEFAULT 0,
            wins INTEGER DEFAULT 0,
            losses INTEGER DEFAULT 0,
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

    ensure_column(conn, "paintings", "mu", "REAL DEFAULT 25.0")
    ensure_column(conn, "paintings", "sigma", "REAL DEFAULT 8.333")
    ensure_column(conn, "paintings", "tags", "TEXT")
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


def softmax(xs: List[float]) -> List[float]:
    m = max(xs)
    ex = [math.exp(x - m) for x in xs]
    s = sum(ex) + 1e-12
    return [e / s for e in ex]


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
    try:
        return fr.data.decode("utf-8", errors="replace"), fr
    except Exception:
        return fr.data.decode(errors="replace"), fr


# ----------------------------
# Simple HTML link parser
# ----------------------------
class HrefParser(HTMLParser):
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


def extract_links(html: str, base_url: str) -> Tuple[List[str], List[str]]:
    p = HrefParser()
    p.feed(html)

    def dedupe(xs):
        seen = set()
        out = []
        for x in xs:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    hrefs = []
    imgs = []
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


# ----------------------------
# Gallerix extraction
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
        if "cdn.gallerix" in u and u.lower().endswith((".webp", ".jpg", ".jpeg", ".png")):
            return u
    for u in imgs:
        if u.lower().endswith((".webp", ".jpg", ".jpeg", ".png")):
            return u

    m = re.search(r"(https?://cdn\.gallerix\.[^\"'\s>]+?\.(?:webp|jpg|jpeg|png))", html, re.I)
    if m:
        return m.group(1)
    return None


def infer_style_tags(text: str) -> List[str]:
    t = (text or "").lower()
    tags = []
    for label, kws in STYLE_KEYWORDS.items():
        if any(k in t for k in kws):
            tags.append(label)
    return tags[:5]


# ----------------------------
# Storeroom crawling (NEW)
# ----------------------------
ARTIST_URL_RE = re.compile(r"^https://gallerix\.org/storeroom/(\d+)/?$")
PAINTING_URL_RE = re.compile(r"^https://gallerix\.org/storeroom/\d+/N/\d+/?$")


def random_letter_url(rng: random.Random) -> str:
    L = rng.choice(LATIN_LETTERS)
    return LETTER_URL.format(L=L)


def extract_artist_urls_from_letter_page(html: str, letter_page_url: str) -> List[str]:
    links, _ = extract_links(html, letter_page_url)
    artists = []
    for u in links:
        if ARTIST_URL_RE.match(u):
            artists.append(u.rstrip("/") + "/")
    # Deduplicate
    seen = set()
    out = []
    for a in artists:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


def extract_painting_urls_from_artist_page(html: str, artist_url: str) -> List[str]:
    links, _ = extract_links(html, artist_url)
    pics = []
    for u in links:
        if PAINTING_URL_RE.match(u):
            pics.append(u.rstrip("/") + "/")
    # Deduplicate
    seen = set()
    out = []
    for p in pics:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def choose_random_artist_and_paintings(
    *,
    user_agent: str,
    timeout: float,
    delay: float,
    seed: int,
    max_artist_trials: int = 6,
    paintings_to_take: int = 10,
) -> Tuple[Optional[str], List[str], str]:
    """
    Picks: random letter -> random artist -> sample paintings from that artist.
    Returns: (artist_url, painting_urls, debug_status)
    """
    rng = random.Random(seed ^ (now_ts() // 3))  # changes often => fresh
    status_bits = []

    for attempt in range(max_artist_trials):
        letter_url = random_letter_url(rng)
        status_bits.append(f"letter={letter_url.split('/')[-2]}")

        l_html, fr = fetch_text(letter_url, user_agent=user_agent, timeout=timeout, max_bytes=3_000_000, referer=STOREROOM_ROOT)
        if not l_html:
            status_bits.append(f"letter_fetch_fail({fr.status})")
            if delay > 0:
                time.sleep(delay)
            continue

        artists = extract_artist_urls_from_letter_page(l_html, letter_url)
        if not artists:
            status_bits.append("no_artists")
            if delay > 0:
                time.sleep(delay)
            continue

        artist_url = rng.choice(artists)
        status_bits.append(f"artist={artist_url.rstrip('/').split('/')[-1]}")

        a_html, fr2 = fetch_text(artist_url, user_agent=user_agent, timeout=timeout, max_bytes=4_000_000, referer=letter_url)
        if not a_html:
            status_bits.append(f"artist_fetch_fail({fr2.status})")
            if delay > 0:
                time.sleep(delay)
            continue

        pics = extract_painting_urls_from_artist_page(a_html, artist_url)
        if not pics:
            status_bits.append("no_paintings_found")
            if delay > 0:
                time.sleep(delay)
            continue

        rng.shuffle(pics)
        chosen = pics[: max(1, paintings_to_take)]
        return artist_url, chosen, " | ".join(status_bits)

    return None, [], " | ".join(status_bits) if status_bits else "no_attempts"


# ----------------------------
# SQLite storage
# ----------------------------
def upsert_painting(url: str, img_url: str, title: str, artist: str, meta: str, tags: List[str]):
    conn = db()
    try:
        migrate_schema(conn)
        conn.execute(
            """
            INSERT INTO paintings (url, img_url, title, artist, meta, tags, elo, mu, sigma, last_seen, last_vote)
            VALUES (?, ?, ?, ?, ?, ?, 1500.0, ?, ?, ?, 0)
            ON CONFLICT(url) DO UPDATE SET
                img_url=excluded.img_url,
                title=COALESCE(excluded.title, paintings.title),
                artist=COALESCE(excluded.artist, paintings.artist),
                meta=COALESCE(excluded.meta, paintings.meta),
                tags=COALESCE(excluded.tags, paintings.tags),
                last_seen=excluded.last_seen
            """,
            (url, img_url, title, artist, meta, json.dumps(tags), TS_MU0, TS_SIGMA0, now_ts()),
        )
        conn.commit()
    finally:
        conn.close()


def get_painting(url: str) -> Optional[Dict]:
    conn = db()
    try:
        migrate_schema(conn)
        cur = conn.execute(
            """
            SELECT url, img_url, title, artist, meta, tags, elo, mu, sigma, games, wins, losses, last_seen, last_vote
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
        title=r[2] or "",
        artist=r[3] or "",
        meta=r[4] or "",
        tags=json.loads(r[5]) if r[5] else [],
        elo=float(r[6] or 1500.0),
        mu=float(r[7] or TS_MU0),
        sigma=float(r[8] or TS_SIGMA0),
        games=int(r[9] or 0),
        wins=int(r[10] or 0),
        losses=int(r[11] or 0),
        last_seen=int(r[12] or 0),
        last_vote=int(r[13] or 0),
    )


def get_pool(limit: int = 2000) -> List[Dict]:
    def _run(conn):
        cur = conn.execute(
            """
            SELECT url, img_url, title, artist, meta, tags, elo, mu, sigma, games, wins, losses, last_seen, last_vote
            FROM paintings
            ORDER BY last_seen DESC
            LIMIT ?
            """,
            (limit,),
        )
        return cur.fetchall()

    conn = db()
    try:
        try:
            rows = _run(conn)
        except sqlite3.OperationalError:
            migrate_schema(conn)
            rows = _run(conn)
    finally:
        conn.close()

    out = []
    for r in rows:
        out.append(
            dict(
                url=r[0],
                img_url=r[1] or "",
                title=r[2] or "",
                artist=r[3] or "",
                meta=r[4] or "",
                tags=json.loads(r[5]) if r[5] else [],
                elo=float(r[6] or 1500.0),
                mu=float(r[7] or TS_MU0),
                sigma=float(r[8] or TS_SIGMA0),
                games=int(r[9] or 0),
                wins=int(r[10] or 0),
                losses=int(r[11] or 0),
                last_seen=int(r[12] or 0),
                last_vote=int(r[13] or 0),
            )
        )
    return out


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
# Rating systems
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

    sig_a_new = math.sqrt(max(sig_a2, 1e-6))
    sig_b_new = math.sqrt(max(sig_b2, 1e-6))
    return (mu_a_new, sig_a_new), (mu_b_new, sig_b_new)


def elo_expected(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))


def elo_update(r_a: float, r_b: float, score_a: float, k: float) -> Tuple[float, float]:
    ea = elo_expected(r_a, r_b)
    eb = 1.0 - ea
    new_a = r_a + k * (score_a - ea)
    new_b = r_b + k * ((1.0 - score_a) - eb)
    return new_a, new_b


def k_factor(games: int) -> float:
    if games < 10:
        return 48.0
    if games < 50:
        return 28.0
    return 16.0


def mu_sigma_to_value(mu: float, sigma: float) -> float:
    return mu - 3.0 * sigma


def value_score_0_100(v: float) -> float:
    return clamp((v / 40.0) * 100.0, 0.0, 100.0)


def apply_vote(winner_url: str, loser_url: str, mode: str = "vote"):
    w = get_painting(winner_url)
    l = get_painting(loser_url)
    if not w or not l:
        return

    rw, rl = w["elo"], l["elo"]
    k = (k_factor(w["games"]) + k_factor(l["games"])) / 2.0
    new_rw, new_rl = elo_update(rw, rl, score_a=1.0, k=k)

    (mu_w, sig_w), (mu_l, sig_l) = trueskill_lite_update(
        w["mu"], w["sigma"], l["mu"], l["sigma"], a_wins=True
    )

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
# Ingest painting page (HTML only)
# ----------------------------
def ingest_painting_page(
    painting_url: str,
    user_agent: str,
    timeout: float,
    delay: float,
) -> bool:
    html, _ = fetch_text(painting_url, user_agent=user_agent, timeout=timeout, max_bytes=2_500_000, referer=painting_url)
    if not html:
        return False

    title, artist, meta, text = extract_title_artist_meta_and_text(html)
    img_url = extract_primary_image_url(html, painting_url) or ""
    tags = infer_style_tags(text)
    upsert_painting(painting_url, img_url, title, artist, meta, tags)

    if delay > 0:
        time.sleep(delay)
    return True


# ----------------------------
# Display helpers (no server-side downloads)
# ----------------------------
def render_painting_display(p: Dict, display_mode: str, height: int = 620):
    url = p["url"]
    img_url = p.get("img_url") or ""

    if display_mode == "iframe":
        components.iframe(url, height=height, scrolling=True)
        return

    if not img_url:
        st.warning("No image URL extracted; using iframe instead.")
        components.iframe(url, height=height, scrolling=True)
        return

    html = f"""
    <div style="width:100%; height:{height}px; display:flex; align-items:center; justify-content:center; background:#111; border-radius:12px; overflow:hidden;">
      <img src="{img_url}" style="max-width:100%; max-height:100%; object-fit:contain;" />
    </div>
    """
    components.html(html, height=height, scrolling=False)


# ----------------------------
# Matchmaking + feeds
# ----------------------------
def choose_pair_uncertainty(pool: List[Dict], seed: int) -> Optional[Tuple[Dict, Dict]]:
    if len(pool) < 2:
        return None

    weights = []
    for p in pool:
        w = (p["sigma"] ** 1.6)
        w *= 1.0 + 0.4 / (p["games"] + 1.0)
        weights.append(w)

    probs = softmax([math.log(w + 1e-9) for w in weights])

    bucket = int(time.time() // 5)
    h = hashlib.sha1(f"{seed}-{bucket}".encode("utf-8")).hexdigest()
    r = [int(h[i:i+8], 16) / 0xFFFFFFFF for i in range(0, 32, 8)]

    def pick(x: float) -> Dict:
        acc = 0.0
        for p, pr in zip(pool, probs):
            acc += pr
            if acc >= x:
                return p
        return pool[-1]

    a = pick(r[0])
    b = pick(r[1])
    tries = 0
    while b["url"] == a["url"] and tries < 10:
        b = pick(r[2])
        tries += 1
    if a["url"] == b["url"]:
        return pool[0], pool[1]

    target = a["mu"]
    candidates = sorted(pool, key=lambda p: abs(p["mu"] - target))
    for cand in candidates[:30]:
        if cand["url"] != a["url"]:
            b = cand
            break
    return a, b


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title=APP_NAME, page_icon="🖼️", layout="wide")
init_db()

st.title("🖼️ Art Mash (Storeroom Randomized)")
st.caption(
    "Now crawls from the Storeroom index (random letter → random artist → random painting pages). "
    "Images are displayed without server-side downloading."
)

with st.sidebar:
    st.header("Source + ingest")
    user_agent = st.text_input("User-Agent", value=DEFAULT_UA)
    timeout = st.slider("Timeout (sec)", 3, 30, 10, 1)
    crawl_delay = st.slider("Crawl delay (sec)", 0.0, 2.0, 0.2, 0.05)

    st.divider()
    st.header("Random ingestion (new)")
    ingest_paintings_per_click = st.slider("Paintings to ingest per click", 1, 40, 10, 1)
    artist_trials = st.slider("Artist selection retries", 1, 10, 5, 1)
    surprise_seed = st.number_input("Random seed", 0, 10_000_000, 1337, 1)

    st.divider()
    st.header("Display")
    display_mode = st.selectbox("Display mode", ["iframe (reliable)", "img tag (fast)"], index=0)
    disp_mode_key = "iframe" if display_mode.startswith("iframe") else "img"
    iframe_height = st.slider("Display height", 360, 900, 620, 10)

    st.divider()
    st.header("Voting / matchmaking")
    match_seed = st.number_input("Match seed", 0, 10_000_000, 2026, 1)
    show_meta = st.checkbox("Show metadata", value=True)

    st.caption(f"DB path: {DB_PATH}")

tab_vote, tab_ingest = st.tabs(["Vote", "Ingest (Storeroom Random)"])


# ----------------------------
# Vote tab
# ----------------------------
with tab_vote:
    pool = get_pool(limit=6000)
    st.write(f"Pool size (ingested paintings): **{len(pool)}**")

    # Quick “make it fresh” button
    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("🎲 Surprise me (ingest new art)", type="primary", use_container_width=True):
            artist_url, painting_urls, dbg = choose_random_artist_and_paintings(
                user_agent=user_agent,
                timeout=float(timeout),
                delay=float(crawl_delay),
                seed=int(surprise_seed),
                max_artist_trials=int(artist_trials),
                paintings_to_take=int(ingest_paintings_per_click),
            )
            if not painting_urls:
                st.error(f"Could not find paintings. Debug: {dbg}")
            else:
                ok = 0
                for pu in painting_urls:
                    if ingest_painting_page(pu, user_agent=user_agent, timeout=float(timeout), delay=float(crawl_delay)):
                        ok += 1
                st.success(f"Ingested {ok}/{len(painting_urls)} paintings from random artist page.")
                st.caption(f"Debug: {dbg}")
                st.rerun()

    if len(pool) < 2:
        st.info("Pool is empty. Go to **Ingest** tab (or click Surprise me) to ingest paintings first.")
    else:
        pair = choose_pair_uncertainty(pool, seed=int(match_seed))
        if not pair:
            st.warning("Unable to choose a pair.")
        else:
            left, right = pair

            colL, colR = st.columns(2, gap="large")

            def card(col, p: Dict, label: str):
                with col:
                    st.subheader(label)
                    render_painting_display(p, disp_mode_key, height=int(iframe_height))

                    if show_meta:
                        st.markdown(f"**Title:** {p['title'] or '(unknown)'}")
                        st.markdown(f"**Artist:** {p['artist'] or '(unknown)'}")
                        if p["meta"]:
                            st.markdown(f"**Info:** {p['meta']}")
                        if p.get("tags"):
                            st.markdown(f"**Tags:** {', '.join(p['tags'])}")

                        v = mu_sigma_to_value(p["mu"], p["sigma"])
                        st.markdown(f"**μ / σ:** `{p['mu']:.2f}` / `{p['sigma']:.2f}`")
                        st.markdown(f"**Score:** `{value_score_0_100(v):.1f}/100`  |  **Games:** `{p['games']}`")
                        st.markdown(f"[Open on Gallerix]({p['url']})")

            card(colL, left, "A")
            card(colR, right, "B")

            b1, b2, b3 = st.columns([1, 1, 1])
            vote_a = b1.button("✅ Vote A (left)", use_container_width=True)
            vote_b = b2.button("✅ Vote B (right)", use_container_width=True)
            skip = b3.button("↩ Skip", use_container_width=True)

            if vote_a:
                record_vote(left["url"], right["url"], left["url"], mode="vote")
                apply_vote(left["url"], right["url"], mode="vote")
                st.success("Saved vote for A.")
                st.rerun()

            if vote_b:
                record_vote(left["url"], right["url"], right["url"], mode="vote")
                apply_vote(right["url"], left["url"], mode="vote")
                st.success("Saved vote for B.")
                st.rerun()

            if skip:
                st.rerun()


# ----------------------------
# Ingest tab (Storeroom Random)
# ----------------------------
with tab_ingest:
    st.subheader("Ingest from Storeroom (random letter → random artist → paintings)")
    st.write(
        "This mode pulls artists from the Storeroom index and samples paintings from a random artist page. "
        "It’s designed to keep the pool fresh so you see new paintings almost every time."
    )

    if st.button("🔄 Ingest now (random artist)", type="primary", use_container_width=True):
        artist_url, painting_urls, dbg = choose_random_artist_and_paintings(
            user_agent=user_agent,
            timeout=float(timeout),
            delay=float(crawl_delay),
            seed=int(surprise_seed),
            max_artist_trials=int(artist_trials),
            paintings_to_take=int(ingest_paintings_per_click),
        )

        if not painting_urls:
            st.error(f"Could not find paintings. Debug: {dbg}")
        else:
            prog = st.progress(0)
            ok = 0
            for i, pu in enumerate(painting_urls):
                if ingest_painting_page(pu, user_agent=user_agent, timeout=float(timeout), delay=float(crawl_delay)):
                    ok += 1
                prog.progress(int(100 * (i + 1) / max(1, len(painting_urls))))
            st.success(f"Ingested {ok}/{len(painting_urls)} paintings from: {artist_url or '(unknown)'}")
            st.caption(f"Debug: {dbg}")
            st.rerun()

    st.divider()
    st.subheader("Manual ingest (painting page URLs)")
    manual_urls = st.text_area("Paste painting page URLs (one per line)", height=140)
    if st.button("Ingest pasted URLs", use_container_width=True):
        urls = [u.strip() for u in manual_urls.splitlines() if u.strip()]
        if not urls:
            st.warning("No URLs provided.")
        else:
            ok = 0
            for u in urls[:200]:
                if ingest_painting_page(u, user_agent=user_agent, timeout=float(timeout), delay=float(crawl_delay)):
                    ok += 1
            st.success(f"Ingested {ok}/{min(len(urls),200)} pages.")
            st.rerun()

st.caption(
    "If images fail in 'img tag' mode, switch to 'iframe' mode — it avoids hotlink/CDN restrictions entirely."
)
