# app.py
# Art Mash — pairwise voting + Elo ranking for historical art (Gallerix)
#
# How it works:
# - Seed album pages from https://gallerix.org/a1/ (Famous artists / popular albums).
# - Crawl a small number of album pages to collect painting page URLs.
# - For each painting page, extract:
#     - title, artist, year/size line (best-effort)
#     - a primary image URL (usually cdn.gallerix.asia)
# - Show two paintings side-by-side; user votes.
# - Update Elo ratings and show leaderboard.
#
# Respectful crawling:
# - Disk cache of HTML + images
# - Crawl delay
# - Small default limits
#
# Dependencies: streamlit (stdlib otherwise)

import os
import re
import io
import time
import json
import math
import hashlib
import sqlite3
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from html.parser import HTMLParser

import streamlit as st


# ----------------------------
# Constants
# ----------------------------
APP_NAME = "Art Mash"
BASE = "https://gallerix.org"
SEED_PAGE = "https://gallerix.org/a1/"  # "Famous artists" / popular albums list

DEFAULT_UA = "ArtMash/1.0 (Streamlit; respectful crawler; +https://streamlit.io)"
CACHE_DIR = ".cache_artmash"
DB_PATH = "artmash.sqlite3"


# ----------------------------
# Utilities
# ----------------------------
def ensure_dirs():
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(os.path.join(CACHE_DIR, "html"), exist_ok=True)
    os.makedirs(os.path.join(CACHE_DIR, "img"), exist_ok=True)


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def now_ts() -> int:
    return int(time.time())


# ----------------------------
# Disk cache fetcher (HTML / images)
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
    # kind in {"html","img"}
    return os.path.join(CACHE_DIR, kind, key)


def fetch_bytes(
    url: str,
    *,
    user_agent: str,
    timeout: float,
    max_bytes: int,
    referer: Optional[str] = None,
    cache_kind: str = "html",
    cache_ttl_sec: int = 7 * 24 * 3600,  # 7 days
) -> FetchResult:
    """
    Fetch bytes with a small disk cache.
    """
    ensure_dirs()
    key = sha1(url)
    path = cache_path(cache_kind, key)

    # cache hit
    if os.path.exists(path):
        age = now_ts() - int(os.path.getmtime(path))
        if age <= cache_ttl_sec:
            try:
                data = open(path, "rb").read()
                return FetchResult(True, "cache", "", data, url, True)
            except Exception:
                pass  # fall through to network

    headers = {"User-Agent": user_agent, "Accept": "*/*"}
    if referer:
        headers["Referer"] = referer

    try:
        req = Request(url, headers=headers)
        with urlopen(req, timeout=timeout) as resp:
            ctype = (resp.headers.get("Content-Type") or "")
            data = resp.read(max_bytes)

        # write cache
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
    cache_ttl_sec: int = 7 * 24 * 3600,
) -> Tuple[Optional[str], FetchResult]:
    fr = fetch_bytes(
        url,
        user_agent=user_agent,
        timeout=timeout,
        max_bytes=max_bytes,
        referer=referer,
        cache_kind="html",
        cache_ttl_sec=cache_ttl_sec,
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
    hrefs = []
    imgs = []

    for h in p.hrefs:
        if not h:
            continue
        try:
            hrefs.append(urljoin(base_url, h))
        except Exception:
            pass

    for s in p.img_srcs:
        if not s:
            continue
        try:
            imgs.append(urljoin(base_url, s))
        except Exception:
            pass

    # dedupe preserve order
    def dedupe(xs):
        seen = set()
        out = []
        for x in xs:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    return dedupe(hrefs), dedupe(imgs)


# ----------------------------
# Gallerix-specific extraction
# ----------------------------
def is_album_url(u: str) -> bool:
    # Examples: https://gallerix.org/album/Vincent-Van-Gogh
    return "/album/" in u and "/pic/" not in u


def is_painting_url(u: str) -> bool:
    # Examples: https://gallerix.org/album/Vincent-Van-Gogh/pic/glrx-413874770
    return "/pic/" in u and "/album/" in u


def pick_seed_albums_from_a1(html: str) -> List[str]:
    """
    Pull album URLs from the Famous artists page.
    """
    links, _ = extract_links(html, SEED_PAGE)
    albums = [u for u in links if is_album_url(u)]
    # Some links may be repeated/irrelevant; keep reasonable subset
    return albums[:400]


def extract_paintings_from_album(html: str, album_url: str) -> List[str]:
    """
    Album page contains lots of /pic/ links. We’ll collect them.
    """
    links, _ = extract_links(html, album_url)
    pics = [u for u in links if is_painting_url(u)]
    return pics


def extract_title_artist_meta(html: str) -> Tuple[str, str, str]:
    """
    Best-effort extraction:
    - title and artist often appear in <h1> like: "Starry Night Vincent van Gogh (1853-1890)"
    - page also has a line: "Vincent van Gogh – Starry Night"
    """
    title = ""
    artist = ""
    meta = ""

    # h1
    m = re.search(r"(?is)<h1[^>]*>(.*?)</h1>", html)
    if m:
        h1 = re.sub(r"(?is)<.*?>", " ", m.group(1))
        h1 = re.sub(r"\s+", " ", h1).strip()
        # heuristic split: last name chunk may be artist; prefer second line:
        title = h1

    # "Painter: <a ...>Artist</a>"
    m = re.search(r"(?is)Painter:\s*<a[^>]*>(.*?)</a>", html)
    if m:
        artist = re.sub(r"(?is)<.*?>", " ", m.group(1))
        artist = re.sub(r"\s+", " ", artist).strip()

    # A “Vincent van Gogh – Starry Night” line (en dash)
    m = re.search(r"(?is)([A-Za-zÀ-ÿ0-9'’\-\s]+)\s*[–-]\s*([A-Za-zÀ-ÿ0-9'’\-\s]+)\s*</", html)
    if m and not artist:
        artist = m.group(1).strip()
        title2 = m.group(2).strip()
        if len(title2) > 0:
            title = title2

    # Year/size line is often near: "1889. 73.0 x 92.0 cm."
    m = re.search(r"(?m)^\s*(\d{3,4}\.\s*[^<]{0,120}cm\.)\s*$", html)
    if m:
        meta = m.group(1).strip()

    return title.strip(), artist.strip(), meta.strip()


def extract_primary_image_url(html: str, page_url: str) -> Optional[str]:
    """
    Many painting pages include an <img> that points to:
      https://cdn.gallerix.asia/... .webp
    We’ll prefer cdn.gallerix.asia, then any .webp/.jpg/.jpeg.
    """
    _, imgs = extract_links(html, page_url)

    # Prefer CDN images
    for u in imgs:
        if "cdn.gallerix" in u and (u.lower().endswith(".webp") or u.lower().endswith(".jpg") or u.lower().endswith(".jpeg") or u.lower().endswith(".png")):
            return u

    # Fallback: any image-like URL
    for u in imgs:
        if u.lower().endswith((".webp", ".jpg", ".jpeg", ".png")):
            return u

    # As a last resort, regex for cdn
    m = re.search(r"(https?://cdn\.gallerix\.[^\"'\s>]+?\.(?:webp|jpg|jpeg|png))", html, re.I)
    if m:
        return m.group(1)

    return None


# ----------------------------
# SQLite storage
# ----------------------------
def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db():
    c = db()
    c.execute(
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
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS votes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts INTEGER,
            left_url TEXT,
            right_url TEXT,
            winner_url TEXT
        )
        """
    )
    c.commit()
    c.close()


def upsert_painting(url: str, img_url: str, title: str, artist: str, meta: str):
    c = db()
    c.execute(
        """
        INSERT INTO paintings (url, img_url, title, artist, meta, last_seen)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(url) DO UPDATE SET
            img_url=excluded.img_url,
            title=COALESCE(excluded.title, paintings.title),
            artist=COALESCE(excluded.artist, paintings.artist),
            meta=COALESCE(excluded.meta, paintings.meta),
            last_seen=excluded.last_seen
        """,
        (url, img_url, title, artist, meta, now_ts()),
    )
    c.commit()
    c.close()


def get_pool(limit: int = 2000) -> List[Dict]:
    c = db()
    cur = c.execute(
        "SELECT url, img_url, title, artist, meta, elo, games, wins, losses FROM paintings ORDER BY last_seen DESC LIMIT ?",
        (limit,),
    )
    rows = cur.fetchall()
    c.close()
    out = []
    for r in rows:
        out.append(
            dict(
                url=r[0],
                img_url=r[1],
                title=r[2] or "",
                artist=r[3] or "",
                meta=r[4] or "",
                elo=float(r[5] or 1500.0),
                games=int(r[6] or 0),
                wins=int(r[7] or 0),
                losses=int(r[8] or 0),
            )
        )
    return out


def get_painting(url: str) -> Optional[Dict]:
    c = db()
    cur = c.execute(
        "SELECT url, img_url, title, artist, meta, elo, games, wins, losses FROM paintings WHERE url=?",
        (url,),
    )
    r = cur.fetchone()
    c.close()
    if not r:
        return None
    return dict(
        url=r[0],
        img_url=r[1],
        title=r[2] or "",
        artist=r[3] or "",
        meta=r[4] or "",
        elo=float(r[5] or 1500.0),
        games=int(r[6] or 0),
        wins=int(r[7] or 0),
        losses=int(r[8] or 0),
    )


def record_vote(left_url: str, right_url: str, winner_url: str):
    c = db()
    c.execute(
        "INSERT INTO votes (ts, left_url, right_url, winner_url) VALUES (?, ?, ?, ?)",
        (now_ts(), left_url, right_url, winner_url),
    )
    c.commit()
    c.close()


# ----------------------------
# Elo ranking + "value"
# ----------------------------
def elo_expected(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))


def elo_update(r_a: float, r_b: float, score_a: float, k: float) -> Tuple[float, float]:
    ea = elo_expected(r_a, r_b)
    eb = 1.0 - ea
    new_a = r_a + k * (score_a - ea)
    new_b = r_b + k * ((1.0 - score_a) - eb)
    return new_a, new_b


def k_factor(games: int) -> float:
    # bigger K early to learn faster; then stabilize
    if games < 10:
        return 48.0
    if games < 50:
        return 28.0
    return 16.0


def elo_to_value_score(elo: float) -> float:
    # Convert Elo into a 0..100-ish score (center at 1500).
    # 400 Elo ~ 10x odds => map 1100..1900 roughly to 0..100.
    return clamp((elo - 1100.0) / 8.0, 0.0, 100.0)


def elo_to_usd_estimate(elo: float) -> float:
    # purely illustrative "value":
    # - baseline $100 at 1500
    # - every +400 Elo multiplies by ~e^1 ≈ 2.718 (smooth growth)
    base = 100.0
    return base * math.exp((elo - 1500.0) / 400.0)


def apply_vote(winner_url: str, loser_url: str):
    w = get_painting(winner_url)
    l = get_painting(loser_url)
    if not w or not l:
        return

    rw, rl = w["elo"], l["elo"]
    kw = k_factor(w["games"])
    kl = k_factor(l["games"])
    k = (kw + kl) / 2.0

    new_w, new_l = elo_update(rw, rl, score_a=1.0, k=k)

    c = db()
    c.execute(
        """
        UPDATE paintings
        SET elo=?, games=games+1, wins=wins+1
        WHERE url=?
        """,
        (new_w, winner_url),
    )
    c.execute(
        """
        UPDATE paintings
        SET elo=?, games=games+1, losses=losses+1
        WHERE url=?
        """,
        (new_l, loser_url),
    )
    c.commit()
    c.close()


# ----------------------------
# Crawling pipeline
# ----------------------------
def crawl_seed_albums(user_agent: str, timeout: float, delay: float, max_albums: int) -> List[str]:
    html, fr = fetch_text(SEED_PAGE, user_agent=user_agent, timeout=timeout, max_bytes=2_000_000)
    if not html:
        return []
    albums = pick_seed_albums_from_a1(html)
    albums = albums[:max_albums]
    if delay > 0:
        time.sleep(delay)
    return albums


def crawl_album_for_paintings(
    album_url: str,
    user_agent: str,
    timeout: float,
    delay: float,
    max_paintings_per_album: int,
) -> List[str]:
    html, fr = fetch_text(album_url, user_agent=user_agent, timeout=timeout, max_bytes=3_000_000, referer=SEED_PAGE)
    if not html:
        return []
    pics = extract_paintings_from_album(html, album_url)
    # albums can be huge; cap
    pics = pics[:max_paintings_per_album]
    if delay > 0:
        time.sleep(delay)
    return pics


def ingest_painting_page(
    painting_url: str,
    user_agent: str,
    timeout: float,
    delay: float,
) -> bool:
    html, fr = fetch_text(painting_url, user_agent=user_agent, timeout=timeout, max_bytes=2_500_000, referer=painting_url)
    if not html:
        return False

    title, artist, meta = extract_title_artist_meta(html)
    img_url = extract_primary_image_url(html, painting_url)
    if not img_url:
        return False

    upsert_painting(painting_url, img_url, title, artist, meta)
    if delay > 0:
        time.sleep(delay)
    return True


def fetch_image_bytes(img_url: str, referer: str, user_agent: str, timeout: float) -> Optional[bytes]:
    fr = fetch_bytes(
        img_url,
        user_agent=user_agent,
        timeout=timeout,
        max_bytes=6_000_000,
        referer=referer,
        cache_kind="img",
        cache_ttl_sec=30 * 24 * 3600,  # 30 days
    )
    if not fr.ok or not fr.data:
        return None
    return fr.data


# ----------------------------
# Matchmaking (choose two paintings)
# ----------------------------
def choose_pair(pool: List[Dict], seed: int) -> Optional[Tuple[Dict, Dict]]:
    if len(pool) < 2:
        return None

    # Weight towards paintings with fewer games (more uncertainty),
    # but still allow all.
    rng = hashlib.sha1(f"{seed}-{now_ts()}".encode("utf-8")).hexdigest()
    # convert hex to ints for deterministic-ish sampling
    rints = [int(rng[i:i+8], 16) for i in range(0, 32, 8)]

    def weight(p):
        # 1/(games+1) encourages exploration
        return 1.0 / (p["games"] + 1.0)

    weights = [weight(p) for p in pool]
    total = sum(weights)
    if total <= 0:
        return pool[0], pool[1]

    def pick(offset: int) -> Dict:
        x = (rints[offset] % 10_000_000) / 10_000_000.0
        t = x * total
        acc = 0.0
        for p, w in zip(pool, weights):
            acc += w
            if acc >= t:
                return p
        return pool[-1]

    a = pick(0)
    b = pick(1)
    tries = 0
    while b["url"] == a["url"] and tries < 10:
        b = pick(2)
        tries += 1
    if a["url"] == b["url"]:
        return pool[0], pool[1]
    return a, b


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title=APP_NAME, page_icon="🖼️", layout="wide")
init_db()

st.title("🖼️ Art Mash")
st.caption(
    "FaceMash-style voting for historical paintings (source: Gallerix). "
    "Votes update Elo rankings + a derived 'value' score. "
    "Please keep crawling limits low and be respectful."
)

with st.sidebar:
    st.header("Source + crawling")
    user_agent = st.text_input("User-Agent", value=DEFAULT_UA)
    timeout = st.slider("Timeout (sec)", 3, 30, 10, 1)
    crawl_delay = st.slider("Crawl delay (sec)", 0.0, 2.0, 0.2, 0.05)

    max_albums = st.slider("Albums to sample", 1, 40, 10, 1)
    max_paintings_per_album = st.slider("Painting links per album", 5, 200, 40, 5)
    ingest_cap = st.slider("Max painting pages to ingest now", 5, 300, 60, 5)

    st.divider()
    st.header("Voting")
    seed = st.number_input("Match seed", 0, 10_000_000, 1337, 1)
    show_value_usd = st.checkbox("Show $ estimate", value=True)
    image_max_width = st.slider("Image max width (px)", 250, 900, 520, 10)

    st.divider()
    if st.button("🧹 Clear cache (local)", use_container_width=True):
        # careful: remove only our cache folder
        try:
            import shutil
            shutil.rmtree(CACHE_DIR, ignore_errors=True)
            st.success("Cache cleared.")
        except Exception as e:
            st.error(f"Failed to clear cache: {e}")


tab1, tab2, tab3 = st.tabs(["Vote", "Crawl/Refresh Pool", "Leaderboard"])

# ----------------------------
# Vote tab
# ----------------------------
with tab1:
    pool = get_pool(limit=2000)

    st.write(f"Pool size (ingested paintings): **{len(pool)}**")

    if len(pool) < 2:
        st.info("Pool is empty. Go to **Crawl/Refresh Pool** to ingest paintings first.")
    else:
        pair = choose_pair(pool, seed=seed)
        if not pair:
            st.warning("Unable to choose a pair.")
        else:
            left, right = pair

            # fetch images
            left_img = fetch_image_bytes(left["img_url"], referer=left["url"], user_agent=user_agent, timeout=float(timeout))
            right_img = fetch_image_bytes(right["img_url"], referer=right["url"], user_agent=user_agent, timeout=float(timeout))

            colL, colR = st.columns(2, gap="large")

            def render_card(col, p, img_bytes, side_label: str):
                with col:
                    st.subheader(side_label)
                    if img_bytes:
                        st.image(img_bytes, use_container_width=True)
                    else:
                        st.warning("Image fetch failed (site may block hotlinking). Try later / smaller crawl / different UA.")
                    st.markdown(f"**Title:** {p['title'] or '(unknown)'}")
                    st.markdown(f"**Artist:** {p['artist'] or '(unknown)'}")
                    if p["meta"]:
                        st.markdown(f"**Info:** {p['meta']}")
                    st.markdown(f"**Elo:** `{p['elo']:.1f}`  |  **Games:** `{p['games']}`")
                    vs = elo_to_value_score(p["elo"])
                    st.markdown(f"**Value score:** `{vs:.1f}/100`")
                    if show_value_usd:
                        usd = elo_to_usd_estimate(p["elo"])
                        st.markdown(f"**Est. value (illustrative):** `${usd:,.0f}`")
                    st.markdown(f"[Open on Gallerix]({p['url']})")

            render_card(colL, left, left_img, "A")
            render_card(colR, right, right_img, "B")

            b1, b2, b3 = st.columns([1, 1, 1])
            vote_a = b1.button("✅ Vote A (left)", use_container_width=True)
            vote_b = b2.button("✅ Vote B (right)", use_container_width=True)
            skip = b3.button("↩ Skip", use_container_width=True)

            if vote_a:
                record_vote(left["url"], right["url"], left["url"])
                apply_vote(left["url"], right["url"])
                st.success("Saved vote for A. Refreshing matchup…")
                st.rerun()

            if vote_b:
                record_vote(left["url"], right["url"], right["url"])
                apply_vote(right["url"], left["url"])
                st.success("Saved vote for B. Refreshing matchup…")
                st.rerun()

            if skip:
                st.rerun()

# ----------------------------
# Crawl tab
# ----------------------------
with tab2:
    st.subheader("Crawl / Refresh Pool")
    st.write(
        "This step ingests painting pages into the local SQLite database. "
        "Keep limits small and delays non-zero to avoid hammering the site."
    )

    if st.button("🔎 Crawl seed albums from /a1 and ingest paintings", type="primary", use_container_width=True):
        prog = st.progress(0)
        status = st.empty()

        albums = crawl_seed_albums(user_agent=user_agent, timeout=float(timeout), delay=float(crawl_delay), max_albums=int(max_albums))
        if not albums:
            st.error("Failed to load seed albums.")
        else:
            status.info(f"Found {len(albums)} albums. Extracting painting links…")

            # gather painting urls
            painting_urls: List[str] = []
            for i, alb in enumerate(albums):
                pics = crawl_album_for_paintings(
                    alb,
                    user_agent=user_agent,
                    timeout=float(timeout),
                    delay=float(crawl_delay),
                    max_paintings_per_album=int(max_paintings_per_album),
                )
                painting_urls.extend(pics)
                prog.progress(int(30 * (i + 1) / max(1, len(albums))))

            # dedupe
            seen = set()
            uniq = []
            for u in painting_urls:
                if u not in seen:
                    seen.add(u)
                    uniq.append(u)
            painting_urls = uniq[: int(ingest_cap)]

            status.info(f"Ingesting {len(painting_urls)} painting pages…")
            ok = 0
            for i, pu in enumerate(painting_urls):
                if ingest_painting_page(pu, user_agent=user_agent, timeout=float(timeout), delay=float(crawl_delay)):
                    ok += 1
                prog.progress(30 + int(70 * (i + 1) / max(1, len(painting_urls))))

            status.success(f"Ingest complete. Added/updated {ok} paintings.")
            st.rerun()

    st.divider()
    st.subheader("Manual ingest")
    manual_urls = st.text_area(
        "Paste painting page URLs (one per line), e.g. https://gallerix.org/album/<album>/pic/<...>",
        height=140,
    )
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

# ----------------------------
# Leaderboard tab
# ----------------------------
with tab3:
    st.subheader("Leaderboard")
    st.write("Top-ranked paintings by Elo. (Rankings improve with more votes.)")

    c = db()
    cur = c.execute(
        """
        SELECT url, img_url, title, artist, meta, elo, games, wins, losses
        FROM paintings
        WHERE games >= 1
        ORDER BY elo DESC
        LIMIT 200
        """
    )
    rows = cur.fetchall()
    c.close()

    if not rows:
        st.info("No votes yet. Go to **Vote** and start ranking.")
    else:
        table = []
        for i, r in enumerate(rows, 1):
            elo = float(r[5] or 1500.0)
            table.append(
                {
                    "rank": i,
                    "title": r[2] or "",
                    "artist": r[3] or "",
                    "elo": round(elo, 1),
                    "value_score": round(elo_to_value_score(elo), 1),
                    "games": int(r[6] or 0),
                    "wins": int(r[7] or 0),
                    "losses": int(r[8] or 0),
                    "link": r[0],
                }
            )

        st.dataframe(table, use_container_width=True, height=520)

        st.download_button(
            "Download leaderboard JSON",
            data=json.dumps(table, indent=2).encode("utf-8"),
            file_name="artmash_leaderboard.json",
            mime="application/json",
            use_container_width=True,
        )

st.caption(
    "Tip: The first 50–200 votes will stabilize rankings a lot. "
    "Keep crawl limits small and use the cache."
)
