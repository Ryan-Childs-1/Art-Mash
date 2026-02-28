# app.py
# Art Mash — historical art FaceMash using Gallerix (no server-side image downloading)
#
# FIXES:
# - Images are displayed client-side using either:
#   (A) iframe embedding the painting page (most reliable)
#   (B) HTML <img src="..."> (browser fetch, not server fetch)
#
# FEATURES:
# - TrueSkill-lite ratings: mu/sigma with uncertainty-based matchmaking
# - Tournament mode (single elimination bracket)
# - Hot/New feeds
# - Artist-level rankings
# - Style clustering (heuristic tags)
#
# Dependencies: streamlit only (stdlib otherwise)

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
import streamlit.components.v1 as components


# ----------------------------
# Constants
# ----------------------------
APP_NAME = "Art Mash"
BASE = "https://gallerix.org"
SEED_PAGE = "https://gallerix.org/a1/"  # popular albums list

DEFAULT_UA = "ArtMash/2.0 (Streamlit; respectful crawler)"
CACHE_DIR = ".cache_artmash"
DB_PATH = "artmash.sqlite3"

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
# Gallerix-specific extraction
# ----------------------------
def is_album_url(u: str) -> bool:
    return "/album/" in u and "/pic/" not in u


def is_painting_url(u: str) -> bool:
    return "/pic/" in u and "/album/" in u


def pick_seed_albums_from_a1(html: str) -> List[str]:
    links, _ = extract_links(html, SEED_PAGE)
    albums = [u for u in links if is_album_url(u)]
    return albums[:400]


def extract_paintings_from_album(html: str, album_url: str) -> List[str]:
    links, _ = extract_links(html, album_url)
    pics = [u for u in links if is_painting_url(u)]
    return pics


def extract_title_artist_meta_and_text(html: str) -> Tuple[str, str, str, str]:
    title, artist, meta = "", "", ""
    # quick title
    m = re.search(r"(?is)<h1[^>]*>(.*?)</h1>", html)
    if m:
        h1 = re.sub(r"(?is)<.*?>", " ", m.group(1))
        title = re.sub(r"\s+", " ", h1).strip()

    m = re.search(r"(?is)Painter:\s*<a[^>]*>(.*?)</a>", html)
    if m:
        artist = re.sub(r"(?is)<.*?>", " ", m.group(1))
        artist = re.sub(r"\s+", " ", artist).strip()

    # year/size line e.g. "1889. 73.0 x 92.0 cm."
    m = re.search(r"(?m)^\s*(\d{3,4}\.\s*[^<]{0,140}cm\.)\s*$", html)
    if m:
        meta = m.group(1).strip()

    # strip text for style heuristics
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
            tags TEXT,
            -- legacy Elo:
            elo REAL DEFAULT 1500.0,
            -- TrueSkill-lite:
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
    c.execute(
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
    c.commit()
    c.close()


def upsert_painting(url: str, img_url: str, title: str, artist: str, meta: str, tags: List[str]):
    c = db()
    c.execute(
        """
        INSERT INTO paintings (url, img_url, title, artist, meta, tags, elo, mu, sigma, last_seen)
        VALUES (?, ?, ?, ?, ?, ?, 1500.0, ?, ?, ?)
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
    c.commit()
    c.close()


def get_pool(limit: int = 2000) -> List[Dict]:
    c = db()
    cur = c.execute(
        """
        SELECT url, img_url, title, artist, meta, tags, elo, mu, sigma, games, wins, losses, last_seen, last_vote
        FROM paintings
        ORDER BY last_seen DESC
        LIMIT ?
        """,
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


def get_painting(url: str) -> Optional[Dict]:
    c = db()
    cur = c.execute(
        """
        SELECT url, img_url, title, artist, meta, tags, elo, mu, sigma, games, wins, losses, last_seen, last_vote
        FROM paintings WHERE url=?
        """,
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


def record_vote(left_url: str, right_url: str, winner_url: str, mode: str):
    c = db()
    c.execute(
        "INSERT INTO votes (ts, left_url, right_url, winner_url, mode) VALUES (?, ?, ?, ?, ?)",
        (now_ts(), left_url, right_url, winner_url, mode),
    )
    c.commit()
    c.close()


# ----------------------------
# Rating systems
# ----------------------------
def normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def normal_cdf(x: float) -> float:
    # Abramowitz & Stegun approximation via erf
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def trueskill_lite_update(
    mu_a: float, sig_a: float,
    mu_b: float, sig_b: float,
    a_wins: bool,
    beta: float = TS_BETA,
    tau: float = TS_TAU,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Two-player TrueSkill-like update (no draws):
    - Inflate sigmas slightly by tau (dynamics)
    - Compute performance difference distribution
    - Update mu and sigma using v,w factors
    """
    # dynamics
    sig_a = math.sqrt(sig_a * sig_a + tau * tau)
    sig_b = math.sqrt(sig_b * sig_b + tau * tau)

    c2 = 2 * beta * beta + sig_a * sig_a + sig_b * sig_b
    c = math.sqrt(c2) + 1e-12

    # if A wins: t = (mu_a - mu_b)/c ; else flipped sign
    t = (mu_a - mu_b) / c
    if not a_wins:
        t = -t

    # v, w
    Phi = normal_cdf(t)
    phi = normal_pdf(t)
    v = phi / max(Phi, 1e-12)
    w = v * (v + t)

    # update
    rank_mult = 1.0  # no draws/tiers
    delta_mu_a = (sig_a * sig_a / c) * v * rank_mult
    delta_mu_b = (sig_b * sig_b / c) * v * rank_mult

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
    # conservative "skill" value like TrueSkill: mu - 3*sigma
    return mu - 3.0 * sigma


def value_score_0_100(v: float) -> float:
    # Map a typical TrueSkill conservative range into 0..100
    # v ~ [0..40] generally, so scale to 0..100
    return clamp((v / 40.0) * 100.0, 0.0, 100.0)


def apply_vote(winner_url: str, loser_url: str, mode: str = "vote"):
    w = get_painting(winner_url)
    l = get_painting(loser_url)
    if not w or not l:
        return

    # Update Elo for backwards-compat + simple leaderboard
    rw, rl = w["elo"], l["elo"]
    k = (k_factor(w["games"]) + k_factor(l["games"])) / 2.0
    new_rw, new_rl = elo_update(rw, rl, score_a=1.0, k=k)

    # Update TrueSkill-lite
    (mu_w, sig_w), (mu_l, sig_l) = trueskill_lite_update(
        w["mu"], w["sigma"], l["mu"], l["sigma"], a_wins=True
    )

    c = db()
    c.execute(
        """
        UPDATE paintings
        SET elo=?, mu=?, sigma=?, games=games+1, wins=wins+1, last_vote=?
        WHERE url=?
        """,
        (new_rw, mu_w, sig_w, now_ts(), winner_url),
    )
    c.execute(
        """
        UPDATE paintings
        SET elo=?, mu=?, sigma=?, games=games+1, losses=losses+1, last_vote=?
        WHERE url=?
        """,
        (new_rl, mu_l, sig_l, now_ts(), loser_url),
    )
    c.commit()
    c.close()


# ----------------------------
# Crawling pipeline (HTML only)
# ----------------------------
def crawl_seed_albums(user_agent: str, timeout: float, delay: float, max_albums: int) -> List[str]:
    html, fr = fetch_text(SEED_PAGE, user_agent=user_agent, timeout=timeout, max_bytes=2_000_000)
    if not html:
        return []
    albums = pick_seed_albums_from_a1(html)[:max_albums]
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
    pics = extract_paintings_from_album(html, album_url)[:max_paintings_per_album]
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

    title, artist, meta, text = extract_title_artist_meta_and_text(html)
    img_url = extract_primary_image_url(html, painting_url)
    if not img_url:
        # Still usable via iframe even without img_url, but we store page anyway with blank img_url.
        img_url = ""

    tags = infer_style_tags(text)
    upsert_painting(painting_url, img_url, title, artist, meta, tags)

    if delay > 0:
        time.sleep(delay)
    return True


# ----------------------------
# Display helpers (no server-side downloads)
# ----------------------------
def render_painting_display(p: Dict, display_mode: str, height: int = 620):
    """
    display_mode:
      - "iframe": embed painting page (most reliable)
      - "img": render img_url as browser-side <img> (fast, may fail if blocked)
    """
    url = p["url"]
    img_url = p.get("img_url") or ""

    if display_mode == "iframe":
        # iframe is most robust and avoids server-side downloading.
        components.iframe(url, height=height, scrolling=True)
        return

    # "img" mode
    if not img_url:
        st.warning("No image URL extracted; switching to iframe view is recommended.")
        components.iframe(url, height=height, scrolling=True)
        return

    # Browser-side <img> so the user's browser fetches the image (not Streamlit server).
    # Add max-width and a dark background frame.
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
    """
    TrueSkill-style matchmaking:
    - Prefer high sigma (uncertain paintings) so ranking improves fast.
    - Also prefer not-yet-voted items.
    """
    if len(pool) < 2:
        return None

    # build weights based on sigma and recency
    weights = []
    for p in pool:
        # higher sigma => more uncertain => explore
        w = (p["sigma"] ** 1.6)
        # encourage new paintings
        w *= 1.0 + 0.4 / (p["games"] + 1.0)
        weights.append(w)

    probs = softmax([math.log(w + 1e-9) for w in weights])

    # deterministic-ish pick based on hash of seed + time bucket
    bucket = int(time.time() // 5)  # changes every 5 seconds
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

    # OPTIONAL: make matchups closer in skill (reduce blowouts)
    # Find candidate b near a’s mu
    target = a["mu"]
    candidates = sorted(pool, key=lambda p: abs(p["mu"] - target))
    for cand in candidates[:30]:
        if cand["url"] != a["url"]:
            b = cand
            break

    return a, b


def feed_new(pool: List[Dict], n: int = 30) -> List[Dict]:
    # least games, most recently seen
    return sorted(pool, key=lambda p: (p["games"], -p["last_seen"]))[:n]


def feed_hot(pool: List[Dict], n: int = 30) -> List[Dict]:
    # "hot" = recent activity + good winrate with at least a few games
    def score(p):
        g = p["games"]
        if g < 5:
            return -1e9
        winrate = p["wins"] / max(1, g)
        rec = p["last_vote"]
        return 2.5 * winrate + 0.002 * rec + 0.08 * mu_sigma_to_value(p["mu"], p["sigma"])
    return sorted(pool, key=score, reverse=True)[:n]


def artist_rankings(pool: List[Dict], topk: int = 5) -> List[Dict]:
    by_artist: Dict[str, List[Dict]] = {}
    for p in pool:
        a = (p["artist"] or "").strip()
        if not a:
            continue
        by_artist.setdefault(a, []).append(p)

    rows = []
    for artist, items in by_artist.items():
        items_sorted = sorted(items, key=lambda p: mu_sigma_to_value(p["mu"], p["sigma"]), reverse=True)
        pick = items_sorted[:topk]
        agg = sum(mu_sigma_to_value(p["mu"], p["sigma"]) for p in pick) / max(1, len(pick))
        games = sum(p["games"] for p in items)
        rows.append({"artist": artist, "rating": agg, "paintings": len(items), "games": games})
    rows.sort(key=lambda r: r["rating"], reverse=True)
    return rows[:200]


def style_clusters(pool: List[Dict]) -> Dict[str, List[Dict]]:
    clusters: Dict[str, List[Dict]] = {}
    for p in pool:
        tags = p.get("tags") or []
        if not tags:
            clusters.setdefault("Unlabeled", []).append(p)
        else:
            for t in tags[:2]:
                clusters.setdefault(t, []).append(p)

    # sort within cluster by conservative value
    for k in clusters:
        clusters[k] = sorted(clusters[k], key=lambda p: mu_sigma_to_value(p["mu"], p["sigma"]), reverse=True)
    return clusters


# ----------------------------
# Tournament mode
# ----------------------------
def start_tournament(pool: List[Dict], size: int, seed: int) -> List[str]:
    """
    Return a list of painting URLs in initial bracket order.
    """
    if len(pool) < size:
        size = len(pool)

    # choose a mix: half uncertain, half top-value
    pool_sorted_unc = sorted(pool, key=lambda p: p["sigma"], reverse=True)
    pool_sorted_top = sorted(pool, key=lambda p: mu_sigma_to_value(p["mu"], p["sigma"]), reverse=True)

    chosen = []
    i = 0
    while len(chosen) < size and i < max(len(pool_sorted_unc), len(pool_sorted_top)):
        if i < len(pool_sorted_unc):
            chosen.append(pool_sorted_unc[i]["url"])
        if len(chosen) < size and i < len(pool_sorted_top):
            chosen.append(pool_sorted_top[i]["url"])
        i += 1
    chosen = list(dict.fromkeys(chosen))[:size]

    # deterministic shuffle by seed
    h = hashlib.sha1(f"tourn-{seed}".encode("utf-8")).hexdigest()
    rot = int(h[:8], 16) % max(1, len(chosen))
    chosen = chosen[rot:] + chosen[:rot]
    return chosen


def tournament_round_pairs(urls: List[str]) -> List[Tuple[str, str]]:
    pairs = []
    for i in range(0, len(urls) - 1, 2):
        pairs.append((urls[i], urls[i + 1]))
    return pairs


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title=APP_NAME, page_icon="🖼️", layout="wide")
init_db()

st.title("🖼️ Art Mash (Fixed Display + TrueSkill + Tournament)")
st.caption(
    "Vote between two paintings. Rankings use TrueSkill-lite (mu/sigma). "
    "Images are displayed without server-side downloading."
)

with st.sidebar:
    st.header("Source + crawl")
    user_agent = st.text_input("User-Agent", value=DEFAULT_UA)
    timeout = st.slider("Timeout (sec)", 3, 30, 10, 1)
    crawl_delay = st.slider("Crawl delay (sec)", 0.0, 2.0, 0.2, 0.05)

    max_albums = st.slider("Albums to sample", 1, 40, 10, 1)
    max_paintings_per_album = st.slider("Painting links per album", 5, 200, 40, 5)
    ingest_cap = st.slider("Max painting pages to ingest now", 5, 300, 60, 5)

    st.divider()
    st.header("Display")
    display_mode = st.selectbox("Display mode", ["iframe (reliable)", "img tag (fast)"], index=0)
    disp_mode_key = "iframe" if display_mode.startswith("iframe") else "img"
    iframe_height = st.slider("Display height", 360, 900, 620, 10)

    st.divider()
    st.header("Voting / matchmaking")
    seed = st.number_input("Match seed", 0, 10_000_000, 1337, 1)
    show_meta = st.checkbox("Show metadata", value=True)

    st.divider()
    if st.button("🧹 Clear HTML cache (local)", use_container_width=True):
        try:
            import shutil
            shutil.rmtree(CACHE_DIR, ignore_errors=True)
            st.success("Cache cleared.")
        except Exception as e:
            st.error(f"Failed: {e}")


tab_vote, tab_crawl, tab_tourn, tab_feeds, tab_leaders = st.tabs(
    ["Vote", "Crawl/Refresh", "Tournament", "Hot/New Feeds", "Leaderboards"]
)

# ----------------------------
# Vote tab
# ----------------------------
with tab_vote:
    pool = get_pool(limit=4000)
    st.write(f"Pool size (ingested paintings): **{len(pool)}**")

    if len(pool) < 2:
        st.info("Pool is empty. Go to **Crawl/Refresh** to ingest paintings first.")
    else:
        pair = choose_pair_uncertainty(pool, seed=seed)
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
                        st.markdown(f"**Conservative value:** `{v:.2f}`  |  **Score:** `{value_score_0_100(v):.1f}/100`")
                        st.markdown(f"**Games:** `{p['games']}`  |  **W-L:** `{p['wins']}-{p['losses']}`")
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
# Crawl tab
# ----------------------------
with tab_crawl:
    st.subheader("Crawl / Refresh Pool (HTML only)")
    st.write(
        "This ingests painting pages into SQLite. "
        "The app never downloads image bytes server-side. "
        "Keep limits small + delays non-zero."
    )

    if st.button("🔎 Crawl seed albums and ingest paintings", type="primary", use_container_width=True):
        prog = st.progress(0)
        status = st.empty()

        albums = crawl_seed_albums(
            user_agent=user_agent,
            timeout=float(timeout),
            delay=float(crawl_delay),
            max_albums=int(max_albums),
        )
        if not albums:
            st.error("Failed to load seed albums.")
        else:
            status.info(f"Found {len(albums)} albums. Extracting painting links…")

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

            # dedupe + cap
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
        "Paste painting page URLs (one per line)",
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
# Tournament tab
# ----------------------------
with tab_tourn:
    st.subheader("Tournament Mode (Single Elimination)")
    pool = get_pool(limit=4000)

    if len(pool) < 8:
        st.info("Need at least 8 ingested paintings. Crawl more first.")
    else:
        size = st.select_slider("Tournament size", options=[8, 16, 32, 64], value=16)
        t_seed = st.number_input("Tournament seed", 0, 10_000_000, 2026, 1)
        if st.button("Start new tournament", type="primary", use_container_width=True):
            bracket = start_tournament(pool, size=size, seed=int(t_seed))
            st.session_state["tourn_urls"] = bracket
            st.session_state["tourn_round"] = 1
            st.session_state["tourn_winners"] = []
            st.rerun()

        urls = st.session_state.get("tourn_urls")
        rnd = st.session_state.get("tourn_round")
        winners = st.session_state.get("tourn_winners", [])

        if urls:
            st.markdown(f"**Round {rnd}** — remaining: `{len(urls)}`")
            pairs = tournament_round_pairs(urls)

            if not pairs:
                champ_url = urls[0]
                champ = get_painting(champ_url)
                st.success(f"🏆 Champion: {champ.get('title','(unknown)')} — {champ.get('artist','')}")
                render_painting_display(champ, disp_mode_key, height=int(iframe_height))
            else:
                # play the first matchup only (clean UX)
                a_url, b_url = pairs[0]
                A = get_painting(a_url)
                B = get_painting(b_url)
                col1, col2 = st.columns(2, gap="large")
                with col1:
                    st.subheader("A")
                    render_painting_display(A, disp_mode_key, height=int(iframe_height))
                    st.caption(A["title"])
                with col2:
                    st.subheader("B")
                    render_painting_display(B, disp_mode_key, height=int(iframe_height))
                    st.caption(B["title"])

                c1, c2, c3 = st.columns(3)
                if c1.button("Vote A wins", use_container_width=True):
                    record_vote(A["url"], B["url"], A["url"], mode="tournament")
                    apply_vote(A["url"], B["url"], mode="tournament")
                    winners.append(A["url"])
                    # remove first two
                    rest = urls[2:]
                    st.session_state["tourn_urls"] = winners + rest
                    st.session_state["tourn_winners"] = winners
                    st.rerun()

                if c2.button("Vote B wins", use_container_width=True):
                    record_vote(A["url"], B["url"], B["url"], mode="tournament")
                    apply_vote(B["url"], A["url"], mode="tournament")
                    winners.append(B["url"])
                    rest = urls[2:]
                    st.session_state["tourn_urls"] = winners + rest
                    st.session_state["tourn_winners"] = winners
                    st.rerun()

                if c3.button("Skip matchup", use_container_width=True):
                    # rotate
                    st.session_state["tourn_urls"] = urls[2:] + urls[:2]
                    st.rerun()

            # advance round when all pairs voted
            # when length is power of two and winners filled up to half, start next round
            if len(urls) in (8, 16, 32, 64) and len(winners) == len(urls) // 2:
                st.session_state["tourn_urls"] = winners
                st.session_state["tourn_round"] = int(rnd) + 1
                st.session_state["tourn_winners"] = []
                st.rerun()

# ----------------------------
# Feeds tab
# ----------------------------
with tab_feeds:
    pool = get_pool(limit=4000)
    st.subheader("Hot & New")

    colA, colB = st.columns(2, gap="large")
    with colA:
        st.markdown("### 🔥 Hot")
        hot = feed_hot(pool, n=40)
        for p in hot[:15]:
            v = mu_sigma_to_value(p["mu"], p["sigma"])
            st.write(f"**{p['title'][:70]}** — {p['artist']} | score {value_score_0_100(v):.1f} | games {p['games']}")
            st.caption(p["url"])

    with colB:
        st.markdown("### 🆕 New")
        new = feed_new(pool, n=40)
        for p in new[:15]:
            st.write(f"**{p['title'][:70]}** — {p['artist']} | games {p['games']}")
            st.caption(p["url"])

# ----------------------------
# Leaderboards tab
# ----------------------------
with tab_leaders:
    pool = get_pool(limit=4000)
    st.subheader("Leaderboards (TrueSkill-lite)")

    # Painting leaderboard by conservative value
    top = sorted(pool, key=lambda p: mu_sigma_to_value(p["mu"], p["sigma"]), reverse=True)[:200]
    rows = []
    for i, p in enumerate(top, 1):
        v = mu_sigma_to_value(p["mu"], p["sigma"])
        rows.append({
            "rank": i,
            "title": p["title"],
            "artist": p["artist"],
            "score_0_100": round(value_score_0_100(v), 1),
            "mu": round(p["mu"], 2),
            "sigma": round(p["sigma"], 2),
            "games": p["games"],
            "wins": p["wins"],
            "losses": p["losses"],
            "tags": ", ".join(p.get("tags") or []),
            "url": p["url"],
        })

    st.dataframe(rows, use_container_width=True, height=520)
    st.download_button(
        "Download paintings leaderboard JSON",
        data=json.dumps(rows, indent=2).encode("utf-8"),
        file_name="artmash_paintings_leaderboard.json",
        mime="application/json",
        use_container_width=True,
    )

    st.divider()
    st.subheader("Artist-level rankings")
    artist_rows = artist_rankings(pool, topk=5)
    st.dataframe(artist_rows, use_container_width=True, height=420)

    st.divider()
    st.subheader("Style clusters (heuristic tags)")
    clusters = style_clusters(pool)
    sel = st.selectbox("Pick a style cluster", sorted(clusters.keys()))
    sample = clusters.get(sel, [])[:30]
    for p in sample[:15]:
        v = mu_sigma_to_value(p["mu"], p["sigma"])
        st.write(f"**{p['title'][:70]}** — {p['artist']} | score {value_score_0_100(v):.1f} | games {p['games']}")
        st.caption(p["url"])

st.caption("Tip: Use iframe display mode if images aren’t showing. It avoids hotlink/CDN restrictions entirely.")
