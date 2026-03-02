# app.py
# Art Mash — Flat-file version (all code runs from the same folder)
#
# Run:
#   streamlit run app.py
#
# Optional DB path override:
#   ARTMASH_DB_PATH=/path/to/artmash.sqlite3 streamlit run app.py

import json
import random

import streamlit as st

import core as C

st.set_page_config(page_title=C.APP_NAME, page_icon="🖼️", layout="wide")
C.init_db()

st.title("🖼️ Art Mash")
st.caption("Votes persist to SQLite and leaderboards update live after every vote. Artist leaderboard is canonical by artist URL.")

with st.sidebar:
    st.header("Queue")
    queue_limit = st.slider("Queue size (from DB)", 200, 20000, 4000, 200, key="sb_queue_limit")
    queue_seed = st.number_input("Queue seed", 0, 10_000_000, 1337, 1, key="sb_queue_seed")
    rebuild_queue = st.button("🔀 Rebuild session queue", use_container_width=True, key="sb_rebuild_queue")

    st.divider()
    st.header("Display")
    display_mode = st.selectbox("Display mode", ["iframe (reliable)", "img tag (fast)"], index=0, key="sb_display_mode")
    disp_mode_key = "iframe" if display_mode.startswith("iframe") else "img"
    iframe_height = st.slider("Display height", 360, 900, 620, 10, key="sb_iframe_height")
    show_meta = st.checkbox("Show metadata", value=True, key="sb_show_meta")

tabs = st.tabs(["Vote", "Bulk Load", "Leaderboards"])

# Session init
if rebuild_queue or "queue_urls" not in st.session_state:
    st.session_state["queue_urls"] = C.build_session_queue(limit=int(queue_limit), seed=int(queue_seed))
    st.session_state["queue_idx"] = 0

if "seen_urls" not in st.session_state:
    st.session_state["seen_urls"] = set()

DEFAULT_TIMEOUT = 10.0
DEFAULT_DELAY = 0.2
DEFAULT_UA_LOCAL = C.DEFAULT_UA

# ----------------------------
# Vote tab
# ----------------------------
with tabs[0]:
    st.write(
        f"Session queue size: **{len(st.session_state['queue_urls'])}**  |  Queue position: **{st.session_state.get('queue_idx',0)}**"
    )

    pair = C.pick_pair_from_queue(st.session_state)
    if not pair:
        st.warning("Not enough paintings in the queue. Use Bulk Load to ingest more paintings, or increase queue size.")
    else:
        left, right = pair
        st.session_state["seen_urls"].add(left["url"])
        st.session_state["seen_urls"].add(right["url"])

        # For img-tag mode, enrich missing img_url/title/artist lazily
        if disp_mode_key == "img":
            if (not left.get("img_url")) or (not left.get("title")) or (not left.get("artist")):
                C.ingest_painting_page_full(left["url"], DEFAULT_UA_LOCAL, DEFAULT_TIMEOUT, 0.0)
                left = C.get_painting(left["url"]) or left
            if (not right.get("img_url")) or (not right.get("title")) or (not right.get("artist")):
                C.ingest_painting_page_full(right["url"], DEFAULT_UA_LOCAL, DEFAULT_TIMEOUT, 0.0)
                right = C.get_painting(right["url"]) or right

        colL, colR = st.columns(2, gap="large")

        def card(col, p, label):
            with col:
                st.subheader(label)
                C.render_painting_display(p, disp_mode_key, height=int(iframe_height))

                if show_meta:
                    v = C.mu_sigma_to_value(p["mu"], p["sigma"])
                    st.markdown(
                        f"**Score:** `{C.value_score_0_100(v):.1f}/100`  |  **μ/σ:** `{p['mu']:.2f}/{p['sigma']:.2f}`  |  **Games:** `{p['games']}`"
                    )
                    if p.get("title"):
                        st.markdown(f"**Title:** {p['title']}")
                    if p.get("artist"):
                        st.markdown(f"**Artist:** {p['artist']}")
                    st.markdown(f"[Open]({p['url']})")

        card(colL, left, "A")
        card(colR, right, "B")

        b1, b2, b3 = st.columns([1, 1, 1])
        vote_a = b1.button("✅ Vote A", use_container_width=True)
        vote_b = b2.button("✅ Vote B", use_container_width=True)
        skip = b3.button("↩ Skip", use_container_width=True)

        if vote_a:
            C.record_vote(left["url"], right["url"], left["url"], mode="vote")
            C.apply_vote(left["url"], right["url"])
            st.rerun()

        if vote_b:
            C.record_vote(left["url"], right["url"], right["url"], mode="vote")
            C.apply_vote(right["url"], left["url"])
            st.rerun()

        if skip:
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
        user_agent = st.text_input("User-Agent", value=DEFAULT_UA_LOCAL)
        timeout = st.slider("Timeout (sec)", 3, 30, 10, 1)
        crawl_delay = st.slider("Crawl delay (sec)", 0.0, 2.0, 0.2, 0.05)

        cA, cB, cC = st.columns(3)
        with cA:
            letters_per_run = st.slider("Letters per run (artists)", 1, 26, 6, 1)
        with cB:
            artists_per_run = st.slider("Artists per run (paintings)", 1, 500, 50, 1)
        with cC:
            cap_per_artist = st.slider("Cap paintings per artist (0 = no cap)", 0, 5000, 0, 50)

    artists_json = C.ingest_state_get("artists_json", "[]")
    try:
        artists_list = json.loads(artists_json)
    except Exception:
        artists_list = []
    letter_pos = int(C.ingest_state_get("artist_letter_pos", "0") or "0")
    artist_idx = int(C.ingest_state_get("artist_idx", "0") or "0")

    st.info(
        f"Artists loaded: **{len(artists_list)}** | Letter progress: **{letter_pos}/26** | Artist progress: **{artist_idx}/{len(artists_list) or 0}**"
    )

    c1, c2 = st.columns(2, gap="large")

    with c1:
        if st.button("1) Load artists (batch)", type="primary", use_container_width=True):
            added_urls, updated_names, status = C.load_all_artists_batch(
                user_agent=user_agent,
                timeout=float(timeout),
                delay=float(crawl_delay),
                letters_per_run=int(letters_per_run),
            )
            st.success(f"Added {added_urls} artist URLs; updated {updated_names} names. {status}")
            st.rerun()

        if st.button("Reset letter progress", use_container_width=True):
            C.ingest_state_set("artist_letter_pos", "0")
            st.warning("Reset letter progress to 0.")
            st.rerun()

    with c2:
        if st.button("2) Load paintings (batch)", type="primary", use_container_width=True):
            unique_inserts, artists_done, missing_names, status = C.load_all_paintings_batch(
                user_agent=user_agent,
                timeout=float(timeout),
                delay=float(crawl_delay),
                artists_per_run=int(artists_per_run),
                paintings_cap_per_artist=int(cap_per_artist),
            )
            st.success(
                f"Inserted {unique_inserts} new painting URLs from {artists_done} artists. Missing names: {missing_names}. {status}"
            )
            st.session_state["queue_urls"] = C.build_session_queue(limit=int(queue_limit), seed=int(queue_seed))
            st.session_state["queue_idx"] = 0
            st.rerun()

        if st.button("Reset artist crawl index", use_container_width=True):
            C.ingest_state_set("artist_idx", "0")
            st.warning("Reset artist crawl index to 0.")
            st.rerun()

    st.divider()
    st.subheader("Optional: enrich a random sample (titles / img URLs)")
    st.write("Fetches ONLY HTML for a sample of painting pages to improve metadata. Still no image downloads server-side.")
    enrich_n = st.slider("Enrich N paintings now", 0, 200, 20, 5)
    if st.button("Enrich sample", use_container_width=True) and enrich_n > 0:
        pool = C.get_pool(limit=max(500, enrich_n * 10))
        rng = random.Random(int(queue_seed) ^ (C.now_ts() // 10))
        rng.shuffle(pool)
        sample = pool[:enrich_n]
        prog = st.progress(0)
        ok = 0
        for i, p in enumerate(sample):
            if C.ingest_painting_page_full(p["url"], user_agent, float(timeout), float(crawl_delay)):
                ok += 1
            prog.progress(int(100 * (i + 1) / max(1, len(sample))))
        st.success(f"Enriched {ok}/{len(sample)} paintings.")
        st.rerun()

# ----------------------------
# Leaderboards tab
# ----------------------------
with tabs[2]:
    st.subheader("Leaderboards (Live)")
    st.caption("Sorting: **lowest numbers are best**.")

    top_cols = st.columns(2)
    with top_cols[0]:
        top_p = st.slider("Top N paintings", 25, 1000, 200, 25)
    with top_cols[1]:
        min_pg = st.slider("Min games per painting", 0, 50, 1, 1)

    prow = C.paintings_leaderboard_live(limit=int(top_p), min_games=int(min_pg))
    st.markdown("### 🏆 Paintings (Lowest is Best)")
    st.dataframe(prow, use_container_width=True, height=520)

    st.download_button(
        "Download paintings leaderboard JSON",
        data=json.dumps(prow, indent=2).encode("utf-8"),
        file_name="artmash_paintings_leaderboard.json",
        mime="application/json",
        use_container_width=True,
    )

    st.divider()
    st.markdown("### 🎨 Artists (Canonical by artist URL, Lowest is Best)")
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        top_a = st.slider("Top N artists", 25, 1000, 200, 25)
    with a2:
        topk = st.slider("Aggregate top-k paintings", 1, 20, 5, 1)
    with a3:
        min_ag = st.slider("Min total games per artist", 0, 500, 5, 1)
    with a4:
        min_paint_games = st.slider("Min games per painting (artist agg)", 0, 50, 1, 1)

    include_unknown = st.checkbox("Include Unknown artist bucket", value=False)

    arow = C.artists_leaderboard_live(
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
    )

st.caption("Tip: If images do not show in img-tag mode (hotlink restrictions), use iframe mode.")
