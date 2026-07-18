from __future__ import annotations

import json
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Spotify Listening Insights", page_icon="🎧", layout="wide")
st.title("🎧 Spotify Listening Insights")
st.caption("Drag in your Spotify Extended Streaming History folder to analyse it, including genre insights.")

def find_col(columns, options):
    return next((c for c in options if c in columns), None)


@st.cache_data(show_spinner=False)
def load_genres(path="artists_and_genres.csv"):
    """Load the bundled artist-to-genre lookup table."""
    try:
        genres = pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame(columns=["artist_key", "genre"])

    required = {"artist", "genre"}
    if not required.issubset(genres.columns):
        raise ValueError("artists_and_genres.csv must contain 'artist' and 'genre' columns.")

    genres = genres[["artist", "genre"]].dropna().copy()
    genres["artist_key"] = genres["artist"].astype(str).str.strip().str.casefold()
    genres["genre"] = genres["genre"].astype(str).str.strip()
    return genres[["artist_key", "genre"]].drop_duplicates("artist_key")

@st.cache_data(show_spinner=False)
def load_history(files, threshold):
    records = []
    for name, content in files:
        try:
            data = json.loads(content.decode("utf-8-sig"))
        except Exception:
            continue
        if isinstance(data, list):
            records.extend(x for x in data if isinstance(x, dict))

    if not records:
        raise ValueError("No Spotify listening records were found.")

    df = pd.DataFrame(records)
    track = find_col(df.columns, ["master_metadata_track_name", "trackName"])
    artist = find_col(df.columns, ["master_metadata_album_artist_name", "artistName"])
    ts = find_col(df.columns, ["ts", "endTime"])
    ms = find_col(df.columns, ["ms_played", "msPlayed"])

    if not all([track, artist, ts, ms]):
        raise ValueError("These files do not look like Extended Streaming History files.")

    df = df.rename(columns={track:"track", artist:"artist", ts:"timestamp", ms:"ms_played"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["ms_played"] = pd.to_numeric(df["ms_played"], errors="coerce").fillna(0)
    df = df.dropna(subset=["track", "artist", "timestamp"]).copy()
    df["hours_played"] = df["ms_played"] / 3_600_000
    df["meaningful_play"] = df["ms_played"] >= threshold * 1000

    if "skipped" not in df:
        df["skipped"] = False
    df["skipped"] = df["skipped"].fillna(False).astype(str).str.lower().isin({"true","1","yes"})

    local = df["timestamp"].dt.tz_convert(None)
    df["year"] = local.dt.year
    df["month"] = local.dt.to_period("M")
    df["hour"] = local.dt.hour
    df["weekday"] = local.dt.day_name()
    return df.sort_values("timestamp")

def rankings(df):
    artists = (df.groupby("artist")
        .agg(streams=("track","size"), meaningful_plays=("meaningful_play","sum"),
             listening_hours=("hours_played","sum"), skipped=("skipped","sum"))
        .reset_index())
    artists["skip_rate_percent"] = artists["skipped"] / artists["streams"] * 100
    artists = artists.sort_values("listening_hours", ascending=False)

    tracks = (df.groupby(["track","artist"])
        .agg(streams=("track","size"), meaningful_plays=("meaningful_play","sum"),
             listening_hours=("hours_played","sum"), skipped=("skipped","sum"),
             first_month=("month","min"))
        .reset_index())

    last_month = df["month"].max()
    tracks["months_known"] = tracks["first_month"].map(
        lambda p: (last_month.year-p.year)*12 + last_month.month-p.month + 1
    )
    tracks["raw_plays_per_month"] = tracks["meaningful_plays"] / tracks["months_known"]
    baseline = float(tracks["raw_plays_per_month"].median())
    prior_months = 3
    tracks["adjusted_plays_per_month"] = (
        tracks["meaningful_plays"] + prior_months * baseline
    ) / (tracks["months_known"] + prior_months)
    tracks["skip_rate"] = tracks["skipped"] / tracks["streams"].clip(lower=1)
    tracks["favourite_score"] = (
        tracks["adjusted_plays_per_month"] * (1 - 0.5 * tracks["skip_rate"]) * 100
    )
    tracks = tracks.sort_values("favourite_score", ascending=False)
    return artists.round(2), tracks.round(2)

def csv_zip(tables):
    memory = BytesIO()
    with ZipFile(memory, "w", ZIP_DEFLATED) as z:
        for name, frame in tables.items():
            z.writestr(name, frame.to_csv(index=False))
    return memory.getvalue()

with st.sidebar:
    threshold = st.slider("Meaningful-play threshold (seconds)", 5, 120, 30, 5)
    top_n = st.slider("Number of results shown", 5, 50, 20, 5)

uploads = st.file_uploader(
    "Drop your Extended Streaming History folder here",
    type=["json"],
    accept_multiple_files="directory",
)

if not uploads:
    st.info("Upload the folder containing your Streaming_History_Audio JSON files.")
    st.stop()

payload = tuple((f.name, f.getvalue()) for f in uploads)

try:
    with st.spinner("Analysing your listening history..."):
        df = load_history(payload, threshold)

        genre_lookup = load_genres()
        df["artist_key"] = df["artist"].astype(str).str.strip().str.casefold()
        df = df.merge(genre_lookup, on="artist_key", how="left")
        df["genre"] = df["genre"].fillna("Unknown")
        df = df.drop(columns=["artist_key"])

        artists, tracks = rankings(df)
        artist_genres = (
            df[["artist", "genre"]]
            .drop_duplicates("artist")
            .set_index("artist")["genre"]
        )
        artists["genre"] = artists["artist"].map(artist_genres).fillna("Unknown")
        tracks["genre"] = tracks["artist"].map(artist_genres).fillna("Unknown")
except ValueError as exc:
    st.error(str(exc))
    st.stop()

yearly = (df.groupby("year")
    .agg(listening_hours=("hours_played","sum"), streams=("track","size"),
         unique_artists=("artist","nunique"), unique_tracks=("track","nunique"))
    .reset_index().round(2))
monthly = (df.groupby("month")
    .agg(listening_hours=("hours_played","sum"), streams=("track","size"))
    .reset_index())
monthly["month"] = monthly["month"].astype(str)

cols = st.columns(4)
cols[0].metric("Listening hours", f"{df['hours_played'].sum():,.0f}")
cols[1].metric("Streams", f"{len(df):,}")
cols[2].metric("Artists", f"{df['artist'].nunique():,}")
cols[3].metric("Tracks", f"{df['track'].nunique():,}")

overview, favourites, artist_tab, track_tab, genre_tab, habits, downloads = st.tabs(
    ["Overview","Favourite songs","Artists","Tracks","Genres","Habits","Downloads"]
)

with overview:
    a,b = st.columns(2)
    with a:
        st.subheader("Listening by year")
        st.line_chart(yearly.set_index("year")[["listening_hours"]])
    with b:
        st.subheader("Monthly trend")
        st.line_chart(monthly.set_index("month")[["listening_hours"]])
    st.dataframe(yearly, use_container_width=True, hide_index=True)

with favourites:
    winner = tracks.iloc[0]
    st.subheader("Your fair favourite song")
    st.markdown(f"### {winner['track']}\n**{winner['artist']}**")
    st.caption("Adjusted for how long the song has been present in your history.")
    st.dataframe(tracks.head(top_n), use_container_width=True, hide_index=True)

with artist_tab:
    chart = artists.head(top_n).set_index("artist")[["listening_hours"]].sort_values("listening_hours")
    st.bar_chart(chart, horizontal=True)
    st.dataframe(artists.head(top_n), use_container_width=True, hide_index=True)

with track_tab:
    shown = tracks.head(top_n).copy()
    shown["song"] = shown["track"] + " — " + shown["artist"]
    st.bar_chart(shown.set_index("song")[["listening_hours"]].sort_values("listening_hours"), horizontal=True)
    st.dataframe(shown, use_container_width=True, hide_index=True)


with genre_tab:
    genre_summary = (
        df.groupby("genre")
        .agg(
            listening_hours=("hours_played", "sum"),
            streams=("track", "size"),
            unique_artists=("artist", "nunique"),
            meaningful_plays=("meaningful_play", "sum"),
        )
        .reset_index()
        .sort_values("listening_hours", ascending=False)
        .round(2)
    )

    known_plays = df["genre"].ne("Unknown").sum()
    coverage = known_plays / len(df) * 100 if len(df) else 0

    a, b, c = st.columns(3)
    a.metric("Top genre", genre_summary.iloc[0]["genre"] if not genre_summary.empty else "—")
    b.metric("Genres found", f"{df.loc[df['genre'] != 'Unknown', 'genre'].nunique():,}")
    c.metric("Genre coverage", f"{coverage:.1f}%")

    chart = (
        genre_summary[genre_summary["genre"] != "Unknown"]
        .head(top_n)
        .set_index("genre")[["listening_hours"]]
        .sort_values("listening_hours")
    )
    st.bar_chart(chart, horizontal=True)
    st.dataframe(genre_summary, use_container_width=True, hide_index=True)

with habits:
    hourly = df.groupby("hour")["hours_played"].sum().reindex(range(24), fill_value=0)
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    weekday = df.groupby("weekday")["hours_played"].sum().reindex(order, fill_value=0)
    a,b = st.columns(2)
    with a:
        st.subheader("By hour")
        st.bar_chart(hourly)
    with b:
        st.subheader("By weekday")
        st.bar_chart(weekday)

with downloads:
    archive = csv_zip({
        "yearly_summary.csv": yearly,
        "monthly_listening.csv": monthly,
        "top_artists.csv": artists,
        "fair_favourite_songs.csv": tracks,
        "genre_summary.csv": genre_summary,
    })
    st.download_button(
        "Download all CSV reports",
        archive,
        "spotify_listening_report.zip",
        "application/zip",
        use_container_width=True,
    )
    st.caption("Genre data is matched from the bundled artists_and_genres.csv lookup file.")
