"""
Movie Recommender API
=====================
Loads pre-trained movie embeddings, exposes endpoints to:
  - search for a movie by (approximate) title  (GET /search)
  - get k nearest-neighbour recommendations     (GET /recommend)

Fuzzy matching handles the fact that titles in the dataset include the year,
e.g. "Toy Story (1995)", while users will typically type just "toy story".

If the environment variable TMDB_API_KEY is set, the /recommend endpoint
enriches each result with overview, cast, poster, and trailer from TMDB.

Run:
    uvicorn userModelApi:app --reload
"""

from __future__ import annotations

import os
import re
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from rapidfuzz import fuzz, process
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Optional: TMDB enrichment (set TMDB_API_KEY env var to enable)
# ---------------------------------------------------------------------------
TMDB_API_KEY: str = os.getenv("TMDB_API_KEY", "")
TMDB_BASE_URL = "https://api.themoviedb.org/3"

try:
    import requests as _requests  # only needed when TMDB key is present
except ImportError:  # pragma: no cover
    _requests = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Movie Recommender API",
    description="Embedding-based movie recommendations with fuzzy title search.",
)

# ---------------------------------------------------------------------------
# Data loaded once at startup
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

movie_lookup: pd.DataFrame = pd.read_csv(os.path.join(BASE_DIR, "movie_lookup.csv"))
movie_embeddings: np.ndarray = np.load(os.path.join(BASE_DIR, "movie_embeddings.npy"))


def _clean_title(title: str) -> str:
    """Strip trailing ` (YYYY)` so fuzzy matching works on the actual name."""
    return re.sub(r"\s*\(\d{4}\)\s*$", "", title).strip()


# Pre-compute cleaned titles & a list for rapidfuzz
movie_lookup["clean_title"] = movie_lookup["title"].apply(_clean_title)
_clean_titles_lower: list[str] = movie_lookup["clean_title"].str.lower().tolist()

# Pre-compute pairwise norms once for fast cosine search
_norms = np.linalg.norm(movie_embeddings, axis=1, keepdims=True)
_norms[_norms == 0] = 1  # avoid division by zero
_normed_embeddings = movie_embeddings / _norms


# ---------------------------------------------------------------------------
# Helper: fuzzy title search
# ---------------------------------------------------------------------------
def search_movie(query: str, top_n: int = 5) -> list[dict[str, Any]]:
    """Return *top_n* movies whose title best matches *query* (fuzzy)."""
    results = process.extract(
        query.lower(),
        _clean_titles_lower,
        scorer=fuzz.WRatio,
        limit=top_n,
    )
    matches: list[dict[str, Any]] = []
    for _match_title, score, idx in results:
        row = movie_lookup.iloc[idx]
        matches.append(
            {
                "movie_id": int(row["movie_id"]),
                "title": row["title"],
                "genres": row["genres"],
                "match_score": round(score, 1),
            }
        )
    return matches


# ---------------------------------------------------------------------------
# Helper: k-NN via cosine similarity on embeddings
# ---------------------------------------------------------------------------
def get_recommendations(movie_id: int, k: int = 4) -> list[dict[str, Any]]:
    """Return *k* nearest movies by cosine similarity of embeddings."""
    query_vec = _normed_embeddings[movie_id].reshape(1, -1)
    similarities = cosine_similarity(query_vec, _normed_embeddings)[0]

    # Exclude the query movie itself, then take top-k
    similarities[movie_id] = -1.0
    top_indices = np.argsort(similarities)[::-1][:k]

    recommendations: list[dict[str, Any]] = []
    for idx in top_indices:
        row = movie_lookup[movie_lookup["movie_id"] == idx]
        if row.empty:
            continue
        row = row.iloc[0]
        recommendations.append(
            {
                "movie_id": int(idx),
                "title": row["title"],
                "genres": row["genres"],
                "similarity": round(float(similarities[idx]), 4),
                "imdb_url": _imdb_url(row["imdbId"]),
            }
        )
    return recommendations


# ---------------------------------------------------------------------------
# Helper: IMDB / TMDB URLs & details
# ---------------------------------------------------------------------------
def _imdb_url(imdb_id) -> str:
    """Build a full IMDB link from the numeric imdbId stored in the CSV."""
    try:
        return f"https://www.imdb.com/title/tt{int(imdb_id):07d}/"
    except (ValueError, TypeError):
        return ""


def get_tmdb_details(tmdb_id: int) -> dict[str, Any] | None:
    """Fetch overview, rating, cast, poster, and trailer from TMDB."""
    if not TMDB_API_KEY or _requests is None:
        return None
    params: dict[str, str] = {"api_key": TMDB_API_KEY, "language": "en-US"}
    try:
        # --- movie details ---
        detail_resp = _requests.get(
            f"{TMDB_BASE_URL}/movie/{tmdb_id}", params=params, timeout=5
        )
        detail_resp.raise_for_status()
        details = detail_resp.json()

        # --- credits (cast) ---
        credits_resp = _requests.get(
            f"{TMDB_BASE_URL}/movie/{tmdb_id}/credits", params=params, timeout=5
        )
        credits_resp.raise_for_status()
        credits = credits_resp.json()

        # --- videos (trailer) ---
        videos_resp = _requests.get(
            f"{TMDB_BASE_URL}/movie/{tmdb_id}/videos", params=params, timeout=5
        )
        videos_resp.raise_for_status()
        videos = videos_resp.json()

        trailer_url: str | None = None
        for video in videos.get("results", []):
            if video.get("type") == "Trailer" and video.get("site") == "YouTube":
                trailer_url = f"https://www.youtube.com/watch?v={video['key']}"
                break

        poster_path = details.get("poster_path")
        return {
            "overview": details.get("overview", ""),
            "vote_average": details.get("vote_average"),
            "poster_url": (
                f"https://image.tmdb.org/t/p/w500{poster_path}"
                if poster_path
                else None
            ),
            "cast": [a["name"] for a in credits.get("cast", [])[:10]],
            "trailer_url": trailer_url,
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    """Quick liveness / readiness check."""
    return {
        "status": "ok",
        "movies_loaded": len(movie_lookup),
        "embedding_shape": list(movie_embeddings.shape),
    }


@app.get("/search")
def search(
    query: str = Query(..., description="Movie title (or part of it) to search for"),
    top_n: int = Query(5, ge=1, le=20, description="Max results to return"),
):
    """
    Fuzzy-search movies by title.

    Useful as an auto-complete / "did you mean?" step before calling /recommend.
    """
    matches = search_movie(query, top_n)
    return {"query": query, "results": matches}


@app.get("/recommend")
def recommend(
    title: str = Query(..., description="Movie title (fuzzy matching is applied)"),
    k: int = Query(4, ge=1, le=20, description="Number of recommendations"),
):
    """
    Main recommendation endpoint.

    1. Fuzzy-match *title* to find the closest movie in the database.
    2. Retrieve *k* nearest neighbours by cosine similarity of embeddings.
    3. (Optional) Enrich results with TMDB details if TMDB_API_KEY is set.
    """
    # Step 1 — resolve the movie
    matches = search_movie(title, top_n=1)
    if not matches or matches[0]["match_score"] < 50:
        raise HTTPException(
            status_code=404,
            detail=(
                "Movie not found in the database. "
                "Try /search to find the closest title."
            ),
        )

    best_match = matches[0]
    movie_id: int = best_match["movie_id"]

    # Step 2 — embedding-based recommendations
    recs = get_recommendations(movie_id, k)

    # Build source movie info
    source_row = movie_lookup[movie_lookup["movie_id"] == movie_id].iloc[0]
    source_info: dict[str, Any] = {
        "movie_id": movie_id,
        "title": best_match["title"],
        "genres": best_match["genres"],
        "imdb_url": _imdb_url(source_row["imdbId"]),
    }

    # Step 3 — optional TMDB enrichment
    if TMDB_API_KEY:
        source_tmdb = source_row.get("tmdbId")
        if pd.notna(source_tmdb):
            source_info["tmdb_details"] = get_tmdb_details(int(source_tmdb))
        for rec in recs:
            rec_row = movie_lookup[movie_lookup["movie_id"] == rec["movie_id"]]
            if not rec_row.empty:
                tmdb_val = rec_row.iloc[0].get("tmdbId")
                if pd.notna(tmdb_val):
                    rec["tmdb_details"] = get_tmdb_details(int(tmdb_val))

    return {
        "source_movie": source_info,
        "recommendations": recs,
    }
