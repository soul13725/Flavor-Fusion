"""
retrieval.py — Pandas-based Hybrid RAG retrieval engine.

Loads local CSV datasets, applies hard constraint filters (time, equipment,
skill level, cuisine, meal category), and performs fuzzy string matching
against the user's available ingredients to surface the best candidate
recipes and beverages.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd
from thefuzz import fuzz

from config import (
    BEVERAGES_CSV,
    GLOBAL_RECIPES_CSV,
    INDIAN_RECIPES_CSV,
    INDIAN_REGIONAL_RECIPES_CSV,
    INGREDIENT_MATCH_THRESHOLD,
    MIN_INGREDIENT_OVERLAP,
    OFFLINE_WORLD_BEVERAGES_CSV,
    OFFLINE_WORLD_RECIPES_CSV,
    TOP_K_RETRIEVAL,
)
from models import UserConstraints

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Loader (singleton-style cache)
# ─────────────────────────────────────────────────────────────────────────────

_DATASET_CACHE: Dict[str, pd.DataFrame] = {}


def _load_csv(path: str, label: str) -> pd.DataFrame:
    """Load a CSV into a DataFrame with caching and basic validation."""
    if label in _DATASET_CACHE:
        return _DATASET_CACHE[label]
    try:
        df = pd.read_csv(path, dtype=str).fillna("")
        logger.info("Loaded %s: %d rows from %s", label, len(df), path)
        _DATASET_CACHE[label] = df
        return df
    except FileNotFoundError:
        logger.warning("Dataset not found: %s — returning empty DataFrame", path)
        return pd.DataFrame()
    except Exception as exc:
        logger.error("Failed to load %s: %s", path, exc)
        return pd.DataFrame()


def load_all_recipes() -> pd.DataFrame:
    """Concatenate Indian + Global recipe CSVs into a unified DataFrame."""
    indian = _load_csv(str(INDIAN_RECIPES_CSV), "indian_recipes")
    if not indian.empty:
        indian = indian.copy()
        indian["_source_dataset"] = "indian_recipes"
    indian_regional = _load_csv(
        str(INDIAN_REGIONAL_RECIPES_CSV),
        "indian_regional_recipes",
    )
    if not indian_regional.empty:
        indian_regional = indian_regional.copy()
        indian_regional["_source_dataset"] = "indian_regional_recipes"
    global_ = _load_csv(str(GLOBAL_RECIPES_CSV), "global_recipes")
    if not global_.empty:
        global_ = global_.copy()
        global_["_source_dataset"] = "global_recipes"
    offline_world = _load_csv(
        str(OFFLINE_WORLD_RECIPES_CSV),
        "offline_world_recipes",
    )
    if not offline_world.empty:
        offline_world = offline_world.copy()
        offline_world["_source_dataset"] = "offline_world_recipes"
    combined = pd.concat(
        [indian, indian_regional, global_, offline_world],
        ignore_index=True,
    )
    # Normalise numeric columns
    for col in ("prep_time_min", "total_time_min"):
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce").fillna(9999)
    return combined


def load_beverages() -> pd.DataFrame:
    """Load the beverages CSV."""
    base_df = _load_csv(str(BEVERAGES_CSV), "beverages")
    offline_df = _load_csv(
        str(OFFLINE_WORLD_BEVERAGES_CSV),
        "offline_world_beverages",
    )
    df = pd.concat([base_df, offline_df], ignore_index=True)
    for col in ("prep_time_min", "total_time_min"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(9999)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Hard Constraint Filters
# ─────────────────────────────────────────────────────────────────────────────

def _filter_time(df: pd.DataFrame, max_prep: int, max_total: int) -> pd.DataFrame:
    """Remove rows that exceed the user's time budget."""
    if "prep_time_min" not in df.columns or "total_time_min" not in df.columns:
        return df
    mask = (df["prep_time_min"] <= max_prep) & (df["total_time_min"] <= max_total)
    return df[mask]


def _filter_equipment(df: pd.DataFrame, user_equipment: List[str]) -> pd.DataFrame:
    """
    Keep rows whose required equipment is a subset of what the user has.
    If the user list is empty, skip this filter (assume unlimited equipment).
    """
    if not user_equipment:
        return df
    if "equipment" not in df.columns:
        return df
    user_set = {e.strip().lower() for e in user_equipment}

    def _equipment_ok(row_equip: str) -> bool:
        if not row_equip.strip():
            return True  # recipe needs no special equipment
        required = {e.strip().lower() for e in row_equip.split(",")}
        return required.issubset(user_set)

    return df[df["equipment"].apply(_equipment_ok)]


def _filter_skill(df: pd.DataFrame, user_skill: str) -> pd.DataFrame:
    """Remove recipes above the user's skill level."""
    hierarchy = {"Beginner": 1, "Intermediate": 2, "Pro": 3}
    user_rank = hierarchy.get(user_skill, 3)
    if "skill_level" not in df.columns:
        logger.warning("Dataset missing skill_level column; skipping skill filter")
        return df
    return df[df["skill_level"].map(lambda s: hierarchy.get(s, 3)) <= user_rank]


def _filter_cuisine(df: pd.DataFrame, cuisine: str) -> pd.DataFrame:
    """Filter by cuisine preference.  'Global' disables this filter."""
    if cuisine.lower() == "global":
        return df
    if cuisine.lower() == "indian food":
        if "_source_dataset" in df.columns:
            return df[
                df["_source_dataset"].isin(
                    ["indian_recipes", "indian_regional_recipes"]
                )
            ]
        return df[df["cuisine"].str.lower().str.contains("indian", na=False)]
    if "cuisine" not in df.columns:
        return df
    return df[df["cuisine"].str.lower() == cuisine.lower()]


def _filter_meal_category(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """Filter by meal_category column (falls back to 'category' for beverages)."""
    if "meal_category" in df.columns:
        return df[df["meal_category"].str.lower() == category.lower()]
    if "category" in df.columns:
        return df[df["category"].str.lower() == category.lower()]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Fuzzy Ingredient Matching & Scoring
# ─────────────────────────────────────────────────────────────────────────────

def _parse_ingredient_names(raw: str) -> List[str]:
    """
    Parse the ingredients cell (format: 'name:qty,name:qty,...')
    and return only the lowercased names.
    """
    names = []
    for part in raw.split(","):
        part = part.strip()
        if ":" in part:
            names.append(part.split(":")[0].strip().lower())
        elif part:
            names.append(part.lower())
    return names


def _ingredient_overlap_score(
    recipe_ingredients: List[str],
    user_ingredients: List[str],
) -> Tuple[float, int, int]:
    """
    Return (overlap_ratio, total_matched_count, direct_match_count).
    Uses fuzzy token_set_ratio to handle plurals, abbreviations, etc.
    """
    if not user_ingredients or not recipe_ingredients:
        return 0.0, 0, 0

    user_lower = [u.lower() for u in user_ingredients]
    user_set = {u.strip().lower() for u in user_ingredients}
    matched = 0
    direct = 0
    for r_ing in recipe_ingredients:
        if r_ing in user_set:
            direct += 1
        best = max(fuzz.token_set_ratio(r_ing, u) for u in user_lower)
        if best >= INGREDIENT_MATCH_THRESHOLD:
            matched += 1

    overlap = matched / len(recipe_ingredients) if recipe_ingredients else 0.0
    return overlap, matched, direct


def _score_and_rank(
    df: pd.DataFrame,
    user_ingredients: List[str],
    top_k: int = TOP_K_RETRIEVAL,
) -> pd.DataFrame:
    """
    Score each recipe by ingredient overlap with the user's pantry.
    Returns the top-k highest scored rows.
    """
    if df.empty or not user_ingredients:
        return df.head(top_k)

    scores: List[float] = []
    match_counts: List[int] = []
    direct_counts: List[int] = []
    for _, row in df.iterrows():
        names = _parse_ingredient_names(row.get("ingredients", ""))
        overlap, cnt, direct = _ingredient_overlap_score(names, user_ingredients)
        scores.append(overlap)
        match_counts.append(cnt)
        direct_counts.append(direct)

    df = df.copy()
    df["_overlap_score"] = scores
    df["_match_count"] = match_counts
    df["_direct_match_count"] = direct_counts

    # Be more permissive for short ingredient queries (e.g. only "paneer")
    dynamic_min_overlap = MIN_INGREDIENT_OVERLAP if len(user_ingredients) >= 3 else 0.08

    filtered = df[
        (df["_overlap_score"] >= dynamic_min_overlap)
        | (df["_direct_match_count"] > 0)
    ]

    if filtered.empty:
        filtered = df[df["_match_count"] > 0]

    df = filtered.sort_values(
        by=["_direct_match_count", "_overlap_score", "_match_count"],
        ascending=[False, False, False],
    ).head(top_k)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Public Retrieval API
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_candidate_recipes(
    constraints: UserConstraints,
) -> List[Dict]:
    """
    Main retrieval pipeline:
      1. Load datasets
      2. Apply hard filters (time, equipment, skill, cuisine, category)
      3. Score by ingredient overlap
      4. Return top-k candidates as list of dicts

    Returns an empty list when no candidates survive filtering.
    """
    is_beverage = constraints.meal_category in {
        "Cocktail", "Mocktail", "Coffee", "Tea", "Smoothie",
    }

    if is_beverage:
        df = load_beverages()
        # Beverages use 'category' or 'type' instead of 'cuisine'
        df = _filter_meal_category(df, constraints.meal_category)
    else:
        df = load_all_recipes()
        df = _filter_cuisine(df, constraints.cuisine_preference)
        df = _filter_meal_category(df, constraints.meal_category)

    # Common hard filters
    df = _filter_time(
        df,
        constraints.time_constraints.max_prep_minutes,
        constraints.time_constraints.max_total_minutes,
    )
    df = _filter_equipment(df, constraints.available_equipment)
    df = _filter_skill(df, constraints.skill_level.value)

    logger.info("After hard filters: %d candidates remain", len(df))

    # Soft scoring by ingredient overlap
    df = _score_and_rank(df, constraints.available_ingredients)

    candidates = df.drop(
        columns=["_overlap_score", "_match_count", "_direct_match_count"],
        errors="ignore",
    ).to_dict(orient="records")

    logger.info("Returning %d candidate recipes/beverages", len(candidates))
    return candidates


def retrieve_beverage_pairing(
    constraints: UserConstraints,
) -> List[Dict]:
    """
    Retrieve candidate beverage pairings that complement the main recipe.
    Applies the same equipment / time / ingredient filters.
    """
    df = load_beverages()

    df = _filter_time(
        df,
        constraints.time_constraints.max_prep_minutes,
        constraints.time_constraints.max_total_minutes,
    )
    df = _filter_equipment(df, constraints.available_equipment)

    # Soft scoring
    df = _score_and_rank(
        df, constraints.available_ingredients, top_k=3,
    )

    return df.drop(
        columns=["_overlap_score", "_match_count", "_direct_match_count"],
        errors="ignore",
    ).to_dict(orient="records")
