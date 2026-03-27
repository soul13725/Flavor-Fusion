"""
nutrition.py — Offline Nutritional Calculation Engine.

Parses the LLM-generated recipe JSON, fuzzy-matches each ingredient to a
local nutrition_database.csv, normalises units to grams, and returns a
complete per-serving macronutrient profile.

Key design decisions:
  • All matching is done in English (enforced by the LLM prompt).
  • Volumetric → gram conversion uses a built-in lookup table.
  • Fuzzy matching uses thefuzz (Levenshtein-based) with configurable threshold.
  • Unmatched ingredients are reported but do NOT block the pipeline.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
from thefuzz import fuzz, process as fuzz_process

from config import (
    NUTRITION_DB_CSV,
    NUTRITION_FUZZ_THRESHOLD,
    OFFLINE_WORLD_NUTRITION_CSV,
)
from models import (
    GeneratedRecipe,
    IngredientItem,
    IngredientNutrition,
    NutrientProfile,
    RecipeNutritionReport,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Volumetric → Gram Conversion Table
# ─────────────────────────────────────────────────────────────────────────────
# Approximate densities for common culinary measures (ml → g is ~1:1 for water-
# based liquids; denser/lighter items get specific factors).

UNIT_TO_GRAMS: Dict[str, float] = {
    # Volume units → ml equivalent first, then density-adjusted
    "ml": 1.0,
    "l": 1000.0,
    "liter": 1000.0,
    "litre": 1000.0,
    "cup": 240.0,
    "cups": 240.0,
    "tbsp": 15.0,
    "tablespoon": 15.0,
    "tablespoons": 15.0,
    "tsp": 5.0,
    "teaspoon": 5.0,
    "teaspoons": 5.0,
    "fl oz": 30.0,
    "fluid ounce": 30.0,
    "fluid ounces": 30.0,
    # Weight units → gram
    "g": 1.0,
    "gram": 1.0,
    "grams": 1.0,
    "kg": 1000.0,
    "kilogram": 1000.0,
    "oz": 28.35,
    "ounce": 28.35,
    "ounces": 28.35,
    "lb": 453.6,
    "lbs": 453.6,
    "pound": 453.6,
    "pounds": 453.6,
    # Discrete units (mapped to a sensible default gram weight)
    "pinch": 0.5,
    "dash": 0.5,
    "piece": 100.0,   # overridden by unit_weight_grams in nutrition DB
    "pieces": 100.0,
    "slice": 30.0,
    "slices": 30.0,
    "clove": 3.0,      # garlic clove
    "cloves": 3.0,
    "bunch": 30.0,
    "sprig": 2.0,
    "sprigs": 2.0,
    "sheet": 3.0,      # nori sheet
    "sheets": 3.0,
    "whole": 100.0,
    "can": 400.0,
    "jar": 350.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# Nutrition Database Loader
# ─────────────────────────────────────────────────────────────────────────────

_NUTRITION_DF: Optional[pd.DataFrame] = None


def _load_nutrition_db() -> pd.DataFrame:
    """Load and cache the nutrition database CSV."""
    global _NUTRITION_DF
    if _NUTRITION_DF is not None:
        return _NUTRITION_DF
    try:
        frames: List[pd.DataFrame] = []
        for path in (NUTRITION_DB_CSV, OFFLINE_WORLD_NUTRITION_CSV):
            try:
                frames.append(pd.read_csv(path))
            except FileNotFoundError:
                logger.warning("Nutrition dataset not found: %s", path)
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        df["food_name_lower"] = df["food_name"].str.strip().str.lower()
        # Ensure numeric columns
        for col in (
            "calories_per_100g", "protein_per_100g", "fat_per_100g",
            "carbs_per_100g", "fiber_per_100g", "sugar_per_100g",
            "sodium_per_100g", "unit_weight_grams",
        ):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        _NUTRITION_DF = df
        logger.info("Nutrition DB loaded: %d entries", len(df))
        return df
    except Exception as exc:
        logger.error("Cannot load nutrition DB: %s", exc)
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Fuzzy Matching
# ─────────────────────────────────────────────────────────────────────────────

def _clean_ingredient_name(raw: str) -> str:
    """
    Strip preparation notes, quantities, and common filler words to
    leave only the core food name for matching.
    """
    # Remove parenthetical notes
    cleaned = re.sub(r"\(.*?\)", "", raw)
    # Remove leading numbers & units
    cleaned = re.sub(r"^\d+[\./]?\d*\s*", "", cleaned)
    # Remove common prep words
    for word in (
        "fresh", "dried", "ground", "chopped", "minced", "sliced",
        "diced", "crushed", "grated", "frozen", "canned", "packed",
        "loosely", "tightly", "large", "small", "medium", "whole",
        "boneless", "skinless", "raw", "cooked",
    ):
        cleaned = re.sub(rf"\b{word}\b", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip().lower()


def _fuzzy_match_ingredient(
    name: str,
    db: pd.DataFrame,
    threshold: int = NUTRITION_FUZZ_THRESHOLD,
) -> Tuple[Optional[pd.Series], int]:
    """
    Find the best match in the nutrition DB for a given ingredient name.

    Returns:
        (matched_row | None, match_score)
    """
    cleaned = _clean_ingredient_name(name)
    if not cleaned:
        return None, 0

    choices = db["food_name_lower"].tolist()

    # First try exact match (fastest path)
    if cleaned in choices:
        row = db[db["food_name_lower"] == cleaned].iloc[0]
        return row, 100

    # Fuzzy match via token_set_ratio (handles word order & partial matches)
    result = fuzz_process.extractOne(
        cleaned,
        choices,
        scorer=fuzz.token_set_ratio,
        score_cutoff=threshold,
    )

    if result is None:
        logger.debug("No match for '%s' (cleaned: '%s')", name, cleaned)
        return None, 0

    # thefuzz/rapidfuzz can return either (match, score) or
    # (match, score, index) depending on installed versions.
    if len(result) == 2:
        matched_name, score = result
    else:
        matched_name, score = result[0], result[1]
    row = db[db["food_name_lower"] == matched_name].iloc[0]
    return row, score


# ─────────────────────────────────────────────────────────────────────────────
# Unit Normalisation
# ─────────────────────────────────────────────────────────────────────────────

def _parse_original_measure(measure: str) -> Tuple[float, str]:
    """
    Parse a human-readable measure string like '200ml', '2 cups', '1/2 tsp'
    into (numeric_value, unit_string).
    """
    if not measure:
        return 0.0, ""

    measure = measure.strip().lower()

    # Match patterns like "200ml", "2 cups", "1/2 tsp", "0.5 liter"
    pattern = r"^(\d+[\./]?\d*)\s*(.*)$"
    m = re.match(pattern, measure)
    if not m:
        return 0.0, measure

    num_str, unit = m.group(1), m.group(2).strip()

    # Handle fractions like "1/2"
    if "/" in num_str:
        parts = num_str.split("/")
        try:
            value = float(parts[0]) / float(parts[1])
        except (ValueError, ZeroDivisionError):
            value = 0.0
    else:
        try:
            value = float(num_str)
        except ValueError:
            value = 0.0

    return value, unit


def normalise_to_grams(
    ingredient: IngredientItem,
    db_row: Optional[pd.Series] = None,
) -> float:
    """
    Convert an ingredient's quantity to grams.

    Priority:
      1. If quantity_grams is already provided and > 0, use it directly.
      2. Parse original_measure for unit-based conversion.
      3. Fall back to the DB entry's unit_weight_grams.
    """
    # LLM should already provide gram weights — fast path
    if ingredient.quantity_grams > 0:
        return ingredient.quantity_grams

    value, unit = _parse_original_measure(ingredient.original_measure)
    if value <= 0:
        # Last resort: use DB unit weight
        if db_row is not None and "unit_weight_grams" in db_row.index:
            return float(db_row["unit_weight_grams"])
        return 100.0  # absolute fallback

    if unit in UNIT_TO_GRAMS:
        return value * UNIT_TO_GRAMS[unit]

    # Try partial match on unit string (e.g. "tablespoons" in keys)
    for key, factor in UNIT_TO_GRAMS.items():
        if key in unit or unit in key:
            return value * factor

    # Unknown unit — check if DB has unit_weight_grams, multiply by count
    if db_row is not None and "unit_weight_grams" in db_row.index:
        return value * float(db_row["unit_weight_grams"])

    return value  # assume grams


# ─────────────────────────────────────────────────────────────────────────────
# Core Nutritional Calculation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_macros(grams: float, db_row: pd.Series) -> NutrientProfile:
    """Scale per-100g macro values to the actual gram weight."""
    factor = grams / 100.0
    return NutrientProfile(
        calories_kcal=round(db_row["calories_per_100g"] * factor, 1),
        protein_g=round(db_row["protein_per_100g"] * factor, 1),
        fat_g=round(db_row["fat_per_100g"] * factor, 1),
        carbs_g=round(db_row["carbs_per_100g"] * factor, 1),
        fiber_g=round(db_row.get("fiber_per_100g", 0) * factor, 1),
        sugar_g=round(db_row.get("sugar_per_100g", 0) * factor, 1),
        sodium_mg=round(db_row.get("sodium_per_100g", 0) * factor, 1),
    )


def _aggregate_profiles(profiles: List[NutrientProfile]) -> NutrientProfile:
    """Sum a list of NutrientProfiles into one total."""
    total = NutrientProfile()
    for p in profiles:
        total.calories_kcal += p.calories_kcal
        total.protein_g += p.protein_g
        total.fat_g += p.fat_g
        total.carbs_g += p.carbs_g
        total.fiber_g += p.fiber_g
        total.sugar_g += p.sugar_g
        total.sodium_mg += p.sodium_mg
    # Round totals
    for field in total.model_fields:
        setattr(total, field, round(getattr(total, field), 1))
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def calculate_recipe_nutrition(
    recipe: GeneratedRecipe,
) -> RecipeNutritionReport:
    """
    Main entry point: compute full nutritional breakdown for a GeneratedRecipe.

    Steps:
        1. Load the nutrition database.
        2. For each ingredient in the recipe (plus beverage pairing):
           a. Fuzzy-match to the DB.
           b. Normalise quantity to grams.
           c. Calculate scaled macros.
        3. Aggregate totals and divide by servings for per-serving values.
    """
    db = _load_nutrition_db()
    breakdown: List[IngredientNutrition] = []
    unmatched: List[str] = []
    all_profiles: List[NutrientProfile] = []

    # Combine recipe ingredients + beverage ingredients
    all_ingredients: List[IngredientItem] = list(recipe.ingredients)
    if recipe.beverage_pairing:
        all_ingredients.extend(recipe.beverage_pairing.ingredients)

    for ing in all_ingredients:
        matched_row, score = _fuzzy_match_ingredient(ing.name, db)

        if matched_row is None:
            unmatched.append(ing.name)
            logger.warning(
                "Unmatched ingredient: '%s' — excluded from nutrition totals",
                ing.name,
            )
            continue

        grams = normalise_to_grams(ing, matched_row)
        macros = _compute_macros(grams, matched_row)
        all_profiles.append(macros)

        breakdown.append(
            IngredientNutrition(
                ingredient_name=ing.name,
                matched_db_name=str(matched_row["food_name"]),
                match_confidence=score,
                quantity_grams=round(grams, 1),
                nutrients=macros,
            )
        )

    total = _aggregate_profiles(all_profiles)
    servings = max(recipe.servings, 1)
    per_serving = NutrientProfile(
        calories_kcal=round(total.calories_kcal / servings, 1),
        protein_g=round(total.protein_g / servings, 1),
        fat_g=round(total.fat_g / servings, 1),
        carbs_g=round(total.carbs_g / servings, 1),
        fiber_g=round(total.fiber_g / servings, 1),
        sugar_g=round(total.sugar_g / servings, 1),
        sodium_mg=round(total.sodium_mg / servings, 1),
    )

    return RecipeNutritionReport(
        recipe_name=recipe.recipe_name,
        servings=servings,
        per_serving=per_serving,
        total=total,
        ingredient_breakdown=breakdown,
        unmatched_ingredients=unmatched,
    )


def calculate_nutrition(recipe: GeneratedRecipe) -> RecipeNutritionReport:
    """Backward-compatible alias matching the Flavor Fusion AI spec."""
    return calculate_recipe_nutrition(recipe)
