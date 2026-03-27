"""
config.py — Central configuration for the AI Recipe Generator backend.

All tunable constants, file paths, and schema definitions live here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

INDIAN_RECIPES_CSV   = DATA_DIR / "indian_recipes.csv"
INDIAN_REGIONAL_RECIPES_CSV = DATA_DIR / "indian_regional_recipes.csv"
GLOBAL_RECIPES_CSV   = DATA_DIR / "global_recipes.csv"
OFFLINE_WORLD_RECIPES_CSV = DATA_DIR / "offline_world_recipes.csv"
BEVERAGES_CSV        = DATA_DIR / "beverages.csv"
OFFLINE_WORLD_BEVERAGES_CSV = DATA_DIR / "offline_world_beverages.csv"
NUTRITION_DB_CSV     = DATA_DIR / "nutrition_database.csv"
OFFLINE_WORLD_NUTRITION_CSV = DATA_DIR / "offline_world_nutrition.csv"

# ─────────────────────────────────────────────────────────────────────────────
# LLM Configuration
# ─────────────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY    = os.getenv("ANTHROPIC_API_KEY", os.getenv("ANTHROPIC_CLAUDE_API_KEY", ""))
LLM_PROVIDER         = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_BASE_URL      = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
LLM_MODEL            = os.getenv("LLM_MODEL", "mistral")
LLM_TEMPERATURE      = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_TEMPERATURE_HIGH = float(os.getenv("LLM_TEMPERATURE_HIGH", "1.3"))  # absurd combos
LLM_MAX_TOKENS       = int(os.getenv("LLM_MAX_TOKENS", "4096"))
TRANSLATION_MODEL    = os.getenv("TRANSLATION_MODEL", LLM_MODEL)

# ─────────────────────────────────────────────────────────────────────────────
# Retrieval Tuning
# ─────────────────────────────────────────────────────────────────────────────
INGREDIENT_MATCH_THRESHOLD  = 60   # Minimum fuzz ratio to count an ingredient hit
MIN_INGREDIENT_OVERLAP      = 0.3  # At least 30 % of user ingredients must match
TOP_K_RETRIEVAL             = 5    # Number of candidate recipes to return to LLM

# ─────────────────────────────────────────────────────────────────────────────
# Nutrition Matching
# ─────────────────────────────────────────────────────────────────────────────
NUTRITION_FUZZ_THRESHOLD    = 70   # Minimum score for ingredient→nutrition mapping

# ─────────────────────────────────────────────────────────────────────────────
# Supported Enumerations
# ─────────────────────────────────────────────────────────────────────────────
VALID_SKILL_LEVELS = {"Beginner", "Intermediate", "Pro"}

VALID_EQUIPMENT = {
    "Stove", "Oven", "Microwave", "Air Fryer", "Blender",
    "Food Processor", "Grill", "Pressure Cooker", "Instant Pot",
    "Espresso Machine", "Sous Vide", "Wok", "Slow Cooker",
    "Stand Mixer", "Toaster", "Smoker",
}

VALID_CUISINES = {
    "Indian", "Italian", "Mexican", "Chinese", "Japanese",
    "Thai", "French", "Mediterranean", "American", "Korean",
    "Middle Eastern", "Ethiopian", "Global",
}

VALID_MEAL_CATEGORIES = {
    "Breakfast", "Lunch", "Dinner", "Snack", "Dessert",
    "Cocktail", "Mocktail", "Coffee", "Tea", "Smoothie",
    "Appetizer", "Side Dish",
}

# ─────────────────────────────────────────────────────────────────────────────
# LLM Output JSON Schema (enforced in the system prompt)
# ─────────────────────────────────────────────────────────────────────────────
RECIPE_JSON_SCHEMA = {
    "recipe_name": "string",
    "cuisine_type": "string",
    "estimated_time_minutes": "integer",
    "equipment_used": ["string"],
    "servings": "integer",
    "ingredients": [
        {
            "name": "string (English only)",
            "quantity_grams": "number",
            "original_measure": "string",
            "preparation_note": "string",
        }
    ],
    "step_by_step_instructions": ["string"],
    "beverage_pairing": {
        "name": "string",
        "type": "string",
        "ingredients": [
            {
                "name": "string (English only)",
                "quantity_grams": "number",
                "original_measure": "string",
            }
        ],
        "instructions": ["string"],
    },
}
