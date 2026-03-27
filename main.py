"""
main.py — CLI entry point and demonstration harness for the CulinaryEngine.

Run with:
    python main.py                  # default Indian dinner demo
    python main.py --absurd         # absurd combos mode
    python main.py --nutrition-only # offline nutrition from JSON file

Ensure Ollama is running locally and `LLM_MODEL` is available before running.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from config import BASE_DIR
from engine import CulinaryEngine
from models import TimeConstraints, UserConstraints

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-20s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


# ─────────────────────────────────────────────────────────────────────────────
# Demo Constraint Presets
# ─────────────────────────────────────────────────────────────────────────────

DEMO_CONSTRAINTS = UserConstraints.model_validate(
    {
        "skill_level": "Intermediate",
        "available_equipment": ["Stove", "Oven", "Blender", "Pressure Cooker"],
        "available_ingredients": [
            "chicken breast", "onion", "tomato", "garlic", "ginger",
            "yogurt", "cream", "rice", "cilantro", "garam masala",
            "turmeric", "cumin seeds", "salt", "oil",
        ],
        "cuisine_preference": "Indian",
        "time_constraints": TimeConstraints(max_prep_minutes=30, max_total_minutes=60),
        "meal_category": "Dinner",
        "absurd_combos": False,
        "target_language": "en",
        "servings": 2,
    }
)

DEMO_ABSURD_CONSTRAINTS = UserConstraints.model_validate(
    {
        "skill_level": "Pro",
        "available_equipment": ["Stove", "Oven", "Blender", "Espresso Machine"],
        "available_ingredients": [
            "espresso", "salmon", "chocolate shavings", "mango",
            "soy sauce", "basil", "cream", "rice noodles", "lime",
        ],
        "cuisine_preference": "Global",
        "time_constraints": TimeConstraints(max_prep_minutes=30, max_total_minutes=60),
        "meal_category": "Dinner",
        "absurd_combos": True,
        "target_language": "en",
        "servings": 2,
    }
)


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-Print Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_section(title: str) -> None:
    print(f"\n{'═' * 70}")
    print(f"  {title}")
    print(f"{'═' * 70}")


def _print_recipe(result) -> None:
    """Print the generation result in a human-readable format."""
    recipe = result.recipe
    nutrition = result.nutrition

    _print_section(f"🍽  {recipe.recipe_name}")
    print(f"  Cuisine     : {recipe.cuisine_type}")
    print(f"  Time        : {recipe.estimated_time_minutes} min")
    print(f"  Servings    : {recipe.servings}")
    print(f"  Equipment   : {', '.join(recipe.equipment_used)}")

    _print_section("📝  Ingredients")
    for ing in recipe.ingredients:
        note = f" ({ing.preparation_note})" if ing.preparation_note else ""
        print(f"  • {ing.name}: {ing.quantity_grams}g ({ing.original_measure}){note}")

    _print_section("👨‍🍳  Instructions")
    for i, step in enumerate(recipe.step_by_step_instructions, 1):
        print(f"  {i}. {step}")

    if recipe.beverage_pairing:
        bp = recipe.beverage_pairing
        _print_section(f"🥂  Beverage Pairing: {bp.name} ({bp.type})")
        for ing in bp.ingredients:
            print(f"  • {ing.name}: {ing.quantity_grams}g ({ing.original_measure})")
        for i, step in enumerate(bp.instructions, 1):
            print(f"  {i}. {step}")

    _print_section("📊  Nutrition (per serving)")
    ps = nutrition.per_serving
    print(f"  Calories : {ps.calories_kcal} kcal")
    print(f"  Protein  : {ps.protein_g} g")
    print(f"  Fat      : {ps.fat_g} g")
    print(f"  Carbs    : {ps.carbs_g} g")
    print(f"  Fiber    : {ps.fiber_g} g")
    print(f"  Sugar    : {ps.sugar_g} g")
    print(f"  Sodium   : {ps.sodium_mg} mg")

    if nutrition.unmatched_ingredients:
        print(f"\n  ⚠ Unmatched ingredients (excluded from totals):")
        for name in nutrition.unmatched_ingredients:
            print(f"    - {name}")

    _print_section("📊  Nutrition (breakdown by ingredient)")
    for ib in nutrition.ingredient_breakdown:
        print(
            f"  {ib.ingredient_name:25s} → {ib.matched_db_name:20s} "
            f"(confidence: {ib.match_confidence}%, {ib.quantity_grams}g) "
            f"= {ib.nutrients.calories_kcal} kcal"
        )

    if result.translated_recipe:
        tr = result.translated_recipe
        _print_section(f"🌐  Translated: {tr.recipe_name}")
        for i, step in enumerate(tr.step_by_step_instructions, 1):
            print(f"  {i}. {step}")


# ─────────────────────────────────────────────────────────────────────────────
# Offline Nutrition-Only Mode
# ─────────────────────────────────────────────────────────────────────────────

def _run_nutrition_only(json_path: str) -> None:
    """Load a recipe JSON file and compute nutrition offline."""
    path = Path(json_path)
    if not path.exists():
        logger.error("File not found: %s", path)
        sys.exit(1)

    with open(path) as f:
        recipe_data = json.load(f)

    engine = CulinaryEngine()
    report = engine.calculate_nutrition_only(recipe_data)

    _print_section("📊  Offline Nutrition Report")
    print(json.dumps(report.model_dump(), indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval-Only Demo (no LLM — for testing without API key)
# ─────────────────────────────────────────────────────────────────────────────

def _run_retrieval_demo(constraints: UserConstraints) -> None:
    """Run only the retrieval pipeline to show candidate matching."""
    from retrieval import retrieve_candidate_recipes, retrieve_beverage_pairing

    _print_section("🔍  Retrieval Demo (no LLM call)")

    candidates = retrieve_candidate_recipes(constraints)
    print(f"\n  Found {len(candidates)} recipe candidate(s):\n")
    for i, c in enumerate(candidates, 1):
        name = c.get("recipe_name", c.get("beverage_name", "Unknown"))
        score = c.get("_overlap_score", "N/A")
        print(f"  {i}. {name} (overlap: {score})")
        print(f"     Ingredients: {c.get('ingredients', '')[:100]}…")

    beverages = retrieve_beverage_pairing(constraints)
    print(f"\n  Found {len(beverages)} beverage candidate(s):\n")
    for i, b in enumerate(beverages, 1):
        name = b.get("beverage_name", "Unknown")
        print(f"  {i}. {name} ({b.get('type', '')})")

    # Show what prompt would have been constructed
    from prompts import build_system_prompt, build_user_prompt
    sys_prompt = build_system_prompt(constraints, candidates)
    usr_prompt = build_user_prompt(constraints)

    _print_section("📝  System Prompt (first 800 chars)")
    print(sys_prompt[:800] + "\n…")

    _print_section("📝  User Prompt")
    print(usr_prompt)


# ─────────────────────────────────────────────────────────────────────────────
# Offline Nutrition Demo (hardcoded recipe, no LLM)
# ─────────────────────────────────────────────────────────────────────────────

def _run_nutrition_demo() -> None:
    """Demonstrate the nutritional calculator with a hardcoded recipe."""
    from models import GeneratedRecipe, IngredientItem, BeveragePairing

    sample_recipe = GeneratedRecipe(
        recipe_name="Demo Chicken Tikka Masala",
        cuisine_type="Indian",
        estimated_time_minutes=50,
        equipment_used=["Stove", "Oven"],
        servings=2,
        ingredients=[
            IngredientItem(name="chicken breast", quantity_grams=500,
                           original_measure="500g", preparation_note="cubed"),
            IngredientItem(name="yogurt", quantity_grams=100,
                           original_measure="100g"),
            IngredientItem(name="onion", quantity_grams=150,
                           original_measure="1 large"),
            IngredientItem(name="tomato puree", quantity_grams=200,
                           original_measure="200ml"),
            IngredientItem(name="cream", quantity_grams=100,
                           original_measure="100ml"),
            IngredientItem(name="garlic", quantity_grams=10,
                           original_measure="3 cloves"),
            IngredientItem(name="ginger", quantity_grams=10,
                           original_measure="1 inch piece"),
            IngredientItem(name="rice", quantity_grams=300,
                           original_measure="300g"),
            IngredientItem(name="garam masala", quantity_grams=5,
                           original_measure="1 tsp"),
            IngredientItem(name="turmeric", quantity_grams=3,
                           original_measure="1/2 tsp"),
            IngredientItem(name="salt", quantity_grams=5,
                           original_measure="1 tsp"),
            IngredientItem(name="oil", quantity_grams=30,
                           original_measure="2 tbsp"),
        ],
        step_by_step_instructions=[
            "Marinate chicken in yogurt and spices for 30 minutes.",
            "Grill chicken in oven at 200°C for 15 minutes.",
            "Sauté onion, garlic, and ginger in oil until golden.",
            "Add tomato puree and simmer for 10 minutes.",
            "Add cream and grilled chicken, mix well.",
            "Serve over steamed rice, garnished with cilantro.",
        ],
        beverage_pairing=BeveragePairing(
            name="Mango Lassi",
            type="Mocktail",
            ingredients=[
                IngredientItem(name="mango", quantity_grams=200,
                               original_measure="1 cup"),
                IngredientItem(name="yogurt", quantity_grams=150,
                               original_measure="150g"),
                IngredientItem(name="milk", quantity_grams=100,
                               original_measure="100ml"),
                IngredientItem(name="sugar", quantity_grams=20,
                               original_measure="1 tbsp"),
            ],
            instructions=[
                "Blend mango, yogurt, milk, and sugar until smooth.",
                "Pour into a glass with ice.",
                "Garnish with cardamom powder.",
            ],
        ),
    )

    from nutrition import calculate_recipe_nutrition
    from engine import GenerationResult

    nutrition = calculate_recipe_nutrition(sample_recipe)

    result = GenerationResult(
        recipe=sample_recipe,
        nutrition=nutrition,
    )
    _print_recipe(result)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI Culinary Engine — Recipe & Beverage Generator",
    )
    parser.add_argument(
        "--absurd", action="store_true",
        help="Enable absurd/avant-garde fusion mode",
    )
    parser.add_argument(
        "--nutrition-only", type=str, default=None,
        help="Path to a recipe JSON file — compute nutrition offline (no LLM)",
    )
    parser.add_argument(
        "--retrieval-demo", action="store_true",
        help="Run retrieval pipeline only (no LLM call required)",
    )
    parser.add_argument(
        "--nutrition-demo", action="store_true",
        help="Run offline nutrition demo with hardcoded recipe (no LLM)",
    )
    parser.add_argument(
        "--language", type=str, default="en",
        help="Target language ISO code (e.g. 'hi', 'es', 'fr')",
    )

    args = parser.parse_args()

    # ── Nutrition-only mode ──────────────────────────────────────────────
    if args.nutrition_only:
        _run_nutrition_only(args.nutrition_only)
        return

    # ── Retrieval demo (no LLM needed) ──────────────────────────────────
    if args.retrieval_demo:
        constraints = DEMO_ABSURD_CONSTRAINTS if args.absurd else DEMO_CONSTRAINTS
        _run_retrieval_demo(constraints)
        return

    # ── Nutrition demo (no LLM needed) ──────────────────────────────────
    if args.nutrition_demo:
        _run_nutrition_demo()
        return

    # ── Full generation pipeline (requires API key) ─────────────────────
    constraints = DEMO_ABSURD_CONSTRAINTS if args.absurd else DEMO_CONSTRAINTS
    constraints.target_language = args.language

    engine = CulinaryEngine()

    try:
        result = engine.generate(constraints)
        _print_recipe(result)

        # Save full result to JSON for inspection
        output_path = BASE_DIR / "last_result.json"
        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info("Full result saved to %s", output_path)

    except Exception as exc:
        logger.exception("Generation failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
