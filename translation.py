"""
translation.py - Multilingual display-string translation.

All core LLM reasoning and ingredient JSON is produced in English for the
nutrition matcher. This module translates only user-facing strings (title and
instruction text) and keeps ingredient names unchanged.
"""

from __future__ import annotations

import asyncio
import json
import logging

import aiohttp

from config import LLM_MAX_TOKENS, OLLAMA_BASE_URL, TRANSLATION_MODEL
from models import GeneratedRecipe

logger = logging.getLogger(__name__)

_OLLAMA_TRANSLATION_URL = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"

LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "ar": "Arabic",
    "ru": "Russian",
    "ta": "Tamil",
    "te": "Telugu",
    "bn": "Bengali",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "ur": "Urdu",
    "th": "Thai",
    "vi": "Vietnamese",
    "tr": "Turkish",
    "nl": "Dutch",
    "pl": "Polish",
    "sv": "Swedish",
}

_TRANSLATION_SYSTEM_PROMPT = """\
You are a professional culinary translator. Translate the following JSON fields
into {target_language}.

RULES:
1. Translate ONLY the values - never change JSON keys.
2. Do NOT translate ingredient names - leave them in English.
3. Translate: recipe_name, step_by_step_instructions, beverage_pairing.name,
   beverage_pairing.instructions.
4. Keep all numeric values and units unchanged.
5. Respond with ONLY the translated JSON object - no markdown, no commentary.
"""


def _extract_ollama_text(payload: dict) -> str:
    response = payload.get("response", "")
    if isinstance(response, str):
        return response.strip()
    raise ValueError("Unexpected Ollama translation response format")


async def translate_display_strings(
    recipe: GeneratedRecipe,
    target_language: str,
) -> GeneratedRecipe:
    """Translate display-only fields while preserving nutrition-safe ingredient names."""
    if target_language.lower() in ("en", "eng", "english"):
        return recipe

    lang_name = LANGUAGE_NAMES.get(target_language.lower(), target_language)

    translatable = {
        "recipe_name": recipe.recipe_name,
        "step_by_step_instructions": recipe.step_by_step_instructions,
    }
    if recipe.beverage_pairing:
        translatable["beverage_pairing"] = {
            "name": recipe.beverage_pairing.name,
            "instructions": recipe.beverage_pairing.instructions,
        }

    system_msg = _TRANSLATION_SYSTEM_PROMPT.format(target_language=lang_name)
    prompt = (
        f"{system_msg}\n\n"
        f"Translate this JSON object:\n"
        f"{json.dumps(translatable, ensure_ascii=False)}"
    )
    payload = {
        "model": TRANSLATION_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": LLM_MAX_TOKENS,
        },
    }

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
            async with session.post(_OLLAMA_TRANSLATION_URL, json=payload) as response:
                response.raise_for_status()
                raw = _extract_ollama_text(await response.json())

        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        translated = json.loads(raw)

        recipe_dict = recipe.model_dump()
        recipe_dict["recipe_name"] = translated.get("recipe_name", recipe.recipe_name)
        recipe_dict["step_by_step_instructions"] = translated.get(
            "step_by_step_instructions", recipe.step_by_step_instructions
        )

        if recipe.beverage_pairing and "beverage_pairing" in translated:
            bp = translated["beverage_pairing"]
            recipe_dict["beverage_pairing"]["name"] = bp.get(
                "name", recipe.beverage_pairing.name
            )
            recipe_dict["beverage_pairing"]["instructions"] = bp.get(
                "instructions", recipe.beverage_pairing.instructions
            )

        return GeneratedRecipe.model_validate(recipe_dict)
    except json.JSONDecodeError as exc:
        logger.error("Translation JSON parse failed: %s", exc)
        return recipe
    except Exception as exc:
        logger.error("Translation call failed: %s", exc)
        return recipe


def translate_display_strings_sync(
    recipe: GeneratedRecipe,
    target_language: str,
) -> GeneratedRecipe:
    """Blocking wrapper around async translation function."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(
                asyncio.run,
                translate_display_strings(recipe, target_language),
            ).result()

    return asyncio.run(translate_display_strings(recipe, target_language))
