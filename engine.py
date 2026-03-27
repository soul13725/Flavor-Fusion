"""
engine.py — CulinaryEngine: the top-level orchestrator.

This class is the single entry point for the entire backend.  It:
  1. Validates and ingests user constraints.
    2. Constructs the engineered LLM prompt.
        3. Calls the local Ollama API (with absurd-mode temperature override).
  5. Parses and validates the JSON response via Pydantic.
  6. Computes offline nutritional values.
  7. Optionally translates display strings to the target language.

Usage:
    engine = CulinaryEngine()
    result = engine.generate(UserConstraints(...))
    # or async:
    result = await engine.agenerate(UserConstraints(...))
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import aiohttp
import requests

from config import (
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_TEMPERATURE_HIGH,
    OLLAMA_BASE_URL,
)
from models import (
    GeneratedRecipe,
    RecipeNutritionReport,
    UserConstraints,
)
from nutrition import calculate_recipe_nutrition
from prompts import build_system_prompt, build_user_prompt
from translation import translate_display_strings, translate_display_strings_sync

logger = logging.getLogger(__name__)

SPICE_HINTS = {
    "salt", "pepper", "chili", "chilli", "paprika", "turmeric", "cumin",
    "coriander", "garam masala", "masala", "oregano", "thyme", "rosemary",
    "basil", "cardamom", "clove", "cinnamon", "nutmeg", "fenugreek",
    "mustard", "asafoetida", "hing", "saffron", "bay leaf", "ginger",
    "garlic powder", "onion powder",
}


# ─────────────────────────────────────────────────────────────────────────────
# Result Container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GenerationResult:
    """Bundles all outputs from a single generation run."""
    recipe: GeneratedRecipe
    nutrition: RecipeNutritionReport
    translated_recipe: Optional[GeneratedRecipe] = None
    candidates_used: List[Dict] = field(default_factory=list)
    raw_llm_response: str = ""
    system_prompt: str = ""
    user_prompt: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the entire result to a JSON-safe dictionary."""
        return {
            "recipe": self.recipe.model_dump(),
            "nutrition": self.nutrition.model_dump(),
            "translated_recipe": (
                self.translated_recipe.model_dump()
                if self.translated_recipe else None
            ),
            "candidates_used_count": len(self.candidates_used),
            "system_prompt_preview": self.system_prompt[:300] + "…",
        }


# ─────────────────────────────────────────────────────────────────────────────
# JSON Parsing with Fallback
# ─────────────────────────────────────────────────────────────────────────────

def _extract_json(raw: str) -> Dict[str, Any]:
    """
    Extract a JSON object from the LLM's raw text output.

    Handles common edge cases:
      • Markdown code fences (```json ... ```)
      • Leading/trailing commentary
      • Multiple JSON blocks (takes the first valid one)
    """
    # Strip markdown fences
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)

    # Attempt direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fallback: find the first { … } block
    brace_depth = 0
    start = None
    for i, ch in enumerate(cleaned):
        if ch == "{":
            if brace_depth == 0:
                start = i
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1
            if brace_depth == 0 and start is not None:
                candidate = cleaned[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    start = None  # try next block

    raise ValueError(
        f"Could not extract valid JSON from LLM response. "
        f"Raw output (first 500 chars): {raw[:500]}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CulinaryEngine
# ─────────────────────────────────────────────────────────────────────────────

class CulinaryEngine:
    """
    Production-grade culinary generation engine.

    Attributes:
        model: Model name for generation.
        base_url: Ollama server base URL.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        _ = api_key
        self.model = model or LLM_MODEL
        self.base_url = (base_url or OLLAMA_BASE_URL).rstrip("/")

    # ── Ollama helpers ────────────────────────────────────────────────────

    @staticmethod
    def _response_text_from_ollama_payload(payload: Dict[str, Any]) -> str:
        response = payload.get("response", "")
        if isinstance(response, str):
            return response.strip()
        raise ValueError("Unexpected Ollama response payload format")

    # ── Step 1: Prompt Construction ──────────────────────────────────────

    def _retrieve(self, constraints: UserConstraints) -> List[Dict]:
        """Retrieval is intentionally disabled; generation is LLM-only."""
        _ = constraints
        return []

    def _build_prompts(
        self,
        constraints: UserConstraints,
        candidates: List[Dict],
        session_messages: Optional[List[Dict[str, str]]] = None,
        user_request: str = "",
    ) -> tuple[str, str]:
        """Build system and user prompts."""
        system_prompt = build_system_prompt(constraints, candidates)
        user_prompt = build_user_prompt(
            constraints,
            user_request=user_request,
            session_messages=session_messages,
        )
        return system_prompt, user_prompt

    # ── Step 2: LLM Call ─────────────────────────────────────────────────

    @staticmethod
    def _build_messages(
        user_prompt: str,
        session_messages: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        """Compose message history from prior turns + current user turn."""
        allowed_roles = {"user", "assistant"}
        messages: List[Dict[str, str]] = []
        for item in session_messages or []:
            role = str(item.get("role", "")).strip().lower()
            content = str(item.get("content", "")).strip()
            if role in allowed_roles and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def _call_llm_sync(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        temperature: float,
    ) -> str:
        """Synchronous Ollama generate API call."""
        combined_user = "\n\n".join(m.get("content", "") for m in messages if m.get("content"))
        prompt = f"{system_prompt}\n\n{combined_user}".strip()
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": LLM_MAX_TOKENS,
            },
        }
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=180,
        )
        response.raise_for_status()
        return self._response_text_from_ollama_payload(response.json())

    async def _call_llm_async(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        temperature: float,
    ) -> str:
        """Asynchronous Ollama generate API call."""
        combined_user = "\n\n".join(m.get("content", "") for m in messages if m.get("content"))
        prompt = f"{system_prompt}\n\n{combined_user}".strip()
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": LLM_MAX_TOKENS,
            },
        }
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as session:
            async with session.post(f"{self.base_url}/api/generate", json=payload) as response:
                response.raise_for_status()
                data = await response.json()
                return self._response_text_from_ollama_payload(data)

    # ── Step 4: Parse & Validate ─────────────────────────────────────────

    @staticmethod
    def _parse_recipe(raw_json: str) -> GeneratedRecipe:
        """Parse raw LLM output into a validated GeneratedRecipe."""
        data = _extract_json(raw_json)
        return GeneratedRecipe.model_validate(data)

    @staticmethod
    def _normalize_ingredient_name(name: str) -> str:
        """Normalize ingredient text for robust exact-set matching."""
        cleaned = re.sub(r"\(.*?\)", "", name.lower())
        cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        aliases = {
            "chillies": "chili",
            "chilies": "chili",
            "tomatoes": "tomato",
            "onions": "onion",
            "potatoes": "potato",
            "capsicums": "capsicum",
        }
        return aliases.get(cleaned, cleaned)

    def _validate_recipe_constraints(
        self,
        recipe: GeneratedRecipe,
        constraints: UserConstraints,
    ) -> List[str]:
        """Validate recipe quality and hard constraints requested by user."""
        errors: List[str] = []

        if recipe.estimated_time_minutes > constraints.time_constraints.max_total_minutes:
            errors.append("estimated_time_minutes exceeds max_total_minutes")

        if recipe.servings != constraints.servings:
            errors.append("servings does not match requested servings")

        allowed_tools = {tool.strip().lower() for tool in constraints.available_equipment}
        recipe_tools = {tool.strip().lower() for tool in recipe.equipment_used}
        if allowed_tools and not recipe_tools.issubset(allowed_tools):
            errors.append("recipe uses tools outside available_equipment")

        user_ings = {
            self._normalize_ingredient_name(name)
            for name in constraints.available_ingredients
            if self._normalize_ingredient_name(name)
        }
        recipe_ings = {
            self._normalize_ingredient_name(item.name)
            for item in recipe.ingredients
            if self._normalize_ingredient_name(item.name)
        }

        if recipe_ings != user_ings:
            missing = sorted(user_ings - recipe_ings)
            added = sorted(recipe_ings - user_ings)
            if missing:
                errors.append(f"recipe omitted user ingredients: {', '.join(missing)}")
            if added:
                errors.append(f"recipe added non-user ingredients: {', '.join(added)}")

        for item in recipe.ingredients:
            if item.quantity_grams <= 0:
                errors.append(f"ingredient '{item.name}' has non-positive quantity_grams")

        spice_lines = [
            ing for ing in recipe.ingredients
            if any(hint in ing.name.lower() for hint in SPICE_HINTS)
        ]
        if not spice_lines:
            errors.append("no spice ingredient with quantity was provided")

        steps = recipe.step_by_step_instructions
        if len(steps) < 5:
            errors.append("instructions are not detailed enough (minimum 5 steps)")

        vague_markers = ("cook it", "as needed", "until done", "mix well", "serve")
        detailed_steps = 0
        for step in steps:
            text = step.strip().lower()
            has_time_or_heat = bool(re.search(r"\b\d+\s*(min|minute|minutes|sec|second|seconds|c|f)\b", text))
            has_detail_length = len(text) >= 45
            is_not_vague = not any(marker == text or text.startswith(marker + " ") for marker in vague_markers)
            if has_detail_length and (has_time_or_heat or is_not_vague):
                detailed_steps += 1

        if detailed_steps < max(4, len(steps) - 1):
            errors.append("instructions contain vague or low-detail steps")

        return errors

    def _repair_with_constraints(
        self,
        constraints: UserConstraints,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        errors: List[str],
        prior_raw_response: str,
        session_messages: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """One-shot corrective call asking the model to regenerate strict JSON."""
        correction = (
            "Your previous JSON failed strict validation. "
            "Regenerate JSON now and fix all issues exactly. "
            f"Validation errors: {'; '.join(errors)}. "
            f"Original user prompt: {user_prompt}. "
            "Keep output strictly valid JSON, no commentary."
        )
        messages = self._build_messages(correction, session_messages)
        _ = constraints
        _ = prior_raw_response
        return self._call_llm_sync(system_prompt, messages, temperature)

    # ── Step 5: Temperature Selection ────────────────────────────────────

    @staticmethod
    def _select_temperature(constraints: UserConstraints) -> float:
        """High temperature for absurd combos, normal otherwise."""
        return (
            LLM_TEMPERATURE_HIGH
            if constraints.absurd_combos
            else LLM_TEMPERATURE
        )

    # ── Main Synchronous Pipeline ────────────────────────────────────────

    def generate(
        self,
        constraints: UserConstraints,
        session_messages: Optional[List[Dict[str, str]]] = None,
        user_request: str = "",
    ) -> GenerationResult:
        """
        Full synchronous generation pipeline.

        Args:
            constraints: Validated UserConstraints object.

        Returns:
            GenerationResult with recipe, nutrition, and optional translation.

        Raises:
            ValueError: If LLM output cannot be parsed into valid JSON.
        """
        logger.info("Starting generation pipeline (sync)")
        logger.info("Constraints: %s", constraints.model_dump_json(indent=2))

        # 1. Retrieval disabled by design (LLM-only generation)
        candidates: List[Dict] = []

        # 2. Prompt construction
        system_prompt, user_prompt = self._build_prompts(
            constraints,
            candidates,
            session_messages=session_messages,
            user_request=user_request,
        )
        messages = self._build_messages(user_prompt, session_messages)

        # 3. LLM call
        temperature = self._select_temperature(constraints)
        raw_response = self._call_llm_sync(system_prompt, messages, temperature)
        logger.info("LLM response received (%d chars)", len(raw_response))

        # 4. Parse & validate
        recipe = self._parse_recipe(raw_response)
        validation_errors = self._validate_recipe_constraints(recipe, constraints)
        if validation_errors:
            logger.warning("Initial response failed strict validation: %s", validation_errors)
            repaired_raw = self._repair_with_constraints(
                constraints=constraints,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                errors=validation_errors,
                prior_raw_response=raw_response,
                session_messages=session_messages,
            )
            recipe = self._parse_recipe(repaired_raw)
            raw_response = repaired_raw
            validation_errors = self._validate_recipe_constraints(recipe, constraints)
            if validation_errors:
                raise ValueError(
                    "Model output failed strict recipe constraints after retry: "
                    + "; ".join(validation_errors)
                )
        logger.info("Parsed recipe: %s", recipe.recipe_name)

        # 5. Nutritional calculation (strictly offline)
        nutrition = calculate_recipe_nutrition(recipe)
        logger.info(
            "Nutrition calculated: %.0f kcal/serving",
            nutrition.per_serving.calories_kcal,
        )

        # 6. Translation (if needed)
        translated = None
        if constraints.target_language.lower() not in ("en", "eng", "english"):
            translated = translate_display_strings_sync(
                recipe, constraints.target_language,
            )
            logger.info("Translated to %s", constraints.target_language)

        return GenerationResult(
            recipe=recipe,
            nutrition=nutrition,
            translated_recipe=translated,
            candidates_used=candidates,
            raw_llm_response=raw_response,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    # ── Main Asynchronous Pipeline ───────────────────────────────────────

    async def agenerate(
        self,
        constraints: UserConstraints,
        session_messages: Optional[List[Dict[str, str]]] = None,
        user_request: str = "",
    ) -> GenerationResult:
        """
        Full asynchronous generation pipeline.

        Identical logic to generate() but uses async LLM calls and
        async translation.
        """
        logger.info("Starting generation pipeline (async)")

        # 1. Retrieval disabled by design (LLM-only generation)
        loop = asyncio.get_running_loop()
        candidates: List[Dict] = []

        # 2. Prompt construction
        system_prompt, user_prompt = self._build_prompts(
            constraints,
            candidates,
            session_messages=session_messages,
            user_request=user_request,
        )
        messages = self._build_messages(user_prompt, session_messages)

        # 3. LLM call
        temperature = self._select_temperature(constraints)
        raw_response = await self._call_llm_async(
            system_prompt, messages, temperature,
        )

        # 4. Parse & validate
        recipe = self._parse_recipe(raw_response)
        validation_errors = self._validate_recipe_constraints(recipe, constraints)
        if validation_errors:
            raise ValueError(
                "Model output failed strict recipe constraints: "
                + "; ".join(validation_errors)
            )

        # 5. Nutritional calculation
        nutrition = await loop.run_in_executor(
            None, calculate_recipe_nutrition, recipe,
        )

        # 6. Translation (async)
        translated = None
        if constraints.target_language.lower() not in ("en", "eng", "english"):
            translated = await translate_display_strings(
                recipe, constraints.target_language,
            )

        return GenerationResult(
            recipe=recipe,
            nutrition=nutrition,
            translated_recipe=translated,
            candidates_used=candidates,
            raw_llm_response=raw_response,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    # ── Nutrition-Only Mode (no LLM) ────────────────────────────────────

    @staticmethod
    def calculate_nutrition_only(
        recipe_json: Dict[str, Any],
    ) -> RecipeNutritionReport:
        """
        Calculate nutrition for an arbitrary recipe JSON dict
        without calling the LLM.  Useful for re-processing saved recipes.
        """
        recipe = GeneratedRecipe.model_validate(recipe_json)
        return calculate_recipe_nutrition(recipe)


class FlavorEngine(CulinaryEngine):
    """Backward-compatible alias matching the Flavor Fusion AI spec."""
