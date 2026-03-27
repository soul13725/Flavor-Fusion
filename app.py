"""
app.py — Streamlit frontend for Flavor Fusion AI.

Layout:
    • Sidebar  : theme toggle, language, and all user constraints
  • Main     : ingredient input, generate button, recipe / nutrition / beverage tabs

Dark mode is implemented by injecting custom CSS overrides via st.markdown and
caching the preference in st.session_state so it persists across reruns.

Multilingual support: all UI labels are served from UI_STRINGS; the LLM
always reasons in English (ingredient names stay English for the nutrition
engine), but display strings are translated to the chosen language.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import streamlit as st

# ── Ensure project root is importable (handles stlite and direct `streamlit run`)
sys.path.insert(0, str(Path(__file__).parent))

from config import GROQ_API_KEY, VALID_EQUIPMENT, VALID_CUISINES
from engine import CulinaryEngine, GenerationResult
from models import BeveragePairing, GeneratedRecipe, TimeConstraints, UserConstraints
from retrieval import (
    load_all_recipes,
    load_beverages,
)
from user_store import (
    add_favorite,
    add_recent,
    authenticate_user,
    get_favorites,
    get_recents,
    get_user,
    init_user_db,
    request_email_verification,
    request_password_reset,
    register_user,
    reset_password,
    update_profile,
    verify_email,
)

BEVERAGE_CATEGORIES = ["Cocktail", "Mocktail", "Coffee", "Tea", "Smoothie"]
FOOD_MEAL_CATEGORIES = [
    "Breakfast",
    "Lunch",
    "Dinner",
    "Snack",
    "Dessert",
    "Appetizer",
    "Side Dish",
]

# ─────────────────────────────────────────────────────────────────────────────
# UI Label Translations
# ─────────────────────────────────────────────────────────────────────────────

UI_STRINGS: dict = {
    "en": {
        "title": "🍽️ FUSION FLAVOUR",
        "subtitle": "Discover recipes, beverages, and nutrition in one smart kitchen workspace",
        "offline_mode_label": "Offline Mode (No API Key)",
        "offline_mode_help": "Use local dataset retrieval only. No LLM creativity or translation.",
        "backend_label": "Model Backend",
        "backend_help": "Use local Mistral via Ollama, or a faster API model via Groq.",
        "language_label": "Output Language",
        "skill_label": "Skill Level",
        "equipment_label": "Available Equipment",
        "cuisine_label": "Cuisine Preference",
        "meal_label": "Meal Category",
        "prep_time_label": "Max Prep Time (min)",
        "total_time_label": "Max Total Time (min)",
        "servings_label": "Servings",
        "absurd_label": "🎲 Absurd Mode",
        "absurd_help": "Invent a wildly creative fusion dish — logic optional.",
        "ingredients_label": "🥦 Your Pantry Ingredients",
        "ingredients_placeholder": "Type ingredients manually, one per line",
        "ingredients_help": "Enter ingredients from your keyboard, one per line.",
        "generate_btn": "✨ Generate Recipe",
        "beverage_input_label": "🥂 Beverage Ingredients",
        "beverage_input_placeholder": "Type beverage ingredients manually, one per line",
        "beverage_input_help": "Enter beverage ingredients from your keyboard, one per line.",
        "beverage_type_label": "Beverage Type",
        "generate_beverage_btn": "🥂 Generate Beverage",
        "beverage_result_header": "Beverage Result",
        "no_beverage_ingredients": "⚠️ Please add at least one beverage ingredient.",
        "no_beverage_match": "No beverage match found for the current ingredients and filters.",
        "generating": "🔥 Cooking up something amazing…",
        "no_ingredients": "⚠️ Please add at least one ingredient.",
        "recipe_tab": "🍽️ Recipe",
        "nutrition_tab": "📊 Nutrition",
        "beverage_tab": "🥂 Beverage Pairing",
        "ingredients_header": "Ingredients",
        "instructions_header": "Instructions",
        "veg_symbol": "🥬",
        "beverage_symbol": "🥤",
        "per_serving": "Per Serving",
        "total_header": "Total",
        "calories": "Calories (kcal)",
        "protein": "Protein (g)",
        "fat": "Fat (g)",
        "carbs": "Carbs (g)",
        "fiber": "Fiber (g)",
        "sugar": "Sugar (g)",
        "sodium": "Sodium (mg)",
        "unmatched": "⚠️ Unmatched ingredients (excluded from totals)",
        "dark_mode": "🌙 Dark Mode",
        "db_stats": "📂 Database",
        "recipes_in_db": "recipes in DB",
        "beverages_in_db": "beverages in DB",
        "nutrition_entries": "nutrition entries",
        "error_prefix": "❌ Generation failed",
        "offline_recipe_title": "Offline Recipe",
        "offline_no_candidates": "No local dataset match found for current filters. Try broader equipment/time/cuisine settings.",
        "cuisine_badge": "🌍 {c}",
        "time_badge": "⏱️ {min} min",
        "servings_badge": "👥 {n} serving(s)",
        "equipment_badge": "🔧 {eq}",
        "candidates_used": "Candidates retrieved from DB",
        "raw_prompt_preview": "System prompt preview",
        "account_header": "👤 Account",
        "login_tab": "Login",
        "register_tab": "Register",
        "login_id_label": "Username or Email",
        "password_label": "Password",
        "login_btn": "Sign In",
        "logout_btn": "Sign Out",
        "register_user_label": "Username",
        "register_email_label": "Email",
        "register_password_label": "Create Password",
        "register_confirm_label": "Confirm Password",
        "register_btn": "Create Account",
        "password_mismatch": "Passwords do not match.",
        "auth_failed": "Invalid username/email or password.",
        "verify_tab": "Verify Email",
        "verify_code_label": "Verification Code",
        "verify_btn": "Verify Account",
        "resend_code_btn": "Generate Verification Code",
        "reset_tab": "Reset Password",
        "reset_code_label": "Reset Code",
        "new_password_label": "New Password",
        "request_reset_btn": "Generate Reset Code",
        "reset_password_btn": "Reset Password",
        "verify_required": "Please verify your account before logging in.",
        "dev_code_note": "Development mode: use the generated code below.",
        "dashboard_header": "📊 Personal Dashboard",
        "fav_tab": "Favorites",
        "recent_tab": "Recent",
        "profile_tab": "Profile",
        "recommend_tab": "Recommended",
        "no_recommend": "No recommendations yet. Save or open a few items first.",
        "save_recipe_btn": "⭐ Save Recipe",
        "save_beverage_btn": "⭐ Save Beverage",
        "saved_ok": "Saved to favorites.",
        "no_favorites": "No favorites yet.",
        "no_recent": "No recent items yet.",
        "display_name_label": "Display Name",
        "bio_label": "Bio",
        "profile_save_btn": "Update Profile",
        "datasets_header": "🌐 Dataset Sources",
        "datasets_help": "Use these public sources to build your own production database.",
    },
    "hi": {
        "title": "🍽️ FUSION FLAVOUR",
        "subtitle": "आपका AI-संचालित पाक और मिक्सोलॉजी सहायक",
        "language_label": "आउटपुट भाषा",
        "skill_label": "कौशल स्तर",
        "equipment_label": "उपलब्ध उपकरण",
        "cuisine_label": "व्यंजन प्राथमिकता",
        "meal_label": "भोजन श्रेणी",
        "prep_time_label": "अधिकतम तैयारी समय (मिनट)",
        "total_time_label": "अधिकतम कुल समय (मिनट)",
        "servings_label": "सर्विंग्स",
        "absurd_label": "🎲 बेतुका मोड",
        "absurd_help": "सभी तर्क को नजरअंदाज करते हुए एक जंगली फ्यूजन डिश।",
        "ingredients_label": "🥦 आपकी पैंट्री सामग्री",
        "ingredients_placeholder": "सामग्री स्वयं टाइप करें, हर पंक्ति में एक",
        "ingredients_help": "कीबोर्ड से सामग्री दर्ज करें, प्रति पंक्ति एक।",
        "generate_btn": "✨ रेसिपी बनाएं",
        "generating": "🔥 कुछ अद्भुत पका रहे हैं…",
        "no_ingredients": "⚠️ कृपया कम से कम एक सामग्री जोड़ें।",
        "recipe_tab": "🍽️ रेसिपी",
        "nutrition_tab": "📊 पोषण",
        "beverage_tab": "🥂 पेय पदार्थ",
        "debug_tab": "🔍 डीबग",
        "ingredients_header": "सामग्री",
        "instructions_header": "निर्देश",
        "per_serving": "प्रति सर्विंग",
        "total_header": "कुल",
        "calories": "कैलोरी (kcal)",
        "protein": "प्रोटीन (g)",
        "fat": "वसा (g)",
        "carbs": "कार्ब्स (g)",
        "fiber": "फाइबर (g)",
        "sugar": "चीनी (g)",
        "sodium": "सोडियम (mg)",
        "unmatched": "⚠️ असंबद्ध सामग्री (कुल से बाहर)",
        "dark_mode": "🌙 डार्क मोड",
        "db_stats": "📂 डेटाबेस",
        "recipes_in_db": "रेसिपी डेटाबेस में",
        "beverages_in_db": "पेय डेटाबेस में",
        "nutrition_entries": "पोषण प्रविष्टियाँ",
        "error_prefix": "❌ जनरेशन विफल",
        "cuisine_badge": "🌍 {c}",
        "time_badge": "⏱️ {min} मिनट",
        "servings_badge": "👥 {n} सर्विंग्स",
        "equipment_badge": "🔧 {eq}",
        "candidates_used": "DB से पुनर्प्राप्त उम्मीदवार",
        "raw_prompt_preview": "सिस्टम प्रॉम्प्ट पूर्वावलोकन",
    },
    "it": {
        "title": "🍽️ FUSION FLAVOUR",
        "subtitle": "Il tuo assistente culinario e mixologico alimentato dall'AI",
        "language_label": "Lingua di output",
        "skill_label": "Livello di abilità",
        "equipment_label": "Attrezzatura disponibile",
        "cuisine_label": "Preferenza culinaria",
        "meal_label": "Categoria del pasto",
        "prep_time_label": "Tempo max di preparazione (min)",
        "total_time_label": "Tempo totale massimo (min)",
        "servings_label": "Porzioni",
        "absurd_label": "🎲 Modalità Assurda",
        "absurd_help": "Inventa un piatto fusion selvaggiamente creativo.",
        "ingredients_label": "🥦 Ingredienti nella tua dispensa",
        "ingredients_placeholder": "Digita manualmente gli ingredienti, uno per riga",
        "ingredients_help": "Inserisci gli ingredienti dalla tastiera, uno per riga.",
        "generate_btn": "✨ Genera Ricetta",
        "generating": "🔥 Preparando qualcosa di straordinario…",
        "no_ingredients": "⚠️ Aggiungi almeno un ingrediente.",
        "recipe_tab": "🍽️ Ricetta",
        "nutrition_tab": "📊 Nutrizione",
        "beverage_tab": "🥂 Abbinamento bevanda",
        "debug_tab": "🔍 Debug",
        "ingredients_header": "Ingredienti",
        "instructions_header": "Istruzioni",
        "per_serving": "Per porzione",
        "total_header": "Totale",
        "calories": "Calorie (kcal)",
        "protein": "Proteine (g)",
        "fat": "Grassi (g)",
        "carbs": "Carboidrati (g)",
        "fiber": "Fibre (g)",
        "sugar": "Zucchero (g)",
        "sodium": "Sodio (mg)",
        "unmatched": "⚠️ Ingredienti non trovati (esclusi dai totali)",
        "dark_mode": "🌙 Modalità scura",
        "db_stats": "📂 Database",
        "recipes_in_db": "ricette nel DB",
        "beverages_in_db": "bevande nel DB",
        "nutrition_entries": "voci nutrizionali",
        "error_prefix": "❌ Generazione fallita",
        "cuisine_badge": "🌍 {c}",
        "time_badge": "⏱️ {min} min",
        "servings_badge": "👥 {n} porzione/i",
        "equipment_badge": "🔧 {eq}",
        "candidates_used": "Candidati recuperati dal DB",
        "raw_prompt_preview": "Anteprima del prompt di sistema",
    },
    "zh": {
        "title": "🍽️ FUSION FLAVOUR",
        "subtitle": "您的 AI 驱动烹饪和调酒助手",
        "language_label": "输出语言",
        "skill_label": "技能等级",
        "equipment_label": "可用设备",
        "cuisine_label": "菜系偏好",
        "meal_label": "餐点类别",
        "prep_time_label": "最长准备时间（分钟）",
        "total_time_label": "最长总时间（分钟）",
        "servings_label": "份数",
        "absurd_label": "🎲 荒诞模式",
        "absurd_help": "无视所有逻辑，发明一道极具创意的融合菜。",
        "ingredients_label": "🥦 您的食材清单",
        "ingredients_placeholder": "请手动输入食材，每行一种",
        "ingredients_help": "通过键盘输入食材，每行一种。",
        "generate_btn": "✨ 生成食谱",
        "generating": "🔥 正在烹饪美味…",
        "no_ingredients": "⚠️ 请至少添加一种食材。",
        "recipe_tab": "🍽️ 食谱",
        "nutrition_tab": "📊 营养信息",
        "beverage_tab": "🥂 饮品搭配",
        "debug_tab": "🔍 调试",
        "ingredients_header": "食材",
        "instructions_header": "烹饪步骤",
        "per_serving": "每份",
        "total_header": "总计",
        "calories": "卡路里 (kcal)",
        "protein": "蛋白质 (g)",
        "fat": "脂肪 (g)",
        "carbs": "碳水化合物 (g)",
        "fiber": "膳食纤维 (g)",
        "sugar": "糖 (g)",
        "sodium": "钠 (mg)",
        "unmatched": "⚠️ 未匹配食材（不计入总量）",
        "dark_mode": "🌙 深色模式",
        "db_stats": "📂 数据库",
        "recipes_in_db": "条食谱",
        "beverages_in_db": "条饮品",
        "nutrition_entries": "条营养数据",
        "error_prefix": "❌ 生成失败",
        "cuisine_badge": "🌍 {c}",
        "time_badge": "⏱️ {min} 分钟",
        "servings_badge": "👥 {n} 份",
        "equipment_badge": "🔧 {eq}",
        "candidates_used": "从数据库检索到的候选数",
        "raw_prompt_preview": "系统提示预览",
    },
    "mr": {
        "title": "🍽️ FUSION FLAVOUR",
        "subtitle": "तुमचा AI-आधारित पाककला आणि पेय सहाय्यक",
        "offline_mode_label": "ऑफलाइन मोड (API की शिवाय)",
        "offline_mode_help": "फक्त स्थानिक डेटासेट वापरा. LLM सर्जनशीलता किंवा भाषांतर नाही.",
        "language_label": "आउटपुट भाषा",
        "skill_label": "कौशल्य पातळी",
        "equipment_label": "उपलब्ध उपकरणे",
        "cuisine_label": "पाककृती प्राधान्य",
        "meal_label": "अन्न प्रकार",
        "prep_time_label": "कमाल तयारी वेळ (मिनिटे)",
        "total_time_label": "कमाल एकूण वेळ (मिनिटे)",
        "servings_label": "सर्विंग्स",
        "absurd_label": "🎲 विचित्र मोड",
        "absurd_help": "अतिशय अनोखी फ्युजन डिश तयार करा.",
        "ingredients_label": "🥦 तुमचे साहित्य",
        "ingredients_placeholder": "साहित्य हाताने टाइप करा, प्रत्येक ओळीत एक",
        "ingredients_help": "कीबोर्डवरून साहित्य लिहा, प्रत्येक ओळीत एक.",
        "generate_btn": "✨ रेसिपी तयार करा",
        "beverage_input_label": "🥂 पेयासाठी साहित्य",
        "beverage_input_placeholder": "पेयासाठी साहित्य हाताने टाइप करा, प्रत्येक ओळीत एक",
        "beverage_input_help": "कीबोर्डवरून पेयाचे साहित्य लिहा, प्रत्येक ओळीत एक.",
        "beverage_type_label": "पेय प्रकार",
        "generate_beverage_btn": "🥂 पेय तयार करा",
        "beverage_result_header": "पेय निकाल",
        "no_beverage_ingredients": "⚠️ कृपया पेयासाठी किमान एक साहित्य जोडा.",
        "no_beverage_match": "सध्याच्या साहित्य आणि फिल्टरसाठी कोणतेही पेय जुळले नाही.",
        "generating": "🔥 काहीतरी अप्रतिम तयार होत आहे…",
        "no_ingredients": "⚠️ कृपया किमान एक साहित्य जोडा.",
        "recipe_tab": "🍽️ रेसिपी",
        "nutrition_tab": "📊 पोषण",
        "beverage_tab": "🥂 पेय जोडी",
        "debug_tab": "🔍 डीबग",
        "ingredients_header": "साहित्य",
        "instructions_header": "कृती",
        "per_serving": "प्रति सर्विंग",
        "total_header": "एकूण",
        "calories": "कॅलरी (kcal)",
        "protein": "प्रथिने (g)",
        "fat": "चरबी (g)",
        "carbs": "कार्ब्स (g)",
        "fiber": "तंतू (g)",
        "sugar": "साखर (g)",
        "sodium": "सोडियम (mg)",
        "unmatched": "⚠️ न जुळलेले साहित्य (एकूणात समाविष्ट नाही)",
        "dark_mode": "🌙 डार्क मोड",
        "db_stats": "📂 डेटाबेस",
        "recipes_in_db": "डेटाबेसमधील रेसिपी",
        "beverages_in_db": "डेटाबेसमधील पेये",
        "nutrition_entries": "पोषण नोंदी",
        "error_prefix": "❌ निर्मिती अयशस्वी",
        "offline_recipe_title": "ऑफलाइन रेसिपी",
        "offline_no_candidates": "सध्याच्या फिल्टरसाठी स्थानिक डेटासेटमध्ये जुळणारी रेसिपी सापडली नाही. फिल्टर विस्तृत करून पहा.",
        "cuisine_badge": "🌍 {c}",
        "time_badge": "⏱️ {min} मिनिटे",
        "servings_badge": "👥 {n} सर्विंग्स",
        "equipment_badge": "🔧 {eq}",
        "candidates_used": "डेटाबेसमधून मिळालेले पर्याय",
        "raw_prompt_preview": "सिस्टम प्रॉम्प्ट पूर्वावलोकन",
    },
}

LANGUAGE_OPTIONS: dict = {
    "English": "en",
    "हिन्दी (Hindi)": "hi",
    "Italiano (Italian)": "it",
    "मराठी (Marathi)": "mr",
    "中文 (Mandarin)": "zh",
}

# ─────────────────────────────────────────────────────────────────────────────
# Theme CSS Injection
# ─────────────────────────────────────────────────────────────────────────────

_DARK_CSS = """
<style>
/* ── Dark Mode Overrides ── */
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stSidebar"] {
    background: radial-gradient(circle at 15% 15%, #1d2555 0%, #0f0f23 45%, #090913 100%) !important;
    color: #e2e2f0 !important;
}
[data-testid="stSidebar"] {
    background-color: #16213e !important;
}
.stMarkdown, .stText, label, p, h1, h2, h3, h4, h5, h6,
[data-testid="stSidebar"] * {
    color: #e2e2f0 !important;
}
.stTextArea textarea,
.stTextInput input,
.stSelectbox select,
[data-baseweb="select"] > div,
[data-baseweb="select"] input,
[data-baseweb="tag"],
[data-testid="stNumberInput"] input {
    background-color: #1a1a3e !important;
    color: #e2e2f0 !important;
    border-color: #4a4a7a !important;
}
[data-testid="stNumberInput"] button,
[data-testid="stNumberInput"] button svg {
    color: #e2e2f0 !important;
}
.stButton > button {
    background: linear-gradient(135deg, #6c63ff, #e040fb) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #5a52e0, #c832e8) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(108, 99, 255, 0.4) !important;
}
[data-testid="metric-container"] {
    background-color: #1a1a3e !important;
    border: 1px solid #4a4a7a !important;
    border-radius: 8px !important;
    padding: 12px !important;
}
.stTabs [data-baseweb="tab-list"] {
    background-color: #16213e !important;
}
.stTabs [data-baseweb="tab"] {
    color: #a0a0d0 !important;
}
.stTabs [aria-selected="true"] {
    color: #e2e2f0 !important;
    border-bottom-color: #6c63ff !important;
}
[data-testid="stCaptionContainer"],
[data-testid="stWidgetLabel"],
.st-emotion-cache-16txtl3,
.st-emotion-cache-1wmy9hl {
    color: #cfd3ff !important;
}
[data-testid="stExpander"] {
    background-color: #131832 !important;
    border: 1px solid #303968 !important;
    border-radius: 10px !important;
}
[data-testid="stAlertContainer"] {
    background-color: #151b38 !important;
    border-color: #3b467e !important;
}
[data-testid="stVerticalBlock"] > [style*="flex-direction: column"] > [data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 14px !important;
}
.recipe-card {
    background-color: #1a1a3e !important;
    border: 1px solid #4a4a7a !important;
}
</style>
"""

_LIGHT_CSS = """
<style>
/* ── Light Mode Overrides ── */
[data-testid="stAppViewContainer"] {
    background-color: #fafafa !important;
}
[data-testid="stSidebar"] {
    background-color: #f0f2f6 !important;
}
.stButton > button {
    background: linear-gradient(135deg, #6c63ff, #e040fb) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(108, 99, 255, 0.3) !important;
}
[data-testid="metric-container"] {
    border-radius: 8px !important;
    border: 1px solid #e0e0e0 !important;
}
</style>
"""

_BASE_CSS = """
<style>
/* ── Shared Styles ── */
.recipe-card {
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-right: 6px;
    background: rgba(108, 99, 255, 0.15);
    color: #6c63ff;
    border: 1px solid rgba(108, 99, 255, 0.3);
}
.ingredient-row {
    display: flex;
    justify-content: space-between;
    padding: 4px 0;
    border-bottom: 1px solid rgba(128,128,128,0.1);
}
.step-number {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: linear-gradient(135deg, #6c63ff, #e040fb);
    color: white;
    font-size: 0.8rem;
    font-weight: 700;
    flex-shrink: 0;
    margin-right: 10px;
}
</style>
"""


# ─────────────────────────────────────────────────────────────────────────────
# Session State Initialisation
# ─────────────────────────────────────────────────────────────────────────────

def _init_session_state() -> None:
    legacy_pantry_seed = (
        "chicken breast\nonion\ngarlic\ntomato\nyogurt\ncream\nrice\ncilantro"
    )
    defaults = {
        "dark_mode": True,
        "language": "en",
        "user_id": None,
        "user_name": "",
        "last_result": None,
        "last_beverage": None,
        "last_constraints": None,
        "pantry_text": "",
        "beverage_text": "",
        "conversation_messages": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if st.session_state.get("pantry_text", "").strip() == legacy_pantry_seed:
        st.session_state.pantry_text = ""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _t(key: str) -> str:
    """Look up a UI string in the active language."""
    lang = st.session_state.get("language", "en")
    return UI_STRINGS.get(lang, UI_STRINGS["en"]).get(
        key, UI_STRINGS["en"].get(key, key)
    )


def _apply_theme() -> None:
    """Inject a consistent dark theme and set Streamlit theme options."""
    st.markdown(_BASE_CSS, unsafe_allow_html=True)
    st.markdown(_DARK_CSS, unsafe_allow_html=True)
    try:
        st._config.set_option("theme.base", "dark")
        st._config.set_option("theme.backgroundColor", "#0f0f23")
        st._config.set_option("theme.secondaryBackgroundColor", "#16213e")
        st._config.set_option("theme.textColor", "#e2e2f0")
    except Exception:
        pass  # Internal API may not be available in all versions


def _parse_ingredients(raw: str) -> list[str]:
    """Split textarea input into ingredient tokens from lines, commas, and semicolons."""
    if not raw:
        return []

    chunks = [c.strip() for c in re.split(r"[\n,;]+", raw) if c.strip()]
    parsed: list[str] = []

    for chunk in chunks:
        # Handle inputs like "tomatoes or tomato sauce".
        if " or " in chunk.lower():
            options = [p.strip() for p in re.split(r"\s+or\s+", chunk, flags=re.IGNORECASE) if p.strip()]
            parsed.extend(options)
            continue
        parsed.append(chunk)

    cleaned_parsed: list[str] = []
    for item in parsed:
        normalized = re.sub(r"^(and|or)\s+", "", item.strip(), flags=re.IGNORECASE)
        if normalized:
            cleaned_parsed.append(normalized)

    # De-duplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for item in cleaned_parsed:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _ingredient_symbol(name: str) -> str:
    """Return a visual symbol for ingredient type."""
    lower = name.lower()
    non_veg_tokens = [
        "chicken", "mutton", "beef", "pork", "fish", "shrimp", "egg",
        "salmon", "tuna", "anchovy", "lamb", "bacon", "ham", "turkey",
    ]
    if any(token in lower for token in non_veg_tokens):
        return "🍖"
    return _t("veg_symbol")


def _is_vegetarian_recipe(recipe: GeneratedRecipe) -> bool:
    """Check if a recipe appears vegetarian based on ingredient names."""
    non_veg_tokens = {
        "chicken", "mutton", "beef", "pork", "fish", "shrimp", "egg",
        "salmon", "tuna", "anchovy", "lamb", "bacon", "ham", "turkey",
    }
    for ing in recipe.ingredients:
        lower = ing.name.lower()
        if any(token in lower for token in non_veg_tokens):
            return False
    return True


def _render_account_panel() -> None:
    """Render login/register controls in sidebar."""
    with st.expander(_t("account_header"), expanded=False):
        if st.session_state.get("user_id"):
            st.success(f"Signed in as {st.session_state.get('user_name', '')}")
            if st.button(_t("logout_btn"), key="logout_btn"):
                st.session_state.user_id = None
                st.session_state.user_name = ""
                st.rerun()
            return

        tab_login, tab_register, tab_verify, tab_reset = st.tabs([
            _t("login_tab"),
            _t("register_tab"),
            _t("verify_tab"),
            _t("reset_tab"),
        ])

        with tab_login:
            with st.form("login_form"):
                identifier = st.text_input(_t("login_id_label"))
                password = st.text_input(_t("password_label"), type="password")
                submitted = st.form_submit_button(_t("login_btn"))
                if submitted:
                    user = authenticate_user(identifier, password)
                    if not user:
                        st.error(_t("auth_failed"))
                    elif not int(user.get("email_verified", 0)):
                        st.warning(_t("verify_required"))
                    else:
                        st.session_state.user_id = user["id"]
                        st.session_state.user_name = user["username"]
                        st.rerun()

        with tab_register:
            with st.form("register_form"):
                reg_user = st.text_input(_t("register_user_label"))
                reg_email = st.text_input(_t("register_email_label"))
                reg_pass = st.text_input(_t("register_password_label"), type="password")
                reg_confirm = st.text_input(_t("register_confirm_label"), type="password")
                reg_submit = st.form_submit_button(_t("register_btn"))
                if reg_submit:
                    if reg_pass != reg_confirm:
                        st.error(_t("password_mismatch"))
                    elif not reg_user or not reg_email or not reg_pass:
                        st.error("Please fill all registration fields.")
                    else:
                        ok, message, code = register_user(reg_user, reg_email, reg_pass)
                        if ok:
                            st.success(message)
                            st.caption(_t("dev_code_note"))
                            st.code(str(code), language="text")
                        else:
                            st.error(message)

        with tab_verify:
            with st.form("verify_form"):
                verify_id = st.text_input(_t("login_id_label"))
                verify_code = st.text_input(_t("verify_code_label"))
                col_v1, col_v2 = st.columns(2)
                verify_submit = col_v1.form_submit_button(_t("verify_btn"))
                resend_submit = col_v2.form_submit_button(_t("resend_code_btn"))

                if verify_submit:
                    ok, message = verify_email(verify_id, verify_code)
                    if ok:
                        st.success(message)
                    else:
                        st.error(message)

                if resend_submit:
                    ok, message, code = request_email_verification(verify_id)
                    if ok:
                        st.success(message)
                        st.caption(_t("dev_code_note"))
                        st.code(str(code), language="text")
                    else:
                        st.error(message)

        with tab_reset:
            with st.form("reset_form"):
                reset_id = st.text_input(_t("login_id_label"), key="reset_identifier")
                reset_code = st.text_input(_t("reset_code_label"))
                new_password = st.text_input(_t("new_password_label"), type="password")
                col_r1, col_r2 = st.columns(2)
                req_submit = col_r1.form_submit_button(_t("request_reset_btn"))
                reset_submit = col_r2.form_submit_button(_t("reset_password_btn"))

                if req_submit:
                    ok, message, code = request_password_reset(reset_id)
                    if ok:
                        st.success(message)
                        st.caption(_t("dev_code_note"))
                        st.code(str(code), language="text")
                    else:
                        st.error(message)

                if reset_submit:
                    ok, message = reset_password(reset_id, reset_code, new_password)
                    if ok:
                        st.success(message)
                    else:
                        st.error(message)


def _render_payload_card(item: dict, key_prefix: str) -> None:
    """Render rich details for a favorite/recent item payload."""
    payload = item.get("payload", {})
    item_type = item.get("item_type", "")
    symbol = "🍽️" if item_type == "recipe" else _t("beverage_symbol")
    title = item.get("item_name", "Unnamed")

    with st.expander(f"{symbol} {title} · {item_type.title()}"):
        if item_type == "recipe":
            ingredients = payload.get("ingredients", [])
            steps = payload.get("step_by_step_instructions", [])
            st.markdown(f"**Ingredients:** {len(ingredients)}")
            for ing in ingredients[:12]:
                st.markdown(f"- {_ingredient_symbol(ing.get('name', ''))} {ing.get('name', '')} ({ing.get('original_measure', '')})")
            st.markdown("**Steps:**")
            for idx, step in enumerate(steps[:8], 1):
                st.markdown(f"{idx}. {step}")
        else:
            ingredients = payload.get("ingredients", [])
            steps = payload.get("instructions", [])
            st.markdown(f"**Type:** {payload.get('type', '')}")
            st.markdown("**Ingredients:**")
            for ing in ingredients[:10]:
                st.markdown(f"- {_t('beverage_symbol')} {ing.get('name', '')} ({ing.get('original_measure', '')})")
            st.markdown("**Instructions:**")
            for idx, step in enumerate(steps[:8], 1):
                st.markdown(f"{idx}. {step}")

        if st.button("Save Again", key=f"{key_prefix}_{item['id']}"):
            add_favorite(
                st.session_state.user_id,
                item_type,
                title,
                payload,
            )
            st.success(_t("saved_ok"))


def _recommend_from_history(user_id: int) -> list[dict]:
    """Generate lightweight personalized recommendations from favorites/recents."""
    history = get_favorites(user_id) + get_recents(user_id, limit=25)
    tokens: set[str] = set()
    for item in history:
        payload = item.get("payload", {})
        for ing in payload.get("ingredients", []):
            name = str(ing.get("name", "")).strip().lower()
            if len(name) > 2:
                tokens.add(name)

    if not tokens:
        return []

    token_list = list(tokens)
    recipes_df = load_all_recipes()
    beverages_df = load_beverages()

    candidates: list[dict] = []
    for _, row in recipes_df.head(3000).iterrows():
        raw_ing = str(row.get("ingredients", "")).lower()
        score = sum(1 for t in token_list if t in raw_ing)
        if score > 0:
            candidates.append({
                "item_type": "recipe",
                "item_name": row.get("recipe_name", "Unknown"),
                "score": score,
            })

    for _, row in beverages_df.head(3000).iterrows():
        raw_ing = str(row.get("ingredients", "")).lower()
        score = sum(1 for t in token_list if t in raw_ing)
        if score > 0:
            candidates.append({
                "item_type": "beverage",
                "item_name": row.get("beverage_name", "Unknown"),
                "score": score,
            })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    seen: set[str] = set()
    output: list[dict] = []
    for item in candidates:
        key = f"{item['item_type']}::{item['item_name']}"
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
        if len(output) >= 12:
            break
    return output


def _render_dashboard() -> None:
    """Render user dashboard with favorites, recents, and profile."""
    user_id = st.session_state.get("user_id")
    if not user_id:
        return

    user = get_user(user_id)
    if not user:
        return

    st.divider()
    st.markdown(f"## {_t('dashboard_header')}")
    fav_tab, recent_tab, recommend_tab, profile_tab = st.tabs([
        _t("fav_tab"),
        _t("recent_tab"),
        _t("recommend_tab"),
        _t("profile_tab"),
    ])

    with fav_tab:
        favorites = get_favorites(user_id)
        if not favorites:
            st.info(_t("no_favorites"))
        for item in favorites:
            _render_payload_card(item, "fav")

    with recent_tab:
        recents = get_recents(user_id, limit=20)
        if not recents:
            st.info(_t("no_recent"))
        for item in recents:
            _render_payload_card(item, "recent")

    with recommend_tab:
        recs = _recommend_from_history(user_id)
        if not recs:
            st.info(_t("no_recommend"))
        for item in recs:
            symbol = "🍽️" if item["item_type"] == "recipe" else _t("beverage_symbol")
            st.markdown(f"{symbol} **{item['item_name']}** · match score `{item['score']}`")

    with profile_tab:
        with st.form("profile_form"):
            display_name = st.text_input(
                _t("display_name_label"),
                value=user.get("display_name", ""),
            )
            bio = st.text_area(
                _t("bio_label"),
                value=user.get("bio", ""),
                height=100,
            )
            if st.form_submit_button(_t("profile_save_btn")):
                update_profile(user_id, display_name, bio)
                st.success("Profile updated")


def _render_beverage_pairing(bp: BeveragePairing) -> None:
    """Render a beverage pairing or standalone beverage card."""
    st.markdown(f"## {bp.name}")
    st.caption(f"Type: **{bp.type}**")
    st.divider()

    st.markdown(f"### {_t('ingredients_header')}")
    for ing in bp.ingredients:
        cols = st.columns([3, 2, 2])
        cols[0].markdown(f"{_t('beverage_symbol')} **{ing.name}**")
        cols[1].markdown(f"`{ing.quantity_grams}g`")
        cols[2].markdown(f"_{ing.original_measure}_")

    st.divider()
    st.markdown(f"### {_t('instructions_header')}")
    for i, step in enumerate(bp.instructions, 1):
        st.markdown(
            f'<div style="display:flex;align-items:flex-start;margin-bottom:10px;">'
            f'<span class="step-number">{i}</span>'
            f'<span style="padding-top:4px">{step}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def _render_sidebar() -> dict:
    """
    Render the sidebar and return a dict of all user-selected constraint values.
    """
    with st.sidebar:
        _render_account_panel()
        st.divider()

        # ── Language selector ──────────────────────────────────────────────
        lang_label = st.selectbox(
            _t("language_label"),
            options=list(LANGUAGE_OPTIONS.keys()),
            index=list(LANGUAGE_OPTIONS.values()).index(
                st.session_state.language
            ),
        )
        new_lang = LANGUAGE_OPTIONS[lang_label]
        if new_lang != st.session_state.language:
            st.session_state.language = new_lang
            st.rerun()

        st.divider()

        backend_option = st.selectbox(
            _t("backend_label"),
            options=["Mistral (Ollama)", "Fast API (Groq)"],
            index=0,
            help=_t("backend_help"),
        )
        llm_provider = "groq" if backend_option == "Fast API (Groq)" else "ollama"
        if llm_provider == "groq" and not GROQ_API_KEY:
            st.warning("GROQ_API_KEY not found. Set it in your environment before using Fast API mode.")

        st.divider()

        # ── Skill level ────────────────────────────────────────────────────
        skill = st.selectbox(
            _t("skill_label"),
            options=["Beginner", "Intermediate", "Pro"],
            index=1,
        )

        # ── Equipment ─────────────────────────────────────────────────────
        equipment = st.multiselect(
            _t("equipment_label"),
            options=sorted(VALID_EQUIPMENT),
            default=["Stove", "Oven", "Blender"],
        )

        # ── Cuisine ────────────────────────────────────────────────────────
        cuisine = st.selectbox(
            _t("cuisine_label"),
            options=sorted(VALID_CUISINES),
            index=sorted(VALID_CUISINES).index("Global"),
        )

        # ── Meal category ──────────────────────────────────────────────────
        meal_cat = st.selectbox(
            _t("meal_label"),
            options=FOOD_MEAL_CATEGORIES,
            index=FOOD_MEAL_CATEGORIES.index("Dinner"),
        )

        # ── Time constraints ───────────────────────────────────────────────
        max_prep = st.slider(_t("prep_time_label"), 5, 120, 30, step=5)
        max_total = st.slider(_t("total_time_label"), 10, 240, 60, step=10)

        # ── Servings ───────────────────────────────────────────────────────
        servings = st.number_input(
            _t("servings_label"), min_value=1, max_value=12, value=2
        )

        # ── Absurd mode ────────────────────────────────────────────────────
        absurd = st.toggle(_t("absurd_label"), help=_t("absurd_help"))

    return {
        "llm_provider": llm_provider,
        "skill": skill,
        "equipment": equipment,
        "cuisine": cuisine,
        "meal_category": meal_cat,
        "max_prep": max_prep,
        "max_total": max_total,
        "servings": servings,
        "absurd": absurd,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Recipe Card Renderer
# ─────────────────────────────────────────────────────────────────────────────

def _render_recipe_tab(recipe, lang: str) -> None:
    """Render the recipe information inside the Recipe tab."""
    # Display the translated recipe if available, otherwise the English one
    display = recipe

    # Header badges
    badges = "".join([
        f'<span class="badge">{_t("time_badge").format(min=display.estimated_time_minutes)}</span>',
        f'<span class="badge">{_t("servings_badge").format(n=display.servings)}</span>',
        f'<span class="badge">{_t("cuisine_badge").format(c=display.cuisine_type)}</span>',
    ])
    if display.equipment_used:
        badges += f'<span class="badge">{_t("equipment_badge").format(eq=", ".join(display.equipment_used))}</span>'
    if _is_vegetarian_recipe(display):
        badges += '<span class="badge">🥬 Vegetarian</span>'
    else:
        badges += '<span class="badge">🍖 Non-Veg</span>'
    st.markdown(badges, unsafe_allow_html=True)
    st.markdown("")

    # Ingredients
    st.markdown(f"### {_t('ingredients_header')}")
    for ing in display.ingredients:
        note = f" *({ing.preparation_note})*" if ing.preparation_note else ""
        cols = st.columns([3, 2, 2])
        cols[0].markdown(f"{_ingredient_symbol(ing.name)} **{ing.name}**{note}")
        cols[1].markdown(f"`{ing.quantity_grams}g`")
        cols[2].markdown(f"_{ing.original_measure}_")

    st.divider()

    # Instructions
    st.markdown(f"### {_t('instructions_header')}")
    for i, step in enumerate(display.step_by_step_instructions, 1):
        st.markdown(
            f'<div style="display:flex;align-items:flex-start;margin-bottom:10px;">'
            f'<span class="step-number">{i}</span>'
            f'<span style="padding-top:4px">{step}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )


def _render_nutrition_tab(nutrition) -> None:
    """Render the nutritional breakdown inside the Nutrition tab."""
    ps = nutrition.per_serving
    total = nutrition.total

    st.markdown(f"#### {_t('per_serving')}")
    cols = st.columns(4)
    cols[0].metric(_t("calories"), f"{ps.calories_kcal:.0f}")
    cols[1].metric(_t("protein"), f"{ps.protein_g:.1f}")
    cols[2].metric(_t("fat"), f"{ps.fat_g:.1f}")
    cols[3].metric(_t("carbs"), f"{ps.carbs_g:.1f}")

    cols2 = st.columns(3)
    cols2[0].metric(_t("fiber"), f"{ps.fiber_g:.1f}")
    cols2[1].metric(_t("sugar"), f"{ps.sugar_g:.1f}")
    cols2[2].metric(_t("sodium"), f"{ps.sodium_mg:.0f} mg")

    st.divider()
    st.markdown(f"#### {_t('total_header')} ({nutrition.servings} servings)")
    cols3 = st.columns(4)
    cols3[0].metric(_t("calories"), f"{total.calories_kcal:.0f}")
    cols3[1].metric(_t("protein"), f"{total.protein_g:.1f}")
    cols3[2].metric(_t("fat"), f"{total.fat_g:.1f}")
    cols3[3].metric(_t("carbs"), f"{total.carbs_g:.1f}")

    if nutrition.ingredient_breakdown:
        st.divider()
        st.markdown("#### Ingredient Breakdown")
        import pandas as pd
        rows = [
            {
                "Ingredient": ib.ingredient_name,
                "DB Match": ib.matched_db_name,
                "Confidence": f"{ib.match_confidence}%",
                "Grams": ib.quantity_grams,
                "kcal": ib.nutrients.calories_kcal,
                "Protein g": ib.nutrients.protein_g,
                "Fat g": ib.nutrients.fat_g,
                "Carbs g": ib.nutrients.carbs_g,
            }
            for ib in nutrition.ingredient_breakdown
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    if nutrition.unmatched_ingredients:
        st.warning(
            _t("unmatched") + ": " + ", ".join(nutrition.unmatched_ingredients)
        )


def _render_beverage_tab(recipe) -> None:
    """Render beverage pairing inside the Beverage tab."""
    display = recipe
    bp = display.beverage_pairing
    if not bp:
        st.info("No beverage pairing was generated.")
        return

    _render_beverage_pairing(bp)


def _render_debug_tab(result) -> None:
    """Render debug information for developers."""
    st.metric(_t("candidates_used"), len(result.candidates_used))
    with st.expander(_t("raw_prompt_preview")):
        st.code(result.system_prompt[:600] + "\n…", language="text")
    with st.expander("Raw LLM Response"):
        st.code(result.raw_llm_response[:800], language="json")
    with st.expander("Full Result JSON"):
        st.json(result.to_dict())


# ─────────────────────────────────────────────────────────────────────────────
# Generation Logic
# ─────────────────────────────────────────────────────────────────────────────

def _run_generation(
    ingredients: list[str],
    sidebar_vals: dict,
) -> None:
    """Build constraints, call CulinaryEngine.generate(), store in session state."""
    constraints = UserConstraints(
        skill_level=sidebar_vals["skill"],
        available_equipment=sidebar_vals["equipment"],
        available_ingredients=ingredients,
        cuisine_preference=sidebar_vals["cuisine"],
        time_constraints=TimeConstraints(
            max_prep_minutes=sidebar_vals["max_prep"],
            max_total_minutes=sidebar_vals["max_total"],
        ),
        meal_category=sidebar_vals["meal_category"],
        absurd_combos=sidebar_vals["absurd"],
        target_language=st.session_state.language,
        servings=sidebar_vals["servings"],
    )

    engine = CulinaryEngine(provider=sidebar_vals.get("llm_provider", "ollama"))

    with st.spinner(_t("generating")):
        result = engine.generate(
            constraints,
            session_messages=st.session_state.get("conversation_messages", []),
            user_request="",
        )

    st.session_state.last_result = result
    st.session_state.last_constraints = constraints
    summary_line = f"Generated recipe: {result.recipe.recipe_name}"
    turns = st.session_state.get("conversation_messages", [])
    turns.append(
        {
            "role": "user",
            "content": f"Ingredients: {', '.join(ingredients)}",
        }
    )
    turns.append({"role": "assistant", "content": summary_line})
    st.session_state.conversation_messages = turns[-20:]

    user_id = st.session_state.get("user_id")
    if user_id:
        add_recent(user_id, "recipe", result.recipe.recipe_name, result.recipe.model_dump())
        if result.recipe.beverage_pairing:
            add_recent(
                user_id,
                "beverage",
                result.recipe.beverage_pairing.name,
                result.recipe.beverage_pairing.model_dump(),
            )


def _run_beverage_generation(
    ingredients: list[str],
    beverage_type: str,
    sidebar_vals: dict,
) -> None:
    """Generate a standalone beverage from manual user input."""
    constraints = UserConstraints(
        skill_level=sidebar_vals["skill"],
        available_equipment=sidebar_vals["equipment"],
        available_ingredients=ingredients,
        cuisine_preference="Global",
        time_constraints=TimeConstraints(
            max_prep_minutes=sidebar_vals["max_prep"],
            max_total_minutes=sidebar_vals["max_total"],
        ),
        meal_category=beverage_type,
        absurd_combos=False,
        target_language="en",
        servings=sidebar_vals["servings"],
    )

    engine = CulinaryEngine(provider=sidebar_vals.get("llm_provider", "ollama"))
    beverage_request = "Generate a beverage only response."
    result = engine.generate(
        constraints,
        session_messages=st.session_state.get("conversation_messages", []),
        user_request=beverage_request,
    )

    beverage = result.recipe.beverage_pairing
    if not beverage:
        raise ValueError(_t("no_beverage_match"))

    st.session_state.last_beverage = beverage
    turns = st.session_state.get("conversation_messages", [])
    turns.append({"role": "user", "content": beverage_request})
    turns.append({"role": "assistant", "content": f"Generated beverage: {beverage.name}"})
    st.session_state.conversation_messages = turns[-20:]

    user_id = st.session_state.get("user_id")
    if user_id:
        add_recent(
            user_id,
            "beverage",
            st.session_state.last_beverage.name,
            st.session_state.last_beverage.model_dump(),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="FUSION FLAVOUR",
        page_icon="🍽️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_user_db()
    _init_session_state()
    _apply_theme()

    # ── Sidebar ──────────────────────────────────────────────────────────────
    sidebar_vals = _render_sidebar()

    # ── Main header ──────────────────────────────────────────────────────────
    st.title(_t("title"))
    st.caption(_t("subtitle"))
    st.divider()

    # ── Ingredient input ─────────────────────────────────────────────────────
    col_input, col_generate = st.columns([3, 1], gap="large")

    with col_input:
        pantry_raw = st.text_area(
            _t("ingredients_label"),
            value=st.session_state.pantry_text,
            height=180,
            placeholder=_t("ingredients_placeholder"),
            help=_t("ingredients_help"),
        )
        st.session_state.pantry_text = pantry_raw

    with col_generate:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

        if sidebar_vals["absurd"]:
            st.info("🎲 Absurd Mode ON — expect the unexpected!")

        generate_clicked = st.button(
            _t("generate_btn"),
            use_container_width=True,
            type="primary",
        )

    # ── Validation ───────────────────────────────────────────────────────────
    if generate_clicked:
        ingredients = _parse_ingredients(pantry_raw)
        if not ingredients:
            st.warning(_t("no_ingredients"))
        else:
            try:
                _run_generation(ingredients, sidebar_vals)
            except Exception as exc:
                st.error(f"{_t('error_prefix')}: {exc}")

    st.divider()

    beverage_input_col, beverage_action_col = st.columns([3, 1], gap="large")

    with beverage_input_col:
        beverage_raw = st.text_area(
            _t("beverage_input_label"),
            value=st.session_state.beverage_text,
            height=140,
            placeholder=_t("beverage_input_placeholder"),
            help=_t("beverage_input_help"),
        )
        st.session_state.beverage_text = beverage_raw

    with beverage_action_col:
        beverage_type = st.selectbox(
            _t("beverage_type_label"),
            options=BEVERAGE_CATEGORIES,
            index=BEVERAGE_CATEGORIES.index("Mocktail"),
        )
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        beverage_clicked = st.button(
            _t("generate_beverage_btn"),
            use_container_width=True,
        )

    if beverage_clicked:
        beverage_ingredients = _parse_ingredients(beverage_raw)
        if not beverage_ingredients:
            st.warning(_t("no_beverage_ingredients"))
        else:
            try:
                _run_beverage_generation(
                    beverage_ingredients,
                    beverage_type,
                    sidebar_vals,
                )
            except Exception as exc:
                st.error(f"{_t('error_prefix')}: {exc}")

    beverage_result = st.session_state.get("last_beverage")
    if beverage_result is not None:
        st.divider()
        st.markdown(f"## {_t('beverage_result_header')}")
        if st.session_state.get("user_id") and st.button(_t("save_beverage_btn"), key="save_standalone_beverage"):
            add_favorite(
                st.session_state.user_id,
                "beverage",
                beverage_result.name,
                beverage_result.model_dump(),
            )
            st.success(_t("saved_ok"))
        _render_beverage_pairing(beverage_result)

    # ── Results ──────────────────────────────────────────────────────────────
    result = st.session_state.get("last_result")
    if result is not None:
        st.divider()

        # Determine which recipe to display (translated or English)
        display_recipe = result.translated_recipe or result.recipe

        st.markdown(f"## {display_recipe.recipe_name}")

        tab_recipe, tab_nutrition, tab_beverage = st.tabs([
            _t("recipe_tab"),
            _t("nutrition_tab"),
            _t("beverage_tab"),
        ])

        if st.session_state.get("user_id") and st.button(_t("save_recipe_btn"), key="save_recipe"):
            add_favorite(
                st.session_state.user_id,
                "recipe",
                display_recipe.recipe_name,
                display_recipe.model_dump(),
            )
            st.success(_t("saved_ok"))

        if (
            st.session_state.get("user_id")
            and display_recipe.beverage_pairing
            and st.button(_t("save_beverage_btn"), key="save_paired_beverage")
        ):
            add_favorite(
                st.session_state.user_id,
                "beverage",
                display_recipe.beverage_pairing.name,
                display_recipe.beverage_pairing.model_dump(),
            )
            st.success(_t("saved_ok"))

        with tab_recipe:
            _render_recipe_tab(display_recipe, st.session_state.language)

        with tab_nutrition:
            _render_nutrition_tab(result.nutrition)

        with tab_beverage:
            _render_beverage_tab(display_recipe)

    _render_dashboard()


if __name__ == "__main__":
    main()
