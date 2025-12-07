"""LLM integration for keyword expansion.

Supports multiple providers via LiteLLM or direct API calls:
- Gemini (GEMINI_API_KEY)
- OpenAI (OPENAI_API_KEY)  
- Anthropic (ANTHROPIC_API_KEY)
- Any LiteLLM-supported provider
"""

import logging
import os
from typing import List, Optional, Dict


# LiteLLM availability
try:
    import litellm
    HAS_LITELLM = True
except ImportError:
    HAS_LITELLM = False

# Direct Gemini availability  
try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


# GEO-centric intent categories for Generative Engine Optimization
GEO_INTENT_CATEGORIES = {
    "direct_answer": "Simple factual queries that can be answered in a sentence or two (e.g., 'what is the capital of france')",
    "complex_research": "Multi-faceted queries requiring comprehensive exploration (e.g., 'how to start a business')",
    "transactional": "Queries with purchase or action intent (e.g., 'buy coffee beans online')",
    "local": "Location-based queries (e.g., 'coffee shops near me')",
    "comparative": "Queries comparing options or seeking recommendations (e.g., 'best crm for small business')",
}


# =============================================================================
# Locale-Specific Terminology for LLM Prompts
# =============================================================================

# UAE/Gulf-specific terminology by niche
LOCALE_TERMINOLOGY = {
    "ae": {
        "general": [
            "Include UAE-specific terms like 'DED license', 'trade license', 'free zone', 'mainland'.",
            "Reference Dubai, Abu Dhabi, Sharjah where relevant.",
            "Include Arabic transliterations common in UAE business (e.g., 'musataha', 'ejari').",
        ],
        "contracting": [
            "Include terms: 'Civil Defense approval', 'Municipality regulations', 'NOC', 'building permit'.",
            "Reference construction authorities: 'Dubai Municipality', 'Trakhees', 'DDA'.",
            "Include fit-out terminology: 'MEP', 'HVAC', 'turnkey', 'BOQ'.",
            "Add property-specific terms: 'villa renovation', 'warehouse construction', 'Musataha lease'.",
        ],
        "real_estate": [
            "Include terms: 'freehold', 'leasehold', 'off-plan', 'ready property'.",
            "Reference developers: 'Emaar', 'Nakheel', 'DAMAC', 'Aldar'.",
            "Add area names: 'Downtown Dubai', 'Palm Jumeirah', 'Business Bay'.",
        ],
        "healthcare": [
            "Include terms: 'DHA license', 'HAAD license', 'MOH approval'.",
            "Reference: 'medical free zone', 'DHCC', 'healthcare city'.",
        ],
    },
    "sa": {
        "general": [
            "Include Saudi-specific terms like 'Saudization', 'Nitaqat', 'commercial registration'.",
            "Reference cities: 'Riyadh', 'Jeddah', 'Dammam', 'NEOM'.",
            "Include Vision 2030 related terminology where relevant.",
        ],
        "contracting": [
            "Include terms: 'Saudi Building Code', 'Momra', 'Baladi permit'.",
            "Reference mega projects: 'NEOM', 'The Line', 'Red Sea Project'.",
        ],
    },
}


def _get_locale_instructions(geo: str, niche: Optional[str] = None) -> str:
    """
    Get locale-specific instructions for LLM prompts.
    
    Args:
        geo: Geographic target (e.g., 'ae', 'sa')
        niche: Optional niche/vertical (e.g., 'contracting', 'real_estate')
        
    Returns:
        String with locale-specific instructions for the prompt
    """
    geo_lower = geo.lower().strip()
    locale_terms = LOCALE_TERMINOLOGY.get(geo_lower, {})
    
    if not locale_terms:
        return ""
    
    instructions = []
    
    # Add general locale terms
    if "general" in locale_terms:
        instructions.extend(locale_terms["general"])
    
    # Add niche-specific terms if available
    if niche and niche.lower() in locale_terms:
        instructions.extend(locale_terms[niche.lower()])
    
    if instructions:
        return "\n" + "\n".join(instructions)
    
    return ""


def _build_prompt(
    seed: str,
    audience: str,
    language: str,
    geo: str,
    max_results: int,
    niche: Optional[str] = None,
) -> str:
    """
    Build the keyword expansion prompt with locale-specific instructions.
    
    Args:
        seed: Seed topic for expansion
        audience: Target audience
        language: Language code
        geo: Geographic target
        max_results: Maximum keywords to generate
        niche: Optional niche/vertical for specialized terminology
        
    Returns:
        Complete prompt string
    """
    # Base prompt
    base_prompt = (
        "Generate a diverse, high-quality list of long-tail SEO keywords and questions as plain text, one per line.\n"
        f"Seed topic: {seed}\nAudience: {audience}\nLanguage: {language}\nGeo: {geo}\n"
        "Rules: lowercase, bigrams/trigrams or longer, focus on search intent, include questions "
        "(who/what/why/how/best/vs/for/near me/beginner/advanced/guide/checklist/template).\n"
    )
    
    # Add locale-specific instructions
    locale_instructions = _get_locale_instructions(geo, niche)
    if locale_instructions:
        base_prompt += f"\nLocale-specific requirements:{locale_instructions}\n"
    
    # Add bilingual instruction for Arabic-speaking regions
    if geo.lower() in ("ae", "sa", "qa", "kw", "bh", "om", "eg"):
        base_prompt += (
            "\nFor this market, include both English keywords and common Arabic transliterations "
            "or mixed-language queries that locals use (e.g., 'شركات مقاولات دبي', 'villa renovation dubai').\n"
        )
    
    base_prompt += f"Return at most {max_results} items. No numbering, no extra prose."
    
    return base_prompt


def _parse_response(text: str, max_results: int) -> List[str]:
    """Parse LLM response into keyword list."""
    lines = [ln.strip().lower() for ln in text.splitlines()]
    items = [ln for ln in lines if ln and len(ln.split()) >= 2]
    # Deduplicate while preserving order
    seen = set()
    out = []
    for k in items:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out[:max_results]


def _detect_provider() -> Optional[str]:
    """Auto-detect available LLM provider based on environment variables."""
    if os.getenv("GEMINI_API_KEY"):
        return "gemini"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    return None


def _expand_with_litellm(
    seed: str,
    audience: str,
    language: str,
    geo: str,
    max_results: int,
    model: Optional[str] = None,
    niche: Optional[str] = None,
) -> List[str]:
    """Expand keywords using LiteLLM (supports OpenAI, Anthropic, etc.)."""
    if not HAS_LITELLM:
        return []
    
    # Default models by detected provider
    if model is None:
        if os.getenv("OPENAI_API_KEY"):
            model = "gpt-4o-mini"
        elif os.getenv("ANTHROPIC_API_KEY"):
            model = "claude-3-haiku-20240307"
        else:
            return []
    
    prompt = _build_prompt(seed, audience, language, geo, max_results, niche)
    
    try:
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
        )
        text = response.choices[0].message.content or ""
        return _parse_response(text, max_results)
    except Exception as e:
        logging.debug(f"LiteLLM expansion failed: {e}")
        return []


def _expand_with_gemini(
    seed: str,
    audience: str,
    language: str,
    geo: str,
    max_results: int,
    model: Optional[str] = None,
    niche: Optional[str] = None,
) -> List[str]:
    """Expand keywords using Google Gemini directly."""
    if not HAS_GENAI:
        return []
    
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return []
    
    model_name = model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    prompt = _build_prompt(seed, audience, language, geo, max_results, niche)
    
    try:
        genai.configure(api_key=api_key)
        model_instance = genai.GenerativeModel(model_name)
        resp = model_instance.generate_content(prompt)
        text = getattr(resp, "text", "") or ""
        return _parse_response(text, max_results)
    except Exception as e:
        logging.debug(f"Gemini expansion failed: {e}")
        return []


def expand_with_llm(
    seed: str,
    audience: str = "",
    language: str = "en",
    geo: str = "global",
    max_results: int = 30,
    provider: str = "auto",
    model: Optional[str] = None,
    niche: Optional[str] = None,
) -> List[str]:
    """
    Expand keywords using an LLM provider.
    
    Args:
        seed: Seed topic for keyword expansion
        audience: Target audience description
        language: Language code (e.g., 'en')
        geo: Geographic target (e.g., 'global', 'us', 'ae')
        max_results: Maximum keywords to generate
        provider: LLM provider ('auto', 'gemini', 'openai', 'anthropic', 'none')
        model: Optional specific model name
        niche: Optional niche/vertical for locale-specific terminology
               (e.g., 'contracting', 'real_estate', 'healthcare')
        
    Returns:
        List of generated keyword strings
        
    Supported environment variables:
        - GEMINI_API_KEY: For Gemini provider
        - OPENAI_API_KEY: For OpenAI provider
        - ANTHROPIC_API_KEY: For Anthropic provider
    """
    if provider == "none":
        return []
    
    # Auto-detect provider
    if provider == "auto":
        provider = _detect_provider()
        if provider is None:
            logging.warning("LLM expansion skipped: No API keys found (GEMINI_API_KEY/OPENAI_API_KEY/ANTHROPIC_API_KEY).")
            return []
    
    # Route to appropriate provider (pass niche for locale-specific prompting)
    if provider == "gemini":
        return _expand_with_gemini(seed, audience, language, geo, max_results, model, niche)
    elif provider in ("openai", "anthropic"):
        return _expand_with_litellm(seed, audience, language, geo, max_results, model, niche)
    else:
        # Try LiteLLM for unknown providers
        if HAS_LITELLM:
            return _expand_with_litellm(seed, audience, language, geo, max_results, model or provider, niche)
        return []


# Backward compatibility alias
def expand_with_gemini(
    seed: str,
    audience: str = "",
    language: str = "en", 
    geo: str = "global",
    max_results: int = 30,
) -> List[str]:
    """Legacy function - use expand_with_llm instead."""
    return expand_with_llm(seed, audience, language, geo, max_results, provider="gemini")


# ============================================================================
# GEO-Centric Intent Classification
# ============================================================================

def _build_intent_prompt(keywords: List[str]) -> str:
    """Build the intent classification prompt for GEO categories."""
    categories = "\n".join(f"- {k}: {v}" for k, v in GEO_INTENT_CATEGORIES.items())
    keyword_list = "\n".join(f"- {kw}" for kw in keywords[:50])  # Limit batch size
    
    return f"""Classify each keyword into exactly ONE of these GEO-centric intent categories:

{categories}

Keywords to classify:
{keyword_list}

Return ONLY a JSON object mapping each keyword to its intent category.
Example format: {{"keyword1": "direct_answer", "keyword2": "comparative"}}
No additional text or explanation."""


def _parse_intent_response(text: str, keywords: List[str]) -> Dict[str, str]:
    """Parse LLM intent classification response."""
    import json
    
    # Try to extract JSON from response
    try:
        # Find JSON object in response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            json_str = text[start:end]
            result = json.loads(json_str)
            
            # Validate and normalize
            valid_intents = set(GEO_INTENT_CATEGORIES.keys())
            normalized = {}
            for kw in keywords:
                intent = result.get(kw, result.get(kw.lower(), "informational"))
                if isinstance(intent, str):
                    intent = intent.lower().replace(" ", "_")
                    if intent in valid_intents:
                        normalized[kw] = intent
                    else:
                        # Map to closest match or default
                        normalized[kw] = "informational"
                else:
                    normalized[kw] = "informational"
            return normalized
    except (json.JSONDecodeError, Exception) as e:
        logging.debug(f"Failed to parse intent response: {e}")
    
    # Fallback: return informational for all
    return {kw: "informational" for kw in keywords}


def classify_intents_with_llm(
    keywords: List[str],
    provider: str = "auto",
    model: Optional[str] = None,
    batch_size: int = 50,
) -> Dict[str, str]:
    """
    Classify keywords into GEO-centric intent categories using LLM.
    
    Categories:
    - direct_answer: Simple factual queries
    - complex_research: Multi-faceted exploratory queries
    - transactional: Purchase/action intent
    - local: Location-based queries
    - comparative: Comparison/recommendation queries
    
    Args:
        keywords: List of keywords to classify
        provider: LLM provider ('auto', 'gemini', 'openai', 'anthropic')
        model: Optional specific model name
        batch_size: Max keywords per LLM call
        
    Returns:
        Dict mapping keyword -> intent category
    """
    if not keywords:
        return {}
    
    if provider == "none":
        return {kw: "informational" for kw in keywords}
    
    # Auto-detect provider
    if provider == "auto":
        provider = _detect_provider()
        if provider is None:
            logging.debug("No LLM available for intent classification")
            return {kw: "informational" for kw in keywords}
    
    all_intents: Dict[str, str] = {}
    
    # Process in batches
    for i in range(0, len(keywords), batch_size):
        batch = keywords[i:i + batch_size]
        prompt = _build_intent_prompt(batch)
        
        try:
            if provider == "gemini" and HAS_GENAI:
                api_key = os.getenv("GEMINI_API_KEY", "").strip()
                if api_key:
                    genai.configure(api_key=api_key)
                    model_name = model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
                    model_instance = genai.GenerativeModel(model_name)
                    resp = model_instance.generate_content(prompt)
                    text = getattr(resp, "text", "") or ""
                    batch_intents = _parse_intent_response(text, batch)
                    all_intents.update(batch_intents)
            elif HAS_LITELLM:
                llm_model = model
                if llm_model is None:
                    if provider == "openai" or os.getenv("OPENAI_API_KEY"):
                        llm_model = "gpt-4o-mini"
                    elif provider == "anthropic" or os.getenv("ANTHROPIC_API_KEY"):
                        llm_model = "claude-3-haiku-20240307"
                
                if llm_model:
                    response = litellm.completion(
                        model=llm_model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=2000,
                    )
                    text = response.choices[0].message.content or ""
                    batch_intents = _parse_intent_response(text, batch)
                    all_intents.update(batch_intents)
        except Exception as e:
            logging.debug(f"LLM intent classification failed: {e}")
            # Fallback for failed batch
            for kw in batch:
                if kw not in all_intents:
                    all_intents[kw] = "informational"
    
    # Fill any missing keywords
    for kw in keywords:
        if kw not in all_intents:
            all_intents[kw] = "informational"
    
    logging.info(f"Classified {len(all_intents)} keywords with LLM intent detection")
    return all_intents


def assign_parent_topics(
    keywords: List[str],
    provider: str = "auto",
    model: Optional[str] = None,
    max_topics: int = 10,
) -> Dict[str, str]:
    """
    Assign parent topics to keywords for hub-spoke SEO architecture.
    
    Creates topical clusters where each keyword maps to a broader "pillar" topic.
    This enables SEO silo architecture for building topical authority.
    
    Args:
        keywords: List of keywords to assign topics
        provider: LLM provider
        model: Optional model name
        max_topics: Maximum number of parent topics to create
        
    Returns:
        Dict mapping keyword -> parent topic
    """
    if not keywords:
        return {}
    
    if provider == "none":
        return {kw: kw.split()[0] for kw in keywords}  # Use first word as topic
    
    # Auto-detect provider
    if provider == "auto":
        provider = _detect_provider()
        if provider is None:
            return {kw: kw.split()[0] for kw in keywords}
    
    keyword_list = "\n".join(f"- {kw}" for kw in keywords[:100])
    
    prompt = f"""Analyze these SEO keywords and assign each to a parent topic (pillar).
Create at most {max_topics} parent topics that represent the main themes.
Parent topics should be 2-4 words, descriptive, and SEO-friendly.

Keywords:
{keyword_list}

Return ONLY a JSON object mapping each keyword to its parent topic.
Example: {{"how to brew coffee": "coffee brewing", "best espresso machine": "espresso equipment"}}
No additional text."""

    try:
        text = ""
        if provider == "gemini" and HAS_GENAI:
            api_key = os.getenv("GEMINI_API_KEY", "").strip()
            if api_key:
                genai.configure(api_key=api_key)
                model_name = model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
                model_instance = genai.GenerativeModel(model_name)
                resp = model_instance.generate_content(prompt)
                text = getattr(resp, "text", "") or ""
        elif HAS_LITELLM:
            llm_model = model or ("gpt-4o-mini" if os.getenv("OPENAI_API_KEY") else "claude-3-haiku-20240307")
            response = litellm.completion(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=3000,
            )
            text = response.choices[0].message.content or ""
        
        if text:
            import json
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(text[start:end])
                # Normalize and fill missing
                topics = {}
                for kw in keywords:
                    topic = result.get(kw, result.get(kw.lower()))
                    if isinstance(topic, str) and topic.strip():
                        topics[kw] = topic.strip().lower()
                    else:
                        topics[kw] = kw.split()[0]
                logging.info(f"Assigned {len(set(topics.values()))} parent topics to {len(topics)} keywords")
                return topics
    except Exception as e:
        logging.debug(f"Parent topic assignment failed: {e}")
    
    # Fallback
    return {kw: kw.split()[0] for kw in keywords}
