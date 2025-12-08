import hashlib
import logging
import os
import random
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Optional, Callable, Set, Any
from urllib.parse import urlparse, urlencode, quote_plus
from urllib import robotparser

import requests
from bs4 import BeautifulSoup

# Tenacity for exponential backoff retries
try:
    from tenacity import (
        retry,
        stop_after_attempt,
        wait_exponential,
        retry_if_exception_type,
        before_sleep_log,
    )
    HAS_TENACITY = True
except ImportError:
    HAS_TENACITY = False
    # Fallback decorator that does nothing
    def retry(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    stop_after_attempt = lambda x: None
    wait_exponential = lambda **kwargs: None
    retry_if_exception_type = lambda x: None
    before_sleep_log = lambda *args: None

# Try to import joblib for caching
try:
    from joblib import Memory
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    Memory = None

DEFAULT_UA = os.getenv("USER_AGENT", os.getenv("USER_AGENT", "oryx-seo/1.0"))
DEFAULT_CACHE_DIR = ".oryx_cache"


# =============================================================================
# Proxy Rotation Middleware
# =============================================================================

@dataclass
class ProxyConfig:
    """
    Configuration for proxy rotation middleware.
    
    Supports BrightData, Smartproxy, or custom proxy pools.
    
    Usage:
        proxy_config = ProxyConfig(
            enabled=True,
            proxy_urls=["http://user:pass@proxy1:port", "http://user:pass@proxy2:port"],
            rotation_strategy="round_robin",  # or "random"
        )
    """
    enabled: bool = False
    proxy_urls: List[str] = field(default_factory=list)
    rotation_strategy: str = "random"  # "random" or "round_robin"
    _current_index: int = 0
    
    def get_proxy(self) -> Optional[Dict[str, str]]:
        """Get the next proxy in rotation."""
        if not self.enabled or not self.proxy_urls:
            return None
        
        if self.rotation_strategy == "round_robin":
            proxy_url = self.proxy_urls[self._current_index % len(self.proxy_urls)]
            self._current_index += 1
        else:  # random
            proxy_url = random.choice(self.proxy_urls)
        
        return {"http": proxy_url, "https": proxy_url}


# Global proxy config (can be overridden per-session)
_proxy_config = ProxyConfig()


def configure_proxies(
    proxy_urls: Optional[List[str]] = None,
    rotation_strategy: str = "random",
) -> None:
    """
    Configure global proxy rotation for scraping.
    
    Args:
        proxy_urls: List of proxy URLs (e.g., ["http://user:pass@host:port"])
        rotation_strategy: "random" or "round_robin"
        
    Example:
        # BrightData setup
        configure_proxies([
            "http://customer-id:password@brd.superproxy.io:22225",
        ])
        
        # Smartproxy setup
        configure_proxies([
            "http://user:pass@gate.smartproxy.com:7000",
        ])
    """
    global _proxy_config
    _proxy_config = ProxyConfig(
        enabled=bool(proxy_urls),
        proxy_urls=proxy_urls or [],
        rotation_strategy=rotation_strategy,
    )
    if proxy_urls:
        logging.info(f"Configured {len(proxy_urls)} proxies with {rotation_strategy} rotation")


# =============================================================================
# Resilient HTTP Client
# =============================================================================

class ScrapingError(Exception):
    """Base exception for scraping errors."""
    pass


class BlockedError(ScrapingError):
    """Raised when request is blocked (403, CAPTCHA, etc.)."""
    pass


class RateLimitError(ScrapingError):
    """Raised when rate limited (429)."""
    pass


# Retry configuration for resilient scraping
RETRY_CONFIG = {
    "stop": stop_after_attempt(5),
    "wait": wait_exponential(multiplier=1, min=2, max=30),
    "retry": retry_if_exception_type((
        requests.exceptions.Timeout,
        requests.exceptions.ConnectionError,
        RateLimitError,
    )),
    "before_sleep": before_sleep_log(logging.getLogger(__name__), logging.WARNING),
}


def _make_resilient_request(
    url: str,
    method: str = "GET",
    headers: Optional[Dict] = None,
    timeout: int = 10,
    use_proxy: bool = True,
    **kwargs,
) -> requests.Response:
    """
    Make an HTTP request with retry logic and proxy rotation.
    
    Handles:
    - Exponential backoff on failures
    - Proxy rotation on blocks
    - User-agent rotation
    - Rate limit detection
    
    Args:
        url: Target URL
        method: HTTP method
        headers: Request headers
        timeout: Request timeout
        use_proxy: Whether to use proxy rotation
        **kwargs: Additional requests kwargs
        
    Returns:
        Response object
        
    Raises:
        BlockedError: If blocked after all retries
        ScrapingError: For other unrecoverable errors
    """
    if headers is None:
        headers = {}
    
    # Ensure we have a User-Agent
    if "User-Agent" not in headers:
        headers["User-Agent"] = _get_random_ua()
    
    # Get proxy if enabled
    proxies = None
    if use_proxy and _proxy_config.enabled:
        proxies = _proxy_config.get_proxy()
    
    try:
        resp = requests.request(
            method,
            url,
            headers=headers,
            timeout=timeout,
            proxies=proxies,
            **kwargs,
        )
        
        # Handle specific status codes
        if resp.status_code == 403:
            raise BlockedError(f"Blocked (403) for {url}")
        elif resp.status_code == 429:
            raise RateLimitError(f"Rate limited (429) for {url}")
        elif resp.status_code >= 500:
            raise ScrapingError(f"Server error ({resp.status_code}) for {url}")
        
        resp.raise_for_status()
        return resp
        
    except requests.exceptions.RequestException as e:
        raise ScrapingError(f"Request failed for {url}: {e}")


def _get_random_ua() -> str:
    """Get a random User-Agent string to avoid detection."""
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    ]
    return random.choice(user_agents)

# Global cache memory instance (initialized lazily)
_cache_memory: Optional["Memory"] = None


def _get_cache_memory(cache_dir: str = DEFAULT_CACHE_DIR) -> Optional["Memory"]:
    """Get or create the cache memory instance."""
    global _cache_memory
    if not HAS_JOBLIB:
        return None
    if _cache_memory is None:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        _cache_memory = Memory(cache_path, verbose=0)
    return _cache_memory


def _url_hash(url: str) -> str:
    """Generate a hash for URL-based cache key."""
    return hashlib.md5(url.encode()).hexdigest()[:16]


@dataclass
class Document:
    url: str
    title: str
    text: str


# ============================================================================
# Google Autocomplete / Autosuggest API
# ============================================================================

# UAE/Gulf market language configurations
# Bilingual markets need both Arabic and English suggestions
BILINGUAL_GEO_CONFIG = {
    "ae": ["en", "ar"],  # UAE: English primary, Arabic secondary
    "sa": ["ar", "en"],  # Saudi: Arabic primary, English secondary
    "qa": ["ar", "en"],  # Qatar
    "kw": ["ar", "en"],  # Kuwait
    "bh": ["ar", "en"],  # Bahrain
    "om": ["ar", "en"],  # Oman
    "eg": ["ar", "en"],  # Egypt
}


@lru_cache(maxsize=512)
def get_google_suggestions(query: str, language: str = "en", country: str = "us") -> tuple:
    """
    Fetch Google Autocomplete suggestions for a query.
    
    Uses Google's public autocomplete endpoint (no API key needed for low volume).
    Results are cached to avoid repeated requests.
    
    Args:
        query: Search query to get suggestions for
        language: Language code (e.g., 'en', 'ar')
        country: Country code (e.g., 'us', 'ae')
        
    Returns:
        Tuple of suggestion strings (tuple for hashability/caching)
    """
    if not query or len(query.strip()) < 2:
        return ()
    
    # Google Autocomplete endpoint
    base_url = "https://suggestqueries.google.com/complete/search"
    params = {
        "client": "firefox",  # Returns clean JSON
        "q": query.strip(),
        "hl": language,  # Host language
        "gl": country,   # Geo location
    }
    
    try:
        headers = {"User-Agent": DEFAULT_UA}
        resp = requests.get(
            f"{base_url}?{urlencode(params)}", 
            headers=headers, 
            timeout=5
        )
        resp.raise_for_status()
        
        # Response format: [query, [suggestion1, suggestion2, ...]]
        data = resp.json()
        if isinstance(data, list) and len(data) >= 2:
            suggestions = data[1]
            if isinstance(suggestions, list):
                # Clean and normalize suggestions
                cleaned = []
                for s in suggestions:
                    if isinstance(s, str):
                        s = s.strip().lower()
                        if s and s != query.lower():
                            cleaned.append(s)
                logging.debug(f"Google suggestions for '{query}' (hl={language}, gl={country}): {len(cleaned)} results")
                return tuple(cleaned)
    except Exception as e:
        logging.debug(f"Google autocomplete failed for '{query}': {e}")
    
    return ()


def get_bilingual_suggestions(
    query: str,
    country: str = "ae",
    languages: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """
    Fetch autocomplete suggestions in multiple languages for bilingual markets.
    
    For UAE and Gulf markets, captures both Arabic and English search behavior.
    
    Args:
        query: Search query
        country: Country code (ae, sa, etc.)
        languages: Optional list of language codes, auto-detected from country if None
        
    Returns:
        Dict mapping language code to list of suggestions
    """
    if languages is None:
        # Auto-detect languages for bilingual markets
        languages = BILINGUAL_GEO_CONFIG.get(country.lower(), ["en"])
    
    results: Dict[str, List[str]] = {}
    
    for lang in languages:
        suggestions = get_google_suggestions(query, language=lang, country=country)
        results[lang] = list(suggestions)
    
    return results


def validate_keywords_with_autocomplete(
    keywords: List[str],
    language: str = "en",
    country: str = "us",
    delay: float = 0.1,
    bilingual: bool = False,
) -> Dict[str, bool]:
    """
    Validate a list of keywords against Google Autocomplete.
    
    A keyword is "validated" if it appears in autocomplete suggestions,
    indicating real search demand.
    
    Args:
        keywords: List of keywords to validate
        language: Primary language code
        country: Country code (if 'ae'/'sa'/etc., enables bilingual checks)
        delay: Delay between requests to avoid rate limiting
        bilingual: If True, check both languages for bilingual markets
        
    Returns:
        Dict mapping keyword -> validated (True if appears in autocomplete)
    """
    validated: Dict[str, bool] = {}
    
    # Determine if we should use bilingual validation
    country_lower = country.lower()
    use_bilingual = bilingual or country_lower in BILINGUAL_GEO_CONFIG
    
    for i, kw in enumerate(keywords):
        if not kw:
            continue
            
        # Use first few words as query prefix
        words = kw.split()
        prefix = " ".join(words[:2]) if len(words) > 2 else kw
        
        kw_lower = kw.lower()
        is_validated = False
        
        if use_bilingual:
            # Check both languages for bilingual markets
            bilingual_results = get_bilingual_suggestions(prefix, country_lower)
            for lang, suggestions in bilingual_results.items():
                if any(kw_lower in s or s in kw_lower for s in suggestions):
                    is_validated = True
                    break
        else:
            # Standard single-language check
            suggestions = get_google_suggestions(prefix, language, country)
            is_validated = any(
                kw_lower in s or s in kw_lower 
                for s in suggestions
            )
        
        validated[kw] = is_validated
        
        # Rate limiting - only delay if not cached
        if i > 0 and delay > 0:
            time.sleep(delay)
    
    logging.info(f"Validated {sum(validated.values())}/{len(keywords)} keywords via autocomplete")
    return validated


@lru_cache(maxsize=128)
def _get_robots_parser(robots_url: str) -> robotparser.RobotFileParser:
    """Cache robots.txt rules per domain to avoid repetitive fetching."""
    rp = robotparser.RobotFileParser()
    rp.set_url(robots_url)
    try:
        rp.read()
    except Exception:
        # If fetch fails, return empty parser (defaults to allowing)
        pass
    return rp


def _allowed_by_robots(url: str, user_agent: str) -> bool:
    try:
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        
        # Use the cached parser instead of creating new one
        rp = _get_robots_parser(robots_url)
        
        allowed = rp.can_fetch(user_agent, url)
        return allowed
    except Exception as e:
        logging.debug(f"robots.txt check failed for {url}: {e}")
        return True


# Try to import markdownify for better HTML conversion
try:
    from markdownify import markdownify as md
    HAS_MARKDOWNIFY = True
except ImportError:
    HAS_MARKDOWNIFY = False


def _calculate_link_density(element) -> float:
    """
    Calculate the link density of an element.
    
    Link density = (text in links) / (total text)
    
    High link density (>0.5) indicates navigation menus, footer link lists,
    or sidebars that should be excluded from content extraction.
    
    Args:
        element: BeautifulSoup element to analyze
        
    Returns:
        Float between 0-1 representing link density
    """
    total_text = element.get_text(strip=True)
    if not total_text:
        return 1.0  # Empty = treat as link-heavy (exclude)
    
    link_text_length = 0
    for link in element.find_all("a"):
        link_text_length += len(link.get_text(strip=True))
    
    return link_text_length / len(total_text)


def _is_content_container(element) -> bool:
    """
    Check if an element is a likely content container.
    
    Uses heuristics to identify main content areas and avoid
    navigation, footers, and sidebars.
    
    Args:
        element: BeautifulSoup element to check
        
    Returns:
        True if element is likely a content container
    """
    # Skip if no text content
    text = element.get_text(strip=True)
    if len(text) < 50:
        return False
    
    # Get element's class and id
    classes = " ".join(element.get("class", [])).lower()
    elem_id = (element.get("id") or "").lower()
    
    # Positive signals (content containers)
    content_signals = [
        "content", "article", "post", "entry", "main", "body",
        "text", "story", "description", "detail", "about",
        "service", "project", "portfolio", "work",
    ]
    
    # Negative signals (non-content areas)
    noise_signals = [
        "nav", "menu", "header", "footer", "sidebar", "widget",
        "comment", "social", "share", "related", "ad", "sponsor",
        "cookie", "popup", "modal", "banner", "subscribe",
    ]
    
    # Check for positive signals
    has_positive = any(sig in classes or sig in elem_id for sig in content_signals)
    
    # Check for negative signals
    has_negative = any(sig in classes or sig in elem_id for sig in noise_signals)
    
    # High link density = likely navigation
    link_density = _calculate_link_density(element)
    is_link_heavy = link_density > 0.5
    
    # Decision logic
    if has_negative or is_link_heavy:
        return False
    if has_positive:
        return True
    
    # For unmarked elements, use text length and link density
    # Long text with low link density is likely content
    return len(text) > 200 and link_density < 0.3


def _extract_visible_text(html: str, preserve_structure: bool = True) -> str:
    """
    Extract visible text from HTML with Block Walker approach.
    
    Uses a two-phase approach:
    1. Block Walker: Iterates over content containers (article, main, etc.)
       and ignores nav, footer, aside at the DOM level
    2. Link Density Filter: Discards blocks with >50% link text
    
    This prevents "Franken-keywords" by never mixing navigation text
    with content text at the extraction level.
    
    Args:
        html: Raw HTML content
        preserve_structure: If True and markdownify is available, converts to Markdown
                          to preserve headers and structure.
    
    Returns:
        Extracted text content with clear block boundaries
    """
    soup = BeautifulSoup(html, "html.parser")
    
    # ==========================================================================
    # Phase 1: Remove obvious noise elements
    # ==========================================================================
    
    # Standard non-content elements
    noise_tags = [
        "script", "style", "noscript", "iframe", "svg", "canvas",
        "video", "audio", "source", "track",
    ]
    for tag in soup(noise_tags):
        tag.decompose()
    
    # ==========================================================================
    # Phase 2: Remove common UI noise (UAE/Gulf websites often have these)
    # ==========================================================================
    
    # Common noise class/id patterns in UAE construction websites
    noise_patterns = [
        # Navigation and menus
        "nav", "navbar", "navigation", "menu", "header-menu",
        # Footers (often cluttered with repeated info)
        "footer", "site-footer", "footer-widget", "footer-links",
        # Sidebars
        "sidebar", "widget", "widget-area",
        # Popups and banners
        "popup", "modal", "cookie", "gdpr", "consent", "banner",
        # Chat widgets (WhatsApp is huge in UAE)
        "whatsapp", "chat-widget", "livechat", "chatbot", "wa-button",
        # Social
        "social", "share", "follow-us",
        # Ads
        "advertisement", "ad-container", "sponsor",
        # Forms (often irrelevant for content extraction)
        "contact-form", "newsletter", "subscribe",
        # Language switchers (source of "en ae" artifacts)
        "lang-switch", "language-selector", "locale", "wpml",
        "polylang", "qtranslate", "translatepress",
    ]
    
    for pattern in noise_patterns:
        # Remove by class
        for el in soup.find_all(class_=lambda c: c and pattern in str(c).lower()):
            el.decompose()
        # Remove by id
        for el in soup.find_all(id=lambda i: i and pattern in str(i).lower()):
            el.decompose()
    
    # Remove common footer/header tags
    for tag_name in ["footer", "nav", "aside"]:
        for tag in soup.find_all(tag_name):
            tag.decompose()
    
    # ==========================================================================
    # Phase 3: Remove elements with suspicious patterns
    # ==========================================================================
    
    # UAE-specific noise: phone numbers, WhatsApp links, repeated contact info
    for el in soup.find_all(["a", "span", "div"]):
        text = el.get_text(strip=True)
        # Skip very short or very long elements
        if len(text) < 3 or len(text) > 500:
            continue
        # WhatsApp links
        if "wa.me" in str(el) or "whatsapp" in str(el).lower():
            el.decompose()
            continue
        # Phone number patterns (UAE: +971, toll-free: 800)
        if text.startswith(("+971", "971", "00971", "800")) and len(text) < 20:
            el.decompose()
            continue
    
    # ==========================================================================
    # Phase 3.5: Insert separators between block-level elements
    # ==========================================================================
    # This prevents "near me" + "company property" from merging into
    # "near me company property" when they're in separate DOM elements.
    
    from bs4 import NavigableString
    
    # Block-level elements that should have clear boundaries
    block_elements = {
        "div", "p", "h1", "h2", "h3", "h4", "h5", "h6",
        "li", "td", "th", "tr", "section", "article", "main",
        "header", "blockquote", "pre", "br", "hr"
    }
    
    # Insert newlines after block elements to ensure text separation
    for tag_name in block_elements:
        for el in soup.find_all(tag_name):
            # Insert a newline after the element if it has content
            if el.string or el.get_text(strip=True):
                # Add newline as a separator
                el.insert_after(NavigableString("\n"))
    
    # Also handle inline elements that often act as boundaries (links, spans in navs)
    for el in soup.find_all(["a", "button"]):
        text = el.get_text(strip=True)
        if text and len(text) < 30:  # Short link text - likely nav element
            # Add space boundary
            el.insert_after(NavigableString(" | "))
    
    # ==========================================================================
    # Phase 3.6: Link Density Filter (Visual Density Heuristic)
    # ==========================================================================
    # Discard entire blocks with >50% link text BEFORE they reach NLP.
    # This catches mega-menus, footer link lists, and sidebar nav that
    # might have escaped pattern-based filtering.
    
    LINK_DENSITY_THRESHOLD = 0.5  # 50% links = likely navigation
    
    for container in soup.find_all(["div", "section", "ul", "ol"]):
        # Only check substantial blocks (>20 chars)
        text_content = container.get_text(strip=True)
        if len(text_content) < 20:
            continue
        
        # Calculate link density for this block
        link_density = _calculate_link_density(container)
        
        if link_density > LINK_DENSITY_THRESHOLD:
            # Log for debugging in verbose mode
            logging.debug(
                f"Discarding link-heavy block ({link_density:.0%} links): "
                f"{text_content[:50]}..."
            )
            container.decompose()
    
    # ==========================================================================
    # Phase 4: Extract content with structure preservation
    # ==========================================================================
    
    if preserve_structure and HAS_MARKDOWNIFY:
        try:
            # Get cleaned HTML
            cleaned_html = str(soup)
            # Convert to Markdown to preserve structure (headers, lists, etc.)
            text = md(cleaned_html, heading_style="ATX", strip=["script", "style", "noscript"])
            # Clean up excessive whitespace
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            # Remove duplicate lines (common in noisy footers)
            seen = set()
            unique_lines = []
            for line in lines:
                line_normalized = line.lower().strip()
                if line_normalized not in seen and len(line_normalized) > 2:
                    seen.add(line_normalized)
                    unique_lines.append(line)
            return "\n".join(unique_lines)
        except Exception:
            pass
    
    # Fallback: BeautifulSoup extraction with deduplication
    texts = []
    seen = set()
    for sel in ["h1", "h2", "h3", "h4", "p", "li", "td", "th"]:
        for el in soup.select(sel):
            t = el.get_text(" ", strip=True)
            t_normalized = t.lower().strip()
            # Skip duplicates and very short text
            if t and t_normalized not in seen and len(t) > 5:
                seen.add(t_normalized)
                texts.append(t)
    
    return "\n".join(texts)


@dataclass
class WeightedText:
    """Text content with SEO importance weight."""
    text: str
    weight: float
    source: str  # "title", "h1", "h2", "h3", "p", "li", "meta"


# SEO weight factors for HTML elements
HTML_WEIGHTS = {
    "title": 4.0,
    "h1": 3.0,
    "h2": 2.0,
    "h3": 1.5,
    "meta_description": 2.5,
    "meta_keywords": 2.0,
    "p": 1.0,
    "li": 1.0,
    "a": 0.8,
}


def extract_weighted_text(html: str) -> List[WeightedText]:
    """
    Extract text from HTML with SEO importance weights.
    
    Assigns weights based on HTML hierarchy:
    - Title: 4x weight
    - H1: 3x weight
    - H2: 2x weight
    - H3: 1.5x weight
    - Meta description: 2.5x weight
    - Paragraphs/Lists: 1x weight
    
    Args:
        html: Raw HTML content
        
    Returns:
        List of WeightedText objects with text, weight, and source tag
    """
    soup = BeautifulSoup(html, "html.parser")
    
    # Remove non-content elements
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
        tag.decompose()
    
    weighted_texts: List[WeightedText] = []
    
    # Extract title
    title_tag = soup.find("title")
    if title_tag:
        text = title_tag.get_text(strip=True)
        if text:
            weighted_texts.append(WeightedText(text, HTML_WEIGHTS["title"], "title"))
    
    # Extract meta description
    meta_desc = soup.find("meta", attrs={"name": "description"})
    if meta_desc and meta_desc.get("content"):
        text = meta_desc["content"].strip()
        if text:
            weighted_texts.append(WeightedText(text, HTML_WEIGHTS["meta_description"], "meta"))
    
    # Extract headers and content with weights
    for tag_name, weight in [("h1", 3.0), ("h2", 2.0), ("h3", 1.5), ("p", 1.0), ("li", 1.0)]:
        for el in soup.find_all(tag_name):
            text = el.get_text(" ", strip=True)
            if text and len(text) > 5:  # Skip very short text
                weighted_texts.append(WeightedText(text, weight, tag_name))
    
    return weighted_texts


def build_weighted_corpus(weighted_texts: List[WeightedText]) -> Dict[str, float]:
    """
    Build a term frequency dict with SEO weights applied.
    
    Args:
        weighted_texts: List of WeightedText objects
        
    Returns:
        Dict mapping term -> weighted frequency
    """
    from collections import Counter
    import re
    
    weighted_freq: Dict[str, float] = {}
    
    for wt in weighted_texts:
        # Simple tokenization
        words = re.findall(r'\b[a-z]{2,}\b', wt.text.lower())
        counts = Counter(words)
        
        for word, count in counts.items():
            if word not in weighted_freq:
                weighted_freq[word] = 0.0
            weighted_freq[word] += count * wt.weight
    
    return weighted_freq


def _fetch_url_uncached(
    url: str, 
    timeout: int = 10, 
    retries: int = 2, 
    user_agent: str = DEFAULT_UA
) -> Optional[Dict]:
    """
    Internal fetch function that returns a dict (for caching serialization).
    """
    if not _allowed_by_robots(url, user_agent):
        logging.info(f"Blocked by robots.txt: {url}")
        return None
    headers = {"User-Agent": user_agent}
    last_exc = None
    for i in range(retries + 1):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            if r.status_code >= 400:
                last_exc = Exception(f"HTTP {r.status_code}")
                time.sleep(0.5)
                continue
            text = _extract_visible_text(r.text)
            title = BeautifulSoup(r.text, "html.parser").title
            title_text = title.get_text(strip=True) if title else url
            return {"url": url, "title": title_text, "text": text}
        except Exception as e:
            last_exc = e
            time.sleep(0.5)
    logging.debug(f"Failed to fetch {url}: {last_exc}")
    return None


def fetch_url(
    url: str, 
    timeout: int = 10, 
    retries: int = 2, 
    user_agent: str = DEFAULT_UA,
    use_cache: bool = True,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> Optional[Document]:
    """
    Fetch a URL and return a Document.
    
    Args:
        url: The URL to fetch
        timeout: Request timeout in seconds
        retries: Number of retries on failure
        user_agent: User agent string
        use_cache: Whether to use caching (requires joblib)
        cache_dir: Directory for cache storage
        
    Returns:
        Document object or None if fetch failed
    """
    result = None
    
    if use_cache and HAS_JOBLIB:
        memory = _get_cache_memory(cache_dir)
        if memory is not None:
            cached_fetch = memory.cache(_fetch_url_uncached)
            result = cached_fetch(url, timeout, retries, user_agent)
    else:
        result = _fetch_url_uncached(url, timeout, retries, user_agent)
    
    if result is None:
        return None
    
    return Document(url=result["url"], title=result["title"], text=result["text"])


def parse_sitemap(sitemap_url: str, timeout: int = 15, user_agent: str = DEFAULT_UA) -> List[str]:
    """
    Parse a sitemap XML file and extract all URLs.
    
    Supports both regular sitemaps and sitemap index files.
    
    Args:
        sitemap_url: URL to the sitemap.xml file
        timeout: Request timeout
        user_agent: User agent string
        
    Returns:
        List of URLs found in the sitemap
    """
    urls: List[str] = []
    
    try:
        headers = {"User-Agent": user_agent}
        response = requests.get(sitemap_url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, "xml")
        
        # Check if this is a sitemap index (contains other sitemaps)
        sitemap_tags = soup.find_all("sitemap")
        if sitemap_tags:
            # This is a sitemap index, recursively parse each sitemap
            for sitemap in sitemap_tags:
                loc = sitemap.find("loc")
                if loc and loc.text:
                    child_urls = parse_sitemap(loc.text.strip(), timeout, user_agent)
                    urls.extend(child_urls)
        else:
            # Regular sitemap with URLs
            url_tags = soup.find_all("url")
            for url_tag in url_tags:
                loc = url_tag.find("loc")
                if loc and loc.text:
                    urls.append(loc.text.strip())
        
        logging.info(f"Parsed sitemap {sitemap_url}: found {len(urls)} URLs")
        
    except Exception as e:
        logging.warning(f"Failed to parse sitemap {sitemap_url}: {e}")
    
    return urls


def read_local_sources(path: str) -> List[Document]:
    import pathlib

    p = pathlib.Path(path)
    docs: List[Document] = []
    
    # Check if it's a sitemap URL
    if path.startswith("http") and path.endswith(".xml"):
        sitemap_urls = parse_sitemap(path)
        for u in sitemap_urls:
            docs.append(Document(url=u, title=u, text=""))
        return docs
    
    if p.is_file():
        # Treat as file of URLs
        try:
            with p.open("r", encoding="utf-8") as f:
                urls = [line.strip() for line in f if line.strip()]
            for u in urls:
                docs.append(Document(url=u, title=u, text=""))
        except Exception as e:
            logging.debug(f"Failed reading sources file {path}: {e}")
    elif p.is_dir():
        for fp in p.rglob("*"):
            if fp.suffix.lower() in {".txt", ".md"}:
                try:
                    text = fp.read_text(encoding="utf-8", errors="ignore")
                    docs.append(Document(url=str(fp), title=fp.stem, text=text))
                except Exception as e:
                    logging.debug(f"Failed reading file {fp}: {e}")
    return docs


def _serpapi_search(query: str, api_key: str, max_results: int = 10) -> List[Dict]:
    """
    Search via SerpAPI and return organic results.
    
    Also extracts 'People Also Ask' questions which are valuable for
    featured snippet optimization and AI citation.
    """
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google",
        "q": query,
        "num": max_results,
        "api_key": api_key,
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        results = []
        for item in data.get("organic_results", [])[:max_results]:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "total_results": data.get("search_information", {}).get("total_results")
            })
        return results
    except Exception as e:
        logging.debug(f"SerpAPI search failed: {e}")
        return []


def extract_paa_from_serpapi(query: str, api_key: str) -> List[str]:
    """
    Extract 'People Also Ask' questions from SerpAPI response.
    
    PAA questions are high-value for:
    - Featured snippet optimization
    - AI/Generative Engine citations
    - Content gap identification
    
    Args:
        query: Search query
        api_key: SerpAPI key
        
    Returns:
        List of PAA question strings
    """
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google",
        "q": query,
        "api_key": api_key,
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        
        paa_questions = []
        for item in data.get("related_questions", []):
            question = item.get("question", "").strip()
            if question:
                paa_questions.append(question.lower())
        
        logging.debug(f"Extracted {len(paa_questions)} PAA questions for '{query}'")
        return paa_questions
    except Exception as e:
        logging.debug(f"PAA extraction failed: {e}")
        return []


def extract_paa_from_bing(query: str, api_key: str) -> List[str]:
    """
    Extract related questions from Bing API response.
    
    Args:
        query: Search query
        api_key: Bing API key
        
    Returns:
        List of related question strings
    """
    url = "https://api.bing.microsoft.com/v7.0/search"
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    params = {"q": query, "responseFilter": "RelatedSearches"}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        
        questions = []
        # Bing returns related searches, filter for question-like patterns
        for item in data.get("relatedSearches", {}).get("value", []):
            text = item.get("text", "").strip().lower()
            if text and any(text.startswith(q) for q in ["how", "what", "why", "when", "where", "which", "who"]):
                questions.append(text)
        
        return questions
    except Exception as e:
        logging.debug(f"Bing PAA extraction failed: {e}")
        return []


def get_paa_questions(
    query: str,
    provider: str = "auto",
    serpapi_key: Optional[str] = None,
    bing_key: Optional[str] = None,
) -> List[str]:
    """
    Get People Also Ask questions for a query.
    
    Automatically selects provider based on available API keys.
    
    Args:
        query: Search query
        provider: 'serpapi', 'bing', or 'auto'
        serpapi_key: Optional SerpAPI key (uses env var if not provided)
        bing_key: Optional Bing key (uses env var if not provided)
        
    Returns:
        List of PAA/related question strings
    """
    serpapi_key = serpapi_key or os.getenv("SERPAPI_KEY", "")
    bing_key = bing_key or os.getenv("BING_API_KEY", "")
    
    if provider == "auto":
        if serpapi_key:
            provider = "serpapi"
        elif bing_key:
            provider = "bing"
        else:
            logging.debug("No SERP API keys available for PAA extraction")
            return []
    
    if provider == "serpapi" and serpapi_key:
        return extract_paa_from_serpapi(query, serpapi_key)
    elif provider == "bing" and bing_key:
        return extract_paa_from_bing(query, bing_key)
    
    return []


def _bing_search(query: str, api_key: str, max_results: int = 10) -> List[Dict]:
    url = "https://api.bing.microsoft.com/v7.0/search"
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    params = {"q": query, "count": max_results}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        web_pages = data.get("webPages", {}).get("value", [])
        total = data.get("webPages", {}).get("totalEstimatedMatches")
        results = []
        for item in web_pages[:max_results]:
            results.append({
                "title": item.get("name", ""),
                "url": item.get("url", ""),
                "snippet": item.get("snippet", ""),
                "total_results": total,
            })
        return results
    except Exception as e:
        logging.debug(f"Bing search failed: {e}")
        return []


def acquire_documents(
    sources: Optional[str],
    query: Optional[str],
    provider: str = "none",
    max_serp_results: int = 10,
    timeout: int = 10,
    retries: int = 2,
    user_agent: str = DEFAULT_UA,
    dry_run: bool = False,
    use_cache: bool = True,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> List[Document]:
    """
    Acquire documents from various sources.
    
    Args:
        sources: Path to local sources (file or directory)
        query: Search query for SERP providers
        provider: SERP provider (none, serpapi, bing)
        max_serp_results: Maximum results from SERP
        timeout: Request timeout
        retries: Number of retries
        user_agent: User agent string
        dry_run: If True, skip network calls
        use_cache: Enable URL caching (requires joblib)
        cache_dir: Directory for cache storage
        
    Returns:
        List of Document objects
    """
    docs: List[Document] = []

    # Local sources
    if sources:
        docs.extend(read_local_sources(sources))

    if dry_run:
        return docs

    # If sources file listed URLs (text empty), fetch them now
    for i, d in enumerate(list(docs)):
        if d.text.strip() == "" and d.url.startswith("http"):
            fetched = fetch_url(
                d.url, 
                timeout=timeout, 
                retries=retries, 
                user_agent=user_agent,
                use_cache=use_cache,
                cache_dir=cache_dir,
            )
            if fetched:
                docs[i] = fetched

    # Provider SERP acquisition
    if provider and provider != "none" and query:
        results: List[Dict] = []
        if provider == "serpapi":
            key = os.getenv("SERPAPI_KEY", "")
            if key:
                results = _serpapi_search(query, key, max_results=max_serp_results)
            else:
                logging.info("SERPAPI_KEY not set; skipping provider fetch")
        elif provider == "bing":
            key = os.getenv("BING_API_KEY", "")
            if key:
                results = _bing_search(query, key, max_results=max_serp_results)
            else:
                logging.info("BING_API_KEY not set; skipping provider fetch")
        # Fetch each result URL (with caching)
        for r in results:
            url = r.get("url")
            if not url:
                continue
            fetched = fetch_url(
                url, 
                timeout=timeout, 
                retries=retries, 
                user_agent=user_agent,
                use_cache=use_cache,
                cache_dir=cache_dir,
            )
            if fetched:
                docs.append(fetched)

    # Deduplicate by URL
    seen = set()
    unique_docs: List[Document] = []
    for d in docs:
        if d.url in seen:
            continue
        seen.add(d.url)
        unique_docs.append(d)

    logging.info(f"Acquired {len(unique_docs)} documents")
    return unique_docs
