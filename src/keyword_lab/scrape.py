import hashlib
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Callable
from urllib.parse import urlparse
from urllib import robotparser

import requests
from bs4 import BeautifulSoup

# Try to import joblib for caching
try:
    from joblib import Memory
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    Memory = None

DEFAULT_UA = os.getenv("USER_AGENT", os.getenv("USER_AGENT", "keyword-lab/1.0"))
DEFAULT_CACHE_DIR = ".keyword_lab_cache"

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


def _allowed_by_robots(url: str, user_agent: str) -> bool:
    try:
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        rp = robotparser.RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        allowed = rp.can_fetch(user_agent, url)
        return allowed
    except Exception as e:
        logging.debug(f"robots.txt check failed for {url}: {e}")
        return True


def _extract_visible_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    texts = []
    for sel in ["h1", "h2", "h3", "p", "li"]:
        for el in soup.select(sel):
            t = el.get_text(" ", strip=True)
            if t:
                texts.append(t)
    return "\n".join(texts)


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
