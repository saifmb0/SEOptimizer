import logging
import os
import time
from dataclasses import dataclass
from typing import List, Dict, Optional
from urllib.parse import urlparse
from urllib import robotparser

import requests
from bs4 import BeautifulSoup

DEFAULT_UA = os.getenv("USER_AGENT", os.getenv("USER_AGENT", "keyword-lab/1.0"))


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


def fetch_url(url: str, timeout: int = 10, retries: int = 2, user_agent: str = DEFAULT_UA) -> Optional[Document]:
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
            return Document(url=url, title=title_text, text=text)
        except Exception as e:
            last_exc = e
            time.sleep(0.5)
    logging.debug(f"Failed to fetch {url}: {last_exc}")
    return None


def read_local_sources(path: str) -> List[Document]:
    import pathlib

    p = pathlib.Path(path)
    docs: List[Document] = []
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
) -> List[Document]:
    docs: List[Document] = []

    # Local sources
    if sources:
        docs.extend(read_local_sources(sources))

    if dry_run:
        return docs

    # If sources file listed URLs (text empty), fetch them now
    for i, d in enumerate(list(docs)):
        if d.text.strip() == "" and d.url.startswith("http"):
            fetched = fetch_url(d.url, timeout=timeout, retries=retries, user_agent=user_agent)
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
        # Fetch each result URL
        for r in results:
            url = r.get("url")
            if not url:
                continue
            fetched = fetch_url(url, timeout=timeout, retries=retries, user_agent=user_agent)
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
