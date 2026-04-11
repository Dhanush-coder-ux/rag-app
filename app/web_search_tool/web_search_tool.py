"""
web_search_tool.py
==================
Production-grade web search + content extraction tool designed for use
inside AI agents (LangGraph / FastAPI RAG backends).

Capabilities
------------
- Web search via DuckDuckGo (ddgs)
- Clean article extraction via trafilatura, with BeautifulSoup fallback
- Meaningful image extraction from pages (filters out icons/ads/SVGs)
- Query-based image search via DuckDuckGo images
- Structured JSON output ready for LLM consumption
- Rate limiting, deduplication, robust error handling, and timeouts

Author : Senior AI Engineer
Usage  : Instantiate WebSearchTool and call .(query)
"""

import logging
import time
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
import trafilatura
from bs4 import BeautifulSoup
from ddgs import DDGS
from ddgs.exceptions import DDGSException

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("WebSearchTool")

DEFAULT_MAX_RESULTS: int = 5
DEFAULT_TIMEOUT: int = 10          # seconds per HTTP request
DEFAULT_MAX_CONTENT_CHARS: int = 3_000   # ~750 tokens — safe for most LLMs
DEFAULT_MAX_IMAGES_PER_PAGE: int = 5
DEFAULT_MAX_RELATED_IMAGES: int = 5
REQUEST_DELAY_SECONDS: float = 1.0  # politeness delay between page fetches

_SKIP_IMAGE_PATTERNS: tuple[str, ...] = (
    "logo", "icon", "avatar", "favicon",
    "sprite", "ads", "banner", "tracking",
    "pixel", "badge", "button", "spacer",
)


class WebSearchTool:
    """
    Tavily-style web search tool for AI agents.

    Parameters
    ----------
    max_results : int
        Maximum number of search results (and page fetches) to return.
    timeout : int
        HTTP request timeout in seconds.
    max_content_chars : int
        Maximum characters of extracted article text to return per result.
        Keeps LLM token usage predictablesearch.
    max_images_per_page : int
        Maximum content images to extract per page.
    max_related_images : int
        Maximum images returned by the image_search step.
    request_delay : float
        Seconds to sleep between consecutive page fetches (rate limiting).
    user_agent : str
        User-Agent header sent with every HTTP request.
    """

    def __init__(
        self,
        max_results: int = DEFAULT_MAX_RESULTS,
        timeout: int = DEFAULT_TIMEOUT,
        max_content_chars: int = DEFAULT_MAX_CONTENT_CHARS,
        max_images_per_page: int = DEFAULT_MAX_IMAGES_PER_PAGE,
        max_related_images: int = DEFAULT_MAX_RELATED_IMAGES,
        request_delay: float = REQUEST_DELAY_SECONDS,
        user_agent: str = (
            "Mozilla/5.0 (compatible; RAGBot/1.0; +https://example.com/bot)"
        ),
    ) -> None:
        self.max_results = max_results
        self.timeout = timeout
        self.max_content_chars = max_content_chars
        self.max_images_per_page = max_images_per_page
        self.max_related_images = max_related_images
        self.request_delay = request_delay
        self.headers = {"User-Agent": user_agent}

        self._session = requests.Session()
        self._session.headers.update(self.headers)


    def search(self, query: str) -> dict:
        """
        Run a full web search pipeline for *query*.

        Steps
        -----
        1. Fetch top-N search result metadata from DuckDuckGo.
        2. For each unique URL: extract article text + page images.
        3. Run a parallel DuckDuckGo image search for related visuals.

        Returns
        -------
        dict
            Structured payload ready for LLM context injection::

                {
                    "query": str,
                    "results": [
                        {
                            "title": str,
                            "url": str,
                            "snippet": str,
                            "content": str,   # clean article text
                            "images": [str]   # absolute image URLs
                        },
                        ...
                    ],
                    "related_images": [str]   # query-level image search
                }
        """
        logger.info("Starting search for: %r", query)

        raw_results = self._ddg_text_search(query)
        if not raw_results:
            logger.warning("No search results returned for query: %r", query)
            return {"query": query, "results": [], "related_images": []}

        results: list[dict] = []
        seen_urls: set[str] = set()   # deduplication guard

        for item in raw_results:
            url: str = item.get("href", "").strip()

            if not url or not self._is_valid_url(url):
                logger.debug("Skipping invalid URL: %r", url)
                continue
            if url in seen_urls:
                logger.debug("Skipping duplicate URL: %r", url)
                continue
            seen_urls.add(url)

            logger.info("Fetching content from: %s", url)
            content = self.extract_content(url)
            images = self.extract_images(url)

            results.append(
                {
                    "title": item.get("title", "").strip(),
                    "url": url,
                    "snippet": item.get("body", "").strip(),
                    "content": content,
                    "images": images,
                }
            )

            time.sleep(self.request_delay)

            if len(results) >= self.max_results:
                break

        related_images = self.image_search(query)

        logger.info(
            "Search complete — %d results, %d related images",
            len(results),
            len(related_images),
        )
        return {
            "query": query,
            "results": results,
            "related_images": related_images,
        }

    def extract_content(self, url: str) -> str:
        """
        Fetch *url* and return clean, readable article text.

        Strategy
        --------
        1. Primary  — trafilatura (best-in-class article extractor).
        2. Fallback — BeautifulSoup <p> tag concatenation.
        3. Hard cap  — truncated to ``max_content_chars`` to limit tokens.

        Returns an empty string on any failure so the pipeline continues.
        """
        html = self._fetch_html(url)
        if not html:
            return ""

        text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
        )
        if text:
            return self._clean_text(text)[: self.max_content_chars]

        logger.debug("trafilatura returned nothing for %s — using BS4 fallback", url)
        try:
            soup = BeautifulSoup(html, "html.parser")

            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()

            paragraphs = soup.find_all("p")
            text = " ".join(p.get_text(separator=" ") for p in paragraphs)
            return self._clean_text(text)[: self.max_content_chars]

        except Exception as exc:
            logger.warning("BS4 extraction failed for %s: %s", url, exc)
            return ""

    def extract_images(self, url: str) -> list[str]:
        """
        Extract meaningful content images from the page at *url*.

        Filters out:
        - Icons, logos, favicons, sprites, ad images, tracking pixels
        - SVG files (usually decorative/icons)
        - Images lacking a ``src`` or ``data-src`` attribute

        All relative URLs are resolved to absolute using ``urljoin``.

        Returns a list of absolute image URL strings (may be empty).
        """
        html = self._fetch_html(url)
        if not html:
            return []

        images: list[str] = []
        try:
            soup = BeautifulSoup(html, "html.parser")

            for img_tag in soup.find_all("img"):
                src: Optional[str] = img_tag.get("src") or img_tag.get("data-src")
                if not src:
                    continue

                full_url = urljoin(url, src.strip())

                if not full_url.startswith(("http://", "https://")):
                    continue

                if full_url.lower().endswith(".svg"):
                    continue

                if any(pat in full_url.lower() for pat in _SKIP_IMAGE_PATTERNS):
                    continue

                width = img_tag.get("width")
                height = img_tag.get("height")
                if width and height:
                    try:
                        if int(width) <= 2 or int(height) <= 2:
                            continue
                    except ValueError:
                        pass

                images.append(full_url)
                if len(images) >= self.max_images_per_page:
                    break

        except Exception as exc:
            logger.warning("Image extraction failed for %s: %s", url, exc)

        return images

    def image_search(self, query: str) -> list[str]:
        """
        Perform a DuckDuckGo image search for *query*.

        Returns a list of direct image URLs related to the query.
        Returns an empty list on any failure.
        """
        images: list[str] = []
        try:
            with DDGS() as ddgs:
                for result in ddgs.images(
                    query,
                    max_results=self.max_related_images,
                ):
                    image_url: Optional[str] = result.get("image")
                    if image_url:
                        images.append(image_url)

        except DDGSException as exc:
            logger.warning("DDGSException during image search for %r: %s", query, exc)
        except Exception as exc:
            logger.warning("Unexpected error during image search for %r: %s", query, exc)

        return images


    def _ddg_text_search(self, query: str) -> list[dict]:
        """
        Execute a DuckDuckGo text search and return raw result dicts.

        Handles DDGSException and generic exceptions gracefully so the
        caller always receives a list (possibly empty).
        """
        try:
            with DDGS() as ddgs:
                buffer = max(self.max_results * 2, self.max_results + 5)
                return list(ddgs.text(query, max_results=buffer))

        except DDGSException as exc:
            logger.error("DDGSException during text search for %r: %s", query, exc)
        except Exception as exc:
            logger.error("Unexpected error during text search for %r: %s", query, exc)

        return []

    def _fetch_html(self, url: str) -> Optional[str]:
        """
        Fetch the HTML at *url* using the shared session.

        Returns the response text on success, or ``None`` on any failure
        (timeout, connection error, HTTP error, invalid URL, etc.).
        """
        try:
            response = self._session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.text

        except requests.exceptions.Timeout:
            logger.warning("Timeout fetching %s (limit: %ds)", url, self.timeout)
        except requests.exceptions.ConnectionError as exc:
            logger.warning("Connection error fetching %s: %s", url, exc)
        except requests.exceptions.HTTPError as exc:
            logger.warning("HTTP error fetching %s: %s", url, exc)
        except requests.exceptions.RequestException as exc:
            logger.warning("Request failed for %s: %s", url, exc)
        except Exception as exc:
            logger.warning("Unexpected fetch error for %s: %s", url, exc)

        return None

    @staticmethod
    def _is_valid_url(url: str) -> bool:
        """Return True if *url* is a well-formed HTTP/HTTPS URL."""
        try:
            parsed = urlparse(url)
            return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
        except Exception:
            return False

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Normalise whitespace in extracted text.

        - Collapses runs of spaces/tabs into a single space.
        - Collapses 3+ consecutive newlines into 2 (preserve paragraph breaks).
        - Strips leading/trailing whitespace.
        """
        import re
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


