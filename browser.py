"""
Browser Module - Playwright page rendering and URL extraction
"""
import re
import logging
from playwright.sync_api import sync_playwright
from urllib.parse import urlparse

from config import BROWSER_TIMEOUT

logger = logging.getLogger(__name__)


def render_page(url: str) -> dict:
    """
    Render a JavaScript page and extract content
    
    Returns:
        {
            "content": "rendered text content",
            "html": "rendered HTML",
            "submit_url": "extracted submit URL",
            "base_url": "base URL for relative links",
            "files": {"audio": [], "csv": [], "json": [], ...}
        }
    """
    logger.info(f"Rendering: {url}")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        
        try:
            page.goto(url, timeout=BROWSER_TIMEOUT)
            page.wait_for_load_state("networkidle", timeout=BROWSER_TIMEOUT)
            
            content = page.inner_text("body")
            html = page.content()
            
            parsed = urlparse(url)
            base_url = f"{parsed.scheme}://{parsed.netloc}"
            
            submit_url = extract_submit_url(content, html, base_url)
            files = extract_file_urls(html, base_url)
            
            logger.info(f"Content length: {len(content)} chars")
            
            return {
                "content": content,
                "html": html,
                "submit_url": submit_url,
                "base_url": base_url,
                "page_url": url,
                "files": files
            }
            
        finally:
            browser.close()


def extract_submit_url(content: str, html: str, base_url: str) -> str:
    """
    Extract submission URL from page content
    CRITICAL: Extracts from page, NOT hallucinated!
    """
    logger.info("Extracting submit URL")
    
    # Pattern 1: "Post your answer to https://..."
    patterns = [
        r'(?:post|submit|send).*?(https?://[^\s<>"]+/submit[^\s<>"]*)',
        r'(https?://[^\s<>"]+/submit)',
        r'/submit[^\s<>"]*',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            url = match.group(1) if '(' in pattern else match.group(0)
            if url.startswith('/'):
                url = base_url + url
            logger.info(f"Found submit URL: {url}")
            return url
    
    # Check HTML attributes
    for pattern in [r'href=["\']([^"\']*submit[^"\']*)["\']', r'action=["\']([^"\']*submit[^"\']*)["\']']:
        match = re.search(pattern, html, re.IGNORECASE)
        if match:
            url = match.group(1)
            if url.startswith('/'):
                url = base_url + url
            logger.info(f"Found submit URL in HTML: {url}")
            return url
    
    # Fallback
    fallback = f"{base_url}/submit"
    logger.warning(f"Using fallback: {fallback}")
    return fallback


def extract_file_urls(html: str, base_url: str) -> dict:
    """Extract file URLs from page HTML"""
    files = {
        "audio": [],
        "csv": [],
        "json": [],
        "pdf": [],
        "zip": [],
        "image": []
    }
    
    # Find all URLs in HTML
    url_pattern = r'(?:href|src)=["\']([^"\']+)["\']'
    matches = re.findall(url_pattern, html, re.IGNORECASE)
    
    for url in matches:
        if url.startswith('/'):
            url = base_url + url
        elif not url.startswith('http'):
            continue
        
        lower = url.lower()
        if any(ext in lower for ext in ['.mp3', '.wav', '.opus', '.ogg', '.m4a']):
            files["audio"].append(url)
        elif '.csv' in lower:
            files["csv"].append(url)
        elif '.json' in lower:
            files["json"].append(url)
        elif '.pdf' in lower:
            files["pdf"].append(url)
        elif '.zip' in lower:
            files["zip"].append(url)
        elif any(ext in lower for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']):
            files["image"].append(url)
    
    logger.info(f"Found files: {files}")
    return files
