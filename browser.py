"""
Browser automation using Playwright for JavaScript rendering
"""
import re
import logging
from playwright.sync_api import sync_playwright
from urllib.parse import urljoin, urlparse

from config import BROWSER_TIMEOUT

logger = logging.getLogger(__name__)


def render_page(url: str) -> dict:
    """
    Render a JavaScript page and extract content
    
    Args:
        url: The quiz page URL to render
    
    Returns:
        {
            "content": "Full rendered text content",
            "html": "Full rendered HTML",
            "submit_url": "Extracted submission URL",
            "base_url": "Base URL for relative links"
        }
    """
    logger.info(f"Rendering page: {url}")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        
        try:
            # Navigate and wait for JavaScript to execute
            page.goto(url, timeout=BROWSER_TIMEOUT)
            page.wait_for_load_state('networkidle', timeout=BROWSER_TIMEOUT)
            
            # Get rendered content
            html = page.content()
            text_content = page.inner_text('body')
            
            logger.info(f"Page rendered successfully. Content length: {len(text_content)} chars")
            
            # Extract the base URL
            parsed = urlparse(url)
            base_url = f"{parsed.scheme}://{parsed.netloc}"
            
            # Extract submit URL from the page content
            submit_url = extract_submit_url(text_content, html, base_url)
            
            return {
                "content": text_content,
                "html": html,
                "submit_url": submit_url,
                "base_url": base_url,
                "page_url": url
            }
            
        finally:
            browser.close()


def extract_submit_url(text_content: str, html: str, base_url: str) -> str:
    """
    Extract the submission URL from page content
    
    CRITICAL: This function explicitly extracts the URL from page content.
    It does NOT rely on LLM to determine the URL!
    
    Args:
        text_content: Rendered text content of the page
        html: Rendered HTML content
        base_url: Base URL for constructing absolute URLs
    
    Returns:
        The submission URL
    """
    logger.info("Extracting submit URL from page content")
    
    # Pattern 1: "Post your answer to https://..." 
    patterns = [
        r'[Pp]ost\s+(?:your\s+)?answer\s+to\s+(https?://[^\s<>"\']+)',
        r'[Ss]ubmit\s+(?:your\s+)?answer\s+to\s+(https?://[^\s<>"\']+)',
        r'[Ss]end\s+(?:your\s+)?(?:answer|response)\s+to\s+(https?://[^\s<>"\']+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_content)
        if match:
            url = match.group(1).rstrip('.,;:)')
            logger.info(f"Found submit URL via pattern: {url}")
            return url
    
    # Pattern 2: Look for URL in JSON example blocks containing "submit"
    json_url_pattern = r'"url":\s*"(https?://[^"]*submit[^"]*)"'
    match = re.search(json_url_pattern, text_content, re.IGNORECASE)
    if match:
        url = match.group(1)
        logger.info(f"Found submit URL in JSON example: {url}")
        return url
    
    # Pattern 3: Look for any URL containing "/submit"
    submit_pattern = r'(https?://[^\s<>"\']+/submit[^\s<>"\']*)'
    match = re.search(submit_pattern, text_content)
    if match:
        url = match.group(1).rstrip('.,;:)')
        logger.info(f"Found submit URL containing /submit: {url}")
        return url
    
    # Pattern 4: Check HTML for submit URLs in href or action attributes
    html_patterns = [
        r'href=["\'](https?://[^"\']*submit[^"\']*)["\']',
        r'action=["\'](https?://[^"\']*submit[^"\']*)["\']',
    ]
    
    for pattern in html_patterns:
        match = re.search(pattern, html, re.IGNORECASE)
        if match:
            url = match.group(1)
            logger.info(f"Found submit URL in HTML attribute: {url}")
            return url
    
    # Fallback: Use base_url + "/submit"
    fallback_url = f"{base_url}/submit"
    logger.warning(f"No explicit submit URL found, using fallback: {fallback_url}")
    return fallback_url


def download_file(url: str, save_path: str = None) -> str:
    """
    Download a file from a URL
    
    Args:
        url: URL of the file to download
        save_path: Optional path to save the file
    
    Returns:
        Path to the downloaded file
    """
    import requests
    import tempfile
    from pathlib import Path
    
    logger.info(f"Downloading file: {url}")
    
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    
    if save_path:
        path = Path(save_path)
    else:
        # Create temp file with appropriate extension
        ext = Path(urlparse(url).path).suffix or '.tmp'
        fd, path = tempfile.mkstemp(suffix=ext)
        path = Path(path)
    
    path.write_bytes(response.content)
    logger.info(f"File downloaded to: {path}")
    
    return str(path)
