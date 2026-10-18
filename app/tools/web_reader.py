"""
Web page reader for the Deep Research Agent.

Safely downloads public web pages and extracts readable text.

Supports optional HTTP client reuse to reduce connection setup
overhead when reading multiple pages.

Temporary performance diagnostics measure:
- URL validation, including DNS resolution
- HTTP request duration
- Response-body reading
- HTML text extraction

The reader retains its existing security controls and limits.
"""

import ipaddress
import socket
import time
from contextlib import nullcontext
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup


# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------

DEFAULT_TIMEOUT = 15.0
MAX_REDIRECTS = 5
MAX_RESPONSE_BYTES = 2 * 1024 * 1024  # 2 MB
MAX_CONTENT_CHARS = 50_000

USER_AGENT = (
    "Mozilla/5.0 "
    "(Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 "
    "(KHTML, like Gecko) "
    "Chrome/154.0 Safari/537.36"
)

ALLOWED_SCHEMES = {"http", "https"}

ALLOWED_CONTENT_TYPES = {
    "text/html",
    "application/xhtml+xml",
    "text/plain",
}


@dataclass
class WebPage:
    """A successfully fetched web page."""

    url: str
    title: str
    content: str


# ----------------------------------------------------------
# Timing diagnostics
# ----------------------------------------------------------

def _log_timing(stage: str, started: float) -> None:
    """Print the elapsed time for a reader stage."""

    elapsed = time.perf_counter() - started
    print(f"    [Reader Timing] {stage}: {elapsed:.3f}s")


# ----------------------------------------------------------
# URL validation
# ----------------------------------------------------------

def _is_private_or_reserved_host(hostname: str) -> bool:
    """
    Check whether a hostname resolves to private, loopback,
    link-local, reserved, or otherwise non-public IP addresses.
    """

    try:
        addresses = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return True

    for address in addresses:
        ip_string = address[4][0]

        try:
            ip = ipaddress.ip_address(ip_string)
        except ValueError:
            return True

        if not ip.is_global:
            return True

    return False


def _validate_url(url: str) -> tuple[bool, str]:
    """
    Validate a URL before making a network request.

    Returns (True, "") when valid, otherwise (False, reason).
    """

    if not url.strip():
        return False, "URL is empty."

    try:
        parsed = urlparse(url)
    except ValueError:
        return False, "URL could not be parsed."

    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        return False, "Only HTTP and HTTPS URLs are allowed."

    hostname = parsed.hostname

    if not hostname:
        return False, "URL does not contain a hostname."

    if parsed.username or parsed.password:
        return False, (
            "URLs containing username or password "
            "credentials are not allowed."
        )

    try:
        port = parsed.port
    except ValueError:
        return False, "URL contains an invalid port."

    if port is not None and port not in {80, 443}:
        return False, (
            "Only standard HTTP and HTTPS ports are allowed."
        )

    try:
        ip = ipaddress.ip_address(hostname)

        if not ip.is_global:
            return False, (
                "Private or non-public IP addresses are not allowed."
            )
    except ValueError:
        # The hostname is a domain rather than an IP literal.
        pass

    # Validate DNS resolution before making the request.
    if _is_private_or_reserved_host(hostname):
        return False, (
            "Hostname resolves to a private or "
            "non-public network address."
        )

    return True, ""


# ----------------------------------------------------------
# Response validation
# ----------------------------------------------------------

def _is_allowed_content_type(content_type: str) -> bool:
    """Reject binary downloads and unsupported response types."""

    media_type = content_type.split(";")[0].strip().lower()

    return media_type in ALLOWED_CONTENT_TYPES


# ----------------------------------------------------------
# Text extraction
# ----------------------------------------------------------

def _extract_text(html: str) -> tuple[str, str]:
    """Extract the page title and readable text."""

    soup = BeautifulSoup(html, "html.parser")

    for element in soup(
        [
            "script",
            "style",
            "noscript",
            "svg",
            "nav",
            "footer",
            "header",
        ]
    ):
        element.decompose()

    title = ""

    if soup.title:
        title = soup.title.get_text(" ", strip=True)

    main_content = soup.find(["article", "main"])

    if main_content is not None:
        content = main_content.get_text(
            separator=" ",
            strip=True,
        )
    else:
        content = soup.get_text(
            separator=" ",
            strip=True,
        )

    content = " ".join(content.split())

    return title, content


# ----------------------------------------------------------
# Main reader
# ----------------------------------------------------------

def read_webpage(
    url: str,
    timeout: float = DEFAULT_TIMEOUT,
    client: httpx.Client | None = None,
) -> WebPage | None:
    """
    Safely download a web page and extract readable text.

    Args:
        url: URL to fetch.
        timeout: Timeout used when creating an internal client.
        client: Optional reusable HTTP client. When supplied,
            this function does not close the caller's client.

    Redirects are followed manually so every destination can
    be validated before another request is made.

    Existing calls to read_webpage(url) remain supported.
    """

    # 1. Validate the initial URL.
    validation_started = time.perf_counter()
    valid, reason = _validate_url(url)
    _log_timing("Initial URL validation", validation_started)

    if not valid:
        print(f"    Rejected URL: {url}")
        print(f"    Reason: {reason}")
        return None

    current_url = url

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": (
            "text/html, "
            "application/xhtml+xml, "
            "text/plain;q=0.9"
        ),
    }

    # Create a client only when the caller did not supply one.
    # nullcontext ensures a shared client is not closed here.
    client_context = (
        httpx.Client(
            headers=headers,
            timeout=timeout,
            follow_redirects=False,
        )
        if client is None
        else nullcontext(client)
    )

    try:
        with client_context as active_client:

            # 2. Follow redirects manually.
            for redirect_number in range(MAX_REDIRECTS + 1):

                validation_started = time.perf_counter()
                valid, reason = _validate_url(current_url)

                _log_timing(
                    f"URL validation (request {redirect_number + 1})",
                    validation_started,
                )

                if not valid:
                    print(f"    Rejected redirect: {current_url}")
                    print(f"    Reason: {reason}")
                    return None

                request_started = time.perf_counter()
                response = active_client.get(current_url)
                _log_timing("HTTP request", request_started)

                if response.is_redirect:
                    if redirect_number >= MAX_REDIRECTS:
                        print(f"    Too many redirects: {url}")
                        return None

                    location = response.headers.get("location")

                    if not location:
                        print(
                            "    Redirect response "
                            "did not contain a location."
                        )
                        return None

                    current_url = urljoin(current_url, location)
                    continue

                response.raise_for_status()
                break

            else:
                return None

    except httpx.TimeoutException as exc:
        print(f"    Page request timed out: {url}")
        print(f"    Error: {exc}")
        return None

    except httpx.HTTPStatusError as exc:
        print(f"    HTTP error while reading: {url}")
        print(f"    Status: {exc.response.status_code}")
        return None

    except httpx.HTTPError as exc:
        print(f"    Failed to read page: {url}")
        print(f"    Error: {exc}")
        return None

    except Exception as exc:
        print(f"    Unexpected reader error: {url}")
        print(f"    Error: {exc}")
        return None

    # 3. Validate response content type.
    content_type = response.headers.get("content-type", "")

    if not _is_allowed_content_type(content_type):
        print(f"    Unsupported content type: {content_type}")
        return None

    # 4. Check Content-Length when provided.
    content_length = response.headers.get("content-length")

    if content_length:
        try:
            declared_size = int(content_length)
        except ValueError:
            declared_size = None

        if (
            declared_size is not None
            and declared_size > MAX_RESPONSE_BYTES
        ):
            print("    Response is too large.")
            return None

    # 5. Read the response with a hard size limit.
    body = bytearray()
    body_started = time.perf_counter()

    try:
        for chunk in response.iter_bytes(chunk_size=64 * 1024):
            body.extend(chunk)

            if len(body) > MAX_RESPONSE_BYTES:
                _log_timing("Response body reading", body_started)
                print(
                    "    Response exceeded the "
                    "maximum allowed size."
                )
                return None

    except httpx.HTTPError as exc:
        _log_timing("Response body reading", body_started)
        print(f"    Failed while reading response: {current_url}")
        print(f"    Error: {exc}")
        return None

    _log_timing("Response body reading", body_started)

    # 6. Decode the response.
    encoding = response.encoding or "utf-8"

    try:
        html = bytes(body).decode(
            encoding,
            errors="replace",
        )
    except (LookupError, UnicodeError):
        html = bytes(body).decode(
            "utf-8",
            errors="replace",
        )

    if not html.strip():
        print(f"    Page returned empty content: {current_url}")
        return None

    # 7. Extract readable text.
    extraction_started = time.perf_counter()
    title, content = _extract_text(html)
    _log_timing("HTML text extraction", extraction_started)

    if not content:
        print(f"    No readable content found: {current_url}")
        return None

    # 8. Limit content passed toward the LLM.
    if len(content) > MAX_CONTENT_CHARS:
        print(
            f"    Content truncated from {len(content):,} to "
            f"{MAX_CONTENT_CHARS:,} characters."
        )
        content = content[:MAX_CONTENT_CHARS]

    return WebPage(
        url=str(response.url),
        title=title,
        content=content,
    )
