"""
Web page reader for the Deep Research Agent.

This module safely downloads public web pages and extracts
their readable text for downstream research and evidence
extraction.

The reader is intentionally focused on:
- URL validation
- basic SSRF protection
- controlled redirects
- request timeouts
- response-size limits
- HTML/text content validation
- readable text extraction
- graceful failure handling

It does not attempt to render JavaScript-heavy pages or
bypass CAPTCHA, authentication, paywalls, or anti-bot systems.
"""

import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup


# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------

DEFAULT_TIMEOUT = 15.0

MAX_REDIRECTS = 5

# Maximum amount of response data we are willing to read.
# This prevents unexpectedly large pages from consuming
# excessive memory.
MAX_RESPONSE_BYTES = 2 * 1024 * 1024  # 2 MB

# Maximum amount of extracted text that will be stored in
# the Source object and eventually passed toward the LLM.
MAX_CONTENT_CHARS = 50_000


USER_AGENT = (
    "Mozilla/5.0 "
    "(Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 "
    "(KHTML, like Gecko) "
    "Chrome/154.0 Safari/537.36"
)


ALLOWED_SCHEMES = {
    "http",
    "https",
}


ALLOWED_CONTENT_TYPES = {
    "text/html",
    "application/xhtml+xml",
    "text/plain",
}


@dataclass
class WebPage:
    """
    A successfully fetched web page.
    """

    url: str
    title: str
    content: str


# ----------------------------------------------------------
# URL validation
# ----------------------------------------------------------


def _is_private_or_reserved_host(
    hostname: str,
) -> bool:
    """
    Check whether a hostname resolves to a private,
    loopback, link-local, reserved, or otherwise non-public
    IP address.

    This provides basic SSRF protection.
    """

    try:
        addresses = socket.getaddrinfo(
            hostname,
            None,
        )

    except socket.gaierror:
        return True

    for address in addresses:

        ip_string = address[4][0]

        try:
            ip = ipaddress.ip_address(
                ip_string
            )

        except ValueError:
            return True

        if not ip.is_global:
            return True

    return False


def _validate_url(
    url: str,
) -> tuple[bool, str]:
    """
    Validate a URL before making a network request.

    Returns
    -------
    tuple[bool, str]
        (True, "") when valid.

        (False, reason) when rejected.
    """

    if not url.strip():
        return False, "URL is empty."

    try:
        parsed = urlparse(url)

    except ValueError:
        return False, "URL could not be parsed."

    # ------------------------------------------------------
    # Scheme
    # ------------------------------------------------------

    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        return False, (
            "Only HTTP and HTTPS URLs are allowed."
        )

    # ------------------------------------------------------
    # Host
    # ------------------------------------------------------

    hostname = parsed.hostname

    if not hostname:
        return False, "URL does not contain a hostname."

    # ------------------------------------------------------
    # Reject URLs containing credentials.
    #
    # Example:
    #
    # https://username:password@example.com
    # ------------------------------------------------------

    if parsed.username or parsed.password:
        return False, (
            "URLs containing username or password "
            "credentials are not allowed."
        )

    # ------------------------------------------------------
    # Restrict ports to normal web ports.
    # ------------------------------------------------------

    try:
        port = parsed.port

    except ValueError:
        return False, "URL contains an invalid port."

    if port is not None and port not in {
        80,
        443,
    }:
        return False, (
            "Only standard HTTP and HTTPS ports "
            "are allowed."
        )

    # ------------------------------------------------------
    # Direct IP address check.
    # ------------------------------------------------------

    try:
        ip = ipaddress.ip_address(
            hostname
        )

        if not ip.is_global:
            return False, (
                "Private or non-public IP addresses "
                "are not allowed."
            )

    except ValueError:
        # Hostname is a normal domain.
        pass

    # ------------------------------------------------------
    # DNS resolution check.
    #
    # This catches domains that resolve to localhost,
    # private networks, loopback addresses, etc.
    # ------------------------------------------------------

    if _is_private_or_reserved_host(
        hostname
    ):
        return False, (
            "Hostname resolves to a private or "
            "non-public network address."
        )

    return True, ""


# ----------------------------------------------------------
# Response validation
# ----------------------------------------------------------


def _is_allowed_content_type(
    content_type: str,
) -> bool:
    """
    Check whether the response is a readable web/text
    document rather than a binary download.
    """

    media_type = (
        content_type
        .split(";")[0]
        .strip()
        .lower()
    )

    return media_type in ALLOWED_CONTENT_TYPES


# ----------------------------------------------------------
# Text extraction
# ----------------------------------------------------------


def _extract_text(
    html: str,
) -> tuple[str, str]:
    """
    Extract title and readable text from HTML.

    Returns
    -------
    tuple[str, str]
        title, content
    """

    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    # ------------------------------------------------------
    # Remove elements that normally do not contain useful
    # research content.
    # ------------------------------------------------------

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
        title = soup.title.get_text(
            " ",
            strip=True,
        )

    # ------------------------------------------------------
    # Prefer obvious article/main content when available.
    #
    # If the page does not have these elements, fall back
    # to the complete remaining document text.
    # ------------------------------------------------------

    main_content = soup.find(
        [
            "article",
            "main",
        ]
    )

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

    content = " ".join(
        content.split()
    )

    return title, content


# ----------------------------------------------------------
# Main reader
# ----------------------------------------------------------


def read_webpage(
    url: str,
    timeout: float = DEFAULT_TIMEOUT,
) -> WebPage | None:
    """
    Safely download a web page and extract readable text.

    Parameters
    ----------
    url:
        URL of the page to read.

    timeout:
        Maximum time allowed for each HTTP request.

    Returns
    -------
    WebPage | None
        Extracted page content, or None if the page could
        not be safely or successfully read.

    Notes
    -----
    Redirects are followed manually so that every redirect
    destination can be validated before another request is
    made.
    """

    # ------------------------------------------------------
    # 1. Validate the initial URL.
    # ------------------------------------------------------

    valid, reason = _validate_url(
        url
    )

    if not valid:

        print(
            f"    Rejected URL: {url}"
        )

        print(
            f"    Reason: {reason}"
        )

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

    try:

        with httpx.Client(
            headers=headers,
            timeout=timeout,
            follow_redirects=False,
        ) as client:

            # --------------------------------------------------
            # 2. Follow redirects manually.
            # --------------------------------------------------

            for redirect_number in range(
                MAX_REDIRECTS + 1
            ):

                # Validate every URL before requesting it.
                valid, reason = _validate_url(
                    current_url
                )

                if not valid:

                    print(
                        f"    Rejected redirect: "
                        f"{current_url}"
                    )

                    print(
                        f"    Reason: {reason}"
                    )

                    return None

                response = client.get(
                    current_url
                )

                # --------------------------------------------------
                # Redirect
                # --------------------------------------------------

                if response.is_redirect:

                    if redirect_number >= MAX_REDIRECTS:

                        print(
                            f"    Too many redirects: "
                            f"{url}"
                        )

                        return None

                    location = response.headers.get(
                        "location"
                    )

                    if not location:

                        print(
                            "    Redirect response "
                            "did not contain a location."
                        )

                        return None

                    current_url = urljoin(
                        current_url,
                        location,
                    )

                    continue

                # --------------------------------------------------
                # We have reached the final response.
                # --------------------------------------------------

                response.raise_for_status()

                break

            else:

                return None

    except httpx.TimeoutException as exc:

        print(
            f"    Page request timed out: {url}"
        )

        print(
            f"    Error: {exc}"
        )

        return None

    except httpx.HTTPStatusError as exc:

        print(
            f"    HTTP error while reading: {url}"
        )

        print(
            f"    Status: {exc.response.status_code}"
        )

        return None

    except httpx.HTTPError as exc:

        print(
            f"    Failed to read page: {url}"
        )

        print(
            f"    Error: {exc}"
        )

        return None

    except Exception as exc:

        print(
            f"    Unexpected reader error: {url}"
        )

        print(
            f"    Error: {exc}"
        )

        return None

    # ------------------------------------------------------
    # 3. Validate response content type.
    # ------------------------------------------------------

    content_type = response.headers.get(
        "content-type",
        "",
    )

    if not _is_allowed_content_type(
        content_type
    ):

        print(
            f"    Unsupported content type: "
            f"{content_type}"
        )

        return None

    # ------------------------------------------------------
    # 4. Check Content-Length when provided.
    # ------------------------------------------------------

    content_length = response.headers.get(
        "content-length"
    )

    if content_length:

        try:
            declared_size = int(
                content_length
            )

        except ValueError:

            declared_size = None

        if (
            declared_size is not None
            and declared_size > MAX_RESPONSE_BYTES
        ):

            print(
                "    Response is too large."
            )

            return None

    # ------------------------------------------------------
    # 5. Read the response with a hard size limit.
    #
    # We do not blindly accept an arbitrarily large body.
    # ------------------------------------------------------

    body = bytearray()

    try:

        for chunk in response.iter_bytes(
            chunk_size=64 * 1024
        ):

            body.extend(chunk)

            if len(body) > MAX_RESPONSE_BYTES:

                print(
                    "    Response exceeded the "
                    "maximum allowed size."
                )

                return None

    except httpx.HTTPError as exc:

        print(
            f"    Failed while reading response: "
            f"{current_url}"
        )

        print(
            f"    Error: {exc}"
        )

        return None

    # ------------------------------------------------------
    # 6. Decode the response.
    # ------------------------------------------------------

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

        print(
            f"    Page returned empty content: "
            f"{current_url}"
        )

        return None

    # ------------------------------------------------------
    # 7. Extract readable text.
    # ------------------------------------------------------

    title, content = _extract_text(
        html
    )

    if not content:

        print(
            f"    No readable content found: "
            f"{current_url}"
        )

        return None

    # ------------------------------------------------------
    # 8. Limit content passed further into the system.
    #
    # This protects the LLM context from extremely large
    # webpages.
    # ------------------------------------------------------

    if len(content) > MAX_CONTENT_CHARS:

        print(
            f"    Content truncated from "
            f"{len(content):,} to "
            f"{MAX_CONTENT_CHARS:,} characters."
        )

        content = content[
            :MAX_CONTENT_CHARS
        ]

    return WebPage(
        url=str(response.url),
        title=title,
        content=content,
    )
