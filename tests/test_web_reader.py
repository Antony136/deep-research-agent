"""Unit tests for the web page reader."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from app.tools import web_reader


def make_response(
    url="https://example.com/article",
    content=b"<html><title>Example</title><main>Research content</main></html>",
    content_type="text/html; charset=utf-8",
    status_code=200,
    headers=None,
):
    response_headers = {
        "content-type": content_type,
    }

    if headers:
        response_headers.update(headers)

    request = httpx.Request("GET", url)

    response = httpx.Response(
        status_code=status_code,
        headers=response_headers,
        content=content,
        request=request,
    )

    return response


@pytest.fixture
def public_host(monkeypatch):
    """Avoid real DNS lookups during tests."""
    monkeypatch.setattr(
        web_reader,
        "_is_private_or_reserved_host",
        lambda hostname: False,
    )


def test_rejects_non_http_scheme():
    page = web_reader.read_webpage("file:///etc/passwd")

    assert page is None


def test_rejects_private_ip():
    valid, reason = web_reader._validate_url(
        "http://127.0.0.1/"
    )

    assert valid is False
    assert "non-public" in reason.lower()


def test_rejects_nonstandard_port(public_host):
    valid, reason = web_reader._validate_url(
        "https://example.com:8080/"
    )

    assert valid is False
    assert "port" in reason.lower()


def test_extracts_title_and_main_content():
    html = """
    <html>
        <head><title>RAG Guide</title></head>
        <body>
            <nav>Navigation text</nav>
            <main>
                <h1>Retrieval-Augmented Generation</h1>
                <p>Retrieve relevant documents before answering.</p>
                <script>ignore_this()</script>
            </main>
            <footer>Footer text</footer>
        </body>
    </html>
    """

    title, content = web_reader._extract_text(html)

    assert title == "RAG Guide"
    assert "Retrieval-Augmented Generation" in content
    assert "Retrieve relevant documents" in content
    assert "Navigation text" not in content
    assert "Footer text" not in content
    assert "ignore_this" not in content


def test_rejects_unsupported_content_type(public_host):
    response = make_response(
        content=b"%PDF-1.7",
        content_type="application/pdf",
    )

    client = MagicMock()
    client.__enter__.return_value.get.return_value = response

    with patch.object(
        web_reader.httpx,
        "Client",
        return_value=client,
    ):
        page = web_reader.read_webpage(
            "https://example.com/document.pdf"
        )

    assert page is None


def test_reads_html_page(public_host):
    response = make_response()

    client = MagicMock()
    client.__enter__.return_value.get.return_value = response

    with patch.object(
        web_reader.httpx,
        "Client",
        return_value=client,
    ):
        page = web_reader.read_webpage(
            "https://example.com/article"
        )

    assert page is not None
    assert page.title == "Example"
    assert page.content == "Research content"


def test_rejects_response_exceeding_size_limit(public_host, monkeypatch):
    monkeypatch.setattr(
        web_reader,
        "MAX_RESPONSE_BYTES",
        10,
    )

    response = make_response(
        content=b"x" * 20,
        headers={"content-length": "20"},
    )

    client = MagicMock()
    client.__enter__.return_value.get.return_value = response

    with patch.object(
        web_reader.httpx,
        "Client",
        return_value=client,
    ):
        page = web_reader.read_webpage(
            "https://example.com/large"
        )

    assert page is None


def test_revalidates_redirect_destination(public_host):
    redirect_response = make_response(
        status_code=302,
        headers={"location": "http://127.0.0.1/admin"},
    )

    client = MagicMock()
    client.__enter__.return_value.get.return_value = redirect_response

    # Initial hostname is public, but redirect points to loopback.
    def validate(url):
        if "127.0.0.1" in url:
            return False, "Private or non-public IP address."
        return True, ""

    with (
        patch.object(
            web_reader,
            "_validate_url",
            side_effect=validate,
        ),
        patch.object(
            web_reader.httpx,
            "Client",
            return_value=client,
        ),
    ):
        page = web_reader.read_webpage(
            "https://example.com/redirect"
        )

    assert page is None
    assert client.__enter__.return_value.get.call_count == 1
