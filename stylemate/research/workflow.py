"""Two-query, four-page research; external passages remain unreviewed data."""
import html
import re
from html.parser import HTMLParser
from urllib.parse import parse_qs, unquote, urlsplit

import requests

from stylemate.agent.policy import check_request_deadline, remaining_request_seconds

DOMAINS = {"woolmark.com", "gore-tex.com", "patagonia.com", "arcteryx.com", "uniqlo.com"}


class PageText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts, self.hidden = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "nav", "footer"}:
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style", "nav", "footer"}:
            self.hidden = max(0, self.hidden - 1)

    def handle_data(self, data):
        if not self.hidden and data.strip():
            self.parts.append(data.strip())


def allowed_source(url):
    parsed = urlsplit(url)
    host = parsed.hostname or ""
    return parsed.scheme == "https" and not parsed.username and not parsed.password and parsed.port in {None, 443} and any(
        host == domain or host.endswith("." + domain) for domain in DOMAINS)


def read_page(session, url):
    check_request_deadline()
    remaining = remaining_request_seconds()
    timeout = min(5, remaining) if remaining is not None else 5
    if timeout <= 0:
        raise TimeoutError()
    with session.get(url, timeout=(min(2, timeout), timeout), allow_redirects=False, stream=True) as response:
        if response.status_code != 200 or "text/html" not in response.headers.get("Content-Type", ""):
            return ""
        data = bytearray()
        for chunk in response.iter_content(8192):
            check_request_deadline()
            data.extend(chunk)
            if len(data) > 500_000:
                return ""
        return data.decode(response.encoding or "utf-8", errors="replace")


def deep_research(arguments, context):
    query = arguments["query"]
    session = requests.Session()
    urls, passages, failures = [], [], 0
    variants = [query, query + " care instructions official"]
    for variant in variants:
        try:
            page = read_page(session, "https://html.duckduckgo.com/html/?q=" + requests.utils.quote(variant))
            for link in re.findall(r'href=["\']([^"\']+)', page):
                link = html.unescape(link)
                if link.startswith("//"):
                    link = "https:" + link
                target = parse_qs(urlsplit(link).query).get("uddg", [link])[0]
                target = unquote(target)
                if allowed_source(target) and target not in urls:
                    urls.append(target)
        except Exception:
            failures += 1
    for url in urls[:4]:
        try:
            page = read_page(session, url)
            parser = PageText()
            parser.feed(page)
            content = " ".join(parser.parts)
            if content:
                # Pick a query-bearing window; no model-generated claims are
                # promoted to evidence or inserted in the reviewed corpus.
                terms = re.findall(r"[A-Za-z]{3,}|[\u4e00-\u9fff]{2,4}", query)
                offset = next((content.lower().find(term.lower()) for term in terms if term.lower() in content.lower()), 0)
                passages.append({"source_url": url, "snippet": content[max(0, offset - 80):offset + 720],
                                 "review_status": "web_unreviewed"})
        except Exception:
            failures += 1
    return {"message": "以下为联网检索资料，尚未经人工审核；请核对产品范围与洗标。" if passages else "联网研究未获得可核验资料，请补充品牌、款号或洗标。",
            "results": passages, "sources": [{"id": str(i), "title": urlsplit(item["source_url"]).hostname,
                                               "url": item["source_url"]} for i, item in enumerate(passages)],
            "support_status": "partial" if passages else "not_found",
            "research_actions": {"search_requests": 2, "page_requests": min(4, len(urls)), "failures": failures}}
