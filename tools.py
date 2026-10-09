"""tools.py - STUDENT IMPLEMENTS.  Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json  # noqa: F401
import os  # noqa: F401
import random
import re
import sys
import time  # noqa: F401
import xml.etree.ElementTree  # noqa: F401  (arXiv answers with Atom XML)
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from threading import Lock
from urllib.parse import quote, quote_plus, urlsplit

import httpx  # noqa: F401
from langchain_core.tools import tool
from dotenv import load_dotenv

# Nạp khóa trên host cả khi chạy riêng `python tools.py`; không chuyển .env sang sandbox.
load_dotenv(Path(__file__).with_name(".env"))

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"
RETRY_STATUSES = {429, 500, 502, 503, 504}
_ARXIV_LOCK = Lock()
_ARXIV_LAST_CALL = None


def redact_secrets(text):
    """Keep provider credentials out of tool responses and console errors."""
    text = str(text)
    # URL trong HTTPStatusError có thể chứa exaApiKey, kể cả dạng đã URL-encode.
    for name, value in os.environ.items():
        if value and len(value) >= 6 and re.search(r"KEY|TOKEN|SECRET|PASSWORD", name, re.I):
            for encoded in {value, quote(value, safe=""), quote_plus(value)}:
                text = text.replace(encoded, "[REDACTED]")
    return re.sub(r"(?i)(exaApiKey=)[^\s&\"'<>]+", r"\1[REDACTED]", text)


def _error(exc):
    return f"ERROR: {type(exc).__name__}: {redact_secrets(exc)}"


def _retry_after(value):
    """Accept Retry-After seconds, and HTTP dates used by some gateways."""
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (ValueError, TypeError):
        try:
            return max(0.0, (parsedate_to_datetime(str(value)) - datetime.now(timezone.utc)).total_seconds())
        except (ValueError, TypeError, OverflowError):
            return None


def _request(method, url, **kwargs):
    """One network attempt; the caller owns retry policy and result parsing."""
    try:
        response = httpx.request(method, url, timeout=45.0, follow_redirects=True, **kwargs)
    except httpx.TransportError as exc:
        raise RetryableError(redact_secrets(exc)) from exc
    if response.status_code in RETRY_STATUSES:
        raise RetryableError(f"HTTP {response.status_code} from {urlsplit(url).hostname}",
                             _retry_after(response.headers.get("Retry-After")))
    response.raise_for_status()
    return response


def _clean(value):
    return " ".join(str(value or "").split())


def _records_text(records):
    return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


# ---- TODO 1: retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn(); when it raises RetryableError, wait and call it again.

    PSEUDO-CODE:
      for attempt in 0 .. attempts-1:
          try: return fn()
          except RetryableError as e:
              if this was the last attempt: raise
              delay = e.retry_after if the server told us, else exponential backoff base * 2**attempt
              cap the delay at `cap` seconds; add random jitter to the exponential case
              sleep(delay)
    Use it to wrap EVERY network call below. Also treat these as retryable: HTTP 429/500/502/503/504,
    httpx.TransportError (timeouts, connection resets). Read the Retry-After header when present.
    """
    # TODO 1: with_retry
    if attempts < 1 or base < 0 or cap < 0:
        raise ValueError("attempts must be positive; base and cap must be non-negative")
    for attempt in range(attempts):
        try:
            return fn()
        except RetryableError as exc:
            if attempt == attempts - 1:
                raise  # Lần cuối trả lỗi ngay, không ngủ một khoảng vô ích.
            if exc.retry_after is not None:
                delay = max(0.0, float(exc.retry_after))
            else:
                delay = base * 2 ** attempt + random.uniform(0.0, base)
            time.sleep(min(cap, delay))  # Chặn cả jitter để thời gian ngủ không vượt cap.


# ---- TODO 2: arXiv ----
@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first. Returns a JSON list of {id, url, published, title, summary}."""
    # PSEUDO-CODE:
    #   keep only word characters of `query` -> terms; no terms -> "NO RESULTS" (do not call the network)
    #   respect arXiv etiquette: at least 3 seconds between two arXiv calls (remember the time of the last call)
    #   GET ARXIV_URL params: search_query="all:t1 AND all:t2 ...", sortBy=submittedDate, sortOrder=descending,
    #       max_results=clamp(max_results, 1, 30)           (wrap in with_retry)
    #   parse the Atom XML: each <entry> -> {id (last part of <id> after /abs/), url, published[:10], title, summary}
    #       collapse whitespace/newlines in title and summary; cut summary to ~600 chars
    #   no entries -> "NO RESULTS"; else json.dumps(records, ensure_ascii=False)
    #   any exception -> "ERROR: <type>: <message>"
    # TODO 2: arxiv_search
    try:
        # Bỏ cú pháp query do LLM tự thêm; chỉ tạo lại các mệnh đề all:<keyword> an toàn.
        terms = [word for word in re.findall(r"[^\W_]+(?:-[^\W_]+)*", query, re.UNICODE)
                 if word.lower() not in {"and", "or", "not", "all", "ti", "au", "abs", "cat"}]
        if not terms:
            return "NO RESULTS"
        params = {"search_query": " AND ".join(f"all:{word}" for word in terms), "start": 0,
                  "sortBy": "submittedDate", "sortOrder": "descending", "max_results": max(1, min(30, max_results))}

        def fetch():
            global _ARXIV_LAST_CALL
            # Lock dùng chung giữa các researcher; mỗi lần retry cũng phải qua giới hạn này.
            with _ARXIV_LOCK:
                now = time.monotonic()
                if _ARXIV_LAST_CALL is not None:
                    time.sleep(max(0.0, 3.0 - (now - _ARXIV_LAST_CALL)))
                _ARXIV_LAST_CALL = time.monotonic()
                return _request("GET", ARXIV_URL, params=params)

        response = with_retry(fetch, attempts=7, cap=60.0)
        root = xml.etree.ElementTree.fromstring(response.content)
        ns, records = {"a": "http://www.w3.org/2005/Atom"}, []
        for entry in root.findall("a:entry", ns):
            raw_id = entry.findtext("a:id", "", ns)
            paper_id = re.sub(r"v\d+$", "", raw_id.partition("/abs/")[2])
            if not paper_id:
                continue
            records.append({"id": paper_id, "url": f"https://arxiv.org/abs/{paper_id}",
                            "published": entry.findtext("a:published", "", ns)[:10],
                            "title": _clean(entry.findtext("a:title", "", ns)),
                            "summary": _clean(entry.findtext("a:summary", "", ns))[:600]})
        return _records_text(records)
    except Exception as exc:
        return _error(exc)


def _hf_records(items, *, prefer_ai=False):
    """Normalize the shared HF API envelope without inventing missing metadata."""
    if not isinstance(items, list):
        raise ValueError("Hugging Face response must be a list")
    records = []
    for item in items:
        paper = item.get("paper") if isinstance(item, dict) else None
        if not isinstance(paper, dict) or not paper.get("id"):
            continue
        summary = (paper.get("ai_summary") or item.get("ai_summary")) if prefer_ai else None
        records.append({"id": str(paper["id"]), "url": f"https://huggingface.co/papers/{paper['id']}",
                        "published": str(paper.get("publishedAt") or item.get("publishedAt") or "")[:10],
                        "title": _clean(paper.get("title") or item.get("title")),
                        "summary": _clean(summary or paper.get("summary") or item.get("summary"))[:600],
                        "upvotes": paper.get("upvotes") or 0, "github": paper.get("githubRepo") or "",
                        "stars": paper.get("githubStars") or 0})
    return records


# ---- TODO 3: Hugging Face ----
@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    # PSEUDO-CODE:
    #   GET HF_DAILY_URL params: limit (clamp 1..100) and date (only when given)      (with_retry)
    #   response = list of items {"paper": {id, title, summary, upvotes, githubRepo, githubStars, publishedAt}, ...}
    #   map every item to the record shape above (skip items without paper.id); url = https://huggingface.co/papers/<id>
    #   keyword -> keep records whose title+summary contains it (case-insensitive); sort by upvotes descending
    # TODO 3: hf_daily_papers
    try:
        params = {"limit": max(1, min(100, limit))}
        if date:
            params["date"] = date
        response = with_retry(lambda: _request("GET", HF_DAILY_URL, params=params))
        records = _hf_records(response.json())
        if keyword.strip():
            # Lọc tại host vì endpoint Daily Papers không hỗ trợ truy vấn theo chủ đề.
            needle = keyword.strip().casefold()
            records = [r for r in records if needle in (r["title"] + " " + r["summary"]).casefold()]
        return _records_text(sorted(records, key=lambda r: r["upvotes"], reverse=True))
    except Exception as exc:
        return _error(exc)


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}."""
    # PSEUDO-CODE:
    #   GET HF_SEARCH_URL params: q=query, limit (clamp 1..50)                         (with_retry)
    #   same item shape as the daily endpoint; prefer paper["ai_summary"] over paper["summary"] when present
    # TODO 3: hf_search_papers
    try:
        if not query.strip():
            return "NO RESULTS"
        params = {"q": query.strip(), "limit": max(1, min(50, limit))}
        response = with_retry(lambda: _request("GET", HF_SEARCH_URL, params=params))
        return _records_text(_hf_records(response.json(), prefer_ai=True))
    except Exception as exc:
        return _error(exc)


def _rpc_payload(text):
    """Read JSON-RPC from plain JSON or SSE events, including multi-line data fields."""
    try:
        return json.loads(text)
    except ValueError:
        pass
    payloads = []
    for event in re.split(r"\r?\n\r?\n", text):
        data = "\n".join(line[5:].lstrip() for line in event.splitlines() if line.startswith("data:"))
        if data and data != "[DONE]":
            payloads.append(json.loads(data))
    for payload in reversed(payloads):
        if isinstance(payload, dict) and ("result" in payload or "error" in payload):
            return payload
    raise ValueError("Exa returned no JSON-RPC result")


def _is_rate_limited(value):
    """Find an explicit rate-limit flag/code even in a nested MCP _meta object."""
    if isinstance(value, dict):
        for key, item in value.items():
            compact = re.sub(r"[^a-z]", "", str(key).lower())
            if "ratelimit" in compact or "throttl" in compact:
                # Counter như rateLimit.remaining=100 không phải cờ báo lỗi.
                if item is True or (isinstance(item, str) and item.lower() in {"true", "exceeded", "limited", "blocked"}):
                    return True
                if isinstance(item, dict) and any(item.get(flag) is True for flag in ("exceeded", "limited", "blocked")):
                    return True
            if _is_rate_limited(item):
                return True
    elif isinstance(value, list):
        return any(_is_rate_limited(item) for item in value)
    elif isinstance(value, str):
        return bool(re.search(r"rate[_ -]?limit(?:ed|[_ -]?(?:exceeded|error|reached))|too many requests", value, re.I))
    return False


def _exa_call(name, arguments):
    """Run one MCP call with retry around HTTP AND application-level errors."""
    def fetch():
        key = os.getenv("EXA_API_KEY", "").strip()
        response = _request("POST", EXA_URL, params={"exaApiKey": key} if key else {},
                            headers={"Accept": "application/json, text/event-stream"},
                            json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                  "params": {"name": name, "arguments": arguments}})
        payload = _rpc_payload(response.text)
        if not isinstance(payload, dict):
            raise ValueError("Exa JSON-RPC response must be an object")
        if "error" in payload:
            error = payload["error"]
            if _is_rate_limited(error) or (isinstance(error, dict) and error.get("code") == 429):
                raise RetryableError("Exa JSON-RPC rate limit")
            raise ValueError(f"Exa JSON-RPC error: {error}")
        result = payload.get("result")
        if not isinstance(result, dict):
            raise ValueError("Exa response is missing result")
        content = "\n".join(item.get("text", "") for item in result.get("content", [])
                            if isinstance(item, dict) and item.get("type") == "text").strip()
        meta = result.get("_meta", {})
        # HTTP 200 vẫn có thể là rate limit; tuyệt đối không đưa thông báo này vào ghi chú.
        if _is_rate_limited(meta) or (result.get("isError") and _is_rate_limited(content)):
            raise RetryableError("Exa MCP rate limit", _retry_after(response.headers.get("Retry-After")))
        if result.get("isError"):
            raise ValueError(f"Exa tool error: {content}")
        return redact_secrets(content) if content else "NO RESULTS"
    return with_retry(fetch, attempts=7, base=2.0, cap=60.0)


# ---- TODO 4: web search / fetch through the Exa MCP endpoint ----
@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa). Describe the ideal page in natural language. Returns clean text of the top results with URLs."""
    # PSEUDO-CODE:
    #   call the MCP tool "web_search_exa" with arguments {query, objective, numResults}
    #       (objective is REQUIRED by Exa: when empty, build one from the query)
    #   see GUIDE.md part 1.4 for how to call an MCP server over plain HTTP (JSON-RPC "tools/call") and read the answer
    #   read optional env EXA_API_KEY; when present it is sent to the Exa endpoint.
    #       (see GUIDE.md 1.4 for where it goes) => the key then appears in exception text: redact it before returning "ERROR: ..."
    #   WATCH OUT: read GUIDE.md 1.4 about how Exa signals "rate limited" on the free tier, and retry on it
    # TODO 4: web_search
    try:
        if not query.strip():
            return "NO RESULTS"
        return _exa_call("web_search_exa", {"query": query.strip(),
                         "objective": objective.strip() or f"Find primary sources about {query.strip()}",
                         "numResults": max(1, min(10, num_results))})
    except Exception as exc:
        return _error(exc)


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated."""
    # PSEUDO-CODE: MCP tool "web_fetch_exa" with arguments {"urls": [url]}; truncate the text to ~12000 chars
    # TODO 4: web_fetch
    try:
        if urlsplit(url).scheme not in {"http", "https"} or not urlsplit(url).hostname:
            raise ValueError("url must be an absolute http(s) URL")
        # maxCharacters yêu cầu Exa trả đủ văn bản; cắt lần nữa để giữ ngữ cảnh gọn.
        return _exa_call("web_fetch_exa", {"urls": [url], "maxCharacters": 12000})[:12000]
    except Exception as exc:
        return _error(exc)


# ---- TODO 5: registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    # Windows console cũng cần in được tiêu đề/tóm tắt Unicode từ nguồn thật.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    failed = False
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        try:
            print(f"Testing {name}...", flush=True)
            result = fn.invoke(args)
            print(f"== {name}\n{result[:400]}\n", flush=True)
            failed |= result.startswith("ERROR:") or result == "NO RESULTS"
        except NotImplementedError as exc:
            print(f"== {name}: not implemented yet ({exc})\n")
            failed = True
    sys.exit(1 if failed else 0)
