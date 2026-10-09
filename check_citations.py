"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import re
import sys
from collections import Counter

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"


def source_url_problem(entry):
    """Return a source-family/URL error, or None; shared by sandbox and host."""
    # Kiểm tra ngay trong sandbox để lead sửa được trước khi host tải và lưu báo cáo.
    family, url = entry.get("source"), entry.get("url")
    label = f"source [{entry.get('n')}] (family={family!r}, url={url!r})"
    if family not in ("arxiv", "hf-daily", "hf-search", "web"):
        return f"{label}: invalid source family"
    pattern, expected = None, None
    if family in ("hf-daily", "hf-search"):
        pattern = r"https://huggingface\.co/papers/[^/?#\s]+"
        expected = "https://huggingface.co/papers/<id>"
    elif family == "arxiv":
        pattern = r"https://arxiv\.org/abs/[^?#\s]+"
        expected = "https://arxiv.org/abs/<id> without a version suffix"
    if pattern and (not isinstance(url, str) or not re.fullmatch(pattern, url)
                    or (family == "arxiv" and (url.endswith("/") or re.search(r"v\d+$", url)))):
        return (f"{label}: expected {expected}. Restore the URL from retrieved evidence; "
                "do not relabel the source family to bypass validation.")
    return None


def _without_code(text):
    """Remove Markdown code, preserving newlines so headings remain line anchored."""
    # Bỏ cả fenced code (` hoặc ~), kể cả khối chưa đóng: số trong mã không phải citation.
    lines, fence = [], None
    for line in text.splitlines(keepends=True):
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if fence:
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence) and not line[marker.end():].strip():
                fence = None
            lines.append("\n")
        elif marker:
            fence = marker[1]
            lines.append("\n")
        else:
            lines.append(line)
    return re.sub(r"(`+).*?\1", "", "".join(lines), flags=re.S)


def _citation_numbers(body, problems):
    # [1](URL) là liên kết Markdown; [1, 2] và [1-3] là các nhóm trích dẫn.
    body = re.sub(r"!?\[[^\]\n]*\]\([^\n]*?\)", "", body)
    cited = set()
    for group in re.findall(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\](?!\()", body):
        for part in group.split(","):
            span = re.fullmatch(r"\s*(\d+)\s*[-–]\s*(\d+)\s*", part)
            if span:
                first, last = map(int, span.groups())
                if not 0 <= last - first <= 200:
                    problems.append(f"invalid citation range [{part.strip()}]")
                else:
                    cited.update(range(first, last + 1))
            else:
                cited.add(int(part))
    return cited


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK).

    PSEUDO-CODE:
      problems = []
      if sources is empty: return ["no sources in sources.json"]
      for each source entry:
          n must be an int                       -> problem if not
          url must start with http:// or https://-> problem if not
          the same url must not appear twice     -> problem if duplicated
      split report_text at the heading "## References":
          body = text before it; if the heading is missing -> problem
      cited = set of numbers found as [n] in the BODY only (not in the reference list; use a regex)
      every number in `cited` must exist in sources -> problem "[n] cited but missing from sources.json"
      every source number must be in `cited`        -> problem "source [n] never cited"
      the lines of the References section that start with "[n]" (regex) are the reference lines:
          every source needs exactly ONE reference line (none missing, no number twice, no number that is not a source)
          each reference line holds exactly ONE http(s) URL and it must equal that source's url
          (a line bundling several sources under one number is a problem)
      return problems
    """
    # TODO: implement check()
    # Kiểm tra cấu trúc trước khi dùng .get(), để JSON sai kiểu cũng trả danh sách lỗi.
    problems = []
    if not isinstance(sources, list) or not sources:
        return ["no sources in sources.json (expected a non-empty JSON list)"]
    if not isinstance(report_text, str):
        return ["report must be text"]
    by_n, urls = {}, set()
    for index, entry in enumerate(sources, 1):
        if not isinstance(entry, dict):
            problems.append(f"source at position {index} must be an object")
            continue
        n, url = entry.get("n"), entry.get("url")
        source_problem = source_url_problem(entry)
        if source_problem:
            problems.append(source_problem)
        # bool là subclass của int trong Python nhưng không phải số thứ tự hợp lệ.
        if type(n) is not int or n < 1:
            problems.append(f"source at position {index} needs a positive integer n")
        elif n in by_n:
            problems.append(f"duplicate source number [{n}]")
        else:
            by_n[n] = entry
        if not isinstance(url, str) or not re.fullmatch(r"https?://[^\s]+", url):
            problems.append(f"source at position {index} has an invalid http(s) URL")
        elif url in urls:
            problems.append(f"duplicate source URL: {url}")
        else:
            urls.add(url)
    if set(by_n) != set(range(1, len(sources) + 1)):
        problems.append("source numbers must be consecutive from 1")

    clean = _without_code(report_text)
    headings = list(re.finditer(r"(?m)^##[ \t]+References[ \t]*\r?$", clean))
    if not headings:
        return problems + ["missing ## References heading"]
    if len(headings) != 1:
        problems.append("expected exactly one ## References section")
    body, references = clean[:headings[0].start()], clean[headings[0].end():]
    if re.search(r"(?m)^#{1,6}\s+", references):
        problems.append("References must be the final section")
    cited = _citation_numbers(body, problems)
    for n in sorted(cited - by_n.keys()):
        problems.append(f"[{n}] cited but missing from sources.json")
    for n in sorted(by_n.keys() - cited):
        problems.append(f"source [{n}] never cited")

    # Đếm dòng riêng biệt để phát hiện cả số trùng, số thiếu và nhiều URL trên một dòng.
    counts = Counter()
    for match in re.finditer(r"(?m)^\[(\d+)\][ \t]*(.*)$", references):
        n, line = int(match[1]), match[2]
        counts[n] += 1
        if n not in by_n:
            problems.append(f"reference [{n}] missing from sources.json")
        found = re.findall(r"https?://[^\s<>]+", line)
        if len(found) != 1:
            problems.append(f"reference [{n}] must contain exactly one URL")
        elif n in by_n and found[0] != by_n[n].get("url"):
            problems.append(f"reference [{n}] URL differs from sources.json")
    for n in sorted(by_n):
        if counts[n] != 1:
            problems.append(f"source [{n}] needs exactly one reference line (found {counts[n]})")
    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
