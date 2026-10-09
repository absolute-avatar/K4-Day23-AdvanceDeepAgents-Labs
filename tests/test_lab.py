"""Offline regression tests: python -m unittest discover -s tests -v.

Không gọi API thật: mock mạng, thời gian ngủ và sandbox để kiểm tra các nhánh lỗi mà không tốn token.
"""
import json
import asyncio
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
from langchain_core.messages import AIMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

import agents
import check_citations
import finalize_citations
import research
import tools


def fixture():
    sources = [
        {"n": 1, "id": "2501.00001", "url": "https://arxiv.org/abs/2501.00001", "title": "A", "date": "", "source": "arxiv"},
        {"n": 2, "id": "2501.00002", "url": "https://huggingface.co/papers/2501.00002", "title": "B", "date": "", "source": "hf-search"},
        {"n": 3, "id": "C", "url": "https://example.org/c", "title": "C", "date": "", "source": "web"},
    ]
    # Đây là dữ liệu giả chỉ cho test, không được ghi vào reports/ để nộp bài.
    report = "# Test\n\nSupported claims [1][2][3].\n\n## References\n" + "\n".join(
        f"[{s['n']}] {s['title']}. {s['url']} (n.d.)" for s in sources) + "\n"
    return report, sources


class WorkspaceSearchTests(unittest.TestCase):
    def request(self, name, args):
        return ToolCallRequest(tool_call={"name": name, "args": args, "id": "search-1", "type": "tool_call"},
                               tool=None, state={}, runtime=None)

    def test_search_paths_are_normalized_without_mutating_request(self):
        guard = agents.WorkspaceSearchMiddleware()
        for name in ("glob", "ls", "grep"):
            for raw, expected in ((None, "/tmp/work"), ("", "/tmp/work"), ("/", "/tmp/work"),
                                  ("research/notes", agents.NOTES_DIR), (agents.NOTES_DIR, agents.NOTES_DIR),
                                  ("/tmp/work/research/../report", "/tmp/work/report")):
                with self.subTest(name=name, path=raw):
                    args = {"pattern": "*.md"}
                    if raw is not None:
                        args["path"] = raw
                    request = self.request(name, args)
                    handler = Mock(return_value="ok")
                    self.assertEqual(guard.wrap_tool_call(request, handler), "ok")
                    self.assertEqual(handler.call_args.args[0].tool_call["args"],
                                     {**args, "path": expected})
                    self.assertEqual(request.tool_call["args"], args)

    def test_outside_paths_never_reach_handler(self):
        guard = agents.WorkspaceSearchMiddleware()
        for name in ("glob", "ls", "grep"):
            for path in ("/etc", "../", "/tmp/work/../../", "/tmp/work-other", "C:\\temp", 4):
                with self.subTest(name=name, path=path):
                    handler = Mock()
                    result = guard.wrap_tool_call(self.request(name, {"path": path}), handler)
                    handler.assert_not_called()
                    self.assertEqual(result.status, "error")
                    self.assertEqual(result.tool_call_id, "search-1")

    def test_async_paths_and_unrelated_tools(self):
        guard = agents.WorkspaceSearchMiddleware()
        seen = []
        async def handler(request):
            seen.append(request)
            return "ok"
        async def exercise():
            await guard.awrap_tool_call(self.request("glob", {}), handler)
            self.assertEqual(seen[-1].tool_call["args"]["path"], agents.WORKDIR)
            result = await guard.awrap_tool_call(self.request("grep", {"path": "/etc"}), handler)
            self.assertEqual(result.status, "error")
            self.assertEqual(len(seen), 1)
            request = self.request("web_fetch", {"url": "https://example.org"})
            await guard.awrap_tool_call(request, handler)
            self.assertIs(seen[-1], request)
        asyncio.run(exercise())

    @patch("agents.create_deep_agent")
    def test_lead_and_every_subagent_have_guard(self, create):
        agents.build_lead_agent(Mock(), Mock())
        config = create.call_args.kwargs
        for stack in [config["middleware"], *(s["middleware"] for s in config["subagents"])]:
            self.assertEqual(sum(isinstance(m, agents.WorkspaceSearchMiddleware) for m in stack), 1)


class CitationTests(unittest.TestCase):
    def test_hf_url_mismatch_is_reported_before_host_save(self):
        report, sources = fixture()
        original = sources[1]["url"]
        for family in ("hf-daily", "hf-search"):
            for wrong in ("https://arxiv.org/abs/2501.00002", "https://huggingface.co/papers/",
                          "https://huggingface.co/papers/2501.00002?download=1"):
                with self.subTest(family=family, url=wrong):
                    data = [sources[0], dict(sources[1], source=family, url=wrong), sources[2]]
                    # References vẫn khớp sources.json: lỗi cần bắt là họ nguồn/URL, không phải citation.
                    issues = check_citations.check(report.replace(original, wrong), data)
                    self.assertTrue(any("source [2]" in issue and family in issue
                                        and wrong in issue and "expected" in issue for issue in issues))
        # arXiv tìm được qua web vẫn giữ source=web theo GUIDE.
        self.assertIsNone(check_citations.source_url_problem(dict(sources[0], source="web")))

    def test_valid_and_grouped_citations(self):
        report, sources = fixture()
        for citation in ("[1][2][3]", "[1, 2, 3]", "[1-3]", "[1–3]"):
            with self.subTest(citation=citation):
                self.assertEqual(check_citations.check(report.replace("[1][2][3]", citation), sources), [])

    def test_every_required_failure(self):
        report, sources = fixture()
        cases = [
            (report, []), (report, {}), (report, [None]),
            (report, [dict(sources[0], n=True), *sources[1:]]),
            (report, [dict(sources[0], url="file:///tmp/a"), *sources[1:]]),
            (report, [sources[0], dict(sources[1], url=sources[0]["url"]), sources[2]]),
            (report, [sources[0], dict(sources[1], n=1), sources[2]]),
            (report.replace("## References", "## Sources"), sources),
            (report.replace("claims [1]", "claims [9]"), sources),
            (report.replace("claims [1]", "claims "), sources),
            (report.replace("[2] B.", "[1] B."), sources),
            (report.replace("[2] B.", "[9] B."), sources),
            (report.replace("[2] B.", "[2] B. https://example.org/extra"), sources),
            (report.replace("https://example.org/c", "https://example.org/wrong"), sources),
            (report + "\n## Another section\ntext", sources),
        ]
        for candidate, data in cases:
            with self.subTest(candidate=candidate, data=data):
                self.assertTrue(check_citations.check(candidate, data))

    def test_code_and_markdown_links_do_not_count(self):
        report, sources = fixture()
        for ignored in ("`[1]`", "[1](https://example.org)", "```python\n[1]\n```", "~~~\n[1]\n~~~"):
            candidate = report.replace("[1][2][3]", f"\n{ignored}\n[2][3]")
            self.assertIn("source [1] never cited", check_citations.check(candidate, sources))

    def test_references_and_code_heading_do_not_count_as_body(self):
        report, sources = fixture()
        self.assertIn("source [1] never cited", check_citations.check(report.replace("claims [1]", "claims "), sources))
        report = "```\n## References\n[99]\n```\n" + report
        self.assertEqual(check_citations.check(report, sources), [])

    def test_finalizer_and_validator_agree_and_are_idempotent(self):
        _, sources = fixture()
        sources.append(dict(sources[0], n=4))
        text, normalized, errors = finalize_citations.finalize("# Test\nEvidence [4][2-3].", sources)
        self.assertEqual(errors, [])
        self.assertEqual(check_citations.check(text, normalized), [])
        self.assertEqual(finalize_citations.finalize(text, normalized), (text, normalized, []))


class ToolTests(unittest.TestCase):
    @patch("tools.time.sleep")
    @patch("tools.random.uniform", return_value=0.25)
    def test_retry_backoff_cap_and_no_sleep_after_last(self, jitter, sleep):
        fn = Mock(side_effect=tools.RetryableError("temporary"))
        with self.assertRaises(tools.RetryableError):
            tools.with_retry(fn, attempts=4, base=1, cap=3)
        self.assertEqual(fn.call_count, 4)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [1.25, 2.25, 3])

    @patch("tools.time.sleep")
    def test_retry_after_and_non_retryable_error(self, sleep):
        fn = Mock(side_effect=[tools.RetryableError("busy", 8), "ok"])
        self.assertEqual(tools.with_retry(fn, cap=6), "ok")
        sleep.assert_called_once_with(6)
        fn = Mock(side_effect=ValueError("bug"))
        with self.assertRaises(ValueError):
            tools.with_retry(fn)
        self.assertEqual(fn.call_count, 1)

    @patch("tools.httpx.request")
    def test_http_retry_classification(self, request):
        req = httpx.Request("GET", "https://example.org")
        for status in [429, 500, 502, 503, 504]:
            request.return_value = httpx.Response(status, headers={"Retry-After": "7"}, request=req)
            with self.assertRaises(tools.RetryableError) as caught:
                tools._request("GET", str(req.url))
            self.assertEqual(caught.exception.retry_after, 7)
        request.return_value = httpx.Response(401, request=req)
        with self.assertRaises(httpx.HTTPStatusError):
            tools._request("GET", str(req.url))
        request.side_effect = httpx.ConnectTimeout("timeout")
        with self.assertRaises(tools.RetryableError):
            tools._request("GET", str(req.url))

    @patch("tools.time.sleep")
    @patch("tools._request")
    def test_arxiv_normalization_sanitization_and_spacing(self, request, sleep):
        request.return_value = SimpleNamespace(content=b'''<feed xmlns="http://www.w3.org/2005/Atom"><entry>
        <id>http://arxiv.org/abs/2501.00001v2</id><title> A  title\n </title><published>2025-01-01T00:00:00Z</published>
        <summary> A summary </summary></entry></feed>''')
        with patch.object(tools, "_ARXIV_LAST_CALL", None), patch("tools.time.monotonic", return_value=10):
            a = json.loads(tools.arxiv_search.invoke({"query": 'all:"world" AND model', "max_results": 99}))[0]
            tools.arxiv_search.invoke({"query": "world model"})
        self.assertEqual(a["id"], "2501.00001")
        self.assertEqual(a["url"], "https://arxiv.org/abs/2501.00001")
        self.assertEqual(a["title"], "A title")
        self.assertEqual(request.call_args_list[0].kwargs["params"]["search_query"], "all:world AND all:model")
        self.assertEqual(request.call_args_list[0].kwargs["params"]["max_results"], 30)
        sleep.assert_called_with(3.0)
        request.reset_mock()
        self.assertEqual(tools.arxiv_search.invoke({"query": ':: " AND OR'}), "NO RESULTS")
        request.assert_not_called()

    @patch("tools._request")
    def test_hf_mapping_filter_sort_and_ai_summary(self, request):
        items = [{"paper": {"id": str(i), "title": "World model", "summary": "original", "ai_summary": "short",
                            "upvotes": i, "publishedAt": "2026-01-01T00:00:00Z"}} for i in [1, 3, 2]] + [{"paper": {}}]
        request.return_value.json.return_value = items
        daily = json.loads(tools.hf_daily_papers.invoke({"keyword": "WORLD"}))
        self.assertEqual([r["upvotes"] for r in daily], [3, 2, 1])
        self.assertEqual(daily[0]["summary"], "original")
        self.assertEqual(tools.hf_daily_papers.invoke({"keyword": "unrelated"}), "NO RESULTS")
        search = json.loads(tools.hf_search_papers.invoke({"query": "world"}))
        self.assertEqual(search[0]["summary"], "short")

    @patch("tools.time.sleep")
    @patch("tools._request")
    def test_exa_sse_rate_limit_and_key_redaction(self, request, sleep):
        secret = "test-secret-" + "x" * 24
        def response(payload, sse=True):
            text = json.dumps(payload)
            return SimpleNamespace(text=f"event: message\ndata: {text}\n\n" if sse else text, headers={})
        limited = {"result": {"_meta": {"rateLimited": True}, "content": [{"type": "text", "text": "wait"}]}}
        ok = {"result": {"content": [{"type": "text", "text": "evidence " + secret}]}}
        request.side_effect = [response(limited), response(ok, False)]
        with patch.dict("os.environ", {"EXA_API_KEY": secret}):
            result = tools.web_search.invoke({"query": "paper"})
            self.assertNotIn(secret, result)
            self.assertIn("evidence", result)
            self.assertEqual(request.call_count, 2)
            self.assertTrue(request.call_args.kwargs["json"]["params"]["arguments"]["objective"])
            request.side_effect = ValueError("url?exaApiKey=" + secret)
            result = tools.web_fetch.invoke({"url": "https://example.org"})
            self.assertTrue(result.startswith("ERROR:"))
            self.assertNotIn(secret, result)

    @patch("tools._request")
    def test_exa_protocol_error_empty_and_fetch_limit(self, request):
        request.return_value = SimpleNamespace(text=json.dumps({"error": {"code": -32602, "message": "bad input"}}), headers={})
        self.assertTrue(tools.web_search.invoke({"query": "x"}).startswith("ERROR:"))
        request.return_value.text = json.dumps({"result": {"content": []}})
        self.assertEqual(tools.web_search.invoke({"query": "x"}), "NO RESULTS")
        request.return_value.text = json.dumps({"result": {"content": [{"type": "text", "text": "x" * 15000}]}})
        self.assertEqual(len(tools.web_fetch.invoke({"url": "https://example.org"})), 12000)
        self.assertEqual(request.call_args.kwargs["json"]["params"]["arguments"]["urls"], ["https://example.org"])

    def test_exa_rate_limit_flags_are_not_usage_counters(self):
        self.assertFalse(tools._is_rate_limited({"rateLimit": {"remaining": 100}, "rateLimited": False}))
        self.assertTrue(tools._is_rate_limited({"provider": {"rateLimitExceeded": True}}))
        self.assertTrue(tools._is_rate_limited({"rateLimit": {"exceeded": True}}))
        self.assertTrue(tools._is_rate_limited({"code": "RATE_LIMIT_EXCEEDED"}))


class ResearchTests(unittest.TestCase):
    def test_slug_and_token_accounting(self):
        self.assertEqual(research.slugify("../../x"), "x")
        self.assertEqual(research.slugify(" !!! "), "topic")
        self.assertLessEqual(len(research.slugify("a" * 100)), 60)
        msg = AIMessage(content="", tool_calls=[{"name": "task", "args": {}, "id": "1"}],
                        usage_metadata={"input_tokens": 30, "output_tokens": 4, "total_tokens": 34})
        stats = research.summarize([msg, {"tool_calls": [{"name": "execute"}]}], 1.26, "test")
        self.assertEqual(stats["tokens"], {"input": 30, "output": 4})
        self.assertEqual(stats["subagent_calls"], 1)
        self.assertEqual(stats["elapsed_s"], 1.3)

    @patch("research.download")
    def test_save_preserves_bytes_and_rejects_invalid_output(self, download):
        report, sources = fixture()
        raw = json.dumps(sources).encode()
        files = {research.REPORT_PATH: report.encode(), research.SOURCES_PATH: raw}
        download.return_value = files
        messages = [{"tool_calls": [{"name": "task"}] * 3}]
        with tempfile.TemporaryDirectory() as temp:
            path = research.save_outputs(None, "Test", messages, 1, "test", temp)
            self.assertEqual(path.read_bytes(), report.encode())
            self.assertEqual(path.with_suffix(".sources.json").read_bytes(), raw)
            before = {p.name: p.read_bytes() for p in Path(temp).iterdir()}
            for invalid in (None, b"", b"not json", b"{}", b"[]"):
                download.return_value = {**files, research.SOURCES_PATH: invalid}
                with self.assertRaises(RuntimeError):
                    research.save_outputs(None, "Test", messages, 1, "test", temp)
                self.assertEqual({p.name: p.read_bytes() for p in Path(temp).iterdir()}, before)

    @patch("research.download")
    def test_save_rolls_back_write_failure(self, download):
        report, sources = fixture()
        download.return_value = {research.REPORT_PATH: report.encode(), research.SOURCES_PATH: json.dumps(sources).encode()}
        with tempfile.TemporaryDirectory() as temp:
            original_replace = research.os.replace
            count = 0
            def failing_replace(src, dest):
                nonlocal count
                count += 1
                if count == 2:
                    raise OSError("disk error")
                return original_replace(src, dest)
            with patch("research.os.replace", side_effect=failing_replace), self.assertRaises(OSError):
                research.save_outputs(None, "Test", [{"tool_calls": [{"name": "task"}] * 3}], 1, "test", temp)
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_agent_specs_and_limits(self):
        specs = agents.build_subagents()
        self.assertEqual([s["name"] for s in specs], ["researcher", "citation-checker"])
        self.assertEqual(len(specs[0]["tools"]), 5)
        self.assertEqual([t.name for t in specs[1]["tools"]], ["web_fetch"])
        self.assertIsNot(specs[0]["middleware"][0], specs[1]["middleware"][0])

    def test_full_file_pipeline_with_real_citation_scripts(self):
        # Sandbox giả chỉ thay dịch vụ cloud; hai script citation được chạy bằng Python thật.
        # Nhờ đó test bắt được lỗi upload, đường dẫn, thứ tự finalize/check và bytes tải về.
        report, sources = fixture()
        messages = [{"tool_calls": [{"name": "task"}] * 3}]
        with tempfile.TemporaryDirectory() as temp:
            root, output = Path(temp) / "sandbox", Path(temp) / "reports"
            root.mkdir()
            class FileSandbox:
                def local(self, path):
                    return root / Path(path).name

                def upload_files(self, files):
                    for path, content in files:
                        self.local(path).write_bytes(content)

                def download_files(self, paths):
                    return [SimpleNamespace(path=p, content=self.local(p).read_bytes() if self.local(p).exists() else None)
                            for p in paths]

                def execute(self, command):
                    if command.startswith("mkdir"):
                        return SimpleNamespace(exit_code=0, output="")
                    script = command.split()[1]
                    run = subprocess.run([sys.executable, str(self.local(script)), str(self.local(research.REPORT_PATH)),
                                          str(self.local(research.SOURCES_PATH))], capture_output=True, text=True)
                    return SimpleNamespace(exit_code=run.returncode, output=run.stdout + run.stderr)

            backend, closed = FileSandbox(), []
            @contextmanager
            def sandbox():
                try:
                    yield backend
                finally:
                    closed.append(True)
            def invoke(*args, **kwargs):
                # Agent giả ghi draft; finalizer thật phải tự sinh References trong bước cuối.
                backend.local(research.REPORT_PATH).write_text(report.split("## References")[0], encoding="utf-8")
                backend.local(research.SOURCES_PATH).write_text(json.dumps(sources), encoding="utf-8")
                return {"messages": messages}
            real_save = research.save_outputs
            def save(*args):
                return real_save(*args, reports_dir=output)
            with patch("research.open_sandbox", sandbox), patch("research.make_model", return_value=Mock()), \
                    patch("research.build_lead_agent", return_value=SimpleNamespace(invoke=invoke)), \
                    patch("research.save_outputs", side_effect=save):
                self.assertEqual(research.main("pipeline test"), 0)
            self.assertEqual(closed, [True])
            path = output / "pipeline-test.md"
            self.assertEqual(path.read_bytes(), backend.local(research.REPORT_PATH).read_bytes())
            self.assertEqual(path.with_suffix(".sources.json").read_bytes(), backend.local(research.SOURCES_PATH).read_bytes())
            self.assertEqual(check_citations.check(path.read_text(encoding="utf-8"), sources), [])

    @patch("research.download")
    def test_output_rejects_insufficient_delegation_or_false_source_labels(self, download):
        report, sources = fixture()
        with tempfile.TemporaryDirectory() as temp:
            download.return_value = {research.REPORT_PATH: report.encode(), research.SOURCES_PATH: json.dumps(sources).encode()}
            with self.assertRaisesRegex(RuntimeError, "subagent"):
                research.save_outputs(None, "Test", [], 1, "test", temp)
            sources[0]["source"] = "hf-daily"
            download.return_value[research.SOURCES_PATH] = json.dumps(sources).encode()
            with self.assertRaisesRegex(RuntimeError, "huggingface.co/papers"):
                research.save_outputs(None, "Test", [{"tool_calls": [{"name": "task"}] * 3}], 1, "test", temp)
            self.assertEqual(list(Path(temp).iterdir()), [])

    @patch("research.make_model")
    def test_missing_topic_does_not_create_model(self, make_model):
        self.assertEqual(research.main(" "), 2)
        make_model.assert_not_called()

    @patch("research.make_model", return_value=Mock())
    @patch("research.build_lead_agent")
    @patch("research.download")
    @patch("research.upload")
    def test_cleanup_on_agent_error_and_no_output(self, upload, download, build_agent, make_model):
        cleaned = []
        backend = Mock()
        backend.execute.return_value = SimpleNamespace(exit_code=0, output="")
        download.return_value = {research.VALIDATOR_PATH: research.VALIDATOR_SOURCE.read_bytes(),
                                 research.FINALIZER_PATH: research.FINALIZER_SOURCE.read_bytes()}
        build_agent.return_value.invoke.side_effect = RuntimeError("graph failed")
        @contextmanager
        def sandbox():
            try:
                yield backend
            finally:
                cleaned.append(True)
        with patch("research.open_sandbox", sandbox), patch("research.save_outputs") as save:
            self.assertEqual(research.main("test topic"), 1)
            save.assert_not_called()
        self.assertEqual(cleaned, [True])


if __name__ == "__main__":
    unittest.main()
