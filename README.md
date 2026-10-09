# Deep Research Agent (Deep Agents + Sandbox)

Lab dựng một **hệ thống deep research đa tác tử**: người dùng chỉ cần nhập một chủ đề (ví dụ `survey about world model`), hệ thống tự lập kế hoạch, giao việc cho nhiều subagent, tìm tài liệu trên arXiv, Hugging Face và web, rồi viết một **báo cáo có trích dẫn**.

Hình thức: **bài thực hành cá nhân**. Ngôn ngữ lập trình: Python 3.11 trở lên.

## 1. Mục tiêu học tập

Sau lab, bạn có thể:

1. Dựng agent bằng thư viện Deep Agents (LangChain): công cụ (tool), system prompt, subagent, backend.
2. Dùng **sandbox** (Daytona) làm không gian làm việc và nơi chạy mã cho agent; hiểu vì sao khóa API và công cụ mạng phải nằm ở phía host chứ không nằm trong sandbox.
3. Viết công cụ gọi API ngoài **chịu được giới hạn tốc độ** (retry, backoff, jitter, `Retry-After`).
4. Thiết kế quy trình đa tác tử: lead chia nhỏ câu hỏi, giao cho N researcher chạy song song, tổng hợp và kiểm tra trích dẫn.
5. Tạo báo cáo có thể kiểm chứng: mọi khẳng định có `[n]` trỏ tới một nguồn có thật.

## 2. Hệ thống làm gì

```mermaid
flowchart TD
    U["python research.py &quot;survey about world model&quot;"] --> S["open_sandbox() - Daytona"]
    S --> L["Lead agent: write_todos, chia N câu hỏi con"]
    L -->|task x N, song song| R["researcher subagents"]
    R --> T1["arxiv_search"]
    R --> T2["hf_daily_papers / hf_search_papers"]
    R --> T3["web_search / web_fetch (Exa MCP)"]
    R --> N["ghi chú trong sandbox: /tmp/work/research/notes"]
    N --> M["Lead gộp: sources.json + report.md"]
    M --> F["execute: finalize_citations.py (có sẵn)"]
    F --> V["execute: check_citations.py"]
    V --> C["citation-checker subagent kiểm tra mẫu"]
    C --> D["download -> reports/slug.md, .sources.json, .meta.json"]
```

Nguồn dữ liệu:

| Nguồn | Dùng để |
|---|---|
| arXiv API `https://export.arxiv.org/api/query` | Tìm bài theo từ khóa, sắp theo ngày |
| Hugging Face Daily Papers `/api/daily_papers` | Bài đang "trending": upvotes, githubRepo, summary |
| Hugging Face papers search `/api/papers/search?q=` | Tìm bài theo chủ đề |
| Web qua Exa MCP (`web_search_exa`, `web_fetch_exa`) | Blog, survey, trang dự án, nội dung đầy đủ của một URL |

## 3. Cấu trúc thư mục

```
Lab/
├── README.md  GUIDE.md  RUBRIC.md  REPORT_TEMPLATE.md   tài liệu
├── topics.md                 5 chủ đề cần chạy
├── requirements.txt  .env.example  .gitignore
├── model.py                  CÓ SẴN - không sửa: tạo mô hình LLM từ biến môi trường
├── sandbox.py                CÓ SẴN - không sửa: sandbox Daytona (hoặc Docker), upload, download
├── self_check.py             CÓ SẴN - không sửa: tự kiểm tra trước khi nộp (python self_check.py)
├── finalize_citations.py     CÓ SẴN - không sửa: script chạy trong sandbox, tự sinh `## References` và đánh số lại trích dẫn
├── tools.py                  SINH VIÊN CÀI ĐẶT: retry + 5 công cụ nguồn dữ liệu
├── agents.py                 SINH VIÊN CÀI ĐẶT: prompt, subagent, lead agent
├── research.py               SINH VIÊN CÀI ĐẶT: script chính
├── check_citations.py        SINH VIÊN CÀI ĐẶT: kiểm tra trích dẫn, chạy TRONG sandbox
└── reports/                  báo cáo sinh ra (bạn commit vào repo nộp)
```

Các tệp "SINH VIÊN CÀI ĐẶT" đã được triển khai theo các phần bắt buộc của `GUIDE.md`. Mọi `TODO n`, comment và pseudo-code gốc được giữ lại để đối chiếu; comment tiếng Việt giải thích cách xử lý. Trong `agents.py`, các chuỗi `*_PROMPT_GUIDANCE` giữ đề bài gốc, còn các chuỗi `*_PROMPT` là hướng dẫn thực sự gửi cho model. Phần mở rộng (GUIDE 6) không được triển khai.

## 4. Cài đặt

Trên **Windows / PowerShell**, chạy từ thư mục repo (nếu đã có `.venv` thì bỏ lệnh tạo môi trường):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Điền khóa trong `.env`. Ví dụ chọn `LAB_MODEL=openai:gpt-6-luna` và `OPENAI_API_KEY`, cùng `DAYTONA_API_KEY` và `EXA_API_KEY`. Với các bản thư viện được ghim trong repo, `langchain-openai` tự dùng Responses API khi GPT-6 gọi công cụ; không cần sửa `model.py`. Model/provider khác phải hỗ trợ tool calling và có gói tích hợp LangChain tương ứng.

Trên **Linux / macOS**:

```bash
python3 -m venv .venv && source .venv/bin/activate      # Python 3.11+
pip install -r requirements.txt
cp .env.example .env                                     # rồi điền khóa CỦA BẠN
```

Bạn cần ba loại khóa (điền vào `.env`, **không bao giờ commit** `.env`):

| Khóa | Lấy ở đâu | Ghi chú |
|---|---|---|
| LLM (`LAB_MODEL` + khóa nhà cung cấp) | Nhà cung cấp bạn chọn (OpenAI, Anthropic, Google, OpenRouter, Ollama...) | Mô hình **phải hỗ trợ tool calling**. Chép tên mô hình từ tài liệu của nhà cung cấp. |
| `DAYTONA_API_KEY` | https://app.daytona.io | Kiểm tra gói miễn phí / credit hiện hành. Không có tài khoản hoặc hết credit: đặt `SANDBOX=docker` để chạy sandbox trong container Docker cục bộ (xem `.env.example`). |
| `EXA_API_KEY` (khuyến nghị) | https://dashboard.exa.ai/api-keys | Có thể chạy không khóa, nhưng bản miễn phí của MCP bị giới hạn tốc độ rất nhanh. |

## 5. Làm bài

Làm theo thứ tự (chi tiết trong `GUIDE.md`):

1. `check_citations.py`: khởi động nhẹ, thuần Python.
2. `tools.py`: viết `with_retry` và 5 công cụ. Thử riêng từng công cụ: `python tools.py`.
3. `agents.py`: viết prompt, subagent và lead agent.
4. `research.py`: ghép tất cả; chạy một chủ đề:

```bash
python research.py "survey about world model"
```

Kết quả nằm ở `reports/survey-about-world-model.md` cùng `.sources.json` và `.meta.json`.

Trên PowerShell, dùng `.\.venv\Scripts\python.exe` thay cho `python`. Kiểm tra offline trước khi gọi dịch vụ thật:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe tools.py
.\.venv\Scripts\python.exe research.py "survey about world model"
.\.venv\Scripts\python.exe self_check.py
```

Các test dùng dữ liệu giả trong thư mục tạm, không gọi API và không tạo báo cáo nộp bài. `tools.py` gọi mạng thật và trả mã thoát 1 nếu một nguồn lỗi/rỗng. `research.py` in tên công cụ để theo dõi tiến độ, kiểm tra lại script được upload, chạy finalizer/validator trong sandbox trước khi tải về và chỉ lưu đầu ra hợp lệ. Mã thoát: 0 thành công, 1 chạy thất bại, 2 thiếu chủ đề.

Để chạy đủ năm chủ đề từ `topics.md` trên PowerShell, dừng ngay nếu có lần chạy thất bại:

```powershell
Get-Content topics.md | ForEach-Object {
    if ($_ -match '^\d+\. (.+)$') {
        & .\.venv\Scripts\python.exe research.py $Matches[1]
        if ($LASTEXITCODE -ne 0) { throw 'Research failed; fix the error before continuing.' }
    }
}
.\.venv\Scripts\python.exe self_check.py
```

Đọc kết quả: `.md` là báo cáo tiếng Anh; `.sources.json` ánh xạ số trích dẫn sang URL; `.meta.json` ghi topic, model, thời gian, tool calls, số nguồn và họ nguồn. Hai tệp nội dung được lưu nguyên bytes tải từ sandbox; không chỉnh sửa báo cáo trên host. Token trong metadata chỉ tính lead, không phải tổng chi phí hệ thống. Giới hạn bắt buộc: lead 150 model calls / 300 tool calls, researcher 40 / 60, citation-checker 20 / 30; graph lead có `recursion_limit=1000`. Helper `general-purpose` mặc định được giới hạn 2 / 2 và không tham gia nghiên cứu.

## 6. Chủ đề và nộp bài

- Chạy đủ **5 chủ đề** trong [`topics.md`](topics.md), mỗi chủ đề một lần.
- Commit mã nguồn và toàn bộ `reports/`, đẩy lên một **public repo** GitHub và nộp link.
- Kiểm tra trước khi nộp: chạy **`python self_check.py`** (không tốn token): nó kiểm tra đủ 5 báo cáo, `meta.json`, trích dẫn bằng `check_citations.py` của bạn, và không có `.env`/khóa nào trong git.
- Cách chấm: xem [`RUBRIC.md`](RUBRIC.md).

## 7. Thời gian, chi phí và an toàn

- Dùng một mô hình **rẻ nhưng hỗ trợ tool calling**, và **đặt giới hạn** (số lần gọi mô hình/công cụ cho lead và subagent, `recursion_limit`): một prompt hỏng có thể khiến agent lặp rất lâu. Đây là hạng mục 2.5 của `RUBRIC.md`.
- Kết quả có tính ngẫu nhiên: cùng một mã có thể cho báo cáo hợp lệ ở lần này và trích dẫn lỗi ở lần sau. Hãy sửa **prompt và mã**, không sửa tay báo cáo.

- Mỗi lần chạy tốn token LLM và thời gian sandbox. `tokens` trong `meta.json` chỉ đếm tin nhắn của lead, chưa gồm subagent, nên chi phí thật cao hơn. `open_sandbox()` luôn dừng và xóa sandbox khi kết thúc, kể cả khi lỗi. Đừng bỏ qua nó.
- **Không đưa bí mật vào sandbox.** Sandbox không ngăn được prompt injection hay việc đẩy dữ liệu ra mạng; một trang web độc hại có thể khiến agent chạy lệnh bên trong sandbox. Vì vậy mọi công cụ gọi mạng và mọi khóa ở lại phía host.
- Nội dung lấy từ web là **dữ liệu không đáng tin**: agent không được làm theo chỉ dẫn nằm trong đó.
