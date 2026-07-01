# Benchmark Methodology

Placeholder for Step 5–6. Definitions live in `CLAUDE.md`:

- **Model load time** — initialization until ready for tokenized input
- **Warm TTFT** — tokenization/request prep to first generated token (excluding load)
- **Prefill throughput** — prompt tokens / prompt evaluation time
- **Decode throughput** — tokens after the first / decode interval
- **Peak RSS** — maximum resident set during the run
- **Memory stability** — RSS trend across repeated warm requests

No benchmark numbers are recorded until instrumentation and repeatable harnesses exist.
