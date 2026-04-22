---
name: EvalHub references and integration plan
description: EvalHub is Red Hat's AI evaluation platform. User wants to optionally integrate via BYOF adapter after standalone eval works.
type: reference
originSessionId: 30228c4f-4f55-4462-87a1-7dd56a52355f
---
EvalHub (Red Hat AI evaluation platform):
- Docs: https://eval-hub.github.io
- Server: https://github.com/eval-hub/eval-hub
- Python SDK: https://github.com/eval-hub/eval-hub-sdk
- Contrib adapters: https://github.com/eval-hub/eval-hub-contrib
- Blog posts PDF in repo: `docs/WC - EvalHub Blogs for Red Hat Summit 2026.pdf`

Integration approach: build standalone evaluation modules first, wrap in EvalHub BYOF adapter later (~30 lines). BYOF pattern: subclass `FrameworkAdapter`, implement `run_benchmark_job()`, package as container. Local mode: `EVALHUB_MODE=local`, no Kubernetes needed.

The SDK package is `eval-hub-sdk` — install via `uv add eval-hub-sdk` or from GitHub. Not yet confirmed on PyPI as of 2026-04-22.
