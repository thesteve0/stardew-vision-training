---
name: Never use pip
description: User requires uv for all package management — pip silently overwrites ROCm packages
type: feedback
originSessionId: 1cac980d-2f8c-45f6-8704-b8fe85e81113
---
NEVER use `pip install`. ALWAYS use `uv add` to add packages and `uv sync` to install.

**Why:** pip silently overwrites ROCm-provided PyTorch packages, breaking GPU support. This happened in practice and is now the first line of CLAUDE.md.

**How to apply:** For any dependency addition, use `uv add <package>`. For installing from lockfile, use `uv sync`. The only exception is `pip install uv` as a Dockerfile bootstrap.
