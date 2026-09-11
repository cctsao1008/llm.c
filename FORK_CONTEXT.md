# Fork Context

This repository is a fork of [karpathy/llm.c](https://github.com/karpathy/llm.c).

The upstream README is intentionally preserved as the primary documentation for `llm.c` itself. Fork-specific work should not rewrite upstream project history or present upstream design decisions as local authorship.

Within the `cctsao1008` project set, this fork serves as an LLM implementation/reference surface for experiments and comparisons around machine learning and LSMM-related work. The independent LSMM research architecture lives in [`cctsao1008/lsmm.c`](https://github.com/cctsao1008/lsmm.c); `llm.c` is a baseline and implementation resource, not the definition of LSMM.

## Documentation principle

> **README explains the system. Issues explain the journey. Code proves the current state.**

For this upstream-derived repository:

- the upstream README explains the upstream `llm.c` system and usage;
- this file explains the durable purpose of the fork within the local project set;
- GitHub Issues preserve fork-specific experiments, investigations, temporary constraints, and work history;
- code, branches, configuration, and tests prove the state of the fork.

This separation keeps upstream provenance intact while allowing local research context to remain explicit.
