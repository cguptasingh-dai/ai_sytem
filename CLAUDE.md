# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the orchestrator (interactive)
python main.py

# Verify all modules compile
python -m py_compile config.py state.py tools.py agents.py graph.py main.py
```

Required env var in `.env`: `GEMINI_API_KEY`. Optional: `PROJECT_PATH` (default `./generated_project`).

There is no test suite, linter, or build step configured.

## Architecture

This is a **multi-agent LLM orchestrator** that simulates a 14-person AI development team. Given a project brief, it produces a complete codebase + docs in a chosen output directory by running 14 specialized Gemini-backed agents through a LangGraph state machine.

The 14 roles split into: **2 leadership** (Lead Engineer, AI Architect), **2 planning** (PM, System Architect), **4 product engineering** (Backend, Frontend, AI Developer, DevOps), **4 ML/data specialists** (Data Engineer, ML Researcher, Prompt Engineer, MLOps), and **2 quality** (QA, Security).

### Module roles

- **`main.py`** — interactive CLI; collects project name, requirements, output path, then calls `DevTeamOrchestrator.run()`.
- **`config.py`** — env loading, model names (`GEMINI_MODEL_PRO`, `GEMINI_MODEL_PRO_TOOLS`), retry settings, and the **mutable** `BASE_DIR` (output directory; rewritten at runtime).
- **`state.py`** — `AgentRole` enum (10 roles), Pydantic models (`Task`, `CodeArtifact`, `Message`, `Meeting`), and the `ProjectState` TypedDict that flows through the LangGraph.
- **`tools.py`** — `@tool`-decorated file-system + planning tools exposed to every agent. All paths are resolved against `config.BASE_DIR` via `_base()`.
- **`agents.py`** — `BaseDeepAgent` (retry loop + rich context builder) and 14 subclasses, one per role. Each subclass only differs in its system prompt.
- **`graph.py`** — `DevTeamOrchestrator` builds the LangGraph workflow and defines one node per phase.

### State flow

`ProjectState` is a TypedDict whose list/dict fields use `Annotated[..., reducer]` so LangGraph can merge updates from parallel branches:
- Lists use `operator.add` (concat) — `tasks`, `errors`, `code_artifacts`, `error_log`, `chat_log`, `meetings`, `completed_tasks`, `review_comments`, `sub_agent_requests`.
- Dicts use a custom `lambda a, b: {**a, **b}` merger — `agent_responses`, `plan`, `retry_counts`, `phase_gates`.

When adding new state fields, choose the correct reducer; otherwise parallel nodes will overwrite each other or LangGraph will raise on merge.

### Workflow graph (built in `_build_graph()`)

```
START → lead_kickoff → pm → architect → ai_architect
      → [backend ‖ frontend ‖ ai ‖ devops ‖ data_engineer ‖ ml_researcher ‖ prompt_engineer ‖ mlops]
      → lead_review
      → conditional: errors? → error_recovery → [qa ‖ security]
                              else            → [qa ‖ security]
      → integration → meeting → END
```

The conditional after `lead_review` uses `_check_errors_after_dev` which inspects `state["errors"]`. Parallel fan-out is achieved by adding multiple edges from one node — LangGraph runs them concurrently and merges results via the state reducers.

### Deep agent pattern (`BaseDeepAgent`)

Every agent goes through the same pipeline:
1. `invoke()` runs `_invoke_with_retry()` up to `MAX_RETRIES` (3) times with exponential backoff (`RETRY_DELAY * (attempt+1)`).
2. On final failure, returns `{errors, retry_counts, error_log}` instead of raising — the graph keeps running and `error_recovery` picks it up.
3. `_build_context()` injects: project name, current phase, requirements, architecture, last 3 agent outputs, current plan, plus the agent's own system prompt and the current task.
4. Tool-using agents are wrapped in `create_react_agent(self.llm, ALL_TOOLS, prompt=system_prompt)`. All agents currently use tools (`use_tools=True`).

To add a new agent: add an `AgentRole` enum member, subclass `BaseDeepAgent` with a system prompt, register it in `_get_agents()`, then add a node + edges in `graph.py`.

### `BASE_DIR` mutation gotcha

`config.BASE_DIR` starts as `./generated_project` but `DevTeamOrchestrator.run()` mutates it to the user-provided path **and re-imports `_get_agents()` to rebuild `AGENTS`** so tool calls resolve to the new directory. The `tools.py` helpers read `config.BASE_DIR` lazily via `_base()` for this reason — don't cache it at import time.

### Models

`config.py` sets both `GEMINI_MODEL_PRO` and `GEMINI_MODEL_PRO_TOOLS` to `"gemini-1.5-pro"`. If a different Gemini model is needed, change these constants — agents pick the model based on `use_tools` at construction time.
