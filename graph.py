from typing import Dict, Any, List
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
import uuid
import re
from datetime import datetime
from state import ProjectState, AgentRole, Task, Meeting
from agents import AGENTS
import config


VALID_REWORK_ROLES = {
    AgentRole.BACKEND.value,
    AgentRole.FRONTEND.value,
    AgentRole.AI_DEV.value,
    AgentRole.DEVOPS.value,
    AgentRole.DATA_ENGINEER.value,
    AgentRole.ML_RESEARCHER.value,
    AgentRole.PROMPT_ENGINEER.value,
    AgentRole.MLOPS.value,
}


def _parse_rework_requests(review_text: str) -> Dict[str, str]:
    """Extract `{role: feedback}` from the Lead Engineer's structured review block.

    Looks for the `## REWORK_REQUESTS` section and parses bullets shaped like
    `- backend_dev: missing input validation on /api/users`. Unknown role names
    are silently dropped so a typo from the LLM can't poison the rework set.
    """
    if not review_text or "REWORK_REQUESTS" not in review_text:
        return {}
    # Slice from REWORK_REQUESTS heading to next ## heading (or EOF)
    m = re.search(r"##\s*REWORK_REQUESTS\s*\n(.*?)(?:\n##|\Z)", review_text, re.DOTALL | re.IGNORECASE)
    if not m:
        return {}
    section = m.group(1).strip()
    if section.lower().startswith("none"):
        return {}
    requests: Dict[str, str] = {}
    for raw in section.splitlines():
        line = raw.strip().lstrip("-*•").strip()
        if ":" not in line:
            continue
        role_part, feedback = line.split(":", 1)
        role_key = role_part.strip().lower().replace(" ", "_")
        if role_key in VALID_REWORK_ROLES and feedback.strip():
            requests[role_key] = feedback.strip()
    return requests


def _is_ai_heavy(state: ProjectState) -> bool:
    """Decide whether to activate the AI/ML specialist track."""
    text = " ".join([
        state.get("requirements_raw") or "",
        state.get("requirements_doc") or "",
        state.get("architecture_doc") or "",
    ]).lower()
    return any(k in text for k in config.AI_KEYWORDS)


class DevTeamOrchestrator:
    """Multi-agent orchestrator with error recovery and phase gates."""

    def __init__(self):
        self.graph = self._build_graph()
        self.memory = MemorySaver()
        self.app = self.graph.compile(checkpointer=self.memory)

    def _build_graph(self):
        builder = StateGraph(ProjectState)

        # Phase 1: Kickoff and Planning
        builder.add_node("lead_kickoff", self._lead_kickoff_node)

        # Phase 2: Requirements and High-Level Design
        builder.add_node("pm", self._pm_node)
        builder.add_node("architect", self._architect_node)

        # Phase 3: Detailed Architecture
        builder.add_node("ai_architect", self._ai_architect_node)

        # Phase 4: Development (Parallel)
        builder.add_node("backend", self._backend_node)
        builder.add_node("frontend", self._frontend_node)
        builder.add_node("ai", self._ai_node)
        builder.add_node("devops", self._devops_node)
        builder.add_node("data_engineer", self._data_engineer_node)
        builder.add_node("ml_researcher", self._ml_researcher_node)
        builder.add_node("prompt_engineer", self._prompt_engineer_node)
        builder.add_node("mlops", self._mlops_node)

        # Phase 5: Lead Engineer Review
        builder.add_node("lead_review", self._lead_review_node)

        # Phase 5.5: Rework loop — re-runs flagged dev agents with Lead's feedback
        builder.add_node("rework", self._rework_node)

        # Phase 6: QA and Security (Parallel)
        builder.add_node("qa", self._qa_node)
        builder.add_node("security", self._security_node)

        # Phase 7: Error Recovery
        builder.add_node("error_recovery", self._error_recovery_node)

        # Phase 8: Integration
        builder.add_node("integration", self._integration_node)

        # Phase 9: Final Meeting
        builder.add_node("meeting", self._meeting_node)

        # Define workflow edges
        builder.add_edge(START, "lead_kickoff")
        builder.add_edge("lead_kickoff", "pm")
        builder.add_edge("pm", "architect")
        builder.add_edge("architect", "ai_architect")

        # Conditional fan-out: always run backend/frontend/devops; only spin up the
        # AI/ML specialists when the project actually involves AI work. This keeps
        # CRUD-only projects fast and prevents irrelevant artifacts.
        builder.add_conditional_edges(
            "ai_architect",
            self._route_to_dev,
            {
                "backend": "backend",
                "frontend": "frontend",
                "ai": "ai",
                "devops": "devops",
                "data_engineer": "data_engineer",
                "ml_researcher": "ml_researcher",
                "prompt_engineer": "prompt_engineer",
                "mlops": "mlops",
            },
        )

        # Converge to lead review (only the activated dev nodes will fire)
        builder.add_edge("backend", "lead_review")
        builder.add_edge("frontend", "lead_review")
        builder.add_edge("ai", "lead_review")
        builder.add_edge("devops", "lead_review")
        builder.add_edge("data_engineer", "lead_review")
        builder.add_edge("ml_researcher", "lead_review")
        builder.add_edge("prompt_engineer", "lead_review")
        builder.add_edge("mlops", "lead_review")

        # Feedback loop: if Lead Engineer flagged gaps and we're under the
        # iteration cap, route to `rework` (which re-runs only the flagged
        # agents) and bounce BACK to lead_review for re-evaluation. Otherwise
        # proceed to error_recovery → QA + Security.
        builder.add_conditional_edges(
            "lead_review",
            self._route_after_review,
            {
                "rework": "rework",
                "forward": "error_recovery",
            },
        )
        builder.add_edge("rework", "lead_review")

        builder.add_edge("error_recovery", "qa")
        builder.add_edge("error_recovery", "security")

        # Converge to integration
        builder.add_edge("qa", "integration")
        builder.add_edge("security", "integration")

        # Final meeting
        builder.add_edge("integration", "meeting")
        builder.add_edge("meeting", END)

        return builder

    def _route_to_dev(self, state: ProjectState) -> List[str]:
        """Fan-out router: pick which dev agents to activate."""
        base = ["backend", "frontend", "devops"]
        if _is_ai_heavy(state):
            print("   [ROUTER] AI-heavy project detected -> activating full AI/ML track")
            return base + ["ai", "data_engineer", "ml_researcher", "prompt_engineer", "mlops"]
        print("   [ROUTER] Non-AI project -> activating core stack only (backend, frontend, devops)")
        return base

    def _route_after_review(self, state: ProjectState) -> str:
        """Loop or move on based on Lead Engineer's REWORK_REQUESTS block."""
        rework = state.get("rework_requests") or {}
        iteration = state.get("review_iteration") or 0
        if rework and iteration < config.MAX_REVIEW_ITERATIONS:
            print(f"   [ROUTER] Lead flagged {len(rework)} agent(s) for rework "
                  f"(cycle {iteration + 1}/{config.MAX_REVIEW_ITERATIONS}) -> looping")
            return "rework"
        if rework:
            print(f"   [ROUTER] Rework cap reached ({config.MAX_REVIEW_ITERATIONS}) — "
                  f"forwarding {len(rework)} unresolved gap(s) to error_recovery")
        else:
            print("   [ROUTER] Lead approved — proceeding to QA + Security")
        return "forward"

    async def _lead_kickoff_node(self, state: ProjectState) -> Dict[str, Any]:
        """Lead Engineer kickoff: setup team and initial planning."""
        print("\n" + "="*70)
        print("PHASE 0: LEAD ENGINEER KICKOFF")
        print("="*70)
        agent = AGENTS[AgentRole.LEAD_ENGINEER]
        task = Task(
            id=str(uuid.uuid4()),
            title="Team Kickoff and Project Planning",
            description="Setup engineering team, create initial project plan, assign roles",
            assigned_to=AgentRole.LEAD_ENGINEER
        )
        result = await agent.invoke(state, task)
        return {
            **result,
            "current_phase": "kickoff",
            "phase_gates": {"kickoff": True}
        }

    async def _pm_node(self, state: ProjectState) -> Dict[str, Any]:
        """Product Manager: Requirements analysis and PRD."""
        print("\n" + "="*70)
        print("PHASE 1: PRODUCT MANAGER - Requirements Analysis")
        print("="*70)
        agent = AGENTS[AgentRole.PM]
        task = Task(
            id=str(uuid.uuid4()),
            title="Create PRD and Requirements",
            description="Analyze requirements and create detailed PRD with user stories",
            assigned_to=AgentRole.PM
        )
        result = await agent.invoke(state, task)

        tasks = [
            Task(id=str(uuid.uuid4()), title="System Architecture Design",
                 description="Design overall architecture", assigned_to=AgentRole.ARCHITECT),
            Task(id=str(uuid.uuid4()), title="Detailed Architecture Design",
                 description="Design components, APIs, database schema", assigned_to=AgentRole.AI_ARCHITECT),
            Task(id=str(uuid.uuid4()), title="Backend API Implementation",
                 description="Implement backend APIs and database", assigned_to=AgentRole.BACKEND),
            Task(id=str(uuid.uuid4()), title="Frontend UI Development",
                 description="Build user interface", assigned_to=AgentRole.FRONTEND),
            Task(id=str(uuid.uuid4()), title="AI Features Integration",
                 description="Integrate Gemini AI features", assigned_to=AgentRole.AI_DEV),
            Task(id=str(uuid.uuid4()), title="DevOps Setup",
                 description="Docker, CI/CD pipeline, monitoring", assigned_to=AgentRole.DEVOPS),
            Task(id=str(uuid.uuid4()), title="QA Testing",
                 description="Write and run comprehensive tests", assigned_to=AgentRole.QA),
            Task(id=str(uuid.uuid4()), title="Security Audit",
                 description="Review code for vulnerabilities", assigned_to=AgentRole.SECURITY),
        ]

        print(f"   Created {len(tasks)} tasks for execution")
        return {
            **result,
            "requirements_doc": result["agent_responses"].get(AgentRole.PM.value, ""),
            "tasks": tasks,
            "current_phase": "requirements",
            "phase_gates": {**state.get("phase_gates", {}), "requirements": True}
        }

    async def _architect_node(self, state: ProjectState) -> Dict[str, Any]:
        """System Architect: High-level design."""
        print("\n" + "="*70)
        print("PHASE 2: SYSTEM ARCHITECT - High-Level Design")
        print("="*70)
        agent = AGENTS[AgentRole.ARCHITECT]
        task = Task(
            id=str(uuid.uuid4()),
            title="Architecture Design",
            description="Design high-level architecture and approve technology stack",
            assigned_to=AgentRole.ARCHITECT
        )
        result = await agent.invoke(state, task)
        return {
            **result,
            "current_phase": "architecture",
            "phase_gates": {**state.get("phase_gates", {}), "architecture": True}
        }

    async def _ai_architect_node(self, state: ProjectState) -> Dict[str, Any]:
        """AI Architect: Detailed component architecture."""
        print("\n" + "="*70)
        print("PHASE 3: AI ARCHITECT - Detailed Design")
        print("="*70)
        agent = AGENTS[AgentRole.AI_ARCHITECT]
        task = Task(
            id=str(uuid.uuid4()),
            title="Detailed Architecture Design",
            description="Design detailed components, APIs, database schema, integration points",
            assigned_to=AgentRole.AI_ARCHITECT
        )
        result = await agent.invoke(state, task)
        return {
            **result,
            "architecture_doc": result["agent_responses"].get(AgentRole.AI_ARCHITECT.value, ""),
            "current_phase": "detailed_architecture",
            "phase_gates": {**state.get("phase_gates", {}), "detailed_architecture": True}
        }

    async def _backend_node(self, state: ProjectState) -> Dict:
        """Backend Developer: API implementation."""
        print("\n" + "="*70)
        print("PHASE 4A: BACKEND DEVELOPER")
        print("="*70)
        agent = AGENTS[AgentRole.BACKEND]
        task = Task(
            id=str(uuid.uuid4()),
            title="Backend API Implementation",
            description="Implement all APIs, database models, business logic",
            assigned_to=AgentRole.BACKEND
        )
        result = await agent.invoke(state, task)
        return {**result, "current_phase": "backend_dev"}

    async def _frontend_node(self, state: ProjectState) -> Dict:
        """Frontend Developer: UI implementation."""
        print("\n" + "="*70)
        print("PHASE 4B: FRONTEND DEVELOPER")
        print("="*70)
        agent = AGENTS[AgentRole.FRONTEND]
        task = Task(
            id=str(uuid.uuid4()),
            title="Frontend UI Development",
            description="Build all UI components, pages, state management",
            assigned_to=AgentRole.FRONTEND
        )
        result = await agent.invoke(state, task)
        return {**result, "current_phase": "frontend_dev"}

    async def _ai_node(self, state: ProjectState) -> Dict:
        """AI Developer: AI features."""
        print("\n" + "="*70)
        print("PHASE 4C: AI DEVELOPER")
        print("="*70)
        agent = AGENTS[AgentRole.AI_DEV]
        task = Task(
            id=str(uuid.uuid4()),
            title="AI Features Integration",
            description="Integrate Gemini, implement RAG, AI-powered features",
            assigned_to=AgentRole.AI_DEV
        )
        result = await agent.invoke(state, task)
        return {**result, "current_phase": "ai_dev"}

    async def _devops_node(self, state: ProjectState) -> Dict:
        """DevOps Engineer: Infrastructure setup."""
        print("\n" + "="*70)
        print("PHASE 4D: DEVOPS ENGINEER")
        print("="*70)
        agent = AGENTS[AgentRole.DEVOPS]
        task = Task(
            id=str(uuid.uuid4()),
            title="DevOps Setup",
            description="Docker, CI/CD pipeline, monitoring, deployment config",
            assigned_to=AgentRole.DEVOPS
        )
        result = await agent.invoke(state, task)
        return {**result, "current_phase": "devops_setup"}

    async def _data_engineer_node(self, state: ProjectState) -> Dict:
        """Data Engineer: Pipelines, ETL, vector stores."""
        print("\n" + "="*70)
        print("PHASE 4E: DATA ENGINEER")
        print("="*70)
        agent = AGENTS[AgentRole.DATA_ENGINEER]
        task = Task(
            id=str(uuid.uuid4()),
            title="Data Pipelines & Vector Store",
            description="Build ingestion, ETL, embeddings, and vector DB setup for AI workloads",
            assigned_to=AgentRole.DATA_ENGINEER
        )
        result = await agent.invoke(state, task)
        return {**result, "current_phase": "data_engineering"}

    async def _ml_researcher_node(self, state: ProjectState) -> Dict:
        """ML Research Engineer: Model selection, training, evaluation."""
        print("\n" + "="*70)
        print("PHASE 4F: ML RESEARCH ENGINEER")
        print("="*70)
        agent = AGENTS[AgentRole.ML_RESEARCHER]
        task = Task(
            id=str(uuid.uuid4()),
            title="Model Selection & Evaluation",
            description="Select models, design eval framework, plan fine-tuning if needed",
            assigned_to=AgentRole.ML_RESEARCHER
        )
        result = await agent.invoke(state, task)
        return {**result, "current_phase": "ml_research"}

    async def _prompt_engineer_node(self, state: ProjectState) -> Dict:
        """Prompt Engineer: Prompt templates, parsers, guardrails."""
        print("\n" + "="*70)
        print("PHASE 4G: PROMPT ENGINEER")
        print("="*70)
        agent = AGENTS[AgentRole.PROMPT_ENGINEER]
        task = Task(
            id=str(uuid.uuid4()),
            title="Prompt Library & Guardrails",
            description="Design versioned prompts, output parsers, golden tests, and safety guardrails",
            assigned_to=AgentRole.PROMPT_ENGINEER
        )
        result = await agent.invoke(state, task)
        return {**result, "current_phase": "prompt_engineering"}

    async def _mlops_node(self, state: ProjectState) -> Dict:
        """MLOps Engineer: Serving, monitoring, drift detection."""
        print("\n" + "="*70)
        print("PHASE 4H: MLOPS ENGINEER")
        print("="*70)
        agent = AGENTS[AgentRole.MLOPS]
        task = Task(
            id=str(uuid.uuid4()),
            title="Model Serving & Observability",
            description="Setup serving, model registry, monitoring, drift detection, SLOs",
            assigned_to=AgentRole.MLOPS
        )
        result = await agent.invoke(state, task)
        return {**result, "current_phase": "mlops"}

    async def _lead_review_node(self, state: ProjectState) -> Dict[str, Any]:
        """Lead Engineer: Code review and quality gate.

        Invokes the Lead Engineer to inspect every dev agent's output, then parses
        the structured `## REWORK_REQUESTS` block from the response to populate
        `rework_requests` for the conditional edge that follows.
        """
        iteration = state.get("review_iteration") or 0
        print("\n" + "="*70)
        if iteration == 0:
            print("PHASE 5: LEAD ENGINEER CODE REVIEW")
        else:
            print(f"PHASE 5 (RE-REVIEW after rework cycle {iteration}): LEAD ENGINEER")
        print("="*70)
        agent = AGENTS[AgentRole.LEAD_ENGINEER]
        review_focus = (
            "Initial review: inspect every dev agent's artifacts, cross-check contracts, "
            "and emit a structured ## REWORK_REQUESTS block listing each agent that must "
            "fix gaps (or 'None' if everything is acceptable)."
            if iteration == 0
            else f"RE-REVIEW after rework cycle {iteration}. The agents listed in the "
                 f"prior review have re-run. Check whether their gaps are actually fixed. "
                 f"Only re-flag agents whose gaps remain; otherwise emit 'None'."
        )
        task = Task(
            id=str(uuid.uuid4()),
            title="Code Review and Quality Gate" if iteration == 0 else f"Re-review (cycle {iteration})",
            description=(
                f"Review all code, check architecture compliance, identify gaps. {review_focus} "
                f"End your response with the REWORK_REQUESTS and REVIEW_VERDICT blocks "
                f"per the system prompt — the orchestrator parses them."
            ),
            assigned_to=AgentRole.LEAD_ENGINEER,
        )
        result = await agent.invoke(state, task)
        review_text = result.get("agent_responses", {}).get(AgentRole.LEAD_ENGINEER.value, "")
        rework_requests = _parse_rework_requests(review_text)
        if rework_requests:
            print(f"   Lead flagged {len(rework_requests)} agent(s) for rework: "
                  f"{', '.join(rework_requests.keys())}")
        else:
            print("   Lead approved — no rework requested")
        errors = result.get("errors", [])
        if errors:
            print(f"   Review surfaced {len(errors)} agent error(s) for recovery")
        return {
            **result,
            "current_phase": "lead_review" if iteration == 0 else f"lead_review_cycle_{iteration}",
            "rework_requests": rework_requests,
        }

    async def _rework_node(self, state: ProjectState) -> Dict[str, Any]:
        """Re-run dev agents flagged by the Lead Engineer with targeted feedback.

        Each flagged agent gets a Task whose description embeds the exact gap from
        the Lead's review, plus an instruction to update existing files rather than
        regenerate. After all reworks, we increment `review_iteration` and clear
        `rework_requests` so the conditional edge re-evaluates cleanly when we
        loop back to lead_review.
        """
        rework_requests = state.get("rework_requests") or {}
        iteration = state.get("review_iteration") or 0
        print("\n" + "="*70)
        print(f"PHASE 5.A: REWORK CYCLE #{iteration + 1} — {len(rework_requests)} agent(s)")
        print("="*70)

        accumulated_responses: Dict[str, str] = {}
        accumulated_chat_log: list = []
        accumulated_errors: list = []
        accumulated_artifacts: list = []
        accumulated_error_log: list = []

        for role_str, feedback in rework_requests.items():
            try:
                role = AgentRole(role_str)
            except ValueError:
                print(f"   Skipping unknown role in rework: {role_str}")
                continue
            agent = AGENTS.get(role)
            if not agent:
                print(f"   No registered agent for role {role_str}")
                continue
            print(f"   Reworking {role_str}: {feedback[:120]}")
            task = Task(
                id=str(uuid.uuid4()),
                title=f"Rework cycle {iteration + 1}: address Lead Engineer feedback",
                description=(
                    f"The Lead Engineer reviewed your previous output and flagged this gap:\n\n"
                    f"  {feedback}\n\n"
                    f"INSTRUCTIONS:\n"
                    f"1. Use list_directory + read_file to inspect what you produced previously.\n"
                    f"2. UPDATE / EDIT existing files to fix the gap. Do NOT regenerate from scratch.\n"
                    f"3. If new files are needed, create them — but keep prior structure intact.\n"
                    f"4. End your response with a one-line summary of what you changed."
                ),
                assigned_to=role,
            )
            try:
                result = await agent.invoke(state, task)
                accumulated_responses.update(result.get("agent_responses", {}))
                accumulated_chat_log.extend(result.get("chat_log", []))
                accumulated_errors.extend(result.get("errors", []))
                accumulated_artifacts.extend(result.get("code_artifacts", []))
                accumulated_error_log.extend(result.get("error_log", []))
            except Exception as e:
                msg = f"Rework for {role_str} failed: {str(e)[:200]}"
                print(f"   {msg}")
                accumulated_errors.append(msg)

        return {
            "current_phase": f"rework_cycle_{iteration + 1}",
            "rework_requests": {},
            "review_iteration": iteration + 1,
            "agent_responses": accumulated_responses,
            "chat_log": accumulated_chat_log,
            "errors": accumulated_errors,
            "code_artifacts": accumulated_artifacts,
            "error_log": accumulated_error_log,
            "phase_gates": {
                **state.get("phase_gates", {}),
                f"rework_cycle_{iteration + 1}": True,
            },
        }

    async def _qa_node(self, state: ProjectState) -> Dict:
        """QA Engineer: Testing and validation."""
        print("\n" + "="*70)
        print("PHASE 6A: QA ENGINEER")
        print("="*70)
        agent = AGENTS[AgentRole.QA]
        task = Task(
            id=str(uuid.uuid4()),
            title="QA Testing",
            description="Write tests, run test suites, report bugs, validate acceptance criteria",
            assigned_to=AgentRole.QA
        )
        result = await agent.invoke(state, task)
        return {**result, "current_phase": "qa"}

    async def _security_node(self, state: ProjectState) -> Dict:
        """Security Reviewer: Vulnerability audit."""
        print("\n" + "="*70)
        print("PHASE 6B: SECURITY REVIEWER")
        print("="*70)
        agent = AGENTS[AgentRole.SECURITY]
        task = Task(
            id=str(uuid.uuid4()),
            title="Security Audit",
            description="Scan for OWASP vulnerabilities, audit authentication, check data protection",
            assigned_to=AgentRole.SECURITY
        )
        result = await agent.invoke(state, task)
        return {**result, "current_phase": "security"}

    async def _error_recovery_node(self, state: ProjectState) -> Dict[str, Any]:
        """Error Recovery: classify failures, write report, retry recoverable ones."""
        print("\n" + "="*70)
        print("PHASE 5.5: ERROR RECOVERY")
        print("="*70)
        errors = state.get("errors", [])
        error_log = state.get("error_log", [])
        retry_counts = state.get("retry_counts", {})

        print(f"   Analyzing {len(errors)} errors across {len(error_log)} log entries")

        # Classify failures
        recoverable = []   # transient / network / timeout — worth one more shot
        unrecoverable = []  # auth / validation — re-attempting won't help
        for entry in error_log:
            ec = entry.get("error_class", "transient")
            if ec in ("auth", "validation"):
                unrecoverable.append(entry)
            else:
                recoverable.append(entry)

        print(f"   Recoverable: {len(recoverable)} | Unrecoverable: {len(unrecoverable)}")

        # Write a recovery report so QA/Security/Integration know the state
        report = ["# Error Recovery Report", f"Generated: {datetime.now().isoformat()}", ""]
        report.append(f"## Summary\n- Total errors: {len(errors)}\n- Recoverable: {len(recoverable)}\n- Unrecoverable: {len(unrecoverable)}\n")
        if unrecoverable:
            report.append("## Unrecoverable Failures (downstream agents must work around these)\n")
            for entry in unrecoverable:
                report.append(f"- **{entry['agent']}** ({entry.get('error_class')}): {entry.get('error')[:200]}")
            report.append("")
        if recoverable:
            report.append("## Recoverable Failures (will retry once)\n")
            for entry in recoverable:
                report.append(f"- **{entry['agent']}** ({entry.get('error_class')}): {entry.get('error')[:200]}")
            report.append("")

        try:
            from tools import write_file
            await write_file.ainvoke({"file_path": "docs/recovery_report.md", "content": "\n".join(report)})
        except Exception as e:
            print(f"   Could not write recovery report: {e}")

        # If every recoverable failure shares the same root error class (e.g. all
        # validation/auth), retrying via the Lead Engineer — which uses the same
        # LLM config — is futile. Detect this and bail with a single recorded note
        # instead of grinding through N×retries of the same broken request.
        unique_classes = {entry.get("error_class") for entry in recoverable}
        global_block = None
        if recoverable and unique_classes.issubset({"validation", "auth"}):
            global_block = (
                f"All {len(recoverable)} recoverable failures share class "
                f"{unique_classes}; the underlying cause (bad model, bad schema, "
                f"or bad credentials) cannot be fixed by retrying via Lead Engineer "
                f"who uses the same LLM. Skipping per-agent recovery."
            )
            print(f"   {global_block}")

        recovered_responses = {}
        if global_block is None:
            # Re-attempt each recoverable failure ONCE with the Lead Engineer as the executor
            for entry in recoverable:
                agent_role_str = entry["agent"]
                print(f"   Retrying via Lead Engineer for: {agent_role_str}")
                try:
                    recovery_task = Task(
                        id=str(uuid.uuid4()),
                        title=f"Recovery for {agent_role_str}",
                        description=(
                            f"The {agent_role_str} agent failed: {entry.get('error', '')[:200]}. "
                            f"Produce a minimal working stub artifact for this role so downstream "
                            f"agents (QA, Security, Integration) can continue. Document gaps in docs/agent_notes/{agent_role_str}.md."
                        ),
                        assigned_to=AgentRole.LEAD_ENGINEER,
                    )
                    lead = AGENTS[AgentRole.LEAD_ENGINEER]
                    result = await lead.invoke(state, recovery_task)
                    response = result.get("agent_responses", {}).get(AgentRole.LEAD_ENGINEER.value, "")
                    if response:
                        recovered_responses[agent_role_str] = f"[RECOVERED via Lead Engineer]\n{response}"
                except Exception as e:
                    print(f"   Recovery attempt for {agent_role_str} also failed: {e}")
                    recovered_responses[agent_role_str] = f"[UNRECOVERED] {str(e)[:200]}"
        else:
            for entry in recoverable:
                recovered_responses[entry["agent"]] = f"[UNRECOVERED — global block] {global_block}"

        return {
            "current_phase": "error_recovery",
            "agent_responses": recovered_responses,
            "phase_gates": {**state.get("phase_gates", {}), "error_recovery": True},
        }

    async def _integration_node(self, state: ProjectState) -> Dict:
        """Integration: Combine all artifacts and verify generated Python compiles."""
        print("\n" + "="*70)
        print("PHASE 7: INTEGRATION - Combining all artifacts")
        print("="*70)

        from pathlib import Path
        import subprocess
        import sys
        base = Path(config.BASE_DIR)
        on_disk = [p for p in base.rglob("*") if p.is_file()] if base.exists() else []
        artifacts_count = len(state.get('code_artifacts', []))
        errors = state.get("errors", [])

        # Verify: byte-compile every generated .py file. Catches the "complex project
        # generated, none of it imports" failure mode silently rampant in LLM agents.
        py_files = [p for p in on_disk if p.suffix == ".py"]
        syntax_failures: list[tuple[str, str]] = []
        for pf in py_files:
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "py_compile", str(pf)],
                    capture_output=True, text=True, timeout=15,
                )
                if result.returncode != 0:
                    syntax_failures.append((str(pf.relative_to(base)), (result.stderr or result.stdout)[:300]))
            except Exception as e:
                syntax_failures.append((str(pf.relative_to(base)), f"compile check failed: {e}"))

        if syntax_failures:
            verify_lines = ["# Build Verification Report", f"Checked {len(py_files)} Python file(s).", ""]
            verify_lines.append(f"## Syntax Failures ({len(syntax_failures)})\n")
            for path, msg in syntax_failures:
                verify_lines.append(f"- **{path}**\n  ```\n  {msg.strip()}\n  ```")
            try:
                from tools import write_file as _wf
                await _wf.ainvoke({"file_path": "docs/build_verification.md", "content": "\n".join(verify_lines)})
            except Exception as e:
                print(f"   Could not write build verification report: {e}")
            print(f"   Build verification: {len(syntax_failures)}/{len(py_files)} Python files FAILED to compile")
        else:
            print(f"   Build verification: all {len(py_files)} Python files compile [OK]")

        success = (artifacts_count > 0 or len(on_disk) > 1) and not syntax_failures
        if success and not errors:
            verdict = "[OK]"
        elif on_disk:
            verdict = "[PARTIAL]"
        else:
            verdict = "[FAILED]"

        print(f"   Code artifacts (state): {artifacts_count}")
        print(f"   Files on disk: {len(on_disk)}")
        print(f"   Upstream errors: {len(errors)}")
        print(f"   Syntax failures: {len(syntax_failures)}")
        print(f"   Verdict: {verdict}")

        from tools import write_file
        summary = f"""# {state['project_name']} - Integration {verdict}

**Generated by AI Development Team**
- Framework: LangGraph + Gemini 3.1 Pro
- Team: 14 Deep Learning Agents
- Date: {datetime.now().isoformat()}
- Files produced: {len(on_disk)}
- Python files verified: {len(py_files) - len(syntax_failures)} / {len(py_files)}
- Agent failures: {len(errors)}
- Syntax failures: {len(syntax_failures)} (see docs/build_verification.md)

## Team Composition
1. **Lead Engineer** - Supervision & Code Review
2. **AI Architect** - Detailed System Design
3. **Product Manager** - Requirements & PRD
4. **System Architect** - High-Level Design
5. **Backend Developer** - APIs & Business Logic
6. **Frontend Developer** - UI Components
7. **AI Developer** - LLM Integration
8. **DevOps Engineer** - Infrastructure & CI/CD
9. **QA Engineer** - Testing & Validation
10. **Security Reviewer** - Vulnerability Audit

## Artifacts Generated
- Code artifacts: {artifacts_count}
- Requirements: docs/PRD.md
- Architecture: docs/architecture.md
- API Specification: docs/api_spec.md
- Database Schema: docs/database_schema.md
- Security Audit: docs/security_audit.md
- Deployment: Dockerfile, docker-compose.yml, CI/CD pipelines
- Tests: tests/backend, tests/frontend, tests/integration

## Phase Completion
- [x] Kickoff & Planning
- [x] Requirements Analysis
- [x] High-Level Architecture
- [x] Detailed Architecture
- [x] Development (Backend, Frontend, AI, DevOps)
- [x] Code Review
- [x] Testing & Security
- [x] Integration

## Next Steps
1. Review all documentation in docs/
2. Run tests: `pytest tests/`
3. Build Docker image: `docker build -t {state['project_name']} .`
4. Deploy using pipeline: `./deploy.sh`
5. Monitor with configured logging and metrics

---
*This project was generated by an AI development team with deep specialization in each role.*
"""
        await write_file.ainvoke({"file_path": "README.md", "content": summary})
        print("   Integration complete. README.md created.")
        return {
            "current_phase": "integration",
            "phase_gates": {
                **state.get("phase_gates", {}),
                "integration": success,
                "integration_verdict": verdict,
                "syntax_failures": len(syntax_failures),
                "files_on_disk": len(on_disk),
            },
        }

    async def _meeting_node(self, state: ProjectState) -> Dict:
        """Final Meeting: Sprint completion and handoff. Reflects actual run state."""
        print("\n" + "="*70)
        print("PHASE 8: SPRINT REVIEW MEETING")
        print("="*70)

        gates = state.get("phase_gates", {})
        verdict = gates.get("integration_verdict", "[UNKNOWN]")
        errors = state.get("errors", [])
        artifacts = len(state.get("code_artifacts", []))
        delivered = gates.get("integration", False)
        syntax_failures = gates.get("syntax_failures", 0)
        files_on_disk = gates.get("files_on_disk", 0)

        print(f"   Participants: All 14 agents")
        print(f"   Project: {state['project_name']}")
        print(f"   Status: {verdict}")
        print(f"   Code artifacts (tracked): {artifacts}")
        print(f"   Files on disk: {files_on_disk}")
        print(f"   Syntax failures: {syntax_failures}")
        print(f"   Agent failures: {len(errors)}")

        if delivered and not errors and not syntax_failures:
            decisions = [
                "Pipeline ran end-to-end with no agent failures",
                f"All {files_on_disk} generated files on disk; Python syntax verified",
                "Code review passed",
                "Security audit completed",
                "Ready for staging deployment",
            ]
            action_items = ["Deploy to staging", "Monitor in production", "Gather user feedback"]
        elif delivered or files_on_disk > 0:
            decisions = [
                f"Pipeline produced output ({files_on_disk} file(s)) "
                f"with {len(errors)} agent failure(s) and {syntax_failures} syntax failure(s)",
                "Review docs/recovery_report.md and docs/build_verification.md before promoting",
                "Treat affected roles' deliverables as best-effort stubs",
            ]
            action_items = [
                "Re-run failed agents after fixing root cause",
                "Fix Python files flagged in docs/build_verification.md",
                "Validate stub artifacts against acceptance criteria",
                "Hold off on staging until gaps are filled",
            ]
        else:
            decisions = [
                "Pipeline did NOT produce a deliverable project",
                f"{len(errors)} agent failure(s); see docs/errors.md",
                "Do not deploy",
            ]
            action_items = [
                "Inspect docs/errors.md and docs/recovery_report.md",
                "Verify GEMINI_API_KEY and model name in config.py",
                "Re-run after fixing root cause",
            ]

        for d in decisions:
            print(f"   - {d}")
        if action_items:
            print("   Action Items:")
            for a in action_items:
                print(f"   - {a}")

        meeting = Meeting(
            topic="Sprint Completion & Project Handoff",
            participants=list(AgentRole),
            decisions=decisions,
            action_items=action_items,
        )
        print(f"   Meeting concluded. {verdict}")
        return {"meetings": [meeting], "current_phase": "done"}

    async def run(self, project_name: str, raw_requirements: str, base_dir: str) -> Dict[str, Any]:
        """Run the multi-agent orchestrator."""
        config.BASE_DIR = base_dir
        # Reinitialize agents
        from agents import _get_agents
        AGENTS.update(_get_agents())

        print(f"\n{'='*70}")
        print(f"STARTING AI DEVELOPMENT PROJECT: {project_name}")
        print(f"Output directory: {base_dir}")
        print(f"Using Gemini 3.1 Pro for all 14 agents")
        print(f"Workflow: Kickoff -> PM -> Architect -> AI Architect")
        print(f"          -> conditional dev fan-out (AI track skipped for non-AI projects)")
        print(f"          -> Lead Review <-> Rework loop (max {config.MAX_REVIEW_ITERATIONS} cycles)")
        print(f"          -> Error Recovery -> QA + Security -> Integration -> Meeting")
        print(f"{'='*70}\n")

        initial_state: ProjectState = {
            "messages": [HumanMessage(content=raw_requirements)],
            "chat_log": [],
            "project_name": project_name,
            "requirements_raw": raw_requirements,
            "requirements_doc": None,
            "architecture_doc": None,
            "tasks": [],
            "completed_tasks": [],
            "code_artifacts": [],
            "review_comments": [],
            "meetings": [],
            "agent_responses": {},
            "current_phase": "start",
            "errors": [],
            "plan": {},
            "retry_counts": {},
            "error_log": [],
            "phase_gates": {},
            "sub_agent_requests": [],
            "rework_requests": {},
            "review_iteration": 0,
        }

        run_config = {
            "configurable": {"thread_id": f"dev_{project_name}"},
            "recursion_limit": 100
        }

        final_state = await self.app.ainvoke(initial_state, config=run_config)
        return final_state
