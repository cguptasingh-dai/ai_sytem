from typing import Annotated, List, Dict, Any, Optional, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from enum import Enum
import operator
from datetime import datetime

class AgentRole(str, Enum):
    LEAD_ENGINEER = "lead_engineer"
    AI_ARCHITECT = "ai_architect"
    PM = "product_manager"
    ARCHITECT = "system_architect"
    BACKEND = "backend_dev"
    FRONTEND = "frontend_dev"
    QA = "qa_engineer"
    SECURITY = "security_reviewer"
    DEVOPS = "devops_engineer"
    AI_DEV = "ai_developer"
    DATA_ENGINEER = "data_engineer"
    ML_RESEARCHER = "ml_research_engineer"
    PROMPT_ENGINEER = "prompt_engineer"
    MLOPS = "mlops_engineer"

class TaskStatus(str, Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    BLOCKED = "blocked"

class Task(BaseModel):
    id: str
    title: str
    description: str
    assigned_to: AgentRole
    status: TaskStatus = TaskStatus.PENDING
    output: Any = None
    dependencies: List[str] = []
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None

class CodeArtifact(BaseModel):
    file_path: str
    content: str
    language: str
    description: str
    agent: AgentRole

class Message(BaseModel):
    sender: AgentRole
    receiver: Optional[AgentRole]
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class Meeting(BaseModel):
    topic: str
    participants: List[AgentRole]
    decisions: List[str]
    # Sprint action items are free-form blurbs (e.g. "Deploy to staging"),
    # not full Task objects with id/assigned_to. Keep it as a string list so
    # the meeting node can populate it without fabricating Task plumbing.
    action_items: List[str]
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class ProjectState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    chat_log: Annotated[List[Message], operator.add]
    project_name: str
    requirements_raw: str
    requirements_doc: Optional[str]
    architecture_doc: Optional[str]
    tasks: Annotated[List[Task], operator.add]
    completed_tasks: Annotated[List[Task], operator.add]
    code_artifacts: Annotated[List[CodeArtifact], operator.add]
    review_comments: Annotated[List[Dict], operator.add]
    meetings: Annotated[List[Meeting], operator.add]
    agent_responses: Annotated[Dict[str, str], lambda a, b: {**a, **b}]
    current_phase: Annotated[str, lambda a, b: b or a]
    errors: Annotated[List[str], operator.add]
    plan: Annotated[Dict[str, Any], lambda a, b: {**a, **b}]
    retry_counts: Annotated[Dict[str, int], lambda a, b: {**a, **b}]
    error_log: Annotated[List[Dict], operator.add]
    phase_gates: Annotated[Dict[str, bool], lambda a, b: {**a, **b}]
    sub_agent_requests: Annotated[List[Dict], operator.add]
    # Feedback loop: lead_review writes a {role: feedback} map to request rework
    # from specific dev agents. Cleared (set to {}) after rework completes.
    rework_requests: Annotated[Dict[str, str], lambda a, b: b if b is not None else a]
    # Counts how many rework cycles have run; capped by MAX_REVIEW_ITERATIONS.
    review_iteration: Annotated[int, lambda a, b: b if b is not None else a]