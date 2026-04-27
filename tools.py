import os
from pathlib import Path
from langchain_core.tools import tool
import config
import json
from datetime import datetime
from typing import Optional

def _base() -> Path:
    return Path(config.BASE_DIR)

def _ensure_dirs():
    for d in [config.BASE_DIR, f"{config.BASE_DIR}/src", f"{config.BASE_DIR}/tests",
              f"{config.BASE_DIR}/docs", f"{config.BASE_DIR}/logs"]:
        Path(d).mkdir(parents=True, exist_ok=True)

@tool
def write_file(file_path: str, content: str) -> str:
    """Write content to a file inside the project directory."""
    _ensure_dirs()
    full_path = _base() / file_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(content, encoding="utf-8")
    print(f"  [TOOL] write_file -> {full_path} ({len(content)} bytes)")
    return f"Successfully wrote {full_path}"

@tool
def append_file(file_path: str, content: str) -> str:
    """Append content to an existing file (create if not exists)."""
    _ensure_dirs()
    full_path = _base() / file_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    with open(full_path, 'a', encoding="utf-8") as f:
        f.write(content)
    print(f"  [TOOL] append_file -> {full_path} ({len(content)} bytes)")
    return f"Successfully appended to {full_path}"

@tool
def read_file(file_path: str) -> str:
    """Read content from a file."""
    full_path = _base() / file_path
    if not full_path.exists():
        print(f"  [TOOL] read_file -> {full_path} not found")
        return f"Error: File {full_path} not found."
    content = full_path.read_text(encoding="utf-8")
    print(f"  [TOOL] read_file -> {full_path} ({len(content)} bytes)")
    return content

@tool
def list_directory(dir_path: str = ".") -> str:
    """List all files/directories inside the given path."""
    full_path = _base() / dir_path
    if not full_path.exists():
        return f"Error: Directory {full_path} not found."
    items = "\n".join(str(p.relative_to(_base())) for p in full_path.iterdir())
    print(f"  [TOOL] list_directory -> {dir_path}: {len(items.splitlines())} items")
    return f"Contents of {dir_path}:\n{items}"

@tool
def search_files(pattern: str) -> str:
    """Search for files matching a pattern (e.g., '*.py', 'src/**/*.ts')."""
    _ensure_dirs()
    full_pattern = str(_base() / pattern)
    from pathlib import Path
    results = list(Path(_base()).glob(pattern))
    file_list = "\n".join(str(p.relative_to(_base())) for p in results)
    print(f"  [TOOL] search_files -> {pattern}: {len(results)} matches")
    return f"Found {len(results)} files:\n{file_list}"

@tool
def create_directory(dir_path: str) -> str:
    """Create a new directory."""
    full_path = _base() / dir_path
    full_path.mkdir(parents=True, exist_ok=True)
    print(f"  [TOOL] create_directory -> {full_path}")
    return f"Directory created: {full_path}"

@tool
def delete_file(file_path: str) -> str:
    """Delete a file."""
    full_path = _base() / file_path
    if full_path.exists() and full_path.is_file():
        full_path.unlink()
        print(f"  [TOOL] delete_file -> {full_path}")
        return f"Deleted {full_path}"
    return f"File {full_path} not found."

@tool
def create_plan(title: str, steps: list[str]) -> str:
    """Create a project plan with steps. `steps` is a list of step descriptions."""
    _ensure_dirs()
    plan_content = f"""# {title}
Created: {datetime.now().isoformat()}

## Plan Steps

"""
    for i, step in enumerate(steps, 1):
        plan_content += f"{i}. {step} - Status: PENDING\n"

    plan_path = _base() / "docs/plan.md"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(plan_content, encoding="utf-8")
    print(f"  [TOOL] create_plan -> {plan_path}")
    return f"Plan created with {len(steps)} steps"

@tool
def update_plan(step_number: int, status: str, notes: str = "") -> str:
    """Update a plan step status (PENDING, IN_PROGRESS, DONE, BLOCKED)."""
    plan_path = _base() / "docs/plan.md"
    if not plan_path.exists():
        return f"Error: Plan not found at {plan_path}"

    content = plan_path.read_text(encoding="utf-8")
    lines = content.split('\n')

    for i, line in enumerate(lines):
        if line.startswith(f"{step_number}."):
            lines[i] = f"{step_number}. {line.split(' - Status:')[0][len(str(step_number))+2:]} - Status: {status}"
            if notes:
                lines[i] += f" (Notes: {notes})"
            break

    plan_path.write_text('\n'.join(lines), encoding="utf-8")
    print(f"  [TOOL] update_plan -> Step {step_number} set to {status}")
    return f"Step {step_number} updated to {status}"

ALL_TOOLS = [write_file, read_file, append_file, list_directory, search_files,
             create_directory, delete_file, create_plan, update_plan]