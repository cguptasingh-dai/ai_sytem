import asyncio
import sys
import os
import warnings

# Silence Pydantic V1 compat warning emitted by langchain-core on Python 3.14.
# The compat layer still works; the warning just clutters output.
warnings.filterwarnings(
    "ignore",
    message=r"Core Pydantic V1 functionality isn't compatible with Python 3\.14.*",
    category=UserWarning,
)
# Belt-and-braces: silence any langchain_core deprecation warnings at startup.
warnings.filterwarnings("ignore", module=r"langchain_core\..*")

# Force UTF-8 stdout so Unicode chars (emoji, arrows, em-dashes) print on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from graph import DevTeamOrchestrator
from config import GEMINI_API_KEY


def _validate_api_key() -> None:
    """Fail loudly at startup if the API key is missing or still a placeholder.

    Without this, every agent silently hits an auth error and falls through to
    the graceful-fallback path, so the workflow `succeeds` but writes no code.
    """
    placeholder_markers = ("your_", "_here", "<", "xxxx", "REPLACE")
    key = (GEMINI_API_KEY or "").strip()
    looks_placeholder = (not key) or any(m in key.lower() for m in (s.lower() for s in placeholder_markers))
    if looks_placeholder:
        print("="*70)
        print("ERROR: GEMINI_API_KEY is missing or still a placeholder.")
        print("="*70)
        print("Edit the .env file in this directory and set your real key, e.g.:")
        print("  GEMINI_API_KEY=AIzaSy...your_real_key...")
        print()
        print("Get a key from: https://aistudio.google.com/apikey")
        print("="*70)
        sys.exit(1)
    if not key.startswith(("AIza", "sk-")):
        print(f"WARNING: GEMINI_API_KEY does not look like a typical Google AI key "
              f"(prefix: {key[:6]}...). Continuing anyway.")

def get_multiline_input(prompt: str) -> str:
    print(prompt)
    print("(Enter your requirements, then press Ctrl+D or an empty line + Enter to finish)")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "":
            break
        lines.append(line)
    return "\n".join(lines)

async def main():
    _validate_api_key()
    print("\n" + "="*70)
    print("AI SOFTWARE ENGINEERING TEAM - Gemini 3.1 Pro")
    print("14 Agents | Conditional Workflow | Lead Review Loop | LangGraph")
    print("="*70 + "\n")
    
    project_name = input("Project name: ").strip()
    if not project_name:
        project_name = "MyProject"
    
    requirements = get_multiline_input("\nDescribe your project requirements (features, tech stack, goals):")
    if not requirements:
        requirements = "Build a REST API with user auth, CRUD operations, and AI features."
    
    default_path = os.path.join(os.getcwd(), project_name.replace(" ", "_"))
    print(f"\nLocal directory path (press Enter to use '{default_path}'):")
    user_path = input().strip().strip('"').strip("'")
    project_path = user_path if user_path else default_path
    project_path = os.path.abspath(project_path)
    
    print(f"\nStarting development in: {project_path}")
    print(f"Project: {project_name}\n")
    
    orchestrator = DevTeamOrchestrator()
    result = await orchestrator.run(project_name, requirements, project_path)
    
    print("\n" + "="*70)
    print("ALL TASKS COMPLETED")
    print(f"Project files written to: {project_path}")
    print("Final metrics:")
    print(f"   - Code artifacts: {len(result.get('code_artifacts', []))}")
    print(f"   - Meetings held: {len(result.get('meetings', []))}")
    print(f"   - Total agent responses: {len(result.get('agent_responses', {}))}")
    print("="*70)

if __name__ == "__main__":
    asyncio.run(main())