# Quick Start Guide - Deep Multi-Agent System

## Setup

### 1. Environment
```bash
# Copy .env file and add your Gemini API key
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

### 2. Dependencies (Already installed)
```bash
pip install -r requirements.txt
```

---

## Running the System

### Basic Usage
```bash
python main.py
```

### Interactive Prompts
```
Project name: MyAwesomeApp
Describe your project requirements:
  Build a REST API with user authentication, JWT tokens,
  CRUD operations for tasks, and AI-powered task summarization
  using Gemini models.

(Press Ctrl+D or empty line to finish)

Local directory path (press Enter to use default):
./my_awesome_app
```

---

## What Happens

### Automated Workflow
1. **AI Lead Engineer** 🏆
   - Sets up team
   - Creates initial plan

2. **Product Manager** 📋
   - Analyzes requirements
   - Creates detailed PRD
   - Generates user stories

3. **System Architect** 🎯
   - Designs high-level architecture
   - Approves technology choices

4. **AI Architect** 🏗️
   - Creates detailed component design
   - Defines API contracts
   - Designs database schema

5. **Parallel Development** (Simultaneous):
   - **Backend Developer** 💾: APIs, database, business logic
   - **Frontend Developer** 🎨: UI components, state management
   - **AI Developer** 🤖: Gemini integration, RAG systems
   - **DevOps Engineer** ⚙️: Docker, CI/CD, infrastructure

6. **Lead Engineer Review** 🏆
   - Code quality gate
   - Architecture compliance check
   - Error detection

7. **Parallel Testing & Security** (Simultaneous):
   - **QA Engineer** ✅: Automated tests, test plans
   - **Security Reviewer** 🔐: Vulnerability audit, hardening

8. **Integration** 🔗
   - Combines all artifacts
   - Generates comprehensive README

9. **Sprint Review** 📊
   - Team meeting
   - Project approval
   - Handoff documentation

---

## Generated Artifacts

### Code
- `src/backend/` - API implementation
- `src/frontend/` - UI components
- `src/ai/` - LLM integrations
- `tests/` - Comprehensive test suites
- `Dockerfile` - Container configuration
- CI/CD pipelines - `.github/workflows/`

### Documentation
- `docs/PRD.md` - Product requirements
- `docs/architecture.md` - System design
- `docs/api_spec.md` - API documentation
- `docs/database_schema.md` - Database design
- `docs/security_audit.md` - Security findings
- `docs/plan.md` - Project plan
- `README.md` - Integration summary

---

## Key Features

### ✅ Deep Agents
Each agent has:
- Rich, role-specific system prompt
- Planning capabilities
- Error handling with retry logic
- Full file system access
- Comprehensive context from project state

### ✅ Workflow
- Phase-based execution
- Parallel processing where possible
- Error recovery mechanism
- Quality gates at each phase

### ✅ Error Handling
```
Retry Policy:
  - Attempt 1: immediate
  - Attempt 2: after 2 seconds
  - Attempt 3: after 4 seconds
  - Failed: logged and documented
```

### ✅ Tools
- 9 file system tools
- 2 planning tools
- All tools available to every agent
- Full read/write/search capabilities

---

## Example Output

```
======================================================================
STARTING AI DEVELOPMENT PROJECT: MyApp
Output directory: /path/to/MyApp
Using Gemini 1.5 Pro for all 10 agents
Workflow: Kickoff → PM → Architect → Parallel Dev → Review → QA/Security → Integration
======================================================================

======================================================================
PHASE 0: LEAD ENGINEER KICKOFF
======================================================================
[LEAD_ENGINEER] Starting task: Team Kickoff and Project Planning
   [TOOL CALL] create_plan with args: {"title": "MyApp Team Plan", ...}
   Response preview: Team setup complete. 5 engineers assigned...

======================================================================
PHASE 1: PRODUCT MANAGER - Requirements Analysis
======================================================================
[PRODUCT_MANAGER] Starting task: Create PRD and Requirements
   [TOOL CALL] write_file with args: {"file_path": "docs/PRD.md", ...}
   [TOOL RESULT] Successfully wrote /path/to/MyApp/docs/PRD.md

...

======================================================================
ALL TASKS COMPLETED
======================================================================
Project files written to: /path/to/MyApp
Final metrics:
   - Code artifacts: 25
   - Meetings held: 1
   - Total agent responses: 10
```

---

## Monitoring Execution

### Console Output
- Agent name and role shown in brackets: `[BACKEND]`
- Tool calls displayed with arguments
- Response previews (first 200 chars)
- Execution time per agent
- Phase transitions clearly marked

### Error Tracking
- Errors displayed in console
- Error log in state
- Retry counts tracked
- Detailed error info stored

### Project State
- Current phase updated after each agent
- Phase gates set to track progress
- Retry counts maintained per agent
- Complete error history

---

## Customization

### Modify Agent Prompts
Edit `agents.py` - each agent class has its system prompt

### Add New Tools
Edit `tools.py` - add new @tool decorated functions

### Change Workflow
Edit `graph.py` - modify `_build_graph()` method

### Adjust Retry Logic
Edit `config.py`:
```python
MAX_RETRIES = 3      # Change retry attempts
RETRY_DELAY = 2      # Change delay between retries
```

---

## Troubleshooting

### API Key Issues
```bash
# Verify GEMINI_API_KEY is set
echo $GEMINI_API_KEY
```

### File Permission Errors
Ensure write access to output directory:
```bash
chmod -R u+w ./generated_project
```

### Agent Failures
Check console output for:
- Tool execution errors
- API timeouts
- Retry attempts and final failure message

### Model Errors
If Gemini 1.5 Pro is unavailable:
- Check your API quota
- Verify API key validity
- Check rate limits

---

## Performance Tips

### Faster Execution
1. Smaller projects: Fewer agents doing work
2. Simpler requirements: Less complex designs
3. Parallel phases: Naturally faster than sequential

### Monitor Progress
- Watch console output for phase transitions
- Phase gates in state show completion
- Agent responses logged in state

### Resource Usage
- Typical run: 2-5 minutes for complete project
- Each agent: 10-30 seconds depending on task
- Disk usage: ~5-20 MB per generated project

---

## Next Steps

### After Generation
1. **Review Generated Code**
   ```bash
   cd ./generated_project
   find . -type f -name "*.py" | head -10
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt  # if generated
   ```

3. **Run Tests**
   ```bash
   pytest tests/
   ```

4. **Build Docker Image**
   ```bash
   docker build -t myapp .
   ```

5. **Deploy**
   ```bash
   docker-compose up
   ```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────┐
│         DevTeamOrchestrator (LangGraph)            │
│         Manages 10-Agent Workflow                   │
└─────────────────────────────────────────────────────┘
              │
    ┌─────────┴──────────────────────────┬─────────────┐
    │    Phase-Based Execution           │ Error       │
    │    with Quality Gates              │ Recovery    │
    │                                    │ & Retry     │
    │ Kickoff → PM → Architects           │             │
    │      ↓                              │             │
    │ Parallel Dev (4 agents)  ─────→  Lead Review    │
    │      ↓                              │             │
    │ Parallel Test (2 agents)  ────→   Integration   │
    │      ↓                              │             │
    │ Final Meeting                       │             │
    └────────────────────────────────────┴─────────────┘

┌─────────────────────────────────────────────────────┐
│    10 Specialized Deep Agents                       │
│  ┌──────┬──────┬──────┬────────┬─────────┐         │
│  │Lead  │AI    │PM    │Arch    │Backend  │         │
│  │Eng   │Arch  │      │        │         │         │
│  ├──────┼──────┼──────┼────────┼─────────┤         │
│  │Front │AI    │DevOps│QA      │Security │         │
│  │end   │Dev   │      │        │         │         │
│  └──────┴──────┴──────┴────────┴─────────┘         │
└─────────────────────────────────────────────────────┘

Each Agent:
  ✓ Rich System Prompt
  ✓ Planning Tools
  ✓ Error Handling
  ✓ Full File Access
  ✓ Rich Context
```

---

## Support & Debugging

### Enable Detailed Logging
Add to `main.py`:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Generated Project
```bash
tree ./generated_project -L 2
```

### Review Phase Gates
Check `final_state["phase_gates"]` for completion status

### Inspect Error Log
```python
print(final_state["error_log"])
print(final_state["retry_counts"])
```

---

**Status**: Ready to generate AI-powered projects! 🚀
