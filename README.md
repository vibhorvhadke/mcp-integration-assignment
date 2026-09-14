# MCP Integration Assignment

An MCP (Model Context Protocol) based refactor of a LangGraph resume-matching agent, converting custom filesystem tools into standardized MCP servers and integrating a second MCP server for bonus multi-MCP functionality.

## Learning Objectives Covered

- Understand Model Context Protocol
- Replace custom tools with MCP servers
- Implement standardized tool interfaces
- Deploy production-ready systems

## Project Structuremcp-integration-assignment/
├── agent/
│ └── matching_agent.py # LangGraph agent, refactored to use MCP clients
├── mcp_server/
│ ├── filesystem_mcp_server.py # Part A: filesystem tools exposed as MCP
│ └── database_mcp_server.py # Bonus: SQLite tools exposed as MCP
├── tests/
│ └── test_scenarios.py # Test scenarios for all MCP tools + full workflow
├── docs/
│ └── state_machine_diagram.png # LangGraph workflow diagram
├── matching_agent_project/
│ ├── data/resumes/ # Sample candidate resumes (.txt)
│ ├── reports/ # Generated match reports (via MCP)
│ └── screening_history.db # SQLite database (via MCP)
└── README.md

## Part A: MCP Server Implementation

`mcp_server/filesystem_mcp_server.py` exposes the following tools via the `mcp` SDK's `MCPServer`/`@mcp.tool()` decorator, JSON-RPC 2.0 compliant under the hood:

| Tool | Description |
|---|---|
| `read_resumes` | Reads all `.txt` resumes from the resumes folder |
| `write_report` | Writes a candidate match report to a timestamped `.txt` file |
| `watch_directory` | Monitors a directory for new files over a given time window |
| `batch_process` | Processes multiple resume files at once, returning per-file stats |

These replace the original Milestone 1 direct filesystem calls (`load_resumes_from_folder()` and the file-write block inside `generate_report()`).

## Part B: Agent Refactoring

`agent/matching_agent.py` no longer touches the filesystem directly. Instead:

- `search_resumes` (now `async`) calls `read_resumes` via an MCP client (`ClientSession`) over stdio transport
- `generate_report` (now `async`) calls `write_report` via the same MCP client
- The MCP server is launched as a subprocess and connected to using `StdioServerParameters` + `stdio_client`
- A persistent session (`AsyncExitStack`) is reused across the agent's lifetime rather than reconnecting per call

The rest of the LangGraph workflow (`parse_jd`, `extract_requirements`, `rank_candidates`, `human_feedback`) is unchanged, since these nodes never touched the filesystem.

### Reliability improvement

Since this project uses OpenRouter's free-tier router model (`openrouter/free`), which occasionally returns malformed JSON, a `safe_llm_invoke_json()` helper was added with automatic retries (up to 3 attempts) before falling back to a safe default. This is used by `extract_requirements` and `rank_candidates`.

## Bonus: Multi-MCP Integration

`mcp_server/database_mcp_server.py` is a second, independent MCP server exposing SQLite-backed tools:

| Tool | Description |
|---|---|
| `save_screening_result` | Saves a candidate's score and decision to the database |
| `get_candidate_history` | Retrieves past screening records for a candidate |
| `list_all_screenings` | Retrieves all screening records |

`matching_agent.py` connects to **both** MCP servers simultaneously. After generating the match report (via the filesystem MCP server), `generate_report` also saves each candidate's screening result to the database (via the database MCP server) — demonstrating an agent orchestrating multiple independent MCP resources in a single workflow.

## Setup & Running (Google Colab)

1. Clone this repo into Colab:
```python
   !git clone https://github.com/vibhorvhadke/mcp-integration-assignment.git
   %cd mcp-integration-assignment
```

2. Install dependencies:
```python
   !pip install mcp langgraph langchain-core langchain-openai sentence-transformers -q
```

3. Set your OpenRouter API key:
```python
   import os
   from getpass import getpass
   os.environ["OPENROUTER_API_KEY"] = getpass("Enter your OpenRouter API key: ")
```

4. Run the agent:
```python
   import sys
   sys.path.append("agent")
   import matching_agent
   final_state = await matching_agent.main()
```

## Running Tests

```python
import sys
sys.path.append("mcp_server")
sys.path.append("tests")

import filesystem_mcp_server
import matching_agent
import test_scenarios

# mcp_session: a ClientSession connected to filesystem_mcp_server.py
# db_session: a ClientSession connected to database_mcp_server.py
await test_scenarios.run_all_tests(mcp_session, matching_agent)
```

All test scenarios pass, covering:
- Each filesystem MCP tool individually (`read_resumes`, `batch_process`, `write_report`, `watch_directory`)
- The database MCP server (`save_screening_result` + `get_candidate_history`)
- A full end-to-end agent workflow run

## Workflow Diagram

See `docs/state_machine_diagram.png` for the LangGraph state machine, showing the flow:
start → parse_jd → extract_requirements → search_resumes (MCP)
→ rank_candidates → generate_report (MCP x2) → human_feedback
→ (loop back to search_resumes on updated criteria, or → end)

## Known Limitations

- Uses OpenRouter's free-tier `openrouter/free` router model, which has variable availability and occasionally returns non-JSON responses (mitigated with retry logic, but not 100% eliminated)
- SQLite database is local to the Colab runtime and resets if the runtime is deleted (data is preserved in the repo only via committed `.db` file snapshots)

## Author

Vibhor Vhadke
