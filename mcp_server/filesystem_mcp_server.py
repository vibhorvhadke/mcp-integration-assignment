"""
filesystem_mcp_server.py

MCP (Model Context Protocol) server exposing filesystem tools
as standardized MCP resources.

Part A of the MCP Integration Assignment.
"""

import os
import time
import json
from datetime import datetime
from typing import List, Dict, Any

# MCPServer is the mcp v2.x class (renamed from FastMCP in v1.x)
# It lets us turn plain Python functions into MCP tools using
# the @mcp.tool() decorator, instead of manually implementing
# JSON-RPC 2.0 handling ourselves.
from mcp.server.mcpserver import MCPServer

# ---------------------------------------------------------
# Server instance
# ---------------------------------------------------------
# This single object is what exposes tools, handles requests,
# and eventually runs as the MCP server process.
mcp = MCPServer(name="filesystem-mcp-server")

# Paths to the folders our Milestone 1 agent already used
RESUMES_FOLDER = "matching_agent_project/data/resumes"
REPORTS_FOLDER = "matching_agent_project/reports"


# ---------------------------------------------------------
# Milestone 1 tools, converted to MCP tools
# ---------------------------------------------------------
# These two functions are direct MCP-ified versions of:
# - load_resumes_from_folder() from matching_agent.py
# - the file-writing block inside generate_report() from matching_agent.py
#
# The @mcp.tool() decorator is what makes these callable by an
# MCP client (like our future matching_agent.py refactor) using
# the standardized MCP protocol, instead of being called as
# regular in-process Python functions.

@mcp.tool()
def read_resumes(folder_path: str = RESUMES_FOLDER) -> List[Dict[str, str]]:
    """Read all .txt resumes from a folder and return their contents."""
    resumes = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, "r") as f:
                text = f.read()
            resumes.append({"candidate_id": filename.replace(".txt", ""), "text": text})
    return resumes


@mcp.tool()
def write_report(report_text: str, folder_path: str = REPORTS_FOLDER) -> str:
    """Write a report to a timestamped .txt file and return its path."""
    os.makedirs(folder_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"{folder_path}/report_{timestamp}.txt"
    with open(report_filename, "w") as f:
        f.write(report_text)
    return report_filename


# ---------------------------------------------------------
# New MCP-specific capabilities (assignment requirement)
# ---------------------------------------------------------
# The assignment explicitly asks for these two NEW capabilities
# that did NOT exist in Milestone 1 — they only make sense once
# your tools are exposed as a standing server, not simple functions.

@mcp.tool()
def watch_directory(folder_path: str = RESUMES_FOLDER, seconds: int = 5) -> List[str]:
    """Monitor a directory for new files for a given duration and return newly detected filenames."""
    # Snapshot 1: what files exist right now
    before = set(os.listdir(folder_path))

    # Wait for the given duration, simulating "watching" the folder
    time.sleep(seconds)

    # Snapshot 2: what files exist after waiting
    after = set(os.listdir(folder_path))

    # The difference tells us which files are new
    return list(after - before)


@mcp.tool()
def batch_process(folder_path: str = RESUMES_FOLDER) -> Dict[str, Any]:
    """Process multiple resume files at once, returning basic stats for each."""
    results = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, "r") as f:
                text = f.read()
            # Simple stats per file - a placeholder for whatever
            # batch operation you want to demonstrate (word count here)
            results[filename] = {"characters": len(text), "words": len(text.split())}
    return results


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------
# mcp.run() starts the server so it can listen for and respond
# to MCP protocol requests (e.g., from an MCP client in matching_agent.py)
if __name__ == "__main__":
    mcp.run()
