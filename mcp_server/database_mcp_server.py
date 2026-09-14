"""
database_mcp_server.py

MCP (Model Context Protocol) server exposing a simple SQLite
database for storing and retrieving candidate screening history.

Bonus: Multi-MCP Integration for the MCP Integration Assignment.
"""

import sqlite3
from typing import List, Dict, Any

from mcp.server.mcpserver import MCPServer

# ---------------------------------------------------------
# Server instance
# ---------------------------------------------------------
mcp = MCPServer(name="database-mcp-server")

DB_PATH = "matching_agent_project/screening_history.db"


def get_connection():
    """Open a connection and ensure the table exists."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS screening_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_id TEXT NOT NULL,
            score INTEGER,
            decision TEXT,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    return conn


# ---------------------------------------------------------
# Database tools
# ---------------------------------------------------------

@mcp.tool()
def save_screening_result(candidate_id: str, score: int, decision: str) -> str:
    """Save a candidate's screening result (score + decision) to the database."""
    conn = get_connection()
    conn.execute(
        "INSERT INTO screening_history (candidate_id, score, decision) VALUES (?, ?, ?)",
        (candidate_id, score, decision),
    )
    conn.commit()
    conn.close()
    return f"Saved screening result for {candidate_id}"


@mcp.tool()
def get_candidate_history(candidate_id: str) -> List[Dict[str, Any]]:
    """Retrieve all past screening records for a given candidate."""
    conn = get_connection()
    cursor = conn.execute(
        "SELECT candidate_id, score, decision, timestamp FROM screening_history WHERE candidate_id = ? ORDER BY timestamp DESC",
        (candidate_id,),
    )
    rows = cursor.fetchall()
    conn.close()
    return [
        {"candidate_id": r[0], "score": r[1], "decision": r[2], "timestamp": r[3]}
        for r in rows
    ]


@mcp.tool()
def list_all_screenings() -> List[Dict[str, Any]]:
    """Retrieve all screening records across all candidates."""
    conn = get_connection()
    cursor = conn.execute(
        "SELECT candidate_id, score, decision, timestamp FROM screening_history ORDER BY timestamp DESC"
    )
    rows = cursor.fetchall()
    conn.close()
    return [
        {"candidate_id": r[0], "score": r[1], "decision": r[2], "timestamp": r[3]}
        for r in rows
    ]


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    mcp.run()
