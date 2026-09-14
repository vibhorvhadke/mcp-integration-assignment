"""
test_scenarios.py

Test scenarios demonstrating MCP resource usage and the full
agent workflow, as required by the assignment deliverables.

Run inside the Colab notebook (not standalone), since it depends
on the notebook's existing mcp_session and matching_agent imports.
"""

import asyncio


async def test_read_resumes(mcp_session):
    print("\n[TEST] read_resumes")
    result = await mcp_session.call_tool("read_resumes", {})
    resumes = result.structured_content.get("result", [])
    assert len(resumes) > 0, "Expected at least one resume"
    print(f"PASSED - Found {len(resumes)} resumes")


async def test_batch_process(mcp_session):
    print("\n[TEST] batch_process")
    result = await mcp_session.call_tool("batch_process", {})
    stats = result.structured_content.get("result", {})
    assert len(stats) > 0, "Expected stats for at least one file"
    print(f"PASSED - Got stats for {len(stats)} files")


async def test_write_report(mcp_session):
    print("\n[TEST] write_report")
    result = await mcp_session.call_tool(
        "write_report", {"report_text": "Test scenario report."}
    )
    path = result.structured_content.get("result", "")
    assert path.endswith(".txt"), "Expected a .txt file path back"
    print(f"PASSED - Report written to {path}")


async def test_watch_directory(mcp_session):
    print("\n[TEST] watch_directory (2 second window)")
    result = await mcp_session.call_tool(
        "watch_directory", {"seconds": 2}
    )
    new_files = result.structured_content.get("result", [])
    print(f"PASSED - No crash; detected {len(new_files)} new file(s) in window")


async def test_full_agent_workflow(matching_agent_module):
    print("\n[TEST] Full agent workflow (parse -> ... -> report)")
    final_state = await matching_agent_module.main()
    assert final_state.get("shortlist"), "Expected a non-empty shortlist"
    print(f"PASSED - Agent produced a shortlist of {len(final_state['shortlist'])} candidates")


async def run_all_tests(mcp_session, matching_agent_module):
    await test_read_resumes(mcp_session)
    await test_batch_process(mcp_session)
    await test_write_report(mcp_session)
    await test_watch_directory(mcp_session)
    await test_full_agent_workflow(matching_agent_module)
    print("\nAll test scenarios passed!")


async def test_database_mcp_server(db_session):
    print("\n[TEST] database MCP server (save + retrieve)")
    await db_session.call_tool(
        "save_screening_result",
        {"candidate_id": "test_candidate", "score": 8, "decision": "Shortlisted"}
    )
    result = await db_session.call_tool(
        "get_candidate_history", {"candidate_id": "test_candidate"}
    )
    history = result.structured_content.get("result", [])
    assert len(history) > 0, "Expected at least one history record"
    print(f"PASSED - Retrieved {len(history)} history record(s) for test_candidate")
