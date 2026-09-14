# ---------------------------------------------------------
# matching_agent.py (Refactored for MCP Integration - Part B)
#
# LangGraph-based Agentic Profile Matching Agent.
# Filesystem access has been REMOVED and replaced with calls
# to filesystem_mcp_server.py via an MCP client connection.
# ---------------------------------------------------------

import os
import sys
import json
import time
from datetime import datetime
from typing import TypedDict, List, Dict, Any, Annotated
import operator

from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, START, END

# MCP client imports - these replace direct filesystem access
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
MCP_SERVER_SCRIPT = "mcp_server/filesystem_mcp_server.py"
DATABASE_MCP_SERVER_SCRIPT = "mcp_server/database_mcp_server.py"

CANDIDATE_MODELS = [
    "openrouter/free",
]

def get_working_llm(max_retries_per_model=2, wait_seconds=5):
    for model_name in CANDIDATE_MODELS:
        for attempt in range(max_retries_per_model):
            try:
                candidate_llm = ChatOpenAI(
                    model=model_name,
                    openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
                    openai_api_base="https://openrouter.ai/api/v1",
                )
                candidate_llm.invoke("Say OK")
                print(f"Using model: {model_name}")
                return candidate_llm
            except Exception as e:
                err_msg = str(e)[:100]
                if "429" in err_msg or "overloaded" in err_msg.lower():
                    print(f"{model_name} temporarily busy (attempt {attempt+1}/{max_retries_per_model}), waiting {wait_seconds}s...")
                    time.sleep(wait_seconds)
                    continue
                else:
                    print(f"Model unavailable: {model_name} ({err_msg}...) - trying next.")
                    break
    raise RuntimeError("No candidate free models are currently available.")

llm = get_working_llm()
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


def safe_llm_invoke_json(prompt: str, default, max_retries: int = 3):
    """
    Call the LLM and parse its response as JSON, retrying a few times
    if the free/rotating model returns malformed output. Falls back
    to `default` only after all retries are exhausted.
    """
    for attempt in range(max_retries):
        response = llm.invoke(prompt)
        raw_output = response.content.strip()
        if raw_output.startswith("```"):
            raw_output = raw_output.strip("`").replace("json", "", 1).strip()
        try:
            return json.loads(raw_output)
        except json.JSONDecodeError:
            print(f"JSON parse failed (attempt {attempt+1}/{max_retries}), retrying...")
    print("All retries exhausted, using default fallback.")
    return default


# ---------------------------------------------------------
# Agent State
# ---------------------------------------------------------
class AgentState(TypedDict):
    conversation_history: Annotated[List[Dict[str, str]], operator.add]
    job_description: str
    requirements: Dict[str, Any]
    candidates: List[Dict[str, Any]]
    shortlist: List[Dict[str, Any]]
    current_step: str
    last_intent: str


# ---------------------------------------------------------
# MCP Client Connection
# ---------------------------------------------------------
# A persistent connection to filesystem_mcp_server.py, kept open
# for the lifetime of the agent's run (not reopened per tool call).

_mcp_exit_stack = AsyncExitStack()
_mcp_session = None
_db_mcp_session = None

async def get_mcp_session():
    """Lazily connect to the filesystem MCP server on first use, then reuse the session."""
    global _mcp_session
    if _mcp_session is not None:
        return _mcp_session

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[MCP_SERVER_SCRIPT],
    )
    errlog_file = open("mcp_server_stderr.log", "w")

    read, write = await _mcp_exit_stack.enter_async_context(
        stdio_client(server_params, errlog=errlog_file)
    )
    session = await _mcp_exit_stack.enter_async_context(
        ClientSession(read, write)
    )
    await session.initialize()

    _mcp_session = session
    return _mcp_session


async def get_db_mcp_session():
    """Lazily connect to the DATABASE MCP server (bonus: multi-MCP
    integration) on first use, then reuse the session."""
    global _db_mcp_session
    if _db_mcp_session is not None:
        return _db_mcp_session

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[DATABASE_MCP_SERVER_SCRIPT],
    )
    errlog_file = open("db_mcp_server_stderr.log", "w")

    read, write = await _mcp_exit_stack.enter_async_context(
        stdio_client(server_params, errlog=errlog_file)
    )
    session = await _mcp_exit_stack.enter_async_context(
        ClientSession(read, write)
    )
    await session.initialize()

    _db_mcp_session = session
    return _db_mcp_session


async def close_mcp_session():
    """Cleanly close both MCP server subprocesses and connections."""
    await _mcp_exit_stack.aclose()


# ---------------------------------------------------------
# Tools (as named in the assignment brief)
# ---------------------------------------------------------

def extract_requirements(jd: str) -> Dict[str, Any]:
    """Tool: Parse a job description into must-have vs nice-to-have requirements."""
    prompt = f"""
You are a recruiting assistant. Read the job description below and extract:
- "must_have": a list of required skills/experience (short phrases)
- "nice_to_have": a list of optional/preferred skills (short phrases)

Respond with ONLY valid JSON in this exact format, no extra text:
{{"must_have": ["..."], "nice_to_have": ["..."]}}

Job Description:
{jd}
"""
    return safe_llm_invoke_json(prompt, default={"must_have": [], "nice_to_have": []})


def compare_candidates(candidate_ids: List[str], all_candidates: List[Dict[str, Any]], requirements: Dict[str, Any]) -> str:
    """Tool: Head-to-head comparison of specific candidates by ID."""
    selected = [c for c in all_candidates if c["candidate_id"] in candidate_ids]
    if not selected:
        return "No matching candidates found for comparison."

    block = ""
    for c in selected:
        block += f"\nCandidate ID: {c['candidate_id']}\nResume:\n{c['text']}\n---"

    prompt = f"""
You are a recruiting assistant. Requirements:
Must-have: {requirements.get('must_have', [])}
Nice-to-have: {requirements.get('nice_to_have', [])}

Compare these candidates head-to-head:
{block}

Give a clear, short comparison highlighting strengths and gaps for each.
"""
    response = llm.invoke(prompt)
    return response.content.strip()


def generate_interview_questions(candidate_id: str, all_candidates: List[Dict[str, Any]], requirements: Dict[str, Any]) -> str:
    """Tool: Generate screening interview questions tailored to a specific candidate."""
    candidate = next((c for c in all_candidates if c["candidate_id"] == candidate_id), None)
    if not candidate:
        return f"Candidate {candidate_id} not found."

    prompt = f"""
You are a recruiting assistant. Job requirements:
Must-have: {requirements.get('must_have', [])}
Nice-to-have: {requirements.get('nice_to_have', [])}

Candidate resume:
{candidate['text']}

Generate 5 targeted screening interview questions for this candidate,
focused on verifying their fit against the requirements above.
"""
    response = llm.invoke(prompt)
    return response.content.strip()


# ---------------------------------------------------------
# Graph Nodes
# ---------------------------------------------------------

def parse_jd(state: AgentState):
    print("Node running: Parse JD")
    raw_jd = state.get("job_description", "")
    cleaned_jd = raw_jd.strip()
    note = {"role": "system", "content": "Job description received and parsed." if cleaned_jd else "Warning: No job description was provided."}
    return {"job_description": cleaned_jd, "conversation_history": [note], "current_step": "parse_jd"}


def extract_requirements_node(state: AgentState):
    print("Node running: Extract Requirements")
    jd_text = state.get("job_description", "")
    parsed_requirements = extract_requirements(jd_text)
    note = {"role": "system", "content": f"Requirements extracted: {parsed_requirements}"}
    return {"requirements": parsed_requirements, "conversation_history": [note], "current_step": "extract_requirements"}


async def search_resumes(state: AgentState):
    """MCP-refactored: reads resumes via the MCP server instead of
    calling load_resumes_from_folder() directly on the filesystem."""
    print("Node running: Search Resumes (via MCP)")
    session = await get_mcp_session()
    result = await session.call_tool("read_resumes", {})
    resumes = result.structured_content.get("result", [])

    if not resumes:
        note = {"role": "system", "content": "Warning: No resumes found to search."}
        return {"candidates": [], "conversation_history": [note], "current_step": "search_resumes"}

    note = {"role": "system", "content": f"Found {len(resumes)} resumes via MCP server."}
    return {"candidates": resumes, "conversation_history": [note], "current_step": "search_resumes"}


def rank_candidates(state: AgentState):
    print("Node running: Rank Candidates")
    candidates = state.get("candidates", [])
    reqs = state.get("requirements", {})
    if not candidates:
        note = {"role": "system", "content": "Warning: No candidates available to rank."}
        return {"shortlist": [], "conversation_history": [note], "current_step": "rank_candidates"}

    block = ""
    for c in candidates:
        block += f"\nCandidate ID: {c['candidate_id']}\nResume:\n{c['text']}\n---"

    prompt = f"""
You are a recruiting assistant. Requirements:
Must-have: {reqs.get('must_have', [])}
Nice-to-have: {reqs.get('nice_to_have', [])}

Candidates:
{block}

Rank ALL candidates from best to worst fit. Respond with ONLY valid JSON,
a list of objects in this exact format:
[
  {{"candidate_id": "...", "rank": 1, "score": 0-10, "reasoning": "..."}},
  ...
]
No extra text outside the JSON.
"""
    shortlist = safe_llm_invoke_json(prompt, default=[])

    note = {"role": "system", "content": f"Ranked {len(shortlist)} candidates."}
    return {"shortlist": shortlist, "conversation_history": [note], "current_step": "rank_candidates"}


async def generate_report(state: AgentState):
    """MCP-refactored: writes the report via the MCP server instead of
    writing directly to disk with open()."""
    print("Node running: Generate Report (via MCP)")
    shortlist = state.get("shortlist", [])
    reqs = state.get("requirements", {})
    if not shortlist:
        note = {"role": "system", "content": "Warning: No shortlist available to report on."}
        return {"conversation_history": [note], "current_step": "generate_report"}

    report_lines = ["=" * 50, "CANDIDATE MATCH REPORT", "=" * 50,
                    f"Must-have requirements: {reqs.get('must_have', [])}",
                    f"Nice-to-have requirements: {reqs.get('nice_to_have', [])}", "-" * 50]
    sorted_shortlist = sorted(shortlist, key=lambda c: c.get("rank", 999))
    for candidate in sorted_shortlist:
        report_lines.append(f"\nRank #{candidate.get('rank')}: {candidate.get('candidate_id')}")
        report_lines.append(f"Score: {candidate.get('score')}/10")
        report_lines.append(f"Reasoning: {candidate.get('reasoning')}")

    report_text = "\n".join(report_lines)

    # Filesystem MCP server: write the report file
    session = await get_mcp_session()
    result = await session.call_tool("write_report", {"report_text": report_text})
    report_filename = result.structured_content.get("result", "unknown_path")
    print(f"Report saved to: {report_filename}")

    # Database MCP server (bonus - multi-MCP): save each candidate's
    # screening result for historical tracking
    db_session = await get_db_mcp_session()
    for candidate in sorted_shortlist:
        decision = "Shortlisted" if candidate.get("score", 0) >= 7 else "Not shortlisted"
        await db_session.call_tool(
            "save_screening_result",
            {
                "candidate_id": candidate.get("candidate_id"),
                "score": candidate.get("score", 0),
                "decision": decision,
            },
        )
    print(f"Saved {len(sorted_shortlist)} screening result(s) to database via MCP")

    note = {"role": "system", "content": f"Report generated ({report_filename}) and screening results saved to database - both via MCP."}
    return {"conversation_history": [note], "current_step": "generate_report"}


def human_feedback(state: AgentState):
    print("Node running: Human Feedback Loop")
    user_messages = [m for m in state.get("conversation_history", []) if m.get("role") == "user"]
    if not user_messages:
        return {"current_step": "human_feedback"}

    latest_message = user_messages[-1]["content"]
    reqs = state.get("requirements", {})
    shortlist = state.get("shortlist", [])

    prompt = f"""
You are a recruiting assistant managing a conversation about candidate rankings.

Current requirements:
Must-have: {reqs.get('must_have', [])}
Nice-to-have: {reqs.get('nice_to_have', [])}

Current shortlist:
{json.dumps(shortlist, indent=2)}

The user just said: "{latest_message}"

Classify this message and respond with ONLY valid JSON in this exact format:
{{
  "intent": "question" or "update_criteria" or "end",
  "answer": "if intent is 'question', answer it here using the shortlist/reasoning above. Otherwise empty string.",
  "updated_requirements": {{"must_have": [...], "nice_to_have": [...]}}
}}

Rules:
- If the user is asking WHY something ranked a certain way, or asking to compare candidates, set intent to "question" and write a clear answer using the existing shortlist data.
- If the user wants to CHANGE requirements (add/remove/move must-have <-> nice-to-have), set intent to "update_criteria" and return the FULL updated requirements object.
- If the user says they're satisfied/done/thanks/no more questions, set intent to "end".
- No extra text outside the JSON.
"""
    response = llm.invoke(prompt)
    raw_output = response.content.strip()
    if raw_output.startswith("```"):
        raw_output = raw_output.strip("`").replace("json", "", 1).strip()
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError:
        parsed = {"intent": "end", "answer": "", "updated_requirements": reqs}

    intent = parsed.get("intent", "end")
    answer = (parsed.get("answer") or "").strip()

    if intent == "end" and (not answer or answer.strip().lower() == latest_message.strip().lower()):
        intent = "question"
        answer = "Sorry, I didn't quite catch that — could you rephrase your question?"

    note = {"role": "assistant", "content": answer or f"Understood - intent classified as: {intent}"}
    updates = {"conversation_history": [note], "current_step": "human_feedback", "last_intent": intent}
    if intent == "update_criteria":
        updates["requirements"] = parsed.get("updated_requirements", reqs)

    return updates


# ---------------------------------------------------------
# Graph wiring
# ---------------------------------------------------------

def route_after_feedback(state: AgentState):
    intent = state.get("last_intent", "end")
    if intent == "update_criteria":
        return "search_resumes"
    elif intent == "question":
        return "human_feedback"
    else:
        return "end"


def build_graph():
    """Builds and compiles the LangGraph workflow."""
    workflow = StateGraph(AgentState)

    workflow.add_node("parse_jd", parse_jd)
    workflow.add_node("extract_requirements", extract_requirements_node)
    workflow.add_node("search_resumes", search_resumes)
    workflow.add_node("rank_candidates", rank_candidates)
    workflow.add_node("generate_report", generate_report)
    workflow.add_node("human_feedback", human_feedback)

    workflow.add_edge(START, "parse_jd")
    workflow.add_edge("parse_jd", "extract_requirements")
    workflow.add_edge("extract_requirements", "search_resumes")
    workflow.add_edge("search_resumes", "rank_candidates")
    workflow.add_edge("rank_candidates", "generate_report")
    workflow.add_edge("generate_report", "human_feedback")

    workflow.add_conditional_edges(
        "human_feedback",
        route_after_feedback,
        {"search_resumes": "search_resumes", "human_feedback": "human_feedback", "end": END}
    )

    return workflow.compile()


# ---------------------------------------------------------
# CLI entry point - now async because search_resumes and
# generate_report are async (MCP calls). Run with: await main()
# ---------------------------------------------------------
async def main():
    app = build_graph()

    sample_jd = """
    We are hiring a Frontend Developer.
    Must have: React, 3+ years of experience.
    Nice to have: TypeScript, AWS knowledge.
    """

    initial_state = {
        "conversation_history": [],
        "job_description": sample_jd,
        "requirements": {},
        "candidates": [],
        "shortlist": [],
        "current_step": "not_started",
        "last_intent": ""
    }

    # ainvoke() runs the full graph asynchronously, handling both
    # sync and async nodes automatically
    final_state = await app.ainvoke(initial_state)
    print("\nInitial screening complete!")
    return final_state
