# ---------------------------------------------------------
# Step 25 :matching_agent.py
# LangGraph-based Agentic Profile Matching Agent
#
# Consolidates the full agent: state definition, all workflow
# nodes (parse, extract, search, rank, report, feedback), and
# the graph wiring with conditional loops. Includes a resilient
# multi-model LLM fallback and a safety check against bad/echoed
# LLM responses in the feedback loop.
# ---------------------------------------------------------

import os
import json
import time
import numpy as np
from datetime import datetime
from typing import TypedDict, List, Dict, Any, Annotated
import operator

from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, START, END


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
BASE_DIR = "matching_agent_project"
RESUMES_FOLDER = f"{BASE_DIR}/data/resumes"
REPORTS_FOLDER = f"{BASE_DIR}/reports"

# Free model availability on OpenRouter changes frequently and can be
# temporarily rate-limited/overloaded. We try a short list of candidates
# in order, retrying on temporary errors, and falling through to the
# next model on permanent unavailability (404s).
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
    raise RuntimeError("No candidate free models are currently available. Try again shortly, or check https://openrouter.ai/models?max_price=0")

llm = get_working_llm()

# Embedding model for RAG-style resume search (loaded once, reused
# across every call to search_resumes).
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


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
    response = llm.invoke(prompt)
    raw_output = response.content.strip()
    if raw_output.startswith("```"):
        raw_output = raw_output.strip("`").replace("json", "", 1).strip()
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        return {"must_have": [], "nice_to_have": []}


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
# Helper: load resumes from disk
# ---------------------------------------------------------
def load_resumes_from_folder(folder_path: str) -> List[Dict[str, str]]:
    resumes = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, "r") as f:
                text = f.read()
            resumes.append({"candidate_id": filename.replace(".txt", ""), "text": text})
    return resumes


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


def search_resumes(state: AgentState):
    print("Node running: Search Resumes")
    resumes = load_resumes_from_folder(RESUMES_FOLDER)
    if not resumes:
        note = {"role": "system", "content": "Warning: No resumes found to search."}
        return {"candidates": [], "conversation_history": [note], "current_step": "search_resumes"}

    reqs = state.get("requirements", {})
    query_text = " ".join(reqs.get("must_have", []) + reqs.get("nice_to_have", [])) or state.get("job_description", "")

    query_embedding = embedding_model.encode(query_text)
    resume_texts = [r["text"] for r in resumes]
    resume_embeddings = embedding_model.encode(resume_texts)

    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    scored_candidates = []
    for resume, embedding in zip(resumes, resume_embeddings):
        score = float(cosine_similarity(query_embedding, embedding))
        scored_candidates.append({"candidate_id": resume["candidate_id"], "text": resume["text"], "similarity_score": round(score, 4)})

    scored_candidates.sort(key=lambda c: c["similarity_score"], reverse=True)
    note = {"role": "system", "content": f"Found and scored {len(scored_candidates)} candidates."}
    return {"candidates": scored_candidates, "conversation_history": [note], "current_step": "search_resumes"}


def rank_candidates(state: AgentState):
    print("Node running: Rank Candidates")
    candidates = state.get("candidates", [])
    reqs = state.get("requirements", {})
    if not candidates:
        note = {"role": "system", "content": "Warning: No candidates available to rank."}
        return {"shortlist": [], "conversation_history": [note], "current_step": "rank_candidates"}

    candidates_block = ""
    for c in candidates:
        candidates_block += f"\nCandidate ID: {c['candidate_id']}\nResume:\n{c['text']}\n---"

    prompt = f"""
You are a recruiting assistant. Here are the job requirements:
Must-have: {reqs.get('must_have', [])}
Nice-to-have: {reqs.get('nice_to_have', [])}

Here are the candidates:
{candidates_block}

Rank these candidates from best to worst fit for the job.
Respond with ONLY valid JSON, a list of objects, in this exact format:
[
  {{"candidate_id": "...", "rank": 1, "score": 0-10, "reasoning": "short explanation"}},
  ...
]
No extra text outside the JSON.
"""
    response = llm.invoke(prompt)
    raw_output = response.content.strip()
    if raw_output.startswith("```"):
        raw_output = raw_output.strip("`").replace("json", "", 1).strip()
    try:
        shortlist = json.loads(raw_output)
    except json.JSONDecodeError:
        shortlist = []

    note = {"role": "system", "content": f"Ranked {len(shortlist)} candidates."}
    return {"shortlist": shortlist, "conversation_history": [note], "current_step": "rank_candidates"}


def generate_report(state: AgentState):
    print("Node running: Generate Report")
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
    os.makedirs(REPORTS_FOLDER, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"{REPORTS_FOLDER}/report_{timestamp}.txt"
    with open(report_filename, "w") as f:
        f.write(report_text)

    print(f"Report saved to: {report_filename}")
    note = {"role": "system", "content": f"Report generated and saved to {report_filename}."}
    return {"conversation_history": [note], "current_step": "generate_report"}


def human_feedback(state: AgentState):
    """
    Interprets the latest user message (question / update_criteria / end).
    Includes a safety check against bad/echoed LLM responses so the
    conversation doesn't end incorrectly due to a weak free-model reply.
    """
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

    # Safety check: if intent is "end" but the answer is empty or just
    # echoes the user's own message back, treat it as a misclassification
    # (a known failure mode with weaker free models) and default to
    # "question" instead of ending the conversation incorrectly.
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
# CLI entry point - allows running this file directly:
#   python matching_agent.py
# ---------------------------------------------------------
if __name__ == "__main__":
    app = build_graph()

    sample_jd = """
    We are hiring a Frontend Developer.
    Must have: React, 3+ years of experience.
    Nice to have: TypeScript, AWS knowledge.
    """

    state = {
        "conversation_history": [],
        "job_description": sample_jd,
        "requirements": {},
        "candidates": [],
        "shortlist": [],
        "current_step": "not_started",
        "last_intent": ""
    }

    state.update(parse_jd(state))
    state.update(extract_requirements_node(state))
    state.update(search_resumes(state))
    state.update(rank_candidates(state))
    state.update(generate_report(state))

    print("\nInitial screening complete! Type 'done' to exit.\n")
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        state["conversation_history"] = state["conversation_history"] + [{"role": "user", "content": user_input}]
        state.update(human_feedback(state))
        intent = state.get("last_intent", "end")

        if intent == "update_criteria":
            state.update(search_resumes(state))
            state.update(rank_candidates(state))
            state.update(generate_report(state))
            print("\nAgent: Requirements updated - rankings refreshed.\n")
        else:
            print(f"\nAgent: {state['conversation_history'][-1]['content']}\n")
            if intent == "end":
                break
