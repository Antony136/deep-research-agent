"""
Main LangGraph workflow for the Deep Research Agent.

Human-in-the-loop checkpoints:
1. Initial research plan approval.
2. One-time authorization for initial web research.
3. Optional adaptive follow-up question approval.
4. Optional final report approval.

Observability:
- Records node execution start, completion, duration, and failures.
- Preserves existing graph routing and HITL behavior.
"""

from langgraph.graph import END, START, StateGraph

from app.graph.nodes.adaptive_planner import adaptive_planner_node
from app.graph.nodes.adaptive_review import adaptive_review_node
from app.graph.nodes.evidence_extractor import evidence_extractor_node
from app.graph.nodes.evidence_verifier import evidence_verifier_node
from app.graph.nodes.final_report_review import (
    final_report_review_node,
)
from app.graph.nodes.initial_plan_review import (
    initial_plan_review_node,
)
from app.graph.nodes.initial_research_review import (
    initial_research_review_node,
)
from app.graph.nodes.planner import planner_node
from app.graph.nodes.researcher import researcher_node
from app.graph.nodes.research_sufficiency import (
    research_sufficiency_node,
)
from app.graph.nodes.synthesizer import synthesis_node
from app.graph.state import ResearchState
from app.observability.tracing import instrument_node


# ============================================================
# INITIAL HUMAN REVIEW ROUTING
# ============================================================

def route_after_initial_plan_review(
    state: ResearchState,
) -> str:
    """Route according to the initial plan review decision."""

    if state.get("research_complete", False):
        print("\n[Router] Initial research plan was rejected.")
        return "end"

    if state.get("initial_plan_approved", False):
        print("\n[Router] Initial research plan approved.")
        return "research_authorization"

    print("\n[Router] Revised plan requires another review.")
    return "plan_review"


def route_after_initial_research_review(
    state: ResearchState,
) -> str:
    """Start web research only after authorization."""

    if state.get("research_complete", False):
        print("\n[Router] Research was stopped before starting.")
        return "end"

    if (
        state.get("initial_research_approved", False)
        and state.get("research_authorized", False)
    ):
        print("\n[Router] Web research authorized.")
        return "research"

    print("\n[Router] Web research was not authorized.")
    return "end"


# ============================================================
# RESEARCH SUFFICIENCY ROUTING
# ============================================================

def should_continue_research(
    state: ResearchState,
) -> str:
    """Decide whether to research, adapt the plan, or synthesize."""

    current_index = state["current_question_index"]
    total_questions = len(state["research_questions"])
    max_total_questions = state["max_total_research_questions"]

    if current_index < total_questions:
        print("\n[Router] Planned research questions remain.")
        print(
            f"[Router] Progress: {current_index}/{total_questions}"
        )
        return "research"

    if state["research_sufficient"]:
        print("\n[Router] Research is sufficient.")
        return "synthesis"

    if total_questions >= max_total_questions:
        print("\n[Router] Maximum question budget reached.")
        return "synthesis"

    if state["research_round"] < state["max_research_rounds"]:
        print("\n[Router] Research remains insufficient.")
        print("[Router] Starting adaptive planning.")
        return "adaptive_planner"

    print("\n[Router] Maximum research rounds reached.")
    return "synthesis"


# ============================================================
# ADAPTIVE REVIEW ROUTING
# ============================================================

def should_review_adaptive_questions(
    state: ResearchState,
) -> str:
    """Choose human review or automatic acceptance of proposals."""

    proposed_questions = state.get(
        "proposed_research_questions",
        [],
    )

    current_questions = state["research_questions"]
    max_total_questions = state["max_total_research_questions"]

    if not proposed_questions:
        print("\n[Router] No usable follow-up questions proposed.")
        return "synthesis"

    available_capacity = max_total_questions - len(current_questions)

    if available_capacity <= 0:
        print("\n[Router] No capacity for follow-up questions.")
        return "synthesis"

    if state.get("require_adaptive_research_approval", True):
        print("\n[Router] Follow-up questions need human review.")
        return "review"

    print("\n[Router] Automatically accepting valid follow-up questions.")
    return "auto_accept"


def auto_accept_adaptive_questions(
    state: ResearchState,
) -> dict:
    """
    Accept validated adaptive proposals when human approval
    is disabled, while enforcing the question budget.
    """

    current_questions = list(state["research_questions"])
    proposed_questions = state.get(
        "proposed_research_questions",
        [],
    )

    max_total_questions = state["max_total_research_questions"]
    capacity = max(0, max_total_questions - len(current_questions))
    accepted_questions = proposed_questions[:capacity]

    if not accepted_questions:
        return {
            "proposed_research_questions": [],
            "adaptive_review_decision": {
                "action": "reject",
                "reason": "No valid follow-up questions could be accepted.",
            },
        }

    updated_questions = current_questions + accepted_questions

    return {
        "research_questions": updated_questions,
        "current_question_index": len(current_questions),
        "proposed_research_questions": [],
        "adaptive_review_decision": {
            "action": "approve",
            "reason": "Follow-up questions accepted automatically.",
        },
        "research_complete": False,
    }


def should_continue_after_adaptive_review(
    state: ResearchState,
) -> str:
    """Continue only if adaptive questions were accepted."""

    decision = state.get("adaptive_review_decision") or {}

    if decision.get("action") != "approve":
        print("\n[Router] Follow-up questions were not approved.")
        return "synthesis"

    current_index = state["current_question_index"]
    total_questions = len(state["research_questions"])
    max_total_questions = state["max_total_research_questions"]

    if current_index >= total_questions:
        print("\n[Router] No approved follow-up questions remain.")
        return "synthesis"

    if total_questions > max_total_questions:
        print("\n[Router] Question budget exceeded.")
        return "synthesis"

    print("\n[Router] Approved follow-up questions will be researched.")
    print(
        f"[Router] Total questions: "
        f"{total_questions}/{max_total_questions}"
    )
    return "research"


# ============================================================
# FINAL REPORT REVIEW ROUTING
# ============================================================

def route_after_final_report_review(
    state: ResearchState,
) -> str:
    """Finish after approval or return to research when requested."""

    decision = state.get("final_report_review_decision")

    if state.get("final_report_approved", False):
        print("\n[Router] Final report approved.")
        return "end"

    if not decision:
        print("\n[Router] No final-report decision was recorded.")
        return "end"

    if decision.get("action") != "research_more":
        print("\n[Router] Final-report review did not request more research.")
        return "end"

    total_questions = len(state["research_questions"])
    max_total_questions = state["max_total_research_questions"]
    research_round = state["research_round"]
    max_research_rounds = state["max_research_rounds"]

    if (
        total_questions < max_total_questions
        and research_round < max_research_rounds
    ):
        print("\n[Router] Returning to adaptive research planning.")
        return "adaptive_planner"

    print("\n[Router] Research budgets exhausted.")
    print("[Router] Ending with the current report.")
    return "end"


# ============================================================
# GRAPH CONSTRUCTION
# ============================================================

def build_research_graph(checkpointer=None):
    """
    Build and compile the complete research workflow.

    Each executable node is wrapped with observability tracing.
    Routing decisions remain unchanged.

    A checkpointer is required for resumable human-in-the-loop
    interrupts across graph invocations.
    """

    graph = StateGraph(ResearchState)

    # Register instrumented nodes.
    graph.add_node(
        "planner",
        instrument_node("planner", planner_node),
    )
    graph.add_node(
        "initial_plan_review",
        instrument_node(
            "initial_plan_review",
            initial_plan_review_node,
        ),
    )
    graph.add_node(
        "initial_research_review",
        instrument_node(
            "initial_research_review",
            initial_research_review_node,
        ),
    )
    graph.add_node(
        "researcher",
        instrument_node("researcher", researcher_node),
    )
    graph.add_node(
        "evidence_extractor",
        instrument_node(
            "evidence_extractor",
            evidence_extractor_node,
        ),
    )
    graph.add_node(
        "evidence_verifier",
        instrument_node(
            "evidence_verifier",
            evidence_verifier_node,
        ),
    )
    graph.add_node(
        "research_sufficiency",
        instrument_node(
            "research_sufficiency",
            research_sufficiency_node,
        ),
    )
    graph.add_node(
        "adaptive_planner",
        instrument_node(
            "adaptive_planner",
            adaptive_planner_node,
        ),
    )
    graph.add_node(
        "adaptive_review",
        instrument_node(
            "adaptive_review",
            adaptive_review_node,
        ),
    )
    graph.add_node(
        "auto_accept_adaptive",
        instrument_node(
            "auto_accept_adaptive",
            auto_accept_adaptive_questions,
        ),
    )
    graph.add_node(
        "synthesizer",
        instrument_node("synthesizer", synthesis_node),
    )
    graph.add_node(
        "final_report_review",
        instrument_node(
            "final_report_review",
            final_report_review_node,
        ),
    )

    # --------------------------------------------------------
    # Initial plan and research authorization
    # --------------------------------------------------------

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "initial_plan_review")

    graph.add_conditional_edges(
        "initial_plan_review",
        route_after_initial_plan_review,
        {
            "plan_review": "initial_plan_review",
            "research_authorization": "initial_research_review",
            "end": END,
        },
    )

    graph.add_conditional_edges(
        "initial_research_review",
        route_after_initial_research_review,
        {
            "research": "researcher",
            "end": END,
        },
    )

    # --------------------------------------------------------
    # Research and evidence processing
    # --------------------------------------------------------

    graph.add_edge("researcher", "evidence_extractor")
    graph.add_edge("evidence_extractor", "evidence_verifier")
    graph.add_edge("evidence_verifier", "research_sufficiency")

    graph.add_conditional_edges(
        "research_sufficiency",
        should_continue_research,
        {
            "research": "researcher",
            "adaptive_planner": "adaptive_planner",
            "synthesis": "synthesizer",
        },
    )

    # --------------------------------------------------------
    # Adaptive planning and approval
    # --------------------------------------------------------

    graph.add_conditional_edges(
        "adaptive_planner",
        should_review_adaptive_questions,
        {
            "review": "adaptive_review",
            "auto_accept": "auto_accept_adaptive",
            "synthesis": "synthesizer",
        },
    )

    graph.add_conditional_edges(
        "adaptive_review",
        should_continue_after_adaptive_review,
        {
            "research": "researcher",
            "synthesis": "synthesizer",
        },
    )

    graph.add_conditional_edges(
        "auto_accept_adaptive",
        should_continue_after_adaptive_review,
        {
            "research": "researcher",
            "synthesis": "synthesizer",
        },
    )

    # --------------------------------------------------------
    # Synthesis and optional final report approval
    # --------------------------------------------------------

    graph.add_edge("synthesizer", "final_report_review")

    graph.add_conditional_edges(
        "final_report_review",
        route_after_final_report_review,
        {
            "adaptive_planner": "adaptive_planner",
            "end": END,
        },
    )

    return graph.compile(checkpointer=checkpointer)