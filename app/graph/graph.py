"""
Main LangGraph workflow for the Deep Research Agent.

Current workflow:

    START
      |
      v
   Planner
      |
      v
  Researcher
      |
      v
Evidence Extractor
      |
      v
Evidence Verifier
      |
      v
Research Sufficiency
      |
      v
 Planned questions remain?
    /              \
  yes               no
   |                 |
   v                 v
Researcher       Is research sufficient?
   |              /              \
   |            yes               no
   |             |                 |
   |            END        Adaptive Planner
   |                               |
   |                               v
   |                        Follow-up questions?
   |                         /             \
   |                       yes              no
   |                        |                |
   |                        v                v
   |                   Researcher          END
   |                        |
   └────────────────────────┘
"""

from langgraph.graph import END, START, StateGraph

from app.graph.nodes.adaptive_planner import adaptive_planner_node
from app.graph.nodes.evidence_extractor import evidence_extractor_node
from app.graph.nodes.evidence_verifier import evidence_verifier_node
from app.graph.nodes.planner import planner_node
from app.graph.nodes.researcher import researcher_node
from app.graph.nodes.research_sufficiency import research_sufficiency_node
from app.graph.state import ResearchState


def should_continue_research(state: ResearchState) -> str:
    """
    Decide what should happen after research sufficiency evaluation.

    The original research plan always has priority.

    Planned questions must be processed before the sufficiency
    evaluator is allowed to terminate the workflow.

    Once the planned research is exhausted:

        sufficient -> END
        insufficient -> adaptive planner

    This prevents the LLM sufficiency evaluator from ending
    research prematurely after seeing only part of the plan.
    """

    current_index = state["current_question_index"]
    total_questions = len(state["research_questions"])

    # ------------------------------------------------------
    # 1. Complete the existing research plan first.
    # ------------------------------------------------------

    if current_index < total_questions:

        print(
            "\n[Router] Planned research questions remain."
        )

        print(
            "[Router] Continuing with the current research plan."
        )

        print(
            f"[Router] Progress: "
            f"{current_index}/{total_questions}"
        )

        return "research"

    # ------------------------------------------------------
    # 2. The planned research is exhausted.
    #    Now evaluate research sufficiency.
    # ------------------------------------------------------

    if state["research_sufficient"]:

        print(
            "\n[Router] All planned research questions "
            "have been processed."
        )

        print(
            "[Router] Research is sufficient."
        )

        print(
            "[Router] Ending research."
        )

        return "end"

    # ------------------------------------------------------
    # 3. Planned research is exhausted but evidence
    #    is still insufficient.
    # ------------------------------------------------------

    if state["research_round"] < state["max_research_rounds"]:

        print(
            "\n[Router] All planned research questions "
            "have been processed."
        )

        print(
            "[Router] Research is still insufficient."
        )

        print(
            "[Router] Starting adaptive planning."
        )

        return "adaptive_planner"

    # ------------------------------------------------------
    # 4. Safety limit reached.
    # ------------------------------------------------------

    print(
        "\n[Router] Maximum research rounds reached."
    )

    print(
        "[Router] Ending research with the current evidence."
    )

    return "end"


def should_continue_after_adaptive_planner(
    state: ResearchState,
) -> str:
    """
    Decide what should happen after adaptive planning.

    If the adaptive planner generated new research questions,
    continue with the researcher.

    If it failed to generate useful follow-up questions,
    terminate the workflow instead of sending an old question
    back to the researcher.
    """

    questions = state["research_questions"]
    current_index = state["current_question_index"]

    if current_index < len(questions):

        print(
            "\n[Router] Adaptive planner created "
            "new research questions."
        )

        print(
            "[Router] Continuing with researcher."
        )

        return "research"

    print(
        "\n[Router] Adaptive planner created "
        "no new research questions."
    )

    print(
        "[Router] Ending research with the current evidence."
    )

    return "end"


def build_research_graph():
    """
    Build and compile the complete research workflow.
    """

    graph = StateGraph(ResearchState)

    graph.add_node(
        "planner",
        planner_node,
    )

    graph.add_node(
        "researcher",
        researcher_node,
    )

    graph.add_node(
        "evidence_extractor",
        evidence_extractor_node,
    )

    graph.add_node(
        "evidence_verifier",
        evidence_verifier_node,
    )

    graph.add_node(
        "research_sufficiency",
        research_sufficiency_node,
    )

    graph.add_node(
        "adaptive_planner",
        adaptive_planner_node,
    )

    # ------------------------------------------------------
    # Main workflow
    # ------------------------------------------------------

    graph.add_edge(
        START,
        "planner",
    )

    graph.add_edge(
        "planner",
        "researcher",
    )

    graph.add_edge(
        "researcher",
        "evidence_extractor",
    )

    graph.add_edge(
        "evidence_extractor",
        "evidence_verifier",
    )

    graph.add_edge(
        "evidence_verifier",
        "research_sufficiency",
    )

    # ------------------------------------------------------
    # Research sufficiency routing
    # ------------------------------------------------------

    graph.add_conditional_edges(
        "research_sufficiency",
        should_continue_research,
        {
            "research": "researcher",
            "adaptive_planner": "adaptive_planner",
            "end": END,
        },
    )

    # ------------------------------------------------------
    # Adaptive research routing
    # ------------------------------------------------------

    graph.add_conditional_edges(
        "adaptive_planner",
        should_continue_after_adaptive_planner,
        {
            "research": "researcher",
            "end": END,
        },
    )

    return graph.compile()
