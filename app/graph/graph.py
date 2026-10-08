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
   |             |          Budget / round available?
   |             |             /          \
   |             |           yes            no
   |             |            |              |
   |             |            v              |
   |             |      Adaptive Planner     |
   |             |            |              |
   |             |            v              |
   |             |       Follow-up questions?|
   |             |          /       \         |
   |             |        yes        no       |
   |             |         |          |       |
   |             └─────────┘          |       |
   |                                  |       |
   └──────────────────────────────────┘       |
                                              |
                         ┌────────────────────┘
                         |
                         v
                    FINAL SYNTHESIS
                         |
                         v
                        END

The research phase may finish because:
- all required evidence was collected,
- the research-question budget was reached,
- the research-round limit was reached, or
- adaptive planning produced no additional questions.

Regardless of why research ends, the verified evidence is passed
to the final synthesis node.
"""

from langgraph.graph import END, START, StateGraph

from app.graph.nodes.adaptive_planner import adaptive_planner_node
from app.graph.nodes.evidence_extractor import evidence_extractor_node
from app.graph.nodes.evidence_verifier import evidence_verifier_node
from app.graph.nodes.planner import planner_node
from app.graph.nodes.researcher import researcher_node
from app.graph.nodes.research_sufficiency import (
    research_sufficiency_node,
)
from app.graph.nodes.synthesizer import synthesis_node
from app.graph.state import ResearchState


def should_continue_research(
    state: ResearchState,
) -> str:
    """
    Decide what should happen after research sufficiency
    evaluation.

    The original research plan always has priority.

    Once the planned research is exhausted:

        sufficient -> synthesis
        insufficient -> adaptive planner

    Adaptive research is allowed only while the absolute
    research-question budget has not been reached.
    """

    current_index = state["current_question_index"]
    total_questions = len(state["research_questions"])
    max_total_questions = state["max_total_research_questions"]

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
    #    If sufficient, move directly to synthesis.
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
            "[Router] Moving to final synthesis."
        )

        return "synthesis"

    # ------------------------------------------------------
    # 3. Hard total-question budget.
    #
    # No additional research can be started once the
    # absolute question budget has been consumed.
    #
    # The current evidence is still passed to synthesis.
    # ------------------------------------------------------

    if total_questions >= max_total_questions:

        print(
            "\n[Router] Maximum research-question budget "
            "has been reached."
        )

        print(
            f"[Router] Questions in plan: "
            f"{total_questions}"
        )

        print(
            f"[Router] Maximum allowed: "
            f"{max_total_questions}"
        )

        print(
            "[Router] Ending research phase."
        )

        print(
            "[Router] Moving to final synthesis with "
            "the current verified evidence."
        )

        return "synthesis"

    # ------------------------------------------------------
    # 4. Planned research is exhausted but evidence
    #    is still insufficient.
    #
    # Adaptive planning is allowed only if both:
    #
    #   - the research-round limit allows another round
    #   - the total-question budget has capacity
    # ------------------------------------------------------

    if state["research_round"] < state["max_research_rounds"]:

        remaining_capacity = (
            max_total_questions - total_questions
        )

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

        print(
            f"[Router] Remaining question capacity: "
            f"{remaining_capacity}"
        )

        return "adaptive_planner"

    # ------------------------------------------------------
    # 5. Research-round safety limit reached.
    #
    # Move to synthesis rather than ending the application.
    # ------------------------------------------------------

    print(
        "\n[Router] Maximum research rounds reached."
    )

    print(
        "[Router] Ending research phase."
    )

    print(
        "[Router] Moving to final synthesis with "
        "the current verified evidence."
    )

    return "synthesis"


def should_continue_after_adaptive_planner(
    state: ResearchState,
) -> str:
    """
    Decide what should happen after adaptive planning.

    The adaptive planner may add follow-up questions.

    The absolute question budget is checked again here so
    the researcher can never receive more questions than
    the configured application limit.
    """

    questions = state["research_questions"]
    current_index = state["current_question_index"]
    max_total_questions = state["max_total_research_questions"]

    # ------------------------------------------------------
    # 1. Check whether the adaptive planner actually added
    #    usable questions.
    # ------------------------------------------------------

    if current_index >= len(questions):

        print(
            "\n[Router] Adaptive planner created "
            "no new research questions."
        )

        print(
            "[Router] Ending research phase."
        )

        print(
            "[Router] Moving to final synthesis."
        )

        return "synthesis"

    # ------------------------------------------------------
    # 2. Enforce the absolute question budget.
    #
    # This protects the workflow even if an adaptive planner
    # implementation accidentally creates too many questions.
    # ------------------------------------------------------

    if len(questions) > max_total_questions:

        print(
            "\n[Router] Adaptive planner exceeded the "
            "research-question budget."
        )

        print(
            f"[Router] Generated questions: "
            f"{len(questions)}"
        )

        print(
            f"[Router] Maximum allowed: "
            f"{max_total_questions}"
        )

        print(
            "[Router] Ending research phase."
        )

        print(
            "[Router] Moving to final synthesis with "
            "the current verified evidence."
        )

        return "synthesis"

    # ------------------------------------------------------
    # 3. Valid adaptive questions exist.
    # ------------------------------------------------------

    print(
        "\n[Router] Adaptive planner created "
        "new research questions."
    )

    print(
        "[Router] Continuing with researcher."
    )

    print(
        f"[Router] Total questions in plan: "
        f"{len(questions)}/{max_total_questions}"
    )

    return "research"


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

    graph.add_node(
        "synthesizer",
        synthesis_node,
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
            "synthesis": "synthesizer",
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
            "synthesis": "synthesizer",
        },
    )

    # ------------------------------------------------------
    # Final synthesis
    # ------------------------------------------------------

    graph.add_edge(
        "synthesizer",
        END,
    )

    return graph.compile()
