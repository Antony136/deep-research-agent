"""
Experiment 04: LangChain tools.

Purpose:
    Understand how LangChain represents functions as tools
    that an LLM/agent can potentially use.
"""

from langchain_core.tools import tool


# ------------------------------------------------------------
# Tool definition
# ------------------------------------------------------------

@tool
def calculate_sum(a: int, b: int) -> int:
    """
    Add two integers and return the result.
    """
    return a + b


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print("LANGCHAIN TOOL TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Tool metadata
    # --------------------------------------------------------

    print("\nTOOL NAME")
    print("-" * 70)
    print(calculate_sum.name)

    print("\nTOOL DESCRIPTION")
    print("-" * 70)
    print(calculate_sum.description)

    print("\nTOOL SCHEMA")
    print("-" * 70)
    print(calculate_sum.args_schema.model_json_schema())

    # --------------------------------------------------------
    # Direct tool execution
    # --------------------------------------------------------

    print("\nDIRECT TOOL EXECUTION")
    print("-" * 70)

    result = calculate_sum.invoke(
        {
            "a": 125,
            "b": 48,
        }
    )

    print(f"125 + 48 = {result}")

    print("\n" + "=" * 70)
    print("LANGCHAIN TOOL TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
