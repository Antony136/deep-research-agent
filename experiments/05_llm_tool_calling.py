"""
Experiment 05: Native tool-calling compatibility test.

Purpose:
    Determine whether the current Qwen + Ollama +
    LangChain setup produces native LangChain tool calls.

We intentionally inspect the AIMessage instead of
executing anything automatically.
"""

import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama


load_dotenv()


MODEL_NAME = os.getenv(
    "OLLAMA_MODEL",
    "qwen2.5-coder:7b",
)

BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)


# ------------------------------------------------------------
# Tools
# ------------------------------------------------------------

@tool
def calculate_sum(a: int, b: int) -> int:
    """
    Add two integers and return the result.
    """
    return a + b


@tool
def calculate_product(a: int, b: int) -> int:
    """
    Multiply two integers and return the result.
    """
    return a * b


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print("NATIVE TOOL-CALLING COMPATIBILITY TEST")
    print("=" * 70)

    model = ChatOllama(
        model=MODEL_NAME,
        base_url=BASE_URL,
        temperature=0,
    )

    model_with_tools = model.bind_tools(
        [
            calculate_sum,
            calculate_product,
        ]
    )

    messages = [
        SystemMessage(
            content=(
                "You are an AI assistant with access to tools. "
                "When a user's request requires a tool, call the "
                "appropriate tool instead of answering manually."
            )
        ),
        HumanMessage(
            content=(
                "Calculate 125 multiplied by 48 using the "
                "available calculator tool."
            )
        ),
    ]

    response = model_with_tools.invoke(
        messages
    )

    # --------------------------------------------------------
    # Basic response
    # --------------------------------------------------------

    print("\nMODEL CONTENT")
    print("-" * 70)

    print(response.content)

    # --------------------------------------------------------
    # Native tool calls
    # --------------------------------------------------------

    print("\nNATIVE TOOL CALLS")
    print("-" * 70)

    if response.tool_calls:

        for tool_call in response.tool_calls:

            print(
                f"Name: {tool_call['name']}"
            )

            print(
                f"Arguments: {tool_call['args']}"
            )

            print(
                f"ID: {tool_call['id']}"
            )

    else:

        print("No native tool calls detected.")

    # --------------------------------------------------------
    # Invalid tool calls
    # --------------------------------------------------------

    print("\nINVALID TOOL CALLS")
    print("-" * 70)

    print(response.invalid_tool_calls)

    # --------------------------------------------------------
    # Additional kwargs
    # --------------------------------------------------------

    print("\nADDITIONAL KWARGS")
    print("-" * 70)

    print(response.additional_kwargs)

    print("\n" + "=" * 70)
    print("NATIVE TOOL-CALLING TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
