"""
Experiment 02: LangChain messages.

Purpose:
    Understand how LangChain represents conversations
    using SystemMessage, HumanMessage, and AIMessage.
"""

import os

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
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


def main():
    print("=" * 70)
    print("LANGCHAIN MESSAGE TEST")
    print("=" * 70)

    model = ChatOllama(
        model=MODEL_NAME,
        base_url=BASE_URL,
        temperature=0,
    )

    messages = [
        SystemMessage(
            content=(
                "You are a professional research assistant. "
                "Give concise and factual answers."
            )
        ),
        HumanMessage(
            content=(
                "What are the main differences between "
                "LangChain and LangGraph?"
            )
        ),
    ]

    response = model.invoke(messages)

    print("\nMODEL RESPONSE")
    print("-" * 70)
    print(response.content)

    print("\nRESPONSE TYPE")
    print("-" * 70)
    print(type(response).__name__)

    print("\nMESSAGE HISTORY")
    print("-" * 70)

    conversation = [
        *messages,
        AIMessage(content=response.content),
    ]

    for message in conversation:
        print(
            f"{type(message).__name__}: "
            f"{message.content[:200]}"
        )

    print("\n" + "=" * 70)
    print("MESSAGE TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
