"""
Experiment 01: LangChain + Ollama model.

Purpose:
    Verify that LangChain can communicate with our local
    Qwen model through Ollama.

This is the first framework layer of the Deep Research Agent.
"""

import os

from dotenv import load_dotenv
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
    print("LANGCHAIN + OLLAMA TEST")
    print("=" * 70)

    model = ChatOllama(
        model=MODEL_NAME,
        base_url=BASE_URL,
        temperature=0,
    )

    response = model.invoke(
        "Explain what a deep research agent is in one paragraph."
    )

    print("\nMODEL RESPONSE")
    print("-" * 70)
    print(response.content)

    print("\n" + "=" * 70)
    print("LANGCHAIN MODEL TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
