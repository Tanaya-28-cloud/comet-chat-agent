"""
Simple interactive CLI for manually exercising the Aster & Row agent —
used for demoing the required scenarios (KB question with citations,
order lookup, multi-turn conversation, abstention/handoff) in the demo
video/GIF. Not used by run_eval.py, which drives AsterRowAgent directly.
"""

import logging

# Quiet third-party/library logging noise (HTTP requests, model loading,
# AFC warnings) so the interactive demo output stays readable. The
# agent's own structured logging (KB retrieval, decisions, tool calls)
# still exists and still fires — this only stops it from printing to
# the console during a live CLI session. Full logs remain visible when
# running evaluation/run_eval.py or by changing this level back to INFO.
logging.basicConfig(level=logging.WARNING)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("google_genai").setLevel(logging.ERROR)
logging.getLogger("aster_row_agent").setLevel(logging.WARNING)

from src.agent import AsterRowAgent


def main() -> None:
    print("Aster & Row Support Agent (type 'exit' or 'quit' to stop)\n")

    agent = AsterRowAgent()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        if not user_input:
            continue

        response = agent.send(user_input)

        print(f"\nAgent: {response.answer}")
        if response.sources:
            print(f"Sources: {', '.join(response.sources)}")
        print(f"Handoff: {response.handoff}")
        if response.tool_called:
            print(f"Tool called: {response.tool_called} {response.tool_arguments}")
        print()


if __name__ == "__main__":
    main()