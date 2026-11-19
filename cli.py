"""Interactive command-line Q&A over the committed index."""

from chain import load_chain


def main():
    """Read questions from stdin in a loop and print grounded answers."""
    ask = load_chain()
    print("RAG CLI ready. Type your question (or 'q' to quit).")
    while True:
        q = input("\nQ: ").strip()
        if q.lower() in {"q", "quit", "exit"}:
            break
        out = ask(q)
        print(f"\nAnswer:\n{out['answer']}\n")
        print(f"Sources: {', '.join(out['sources'])} | Latency: {out['latency']}s")


if __name__ == "__main__":
    main()
