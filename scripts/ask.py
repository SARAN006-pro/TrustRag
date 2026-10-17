"""Interactive CLI for TrustRAG.

Usage:
    python scripts/ask.py "What is the annual benefit under PM-KISAN?"
    python scripts/ask.py   # starts interactive session
"""

import sys
import json
import time
import urllib.request
import urllib.error

API_URL = "http://127.0.0.1:8000/ask"

def ask_api(question: str) -> None:
    print(f"\n=======================================================")
    print(f"QUESTION: {question}")
    print(f"=======================================================")
    start_time = time.time()
    payload = json.dumps({"question": question}).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            elapsed = time.time() - start_time
            print(f"\nANSWER:\n{data.get('answer')}\n")
            print(f"-------------------------------------------------------")
            print(f"VERIFICATION STATUS:")
            print(f"  Grounded:    {data.get('grounded')}")
            print(f"  Risk Score:  {data.get('risk_score'):.4f} (Threshold: 0.35)")
            print(f"  Latency:     {data.get('latency_ms')} ms (CLI elapsed: {elapsed:.2f}s)")
            print(f"\nCITATIONS / SOURCES ({len(data.get('sources', []))} chunks):")
            for i, src in enumerate(data.get("sources", []), start=1):
                chunk_id = src.get("chunk_id", "unknown")
                snippet = src.get("snippet", "").strip()[:180]
                print(f"  [{i}] {chunk_id}: {snippet}...")
            print(f"=======================================================\n")
    except urllib.error.URLError as e:
        print(f"\n[ERROR] Failed to connect to TrustRAG API at {API_URL}: {e}")
        print("Please ensure the API is running with:")
        print("  uvicorn trustrag.api:app --host 127.0.0.1 --port 8000\n")

def main():
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        ask_api(question)
    else:
        print("TrustRAG Interactive Session (Type 'exit' or 'quit' to quit)")
        while True:
            try:
                question = input("\nAsk a question about government schemes > ").strip()
                if not question:
                    continue
                if question.lower() in ("exit", "quit", "q"):
                    break
                ask_api(question)
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break

if __name__ == "__main__":
    main()
