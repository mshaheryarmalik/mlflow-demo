"""
run_pipeline.py — End-to-end runner for the LangGraph multi-agent pipeline.

What this script demonstrates:
  1. mlflow.langchain.autolog()  — automatic tracing of every LangChain/LangGraph call
  2. mlflow.start_run()          — wrapping the full pipeline in a named MLflow run
  3. mlflow.log_params()         — logging pipeline configuration
  4. mlflow.log_metrics()        — logging output quality metrics
  5. mlflow.log_text()           — persisting the final article as an artifact
  6. Prompt Registry             — registering and loading versioned system prompts

Usage:
    uv run python scripts/run_pipeline.py
    uv run python scripts/run_pipeline.py --topic "LangGraph vs AutoGen"
    uv run python scripts/run_pipeline.py --topic "MLflow in production" --experiment my-exp
"""

import argparse
import time

import mlflow

from agents.config import configure_mlflow, get_ollama_model
from agents.graph import pipeline
from agents.nodes import (
    CRITIC_SYSTEM_PROMPT,
    RESEARCHER_SYSTEM_PROMPT,
    WRITER_SYSTEM_PROMPT,
)


# ── Prompt Registry helpers ────────────────────────────────────────────────────


def register_prompts() -> None:
    """
    Register versioned system prompts in the MLflow Prompt Registry.
    Idempotent — safe to call on every run.
    """
    prompts = {
        "researcher-system-prompt": RESEARCHER_SYSTEM_PROMPT,
        "writer-system-prompt": WRITER_SYSTEM_PROMPT,
        "critic-system-prompt": CRITIC_SYSTEM_PROMPT,
    }
    for name, template in prompts.items():
        try:
            mlflow.register_prompt(name=name, template=template)
            print(f"  [prompt-registry] Registered '{name}'")
        except Exception:
            # Prompt may already exist — that's fine
            pass


# ── Simple quality metrics ─────────────────────────────────────────────────────


def compute_output_metrics(state: dict) -> dict[str, float]:
    """
    Lightweight heuristic metrics on the final output.
    Replace / extend with mlflow.evaluate() for LLM-judge-based scoring.
    """
    final = state.get("final", "") or state.get("draft", "")
    research = state.get("research", "")

    return {
        "output_word_count": len(final.split()),
        "research_word_count": len(research.split()),
        "iterations_completed": float(state.get("iterations", 0)),
        "has_markdown_headers": float("##" in final or "# " in final),
        "latency_seconds": state.get("_latency_seconds", 0.0),
    }


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LangGraph multi-agent pipeline.")
    parser.add_argument(
        "--topic",
        type=str,
        default="MLflow for LLMs and Agents in production",
        help="Research topic for the pipeline.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="llm-agents-demo",
        help="MLflow experiment name.",
    )
    args = parser.parse_args()

    # 1. Configure MLflow
    configure_mlflow(experiment_name=args.experiment)

    # 2. Enable automatic tracing for all LangChain/LangGraph calls
    mlflow.langchain.autolog()
    print("[mlflow] LangChain autolog enabled — all LLM calls will be traced.")

    # 3. Register prompts in the Prompt Registry
    print("[mlflow] Registering system prompts...")
    register_prompts()

    # 4. Run the pipeline inside a named MLflow run
    with mlflow.start_run(run_name=f"pipeline-{args.topic[:40].replace(' ', '-')}") as run:
        print(f"\n[mlflow] Run ID: {run.info.run_id}")
        print(f"[mlflow] Tracking URI: {mlflow.get_tracking_uri()}")

        # Log pipeline parameters
        mlflow.log_params({
            "topic": args.topic,
            "model": get_ollama_model(),
            "max_iterations": 2,
            "framework": "langgraph",
        })

        # 5. Execute the pipeline
        print(f"\n[pipeline] Starting pipeline for topic: '{args.topic}'\n")
        start_time = time.time()

        initial_state = {
            "topic": args.topic,
            "messages": [],
            "research": "",
            "draft": "",
            "feedback": "",
            "final": "",
            "iterations": 0,
        }

        final_state = pipeline.invoke(initial_state)
        elapsed = time.time() - start_time
        final_state["_latency_seconds"] = elapsed

        print(f"[pipeline] Completed in {elapsed:.1f}s")

        # 6. Log quality metrics
        metrics = compute_output_metrics(final_state)
        mlflow.log_metrics(metrics)
        print(f"[mlflow] Logged metrics: {metrics}")

        # 7. Save outputs as artifacts
        output = final_state.get("final") or final_state.get("draft", "No output generated.")
        mlflow.log_text(output, "final_article.md")
        mlflow.log_text(final_state.get("research", ""), "research_notes.txt")
        mlflow.log_text(final_state.get("feedback", ""), "critic_feedback.txt")
        print("[mlflow] Artifacts saved: final_article.md, research_notes.txt, critic_feedback.txt")

        # 8. Print the result
        print("\n" + "=" * 70)
        print("FINAL ARTICLE")
        print("=" * 70)
        print(output)
        print("=" * 70)
        print(f"\nView run in MLflow UI: http://localhost:5000/#/experiments")


if __name__ == "__main__":
    main()
