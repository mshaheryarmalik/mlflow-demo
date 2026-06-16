"""
Shared MLflow configuration and tracking setup for the llms-agents package.
"""

import os

import mlflow
from dotenv import load_dotenv

load_dotenv()


def configure_mlflow(experiment_name: str = "llm-agents-demo") -> None:
    """
    Point MLflow at the local tracking server and set the active experiment.
    Call this once at the start of any script or notebook.
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def get_ollama_base_url() -> str:
    return os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def get_ollama_model() -> str:
    return os.environ.get("OLLAMA_MODEL", "llama3.2:1b")
