"""
MoodMentor CLI Entry Point (Task 9).
Provides commands: run, evaluate, verify, version, help.
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path


def get_base_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(
        prog="moodmentor",
        description="MoodMentor: AI-Based Employee Wellness Management Platform CLI",
    )
    parser.add_argument("--version", action="version", version="MoodMentor 1.0.0")

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: run
    run_parser = subparsers.add_parser("run", help="Launch the interactive Streamlit dashboard")
    run_parser.add_argument("--port", type=int, default=8501, help="Port to run Streamlit app on (default 8501)")
    run_parser.add_argument("--headless", action="store_true", help="Run Streamlit in headless mode")

    # Command: evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Run offline ML recommendation benchmark (Task 9)")
    eval_parser.add_argument("--k", type=int, default=3, help="Top-K recommendations to evaluate (default 3)")

    # Command: verify
    verify_parser = subparsers.add_parser("verify", help="Run all verification scripts (Milestones 1, 2, 3)")
    verify_parser.add_argument("--milestone", type=int, choices=[1, 2, 3], help="Specific milestone verification script to run")

    args = parser.parse_args()
    base_dir = get_base_dir()

    if args.command == "run":
        app_path = base_dir / "app.py"
        cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(args.port)]
        if args.headless:
            cmd.append("--server.headless=true")
        print(f"Launching MoodMentor Dashboard at http://localhost:{args.port}...")
        subprocess.run(cmd, cwd=base_dir)

    elif args.command == "evaluate":
        print(f"Executing MoodMentor Recommendation Benchmark (K={args.k})...")
        from services.recommendation_eval import run_recommendation_evaluation
        run_recommendation_evaluation(k=args.k)

    elif args.command == "verify":
        if args.milestone:
            script_path = base_dir / f"verify_milestone{args.milestone}.py"
            print(f"Running verify_milestone{args.milestone}.py...")
            subprocess.run([sys.executable, str(script_path)], cwd=base_dir)
        else:
            print("Running all verification scripts...")
            for m in [1, 2, 3]:
                script_path = base_dir / f"verify_milestone{m}.py"
                print(f"\n--- Running Milestone {m} Verification ---")
                subprocess.run([sys.executable, str(script_path)], cwd=base_dir)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
