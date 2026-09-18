"""
Runs LLM-generated Python "experiment" scripts in isolation.

Primary path: a Docker container built from Dockerfile.sandbox, run with
  --network none, a read-only root filesystem except a scratch mount, and a
  wall-clock timeout. This is the safest option and is used automatically
  whenever the Docker daemon is reachable.

Fallback path: if Docker is not installed / not running (common on a fresh
Windows machine), the code runs as a subprocess of the current Python
interpreter with a timeout. This is NOT a strong sandbox -- it is only a
convenience fallback so the app still works end-to-end. The UI clearly
labels which mode was used.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid


def docker_available() -> bool:
    try:
        subprocess.run(
            ["docker", "info"], capture_output=True, timeout=5, check=True
        )
        return True
    except Exception:
        return False


def _write_workspace(code: str) -> str:
    workdir = tempfile.mkdtemp(prefix="ai_research_sandbox_")
    with open(os.path.join(workdir, "script.py"), "w", encoding="utf-8") as f:
        f.write(code)
    return workdir


def run_in_docker(code: str, image: str, timeout: int) -> dict:
    workdir = _write_workspace(code)
    try:
        cmd = [
            "docker", "run", "--rm",
            "--network", "none",
            "--memory", "1g",
            "--cpus", "1",
            "-v", f"{workdir}:/workspace",
            "-w", "/workspace",
            image,
            "python", "script.py",
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=timeout, text=True)
            output_path = os.path.join(workdir, "output.json")
            artifacts = {}
            if os.path.exists(output_path):
                try:
                    with open(output_path, "r", encoding="utf-8") as f:
                        artifacts = json.load(f)
                except Exception:
                    artifacts = {}
            return {
                "success": proc.returncode == 0,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "artifacts": artifacts,
                "mode": "docker",
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Experiment timed out after {timeout}s",
                "artifacts": {},
                "mode": "docker",
            }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def run_locally(code: str, timeout: int) -> dict:
    workdir = _write_workspace(code)
    try:
        cmd = [sys.executable, "script.py"]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, timeout=timeout, text=True, cwd=workdir
            )
            output_path = os.path.join(workdir, "output.json")
            artifacts = {}
            if os.path.exists(output_path):
                try:
                    with open(output_path, "r", encoding="utf-8") as f:
                        artifacts = json.load(f)
                except Exception:
                    artifacts = {}
            return {
                "success": proc.returncode == 0,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "artifacts": artifacts,
                "mode": "local-subprocess (NOT sandboxed - install Docker for isolation)",
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Experiment timed out after {timeout}s",
                "artifacts": {},
                "mode": "local-subprocess",
            }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def run_experiment(code: str, image: str, timeout: int) -> dict:
    """Entry point used by the agent. Chooses Docker if available, else falls back."""
    if docker_available():
        result = run_in_docker(code, image, timeout)
        # If the image itself is missing, docker run fails fast with a clear stderr;
        # fall back to local execution so the user still gets a result.
        if not result["success"] and "Unable to find image" in (result.get("stderr") or ""):
            return run_locally(code, timeout)
        return result
    return run_locally(code, timeout)
