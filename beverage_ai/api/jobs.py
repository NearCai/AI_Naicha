"""Tiny in-memory background-job registry for long-running subprocesses.

Used by `/api/v2/update` to launch `scripts/update_from_feedback.py` and
return a job id the caller can poll. Suitable for single-process dev /
small deployments — production should swap this for Celery/RQ/Arq.

Lifecycle:
    queued → running → completed (or failed)
"""
from __future__ import annotations

import subprocess
import sys
import threading
import uuid
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

JobStatus = Literal["queued", "running", "completed", "failed"]


@dataclass
class Job:
    job_id: str
    cmd: list[str]
    status: JobStatus = "queued"
    started_at: datetime | None = None
    finished_at: datetime | None = None
    returncode: int | None = None
    stdout_tail: list[str] = field(default_factory=list)
    stderr_tail: list[str] = field(default_factory=list)
    audit_path: str | None = None
    error: str | None = None
    note: str | None = None

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "returncode": self.returncode,
            "stdout_tail": list(self.stdout_tail),
            "stderr_tail": list(self.stderr_tail),
            "audit_path": self.audit_path,
            "error": self.error,
            "note": self.note,
            "cmd": list(self.cmd),
        }


class JobRegistry:
    """Thread-safe in-memory job table."""

    _MAX_TAIL_LINES = 20
    _MAX_JOBS = 50

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._order: deque[str] = deque()

    def submit(
        self,
        cmd: Sequence[str],
        *,
        cwd: Path | str | None = None,
        env: dict[str, str] | None = None,
        audit_path: str | None = None,
        note: str | None = None,
    ) -> Job:
        job = Job(
            job_id=f"job_{uuid.uuid4().hex[:12]}",
            cmd=list(cmd),
            audit_path=audit_path,
            note=note,
        )
        with self._lock:
            self._jobs[job.job_id] = job
            self._order.append(job.job_id)
            # Drop oldest jobs if we exceed the cap.
            while len(self._order) > self._MAX_JOBS:
                old = self._order.popleft()
                self._jobs.pop(old, None)

        thread = threading.Thread(
            target=self._run,
            args=(job, cwd, env),
            daemon=True,
            name=f"job-{job.job_id}",
        )
        thread.start()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())

    # -------- internal --------

    def _run(self, job: Job, cwd: Path | str | None, env: dict[str, str] | None) -> None:
        with self._lock:
            job.status = "running"
            job.started_at = datetime.now(timezone.utc)

        try:
            proc = subprocess.run(
                job.cmd,
                cwd=str(cwd) if cwd else None,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except Exception as e:  # pragma: no cover - defensive
            with self._lock:
                job.status = "failed"
                job.error = f"{type(e).__name__}: {e}"
                job.finished_at = datetime.now(timezone.utc)
            return

        stdout_lines = (proc.stdout or "").splitlines()
        stderr_lines = (proc.stderr or "").splitlines()
        with self._lock:
            job.returncode = proc.returncode
            job.stdout_tail = stdout_lines[-self._MAX_TAIL_LINES :]
            job.stderr_tail = stderr_lines[-self._MAX_TAIL_LINES :]
            job.finished_at = datetime.now(timezone.utc)
            if proc.returncode == 0:
                job.status = "completed"
            else:
                job.status = "failed"
                if not job.error:
                    job.error = f"subprocess exited {proc.returncode}"


# Module-level singleton — fine because Job state is bounded and process-local.
registry = JobRegistry()


def build_update_cmd(
    session_id: str,
    *,
    feedback_db: str = "data/feedback.duckdb",
    base_model: str = "models/sensory_gnn_stage1_best.pt",
    stage2_epochs: int = 30,
    dirichlet_lr: float = 0.3,
    serving_top_quantile: float = 0.6,
    skip_stage2: bool = True,
    audit_dir: str = "data/feedback",
    python: str | None = None,
) -> list[str]:
    """Build the argv for `scripts/update_from_feedback.py`."""
    cmd: list[str] = [
        python or sys.executable,
        "scripts/update_from_feedback.py",
        "--session", session_id,
        "--feedback-db", feedback_db,
        "--base-model", base_model,
        "--stage2-epochs", str(stage2_epochs),
        "--dirichlet-lr", str(dirichlet_lr),
        "--serving-top-quantile", str(serving_top_quantile),
        "--audit-dir", audit_dir,
    ]
    if skip_stage2:
        cmd.append("--skip-stage2")
    return cmd
