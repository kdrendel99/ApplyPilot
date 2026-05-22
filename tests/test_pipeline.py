from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from applypilot import config
from applypilot import pipeline


def _stub_runtime(monkeypatch, tmp_path: Path) -> Path:
    log_dir = tmp_path / "logs"
    monkeypatch.setattr(config, "LOG_DIR", log_dir)
    monkeypatch.setattr(pipeline, "load_env", lambda: None)
    monkeypatch.setattr(pipeline, "ensure_dirs", lambda: log_dir.mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr(pipeline, "init_db", lambda: None)
    monkeypatch.setattr(
        pipeline,
        "get_stats",
        lambda: {
            "total": 0,
            "pending_detail": 0,
            "with_description": 0,
            "scored": 0,
            "tailored": 0,
            "with_cover_letter": 0,
            "ready_to_apply": 0,
            "applied": 0,
        },
    )
    return log_dir


def test_run_pipeline_creates_per_stage_logs(monkeypatch, tmp_path):
    log_dir = _stub_runtime(monkeypatch, tmp_path)

    def run_score():
        pipeline.console.print("score console marker")
        print("score stdout marker")
        print("score stderr marker", file=sys.stderr)
        logging.getLogger("applypilot.tests").info("score marker")
        return {"status": "ok"}

    def run_tailor(min_score: int = 7, validation_mode: str = "normal"):
        logging.getLogger("applypilot.tests").warning("tailor marker")
        return {"status": "ok"}

    monkeypatch.setitem(pipeline._STAGE_RUNNERS, "score", run_score)
    monkeypatch.setitem(pipeline._STAGE_RUNNERS, "tailor", run_tailor)

    result = pipeline.run_pipeline(stages=["score", "tailor"])

    run_dir = Path(result["run_log_dir"])
    assert run_dir.is_dir()
    assert run_dir.parent == log_dir / "runs"
    assert (log_dir / "runs" / "latest").exists()

    score_log = run_dir / "score.log"
    tailor_log = run_dir / "tailor.log"
    assert result["stage_log_paths"] == {
        "score": str(score_log),
        "tailor": str(tailor_log),
    }
    run_log = run_dir / "run.log"
    assert result["run_log_path"] == str(run_log)
    assert run_log.exists()

    score_text = score_log.read_text(encoding="utf-8")
    assert "Stage: score" in score_text
    assert "Description: LLM scoring" in score_text
    assert "Start time:" in score_text
    assert "Completion time:" in score_text
    assert "Elapsed seconds:" in score_text
    assert "Final status: ok" in score_text
    assert "score marker" in score_text
    assert "score console marker" in score_text
    assert "score stdout marker" in score_text
    assert "score stderr marker" in score_text
    assert "tailor marker" not in score_text

    run_text = run_log.read_text(encoding="utf-8")
    assert "Run logs:" in run_text
    assert "score console marker" in run_text
    assert "score stdout marker" in run_text
    assert "score stderr marker" in run_text

    tailor_text = tailor_log.read_text(encoding="utf-8")
    assert "Stage: tailor" in tailor_text
    assert "tailor marker" in tailor_text
    assert "score marker" not in tailor_text


def test_dry_run_does_not_create_run_logs(monkeypatch, tmp_path):
    log_dir = _stub_runtime(monkeypatch, tmp_path)

    result = pipeline.run_pipeline(stages=["score"], dry_run=True)

    assert result == {"stages": [], "errors": {}, "elapsed": 0.0}
    assert not (log_dir / "runs").exists()


def test_streaming_logs_stay_stage_scoped(monkeypatch, tmp_path):
    _stub_runtime(monkeypatch, tmp_path)
    monkeypatch.setattr(pipeline, "_STREAM_POLL_INTERVAL", 0.01)

    pending_counts = {"score": 0}

    def count_pending(stage: str, min_score: int = 7) -> int:
        if stage != "score":
            return 0
        pending_counts["score"] += 1
        return 1 if pending_counts["score"] == 1 else 0

    def run_discover(workers: int = 1):
        logging.getLogger("applypilot.tests").info("discover stream marker")
        time.sleep(0.02)
        return {"status": "ok"}

    def run_score():
        logging.getLogger("applypilot.tests").info("score stream marker")
        return {"status": "ok"}

    monkeypatch.setattr(pipeline, "_count_pending", count_pending)
    monkeypatch.setitem(pipeline._STAGE_RUNNERS, "discover", run_discover)
    monkeypatch.setitem(pipeline._STAGE_RUNNERS, "score", run_score)

    result = pipeline.run_pipeline(stages=["discover", "score"], stream=True)
    run_dir = Path(result["run_log_dir"])

    discover_text = (run_dir / "discover.log").read_text(encoding="utf-8")
    score_text = (run_dir / "score.log").read_text(encoding="utf-8")

    assert "discover stream marker" in discover_text
    assert "score stream marker" not in discover_text
    assert "score stream marker" in score_text
    assert "discover stream marker" not in score_text
