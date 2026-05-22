"""ApplyPilot Pipeline Orchestrator.

Runs pipeline stages in sequence or concurrently (streaming mode).

Usage (via CLI):
    applypilot run                        # all stages, sequential
    applypilot run --stream               # all stages, concurrent
    applypilot run discover enrich        # specific stages
    applypilot run score tailor cover     # LLM-only stages
    applypilot run --dry-run              # preview without executing
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, TextIO

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from applypilot.config import load_env, ensure_dirs
from applypilot.database import init_db, get_connection, get_stats

log = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------

STAGE_ORDER = ("discover", "enrich", "score", "tailor", "cover", "pdf")

STAGE_META: dict[str, dict] = {
    "discover": {"desc": "Job discovery (JobSpy + Workday + smart extract)"},
    "enrich":   {"desc": "Detail enrichment (full descriptions + apply URLs)"},
    "score":    {"desc": "LLM scoring (fit 1-10)"},
    "tailor":   {"desc": "Resume tailoring (LLM + validation)"},
    "cover":    {"desc": "Cover letter generation"},
    "pdf":      {"desc": "PDF conversion (tailored resumes + cover letters)"},
}

# Upstream dependency: a stage only finishes when its upstream is done AND
# it has no remaining pending work.
_UPSTREAM: dict[str, str | None] = {
    "discover": None,
    "enrich":   "discover",
    "score":    "enrich",
    "tailor":   "score",
    "cover":    "tailor",
    "pdf":      "cover",
}


# ---------------------------------------------------------------------------
# Individual stage runners
# ---------------------------------------------------------------------------

def _run_discover(workers: int = 1) -> dict:
    """Stage: Job discovery — JobSpy, Workday, and smart-extract scrapers."""
    stats: dict = {"jobspy": None, "workday": None, "smartextract": None}

    # JobSpy
    console.print("  [cyan]JobSpy full crawl...[/cyan]")
    try:
        from applypilot.discovery.jobspy import run_discovery
        run_discovery()
        stats["jobspy"] = "ok"
    except Exception as e:
        log.error("JobSpy crawl failed: %s", e)
        console.print(f"  [red]JobSpy error:[/red] {e}")
        stats["jobspy"] = f"error: {e}"

    # Workday corporate scraper
    console.print("  [cyan]Workday corporate scraper...[/cyan]")
    try:
        from applypilot.discovery.workday import run_workday_discovery
        run_workday_discovery(workers=workers)
        stats["workday"] = "ok"
    except Exception as e:
        log.error("Workday scraper failed: %s", e)
        console.print(f"  [red]Workday error:[/red] {e}")
        stats["workday"] = f"error: {e}"

    # Smart extract
    console.print("  [cyan]Smart extract (AI-powered scraping)...[/cyan]")
    try:
        from applypilot.discovery.smartextract import run_smart_extract
        run_smart_extract(workers=workers)
        stats["smartextract"] = "ok"
    except Exception as e:
        log.error("Smart extract failed: %s", e)
        console.print(f"  [red]Smart extract error:[/red] {e}")
        stats["smartextract"] = f"error: {e}"

    return stats


def _run_enrich(workers: int = 1) -> dict:
    """Stage: Detail enrichment — scrape full descriptions and apply URLs."""
    try:
        from applypilot.enrichment.detail import run_enrichment
        run_enrichment(workers=workers)
        return {"status": "ok"}
    except Exception as e:
        log.error("Enrichment failed: %s", e)
        return {"status": f"error: {e}"}


def _run_score() -> dict:
    """Stage: LLM scoring — assign fit scores 1-10."""
    try:
        from applypilot.scoring.scorer import run_scoring
        run_scoring()
        return {"status": "ok"}
    except Exception as e:
        log.error("Scoring failed: %s", e)
        return {"status": f"error: {e}"}


def _run_tailor(min_score: int = 7, validation_mode: str = "normal") -> dict:
    """Stage: Resume tailoring — generate tailored resumes for high-fit jobs."""
    try:
        from applypilot.scoring.tailor import run_tailoring
        run_tailoring(min_score=min_score, validation_mode=validation_mode)
        return {"status": "ok"}
    except Exception as e:
        log.error("Tailoring failed: %s", e)
        return {"status": f"error: {e}"}


def _run_cover(min_score: int = 7, validation_mode: str = "normal") -> dict:
    """Stage: Cover letter generation."""
    try:
        from applypilot.scoring.cover_letter import run_cover_letters
        run_cover_letters(min_score=min_score, validation_mode=validation_mode)
        return {"status": "ok"}
    except Exception as e:
        log.error("Cover letter generation failed: %s", e)
        return {"status": f"error: {e}"}


def _run_pdf() -> dict:
    """Stage: PDF conversion — convert tailored resumes and cover letters to PDF."""
    try:
        from applypilot.scoring.pdf import batch_convert
        batch_convert()
        return {"status": "ok"}
    except Exception as e:
        log.error("PDF conversion failed: %s", e)
        return {"status": f"error: {e}"}


# Map stage names to their runner functions
_STAGE_RUNNERS: dict[str, Callable[..., dict]] = {
    "discover": _run_discover,
    "enrich":   _run_enrich,
    "score":    _run_score,
    "tailor":   _run_tailor,
    "cover":    _run_cover,
    "pdf":      _run_pdf,
}


# ---------------------------------------------------------------------------
# Run logging
# ---------------------------------------------------------------------------

_LOG_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S%z"
_LOGGER_STAGE_PREFIXES = (
    ("applypilot.discovery.", "discover"),
    ("applypilot.enrichment.", "enrich"),
    ("applypilot.scoring.scorer", "score"),
    ("applypilot.scoring.tailor", "tailor"),
    ("applypilot.scoring.cover_letter", "cover"),
    ("applypilot.scoring.pdf", "pdf"),
)


class _ThreadAwareLogStream:
    """Mirror stdout/stderr to a run log and the current thread's stage log."""

    def __init__(
        self,
        original: TextIO,
        run_stream: TextIO,
        stage_streams: dict[int, TextIO],
        lock: threading.RLock,
    ):
        self._original = original
        self._run_stream = run_stream
        self._stage_streams = stage_streams
        self._lock = lock

    def write(self, data: str) -> int:
        written = self._original.write(data)
        self._original.flush()
        if data:
            with self._lock:
                self._run_stream.write(data)
                self._run_stream.flush()
                stage_stream = self._stage_streams.get(threading.get_ident())
                if stage_stream is not None:
                    stage_stream.write(data)
                    stage_stream.flush()
        return len(data) if written is None else written

    def flush(self) -> None:
        self._original.flush()
        with self._lock:
            self._run_stream.flush()
            stage_stream = self._stage_streams.get(threading.get_ident())
            if stage_stream is not None:
                stage_stream.flush()

    def __getattr__(self, name: str):
        return getattr(self._original, name)


class _StageLogRouter(logging.Handler):
    """Route applypilot.* log records to the stage active on the current thread."""

    def __init__(self, stage_files: dict[str, TextIO]):
        super().__init__(level=logging.NOTSET)
        self._stage_files = stage_files
        self._thread_stages: dict[int, str] = {}
        self._lock = threading.RLock()
        self.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))

    def activate(self, stage: str) -> None:
        with self._lock:
            self._thread_stages[threading.get_ident()] = stage

    def deactivate(self) -> None:
        with self._lock:
            self._thread_stages.pop(threading.get_ident(), None)

    def emit(self, record: logging.LogRecord) -> None:
        if not record.name.startswith("applypilot"):
            return

        try:
            msg = self.format(record)
        except Exception:
            self.handleError(record)
            return

        with self._lock:
            stage = self._thread_stages.get(record.thread) or self._stage_for_logger(record.name)
            if not stage:
                return
            stream = self._stage_files.get(stage)
            if stream is None:
                return
            try:
                stream.write(msg + "\n")
                stream.flush()
            except Exception:
                self.handleError(record)

    @staticmethod
    def _stage_for_logger(logger_name: str) -> str | None:
        for prefix, stage in _LOGGER_STAGE_PREFIXES:
            if logger_name.startswith(prefix):
                return stage
        return None


class _PipelineRunLogger:
    """Owns per-run and per-stage log files for one pipeline invocation."""

    def __init__(self, ordered: list[str]):
        from applypilot.config import LOG_DIR

        self.run_dir = self._make_run_dir(LOG_DIR)
        self.run_path = self.run_dir / "run.log"
        self.stage_paths = {stage: self.run_dir / f"{stage}.log" for stage in ordered}
        self._files: dict[str, TextIO] = {}
        self._run_file: TextIO | None = None
        self._router: _StageLogRouter | None = None
        self._logger = logging.getLogger("applypilot")
        self._previous_level = self._logger.level
        self._stdout: TextIO | None = None
        self._stderr: TextIO | None = None
        self._stage_streams: dict[int, TextIO] = {}
        self._output_lock = threading.RLock()

        for stage, path in self.stage_paths.items():
            path.touch()
            self._files[stage] = path.open("a", encoding="utf-8", buffering=1)

    @staticmethod
    def _make_run_dir(log_root: Path) -> Path:
        runs_root = log_root / "runs"
        runs_root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = f"{timestamp}-{os.getpid()}"
        run_dir = runs_root / base
        suffix = 1
        while run_dir.exists():
            run_dir = runs_root / f"{base}-{suffix}"
            suffix += 1
        run_dir.mkdir()
        return run_dir

    def install(self) -> None:
        self._run_file = self.run_path.open("a", encoding="utf-8", buffering=1)
        self._run_file.write(f"Run output log: {self.run_path}\n")
        self._run_file.flush()
        stdout = sys.stdout
        stderr = sys.stderr
        self._stdout = stdout
        self._stderr = stderr
        sys.stdout = _ThreadAwareLogStream(
            stdout,
            self._run_file,
            self._stage_streams,
            self._output_lock,
        )  # type: ignore[assignment]
        sys.stderr = _ThreadAwareLogStream(
            stderr,
            self._run_file,
            self._stage_streams,
            self._output_lock,
        )  # type: ignore[assignment]
        self._router = _StageLogRouter(self._files)
        self._logger.addHandler(self._router)
        if self._logger.level == logging.NOTSET or self._logger.level > logging.INFO:
            self._logger.setLevel(logging.INFO)

    def close(self) -> None:
        if self._stdout is not None:
            sys.stdout = self._stdout  # type: ignore[assignment]
            self._stdout = None
        if self._stderr is not None:
            sys.stderr = self._stderr  # type: ignore[assignment]
            self._stderr = None
        with self._output_lock:
            self._stage_streams.clear()
        if self._router is not None:
            self._logger.removeHandler(self._router)
            self._router.close()
            self._router = None
        self._logger.setLevel(self._previous_level)
        if self._run_file is not None:
            self._run_file.close()
            self._run_file = None
        for stream in self._files.values():
            stream.close()
        self._files.clear()

    def activate_stage(self, stage: str) -> None:
        if self._router is not None:
            self._router.activate(stage)
        stream = self._files.get(stage)
        if stream is not None:
            with self._output_lock:
                self._stage_streams[threading.get_ident()] = stream

    def deactivate_stage(self) -> None:
        if self._router is not None:
            self._router.deactivate()
        with self._output_lock:
            self._stage_streams.pop(threading.get_ident(), None)

    def write_stage_header(self, stage: str, started_at: datetime) -> None:
        meta = STAGE_META[stage]
        self._write(stage, f"Stage: {stage}")
        self._write(stage, f"Description: {meta['desc']}")
        self._write(stage, f"Start time: {started_at.strftime(_LOG_TIME_FORMAT)}")
        self._write(stage, "")

    def write_stage_footer(
        self,
        stage: str,
        completed_at: datetime,
        elapsed: float,
        status: str,
        error: str | None = None,
    ) -> None:
        self._write(stage, "")
        self._write(stage, f"Completion time: {completed_at.strftime(_LOG_TIME_FORMAT)}")
        self._write(stage, f"Elapsed seconds: {elapsed:.3f}")
        self._write(stage, f"Final status: {status}")
        if error:
            self._write(stage, f"Error: {error}")

    def update_latest_pointer(self) -> None:
        latest = self.run_dir.parent / "latest"
        try:
            if latest.is_symlink() or latest.exists():
                latest.unlink()
            latest.symlink_to(self.run_dir, target_is_directory=True)
        except OSError:
            latest.write_text(str(self.run_dir) + "\n", encoding="utf-8")

    def _write(self, stage: str, message: str) -> None:
        stream = self._files[stage]
        stream.write(message + "\n")
        stream.flush()


# ---------------------------------------------------------------------------
# Stage resolution
# ---------------------------------------------------------------------------

def _resolve_stages(stage_names: list[str]) -> list[str]:
    """Resolve 'all' and validate/order stage names."""
    if "all" in stage_names:
        return list(STAGE_ORDER)

    resolved = []
    for name in stage_names:
        if name not in STAGE_META:
            console.print(
                f"[red]Unknown stage:[/red] '{name}'. "
                f"Available: {', '.join(STAGE_ORDER)}, all"
            )
            raise SystemExit(1)
        if name not in resolved:
            resolved.append(name)

    # Maintain canonical order
    return [s for s in STAGE_ORDER if s in resolved]


# ---------------------------------------------------------------------------
# Streaming pipeline helpers
# ---------------------------------------------------------------------------

class _StageTracker:
    """Thread-safe tracker for which stages have finished producing work."""

    def __init__(self):
        self._events: dict[str, threading.Event] = {
            stage: threading.Event() for stage in STAGE_ORDER
        }
        self._results: dict[str, dict] = {}
        self._lock = threading.Lock()

    def mark_done(self, stage: str, result: dict | None = None) -> None:
        with self._lock:
            self._results[stage] = result or {"status": "ok"}
        self._events[stage].set()

    def is_done(self, stage: str) -> bool:
        return self._events[stage].is_set()

    def wait(self, stage: str, timeout: float | None = None) -> bool:
        return self._events[stage].wait(timeout=timeout)

    def get_results(self) -> dict[str, dict]:
        with self._lock:
            return dict(self._results)


# SQL to count pending work for each stage
_PENDING_SQL: dict[str, str] = {
    "enrich": "SELECT COUNT(*) FROM jobs WHERE detail_scraped_at IS NULL",
    "score":  "SELECT COUNT(*) FROM jobs WHERE full_description IS NOT NULL AND fit_score IS NULL",
    "tailor": (
        "SELECT COUNT(*) FROM jobs WHERE fit_score >= ? "
        "AND full_description IS NOT NULL "
        "AND tailored_resume_path IS NULL "
        "AND COALESCE(tailor_attempts, 0) < 5"
    ),
    "cover": (
        "SELECT COUNT(*) FROM jobs WHERE tailored_resume_path IS NOT NULL "
        "AND (cover_letter_path IS NULL OR cover_letter_path = '') "
        "AND COALESCE(cover_attempts, 0) < 5"
    ),
    "pdf": (
        "SELECT COUNT(*) FROM jobs WHERE tailored_resume_path IS NOT NULL "
        "AND tailored_resume_path LIKE '%.txt'"
    ),
}

# How long to sleep between polling loops in streaming mode (seconds)
_STREAM_POLL_INTERVAL = 10


def _count_pending(stage: str, min_score: int = 7) -> int:
    """Count pending work items for a stage."""
    sql = _PENDING_SQL.get(stage)
    if sql is None:
        return 0
    conn = get_connection()
    if "?" in sql:
        return conn.execute(sql, (min_score,)).fetchone()[0]
    return conn.execute(sql).fetchone()[0]


def _run_stage_streaming(
    stage: str,
    tracker: _StageTracker,
    stop_event: threading.Event,
    min_score: int = 7,
    workers: int = 1,
    validation_mode: str = "normal",
) -> None:
    """Run a single stage in streaming mode: loop until upstream done + no work.

    For discover: runs once, then marks done.
    For all others: polls DB for pending work, runs the batch processor,
    and repeats until upstream is done and no pending work remains.
    """
    runner = _STAGE_RUNNERS[stage]
    kwargs: dict = {}
    if stage in ("tailor", "cover"):
        kwargs["min_score"] = min_score
        kwargs["validation_mode"] = validation_mode
    if stage in ("discover", "enrich"):
        kwargs["workers"] = workers

    upstream = _UPSTREAM[stage]

    if stage == "discover":
        # Discover runs once (its sub-scrapers already do their full crawl)
        try:
            result = runner(**kwargs)
            tracker.mark_done(stage, result)
        except Exception as e:
            log.exception("Stage '%s' crashed", stage)
            tracker.mark_done(stage, {"status": f"error: {e}"})
        return

    # For downstream stages: loop until upstream done + no pending work
    passes = 0
    while not stop_event.is_set():
        # Wait for upstream to start producing work (first pass only)
        if passes == 0 and upstream and not tracker.is_done(upstream):
            # Wait a bit for upstream to produce some work before first run
            tracker.wait(upstream, timeout=_STREAM_POLL_INTERVAL)

        pending = _count_pending(stage, min_score)

        if pending > 0:
            try:
                runner(**kwargs)
                passes += 1
            except Exception as e:
                log.error("Stage '%s' error (pass %d): %s", stage, passes, e)
                passes += 1
        else:
            # No work right now
            upstream_done = upstream is None or tracker.is_done(upstream)
            if upstream_done:
                # No work and upstream is done — this stage is finished
                break
            # Upstream still running, wait and retry
            if stop_event.wait(timeout=_STREAM_POLL_INTERVAL):
                break  # Stop requested

    tracker.mark_done(stage, {"status": "ok", "passes": passes})


def _run_stage_streaming_logged(
    stage: str,
    tracker: _StageTracker,
    stop_event: threading.Event,
    run_logger: _PipelineRunLogger,
    min_score: int = 7,
    workers: int = 1,
    validation_mode: str = "normal",
) -> None:
    started_at = datetime.now().astimezone()
    t0 = time.time()
    status = "unknown"
    error: str | None = None

    run_logger.write_stage_header(stage, started_at)
    run_logger.activate_stage(stage)
    try:
        _run_stage_streaming(stage, tracker, stop_event, min_score, workers, validation_mode)
        status = tracker.get_results().get(stage, {}).get("status", "unknown")
    except Exception as e:
        status = f"error: {e}"
        error = str(e)
        log.exception("Stage '%s' crashed", stage)
        tracker.mark_done(stage, {"status": status})
    finally:
        run_logger.deactivate_stage()
        run_logger.write_stage_footer(
            stage,
            datetime.now().astimezone(),
            time.time() - t0,
            status,
            error,
        )


# ---------------------------------------------------------------------------
# Pipeline orchestrators
# ---------------------------------------------------------------------------

def _run_sequential(ordered: list[str], min_score: int, workers: int = 1,
                    validation_mode: str = "normal",
                    run_logger: _PipelineRunLogger | None = None) -> dict:
    """Execute stages one at a time (original behavior)."""
    results: list[dict] = []
    errors: dict[str, str] = {}
    pipeline_start = time.time()

    for name in ordered:
        meta = STAGE_META[name]
        console.print(f"\n{'=' * 70}")
        console.print(f"  [bold]STAGE: {name}[/bold] — {meta['desc']}")
        console.print(f"  Started: {datetime.now().strftime('%H:%M:%S')}")
        console.print(f"{'=' * 70}")

        t0 = time.time()
        started_at = datetime.now().astimezone()
        runner = _STAGE_RUNNERS[name]
        status = "unknown"
        error: str | None = None

        if run_logger:
            run_logger.write_stage_header(name, started_at)
            run_logger.activate_stage(name)

        try:
            kwargs: dict = {}
            if name in ("tailor", "cover"):
                kwargs["min_score"] = min_score
                kwargs["validation_mode"] = validation_mode
            if name in ("discover", "enrich"):
                kwargs["workers"] = workers
            result = runner(**kwargs)
            elapsed = time.time() - t0

            status = "ok"
            if isinstance(result, dict):
                status = result.get("status", "ok")
                if name == "discover":
                    sub_errors = [
                        f"{k}: {v}" for k, v in result.items()
                        if isinstance(v, str) and v.startswith("error")
                    ]
                    if sub_errors:
                        status = "partial"

        except Exception as e:
            elapsed = time.time() - t0
            status = f"error: {e}"
            error = str(e)
            log.exception("Stage '%s' crashed", name)
            console.print(f"\n  [red]STAGE FAILED:[/red] {e}")
        finally:
            if run_logger:
                run_logger.deactivate_stage()
                run_logger.write_stage_footer(
                    name,
                    datetime.now().astimezone(),
                    time.time() - t0,
                    status,
                    error,
                )

        results.append({"stage": name, "status": status, "elapsed": elapsed})
        if status not in ("ok", "partial"):
            errors[name] = status

        console.print(f"\n  Stage '{name}' completed in {elapsed:.1f}s — {status}")

    total_elapsed = time.time() - pipeline_start
    return {"stages": results, "errors": errors, "elapsed": total_elapsed}


def _run_streaming(ordered: list[str], min_score: int, workers: int = 1,
                   validation_mode: str = "normal",
                   run_logger: _PipelineRunLogger | None = None) -> dict:
    """Execute stages concurrently with DB as conveyor belt."""
    tracker = _StageTracker()
    stop_event = threading.Event()
    pipeline_start = time.time()

    console.print("\n  [bold cyan]STREAMING MODE[/bold cyan] — stages run concurrently")
    console.print(f"  Poll interval: {_STREAM_POLL_INTERVAL}s\n")

    # Mark stages NOT in `ordered` as done so downstream doesn't wait for them
    for stage in STAGE_ORDER:
        if stage not in ordered:
            tracker.mark_done(stage, {"status": "skipped"})

    # Launch each stage in its own thread
    threads: dict[str, threading.Thread] = {}
    start_times: dict[str, float] = {}

    for name in ordered:
        start_times[name] = time.time()
        if run_logger:
            target = _run_stage_streaming_logged
            args = (name, tracker, stop_event, run_logger, min_score, workers, validation_mode)
        else:
            target = _run_stage_streaming
            args = (name, tracker, stop_event, min_score, workers, validation_mode)
        t = threading.Thread(
            target=target,
            args=args,
            name=f"stage-{name}",
            daemon=True,
        )
        threads[name] = t
        t.start()
        console.print(f"  [dim]Started thread:[/dim] {name}")

    # Wait for all threads to finish
    try:
        for name in ordered:
            threads[name].join()
            elapsed = time.time() - start_times[name]
            console.print(
                f"  [green]Completed:[/green] {name} ({elapsed:.1f}s)"
            )
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted — stopping stages...[/yellow]")
        stop_event.set()
        for t in threads.values():
            t.join(timeout=10)

    total_elapsed = time.time() - pipeline_start

    # Build results from tracker
    all_results = tracker.get_results()
    results: list[dict] = []
    errors: dict[str, str] = {}

    for name in ordered:
        r = all_results.get(name, {"status": "unknown"})
        elapsed = time.time() - start_times.get(name, pipeline_start)
        status = r.get("status", "ok")

        results.append({"stage": name, "status": status, "elapsed": elapsed})
        if status not in ("ok", "partial", "skipped"):
            errors[name] = status

    return {"stages": results, "errors": errors, "elapsed": total_elapsed}


def run_pipeline(
    stages: list[str] | None = None,
    min_score: int = 7,
    dry_run: bool = False,
    stream: bool = False,
    workers: int = 1,
    validation_mode: str = "normal",
) -> dict:
    """Run pipeline stages.

    Args:
        stages: List of stage names, or None / ["all"] for full pipeline.
        min_score: Minimum fit score for tailor/cover stages.
        dry_run: If True, preview stages without executing.
        stream: If True, run stages concurrently (streaming mode).
        workers: Number of parallel threads for discovery/enrichment stages.

    Returns:
        Dict with keys: stages (list of result dicts), errors (dict), elapsed (float).
    """
    # Bootstrap
    load_env()
    ensure_dirs()
    init_db()

    # Resolve stages
    if stages is None:
        stages = ["all"]
    ordered = _resolve_stages(stages)

    # Banner
    mode = "streaming" if stream else "sequential"
    console.print()
    console.print(Panel.fit(
        f"[bold]ApplyPilot Pipeline[/bold] ({mode})",
        border_style="blue",
    ))
    console.print(f"  Min score:  {min_score}")
    console.print(f"  Workers:    {workers}")
    console.print(f"  Validation: {validation_mode}")
    console.print(f"  Stages:     {' -> '.join(ordered)}")

    # Pre-run stats
    pre_stats = get_stats()
    console.print(f"  DB:        {pre_stats['total']} jobs, {pre_stats['pending_detail']} pending enrichment")

    if dry_run:
        console.print(f"\n  [yellow]DRY RUN[/yellow] — would execute ({mode}):")
        for name in ordered:
            meta = STAGE_META[name]
            console.print(f"    {name:<12s}  {meta['desc']}")
        console.print("\n  No changes made.")
        return {"stages": [], "errors": {}, "elapsed": 0.0}

    run_logger = _PipelineRunLogger(ordered)
    run_logger.install()
    try:
        console.print(f"  Run logs:   {run_logger.run_dir}")
        console.print(f"  Run output: {run_logger.run_path}")
        for name in ordered:
            console.print(f"    {name:<12s} {run_logger.stage_paths[name]}")

        # Execute
        if stream:
            result = _run_streaming(ordered, min_score, workers=workers,
                                    validation_mode=validation_mode,
                                    run_logger=run_logger)
        else:
            result = _run_sequential(ordered, min_score, workers=workers,
                                     validation_mode=validation_mode,
                                     run_logger=run_logger)
    finally:
        run_logger.update_latest_pointer()
        run_logger.close()

    result["run_log_dir"] = str(run_logger.run_dir)
    result["run_log_path"] = str(run_logger.run_path)
    result["stage_log_paths"] = {
        stage: str(path) for stage, path in run_logger.stage_paths.items()
    }

    # Summary table
    console.print(f"\n{'=' * 70}")
    summary = Table(title="Pipeline Summary", show_header=True, header_style="bold")
    summary.add_column("Stage", style="bold")
    summary.add_column("Status")
    summary.add_column("Time", justify="right")

    for r in result["stages"]:
        elapsed_str = f"{r['elapsed']:.1f}s"
        status_display = r["status"][:30]
        if r["status"] == "ok":
            style = "green"
        elif r["status"] in ("partial", "skipped"):
            style = "yellow"
        else:
            style = "red"
        summary.add_row(r["stage"], f"[{style}]{status_display}[/{style}]", elapsed_str)

    summary.add_row("", "", "")
    summary.add_row("[bold]Total[/bold]", "", f"[bold]{result['elapsed']:.1f}s[/bold]")
    console.print(summary)

    # Final DB stats
    final = get_stats()
    console.print("\n  [bold]DB Final State:[/bold]")
    console.print(f"    Total jobs:     {final['total']}")
    console.print(f"    With desc:      {final['with_description']}")
    console.print(f"    Scored:         {final['scored']}")
    console.print(f"    Tailored:       {final['tailored']}")
    console.print(f"    Cover letters:  {final['with_cover_letter']}")
    console.print(f"    Ready to apply: {final['ready_to_apply']}")
    console.print(f"    Applied:        {final['applied']}")
    console.print(f"{'=' * 70}\n")

    return result
