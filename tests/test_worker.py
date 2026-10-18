from threading import Event

from app import worker


def test_run_worker_stops_without_claiming_jobs(monkeypatch):
    shutdown = Event()
    shutdown.set()
    claimed = []

    monkeypatch.setattr(
        worker.repository,
        "claim_next_job",
        lambda *args: claimed.append(args),
    )

    worker.run_worker(
        worker_id="test-worker",
        shutdown_event=shutdown,
        poll_seconds=0,
        lease_seconds=30,
    )

    assert claimed == []


def test_run_worker_waits_when_no_job_is_available(monkeypatch):
    shutdown = Event()
    waits = []
    claimed = []

    def fake_claim(*args):
        claimed.append(args)
        shutdown.set()
        return None

    monkeypatch.setattr(
        worker.repository,
        "claim_next_job",
        fake_claim,
    )
    monkeypatch.setattr(
        shutdown,
        "wait",
        lambda seconds: waits.append(seconds),
    )

    worker.run_worker(
        worker_id="test-worker",
        shutdown_event=shutdown,
        poll_seconds=2,
        lease_seconds=30,
    )

    assert len(claimed) == 1
    assert claimed[0] == ("test-worker", 30)
    assert waits == [2]


def test_run_worker_continues_after_queue_poll_failure(monkeypatch):
    shutdown = Event()
    waits = []
    calls = []

    def fake_claim(*_args):
        calls.append(True)

        if len(calls) == 1:
            raise RuntimeError("database unavailable")

        shutdown.set()
        return None

    monkeypatch.setattr(
        worker.repository,
        "claim_next_job",
        fake_claim,
    )
    monkeypatch.setattr(
        shutdown,
        "wait",
        lambda seconds: waits.append(seconds),
    )

    worker.run_worker(
        worker_id="test-worker",
        shutdown_event=shutdown,
        poll_seconds=3,
        lease_seconds=30,
    )

    assert len(calls) == 2
    assert waits == [3, 3]


def test_run_worker_handles_job_finalization_failure(monkeypatch):
    shutdown = Event()
    processed = []
    calls = []

    def fake_claim(*_args):
        calls.append(True)

        if len(calls) == 1:
            return {
                "id": "job-123",
                "attempt_count": 1,
                "max_attempts": 3,
            }

        shutdown.set()
        return None

    def fake_process(*args):
        processed.append(args)
        raise RuntimeError("finalization failed")

    monkeypatch.setattr(
        worker.repository,
        "claim_next_job",
        fake_claim,
    )
    monkeypatch.setattr(
        worker,
        "process_job",
        fake_process,
    )
    monkeypatch.setattr(
        shutdown,
        "wait",
        lambda *_args: None,
    )

    worker.run_worker(
        worker_id="test-worker",
        shutdown_event=shutdown,
        poll_seconds=1,
        lease_seconds=30,
    )

    assert len(processed) == 1
    assert processed[0][1] == "test-worker"
    assert processed[0][3] == 30
