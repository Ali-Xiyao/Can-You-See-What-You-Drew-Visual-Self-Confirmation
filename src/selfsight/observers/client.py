"""Client for a dedicated observer Python environment/process."""

from __future__ import annotations

import json
import subprocess
import threading
from collections.abc import Sequence
from io import TextIOWrapper
from pathlib import Path

from typing_extensions import Self

from selfsight.observers.protocol import decode_result, encode_request
from selfsight.schemas import BlindObservationRequest, ObservationResult


class ObserverServiceClient:
    def __init__(self, command: Sequence[str], log_path: str | Path | None = None) -> None:
        self.command = tuple(command)
        self.log_path = Path(log_path) if log_path else None
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        self._stderr_handle: TextIOWrapper | None = None

    def start(self) -> None:
        if self._process is not None:
            return
        stderr_target: int | TextIOWrapper = subprocess.PIPE
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            stderr_path = self.log_path.with_suffix(self.log_path.suffix + ".stderr.log")
            self._stderr_handle = stderr_path.open("a", encoding="utf-8")
            stderr_target = self._stderr_handle
        self._process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_target,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )

    def observe(self, request: BlindObservationRequest) -> ObservationResult:
        self.start()
        assert self._process is not None and self._process.stdin and self._process.stdout
        wire = encode_request(request)
        with self._lock:
            if self.log_path:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a", encoding="utf-8") as log:
                    log.write(wire + "\n")
            self._process.stdin.write(wire + "\n")
            self._process.stdin.flush()
            line = self._process.stdout.readline()
        if not line:
            stderr = self._process.stderr.read() if self._process.stderr else ""
            if self.log_path:
                if self._stderr_handle:
                    self._stderr_handle.flush()
                stderr_path = self.log_path.with_suffix(self.log_path.suffix + ".stderr.log")
                if stderr_path.exists():
                    stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
            raise RuntimeError(f"Observer process exited unexpectedly: {stderr[-4000:]}")
        try:
            response = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                "Observer JSONL protocol violation: non-JSON stdout was emitted: "
                f"{line[:300]!r}"
            ) from error
        if not response.get("ok"):
            raise RuntimeError(f"Observer error: {response.get('error')}: {response.get('detail')}")
        return decode_result(json.dumps(response["result"], ensure_ascii=False))

    def close(self) -> None:
        if self._process is None:
            return
        if self._process.stdin:
            self._process.stdin.close()
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            self._process.wait(timeout=10)
        self._process = None
        if self._stderr_handle:
            self._stderr_handle.close()
            self._stderr_handle = None

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
