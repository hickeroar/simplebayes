import threading


class ReadinessState:
    def __init__(self) -> None:
        self._is_ready = True
        self._lock = threading.Lock()

    @property
    def is_ready(self) -> bool:
        with self._lock:
            return self._is_ready

    def mark_ready(self) -> None:
        with self._lock:
            self._is_ready = True

    def mark_not_ready(self) -> None:
        with self._lock:
            self._is_ready = False
