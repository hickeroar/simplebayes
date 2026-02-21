import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_healthz(port: int, timeout_seconds: float = 8.0) -> bool:
    deadline = time.time() + timeout_seconds
    url = f"http://127.0.0.1:{port}/healthz"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=0.5) as response:
                return response.status == 200
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.2)
    return False


def test_cli_module_help_exits_zero():
    result = subprocess.run(
        [sys.executable, "-m", "simplebayes.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0
    assert "Run the simplebayes API server." in result.stdout


def test_cli_module_fails_with_invalid_env_port():
    env = os.environ.copy()
    env["SIMPLEBAYES_PORT"] = "not-a-number"
    result = subprocess.run(
        [sys.executable, "-m", "simplebayes.cli"],
        capture_output=True,
        text=True,
        timeout=10,
        env=env,
        check=False,
    )
    assert result.returncode != 0
    assert "invalid literal for int()" in result.stderr


def test_cli_server_starts_and_serves_healthz():
    port = _find_free_port()
    with subprocess.Popen(
        [
            sys.executable,
            "-m",
            "simplebayes.cli",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ) as process:
        try:
            assert _wait_for_healthz(port)
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
