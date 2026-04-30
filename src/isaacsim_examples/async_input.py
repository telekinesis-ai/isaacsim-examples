"""Non-blocking stdin helpers that keep the Isaac Sim viewport responsive."""

import queue
import threading

_input_queue: queue.Queue = queue.Queue()


def _input_thread_func(prompt: str):
    """Read one line from stdin and put it on the queue (runs in a daemon thread)."""
    try:
        line = input(prompt)
        _input_queue.put(line)
    except EOFError:
        _input_queue.put(None)


def async_input(prompt: str = "> "):
    """Start a background thread that waits for user input.

    Call ``poll_input()`` each frame to check if a line has arrived.
    """
    t = threading.Thread(target=_input_thread_func, args=(prompt,), daemon=True)
    t.start()


def poll_input():
    """Return the user's input string if available, otherwise ``None``.

    A return value of ``False`` signals EOF.
    """
    try:
        line = _input_queue.get_nowait()
        if line is None:
            return False  # EOF
        return line
    except queue.Empty:
        return None
