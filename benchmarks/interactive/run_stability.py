"""Retired Q2.2 entry point; read historical JSON without regenerating inputs."""
from .run_breadth import _original as read_report

def run_stability(*args, **kwargs):
    raise RuntimeError("Historical qualification runner retired by R5; preserved reports remain readable. "
                       "Use the current interactive harness for new checks.")

def main(*args, **kwargs):
    raise RuntimeError("Historical qualification runner retired by R5; preserved reports remain readable. "
                       "Use the current interactive harness for new checks.")

if __name__ == '__main__':
    main()
