"""Launch the local Agent web application with explicit operator configuration."""
from __future__ import annotations

import argparse
import sys

from .config import WebConfigurationError, build_interactive_application, load_web_configuration


def _port(value):
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Port must be an integer.") from exc
    if not 1 <= port <= 65_535:
        raise argparse.ArgumentTypeError("Port must be between 1 and 65535.")
    return port


def _workers(value):
    try:
        count = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Worker count must be an integer.") from exc
    if not 1 <= count <= 8:
        raise argparse.ArgumentTypeError("Worker count must be between 1 and 8.")
    return count


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m agent.web", description="Serve the local Agent browser interface.")
    parser.add_argument("--config", required=True, help="Operator JSON configuration path.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1).")
    parser.add_argument("--port", default=8000, type=_port, help="TCP port (default: 8000).")
    parser.add_argument("--max-workers", default=2, type=_workers,
                        help="Bounded local submission workers (1-8, default: 2).")
    arguments = parser.parse_args(argv)
    try:
        import uvicorn
        from .app import create_app
    except ImportError:
        print("Web dependencies are unavailable. Install requirements-web.txt in the Agent environment.", file=sys.stderr)
        return 2
    try:
        configuration = load_web_configuration(arguments.config)
        interactive = build_interactive_application(configuration)
        options = dict(input_sets=configuration.input_sets, max_workers=arguments.max_workers)
        if configuration.upload_root is not None:
            from agent.application.uploads import H5ADUploadAdmission
            options['uploads'] = H5ADUploadAdmission(interactive.resources, configuration.upload_root,
                max_bytes=configuration.upload_max_bytes, max_concurrent=configuration.upload_max_concurrent)
        application = create_app(interactive, **options)
    except (WebConfigurationError, ValueError, OSError) as exc:
        print(f"Agent web startup failed: {exc}", file=sys.stderr)
        return 2
    uvicorn.run(application, host=arguments.host, port=arguments.port, workers=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
