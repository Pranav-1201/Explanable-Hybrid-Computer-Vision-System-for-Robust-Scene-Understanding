"""Production entry point: the Flask app under waitress (Windows and Linux).

    venv/Scripts/python.exe serve.py            # dev machine
    python serve.py                             # container (PORT/THREADS from env)

`python app.py` remains the Flask development server.
"""
import os

from waitress import serve

from serving.config import env_int


def main() -> None:
    from app import app  # loads the model; STRICT_STARTUP makes failures fatal

    host = os.environ.get("HOST", "0.0.0.0")
    port = env_int("PORT", 5000)
    threads = env_int("THREADS", 4)
    print(f"[INFO] waitress serving on http://{host}:{port} with {threads} threads")
    serve(app, host=host, port=port, threads=threads)


if __name__ == "__main__":
    main()
