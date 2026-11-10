"""Semantic search benchmarker: chunking strategies x BM25 / dense retrieval."""


def main() -> None:
    """Console-script entry point (see ``[project.scripts]`` in pyproject.toml)."""
    # Imported lazily so `import semantic_search_benchmarker` stays lightweight.
    from semantic_search_benchmarker.benchmarker import main as _main

    _main()


def dashboard() -> None:
    """Console-script entry point that launches the Streamlit dashboard."""
    import sys
    from pathlib import Path

    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", str(Path(__file__).with_name("app.py")), *sys.argv[1:]]
    sys.exit(stcli.main())
