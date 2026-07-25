"""
Smoke test for the pipeline entry point (src/main.py).

Run with: just test
"""


def test_main_module_imports_cleanly():
    """
    Importing src.main pulls in every pipeline stage's top-level dependencies
    (prepare_data, features, common) — a regression here means a broken import
    somewhere in the chain, which would otherwise only surface at runtime.
    """
    import src.main as main

    assert callable(main.run_all)
