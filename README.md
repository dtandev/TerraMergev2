## Run (Hydra)

Create and activate env (once), then run the orchestrator:

```bash
# from repo root
python -m pip install -e .
python rule_them_all.py

### Verbosity
Bump console logs:
```bash
python rule_them_all.py logging.console_level=DEBUG