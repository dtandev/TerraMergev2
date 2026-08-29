# Must be the first thing pytest imports in this session — see the identical comment in
# src/main.py. Importing GDAL's `osgeo` bindings before `pyarrow.dataset` segfaults at
# interpreter shutdown in this environment (verified: reversing the order avoids it). Within a
# single pytest process, whichever test module happens to import `osgeo` first (e.g. via
# src.prepare_data.extract_polygons_from_gdb) would otherwise "win" the race by file-collection
# order; importing it here in conftest.py — always loaded before any test module — pins the safe
# order regardless of which test files exist or what order they run in.
import pyarrow.dataset  # noqa: F401
