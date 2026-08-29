"""Reader for SWDE (Standard Wymiany Danych Ewidencyjnych) EGiB deliveries.

SWDE is a proprietary, flat-text relational format from GUGiK (SWDE w.2.00) — not a GIS format
GDAL understands. Reverse-engineered against real files this session (see the plan for the
investigation trail). Key facts that drive this parser:

- Encoding is ISO-8859-2 (confirmed empirically by decoding known Polish text — CP852/CP1250
  produce garbage on the same bytes). Files use CRLF line endings.
- A `.swd` file declares one of two sub-types via the `NS,ZN,<TYPE>` header line: `EWOPIS`
  (descriptive attributes only, no coordinates) or `EWMAPA` (carries real geometry: a point
  dictionary, table `G5PZG` mapping point id -> X/Y, plus parcel geometry records in table
  `G5G_DZE`, each a parcel id + one or more point-id rings). **This declared type is not trusted**
  — confirmed on real data (rok_2021/2023 szczycieński) that a file can declare `NS,ZN,EWOPIS` and
  still carry full, dense ring geometry for `G5DZE`/`G5UZG`/`G5KKL` (tens of thousands of rings,
  file sizes on par with same-unit EWMAPA-declared years) — some vendor's export tool just doesn't
  set this field to match its actual content. `read_all_layers` always attempts to parse and only
  returns `{}` when it genuinely finds no points/rings, regardless of the declared type.
- After the header/schema section (which this parser skips — field names are given inline on every
  data line anyway, so the schema dictionary isn't needed to parse values), every data record has
  the same shape: a header line `XX,,<table>,<id>,...;` (XX in {RO,RD,RC,RL,RP}) starts a record
  for `<table>` with primary id `<id>`; `D,<field>,<type>,<value>` lines set attributes;
  `WG,<relation>,<target_table>,<target_id>;` lines record a foreign-key relation; a geometry
  block is delimited by `GL;` ... `GX;`, containing one or more rings delimited by `K,<sign>;` ...
  `PZ;` (sign `+` = exterior, `-` = hole), each ring being an ordered list of either
  `P,P,<table>,<point_id>;` references into the point dictionary (used by `G5G_DZE` parcels --
  or, in some deliveries, the ring-and-attributes-combined plain `G5DZE` table, see
  `_GEOM_TABLE_CANDIDATES`), or
  `P,G,<x>,<y>,;` inline coordinates given directly in the ring (used by `G5KKL`/`G5G_KKL`, see
  below) — both forms can appear as a *record header's* single point definition too (no ring
  involved), which is how `G5PZG` point-dictionary entries are given; `X;` ends the record
  (a `XC,<checksum>;` variant is used by some deliveries instead — confirmed on a real rok_2016
  ostródzki file where EVERY record in the file, not just one table, uses this terminator; treated
  identically to `X;`).
- A second geometry table, `G5KKL` (or `G5G_KKL` in some deliveries — the exact table name is not
  consistent across real deliveries, confirmed empirically across rok_2014/2016/2020 files for two
  different powiats), carries **self-contained** ring geometry (inline `P,G,` coordinates, no
  `G5PZG` lookup needed) plus two attributes on every single record (confirmed: 100% coverage
  across every sampled file): `G5OZU` (land-use designation, e.g. "R"/"Ł"/"Ps"/"N") and `G5OZK`
  (soil classification, e.g. "IVa"/"VI"). This is the legacy system's *combined*
  classification-and-land-use contour — the modern EGB schema splits the same concept into two
  separate feature classes, `EGB_KonturUzytkuGruntowego` (by `G5OZU`) and
  `EGB_KonturKlasyfikacyjny` (by `G5OZK`) — so this reader exposes the same KKL geometry under
  both canonical names (see `_build_kkl_polygons`, `read_all_layers`). Without this, every SWDE
  delivery in the dataset was silently missing both layers entirely, even where the source file
  genuinely carries the geometry (confirmed: e.g. 18 079 KKL polygons vs 17 832 parcels in one
  real 2020 ostródzki delivery — not a marginal amount of data).

Validated against a real overlapping case: `281701_1` in `rok_2017/.../geometryczne/281701_1.swd`
reconstructs to polygons in the same coordinate range as the same unit's later GML/GDB delivery
(EPSG:2178 / CS2000 zone 7 coordinates in the 5.9M/7.49M range) — see tests/test_readers_swde.py
for the synthetic-fixture regression version of this check.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from loguru import logger
from shapely.geometry import Polygon

_ENCODING = "iso-8859-2"

_RECORD_HEADER_PREFIXES = ("RO,", "RD,", "RC,", "RL,", "RP,")

_GEOM_TABLE = "G5G_DZE"
# Some deliveries (confirmed real: rok_2021/2023 szczycienski) carry ring geometry directly on
# the plain "G5DZE" table (no middle "G_") instead of the standard "G5G_DZE" -- the same table
# name that _ATTRIBUTE_TABLES lists as an attributes-only join source elsewhere. For those
# deliveries G5DZE plays both roles at once (it's the *only* parcel table in the file). Checked in
# this order -- first delivery to have non-empty records under either name wins.
_GEOM_TABLE_CANDIDATES = ("G5G_DZE", "G5DZE")
_POINT_TABLE = "G5PZG"
_ATTRIBUTE_TABLES = ("G5DZE", "G5O_DZE")
_ID_FIELD = "G5IDD"

# Combined classification+land-use contour table — name varies across real deliveries (both seen
# empirically), so both are checked; whichever is present in a given file is used.
_KKL_TABLES = ("G5KKL", "G5G_KKL")
_KKL_ATTR_FIELDS = ("G5IDK", "G5OZU", "G5OZK", "G5PEW")


def _read_lines(path: Path) -> list[str]:
    with open(path, encoding=_ENCODING, newline="") as f:
        text = f.read()
    return text.split("\n")


def _swde_type(lines: list[str]) -> str | None:
    for line in lines[:30]:
        line = line.strip()
        if line.startswith("NS,ZN,"):
            return line.split(",", 2)[2].rstrip(";").strip()
    return None


def _parse_records(lines: list[str]) -> dict[str, list[dict[str, Any]]]:
    """
    Generic SWDE data-section parser: returns {table_name: [record, ...]}.

    Each record dict has: "_id" (primary id from the header line), plain attribute keys from
    `D,` lines, "_xy" (tuple, only for point-definition records), and "_rings" (list of
    (sign, [point_id_or_xy, ...]) — each ring is either a list of point-id strings (`P,P,`
    references, e.g. `G5G_DZE`) or a list of (x, y) tuples (`P,G,` inline coordinates, e.g.
    `G5KKL`/`G5G_KKL`) — never mixed within one real ring, but callers should key off the element
    type rather than assume one or the other).
    """
    records: dict[str, list[dict[str, Any]]] = {}
    in_data_section = False
    current: dict[str, Any] | None = None
    current_table: str | None = None
    current_ring: list[str] | None = None
    current_ring_sign: str | None = None

    for raw_line in lines:
        line = raw_line.rstrip("\r\n").strip()
        if not line:
            continue

        if not in_data_section:
            if line in ("SO;", "ST;"):
                in_data_section = True
            continue

        if line.startswith(_RECORD_HEADER_PREFIXES) and line[2:3] == ",":
            parts = line.split(",")
            current_table = parts[2] if len(parts) > 2 else None
            current = {"_id": parts[3].rstrip(";") if len(parts) > 3 else None}
            # Some deliveries (confirmed real: rok_2020 elblaski) reference ring points by this
            # secondary/alias id (e.g. "PZG_1") via a `P,K,` ring marker instead of referencing
            # the primary id via `P,P,` -- see _build_point_dict, which keys by both.
            if len(parts) > 4:
                current["_alt_id"] = parts[4].rstrip(";")
            continue

        if current is None:
            continue  # stray line outside any record (e.g. trailing schema noise) — ignore

        if line.startswith("D,"):
            parts = line.split(",", 3)
            if len(parts) >= 2:
                field = parts[1]
                value = parts[3] if len(parts) > 3 else ""
                current[field] = value
            continue

        if line.startswith("WG,"):
            parts = line.split(",")
            current.setdefault("_relations", []).append(tuple(parts[1:]))
            continue

        if line.startswith("P,G,"):
            parts = line.split(",")
            try:
                xy = (float(parts[2]), float(parts[3]))
            except (IndexError, ValueError):
                logger.debug("Nie udało się sparsować punktu: {}", line)
                continue
            # Inside a ring (K,.../PZ; block) this is an inline vertex coordinate (G5KKL/G5G_KKL
            # style) — append it to the ring being built. Outside a ring, it's a standalone
            # point-dictionary definition (G5PZG style) — one per record.
            if current_ring is not None:
                current_ring.append(xy)
            else:
                current["_xy"] = xy
            continue

        if line.startswith("P,P,") or line.startswith("P,K,"):
            parts = line.split(",")
            point_id = parts[-1].rstrip(";")
            if current_ring is not None:
                current_ring.append(point_id)
            continue

        if line.startswith("GL;"):
            current["_rings"] = []
            continue

        if line.startswith("GX;"):
            continue

        if line.startswith("K,"):
            current_ring_sign = line.split(",")[1].rstrip(";")
            current_ring = []
            continue

        if line.startswith("PZ;"):
            if current_ring is not None and "_rings" in current:
                current["_rings"].append((current_ring_sign, current_ring))
            current_ring = None
            current_ring_sign = None
            continue

        if line.startswith("X;") or line.startswith("XC,"):
            # `XC,<checksum>;` is a real alternative record terminator (confirmed: a whole
            # rok_2016 ostródzki delivery uses it for every table, not just one) — without
            # recognizing it, no record in such a file would ever be committed, silently
            # returning zero geometry for an otherwise well-formed file.
            if current_table is not None:
                records.setdefault(current_table, []).append(current)
            current = None
            current_table = None
            continue

        # Anything else (stray schema-section leftovers, section markers) is intentionally ignored.

    return records


def _build_point_dict(records: dict[str, list[dict[str, Any]]]) -> dict[str, tuple[float, float]]:
    points: dict[str, tuple[float, float]] = {}
    for rec in records.get(_POINT_TABLE, []):
        xy = rec.get("_xy")
        if xy is None:
            continue
        # Indexed by BOTH the primary id and the secondary/alias id (when present) -- confirmed
        # real case (rok_2020 elblaski): rings reference points via `P,K,<alias>;` using the
        # alias id (e.g. "PZG_1"), not the primary GUID id every other seen delivery's `P,P,`
        # rings reference. Keying by both means either ring-reference convention resolves.
        for pid in (rec.get("_id"), rec.get("_alt_id")):
            if pid is not None:
                points[pid] = xy
    return points


def _build_parcel_polygons(
    records: dict[str, list[dict[str, Any]]],
    points: dict[str, tuple[float, float]],
    crs: str = "EPSG:2178",
) -> gpd.GeoDataFrame:
    # Prefer G5G_DZE (records carry ONLY geometry there; attributes come from a separate G5DZE/
    # G5O_DZE table via _enrich_with_attributes) but fall back to G5DZE itself when that's the
    # only parcel table the delivery has -- see _GEOM_TABLE_CANDIDATES.
    geom_table = next((t for t in _GEOM_TABLE_CANDIDATES if records.get(t)), _GEOM_TABLE)

    rows: list[dict[str, Any]] = []
    skipped = 0

    for rec in records.get(geom_table, []):
        rings = rec.get("_rings") or []
        if not rings:
            skipped += 1
            continue

        exterior = None
        holes: list[list[tuple[float, float]]] = []
        for sign, ring_points in rings:
            # A ring is either a list of G5PZG point-id strings (`P,P,` references -- the
            # standard G5G_DZE style) or a list of inline (x, y) tuples (`P,G,` coordinates given
            # directly in the ring, no point-dictionary lookup needed -- confirmed real case:
            # rok_2013 elblaski's G5G_DZE uses this style, same as G5KKL/G5G_KKL always does, see
            # _build_kkl_polygons). Never mixed within one ring, so checking the first element's
            # type is enough to route the whole ring correctly.
            if ring_points and isinstance(ring_points[0], tuple):
                coords = list(ring_points)
            else:
                coords = [points[p] for p in ring_points if p in points]
            if len(coords) < 3:
                continue
            if sign == "+" and exterior is None:
                exterior = coords
            else:
                holes.append(coords)

        if exterior is None:
            skipped += 1
            continue

        try:
            geom = Polygon(exterior, holes or None)
        except Exception:
            logger.debug("Nie udało się zbudować poligonu dla rekordu {}", rec.get("_id"))
            skipped += 1
            continue

        rows.append({"idDzialki": rec.get(_ID_FIELD), "_geom_id": rec.get("_id"), "geometry": geom})

    if skipped:
        logger.warning(
            "SWDE: pominięto {} rekordów geometrii {} (brak pierścienia / brakujące punkty)",
            skipped,
            geom_table,
        )

    if not rows:
        return gpd.GeoDataFrame(columns=["idDzialki", "geometry"], geometry="geometry")

    return gpd.GeoDataFrame(pd.DataFrame(rows), geometry="geometry", crs=crs)


def _build_kkl_polygons(
    records: dict[str, list[dict[str, Any]]],
    crs: str = "EPSG:2178",
) -> gpd.GeoDataFrame:
    """Build polygons from the legacy combined classification+land-use contour table
    (`G5KKL`/`G5G_KKL` — see module docstring). Unlike `G5G_DZE`, rings here are already lists of
    (x, y) tuples (inline `P,G,` coordinates) — no `G5PZG` point-dictionary lookup needed."""
    table_name = next((t for t in _KKL_TABLES if records.get(t)), None)
    if table_name is None:
        return gpd.GeoDataFrame(columns=[*_KKL_ATTR_FIELDS, "geometry"], geometry="geometry")

    rows: list[dict[str, Any]] = []
    skipped = 0

    for rec in records.get(table_name, []):
        rings = rec.get("_rings") or []
        if not rings:
            skipped += 1
            continue

        exterior = None
        holes: list[list[tuple[float, float]]] = []
        for sign, coords in rings:
            pts = [c for c in coords if isinstance(c, tuple)]
            if len(pts) < 3:
                continue
            if sign == "+" and exterior is None:
                exterior = pts
            else:
                holes.append(pts)

        if exterior is None:
            skipped += 1
            continue

        try:
            geom = Polygon(exterior, holes or None)
        except Exception:
            logger.debug("Nie udało się zbudować poligonu KKL dla rekordu {}", rec.get("_id"))
            skipped += 1
            continue

        rows.append({**{f: rec.get(f) for f in _KKL_ATTR_FIELDS}, "geometry": geom})

    if skipped:
        logger.warning(
            "SWDE: pominięto {} rekordów geometrii {} (brak pierścienia)", skipped, table_name
        )

    if not rows:
        return gpd.GeoDataFrame(columns=[*_KKL_ATTR_FIELDS, "geometry"], geometry="geometry")

    return gpd.GeoDataFrame(pd.DataFrame(rows), geometry="geometry", crs=crs)


def _enrich_with_attributes(
    gdf: gpd.GeoDataFrame, records: dict[str, list[dict[str, Any]]]
) -> gpd.GeoDataFrame:
    """Best-effort join of extra attributes (area, land-use class...) from G5DZE/G5O_DZE by idDzialki."""
    for table in _ATTRIBUTE_TABLES:
        rows = records.get(table, [])
        if not rows:
            continue
        attrs = pd.DataFrame(rows)
        if _ID_FIELD not in attrs.columns:
            continue
        attrs = attrs.rename(columns={_ID_FIELD: "idDzialki"}).drop(
            columns=[c for c in ("_id", "_relations", "_rings", "_xy") if c in attrs.columns]
        )
        attrs = attrs.drop_duplicates(subset="idDzialki", keep="first")
        gdf = gdf.merge(attrs, on="idDzialki", how="left", suffixes=("", f"_{table}"))
    return gdf


def read_all_layers(swd_path: Path, crs: str = "EPSG:2178") -> dict[str, gpd.GeoDataFrame]:
    """
    Read a `.swd` file into {layer_name: GeoDataFrame}.

    Returns `{}` for files with no usable geometry (nothing this pipeline can use for a polygon
    layer) and for anything that fails to parse. Returns at least `{"DzialkaEwidencyjna": gdf}`
    when at least one parcel polygon could be reconstructed, plus `"KonturUzytkuGruntowego"` and
    `"KonturKlasyfikacyjny"` (same geometry, both derived from the combined `G5KKL`/`G5G_KKL`
    table — see module docstring) when that table is present and yields at least one polygon.

    The declared `NS,ZN,<TYPE>` header is NOT used to gate parsing — confirmed on real data
    (rok_2021/2023 szczycieński) that a file can declare `NS,ZN,EWOPIS` and still carry full,
    dense `G5DZE`/`G5UZG`/`G5KKL` ring geometry (tens of thousands of rings, file sizes on par
    with same-unit EWMAPA-declared years). Trusting the header alone silently discarded real,
    fully-reconstructable parcel data. Instead, this always attempts to parse and build geometry;
    genuinely attribute-only files (no `G5PZG` points, no inline `P,G,` ring coordinates) still
    naturally yield `{}` via the empty-points/empty-polygon checks below — the declared type is
    only used for a diagnostic log line when it disagrees with what was actually found.

    `crs` defaults to EPSG:2178 (CS2000 zone 7, covering powiat szczycieński / most of
    woj. warmińsko-mazurskie — the project's contracted scope). SWDE carries no CRS metadata of
    its own anywhere in the file (unlike GDB/GML/SHP, which all embed it), so this can't be
    auto-detected the way `shp_reader._resolve_unresolvable_crs` does — it must be supplied by the
    caller for any delivery outside that zone. Threaded through from `prepare.swde_crs` in
    `conf/prepare/default.yaml`, not hardcoded, so a future run against a different powiat/zone
    doesn't silently mislabel its coordinates.
    """
    try:
        lines = _read_lines(swd_path)
    except Exception:
        logger.exception("Nie można odczytać {}", swd_path)
        return {}

    swde_type = _swde_type(lines)
    records = _parse_records(lines)
    points = _build_point_dict(records)
    # Not gated on `points` being non-empty: a parcel ring can be self-contained inline (x, y)
    # coordinates needing no G5PZG lookup at all (confirmed real case: rok_2013 elblaski's
    # G5G_DZE uses this style, same as G5KKL/G5G_KKL always does) -- _build_parcel_polygons
    # handles both ring styles and is the single source of truth for whether anything usable
    # came out.
    if not points:
        logger.debug(
            "SWDE {} (zadeklarowany typ '{}'): brak tabeli punktów ({}) -- próbuję mimo to (pierścienie mogą mieć współrzędne inline).",
            swd_path.name,
            swde_type,
            _POINT_TABLE,
        )

    gdf = _build_parcel_polygons(records, points, crs=crs)
    if gdf.empty:
        logger.info(
            "SWDE {} (zadeklarowany typ '{}'): nie zrekonstruowano żadnego poligonu działki — brak geometrii, pomijam.",
            swd_path.name,
            swde_type,
        )
        return {}

    if swde_type != "EWMAPA":
        logger.warning(
            "SWDE {} zadeklarowany jako '{}' (nie EWMAPA), ale zawiera {} realnych poligonów "
            "działek — używam mimo to (deklarowany typ jest niewiarygodny dla tej dostawy).",
            swd_path.name,
            swde_type,
            len(gdf),
        )

    gdf = _enrich_with_attributes(gdf, records)
    logger.info("SWDE {}: zrekonstruowano {} poligonów działek.", swd_path.name, len(gdf))
    layers: dict[str, gpd.GeoDataFrame] = {"DzialkaEwidencyjna": gdf}

    kkl_gdf = _build_kkl_polygons(records, crs=crs)
    if not kkl_gdf.empty:
        logger.info(
            "SWDE {}: zrekonstruowano {} konturów klasyfikacyjno-użytkowych (G5KKL/G5G_KKL).",
            swd_path.name,
            len(kkl_gdf),
        )
        layers["KonturUzytkuGruntowego"] = kkl_gdf
        layers["KonturKlasyfikacyjny"] = kkl_gdf.copy()

    return layers


__all__ = ["read_all_layers"]
