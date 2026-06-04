#!/usr/bin/env python3
"""
Australian SPC-style 5-Day Severe Weather Outlook
Real data: NOAA GFS 0.25° global forecast via NOAA AWS Open Data
  https://noaa-gfs-bdp-pds.s3.amazonaws.com
"""

import sys, os, base64
from io import BytesIO
from datetime import datetime, timedelta

import requests
import numpy as np
import eccodes
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.patches import Polygon, PathPatch
from matplotlib.collections import PatchCollection
from matplotlib.path import Path


# ── Config ────────────────────────────────────────────────────────────────────
GFS_BASE   = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
GEOJSON_URLS = [
    "https://raw.githubusercontent.com/tonywr71/GeoJson-Data/master/australian-states.min.geojson",
    "https://raw.githubusercontent.com/rowanhogan/australian-states/master/states.min.geojson",
]
AUS_BOUNDS   = (112.0, 154.5, -44.5, -9.5)   # lon_min, lon_max, lat_min, lat_max
FORECAST_DAYS = 5

HAZARDS = ["Wind", "Hail", "Flood", "Fire", "Tornado"]
RISK = {
    0: ("NONE",  "#b0b8c0"),
    1: ("MRGL",  "#4caf50"),
    2: ("SLGT",  "#ffeb3b"),
    3: ("ENH",   "#ff9800"),
    4: ("MDT",   "#f44336"),
    5: ("HIGH",  "#9c27b0"),
}


# ── GFS index helpers ─────────────────────────────────────────────────────────
def gfs_url(date_s, run_s, fhour, ext=""):
    return (f"{GFS_BASE}/gfs.{date_s}/{run_s}/atmos/"
            f"gfs.t{run_s}z.pgrb2.0p25.f{fhour:03d}{ext}")


def find_latest_run():
    """Walk back through GFS run times until we find one with f024 ready."""
    now = datetime.utcnow()
    for h_back in range(0, 30, 1):
        dt   = now - timedelta(hours=h_back)
        rh   = (dt.hour // 6) * 6
        dt   = dt.replace(hour=rh, minute=0, second=0, microsecond=0)
        ds   = dt.strftime("%Y%m%d")
        rs   = f"{rh:02d}"
        try:
            resp = requests.head(gfs_url(ds, rs, 24, ".idx"), timeout=5)
            if resp.status_code == 200:
                return ds, rs, dt
        except Exception:
            pass
    return None, None, None


def parse_idx(text):
    """Parse GFS .idx into list of record dicts with byte ranges."""
    records = []
    lines   = [l for l in text.strip().split("\n") if l]
    for i, line in enumerate(lines):
        parts = line.split(":")
        if len(parts) < 6:
            continue
        start = int(parts[1])
        end   = int(lines[i + 1].split(":")[1]) if i + 1 < len(lines) else None
        records.append({
            "start": start,
            "end":   end,
            "var":   parts[3],
            "level": parts[4],
            "time":  parts[5],
        })
    return records


def find_record(records, varname, level_substr=None, time_substr=None,
                level_exclude=None):
    """Return the first matching record, with optional string filters."""
    for rec in records:
        if rec["var"] != varname:
            continue
        if level_substr and level_substr not in rec["level"]:
            continue
        if level_exclude and level_exclude in rec["level"]:
            continue
        if time_substr and time_substr not in rec["time"]:
            continue
        return rec
    return None


# ── GRIB2 download & parse ────────────────────────────────────────────────────
def download_range(url, start, end):
    end_s = str(end - 1) if end else ""
    r = requests.get(url, headers={"Range": f"bytes={start}-{end_s}"}, timeout=60)
    r.raise_for_status()
    return r.content


def grib_to_aus(grib_bytes):
    """Decode a GRIB2 message and return (lats, lons, 2D-values) for Australia."""
    gid = eccodes.codes_new_from_message(grib_bytes)
    try:
        ni   = eccodes.codes_get(gid, "Ni")
        nj   = eccodes.codes_get(gid, "Nj")
        lat1 = eccodes.codes_get(gid, "latitudeOfFirstGridPointInDegrees")
        lat2 = eccodes.codes_get(gid, "latitudeOfLastGridPointInDegrees")
        lon1 = eccodes.codes_get(gid, "longitudeOfFirstGridPointInDegrees")
        lon2 = eccodes.codes_get(gid, "longitudeOfLastGridPointInDegrees")
        vals = eccodes.codes_get_values(gid)
    finally:
        eccodes.codes_release(gid)

    lats = np.linspace(lat1, lat2, nj)
    lons = np.linspace(lon1, lon2, ni)
    data = vals.reshape(nj, ni)

    lon_min, lon_max, lat_min, lat_max = AUS_BOUNDS
    lat_mask = (lats >= lat_min) & (lats <= lat_max)
    lon_mask = (lons >= lon_min) & (lons <= lon_max)
    return lats[lat_mask], lons[lon_mask], data[np.ix_(lat_mask, lon_mask)]


def fetch_var(date_s, run_s, fhour, rec):
    """Download one GRIB2 variable and return (lats, lons, grid) for Australia."""
    url   = gfs_url(date_s, run_s, fhour)
    raw   = download_range(url, rec["start"], rec["end"])
    return grib_to_aus(raw)


# ── Fetch one forecast day ────────────────────────────────────────────────────
#
# Variables:
#   ugrd   UGRD  10 m above ground   → 10m wind U-component (m/s)
#   vgrd   VGRD  10 m above ground   → 10m wind V-component (m/s)
#   gust   GUST  surface              → surface wind gust (m/s)
#   tmp2m  TMP   2 m above ground    → 2m air temperature (K)
#   cape   CAPE  surface              → CAPE (J/kg)
#   lftx   LFTX  surface              → surface lifted index (K)
#   apcp   APCP  surface, "day acc"   → cumulative precipitation since run start (mm)

VAR_SPECS = [
    ("ugrd",  "UGRD", "10 m above ground", None,      None,      None),
    ("vgrd",  "VGRD", "10 m above ground", None,      None,      None),
    ("gust",  "GUST", "surface",           None,      None,      "PV="),
    ("tmp2m", "TMP",  "2 m above ground",  None,      None,      None),
    ("cape",  "CAPE", "surface",           None,      None,      None),
    ("lftx",  "LFTX", "surface",           None,      None,      None),
    ("apcp",  "APCP", "surface",           None,      "day acc", None),
]


def fetch_day(date_s, run_s, fhour):
    """Return (lats, lons, fields_dict) for one forecast time."""
    url_idx = gfs_url(date_s, run_s, fhour, ".idx")
    idx_text = requests.get(url_idx, timeout=15).text
    records  = parse_idx(idx_text)

    lats = lons = None
    fields = {}
    for key, var, lev_sub, lev_exc_none, time_sub, lev_exc in VAR_SPECS:
        rec = find_record(records, var,
                          level_substr=lev_sub,
                          time_substr=time_sub,
                          level_exclude=lev_exc)
        if rec is None:
            print(f"    [{key}] not found", flush=True)
            fields[key] = None
            continue
        try:
            l, o, g = fetch_var(date_s, run_s, fhour, rec)
            if lats is None:
                lats, lons = l, o
            fields[key] = g
            print(".", end="", flush=True)
        except Exception as e:
            print(f"  ![{key}: {e}]", end="", flush=True)
            fields[key] = None

    return lats, lons, fields


# ── Risk calculations ─────────────────────────────────────────────────────────
def zeros_like(fields):
    for v in fields.values():
        if v is not None:
            return np.zeros(v.shape, dtype=int)
    return np.zeros((141, 171), dtype=int)


def compute_risks(fields, prev_apcp=None):
    """Return dict hazard→2D risk array (0–5) from raw GFS fields."""

    def f(key):
        v = fields.get(key)
        return v if v is not None else np.zeros_like(zeros_like(fields), dtype=float)

    ugrd = f("ugrd")
    vgrd = f("vgrd")
    gust = f("gust")
    tmp  = f("tmp2m") - 273.15          # K → °C
    cape = f("cape")
    lftx = f("lftx")

    # Cumulative APCP in mm; subtract previous day to get daily total
    apcp_cum = f("apcp")
    if prev_apcp is not None:
        daily_precip = np.maximum(0.0, apcp_cum - prev_apcp)
    else:
        daily_precip = apcp_cum

    wind_ms  = np.sqrt(ugrd**2 + vgrd**2)
    wind_kph = wind_ms * 3.6
    gust_kph = gust * 3.6

    # Precipitation probability proxy (0–100)
    precip_prob = np.clip(daily_precip / 0.15, 0, 95)

    # ── Wind ──────────────────────────────────────────────────────────────
    wr = np.zeros(wind_kph.shape, dtype=int)
    for thr, lvl in [(35,1),(46,2),(58,3),(72,4),(90,5)]:
        wr[wind_kph >= thr] = lvl
    gr = np.zeros_like(wr)
    for thr, lvl in [(44,1),(60,2),(75,3),(95,4),(120,5)]:
        gr[gust_kph >= thr] = lvl
    wind_risk = np.maximum(wr, gr)

    # ── Hail ──────────────────────────────────────────────────────────────
    hail_risk = np.zeros(cape.shape, dtype=int)
    for (c_thr, p_thr), lvl in [
        ((300,  20), 1),
        ((700,  30), 2),
        ((1300, 40), 3),
        ((2000, 50), 4),
        ((2800, 60), 5),
    ]:
        hail_risk[(cape >= c_thr) & (precip_prob >= p_thr)] = lvl

    # ── Flood ─────────────────────────────────────────────────────────────
    flood_risk = np.zeros(daily_precip.shape, dtype=int)
    for thr, lvl in [(10,1),(25,2),(50,3),(100,4),(150,5)]:
        flood_risk[daily_precip >= thr] = lvl

    # ── Fire ──────────────────────────────────────────────────────────────
    fwi = (tmp - 15.0) * 1.8 + wind_kph * 0.9
    fire_risk = np.zeros(fwi.shape, dtype=int)
    for thr, lvl in [(8,1),(20,2),(38,3),(58,4),(80,5)]:
        fire_risk[fwi >= thr] = lvl
    fire_risk[daily_precip >= 3] = 0   # rain suppresses fire risk

    # ── Tornado ───────────────────────────────────────────────────────────
    tor_risk = np.zeros(cape.shape, dtype=int)
    for (c_thr, li_thr, g_thr, p_thr), lvl in [
        ((600,  -2,   0,  0),  1),
        ((1200, -3,   0, 30),  2),
        ((2000, -4,  55, 40),  3),
        ((3000, -5,  75,  0),  4),
    ]:
        mask = (cape >= c_thr) & (lftx <= li_thr)
        if g_thr:  mask &= (gust_kph >= g_thr)
        if p_thr:  mask &= (precip_prob >= p_thr)
        tor_risk[mask] = lvl

    return {
        "Wind":    wind_risk,
        "Hail":    hail_risk,
        "Flood":   flood_risk,
        "Fire":    fire_risk,
        "Tornado": tor_risk,
    }


# ── Map data ──────────────────────────────────────────────────────────────────
def fetch_geojson():
    for url in GEOJSON_URLS:
        try:
            r = requests.get(url, timeout=20)
            if r.ok:
                return r.json()
        except Exception:
            pass
    return None


def geojson_to_polygons(geojson):
    polys = []
    for feat in geojson.get("features", []):
        geom  = feat.get("geometry", {})
        gtype = geom.get("type", "")
        coords = geom.get("coordinates", [])
        rings = ([coords[0]] if gtype == "Polygon"
                 else [p[0] for p in coords] if gtype == "MultiPolygon"
                 else [])
        for ring in rings:
            a = np.array(ring)
            if a.ndim == 2 and a.shape[1] >= 2 and len(a) >= 3:
                polys.append(a[:, :2])
    return polys


def polys_to_clip_path(polys):
    verts, codes = [], []
    for p in polys:
        verts.extend(p.tolist())
        verts.append(p[0].tolist())
        codes += [Path.MOVETO] + [Path.LINETO] * (len(p) - 1) + [Path.CLOSEPOLY]
    return Path(verts, codes) if verts else None


# ── Rendering ─────────────────────────────────────────────────────────────────
CMAP = mcolors.ListedColormap([RISK[i][1] for i in range(6)])
NORM = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5], CMAP.N)
LON0, LON1, LAT0, LAT1 = AUS_BOUNDS


def draw_panel(ax, lats, lons, risk_grid, title, polys, clip_path):
    ax.set_xlim(LON0, LON1)
    ax.set_ylim(LAT0, LAT1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("#1a2e40")

    # Land fill
    if polys:
        ax.add_collection(PatchCollection(
            [Polygon(p, closed=True) for p in polys],
            facecolor="#2c3e50", edgecolor="none", zorder=1,
        ))

    # Risk surface
    GL, GLatG = np.meshgrid(lons, lats)
    cf = ax.contourf(GL, GLatG, risk_grid,
                     levels=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
                     cmap=CMAP, norm=NORM, alpha=0.82, zorder=2)

    # Clip to Australian coastline
    if clip_path is not None:
        cp = PathPatch(clip_path, transform=ax.transData, visible=False)
        ax.add_patch(cp)
        cf.set_clip_path(cp)          # works in matplotlib 3.8+

    # State borders on top
    if polys:
        ax.add_collection(PatchCollection(
            [Polygon(p, closed=True) for p in polys],
            facecolor="none", edgecolor="#5a8faa", linewidth=0.5, zorder=4,
        ))

    ax.set_title(title.upper(), fontsize=9, fontweight="bold",
                 color="#cce8ff", pad=3, fontfamily="monospace")


def render_day(lats, lons, risk_grids, date_label, polys, clip_path):
    fig = plt.figure(figsize=(21, 5.8), facecolor="#0c1824")
    fig.text(0.5, 0.966,
             f"AUSTRALIA SEVERE WEATHER OUTLOOK  ·  {date_label}",
             ha="center", va="top", fontsize=13, fontweight="bold",
             color="white", fontfamily="monospace",
             path_effects=[pe.withStroke(linewidth=3, foreground="#0c1824")])

    panel_w = 0.189
    for i, hazard in enumerate(HAZARDS):
        x0 = 0.005 + i * (panel_w + 0.006)
        ax = fig.add_axes([x0, 0.075, panel_w, 0.855])
        draw_panel(ax, lats, lons, risk_grids[hazard], hazard, polys, clip_path)

    # Legend
    lax = fig.add_axes([0.005, 0.005, 0.99, 0.060])
    lax.axis("off")
    lax.set_facecolor("#060e16")
    for i in range(6):
        label, color = RISK[i]
        x = 0.005 + i * 0.165
        lax.add_patch(plt.Rectangle((x, 0.07), 0.15, 0.86,
                                     color=color, transform=lax.transAxes, clip_on=False))
        fc = "black" if i < 3 else "white"
        lax.text(x + 0.075, 0.50, label, ha="center", va="center",
                 fontsize=8, fontweight="bold", color=fc,
                 fontfamily="monospace", transform=lax.transAxes)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64


# ── HTML ──────────────────────────────────────────────────────────────────────
def make_html(images, dates, run_label):
    labels = ["TODAY", "TOMORROW", "DAY 3", "DAY 4", "DAY 5"]
    blocks = "\n".join(
        f'  <div class="day">\n'
        f'    <div class="day-hdr">{labels[i]} &mdash; {dates[i]}</div>\n'
        f'    <img src="data:image/png;base64,{img}" alt="{dates[i]}">\n'
        f'  </div>'
        for i, img in enumerate(images)
    )
    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Australia Severe Weather Outlook</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0b1520;color:#adc8e0;font-family:"Courier New",monospace;font-size:13px}}
header{{background:#070f18;padding:14px 22px;border-bottom:2px solid #18304a}}
h1{{font-size:17px;color:#6aacde;letter-spacing:3px}}
.sub{{color:#365870;margin-top:5px;font-size:10px;letter-spacing:1px}}
.days{{padding:14px 10px;max-width:1700px;margin:0 auto}}
.day{{margin-bottom:18px}}
.day-hdr{{color:#4a8ab8;font-size:11px;letter-spacing:2px;margin-bottom:5px;padding-left:2px}}
.day img{{width:100%;display:block;border:1px solid #18304a;border-radius:2px}}
footer{{text-align:center;padding:9px;font-size:10px;color:#28506a;
        border-top:1px solid #18304a;margin-top:8px}}
footer a{{color:#285870;text-decoration:none}}
</style>
</head>
<body>
<header>
  <h1>&#9928; AUSTRALIA SEVERE WEATHER OUTLOOK</h1>
  <div class="sub">GENERATED {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")} &nbsp;&bull;&nbsp; {run_label} &nbsp;&bull;&nbsp; WIND / HAIL / FLOOD / FIRE / TORNADO</div>
</header>
<div class="days">
{blocks}
</div>
<footer>NOT FOR OPERATIONAL USE &nbsp;&bull;&nbsp; For official warnings visit <a href="https://www.bom.gov.au">bom.gov.au</a></footer>
</body>
</html>"""


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 56)
    print("  Australia Severe Weather Outlook  [GFS live data]")
    print("=" * 56)

    print("\n[1/4] Fetching Australia map geometry...")
    geojson = fetch_geojson()
    if geojson:
        polys     = geojson_to_polygons(geojson)
        clip_path = polys_to_clip_path(polys)
        print(f"      {len(polys)} state polygons loaded")
    else:
        polys, clip_path = [], None
        print("      WARNING: no map geometry")

    print("\n[2/4] Locating latest GFS run...")
    date_s, run_s, run_dt = find_latest_run()
    if not date_s:
        print("      ERROR: GFS data unavailable")
        sys.exit(1)
    run_label = f"NOAA GFS  {run_dt.strftime('%Y-%m-%d %HZ')}"
    print(f"      {run_label}")

    print("\n[3/4] Downloading GFS fields for Australia (7 vars × 5 days)...")
    all_lats = all_lons = None
    all_risks, dates = [], []
    prev_apcp = None

    for day in range(FORECAST_DAYS):
        fhour    = (day + 1) * 24
        valid_dt = run_dt + timedelta(hours=fhour)
        date_str = valid_dt.strftime("%A, %d %b %Y")
        dates.append(date_str)
        print(f"\n  Day {day+1}  {date_str}  (f{fhour:03d})", end="  ", flush=True)

        lats, lons, fields = fetch_day(date_s, run_s, fhour)
        if all_lats is None:
            all_lats, all_lons = lats, lons

        risks = compute_risks(fields, prev_apcp)
        all_risks.append(risks)

        if fields.get("apcp") is not None:
            prev_apcp = fields["apcp"].copy()

    print("\n\n[4/4] Rendering graphics...")
    images = []
    for day, (risks, date_str) in enumerate(zip(all_risks, dates)):
        print(f"  Day {day+1}...", end=" ", flush=True)
        images.append(render_day(all_lats, all_lons, risks, date_str, polys, clip_path))
    print("done")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    with open(out, "w") as f:
        f.write(make_html(images, dates, run_label))

    print(f"\n  Saved → {out}")
    print(f"  Open index.html in your browser.\n")


if __name__ == "__main__":
    main()
