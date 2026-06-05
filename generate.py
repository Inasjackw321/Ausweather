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
GFS_BASE = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
GEOJSON_URLS = [
    "https://raw.githubusercontent.com/tonywr71/GeoJson-Data/master/australian-states.min.geojson",
    "https://raw.githubusercontent.com/rowanhogan/australian-states/master/states.min.geojson",
]
AUS_BOUNDS    = (112.0, 154.5, -44.5, -9.5)
FORECAST_DAYS = 5

HAZARDS = ["Wind", "Hail", "Flood", "Fire", "Tornado"]

RISK = {
    0: ("NONE", "#b0b8c0"),
    1: ("MRGL", "#4caf50"),
    2: ("SLGT", "#e6d800"),
    3: ("ENH",  "#ff9800"),
    4: ("MDT",  "#f44336"),
    5: ("HIGH", "#9c27b0"),
}

# State label positions (lon, lat, abbrev)
STATE_LABELS = [
    (146.5, -32.0, "NSW"),
    (144.5, -36.8, "VIC"),
    (144.0, -22.0, "QLD"),
    (135.5, -30.0, "SA"),
    (121.0, -27.0, "WA"),
    (133.5, -19.5, "NT"),
    (146.5, -42.0, "TAS"),
]


# ── GFS index helpers ─────────────────────────────────────────────────────────
def gfs_url(date_s, run_s, fhour, ext=""):
    return (f"{GFS_BASE}/gfs.{date_s}/{run_s}/atmos/"
            f"gfs.t{run_s}z.pgrb2.0p25.f{fhour:03d}{ext}")


def find_latest_run():
    now = datetime.utcnow()
    for h_back in range(0, 30):
        dt = now - timedelta(hours=h_back)
        rh = (dt.hour // 6) * 6
        dt = dt.replace(hour=rh, minute=0, second=0, microsecond=0)
        ds, rs = dt.strftime("%Y%m%d"), f"{rh:02d}"
        try:
            if requests.head(gfs_url(ds, rs, 24, ".idx"), timeout=5).status_code == 200:
                return ds, rs, dt
        except Exception:
            pass
    return None, None, None


def parse_idx(text):
    records = []
    lines = [l for l in text.strip().split("\n") if l]
    for i, line in enumerate(lines):
        parts = line.split(":")
        if len(parts) < 6:
            continue
        records.append({
            "start": int(parts[1]),
            "end":   int(lines[i + 1].split(":")[1]) if i + 1 < len(lines) else None,
            "var":   parts[3],
            "level": parts[4],
            "time":  parts[5],
        })
    return records


def find_record(records, varname, level_substr=None, time_substr=None, level_exclude=None):
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
    return (lats[(lats >= lat_min) & (lats <= lat_max)],
            lons[(lons >= lon_min) & (lons <= lon_max)],
            data[np.ix_((lats >= lat_min) & (lats <= lat_max),
                        (lons >= lon_min) & (lons <= lon_max))])


VAR_SPECS = [
    ("ugrd",  "UGRD", "10 m above ground", None,      None),
    ("vgrd",  "VGRD", "10 m above ground", None,      None),
    ("gust",  "GUST", "surface",           "PV=",     None),
    ("tmp2m", "TMP",  "2 m above ground",  None,      None),
    ("cape",  "CAPE", "surface",           None,      None),
    ("lftx",  "LFTX", "surface",           None,      None),
    ("apcp",  "APCP", "surface",           None,      "day acc"),
]


def fetch_day(date_s, run_s, fhour):
    idx_text = requests.get(gfs_url(date_s, run_s, fhour, ".idx"), timeout=15).text
    records  = parse_idx(idx_text)
    lats = lons = None
    fields = {}
    for key, var, lev_sub, lev_exc, time_sub in VAR_SPECS:
        rec = find_record(records, var, level_substr=lev_sub,
                          time_substr=time_sub, level_exclude=lev_exc)
        if rec is None:
            fields[key] = None
            continue
        try:
            url = gfs_url(date_s, run_s, fhour)
            l, o, g = grib_to_aus(download_range(url, rec["start"], rec["end"]))
            if lats is None:
                lats, lons = l, o
            fields[key] = g
            print(".", end="", flush=True)
        except Exception as e:
            print(f"![{key}:{e}]", end="", flush=True)
            fields[key] = None
    return lats, lons, fields


# ── Risk calculations ─────────────────────────────────────────────────────────
def compute_risks(fields, prev_apcp=None):
    def f(key):
        v = fields.get(key)
        fallback = next((x for x in fields.values() if x is not None), np.zeros((141, 171)))
        return v if v is not None else np.zeros(fallback.shape)

    ugrd, vgrd = f("ugrd"), f("vgrd")
    gust  = f("gust")
    tmp   = f("tmp2m") - 273.15
    cape  = f("cape")
    lftx  = f("lftx")

    apcp_cum = f("apcp")
    daily_precip = (np.maximum(0.0, apcp_cum - prev_apcp)
                    if prev_apcp is not None else apcp_cum)

    wind_kph = np.sqrt(ugrd**2 + vgrd**2) * 3.6
    gust_kph = gust * 3.6
    precip_prob = np.clip(daily_precip / 0.15, 0, 95)

    wr = np.zeros(wind_kph.shape, dtype=int)
    for t, l in [(35,1),(46,2),(58,3),(72,4),(90,5)]:
        wr[wind_kph >= t] = l
    gr = np.zeros_like(wr)
    for t, l in [(44,1),(60,2),(75,3),(95,4),(120,5)]:
        gr[gust_kph >= t] = l
    wind_risk = np.maximum(wr, gr)

    hail_risk = np.zeros(cape.shape, dtype=int)
    for (c, p), l in [((300,20),1),((700,30),2),((1300,40),3),((2000,50),4),((2800,60),5)]:
        hail_risk[(cape >= c) & (precip_prob >= p)] = l

    flood_risk = np.zeros(daily_precip.shape, dtype=int)
    for t, l in [(10,1),(25,2),(50,3),(100,4),(150,5)]:
        flood_risk[daily_precip >= t] = l

    fwi = (tmp - 15.0) * 1.8 + wind_kph * 0.9
    fire_risk = np.zeros(fwi.shape, dtype=int)
    for t, l in [(8,1),(20,2),(38,3),(58,4),(80,5)]:
        fire_risk[fwi >= t] = l
    fire_risk[daily_precip >= 3] = 0

    tor_risk = np.zeros(cape.shape, dtype=int)
    for (c, li, g, p), l in [
        ((600,-2,0,0),1), ((1200,-3,0,30),2),
        ((2000,-4,55,40),3), ((3000,-5,75,0),4)
    ]:
        mask = (cape >= c) & (lftx <= li)
        if g: mask &= (gust_kph >= g)
        if p: mask &= (precip_prob >= p)
        tor_risk[mask] = l

    return {"Wind": wind_risk, "Hail": hail_risk,
            "Flood": flood_risk, "Fire": fire_risk, "Tornado": tor_risk}



# ── Map geometry ──────────────────────────────────────────────────────────────
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


# ── Map rendering ─────────────────────────────────────────────────────────────
CMAP = mcolors.ListedColormap([RISK[i][1] for i in range(6)])
NORM = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5], CMAP.N)
LON0, LON1, LAT0, LAT1 = AUS_BOUNDS


def draw_panel(ax, lats, lons, risk_grid, title, polys, clip_path, show_labels=True):
    ax.set_xlim(LON0, LON1)
    ax.set_ylim(LAT0, LAT1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("#1a2e40")

    if polys:
        ax.add_collection(PatchCollection(
            [Polygon(p, closed=True) for p in polys],
            facecolor="#253545", edgecolor="none", zorder=1,
        ))

    GL, GLatG = np.meshgrid(lons, lats)
    cf = ax.contourf(GL, GLatG, risk_grid,
                     levels=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
                     cmap=CMAP, norm=NORM, alpha=0.85, zorder=2)

    if clip_path is not None:
        cp = PathPatch(clip_path, transform=ax.transData, visible=False)
        ax.add_patch(cp)
        cf.set_clip_path(cp)

    if polys:
        ax.add_collection(PatchCollection(
            [Polygon(p, closed=True) for p in polys],
            facecolor="none", edgecolor="#4a7a96", linewidth=0.6, zorder=4,
        ))

    # State abbreviation labels
    if show_labels:
        for lon, lat, abbrev in STATE_LABELS:
            ax.text(lon, lat, abbrev, fontsize=5.5, color="#6aaccc",
                    ha="center", va="center", zorder=5, fontfamily="monospace",
                    fontweight="bold", alpha=0.75,
                    path_effects=[pe.withStroke(linewidth=1.5, foreground="#1a2e40")])

    ax.set_title(title.upper(), fontsize=9, fontweight="bold",
                 color="#cce8ff", pad=3, fontfamily="monospace")


def render_day_image(lats, lons, risk_grids, date_label, polys, clip_path):
    fig = plt.figure(figsize=(22, 5.6), facecolor="#0c1824")
    fig.text(0.5, 0.975,
             f"AUSTRALIA SEVERE WEATHER OUTLOOK  ·  {date_label}",
             ha="center", va="top", fontsize=12, fontweight="bold",
             color="white", fontfamily="monospace",
             path_effects=[pe.withStroke(linewidth=3, foreground="#0c1824")])

    panel_w = 0.188
    for i, hazard in enumerate(HAZARDS):
        x0 = 0.006 + i * (panel_w + 0.006)
        ax = fig.add_axes([x0, 0.06, panel_w, 0.875])
        draw_panel(ax, lats, lons, risk_grids[hazard], hazard, polys, clip_path)

    lax = fig.add_axes([0.006, 0.005, 0.988, 0.050])
    lax.axis("off")
    lax.set_facecolor("#06101a")
    for i in range(6):
        label, color = RISK[i]
        x = 0.005 + i * 0.165
        lax.add_patch(plt.Rectangle((x, 0.05), 0.155, 0.90,
                                     color=color, transform=lax.transAxes, clip_on=False))
        fc = "black" if i < 3 else "white"
        lax.text(x + 0.077, 0.52, label, ha="center", va="center",
                 fontsize=8, fontweight="bold", color=fc,
                 fontfamily="monospace", transform=lax.transAxes)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64


# ── HTML generation ───────────────────────────────────────────────────────────
def make_html(images, dates, run_label, timestamp):
    day_labels = ["Day 1 — Today", "Day 2 — Tomorrow", "Day 3", "Day 4", "Day 5"]

    sections = []
    for label, date_str, img in zip(day_labels, dates, images):
        sections.append(
            f'<div class="day">'
            f'<h2>{label} <span class="date">{date_str}</span></h2>'
            f'<img src="data:image/png;base64,{img}" alt="{date_str} outlook">'
            f'</div>'
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Australia Severe Weather Outlook</title>
<style>
  body {{
    background: #0c1824;
    color: #b0ccde;
    font-family: "Courier New", Courier, monospace;
    margin: 0;
    padding: 16px;
  }}
  h1 {{
    font-size: 18px;
    color: #d8eaf8;
    letter-spacing: 2px;
    margin-bottom: 4px;
  }}
  .meta {{
    font-size: 11px;
    color: #4a7090;
    margin-bottom: 20px;
  }}
  .day {{
    margin-bottom: 32px;
  }}
  h2 {{
    font-size: 13px;
    color: #d8eaf8;
    letter-spacing: 1px;
    margin-bottom: 6px;
  }}
  .date {{
    color: #4a9ed0;
    font-weight: normal;
  }}
  img {{
    width: 100%;
    display: block;
    border: 1px solid #1c3550;
  }}
  footer {{
    font-size: 10px;
    color: #4a7090;
    border-top: 1px solid #1c3550;
    padding-top: 10px;
    margin-top: 10px;
  }}
  footer a {{ color: #3a7090; text-decoration: none; }}
</style>
</head>
<body>
<h1>&#9928; AUSTRALIA SEVERE WEATHER OUTLOOK</h1>
<div class="meta">Data: {run_label} &nbsp;&bull;&nbsp; Generated: {timestamp} UTC</div>

{''.join(sections)}

<footer>
  NOT FOR OPERATIONAL USE &nbsp;&bull;&nbsp;
  For official warnings visit <a href="https://www.bom.gov.au" target="_blank">bom.gov.au</a>
  &nbsp;&bull;&nbsp; Data: NOAA GFS via AWS Open Data
</footer>
</body>
</html>"""


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 58)
    print("  Australia Severe Weather Outlook  [GFS live data]")
    print("=" * 58)

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
    run_label = f"NOAA GFS {run_dt.strftime('%Y-%m-%d %HZ')}"
    print(f"      {run_label}")

    print("\n[3/4] Downloading GFS fields (7 vars × 5 days)...")
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
        images.append(render_day_image(all_lats, all_lons, risks, date_str, polys, clip_path))
    print("done")

    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    with open(out, "w") as f:
        f.write(make_html(images, dates, run_label, timestamp))

    print(f"\n  Saved → {out}\n")


if __name__ == "__main__":
    main()
