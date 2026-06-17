#!/usr/bin/env python3
"""
Ausweather — Australian Severe Weather Outlook
Real data: NOAA GFS 0.25° via AWS Open Data
"""

import sys, os, base64, json
from io import BytesIO
from datetime import datetime, timedelta

import requests
import numpy as np
from scipy.ndimage import gaussian_filter
import eccodes
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.patches import Polygon, PathPatch
from matplotlib.collections import PatchCollection
from matplotlib.path import Path


# ── Config ─────────────────────────────────────────────────────────────────────
GFS_BASE = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
GEOJSON_URLS = [
    "https://raw.githubusercontent.com/tonywr71/GeoJson-Data/master/australian-states.min.geojson",
    "https://raw.githubusercontent.com/rowanhogan/australian-states/master/states.min.geojson",
]
AUS_BOUNDS    = (112.0, 154.5, -44.5, -9.5)
LON0, LON1, LAT0, LAT1 = AUS_BOUNDS
FORECAST_DAYS = 5

HAZARDS = ["Wind", "Hail", "Flood", "Tornado"]
HAZARD_ICONS = {"Wind": "💨", "Hail": "🌨", "Flood": "🌊", "Tornado": "🌪"}

STATE_LABELS = [
    (146.5, -32.0, "NSW"),
    (144.5, -36.8, "VIC"),
    (144.0, -22.0, "QLD"),
    (135.5, -30.0, "SA"),
    (121.0, -27.0, "WA"),
    (133.5, -19.5, "NT"),
    (146.5, -42.0, "TAS"),
]

# Heatmap colour ramp: background → faint cyan → vivid cyan → green → yellow → orange → purple
HEATMAP_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "ausweather", [
        (0.00, "#050a10"),
        (0.04, "#050a10"),   # dead zone for near-zero gaussian bleed
        (0.16, "#004d66"),   # faint dark cyan
        (0.32, "#00bcd4"),   # cyan   (MRGL)
        (0.50, "#4caf50"),   # green  (SLGT)
        (0.65, "#ffeb3b"),   # yellow (ENH)
        (0.80, "#ff5722"),   # orange (MDT)
        (1.00, "#9c27b0"),   # purple (HIGH)
    ], N=256
)


# ── GFS index helpers ──────────────────────────────────────────────────────────
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


# ── GRIB2 download & parse ─────────────────────────────────────────────────────
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
    return (lats[(lats >= LAT0) & (lats <= LAT1)],
            lons[(lons >= LON0) & (lons <= LON1)],
            data[np.ix_((lats >= LAT0) & (lats <= LAT1),
                        (lons >= LON0) & (lons <= LON1))])


VAR_SPECS = [
    ("ugrd",  "UGRD", "10 m above ground", None,  None),
    ("vgrd",  "VGRD", "10 m above ground", None,  None),
    ("gust",  "GUST", "surface",           "PV=", None),
    ("tmp2m", "TMP",  "2 m above ground",  None,  None),
    ("cape",  "CAPE", "surface",           None,  None),
    ("lftx",  "LFTX", "surface",           None,  None),
    ("apcp",  "APCP", "surface",           None,  "day acc"),
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
            l, o, g = grib_to_aus(download_range(gfs_url(date_s, run_s, fhour),
                                                  rec["start"], rec["end"]))
            if lats is None:
                lats, lons = l, o
            fields[key] = g
            print(".", end="", flush=True)
        except Exception as e:
            print(f"![{key}:{e}]", end="", flush=True)
            fields[key] = None
    return lats, lons, fields


# ── Risk calculations ──────────────────────────────────────────────────────────
def compute_risks(fields, prev_apcp=None):
    def f(key):
        v = fields.get(key)
        fallback = next((x for x in fields.values() if x is not None), np.zeros((141, 171)))
        return v if v is not None else np.zeros(fallback.shape)

    ugrd, vgrd = f("ugrd"), f("vgrd")
    gust  = f("gust")
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
            "Flood": flood_risk, "Tornado": tor_risk}


# ── Map geometry ───────────────────────────────────────────────────────────────
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


# ── Heatmap rendering ──────────────────────────────────────────────────────────
def render_hazard_image(lats, lons, risk_grid, polys, clip_path):
    fig = plt.figure(figsize=(14, 10.5), facecolor="#050a10")
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(LON0, LON1)
    ax.set_ylim(LAT0, LAT1)
    ax.axis("off")
    ax.set_facecolor("#050a10")

    # Land base
    if polys:
        ax.add_collection(PatchCollection(
            [Polygon(p, closed=True) for p in polys],
            facecolor="#0c1824", edgecolor="none", zorder=1))

    GL, GLatG = np.meshgrid(lons, lats)
    norm = mcolors.Normalize(vmin=0, vmax=5)

    # Glow pass — heavy blur, lower alpha
    glow = gaussian_filter(risk_grid.astype(float), sigma=4.0)
    cf_glow = ax.contourf(GL, GLatG, glow,
                          levels=np.linspace(0.15, 5, 60),
                          cmap=HEATMAP_CMAP, norm=norm, alpha=0.45, zorder=2)

    # Core pass — tighter blur, stronger alpha
    core = gaussian_filter(risk_grid.astype(float), sigma=2.0)
    cf_core = ax.contourf(GL, GLatG, core,
                          levels=np.linspace(0.15, 5, 60),
                          cmap=HEATMAP_CMAP, norm=norm, alpha=0.90, zorder=3)

    # Clip both to Australia coastline
    if clip_path is not None:
        cp1 = PathPatch(clip_path, transform=ax.transData, visible=False)
        ax.add_patch(cp1)
        cf_glow.set_clip_path(cp1)
        cp2 = PathPatch(clip_path, transform=ax.transData, visible=False)
        ax.add_patch(cp2)
        cf_core.set_clip_path(cp2)

    # State borders
    if polys:
        ax.add_collection(PatchCollection(
            [Polygon(p, closed=True) for p in polys],
            facecolor="none", edgecolor="#1e3a52", linewidth=0.8, zorder=5))

    # State abbreviation labels
    for lon, lat, abbrev in STATE_LABELS:
        ax.text(lon, lat, abbrev, fontsize=9, color="#2a5575",
                ha="center", va="center", zorder=6,
                fontfamily="monospace", fontweight="bold",
                path_effects=[pe.withStroke(linewidth=2, foreground="#050a10")])

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=85, bbox_inches="tight", pad_inches=0,
                facecolor="#050a10")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64


# ── HTML ───────────────────────────────────────────────────────────────────────
def make_html(images, dates, run_label, timestamp):
    """
    images: list of dicts, one per day: {hazard: base64_png_string}
    dates:  list of date strings, one per day
    """
    images_json = json.dumps(images)
    dates_json  = json.dumps(dates)
    icons_json  = json.dumps(HAZARD_ICONS)

    hazard_btns = "\n".join(
        f'<button class="hbtn{" active" if i == 0 else ""}" '
        f'data-h="{h}" onclick="setHazard(\'{h}\')">'
        f'{HAZARD_ICONS[h]} {h.upper()}</button>'
        for i, h in enumerate(HAZARDS)
    )

    day_ticks = "\n".join(
        f'<div class="tick{" active" if i == 0 else ""}" '
        f'data-i="{i}" onclick="seek({i})">'
        f'{"TODAY" if i == 0 else "TOMORROW" if i == 1 else f"DAY {i+1}"}</div>'
        for i in range(FORECAST_DAYS)
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ausweather</title>
<style>
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

body {{
  background: #050a10;
  color: #fff;
  font-family: -apple-system, "Segoe UI", Arial, sans-serif;
  height: 100vh;
  overflow: hidden;
  user-select: none;
}}

/* ── Top HUD ── */
.hud-top {{
  position: fixed; top: 0; left: 0; right: 0; z-index: 30;
  display: flex; align-items: center; gap: 20px; flex-wrap: wrap;
  padding: 10px 18px;
  background: linear-gradient(180deg, rgba(5,10,16,0.95) 0%, transparent 100%);
  pointer-events: none;
}}

.logo {{
  font-size: 22px; font-weight: 800; letter-spacing: 3px;
  color: #fff; text-transform: uppercase;
}}
.logo em {{ color: #4dd0e1; font-style: normal; }}

.cs-wrap {{
  display: flex; align-items: center; gap: 6px;
}}
.cs-labels {{
  display: flex; justify-content: space-between;
  font-size: 9px; color: #5a8faa; letter-spacing: 1px;
  margin-bottom: 3px;
}}
.cs-bar {{
  width: 150px; height: 10px; border-radius: 5px;
  background: linear-gradient(90deg, #004d66, #00bcd4, #4caf50, #ffeb3b, #ff5722, #9c27b0);
}}
.cs-text {{
  display: flex; flex-direction: column;
}}

.time-info {{
  margin-left: auto; text-align: right;
}}
.time-date {{ font-size: 11px; color: #5a8faa; letter-spacing: 1px; }}
.time-day  {{ font-size: 14px; font-weight: 600; color: #fff; letter-spacing: 1px; }}

/* ── Map ── */
#map {{
  position: fixed; inset: 0;
  display: flex; align-items: center; justify-content: center;
  overflow: hidden;
}}
.map-img {{
  position: absolute; inset: 0;
  width: 100%; height: 100%;
  object-fit: contain;
  transition: opacity 0.55s ease;
}}
#img-b {{ opacity: 0; }}

/* ── Bottom HUD ── */
.hud-bot {{
  position: fixed; bottom: 0; left: 0; right: 0; z-index: 30;
  padding: 18px 18px 14px;
  background: linear-gradient(0deg, rgba(5,10,16,0.97) 60%, transparent 100%);
}}

/* Hazard buttons */
.hazard-row {{
  display: flex; justify-content: center; gap: 8px; margin-bottom: 14px;
}}
.hbtn {{
  padding: 7px 18px; border-radius: 20px;
  border: 1px solid #1a3a50;
  background: rgba(5,10,16,0.7);
  color: #5a8faa; font-size: 12px; font-weight: 600; letter-spacing: 0.5px;
  cursor: pointer; transition: all 0.2s;
}}
.hbtn:hover {{ border-color: #4dd0e1; color: #cce8f0; background: rgba(0,30,50,0.8); }}
.hbtn.active {{
  background: rgba(0,60,90,0.8); border-color: #4dd0e1; color: #4dd0e1;
}}

/* Playback row */
.play-row {{
  display: flex; align-items: center; gap: 12px;
}}

.pbtn {{
  width: 38px; height: 38px; border-radius: 50%; flex-shrink: 0;
  background: rgba(77,208,225,0.12);
  border: 1.5px solid rgba(77,208,225,0.5);
  color: #4dd0e1; font-size: 15px; cursor: pointer;
  display: flex; align-items: center; justify-content: center;
  transition: background 0.2s, border-color 0.2s;
}}
.pbtn:hover {{ background: rgba(77,208,225,0.25); border-color: #4dd0e1; }}

.timeline {{ flex: 1; }}
.tl-bar {{
  height: 4px; background: #0d2035; border-radius: 2px; cursor: pointer;
  position: relative; margin-bottom: 8px;
}}
.tl-fill {{
  position: absolute; left: 0; top: 0; height: 100%;
  background: #4dd0e1; border-radius: 2px;
  transition: width 0.4s ease;
  pointer-events: none;
}}
.tl-dot {{
  position: absolute; top: 50%; transform: translate(-50%, -50%);
  width: 12px; height: 12px; border-radius: 50%;
  background: #4dd0e1; border: 2px solid #050a10;
  transition: left 0.4s ease;
  pointer-events: none;
}}
.tl-ticks {{
  display: flex; justify-content: space-between;
}}
.tick {{
  font-size: 9px; color: #3a6070; letter-spacing: 1px; cursor: pointer;
  text-align: center; flex: 1; transition: color 0.2s;
  padding: 2px 0;
}}
.tick:hover {{ color: #8ac0d0; }}
.tick.active {{ color: #4dd0e1; font-weight: 700; }}

/* Source tag */
.src-tag {{
  position: fixed; bottom: 8px; right: 14px; z-index: 40;
  font-size: 9px; color: #1e3a50; letter-spacing: 1px;
  pointer-events: none;
}}
</style>
</head>
<body>

<!-- Top HUD -->
<div class="hud-top">
  <div class="logo">Aus<em>weather</em></div>
  <div class="cs-wrap">
    <div class="cs-text">
      <div class="cs-labels"><span>NONE</span><span>MRGL</span><span>SLGT</span><span>ENH</span><span>MDT</span><span>HIGH</span></div>
      <div class="cs-bar"></div>
    </div>
  </div>
  <div class="time-info">
    <div class="time-date" id="time-date"></div>
    <div class="time-day"  id="time-day"></div>
  </div>
</div>

<!-- Map crossfade -->
<div id="map">
  <img id="img-a" class="map-img" src="" alt="">
  <img id="img-b" class="map-img" src="" alt="">
</div>

<!-- Bottom HUD -->
<div class="hud-bot">
  <div class="hazard-row">
    {hazard_btns}
  </div>
  <div class="play-row">
    <button class="pbtn" id="pbtn" onclick="togglePlay()" title="Play / Pause">&#9654;</button>
    <div class="timeline">
      <div class="tl-bar" id="tl-bar" onclick="barClick(event)">
        <div class="tl-fill" id="tl-fill" style="width:0%"></div>
        <div class="tl-dot"  id="tl-dot"  style="left:0%"></div>
      </div>
      <div class="tl-ticks">
        {day_ticks}
      </div>
    </div>
  </div>
</div>

<div class="src-tag">Data: {run_label} &nbsp;·&nbsp; Generated {timestamp} UTC</div>

<script>
const IMAGES = {images_json};
const DATES  = {dates_json};
const ICONS  = {icons_json};
const HAZARDS = {json.dumps(HAZARDS)};
const DAY_LABELS = ["TODAY","TOMORROW","DAY 3","DAY 4","DAY 5"];

let curDay = 0, curHazard = HAZARDS[0], playing = true;
let activeSlot = 'a', timer = null;

function img(id) {{ return document.getElementById('img-' + id); }}

function showFrame(animate) {{
  const src = 'data:image/png;base64,' + IMAGES[curDay][curHazard];
  const next = activeSlot === 'a' ? 'b' : 'a';

  if (!animate) {{
    img('a').style.transition = 'none';
    img('b').style.transition = 'none';
  }} else {{
    img('a').style.transition = 'opacity 0.55s ease';
    img('b').style.transition = 'opacity 0.55s ease';
  }}

  img(next).src = src;
  img(next).style.opacity = '1';
  img(activeSlot).style.opacity = '0';
  activeSlot = next;

  // Update HUD
  document.getElementById('time-date').textContent = DATES[curDay];
  document.getElementById('time-day').textContent  = DAY_LABELS[curDay];

  const pct = curDay / (DAY_LABELS.length - 1) * 100;
  document.getElementById('tl-fill').style.width = pct + '%';
  document.getElementById('tl-dot').style.left   = pct + '%';

  document.querySelectorAll('.tick').forEach((t, i) => {{
    t.classList.toggle('active', i === curDay);
  }});
}}

function setHazard(h) {{
  curHazard = h;
  document.querySelectorAll('.hbtn').forEach(b => {{
    b.classList.toggle('active', b.dataset.h === h);
  }});
  showFrame(false);
}}

function seek(i) {{
  curDay = i;
  showFrame(true);
}}

function barClick(e) {{
  const rect = document.getElementById('tl-bar').getBoundingClientRect();
  const pct  = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
  seek(Math.round(pct * (DAY_LABELS.length - 1)));
}}

function togglePlay() {{
  playing = !playing;
  document.getElementById('pbtn').innerHTML = playing ? '&#9646;&#9646;' : '&#9654;';
  if (playing) startTimer();
  else clearInterval(timer);
}}

function startTimer() {{
  clearInterval(timer);
  timer = setInterval(() => {{
    curDay = (curDay + 1) % DAY_LABELS.length;
    showFrame(true);
  }}, 1600);
}}

// Kick off
showFrame(false);
startTimer();
</script>
</body>
</html>"""


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 58)
    print("  Ausweather  [GFS live data]")
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

    print(f"\n\n[4/4] Rendering {FORECAST_DAYS * len(HAZARDS)} hazard maps...")
    images = []
    for day, risks in enumerate(all_risks):
        day_imgs = {}
        for hazard in HAZARDS:
            print(f"  Day {day+1} {hazard}...", end=" ", flush=True)
            day_imgs[hazard] = render_hazard_image(
                all_lats, all_lons, risks[hazard], polys, clip_path)
        images.append(day_imgs)
        print()

    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    with open(out, "w") as f:
        f.write(make_html(images, dates, run_label, timestamp))

    print(f"\n  Saved → {out}\n")


if __name__ == "__main__":
    main()
