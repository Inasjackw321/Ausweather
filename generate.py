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
HAZARD_ICONS = {"Wind": "💨", "Hail": "🌨", "Flood": "🌊", "Fire": "🔥", "Tornado": "🌪"}

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

# Australian regions for discussion generation
# (name, lat_min, lat_max, lon_min, lon_max)
AUS_REGIONS = [
    ("Cape York",          -17, -10, 142, 154),
    ("North QLD",          -23, -17, 138, 154),
    ("SE QLD",             -29, -23, 148, 154),
    ("Central QLD",        -26, -19, 138, 148),
    ("Northern NSW",       -32, -28, 141, 154),
    ("Southern NSW",       -37, -32, 141, 154),
    ("Victoria",           -39, -34, 141, 150),
    ("Tasmania",           -44, -39, 143, 149),
    ("South Australia",    -38, -26, 129, 141),
    ("Northern Territory", -26, -10, 129, 139),
    ("Kimberley",          -20, -13, 124, 131),
    ("Pilbara",            -24, -20, 114, 122),
    ("SW Western Australia", -36, -28, 112, 122),
    ("North Western Australia", -22, -13, 112, 124),
    ("Central Australia",  -30, -22, 120, 138),
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


# ── Discussion generation ─────────────────────────────────────────────────────
def region_risks(risk_grid, lats, lons):
    """Return list of (region_name, max_risk) sorted by risk desc."""
    results = []
    for name, lat_min, lat_max, lon_min, lon_max in AUS_REGIONS:
        lm = (lats >= lat_min) & (lats <= lat_max)
        om = (lons >= lon_min) & (lons <= lon_max)
        if lm.any() and om.any():
            mx = int(risk_grid[np.ix_(lm, om)].max())
            if mx >= 1:
                results.append((name, mx))
    return sorted(results, key=lambda x: -x[1])


def top_regions(risk_grid, lats, lons, min_risk=2, n=4):
    return [r[0] for r in region_risks(risk_grid, lats, lons) if r[1] >= min_risk][:n]


def _fmt_regions(names):
    if not names:
        return "isolated areas"
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


DISC_TEMPLATES = {
    "Wind": {
        0: "No significant wind risk forecast for this period.",
        1: "Marginal wind risk across {r}. Isolated gusts to 50–60 km/h possible.",
        2: "Slight wind risk across {r}. Gusty conditions with gusts reaching 60–75 km/h are possible, particularly in elevated areas and along exposed coastlines.",
        3: "Enhanced wind risk across {r}. Damaging gusts of 80–95 km/h are forecast, likely associated with a frontal system or strong pressure gradient. Structural damage and fallen trees are possible.",
        4: "Dangerous wind conditions forecast across {r}. Destructive gusts to 100–120 km/h likely in exposed locations. Significant property damage is possible — secure outdoor items and check with your local emergency services.",
        5: "Extreme wind event forecast across {r}. Life-threatening gusts exceeding 120 km/h anticipated. Avoid all unnecessary travel. Take shelter in a sturdy building away from windows.",
    },
    "Hail": {
        0: "No significant hail risk forecast.",
        1: "Marginal hail risk across {r}. Isolated thunderstorms with small hail are possible.",
        2: "Slight hail risk across {r}. Isolated thunderstorms capable of producing small to moderate hail are possible where instability increases.",
        3: "Enhanced hail risk across {r}. Elevated CAPE values support severe thunderstorm development. Large hail (2–4 cm) is possible with the most organised storms.",
        4: "Significant hail risk across {r}. High CAPE combined with wind shear creates an environment favourable for supercells. Large to very large hail (4–6 cm) is possible.",
        5: "Extreme hail risk across {r}. Supercell thunderstorms capable of producing giant hail (>6 cm) are possible. Protect vehicles and seek shelter indoors.",
    },
    "Flood": {
        0: "No significant flooding risk forecast.",
        1: "Marginal flooding risk across {r}. Rainfall of 10–25 mm possible with minor flooding in low-lying areas.",
        2: "Slight flooding risk across {r}. Rainfall accumulations of 25–50 mm forecast, with localised flash flooding possible in low-lying areas and smaller catchments.",
        3: "Enhanced flooding risk across {r}. Moderate to heavy rainfall of 50–100 mm expected. Flash flooding is likely in susceptible areas; rivers may rise. Do not drive through floodwater.",
        4: "Significant flooding risk across {r}. Rainfall accumulations of 100–150 mm forecast, likely causing major flash and river flooding. Evacuation of some areas may be required.",
        5: "Catastrophic flooding forecast across {r}. Extreme rainfall of 150+ mm anticipated. Life-threatening flash and river flooding likely. Follow all emergency directions immediately.",
    },
    "Fire": {
        0: "No significant fire weather risk forecast. Moist conditions and moderate temperatures prevail.",
        1: "Marginal fire weather risk across {r}. Warm, dry conditions with moderate winds keep fire danger slightly elevated. Exercise caution with ignition sources.",
        2: "Slight fire weather risk across {r}. Above-average temperatures combined with moderate winds and low humidity could support fire spread if ignition occurs.",
        3: "Enhanced fire weather risk across {r}. Elevated temperatures, dry conditions and strong winds will elevate fire danger. Avoid burning off and check local fire restrictions.",
        4: "Severe fire weather conditions forecast across {r}. Combination of very high temperatures, critically low humidity and strong winds creates a dangerous fire environment. Total Fire Bans likely.",
        5: "Catastrophic fire weather conditions across {r}. Any fires that start will be extremely difficult to control and may threaten lives and homes. Do not wait to be told to leave — have your bushfire plan ready.",
    },
    "Tornado": {
        0: "No significant tornado risk forecast.",
        1: "Marginal tornado risk across {r}. Weak tornadoes or gustnadoes cannot be ruled out with any convective activity in the region.",
        2: "Slight tornado risk across {r}. Isolated rotating thunderstorms are possible given marginal instability and wind shear. Short-lived weak tornadoes are possible.",
        3: "Enhanced tornado risk across {r}. CAPE and wind shear profiles are conducive to rotating thunderstorms. Well-organised supercells capable of tornadoes are possible.",
        4: "Significant tornado risk across {r}. Exceptionally high CAPE combined with strong directional wind shear creates a rare severe tornado environment for Australia. Long-track tornadoes are possible.",
        5: "Extreme tornado risk across {r}. This is a rare and dangerous severe weather setup. Violent, long-track tornadoes are possible. Take shelter in an interior room on the lowest floor of a sturdy building.",
    },
}


def generate_discussion(risks, lats, lons):
    """Return dict: hazard → (max_risk, top_region_names, discussion_text)"""
    out = {}
    for h in HAZARDS:
        mx = int(risks[h].max())
        regions = top_regions(risks[h], lats, lons, min_risk=max(1, mx - 1))
        r_str = _fmt_regions(regions)
        text = DISC_TEMPLATES[h][mx].format(r=r_str)
        out[h] = {"max_risk": mx, "regions": regions[:3], "text": text}
    return out


def synoptic_overview(risks, lats, lons):
    """Generate a one-paragraph synoptic context sentence."""
    def regional_max(risk_grid, lat_min, lat_max, lon_min, lon_max):
        lm = (lats >= lat_min) & (lats <= lat_max)
        om = (lons >= lon_min) & (lons <= lon_max)
        return int(risk_grid[np.ix_(lm, om)].max()) if lm.any() and om.any() else 0

    se_wind  = regional_max(risks["Wind"],  -40, -33, 140, 154)
    se_flood = regional_max(risks["Flood"], -40, -33, 140, 154)
    n_fire   = regional_max(risks["Fire"],  -25, -10, 125, 155)
    instab   = int(risks["Hail"].max())
    tor      = int(risks["Tornado"].max())

    features = []
    if se_wind >= 3 or se_flood >= 3:
        features.append("an active frontal system tracking across southeastern Australia")
    if n_fire >= 3:
        features.append("dry season fire weather conditions across the tropical north")
    if instab >= 3:
        features.append("elevated atmospheric instability supporting severe thunderstorm development")
    if tor >= 3:
        features.append("strong convective wind shear increasing the tornado threat")
    if int(risks["Flood"].max()) >= 4:
        features.append("a significant rainfall event driving flooding concerns")

    if not features:
        max_overall = max(int(risks[h].max()) for h in HAZARDS)
        if max_overall <= 1:
            return ("A relatively quiet weather pattern is forecast. No major severe weather systems are expected, though isolated hazards may persist in some regions.")
        return ("A broadly benign pattern is expected, though localised hazards remain possible across parts of the continent.")

    if len(features) == 1:
        return f"The primary weather driver for this period is {features[0]}."
    joined = "; ".join(features[:-1]) + f"; and {features[-1]}"
    return f"The forecast period is characterised by {joined}."


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
_RISK_CSS_VARS = "\n".join(
    f"  --c-{RISK[i][0].lower()}: {RISK[i][1]};"
    for i in range(6)
)

HTML_CSS = """
:root {
  --bg:      #0a1520;
  --bg2:     #0e1e2e;
  --bg3:     #142030;
  --bg4:     #1a2a3c;
  --border:  #1c3550;
  --border2: #254060;
  --text:    #b0ccde;
  --text2:   #d8eaf8;
  --dim:     #4a7090;
  --accent:  #4a9ed0;
""" + _RISK_CSS_VARS + """
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: "Courier New", Courier, monospace;
  font-size: 13px;
  line-height: 1.5;
  min-height: 100vh;
}

/* ── Header ── */
header {
  background: linear-gradient(180deg, #060e18 0%, #0c1a28 100%);
  border-bottom: 2px solid var(--border);
  padding: 12px 20px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  flex-wrap: wrap;
}
.hdr-title h1 {
  font-size: 18px;
  color: var(--text2);
  letter-spacing: 3px;
  font-weight: bold;
}
.hdr-title .subtitle {
  color: var(--dim);
  font-size: 10px;
  letter-spacing: 2px;
  margin-top: 3px;
}
.hdr-meta {
  display: flex;
  gap: 20px;
  flex-wrap: wrap;
}
.meta-item {
  text-align: right;
}
.meta-label {
  font-size: 9px;
  color: var(--dim);
  letter-spacing: 2px;
  text-transform: uppercase;
}
.meta-value {
  font-size: 11px;
  color: var(--accent);
  margin-top: 1px;
}

/* ── Tab bar ── */
.tab-bar {
  background: var(--bg2);
  border-bottom: 2px solid var(--border);
  display: flex;
  overflow-x: auto;
  scrollbar-width: thin;
}
.tab-bar::-webkit-scrollbar { height: 4px; }
.tab-bar::-webkit-scrollbar-track { background: var(--bg2); }
.tab-bar::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }
.tab {
  flex: 1;
  min-width: 110px;
  padding: 10px 14px;
  background: none;
  border: none;
  border-right: 1px solid var(--border);
  color: var(--dim);
  font-family: inherit;
  font-size: 11px;
  font-weight: bold;
  letter-spacing: 1px;
  cursor: pointer;
  transition: background 0.15s, color 0.15s;
  text-align: center;
  position: relative;
}
.tab:hover { background: var(--bg3); color: var(--text); }
.tab.active {
  background: var(--bg3);
  color: var(--text2);
  border-bottom: 3px solid var(--accent);
}
.tab .tab-day  { display: block; font-size: 11px; letter-spacing: 1px; }
.tab .tab-date { display: block; font-size: 9px; color: var(--dim); margin-top: 2px; font-weight: normal; }
.tab .tab-top  {
  display: inline-block;
  width: 8px; height: 8px;
  border-radius: 2px;
  margin-left: 5px;
  vertical-align: middle;
  position: relative;
  top: -1px;
}
.tab.active .tab-date { color: #7aaccc; }

/* ── Day panel ── */
.day-panel { display: none; }
.day-panel.active { display: block; }

/* ── Risk badge row ── */
.badge-row {
  display: flex;
  gap: 8px;
  padding: 10px 12px;
  background: var(--bg2);
  border-bottom: 1px solid var(--border);
  flex-wrap: wrap;
}
.badge {
  display: flex;
  align-items: center;
  gap: 7px;
  background: var(--bg3);
  border: 1px solid var(--border2);
  border-radius: 4px;
  padding: 5px 10px;
  min-width: 180px;
  flex: 1;
}
.badge-icon { font-size: 14px; }
.badge-name {
  font-size: 10px;
  font-weight: bold;
  letter-spacing: 1px;
  color: var(--text);
  min-width: 55px;
}
.badge-level {
  font-size: 9px;
  font-weight: bold;
  padding: 2px 6px;
  border-radius: 3px;
  letter-spacing: 1px;
  white-space: nowrap;
}
.badge-regions {
  font-size: 9px;
  color: var(--dim);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* ── Maps ── */
.maps-wrap {
  padding: 10px 12px 4px;
  background: var(--bg2);
}
.maps-wrap img {
  width: 100%;
  display: block;
  border: 1px solid var(--border);
  border-radius: 3px;
}

/* ── Discussion ── */
.discussion {
  margin: 10px 12px;
  background: var(--bg3);
  border: 1px solid var(--border2);
  border-radius: 4px;
  overflow: hidden;
}
.disc-header {
  background: var(--bg2);
  border-bottom: 1px solid var(--border2);
  padding: 8px 14px;
  font-size: 11px;
  font-weight: bold;
  color: var(--text2);
  letter-spacing: 2px;
  display: flex;
  align-items: center;
  gap: 8px;
}
.disc-synoptic {
  padding: 10px 14px;
  border-bottom: 1px solid var(--border);
  font-size: 12px;
  color: var(--text2);
  line-height: 1.6;
  font-style: italic;
}
.disc-hazards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
}
.disc-hazard {
  padding: 10px 14px;
  border-right: 1px solid var(--border);
  border-bottom: 1px solid var(--border);
}
.disc-hazard:last-child { border-right: none; }
.disc-hazard-hdr {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 6px;
}
.disc-hazard-icon { font-size: 13px; }
.disc-hazard-name {
  font-size: 10px;
  font-weight: bold;
  letter-spacing: 1px;
  color: var(--text2);
}
.disc-hazard-level {
  font-size: 9px;
  font-weight: bold;
  padding: 1px 5px;
  border-radius: 2px;
  margin-left: auto;
  letter-spacing: 1px;
}
.disc-hazard-text {
  font-size: 11px;
  color: var(--text);
  line-height: 1.55;
}
.disc-hazard-text strong { color: var(--text2); }

/* ── Legend ── */
.legend {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 12px;
  background: var(--bg2);
  border-top: 1px solid var(--border);
  flex-wrap: wrap;
}
.legend-label {
  font-size: 9px;
  color: var(--dim);
  letter-spacing: 2px;
  margin-right: 4px;
}
.legend-item {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 9px;
  letter-spacing: 1px;
}
.legend-swatch {
  width: 18px;
  height: 14px;
  border-radius: 2px;
}
.legend-item span { color: var(--dim); }

/* ── Footer ── */
footer {
  text-align: center;
  padding: 10px;
  font-size: 10px;
  color: var(--dim);
  border-top: 1px solid var(--border);
  letter-spacing: 1px;
}
footer a { color: #3a7090; text-decoration: none; }
footer a:hover { color: var(--accent); }

/* ── Responsive ── */
@media (max-width: 700px) {
  .hdr-title h1 { font-size: 14px; letter-spacing: 1px; }
  .hdr-meta { display: none; }
  .badge { min-width: 140px; }
  .disc-hazards { grid-template-columns: 1fr; }
}
"""

HTML_JS = """
function showDay(n) {
  document.querySelectorAll('.day-panel').forEach((p, i) => {
    p.classList.toggle('active', i === n);
  });
  document.querySelectorAll('.tab').forEach((t, i) => {
    t.classList.toggle('active', i === n);
  });
  history.replaceState(null, '', '#day' + n);
}
// Restore from hash
(function() {
  var m = location.hash.match(/#day(\\d)/);
  showDay(m ? parseInt(m[1]) : 0);
})();
"""


def _risk_badge_style(level):
    _, color = RISK[level]
    fg = "black" if level < 3 else "white"
    return f'style="background:{color};color:{fg}"'


def make_html(images, day_metas, dates, run_label, timestamp):
    day_labels = ["TODAY", "TOMORROW", "DAY 3", "DAY 4", "DAY 5"]
    short_dates = [
        (datetime.strptime(d, "%A, %d %b %Y")).strftime("%a %d %b")
        for d in dates
    ]

    # Build tabs
    tabs = []
    for i, (label, short) in enumerate(zip(day_labels, short_dates)):
        # Find highest risk for the dot indicator
        top_risk = max(meta["max_risk"] for k, meta in day_metas[i].items() if k != "_synoptic")
        _, dot_color = RISK[top_risk]
        tabs.append(
            f'<button class="tab" onclick="showDay({i})">'
            f'<span class="tab-day">{label}'
            f'<span class="tab-top" style="background:{dot_color}"></span></span>'
            f'<span class="tab-date">{short}</span>'
            f'</button>'
        )

    # Build day panels
    panels = []
    for i, (img, metas, date_str) in enumerate(zip(images, day_metas, dates)):
        # Badge row
        badges = []
        for h in HAZARDS:
            meta = metas[h]
            mx   = meta["max_risk"]
            lbl, color = RISK[mx]
            fg   = "black" if mx < 3 else "white"
            region_str = " · ".join(meta["regions"][:2]) if meta["regions"] else "—"
            badges.append(
                f'<div class="badge">'
                f'<span class="badge-icon">{HAZARD_ICONS[h]}</span>'
                f'<span class="badge-name">{h.upper()}</span>'
                f'<span class="badge-level" style="background:{color};color:{fg}">{lbl}</span>'
                f'<span class="badge-regions">{region_str}</span>'
                f'</div>'
            )

        # Discussion hazard cards
        hazard_cards = []
        for h in HAZARDS:
            meta = metas[h]
            mx   = meta["max_risk"]
            lbl, color = RISK[mx]
            fg   = "black" if mx < 3 else "white"
            hazard_cards.append(
                f'<div class="disc-hazard">'
                f'<div class="disc-hazard-hdr">'
                f'<span class="disc-hazard-icon">{HAZARD_ICONS[h]}</span>'
                f'<span class="disc-hazard-name">{h.upper()}</span>'
                f'<span class="disc-hazard-level" style="background:{color};color:{fg}">{lbl}</span>'
                f'</div>'
                f'<div class="disc-hazard-text">{meta["text"]}</div>'
                f'</div>'
            )

        synoptic = metas["_synoptic"]

        panels.append(
            f'<div class="day-panel" id="day{i}">'
            f'<div class="badge-row">{"".join(badges)}</div>'
            f'<div class="maps-wrap">'
            f'<img src="data:image/png;base64,{img}" alt="{date_str} outlook">'
            f'</div>'
            f'<div class="discussion">'
            f'<div class="disc-header">📋 FORECAST DISCUSSION &nbsp;·&nbsp; {date_str.upper()}</div>'
            f'<div class="disc-synoptic">{synoptic}</div>'
            f'<div class="disc-hazards">{"".join(hazard_cards)}</div>'
            f'</div>'
            f'</div>'
        )

    # Legend
    legend_items = "".join(
        f'<div class="legend-item">'
        f'<div class="legend-swatch" style="background:{RISK[i][1]}"></div>'
        f'<span>{RISK[i][0]}</span>'
        f'</div>'
        for i in range(6)
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Australia Severe Weather Outlook</title>
<style>{HTML_CSS}</style>
</head>
<body>

<header>
  <div class="hdr-title">
    <h1>&#9928; AUSTRALIA SEVERE WEATHER OUTLOOK</h1>
    <div class="subtitle">5-DAY RISK FORECAST &nbsp;&bull;&nbsp; WIND &nbsp;&bull;&nbsp; HAIL &nbsp;&bull;&nbsp; FLOOD &nbsp;&bull;&nbsp; FIRE &nbsp;&bull;&nbsp; TORNADO</div>
  </div>
  <div class="hdr-meta">
    <div class="meta-item">
      <div class="meta-label">DATA SOURCE</div>
      <div class="meta-value">{run_label}</div>
    </div>
    <div class="meta-item">
      <div class="meta-label">GENERATED</div>
      <div class="meta-value">{timestamp} UTC</div>
    </div>
  </div>
</header>

<div class="tab-bar">{''.join(tabs)}</div>

{''.join(panels)}

<div class="legend">
  <span class="legend-label">RISK SCALE</span>
  {legend_items}
</div>

<footer>
  NOT FOR OPERATIONAL USE &nbsp;&bull;&nbsp;
  For official warnings visit <a href="https://www.bom.gov.au" target="_blank">bom.gov.au</a>
  &nbsp;&bull;&nbsp; Data: NOAA GFS via AWS Open Data
</footer>

<script>{HTML_JS}</script>
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
    all_risks, all_metas, dates = [], [], []
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

        metas = generate_discussion(risks, lats, lons)
        metas["_synoptic"] = synoptic_overview(risks, lats, lons)
        all_metas.append(metas)

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
        f.write(make_html(images, all_metas, dates, run_label, timestamp))

    print(f"\n  Saved → {out}\n")


if __name__ == "__main__":
    main()
