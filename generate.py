#!/usr/bin/env python3
"""
Australian SPC-style 5-Day Severe Weather Outlook Generator
Produces: Wind / Hail / Flood / Fire / Tornado risk maps

Data model: realistic synthetic climatology for the current date,
seeded by date so output is deterministic per day.
"""

import sys, os, json, base64
from io import BytesIO
from datetime import datetime, timedelta

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.patches import Polygon, PathPatch
from matplotlib.collections import PatchCollection
from matplotlib.path import Path

try:
    import requests as _req
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False


# ── Locations ─────────────────────────────────────────────────────────────────
LOCATIONS = [
    (-33.87, 151.21, "Sydney"),
    (-37.81, 144.96, "Melbourne"),
    (-27.47, 153.02, "Brisbane"),
    (-31.95, 115.86, "Perth"),
    (-34.93, 138.60, "Adelaide"),
    (-12.46, 130.84, "Darwin"),
    (-42.88, 147.33, "Hobart"),
    (-35.28, 149.13, "Canberra"),
    (-16.92, 145.77, "Cairns"),
    (-19.26, 146.82, "Townsville"),
    (-23.70, 133.88, "Alice Springs"),
    (-20.73, 139.49, "Mt Isa"),
    (-20.66, 116.85, "Port Hedland"),
    (-25.29, 152.84, "Hervey Bay"),
    (-32.93, 151.77, "Newcastle"),
    (-38.14, 144.36, "Geelong"),
    (-34.18, 142.16, "Mildura"),
    (-32.29, 148.60, "Dubbo"),
    (-17.96, 122.23, "Broome"),
    (-31.49, 145.70, "Cobar"),
    (-30.74, 136.46, "Woomera"),
    (-17.37, 136.78, "Tennant Creek"),
    (-14.47, 132.26, "Katherine"),
    (-33.86, 121.89, "Esperance"),
    (-28.77, 114.62, "Geraldton"),
    (-26.18, 128.30, "Warburton"),
    (-29.00, 153.40, "Lismore"),
    (-18.25, 127.67, "Halls Creek"),
    (-25.30, 152.35, "Toowoomba"),
    (-36.12, 146.89, "Albury"),
    (-21.15, 149.16, "Mackay"),
    (-23.38, 150.51, "Rockhampton"),
    (-26.40, 153.10, "Sunshine Coast"),
    (-37.84, 147.63, "Sale"),
    (-24.88, 152.35, "Bundaberg"),
]

HAZARDS = ["Wind", "Hail", "Flood", "Fire", "Tornado"]

RISK = {
    0: ("NONE",  "#b0b8c0"),
    1: ("MRGL",  "#4caf50"),
    2: ("SLGT",  "#ffeb3b"),
    3: ("ENH",   "#ff9800"),
    4: ("MDT",   "#f44336"),
    5: ("HIGH",  "#9c27b0"),
}

AUS_BOUNDS = (112.0, 154.5, -44.5, -9.5)

GEOJSON_URLS = [
    "https://raw.githubusercontent.com/tonywr71/GeoJson-Data/master/australian-states.min.geojson",
    "https://raw.githubusercontent.com/rowanhogan/australian-states/master/states.min.geojson",
]


# ── Map data ──────────────────────────────────────────────────────────────────
def fetch_geojson():
    if not _HAS_REQUESTS:
        return None
    for url in GEOJSON_URLS:
        try:
            r = _req.get(url, timeout=25)
            if r.ok:
                data = r.json()
                features = data.get("features", [])
                if features and "iso_a3" in (features[0].get("properties") or {}):
                    aus = [f for f in features
                           if (f.get("properties") or {}).get("iso_a3") == "AUS"]
                    return {"type": "FeatureCollection", "features": aus} if aus else None
                return data
        except Exception:
            pass
    return None


def geojson_to_polygons(geojson):
    polys = []
    for feat in geojson.get("features", []):
        geom  = feat.get("geometry", {})
        gtype = geom.get("type", "")
        coords = geom.get("coordinates", [])
        rings = []
        if gtype == "Polygon":
            rings = [coords[0]]
        elif gtype == "MultiPolygon":
            rings = [poly[0] for poly in coords]
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


# ── Synthetic weather data ────────────────────────────────────────────────────
#
# Climatological model for each location:
#   - "base" values for the current month
#   - synoptic perturbations that evolve day-to-day
#
# June = mid-winter in SE Australia, dry season in the north.
#
# Climate profiles: (wind_base, cape_base, precip_base, temp_base, fire_base)
# Profiles encode typical June conditions for each region.

def _month_climate(lat, lon, month):
    """
    Return (wind_kph, cape_jkg, precip_mm, temp_c, fire_idx) climatological
    baseline for the given location and calendar month.
    """
    is_tropical = lat > -22                        # NT/N QLD/N WA
    is_winter   = month in (5, 6, 7, 8)            # May-Aug

    # ── Tropical north ──────────────────────────────────────────────
    if is_tropical:
        if is_winter:   # dry season
            wind    = 18
            cape    = 300
            precip  = 3
            temp    = 28
            fire    = 50        # grass fire season ramps up
        else:           # wet season
            wind    = 22
            cape    = 1800
            precip  = 60
            temp    = 32
            fire    = 5

    # ── Southern Australia (below -22) ──────────────────────────────
    else:
        # West coast (Perth / Geraldton / Broome)
        if lon < 125:
            if is_winter:
                wind   = 30
                cape   = 200
                precip = 20
                temp   = 15
                fire   = 5
            else:
                wind   = 22
                cape   = 800
                precip = 5
                temp   = 28
                fire   = 35

        # Central (Alice / Woomera / Cobar / Warburton)
        elif lon < 140 and lat > -35:
            if is_winter:
                wind   = 22
                cape   = 100
                precip = 4
                temp   = 16
                fire   = 20
            else:
                wind   = 20
                cape   = 600
                precip = 8
                temp   = 32
                fire   = 50

        # SE Australia (Vic / NSW / SA / Tas)
        else:
            if is_winter:
                wind   = 40      # frequent cold fronts
                cape   = 300
                precip = 25
                temp   = 11
                fire   = 3
            else:
                wind   = 28
                cape   = 1200
                precip = 12
                temp   = 26
                fire   = 55

    return wind, cape, precip, temp, fire


def synthetic_risks(day_offset, seed_date):
    """
    Generate risk dicts for all locations for a given day offset.
    seed_date: datetime used as base seed so output is repeatable per run.
    """
    day_seed = int(seed_date.strftime("%Y%m%d")) + day_offset
    rng = np.random.default_rng(day_seed)

    # Synoptic patterns: simulate 1-3 systems over Australia
    n_systems = rng.integers(1, 4)
    systems = []
    for _ in range(n_systems):
        stype = rng.choice(["front", "low", "high", "trough"])
        # random centre position biased toward realistic tracks
        clon = rng.uniform(115, 155)
        clat = rng.uniform(-40, -15)
        strength = rng.uniform(0.4, 1.0)
        radius   = rng.uniform(5, 18)
        systems.append((stype, clon, clat, strength, radius))

    results = []
    for lat, lon, name in LOCATIONS:
        month  = (seed_date + timedelta(days=day_offset)).month
        w_base, cape_base, precip_base, temp_base, fire_base = _month_climate(lat, lon, month)

        # Accumulate system influence at this location
        wind_boost  = 0.0
        cape_boost  = 0.0
        precip_mult = 1.0
        fire_mult   = 1.0

        for stype, clon, clat, strength, radius in systems:
            dist = np.sqrt((lon - clon)**2 + (lat - clat)**2)
            if dist >= radius:
                continue
            influence = strength * (1 - dist / radius)

            if stype == "front":
                wind_boost  += influence * 45
                precip_mult += influence * 3.0
                cape_boost  += influence * 400
                fire_mult   *= max(0.1, 1 - influence * 0.8)
            elif stype == "low":
                wind_boost  += influence * 35
                precip_mult += influence * 4.0
                cape_boost  += influence * 600
                fire_mult   *= max(0.1, 1 - influence * 0.9)
            elif stype == "high":
                wind_boost  += influence * 10
                fire_mult   += influence * 0.5
            elif stype == "trough":
                cape_boost  += influence * 800
                precip_mult += influence * 2.0
                wind_boost  += influence * 20

        # Small local noise
        local = rng.standard_normal(5) * 0.15

        wind   = max(0, w_base     + wind_boost  + local[0] * 15)
        gusts  = wind * rng.uniform(1.3, 1.7)
        cape   = max(0, cape_base  + cape_boost  + local[1] * 200)
        precip = max(0, precip_base * precip_mult + abs(local[2]) * 5)
        prob   = min(100, max(0, (precip / (precip + 5)) * 100 + local[3] * 10))
        temp   = temp_base + local[4] * 2
        fwi    = fire_base * fire_mult

        # ── Risk levels ──────────────────────────────────────────────
        def wind_risk(g, w):
            if g >= 120 or w >= 90: return 5
            if g >= 95  or w >= 72: return 4
            if g >= 75  or w >= 58: return 3
            if g >= 60  or w >= 46: return 2
            if g >= 44  or w >= 35: return 1
            return 0

        def hail_risk(c, p):
            if c >= 2800 and p >= 60: return 5
            if c >= 2000 and p >= 50: return 4
            if c >= 1300 and p >= 40: return 3
            if c >= 700  and p >= 30: return 2
            if c >= 300  and p >= 20: return 1
            return 0

        def flood_risk(prec, p):
            if prec >= 150:           return 5
            if prec >= 100:           return 4
            if prec >= 60:            return 3
            if prec >= 30:            return 2
            if prec >= 15 and p >= 50: return 1
            return 0

        def fire_risk(fi, prec):
            if prec >= 3: return 0
            if fi >= 80: return 5
            if fi >= 58: return 4
            if fi >= 38: return 3
            if fi >= 20: return 2
            if fi >= 8:  return 1
            return 0

        def tornado_risk(c, g, p):
            # simplified: requires instability + shear
            li = -max(0, (c - 500) / 500)   # proxy lifted index
            if c >= 3000 and li <= -5 and g >= 80:             return 4
            if c >= 2000 and li <= -4 and g >= 60 and p >= 50: return 3
            if c >= 1200 and li <= -3 and p >= 40:             return 2
            if c >= 600  and p >= 30:                          return 1
            return 0

        results.append({
            "lat": lat, "lon": lon, "name": name,
            "Wind":    wind_risk(gusts, wind),
            "Hail":    hail_risk(cape, prob),
            "Flood":   flood_risk(precip, prob),
            "Fire":    fire_risk(fwi, precip),
            "Tornado": tornado_risk(cape, gusts, prob),
        })

    return results


# ── Map rendering ─────────────────────────────────────────────────────────────
CMAP = mcolors.ListedColormap([RISK[i][1] for i in range(6)])
NORM = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5], CMAP.N)
LON0, LON1, LAT0, LAT1 = AUS_BOUNDS


def draw_panel(ax, points, hazard, polys, clip_path):
    ax.set_xlim(LON0, LON1)
    ax.set_ylim(LAT0, LAT1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("#1a2e40")     # ocean

    # Land base
    if polys:
        col = PatchCollection(
            [Polygon(p, closed=True) for p in polys],
            facecolor="#2c3e50", edgecolor="none", zorder=1,
        )
        ax.add_collection(col)

    lats = np.array([p["lat"] for p in points])
    lons = np.array([p["lon"] for p in points])
    vals = np.array([p[hazard] for p in points], dtype=float)

    # Interpolated surface
    if len(points) >= 5:
        try:
            from scipy.interpolate import griddata
            glon = np.linspace(LON0 + 0.5, LON1 - 0.5, 140)
            glat = np.linspace(LAT0 + 0.5, LAT1 - 0.5, 95)
            GL, GLatG = np.meshgrid(glon, glat)
            gv = griddata((lons, lats), vals, (GL, GLatG), method="linear", fill_value=0)
            gv = np.clip(np.round(gv), 0, 5)
            cf = ax.contourf(GL, GLatG, gv,
                             levels=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
                             cmap=CMAP, norm=NORM, alpha=0.82, zorder=2)
            if clip_path is not None:
                cp = PathPatch(clip_path, transform=ax.transData, visible=False)
                ax.add_patch(cp)
                for coll in cf.collections:
                    coll.set_clip_path(cp)
        except Exception:
            for p in points:
                rv = p[hazard]
                if rv > 0:
                    ax.scatter(p["lon"], p["lat"], c=RISK[rv][1],
                               s=700, alpha=0.65, zorder=3, edgecolors="none")

    # State borders
    if polys:
        ax.add_collection(PatchCollection(
            [Polygon(p, closed=True) for p in polys],
            facecolor="none", edgecolor="#5a8faa", linewidth=0.5, zorder=4,
        ))

    ax.set_title(hazard.upper(), fontsize=9, fontweight="bold",
                 color="#cce8ff", pad=3, fontfamily="monospace")


def render_day(day_points, date_label, polys, clip_path):
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
        draw_panel(ax, day_points, hazard, polys, clip_path)

    # Risk legend strip
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
def make_html(images, dates):
    day_labels = ["TODAY", "TOMORROW", "DAY 3", "DAY 4", "DAY 5"]
    blocks = []
    for label, date, img in zip(day_labels, dates, images):
        blocks.append(f"""\
  <div class="day">
    <div class="day-hdr">{label} &mdash; {date}</div>
    <img src="data:image/png;base64,{img}" alt="{date} outlook">
  </div>""")

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
  <div class="sub">GENERATED {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")} &nbsp;&bull;&nbsp; WIND / HAIL / FLOOD / FIRE / TORNADO</div>
</header>
<div class="days">
{chr(10).join(blocks)}
</div>
<footer>NOT FOR OPERATIONAL USE &nbsp;&bull;&nbsp; For official warnings visit <a href="https://www.bom.gov.au">bom.gov.au</a></footer>
</body>
</html>"""


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 52)
    print("  Australia Severe Weather Outlook Generator")
    print("=" * 52)

    print("\n[1/3] Downloading Australia map data...")
    geojson   = fetch_geojson()
    if geojson:
        polys     = geojson_to_polygons(geojson)
        clip_path = polys_to_clip_path(polys)
        print(f"      OK — {len(polys)} polygons")
    else:
        polys, clip_path = [], None
        print("      Network unavailable — dots-only mode")

    print("\n[2/3] Building 5-day outlooks (synthetic climatology)...")
    seed_date = datetime.now()
    days_points, dates = [], []
    for d in range(5):
        date = seed_date + timedelta(days=d)
        date_str = date.strftime("%A, %d %b %Y")
        dates.append(date_str)
        days_points.append(synthetic_risks(d, seed_date))
        print(f"      Day {d+1}: {date_str}")

    print("\n[3/3] Rendering graphics...")
    images = []
    for d in range(5):
        print(f"      Day {d+1} ...")
        images.append(render_day(days_points[d], dates[d], polys, clip_path))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    with open(out, "w") as f:
        f.write(make_html(images, dates))

    print(f"\n  Saved → {out}")
    print("  Open index.html in your browser.\n")


if __name__ == "__main__":
    main()
