#!/usr/bin/env python3
"""
Ausweather — Australian Severe Weather Outlook
Real data: NOAA GFS 0.25° via AWS Open Data (hourly out to +24h, 3-hourly to +120h)
Interactive Leaflet map with live RainViewer radar + per-hour GFS risk overlays.
"""

import sys, os, math, base64, json
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
AUS_BOUNDS = (112.0, 154.5, -44.5, -9.5)          # lon0, lon1, lat0(S), lat1(N)
LON0, LON1, LAT0, LAT1 = AUS_BOUNDS

# Forecast frames: every hour to +24h, then every 3h to +120h
FRAME_HOURS = list(range(1, 25)) + list(range(27, 121, 3))
# Optional subset for quick local testing:  AUSWEATHER_MAXFRAMES=4 python3 generate.py
_MAXF = int(os.environ.get("AUSWEATHER_MAXFRAMES", "0"))
if _MAXF:
    FRAME_HOURS = FRAME_HOURS[:_MAXF]

HAZARDS = ["Wind", "Hail", "Flood", "Tornado"]
# "Max" = overall threat (highest of the four); rendered as an extra layer.
DATA_HAZARDS = HAZARDS + ["Max"]
HAZARD_ICONS = {"Wind": "💨", "Hail": "🧊", "Flood": "🌊", "Tornado": "🌪",
                "Max": "⚠️", "Radar": "📡"}

RISK_LABELS = ["NONE", "MRGL", "SLGT", "ENH", "MDT", "HIGH"]

RISK_COLORS = ["#2a3a48","#0090c0","#00d0d0","#ffe23b","#ff6a1f","#c026d3"]

CITIES = [
    ("Sydney",     -33.87, 151.21),
    ("Melbourne",  -37.81, 144.96),
    ("Brisbane",   -27.47, 153.02),
    ("Perth",      -31.95, 115.86),
    ("Adelaide",   -34.93, 138.60),
    ("Darwin",     -12.46, 130.84),
    ("Hobart",     -42.88, 147.33),
    ("Canberra",   -35.28, 149.13),
    ("Cairns",     -16.92, 145.77),
    ("Gold Coast", -28.00, 153.43),
]

STATE_LABELS = [
    (146.5, -32.0, "NSW"), (144.5, -36.8, "VIC"), (144.0, -22.0, "QLD"),
    (135.5, -30.0, "SA"),  (121.0, -27.0, "WA"),  (133.5, -19.5, "NT"),
    (146.5, -42.0, "TAS"),
]

# Heatmap ramp: transparent-ish dark → cyan → green → yellow → orange → purple
HEATMAP_CMAP = mcolors.LinearSegmentedColormap.from_list("ausweather", [
    (0.00, "#0a3550"), (0.18, "#0090c0"), (0.36, "#00d0d0"),
    (0.52, "#43c25a"), (0.66, "#ffe23b"), (0.80, "#ff6a1f"),
    (1.00, "#c026d3"),
], N=256)


# ── Mercator projection (match Leaflet EPSG:3857 so overlays align) ─────────────
def mercY(lat):
    lat = np.clip(lat, -85.0, 85.0)
    return np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))

MY0, MY1 = float(mercY(np.array([LAT0]))[0]), float(mercY(np.array([LAT1]))[0])


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
            if requests.head(gfs_url(ds, rs, max(FRAME_HOURS), ".idx"),
                             timeout=5).status_code == 200:
                return ds, rs, dt
        except Exception:
            pass
    return None, None, None


def parse_idx(text):
    records, lines = [], [l for l in text.strip().split("\n") if l]
    for i, line in enumerate(lines):
        parts = line.split(":")
        if len(parts) < 6:
            continue
        records.append({
            "start": int(parts[1]),
            "end":   int(lines[i + 1].split(":")[1]) if i + 1 < len(lines) else None,
            "var":   parts[3], "level": parts[4], "time": parts[5],
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


def find_precip_bucket(records):
    """
    Return (record, duration_hours) for the *6-hour-resetting* APCP bucket
    (e.g. '18-24 hour acc'), i.e. the most recent localized accumulation —
    chosen as the APCP/surface record with the largest start hour.
    """
    best, best_start, best_dur = None, -1, 1
    for rec in records:
        if rec["var"] != "APCP" or rec["level"] != "surface":
            continue
        t = rec["time"]
        if "day acc" in t:            # skip the 0-N day continuous total
            continue
        # parse "A-B hour acc fcst"
        try:
            span = t.split("hour")[0].strip()
            a, b = span.split("-")
            a, b = int(a), int(b)
        except Exception:
            continue
        if a > best_start:
            best, best_start, best_dur = rec, a, max(1, b - a)
    return best, best_dur


# ── GRIB2 download & parse ─────────────────────────────────────────────────────
def download_range(url, start, end):
    end_s = str(end - 1) if end else ""
    r = requests.get(url, headers={"Range": f"bytes={start}-{end_s}"}, timeout=60)
    r.raise_for_status()
    return r.content


def grib_to_aus(grib_bytes):
    gid = eccodes.codes_new_from_message(grib_bytes)
    try:
        ni  = eccodes.codes_get(gid, "Ni"); nj = eccodes.codes_get(gid, "Nj")
        la1 = eccodes.codes_get(gid, "latitudeOfFirstGridPointInDegrees")
        la2 = eccodes.codes_get(gid, "latitudeOfLastGridPointInDegrees")
        lo1 = eccodes.codes_get(gid, "longitudeOfFirstGridPointInDegrees")
        lo2 = eccodes.codes_get(gid, "longitudeOfLastGridPointInDegrees")
        vals = eccodes.codes_get_values(gid)
    finally:
        eccodes.codes_release(gid)
    lats = np.linspace(la1, la2, nj)
    lons = np.linspace(lo1, lo2, ni)
    data = vals.reshape(nj, ni)
    lm = (lats >= LAT0) & (lats <= LAT1)
    om = (lons >= LON0) & (lons <= LON1)
    return lats[lm], lons[om], data[np.ix_(lm, om)]


# instantaneous variables (TMP dropped — was only used for the removed Fire hazard)
VAR_SPECS = [
    ("ugrd", "UGRD", "10 m above ground", None,  None),
    ("vgrd", "VGRD", "10 m above ground", None,  None),
    ("gust", "GUST", "surface",           "PV=", None),
    ("cape", "CAPE", "surface",           None,  None),
    ("lftx", "LFTX", "surface",           None,  None),
]


def fetch_frame(date_s, run_s, fhour):
    idx_text = requests.get(gfs_url(date_s, run_s, fhour, ".idx"), timeout=15).text
    records  = parse_idx(idx_text)
    url = gfs_url(date_s, run_s, fhour)
    lats = lons = None
    fields = {}
    for key, var, lev_sub, lev_exc, time_sub in VAR_SPECS:
        rec = find_record(records, var, level_substr=lev_sub,
                          time_substr=time_sub, level_exclude=lev_exc)
        if rec is None:
            fields[key] = None
            continue
        try:
            l, o, g = grib_to_aus(download_range(url, rec["start"], rec["end"]))
            if lats is None:
                lats, lons = l, o
            fields[key] = g
            print(".", end="", flush=True)
        except Exception as e:
            print(f"![{key}:{e}]", end="", flush=True); fields[key] = None
    # precip bucket
    prec_rec, dur = find_precip_bucket(records)
    if prec_rec is not None:
        try:
            l, o, g = grib_to_aus(download_range(url, prec_rec["start"], prec_rec["end"]))
            if lats is None:
                lats, lons = l, o
            fields["apcp"] = g; fields["apcp_dur"] = dur
            print("·", end="", flush=True)
        except Exception as e:
            print(f"![apcp:{e}]", end="", flush=True); fields["apcp"] = None
    else:
        fields["apcp"] = None
    return lats, lons, fields


# ── Risk calculations (per-frame, instantaneous + recent precip rate) ───────────
def compute_risks(fields):
    def f(key):
        v = fields.get(key)
        ref = next((x for x in fields.values()
                    if isinstance(x, np.ndarray)), np.zeros((141, 171)))
        return v if isinstance(v, np.ndarray) else np.zeros(ref.shape)

    ugrd, vgrd = f("ugrd"), f("vgrd")
    gust = f("gust"); cape = f("cape"); lftx = f("lftx")
    apcp = f("apcp"); dur = max(1, fields.get("apcp_dur", 1) or 1)

    wind_kph = np.sqrt(ugrd**2 + vgrd**2) * 3.6
    gust_kph = gust * 3.6
    rate = apcp / dur                                  # mm/hr over the recent bucket
    precip_prob = np.clip(rate * 12.0, 0, 95)          # convective coverage proxy

    # Wind — instantaneous sustained + gusts
    wr = np.zeros(wind_kph.shape, int)
    for t, l in [(35,1),(46,2),(58,3),(72,4),(90,5)]:
        wr[wind_kph >= t] = l
    gr = np.zeros_like(wr)
    for t, l in [(44,1),(60,2),(75,3),(95,4),(120,5)]:
        gr[gust_kph >= t] = l
    wind_risk = np.maximum(wr, gr)

    # Hail — instability gated by active convective precip
    hail_risk = np.zeros(cape.shape, int)
    for (c, p), l in [((300,20),1),((700,30),2),((1300,40),3),((2000,50),4),((2800,60),5)]:
        hail_risk[(cape >= c) & (precip_prob >= p)] = l

    # Flood — max of instantaneous rate risk and bucket-accumulation risk
    flood_rate = np.zeros(rate.shape, int)
    for t, l in [(1,1),(3,2),(7,3),(15,4),(30,5)]:
        flood_rate[rate >= t] = l
    flood_acc = np.zeros(apcp.shape, int)
    for t, l in [(5,1),(15,2),(30,3),(60,4),(100,5)]:
        flood_acc[apcp >= t] = l
    flood_risk = np.maximum(flood_rate, flood_acc)

    # Tornado — high CAPE + instability + shear proxy (gusts) + precip
    tor_risk = np.zeros(cape.shape, int)
    for (c, li, g, p), l in [((600,-2,0,0),1),((1200,-3,0,30),2),
                              ((2000,-4,55,40),3),((3000,-5,75,0),4)]:
        m = (cape >= c) & (lftx <= li)
        if g: m &= (gust_kph >= g)
        if p: m &= (precip_prob >= p)
        tor_risk[m] = l

    out = {"Wind": wind_risk, "Hail": hail_risk,
           "Flood": flood_risk, "Tornado": tor_risk}
    out["Max"] = np.maximum.reduce([out[h] for h in HAZARDS])
    return out


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
        geom = feat.get("geometry", {}); gt = geom.get("type", "")
        coords = geom.get("coordinates", [])
        rings = ([coords[0]] if gt == "Polygon"
                 else [p[0] for p in coords] if gt == "MultiPolygon" else [])
        for ring in rings:
            a = np.array(ring)
            if a.ndim == 2 and a.shape[1] >= 2 and len(a) >= 3:
                polys.append(a[:, :2])
    return polys


def project_poly(p):
    """lon/lat polygon → lon/mercatorY for drawing & clipping."""
    out = p.copy().astype(float)
    out[:, 1] = mercY(p[:, 1])
    return out


def polys_to_clip_path(polys_proj):
    verts, codes = [], []
    for p in polys_proj:
        verts.extend(p.tolist()); verts.append(p[0].tolist())
        codes += [Path.MOVETO] + [Path.LINETO] * (len(p) - 1) + [Path.CLOSEPOLY]
    return Path(verts, codes) if verts else None


def city_risks_for_frame(risks, lats, lons):
    """Return {city_name: {hazard: max_risk_int}} for each city."""
    result = {}
    for name, clat, clon in CITIES:
        ili = int(np.argmin(np.abs(lats - clat)))
        ilo = int(np.argmin(np.abs(lons - clon)))
        r1, r2 = max(0, ili-1), min(len(lats), ili+2)
        c1, c2 = max(0, ilo-1), min(len(lons), ilo+2)
        result[name] = {h: int(risks[h][r1:r2, c1:c2].max()) for h in DATA_HAZARDS}
    return result


def coarse_risks(risks, lats, lons, step=4):
    """Sub-sampled risk grid for click-to-inspect (small enough to embed in JS)."""
    clats = lats[::step].tolist()
    clons = lons[::step].tolist()
    data  = {h: base64.b64encode(
                 risks[h][::step, ::step].astype(np.uint8).flatten().tobytes()
             ).decode() for h in DATA_HAZARDS}
    return clats, clons, data


# ── Heatmap rendering (transparent, Mercator, clipped to coastline) ─────────────
_FIG_W = 6.4
# Mercator X is linear in longitude (radians); keep figure proportional in
# consistent Web-Mercator units so blobs aren't pre-distorted before Leaflet
# stretches the overlay onto the map bounds.
_FIG_H = _FIG_W * (MY1 - MY0) / math.radians(LON1 - LON0)

def render_overlay(lats, lons, risk_grid, polys_proj, clip_path):
    fig = plt.figure(figsize=(_FIG_W, _FIG_H))
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.set_xlim(LON0, LON1); ax.set_ylim(MY0, MY1)
    ax.patch.set_alpha(0.0)

    if risk_grid.max() >= 1:
        Y = mercY(lats)
        GX, GY = np.meshgrid(lons, Y)
        norm = mcolors.Normalize(vmin=0, vmax=5)
        lv = np.linspace(0.18, 5, 56)

        glow = gaussian_filter(risk_grid.astype(float), sigma=3.4)
        cf1 = ax.contourf(GX, GY, glow, levels=lv, cmap=HEATMAP_CMAP,
                          norm=norm, alpha=0.40, extend="max")
        core = gaussian_filter(risk_grid.astype(float), sigma=1.6)
        cf2 = ax.contourf(GX, GY, core, levels=lv, cmap=HEATMAP_CMAP,
                          norm=norm, alpha=0.88, extend="max")
        if clip_path is not None:
            for cf in (cf1, cf2):
                cp = PathPatch(clip_path, transform=ax.transData, visible=False)
                ax.add_patch(cp); cf.set_clip_path(cp)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=78, transparent=True,
                bbox_inches=None, pad_inches=0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# ── HTML (placeholder substitution to avoid brace escaping) ─────────────────────
def make_html(frames, run_label, timestamp, coarse_lats, coarse_lons):
    """
    frames: list of dicts: { f, utc, local, day, images:{hazard:b64}, city_risks, coarse }
    """
    images = [fr["images"] for fr in frames]
    meta   = [{"f": fr["f"], "utc": fr["utc"], "local": fr["local"], "day": fr["day"]}
              for fr in frames]

    legend_colors = ["#2a3a48"] + [mcolors.to_hex(HEATMAP_CMAP(i / 5.0)) for i in range(1, 6)]
    legend = "".join(
        f'<div class="lg-item"><span class="lg-sw" style="background:{legend_colors[i]}"></span>{RISK_LABELS[i]}</div>'
        for i in range(6)
    )

    profile = {h: [fr["peak"][h] for fr in frames] for h in DATA_HAZARDS}

    tpl = _HTML_TEMPLATE
    repl = {
        "__IMAGES__":       json.dumps(images, separators=(",", ":")),
        "__META__":         json.dumps(meta, separators=(",", ":")),
        "__HAZARDS__":      json.dumps(DATA_HAZARDS),
        "__ICONS__":        json.dumps(HAZARD_ICONS),
        "__PROFILE__":      json.dumps(profile, separators=(",", ":")),
        "__BOUNDS__":       json.dumps([[LAT0, LON0], [LAT1, LON1]]),
        "__LEGEND__":       legend,
        "__RUNLABEL__":     run_label,
        "__TIMESTAMP__":    timestamp,
        "__NFRAMES__":      str(len(frames)),
        "__CITIES__":       json.dumps([[n, la, lo] for n, la, lo in CITIES]),
        "__RISK_COLORS__":  json.dumps(RISK_COLORS),
        "__CITY_RISKS__":   json.dumps([fr["city_risks"] for fr in frames], separators=(",",":")),
        "__COARSE__":       json.dumps([fr["coarse"]     for fr in frames], separators=(",",":")),
        "__COARSE_LATS__":  json.dumps(coarse_lats),
        "__COARSE_LONS__":  json.dumps(coarse_lons),
        "__COARSE_SHAPE__": json.dumps([len(coarse_lats), len(coarse_lons)]),
    }
    for k, v in repl.items():
        tpl = tpl.replace(k, v)
    return tpl


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>Ausweather</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;overflow:hidden;background:#050a10;
  font-family:-apple-system,"Segoe UI",Roboto,Arial,sans-serif;color:#eaf6ff;-webkit-tap-highlight-color:transparent}
#map{position:fixed;inset:0;background:#050a10;z-index:1}
.leaflet-container{background:#050a10}
.risk-overlay{image-rendering:auto;transition:opacity .4s ease}

/* ── Top bar ── */
.topbar{position:fixed;top:0;left:0;right:0;z-index:500;display:flex;align-items:center;
  gap:12px;padding:10px 14px;pointer-events:none;
  background:linear-gradient(180deg,rgba(5,10,16,.94) 60%,rgba(5,10,16,0))}
.brand{font-size:20px;font-weight:800;letter-spacing:2px;text-transform:uppercase;
  text-shadow:0 2px 10px rgba(0,0,0,.6);white-space:nowrap}
.brand em{color:#36c5e0;font-style:normal}
.brand small{display:block;font-size:8px;letter-spacing:3px;color:#5a8faa;font-weight:600;margin-top:1px}
.legbox{margin-left:auto;pointer-events:auto;background:rgba(8,16,24,.80);border:1px solid #15324a;
  border-radius:10px;padding:7px 10px;backdrop-filter:blur(8px);flex-shrink:0}
.legbox .lg-title{font-size:8px;letter-spacing:2px;color:#5a8faa;margin-bottom:5px}
.lg-scale{display:flex;gap:0}
.lg-item{font-size:8px;letter-spacing:.4px;color:#9fc4d8;text-align:center;width:36px}
.lg-sw{display:block;height:7px;border-radius:2px;margin-bottom:3px}
.radar-legend{display:none}
.radar-legend.show{display:block}
.radar-grad{height:7px;width:190px;border-radius:2px;margin-bottom:3px;
  background:linear-gradient(90deg,#3aa0ff,#19d36b,#e9e337,#ff8a1f,#ff2d2d,#d23bd2)}
.radar-grad-labels{display:flex;justify-content:space-between;font-size:8px;color:#9fc4d8;letter-spacing:.4px}

/* ── Alert banner ── */
.alert{position:fixed;top:54px;left:50%;transform:translateX(-50%);z-index:480;
  display:none;align-items:center;gap:8px;font-size:11px;font-weight:700;letter-spacing:1px;
  padding:6px 16px;border-radius:20px;border:1px solid;backdrop-filter:blur(8px);white-space:nowrap;
  max-width:92vw;overflow:hidden;text-overflow:ellipsis}
.alert.show{display:flex}
.alert-3{background:rgba(255,226,59,.12);border-color:rgba(255,226,59,.4);color:#ffe23b}
.alert-4{background:rgba(255,106,31,.15);border-color:rgba(255,106,31,.5);color:#ff9040}
.alert-5{background:rgba(192,38,211,.18);border-color:rgba(192,38,211,.55);color:#d060e0}
.alert .dot{width:6px;height:6px;border-radius:50%;background:currentColor;animation:blink 1.2s infinite;flex:0 0 auto}
@keyframes blink{0%,100%{opacity:.3}50%{opacity:1}}

/* ── Left icon buttons ── */
.icon-btn{pointer-events:auto;width:36px;height:36px;border-radius:10px;border:1px solid #15324a;
  background:rgba(8,16,24,.80);color:#9fc4d8;font-size:15px;cursor:pointer;display:flex;
  align-items:center;justify-content:center;backdrop-filter:blur(8px);transition:.15s;user-select:none}
.icon-btn:hover{border-color:#36c5e0;color:#cdeefb}
.icon-btn.on{border-color:#36c5e0;color:#36c5e0;background:rgba(0,40,60,.65)}
.left-stack{position:fixed;top:60px;left:14px;z-index:500;display:flex;flex-direction:column;gap:8px}
.zoom-pair{display:flex;flex-direction:column}
.zoom-pair .icon-btn:first-child{border-radius:10px 10px 4px 4px;border-bottom-width:.5px}
.zoom-pair .icon-btn:last-child {border-radius:4px 4px 10px 10px;border-top-width:.5px}

/* ── Radar badge ── */
.radar-flag{position:fixed;left:50%;transform:translateX(-50%);top:56px;z-index:450;
  display:none;align-items:center;gap:6px;font-size:10px;letter-spacing:1.2px;color:#7fe3ff;
  background:rgba(8,16,24,.80);border:1px solid #15506a;border-radius:20px;
  padding:5px 12px;backdrop-filter:blur(8px);pointer-events:none}
.radar-flag.show{display:flex}
.radar-flag .dot{width:6px;height:6px;border-radius:50%;background:#36c5e0;animation:blink 1.4s infinite}
.radar-flag .rtime{margin-left:4px;opacity:.78}

/* ── Layers panel ── */
.panel{position:fixed;right:14px;bottom:172px;z-index:600;width:222px;
  background:rgba(9,17,26,.97);border:1px solid #163450;border-radius:14px;
  padding:14px 14px 10px;backdrop-filter:blur(12px);display:none;
  box-shadow:0 10px 40px rgba(0,0,0,.55)}
.panel.show{display:block}
.panel h4{font-size:9px;letter-spacing:2px;color:#5a8faa;margin-bottom:11px;text-transform:uppercase}
.prow{display:flex;align-items:center;justify-content:space-between;margin-bottom:11px}
.prow:last-child{margin-bottom:0}
.prow label{font-size:12px;color:#cfe6f2}
.sw{position:relative;width:38px;height:21px;border-radius:11px;background:#1a3145;
  cursor:pointer;transition:.2s;flex:0 0 auto}
.sw.on{background:#0c6d8c}
.sw::after{content:"";position:absolute;top:2px;left:2px;width:17px;height:17px;border-radius:50%;
  background:#cdeefb;transition:.2s}
.sw.on::after{left:19px;background:#7fe3ff}
.slider-sm{width:80px;accent-color:#36c5e0}
.pdiv{height:1px;background:#163450;margin:10px 0 12px}

/* ── Bottom HUD ── */
.bottom{position:fixed;left:0;right:0;bottom:0;z-index:500;padding:12px 14px 10px;
  background:linear-gradient(0deg,rgba(5,10,16,.97) 62%,rgba(5,10,16,0));pointer-events:none}
.bottom>*{pointer-events:auto}
.hazard-bar{display:flex;justify-content:center;margin-bottom:10px}
.hsel{display:flex;gap:4px;background:rgba(8,16,24,.82);border:1px solid #15324a;border-radius:24px;
  padding:4px;backdrop-filter:blur(8px);max-width:96vw;overflow-x:auto;scrollbar-width:none}
.hsel::-webkit-scrollbar{display:none}
.hbtn{border:0;background:none;color:#7ba6bd;font-size:12px;font-weight:700;letter-spacing:.5px;
  padding:6px 13px;border-radius:18px;cursor:pointer;transition:.15s;white-space:nowrap;flex:0 0 auto}
.hbtn:hover{color:#cdeefb}
.hbtn.active{background:linear-gradient(180deg,#0a4d66,#073549);color:#5fe0f5;
  box-shadow:0 0 0 1px #1c6f8c inset}
.hbtn.radar.active{background:linear-gradient(180deg,#0a3a66,#062a49);color:#7fb6ff;
  box-shadow:0 0 0 1px #1c5a8c inset}

.scrub-row{max-width:1100px;margin:0 auto;display:flex;align-items:center;gap:10px}
.pbtn{flex:0 0 auto;width:40px;height:40px;border-radius:50%;border:1.5px solid rgba(54,197,224,.55);
  background:rgba(54,197,224,.14);color:#5fe0f5;font-size:15px;cursor:pointer;display:flex;
  align-items:center;justify-content:center;transition:.15s}
.pbtn:hover{background:rgba(54,197,224,.3)}
.sbtn{flex:0 0 auto;width:30px;height:30px;border-radius:50%;border:1px solid #1b3c54;
  background:rgba(8,16,24,.7);color:#9fc4d8;cursor:pointer;font-size:11px;display:flex;
  align-items:center;justify-content:center}
.sbtn:hover{border-color:#36c5e0;color:#cdeefb}
.tl{flex:1 1 auto;min-width:0}
.tl-top{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:4px}
.tl-time{font-size:13px;font-weight:700;letter-spacing:.3px;color:#eaf6ff;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.tl-time b{color:#5fe0f5}
.tl-time .tag{font-size:9px;font-weight:700;letter-spacing:1px;padding:1px 6px;border-radius:4px;margin-right:6px;
  background:#0c4a63;color:#7fe3ff;vertical-align:middle}
.tl-time .tag.obs{background:#10406a;color:#7fb6ff}
.tl-meta{display:flex;gap:10px;align-items:center;flex:0 0 auto}
.tl-utc{font-size:10px;color:#5a8faa;letter-spacing:.4px}
.speed-wrap{display:flex;align-items:center;gap:5px;font-size:9px;color:#3f6377}
.speed-wrap input{width:52px;accent-color:#36c5e0}

/* risk profile strip */
.strip{display:flex;gap:1px;height:7px;margin:3px 0 1px;border-radius:3px;overflow:hidden;cursor:pointer}
.strip.hide{display:none}
.seg{flex:1 1 0;background:#13202f;transition:background .25s,transform .15s}
.seg.cur{transform:scaleY(1.7)}

input[type=range]{-webkit-appearance:none;appearance:none;width:100%;height:5px;border-radius:3px;
  background:#0e2436;outline:none;margin:6px 0 4px;cursor:pointer}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;width:14px;height:14px;
  border-radius:50%;background:#5fe0f5;border:2px solid #04222e;box-shadow:0 0 7px rgba(95,224,245,.6);cursor:pointer}
input[type=range]::-moz-range-thumb{width:14px;height:14px;border-radius:50%;background:#5fe0f5;
  border:2px solid #04222e;cursor:pointer}
.dayticks{display:flex;justify-content:space-between;margin-top:2px}
.dt{font-size:8px;letter-spacing:1px;color:#3f6377;cursor:pointer;padding:3px 4px;border-radius:5px;text-align:center}
.dt:hover{color:#8ac0d0;background:rgba(54,197,224,.08)}
.dt.on{color:#5fe0f5;font-weight:700}
.dt-label{display:block}
.dh-dots{display:flex;justify-content:center;gap:2px;margin-top:3px}
.dh-dot{width:7px;height:5px;border-radius:1.5px;flex:0 0 auto;opacity:.88}

/* ── Daily digest panel ── */
.digest{position:fixed;left:50%;transform:translateX(-50%);bottom:168px;z-index:600;
  background:rgba(9,17,26,.97);border:1px solid #163450;border-radius:14px;
  padding:12px 14px 10px;backdrop-filter:blur(12px);display:none;
  box-shadow:0 10px 40px rgba(0,0,0,.55);min-width:300px}
.digest.show{display:block}
.digest h4{font-size:9px;letter-spacing:2px;color:#5a8faa;margin-bottom:9px;text-transform:uppercase}
.dg-table{border-collapse:collapse;width:100%}
.dg-table td,.dg-table th{padding:3px 6px;font-size:9px;text-align:center;border-radius:3px}
.dg-table th{color:#5a8faa;letter-spacing:1px;font-weight:600;padding-bottom:5px}
.dg-table td.lbl{text-align:left;color:#9fc4d8;font-size:10px;padding-right:10px;white-space:nowrap}
.dg-table td.cell{font-weight:700;letter-spacing:.5px;font-size:9px;min-width:38px}
.dg-table tr:hover td.cell{outline:1px solid rgba(255,255,255,.2)}

/* ── City markers ── */
.city-icon{background:none!important;border:none!important;overflow:visible!important}
.city-pin{display:flex;flex-direction:column;align-items:center;cursor:pointer}
.city-dot{width:9px;height:9px;border-radius:50%;border:2px solid #050a10;
  background:var(--c,#2a3a48);box-shadow:0 0 6px var(--c,transparent);transition:.3s}
.city-label{margin-top:2px;text-align:center;background:rgba(5,10,16,.72);
  border:1px solid rgba(255,255,255,.1);border-radius:5px;padding:2px 5px;
  line-height:1.25;white-space:nowrap;backdrop-filter:blur(4px)}
.city-name{display:block;font-size:8px;font-weight:700;letter-spacing:.5px;color:#9fc4d8}
.city-risk{display:block;font-size:8px;font-weight:800;color:var(--c,#5a8faa)}

/* ── Popup ── */
.leaflet-popup-content-wrapper.risk-popup-wrap{
  background:rgba(9,17,26,.96);border:1px solid #1e405a;border-radius:10px;
  color:#eaf6ff;padding:0;box-shadow:0 8px 30px rgba(0,0,0,.5)}
.leaflet-popup-tip{background:rgba(9,17,26,.96)!important}
.leaflet-popup-content{margin:0!important}
.popup-inner{padding:10px 14px;min-width:158px}
.popup-hdr{font-size:9px;letter-spacing:1.5px;color:#5a8faa;margin-bottom:8px}
.popup-row{display:flex;justify-content:space-between;align-items:center;
  padding:3px 0;border-bottom:1px solid #142032;font-size:12px}
.popup-row.tot{border-bottom:none;border-top:1px solid #1e405a;margin-top:3px;padding-top:5px}
.popup-row:last-child{border-bottom:none}
.popup-row .ph{color:#9fc4d8}
.popup-row .pv{font-weight:800;font-size:11px}

/* ── Info modal ── */
.modal{position:fixed;inset:0;z-index:1000;display:none;align-items:center;justify-content:center;
  background:rgba(2,6,10,.72);backdrop-filter:blur(4px);padding:20px}
.modal.show{display:flex}
.card{max-width:440px;background:#0a1622;border:1px solid #173552;border-radius:16px;padding:22px;
  box-shadow:0 20px 60px rgba(0,0,0,.6);max-height:88vh;overflow-y:auto}
.card h2{font-size:16px;letter-spacing:1px;margin-bottom:3px}
.card h2 em{color:#36c5e0;font-style:normal}
.card .sub{font-size:9px;letter-spacing:2.5px;color:#5a8faa;margin-bottom:13px}
.card p{font-size:12px;line-height:1.65;color:#bcd6e6;margin-bottom:9px}
.card a{color:#5fe0f5;text-decoration:none}
.card kbd{background:#13283a;border:1px solid #1e405a;border-radius:4px;padding:0 5px;font-size:11px;color:#cdeefb}
.card .close{margin-top:6px;width:100%;padding:9px;border:0;border-radius:9px;
  background:#0c6d8c;color:#eaf6ff;font-size:13px;font-weight:700;cursor:pointer}
.card .close:hover{background:#0d7ea1}

.leaflet-control-zoom{display:none}
.leaflet-control-attribution{font-size:9px!important;background:rgba(5,10,16,.55)!important;color:#456!important}
.leaflet-control-attribution a{color:#5a8faa!important}
@media(max-width:640px){
  .brand{font-size:16px}.legbox{display:none}
  .panel{right:8px;bottom:188px;width:204px}
  .hbtn{padding:5px 10px;font-size:11px}
  .speed-wrap{display:none}
}
</style>
</head>
<body>
<div id="map"></div>

<!-- Top bar -->
<div class="topbar">
  <div>
    <div class="brand">Aus<em>weather</em><small>SEVERE WEATHER · LIVE RADAR</small></div>
  </div>
  <div class="legbox">
    <div id="riskLegend">
      <div class="lg-title">RISK SCALE</div>
      <div class="lg-scale">__LEGEND__</div>
    </div>
    <div class="radar-legend" id="radarLegend">
      <div class="lg-title">RADAR · PRECIP INTENSITY</div>
      <div class="radar-grad"></div>
      <div class="radar-grad-labels"><span>LIGHT</span><span>MOD</span><span>HEAVY</span><span>INTENSE</span></div>
    </div>
  </div>
</div>

<!-- Alert banner -->
<div class="alert" id="alert"><span class="dot"></span><span id="alertTxt"></span></div>

<!-- Left controls -->
<div class="left-stack">
  <button class="icon-btn" id="infoBtn" title="About">&#9432;</button>
  <button class="icon-btn" id="locBtn"  title="My location">&#9678;</button>
  <button class="icon-btn" id="fsBtn"   title="Fullscreen">&#9974;</button>
  <button class="icon-btn" id="digestBtn" title="5-day digest">&#9783;</button>
  <button class="icon-btn" id="layerBtn" title="Layers">&#11052;</button>
  <div class="zoom-pair">
    <button class="icon-btn" id="zoomIn"  title="Zoom in">+</button>
    <button class="icon-btn" id="zoomOut" title="Zoom out">&#8722;</button>
  </div>
</div>

<!-- Daily digest panel -->
<div class="digest" id="digest"></div>

<!-- Radar badge -->
<div class="radar-flag" id="radarFlag">
  <span class="dot"></span>
  <span id="radarMode">RADAR</span>
  <span class="rtime" id="radarTime"></span>
  <span class="rtime" id="radarAge" style="opacity:.5;margin-left:2px"></span>
</div>

<!-- Layers panel -->
<div class="panel" id="panel">
  <h4>Layers</h4>
  <div class="prow"><label>Risk overlay</label><div class="sw on" id="swRisk"></div></div>
  <div class="prow"><label>Risk opacity</label><input type="range" class="slider-sm" id="riskOpac" min="20" max="100" value="85"></div>
  <div class="pdiv"></div>
  <div class="prow"><label>Radar underlay</label><div class="sw on" id="swRadar"></div></div>
  <div class="prow"><label>Satellite IR</label><div class="sw" id="swSatIR"></div></div>
  <div class="prow"><label>Radar opacity</label><input type="range" class="slider-sm" id="radarOpac" min="10" max="100" value="70"></div>
  <div class="pdiv"></div>
  <div class="prow"><label>Satellite map</label><div class="sw" id="swSatMap"></div></div>
  <div class="prow"><label>City markers</label><div class="sw on" id="swCities"></div></div>
  <div class="prow"><label>Map labels</label><div class="sw on" id="swLabels"></div></div>
</div>

<!-- Bottom HUD -->
<div class="bottom">
  <div class="hazard-bar"><div class="hsel" id="hsel"></div></div>
  <div class="scrub-row">
    <button class="pbtn" id="playBtn" title="Play / Pause">&#9654;</button>
    <button class="sbtn" id="prevBtn" title="Previous">&#9664;</button>
    <div class="tl">
      <div class="tl-top">
        <div class="tl-time" id="timeMain">—</div>
        <div class="tl-meta">
          <span class="tl-utc" id="timeUtc"></span>
          <span class="speed-wrap"><span>SPEED</span>
            <input type="range" id="speedCtrl" min="1" max="10" value="5" title="Playback speed"></span>
        </div>
      </div>
      <div class="strip" id="strip"></div>
      <input type="range" id="scrub" min="0" max="__NFRAMES__" value="0">
      <div class="dayticks" id="dayticks"></div>
    </div>
    <button class="sbtn" id="nextBtn" title="Next">&#9654;</button>
  </div>
</div>

<!-- Info modal -->
<div class="modal" id="modal">
  <div class="card">
    <h2>Aus<em>weather</em></h2>
    <div class="sub">5-DAY SEVERE WEATHER OUTLOOK</div>
    <p>Hour-by-hour severe weather risk for Australia — <b>wind, hail, flood and tornado</b> — from the
       NOAA GFS model, plus a live <b>radar</b> mode and infrared satellite.</p>
    <p>Pick a hazard or <b>Max</b> (overall threat). Tap <b>📡 Radar</b> for the live precipitation loop.
       The bar under the timeline shows when risk peaks — click it or the day labels to jump. Click the map
       or a city for a local breakdown.</p>
    <p style="color:#8aabb8">Shortcuts: <kbd>Space</kbd> play · <kbd>&larr;</kbd>/<kbd>&rarr;</kbd> step ·
       <kbd>Home</kbd>/<kbd>End</kbd> first/last frame · <kbd>R</kbd> refresh radar ·
       the &#9678; button flies to your location.</p>
    <p style="color:#8aabb8">The <b>&#9783;</b> button shows a 5-day hazard summary grid. Day-tick dots
       below the timeline are colour-coded by peak risk per hazard. The radar loop auto-refreshes every 5 minutes.</p>
    <p style="color:#7e98a8;font-size:11px">Risk levels are computed from model fields and are
       <b>not official warnings</b>. For authoritative forecasts visit
       <a href="https://www.bom.gov.au" target="_blank" rel="noopener">bom.gov.au</a>.</p>
    <p style="color:#5a7488;font-size:10px">Data: __RUNLABEL__ &middot; generated __TIMESTAMP__ UTC
       &middot; radar &copy; RainViewer &middot; map &copy; CARTO/OSM/Esri</p>
    <button class="close" id="closeBtn">Got it</button>
  </div>
</div>

<script>
const IMAGES      = __IMAGES__;
const META        = __META__;
const HAZARDS     = __HAZARDS__;       // 5: Wind,Hail,Flood,Tornado,Max
const ICONS       = __ICONS__;
const BOUNDS      = __BOUNDS__;
const RISK_COLORS = __RISK_COLORS__;
const RISK_LABELS = ["NONE","MRGL","SLGT","ENH","MDT","HIGH"];
const CITIES      = __CITIES__;
const CITY_RISKS  = __CITY_RISKS__;
const COARSE      = __COARSE__;
const COARSE_LATS = __COARSE_LATS__;
const COARSE_LONS = __COARSE_LONS__;
const COARSE_SHAPE= __COARSE_SHAPE__;
const PROFILE     = __PROFILE__;       // {hazard:[peakPerFrame]}
const N = META.length;
const REAL_HAZARDS = HAZARDS.filter(h=>h!=='Max');
const MODES = HAZARDS.concat(['Radar']);
const LAT0=BOUNDS[0][0], LON0=BOUNDS[0][1], LAT1=BOUNDS[1][0], LON1=BOUNDS[1][1];

let mode='forecast', curHazard=HAZARDS[0], curFrame=0, rIdx=0;
let playing=false, playTimer=null, playInterval=650;
let riskOn=true, riskOpac=.85, radarOpacity=.70;
let radarUnderlay=true, satIROn=false;

/* ── Map ── */
const map=L.map('map',{zoomControl:false,attributionControl:true,
  minZoom:3,maxZoom:13,zoomSnap:.25,inertia:true}).fitBounds(BOUNDS);
map.createPane('radarPane');
map.getPane('radarPane').style.zIndex=350;
map.getPane('radarPane').style.pointerEvents='none';

const tOpt={subdomains:'abcd',maxZoom:19};
const baseDark=L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png',
  {attribution:'&copy; OSM &copy; CARTO',...tOpt}).addTo(map);
const baseSat=L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
  {attribution:'&copy; Esri',maxZoom:19,pane:'tilePane'});
const labels=L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}{r}.png',
  {pane:'markerPane',...tOpt}).addTo(map);

/* ── Risk overlay crossfade pair ── */
function dataUri(b64){return 'data:image/png;base64,'+b64;}
let ovA=L.imageOverlay(dataUri(IMAGES[0][curHazard]),BOUNDS,{opacity:.85,className:'risk-overlay',interactive:false}).addTo(map);
let ovB=L.imageOverlay(dataUri(IMAGES[0][curHazard]),BOUNDS,{opacity:0,className:'risk-overlay',interactive:false}).addTo(map);
let ovTop=ovA;

/* ── Forecast frame ── */
function showForecast(i,animate){
  curFrame=(i+N)%N;
  const uri=dataUri(IMAGES[curFrame][curHazard]);
  const back=(ovTop===ovA)?ovB:ovA;
  back.setUrl(uri);
  if(animate){back.setOpacity(riskOn?riskOpac:0);ovTop.setOpacity(0);ovTop=back;}
  else{ovA.setUrl(uri);ovB.setUrl(uri);ovTop.setOpacity(riskOn?riskOpac:0);back.setOpacity(0);}
  const m=META[curFrame];
  document.getElementById('timeMain').innerHTML=
    '<span class="tag">FORECAST</span>'+m.local;
  document.getElementById('timeUtc').textContent=m.utc;
  document.getElementById('scrub').value=curFrame;
  updateDayticks();updateStripCursor();updateCityMarkers();updateAlert();syncHash();
}

/* ── Hazard / mode selector ── */
const hsel=document.getElementById('hsel');
MODES.forEach(h=>{
  const b=document.createElement('button');
  b.className='hbtn'+(h==='Radar'?' radar':'')+(h===curHazard?' active':'');
  b.dataset.h=h;
  b.innerHTML=(ICONS[h]||'')+' '+(h==='Max'?'MAX':h.toUpperCase());
  b.onclick=()=>selectMode(h);
  hsel.appendChild(b);
});
function highlightMode(h){
  [...hsel.children].forEach(c=>c.classList.toggle('active',c.dataset.h===h));
}
function selectMode(h){
  if(h==='Radar'){
    if(!playing){}else stop();
    enterRadar();highlightMode('Radar');return;
  }
  if(mode==='radar'){stop();exitRadar();}
  mode='forecast';
  curHazard=h;highlightMode(h);
  buildStrip();
  document.getElementById('riskLegend').style.display='';
  document.getElementById('radarLegend').classList.remove('show');
  showForecast(curFrame,false);
}

/* ── Timeline / dayticks ── */
const scrub=document.getElementById('scrub');
scrub.max=N-1;
scrub.oninput=()=>{stop();const v=parseInt(scrub.value);
  if(mode==='radar')showRadar(v);else showForecast(v,false);};
document.getElementById('prevBtn').onclick=()=>{stop();step(-1);};
document.getElementById('nextBtn').onclick=()=>{stop();step(1);};
function step(d){if(mode==='radar'){const n=radarCount();if(n)showRadar((rIdx+d+n)%n);}
  else showForecast(curFrame+d,true);}

const dayFirst={};
META.forEach((m,i)=>{if(!(m.day in dayFirst))dayFirst[m.day]=i;});
const dayLabels=['TODAY','TMW','DAY 3','DAY 4','DAY 5','DAY 6'];
function buildDayticks(){
  const wrap=document.getElementById('dayticks');
  wrap.innerHTML='';
  Object.keys(dayFirst).forEach(d=>{
    const el=document.createElement('div');el.className='dt';el.dataset.day=d;
    const label=dayLabels[parseInt(d)]||('DAY '+(parseInt(d)+1));
    const dayFrames=META.reduce((a,m,i)=>{if(String(m.day)===String(d))a.push(i);return a;},[]);
    const dots=DATA_HAZARDS.map(h=>{
      const mx=dayFrames.reduce((v,i)=>Math.max(v,PROFILE[h]?.[i]||0),0);
      return '<span class="dh-dot" style="background:'+RISK_COLORS[mx]+'" title="'+h+': '+RISK_LABELS[mx]+'"></span>';
    }).join('');
    el.innerHTML='<span class="dt-label">'+label+'</span><div class="dh-dots">'+dots+'</div>';
    el.onclick=()=>{if(mode==='radar')return;stop();showForecast(dayFirst[d],true);};
    wrap.appendChild(el);
  });
}
function updateDayticks(){const d=META[curFrame].day;
  document.querySelectorAll('.dt').forEach(e=>e.classList.toggle('on',e.dataset.day==d));}
buildDayticks();

/* ── Risk profile strip ── */
const stripEl=document.getElementById('strip');
function buildStrip(){
  stripEl.classList.remove('hide');
  const prof=PROFILE[curHazard]||[];
  stripEl.innerHTML='';
  for(let i=0;i<N;i++){
    const s=document.createElement('div');s.className='seg';
    s.style.background=RISK_COLORS[prof[i]||0];
    s.onclick=()=>{stop();showForecast(i,true);};
    stripEl.appendChild(s);
  }
}
function updateStripCursor(){
  [...stripEl.children].forEach((s,i)=>s.classList.toggle('cur',i===curFrame));
}

/* ── Playback ── */
const playBtn=document.getElementById('playBtn');
document.getElementById('speedCtrl').oninput=function(){
  playInterval=Math.round(1400-this.value*120);if(playing){stop();play();}};
function play(){playing=true;playBtn.innerHTML='&#10074;&#10074;';
  playTimer=setInterval(()=>step(1),playInterval);}
function stop(){playing=false;playBtn.innerHTML='&#9654;';clearInterval(playTimer);}
playBtn.onclick=()=>{playing?stop():play();};

/* ── Zoom / fullscreen ── */
document.getElementById('zoomIn').onclick=()=>map.zoomIn();
document.getElementById('zoomOut').onclick=()=>map.zoomOut();
document.getElementById('fsBtn').onclick=()=>{
  if(!document.fullscreenElement)document.documentElement.requestFullscreen?.();
  else document.exitFullscreen?.();};

/* ── City markers ── */
function cityIcon(name,risk){
  const c=RISK_COLORS[risk]||'#2a3a48';
  return L.divIcon({className:'city-icon',iconSize:[0,0],iconAnchor:[0,0],
    html:'<div class="city-pin" style="--c:'+c+'"><div class="city-dot"></div>'+
      '<div class="city-label" style="--c:'+c+'"><span class="city-name">'+name+'</span>'+
      (risk>0?'<span class="city-risk">'+RISK_LABELS[risk]+'</span>':'')+'</div></div>'});
}
const cityMarkers=CITIES.map(([name,lat,lon])=>{
  const mk=L.marker([lat,lon],{icon:cityIcon(name,0),zIndexOffset:100});
  mk.on('click',()=>{
    map.flyTo([lat,lon],8,{duration:1});
    if(mode!=='radar')openCityPopup(name,lat,lon);
  });
  return mk.addTo(map);
});
let citiesOn=true;
function updateCityMarkers(){
  if(!citiesOn||mode==='radar'||!CITY_RISKS[curFrame])return;
  CITIES.forEach(([name],i)=>{
    const r=(CITY_RISKS[curFrame][name]||{})[curHazard]||0;
    cityMarkers[i].setIcon(cityIcon(name,r));
  });
}

/* ── Alert banner ── */
function updateAlert(){
  const el=document.getElementById('alert');
  if(mode==='radar'||!CITY_RISKS[curFrame]){el.classList.remove('show');return;}
  const cr=CITY_RISKS[curFrame];let max=0,maxH='',cs=[];
  REAL_HAZARDS.forEach(h=>CITIES.forEach(([name])=>{
    const r=(cr[name]||{})[h]||0;
    if(r>max){max=r;maxH=h;cs=[name];}
    else if(r===max&&r>=3&&!cs.includes(name))cs.push(name);
  }));
  if(max>=3){el.className='alert show alert-'+max;
    document.getElementById('alertTxt').textContent=
      RISK_LABELS[max]+' '+maxH.toUpperCase()+' — '+cs.slice(0,3).join(', ');}
  else el.classList.remove('show');
}

/* ── Popups ── */
function riskRows(getR){
  let rows='';
  REAL_HAZARDS.forEach(h=>{const r=getR(h);
    rows+='<div class="popup-row"><span class="ph">'+(ICONS[h]||'')+' '+h+
      '</span><span class="pv" style="color:'+RISK_COLORS[r]+'">'+RISK_LABELS[r]+'</span></div>';});
  const mx=getR('Max');
  rows+='<div class="popup-row tot"><span class="ph">⚠️ Overall</span>'+
    '<span class="pv" style="color:'+RISK_COLORS[mx]+'">'+RISK_LABELS[mx]+'</span></div>';
  return rows;
}
const coarseCache={};
function coarseGet(frame,hazard,iLat,iLon){
  const key=frame+'_'+hazard;
  if(!coarseCache[key]){const s=atob(COARSE[frame][hazard]);
    const a=new Uint8Array(s.length);for(let i=0;i<s.length;i++)a[i]=s.charCodeAt(i);coarseCache[key]=a;}
  return coarseCache[key][iLat*COARSE_SHAPE[1]+iLon]||0;
}
function nearest1d(arr,v){let b=0,bd=1e9;for(let i=0;i<arr.length;i++){const d=Math.abs(arr[i]-v);if(d<bd){bd=d;b=i;}}return b;}
function openCityPopup(name,lat,lon){
  const cr=(CITY_RISKS[curFrame]||{})[name]||{};
  L.popup({className:'risk-popup-wrap',maxWidth:210,autoPanPadding:L.point(20,90)})
   .setLatLng([lat,lon])
   .setContent('<div class="popup-inner"><div class="popup-hdr">'+name.toUpperCase()+' · '+META[curFrame].utc+
     '</div>'+riskRows(h=>cr[h]||0)+'</div>').openOn(map);
}
map.on('click',e=>{
  if(mode==='radar')return;
  const lat=e.latlng.lat,lon=e.latlng.lng;
  if(lat<LAT0||lat>LAT1||lon<LON0||lon>LON1)return;
  const iLat=nearest1d(COARSE_LATS,lat),iLon=nearest1d(COARSE_LONS,lon);
  L.popup({className:'risk-popup-wrap',maxWidth:210,autoPanPadding:L.point(20,90)})
   .setLatLng(e.latlng)
   .setContent('<div class="popup-inner"><div class="popup-hdr">'+META[curFrame].utc+
     '</div>'+riskRows(h=>coarseGet(curFrame,h,iLat,iLon))+'</div>').openOn(map);
});

/* ── Layers panel ── */
const panel=document.getElementById('panel');
document.getElementById('layerBtn').onclick=()=>{panel.classList.toggle('show');
  document.getElementById('layerBtn').classList.toggle('on',panel.classList.contains('show'));
  digestEl.classList.remove('show');digestBtn.classList.remove('on');};
function bindSw(id,init,fn){const el=document.getElementById(id);el.classList.toggle('on',init);
  el.onclick=()=>{el.classList.toggle('on');fn(el.classList.contains('on'));};}
bindSw('swRisk',true,on=>{riskOn=on;if(mode==='forecast')ovTop.setOpacity(on?riskOpac:0);});
bindSw('swLabels',true,on=>{on?labels.addTo(map):map.removeLayer(labels);});
bindSw('swCities',true,on=>{citiesOn=on;cityMarkers.forEach(m=>on?m.addTo(map):m.remove());if(on)updateCityMarkers();});
bindSw('swSatMap',false,on=>{if(on){map.removeLayer(baseDark);baseSat.addTo(map);}else{map.removeLayer(baseSat);baseDark.addTo(map);}});
bindSw('swRadar',true,on=>{radarUnderlay=on;paintRadar();});
bindSw('swSatIR',false,on=>{satIROn=on;if(mode==='radar'){rIdx=Math.min(rIdx,radarCount()-1);enterRadar();}else paintRadar();});
document.getElementById('riskOpac').oninput=function(){riskOpac=this.value/100;if(riskOn&&mode==='forecast')ovTop.setOpacity(riskOpac);};
document.getElementById('radarOpac').oninput=function(){radarOpacity=this.value/100;paintRadar();};

/* ── Radar (RainViewer) ── */
let radarFrames=[],satFrames=[],pastCount=0,radarReady=false,radarTimer=null;
const radarFlag=document.getElementById('radarFlag');
function curSet(){return satIROn?satFrames:radarFrames;}
function radarCount(){return curSet().length;}
let radarLoadedAt=null;
function loadRadar(){
  fetch('https://api.rainviewer.com/public/weather-maps.json')
   .then(r=>r.json()).then(d=>{
     const host=d.host;
     const past=(d.radar&&d.radar.past?d.radar.past:[]).slice(-12);
     const now=(d.radar&&d.radar.nowcast?d.radar.nowcast:[]).slice(0,3);
     pastCount=past.length;
     radarFrames=past.concat(now).map(fr=>({time:fr.time,
       layer:L.tileLayer(host+fr.path+'/256/{z}/{x}/{y}/6/1_1.png',
         {opacity:0,pane:'radarPane',maxZoom:12}).addTo(map)}));
     const ir=(d.satellite&&d.satellite.infrared?d.satellite.infrared:[]).slice(-12);
     satFrames=ir.map(fr=>({time:fr.time,
       layer:L.tileLayer(host+fr.path+'/256/{z}/{x}/{y}/0/0_0.png',
         {opacity:0,pane:'radarPane',maxZoom:12}).addTo(map)}));
     radarReady=true;
     radarLoadedAt=Date.now();
     document.getElementById('radarAge').textContent='Updated '+fmtTime(Math.floor(radarLoadedAt/1000));
     paintRadar();
     if(mode==='radar')enterRadar();
   }).catch(e=>{radarReady=false;console.warn('Radar load failed:',e);
     setTimeout(loadRadar,30000);});
}
function hideAllRadar(){radarFrames.forEach(f=>f.layer.setOpacity(0));satFrames.forEach(f=>f.layer.setOpacity(0));}
function reloadRadar(){
  radarFrames.forEach(f=>{try{map.removeLayer(f.layer);}catch(e){}});
  satFrames.forEach(f=>{try{map.removeLayer(f.layer);}catch(e){}});
  radarFrames=[];satFrames=[];pastCount=0;radarReady=false;
  document.getElementById('radarAge').textContent='';
  loadRadar();
}
setInterval(reloadRadar,300000);
function fmtTime(ts){const d=new Date(ts*1000);
  return ('0'+d.getUTCHours()).slice(-2)+':'+('0'+d.getUTCMinutes()).slice(-2)+' UTC';}
function paintRadar(){
  // forecast underlay: show most-recent observation under the risk
  if(mode==='forecast'){
    hideAllRadar();
    if(radarUnderlay&&radarReady){
      const set=curSet();const idx=satIROn?set.length-1:Math.max(0,pastCount-1);
      if(set[idx])set[idx].layer.setOpacity(radarOpacity*.85);
      radarFlag.classList.add('show');
      document.getElementById('radarMode').textContent=satIROn?'SAT IR · LATEST':'RADAR · LATEST';
      document.getElementById('radarTime').textContent=set[idx]?fmtTime(set[idx].time):'';
    }else radarFlag.classList.remove('show');
  }
}
function enterRadar(){
  mode='radar';
  ovA.setOpacity(0);ovB.setOpacity(0);
  stripEl.classList.add('hide');
  document.getElementById('riskLegend').style.display='none';
  document.getElementById('radarLegend').classList.add('show');
  document.getElementById('alert').classList.remove('show');
  clearInterval(radarTimer);
  if(!radarReady||!radarCount()){
    document.getElementById('timeMain').innerHTML='<span class="tag obs">RADAR</span>loading…';
    return;
  }
  rIdx=Math.min(rIdx,radarCount()-1);
  if(rIdx===0)rIdx=Math.max(0,pastCount-1);
  scrub.max=radarCount()-1;
  showRadar(rIdx);
  if(!playing)play();
}
function exitRadar(){clearInterval(radarTimer);hideAllRadar();mode='forecast';scrub.max=N-1;paintRadar();}
function showRadar(i){
  const set=curSet();const L0=set.length;if(!L0)return;
  rIdx=(i+L0)%L0;
  hideAllRadar();
  set[rIdx].layer.setOpacity(radarOpacity);
  const nc=rIdx>=pastCount&&!satIROn;
  const tag=nc?'NOWCAST':(satIROn?'SAT IR':'OBSERVED');
  const frameLabel='('+(rIdx+1)+'/'+L0+')';
  document.getElementById('timeMain').innerHTML=
    '<span class="tag obs">'+tag+'</span>'+fmtTime(set[rIdx].time)+
    ' <span style="font-size:9px;color:#5a8faa;margin-left:4px">'+frameLabel+'</span>';
  document.getElementById('timeUtc').textContent=satIROn?'Infrared satellite':'Precipitation radar';
  document.getElementById('radarMode').textContent=satIROn?'SATELLITE IR':'RADAR';
  document.getElementById('radarTime').textContent=fmtTime(set[rIdx].time);
  radarFlag.classList.add('show');
  scrub.max=L0-1;scrub.value=rIdx;syncHash();
}

/* ── Daily digest panel ── */
const digestEl=document.getElementById('digest');
function buildDigest(){
  const days=Object.keys(dayFirst);
  let html='<h4>5-Day Risk Outlook</h4><table class="dg-table"><thead><tr><th></th>';
  days.forEach(d=>{html+='<th>'+( dayLabels[parseInt(d)]||'DAY '+(parseInt(d)+1))+'</th>';});
  html+='</tr></thead><tbody>';
  REAL_HAZARDS.forEach(h=>{
    html+='<tr><td class="lbl">'+(ICONS[h]||'')+' '+h+'</td>';
    days.forEach(d=>{
      const dayFrames=META.reduce((a,m,i)=>{if(String(m.day)===String(d))a.push(i);return a;},[]);
      const mx=dayFrames.reduce((v,i)=>Math.max(v,PROFILE[h]?.[i]||0),0);
      const tc=mx>=3?'#fff':mx>0?'#9fc4d8':'#3f6377';
      html+='<td class="cell" style="background:'+RISK_COLORS[mx]+';color:'+tc+'" '+
        'onclick="stop();showForecast(dayFirst[\''+d+'\'],true);digestEl.classList.remove(\'show\');digestBtn.classList.remove(\'on\')">'+
        RISK_LABELS[mx]+'</td>';
    });
    html+='</tr>';
  });
  html+='</tbody></table>';
  digestEl.innerHTML=html;
}
const digestBtn=document.getElementById('digestBtn');
digestBtn.onclick=()=>{
  const showing=digestEl.classList.toggle('show');
  digestBtn.classList.toggle('on',showing);
  if(showing&&!digestEl.querySelector('table'))buildDigest();
  document.getElementById('panel').classList.remove('show');
  document.getElementById('layerBtn').classList.remove('on');
};

/* ── Info modal ── */
const modal=document.getElementById('modal');
document.getElementById('infoBtn').onclick=()=>modal.classList.add('show');
document.getElementById('closeBtn').onclick=()=>modal.classList.remove('show');
modal.onclick=e=>{if(e.target===modal)modal.classList.remove('show');};

/* ── Geolocation ── */
document.getElementById('locBtn').onclick=()=>{if(!navigator.geolocation)return;
  navigator.geolocation.getCurrentPosition(
    p=>map.flyTo([p.coords.latitude,p.coords.longitude],9,{duration:1.2}),
    ()=>{},{enableHighAccuracy:true,timeout:8000});};

/* ── Keyboard ── */
document.addEventListener('keydown',e=>{
  if(e.code==='Space'){e.preventDefault();playing?stop():play();}
  else if(e.code==='ArrowRight'){stop();step(1);}
  else if(e.code==='ArrowLeft'){stop();step(-1);}
  else if(e.code==='Home'){stop();if(mode==='radar')showRadar(0);else showForecast(0,true);}
  else if(e.code==='End'){stop();if(mode==='radar')showRadar(radarCount()-1);else showForecast(N-1,true);}
  else if(e.code==='KeyR'&&!e.ctrlKey){if(mode==='radar')reloadRadar();}});

/* ── Share hash ── */
function syncHash(){
  const v=(mode==='radar')?('Radar/'+rIdx):(curHazard+'/'+curFrame);
  history.replaceState(null,'','#'+v);
}
function restoreHash(){
  const m=(location.hash||'').replace('#','').split('/');
  if(m.length===2){
    const h=decodeURIComponent(m[0]),i=parseInt(m[1])||0;
    if(h==='Radar'){rIdx=i;return 'Radar';}
    if(HAZARDS.includes(h)){curHazard=h;curFrame=Math.max(0,Math.min(N-1,i));return h;}
  }
  return null;
}

/* ── Boot ── */
const want=restoreHash();
buildStrip();
if(want==='Radar'){highlightMode('Radar');showForecast(curFrame,false);enterRadar();}
else{highlightMode(curHazard);showForecast(curFrame,false);}
loadRadar();
setTimeout(()=>{if(!sessionStorage.getItem('aw_seen')){modal.classList.add('show');sessionStorage.setItem('aw_seen','1');}},500);
</script>
</body>
</html>"""


# ── Main ───────────────────────────────────────────────────────────────────────
def aest_label(utc_dt):
    """Eastern Australia local time (AEST, UTC+10; ignores DST)."""
    lt = utc_dt + timedelta(hours=10)
    h = lt.hour
    ampm = "am" if h < 12 else "pm"
    h12 = h % 12 or 12
    return f"{lt.strftime('%a %d %b')} · <b>{h12}:{lt.minute:02d}{ampm}</b> AEST"


def main():
    print("=" * 60)
    print(f"  Ausweather  [GFS hourly · {len(FRAME_HOURS)} frames × {len(HAZARDS)} hazards]")
    print("=" * 60)

    print("\n[1/4] Fetching Australia map geometry...")
    geojson = fetch_geojson()
    if geojson:
        polys_ll  = geojson_to_polygons(geojson)
        polys_proj = [project_poly(p) for p in polys_ll]
        clip_path  = polys_to_clip_path(polys_proj)
        print(f"      {len(polys_proj)} polygons loaded")
    else:
        polys_proj, clip_path = [], None
        print("      WARNING: no map geometry (overlays will not be coast-clipped)")

    print("\n[2/4] Locating latest GFS run...")
    date_s, run_s, run_dt = find_latest_run()
    if not date_s:
        print("      ERROR: GFS data unavailable"); sys.exit(1)
    run_label = f"NOAA GFS {run_dt.strftime('%Y-%m-%d %HZ')}"
    print(f"      {run_label}")

    print(f"\n[3/4] Downloading + rendering {len(FRAME_HOURS)} frames...")
    frames = []
    first_date = (run_dt + timedelta(hours=FRAME_HOURS[0])).date()
    coarse_lats = coarse_lons = None

    for n, fhour in enumerate(FRAME_HOURS):
        valid = run_dt + timedelta(hours=fhour)
        print(f"\n  [{n+1}/{len(FRAME_HOURS)}] f{fhour:03d}  {valid:%a %d %b %HZ}", end="  ", flush=True)
        lats, lons, fields = fetch_frame(date_s, run_s, fhour)
        risks = compute_risks(fields)
        city_r = city_risks_for_frame(risks, lats, lons)
        clats, clons, coarse = coarse_risks(risks, lats, lons)
        if coarse_lats is None:
            coarse_lats, coarse_lons = clats, clons
        imgs = {}
        for hz in DATA_HAZARDS:
            imgs[hz] = render_overlay(lats, lons, risks[hz], polys_proj, clip_path)
        peak = {hz: int(risks[hz].max()) for hz in DATA_HAZARDS}
        day = (valid.date() - first_date).days
        frames.append({
            "f": fhour,
            "utc": valid.strftime("%H:%MZ %d %b"),
            "local": aest_label(valid),
            "day": day,
            "images": imgs,
            "city_risks": city_r,
            "coarse": coarse,
            "peak": peak,
        })
        print("✓", end="", flush=True)

    print("\n\n[4/4] Writing index.html...")
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    html = make_html(frames, run_label, timestamp, coarse_lats or [], coarse_lons or [])
    with open(out, "w") as f:
        f.write(html)
    size_mb = len(html.encode()) / 1e6
    print(f"      {out}  ({size_mb:.1f} MB)\n")


if __name__ == "__main__":
    main()
