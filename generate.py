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
HAZARD_ICONS = {"Wind": "💨", "Hail": "🧊", "Flood": "🌊", "Tornado": "🌪"}

RISK_LABELS = ["NONE", "MRGL", "SLGT", "ENH", "MDT", "HIGH"]

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
def make_html(frames, run_label, timestamp):
    """
    frames: list of dicts: { f, utc, local, day, images:{hazard:b64} }
    """
    images = [fr["images"] for fr in frames]
    meta   = [{"f": fr["f"], "utc": fr["utc"], "local": fr["local"], "day": fr["day"]}
              for fr in frames]

    legend_colors = ["#2a3a48"] + [mcolors.to_hex(HEATMAP_CMAP(i / 5.0)) for i in range(1, 6)]
    legend = "".join(
        f'<div class="lg-item"><span class="lg-sw" style="background:{legend_colors[i]}"></span>{RISK_LABELS[i]}</div>'
        for i in range(6)
    )

    tpl = _HTML_TEMPLATE
    repl = {
        "__IMAGES__":  json.dumps(images, separators=(",", ":")),
        "__META__":    json.dumps(meta, separators=(",", ":")),
        "__HAZARDS__": json.dumps(HAZARDS),
        "__ICONS__":   json.dumps(HAZARD_ICONS),
        "__BOUNDS__":  json.dumps([[LAT0, LON0], [LAT1, LON1]]),
        "__LEGEND__":  legend,
        "__RUNLABEL__": run_label,
        "__TIMESTAMP__": timestamp,
        "__NFRAMES__": str(len(frames)),
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
.risk-overlay{image-rendering:auto;transition:opacity .45s ease}

/* ── Top bar ── */
.topbar{position:fixed;top:0;left:0;right:0;z-index:500;display:flex;align-items:flex-start;
  gap:14px;padding:12px 14px;pointer-events:none;
  background:linear-gradient(180deg,rgba(5,10,16,.92),rgba(5,10,16,0))}
.brand{font-size:21px;font-weight:800;letter-spacing:2px;text-transform:uppercase;
  text-shadow:0 2px 10px rgba(0,0,0,.6)}
.brand em{color:#36c5e0;font-style:normal}
.brand small{display:block;font-size:9px;letter-spacing:3px;color:#5a8faa;font-weight:600;margin-top:1px}
.legend{margin-left:auto;pointer-events:auto;background:rgba(8,16,24,.78);border:1px solid #15324a;
  border-radius:10px;padding:8px 11px;backdrop-filter:blur(8px)}
.legend .lg-title{font-size:8px;letter-spacing:2px;color:#5a8faa;margin-bottom:6px}
.legend .lg-scale{display:flex;gap:0}
.lg-item{font-size:8px;letter-spacing:.5px;color:#9fc4d8;text-align:center;width:38px}
.lg-sw{display:block;height:8px;border-radius:2px;margin-bottom:3px}

.icon-btn{pointer-events:auto;width:38px;height:38px;border-radius:10px;border:1px solid #15324a;
  background:rgba(8,16,24,.78);color:#9fc4d8;font-size:16px;cursor:pointer;display:flex;
  align-items:center;justify-content:center;backdrop-filter:blur(8px);transition:.15s}
.icon-btn:hover{border-color:#36c5e0;color:#cdeefb}
.icon-btn.on{border-color:#36c5e0;color:#36c5e0;background:rgba(0,40,60,.6)}

.left-stack{position:fixed;top:64px;left:14px;z-index:500;display:flex;flex-direction:column;gap:8px}

/* ── Bottom HUD ── */
.bottom{position:fixed;left:0;right:0;bottom:0;z-index:500;padding:14px 14px 12px;
  background:linear-gradient(0deg,rgba(5,10,16,.95) 55%,rgba(5,10,16,0));pointer-events:none}
.bottom>*{pointer-events:auto}
.hazard-bar{display:flex;justify-content:center;margin-bottom:12px}
.hsel{display:flex;gap:6px;background:rgba(8,16,24,.8);border:1px solid #15324a;border-radius:24px;
  padding:5px;backdrop-filter:blur(8px)}
.hbtn{border:0;background:none;color:#7ba6bd;font-size:12px;font-weight:700;letter-spacing:.5px;
  padding:7px 15px;border-radius:18px;cursor:pointer;transition:.15s;white-space:nowrap}
.hbtn:hover{color:#cdeefb}
.hbtn.active{background:linear-gradient(180deg,#0a4d66,#073549);color:#5fe0f5;
  box-shadow:0 0 0 1px #1c6f8c inset}

.scrub{max-width:1100px;margin:0 auto;display:flex;align-items:center;gap:12px}
.pbtn{flex:0 0 auto;width:42px;height:42px;border-radius:50%;border:1.5px solid rgba(54,197,224,.55);
  background:rgba(54,197,224,.14);color:#5fe0f5;font-size:16px;cursor:pointer;display:flex;
  align-items:center;justify-content:center;transition:.15s}
.pbtn:hover{background:rgba(54,197,224,.3)}
.sbtn{flex:0 0 auto;width:32px;height:32px;border-radius:50%;border:1px solid #1b3c54;
  background:rgba(8,16,24,.7);color:#9fc4d8;cursor:pointer;font-size:12px}
.sbtn:hover{border-color:#36c5e0;color:#cdeefb}
.tl{flex:1 1 auto}
.tl-top{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:5px}
.tl-time{font-size:13px;font-weight:700;letter-spacing:.4px;color:#eaf6ff}
.tl-time b{color:#5fe0f5}
.tl-utc{font-size:10px;color:#5a8faa;letter-spacing:.5px}
.range{position:relative;height:22px}
input[type=range]{-webkit-appearance:none;appearance:none;width:100%;height:5px;border-radius:3px;
  background:#0e2436;outline:none;margin:8px 0;cursor:pointer}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;width:15px;height:15px;
  border-radius:50%;background:#5fe0f5;border:2px solid #04222e;box-shadow:0 0 8px rgba(95,224,245,.7);cursor:pointer}
input[type=range]::-moz-range-thumb{width:15px;height:15px;border-radius:50%;background:#5fe0f5;
  border:2px solid #04222e;cursor:pointer}
.dayticks{display:flex;justify-content:space-between;margin-top:2px}
.dt{font-size:8.5px;letter-spacing:1px;color:#3f6377}
.dt.on{color:#5fe0f5;font-weight:700}

/* radar badge */
.radar-flag{position:fixed;left:50%;transform:translateX(-50%);top:14px;z-index:450;
  display:none;align-items:center;gap:6px;font-size:10px;letter-spacing:1.5px;color:#7fe3ff;
  background:rgba(8,16,24,.78);border:1px solid #15506a;border-radius:20px;padding:5px 12px;backdrop-filter:blur(8px)}
.radar-flag.show{display:flex}
.radar-flag .dot{width:7px;height:7px;border-radius:50%;background:#36c5e0;animation:pulse 1.4s infinite}
@keyframes pulse{0%,100%{opacity:.35}50%{opacity:1}}

/* layers panel */
.panel{position:fixed;right:14px;bottom:150px;z-index:600;width:210px;background:rgba(9,17,26,.95);
  border:1px solid #163450;border-radius:14px;padding:14px;backdrop-filter:blur(10px);display:none;
  box-shadow:0 10px 40px rgba(0,0,0,.5)}
.panel.show{display:block}
.panel h4{font-size:9px;letter-spacing:2px;color:#5a8faa;margin-bottom:10px;text-transform:uppercase}
.row{display:flex;align-items:center;justify-content:space-between;margin-bottom:11px}
.row label{font-size:12px;color:#cfe6f2}
.row:last-child{margin-bottom:0}
.sw{position:relative;width:38px;height:21px;border-radius:11px;background:#1a3145;cursor:pointer;transition:.2s;flex:0 0 auto}
.sw.on{background:#0c6d8c}
.sw::after{content:"";position:absolute;top:2px;left:2px;width:17px;height:17px;border-radius:50%;
  background:#cdeefb;transition:.2s}
.sw.on::after{left:19px;background:#7fe3ff}
.opac{width:90px}

/* info modal */
.modal{position:fixed;inset:0;z-index:1000;display:none;align-items:center;justify-content:center;
  background:rgba(2,6,10,.7);backdrop-filter:blur(4px);padding:20px}
.modal.show{display:flex}
.card{max-width:440px;background:#0a1622;border:1px solid #173552;border-radius:16px;padding:24px;
  box-shadow:0 20px 60px rgba(0,0,0,.6)}
.card h2{font-size:16px;letter-spacing:1px;margin-bottom:4px}
.card h2 em{color:#36c5e0;font-style:normal}
.card .sub{font-size:10px;letter-spacing:2px;color:#5a8faa;margin-bottom:14px}
.card p{font-size:12.5px;line-height:1.6;color:#bcd6e6;margin-bottom:10px}
.card a{color:#5fe0f5;text-decoration:none}
.card .close{margin-top:8px;width:100%;padding:10px;border:0;border-radius:9px;
  background:#0c6d8c;color:#eaf6ff;font-size:13px;font-weight:700;cursor:pointer}
.card .close:hover{background:#0d7ea1}

.leaflet-control-zoom{display:none}
.leaflet-control-attribution{font-size:9px!important;background:rgba(5,10,16,.6)!important;color:#456!important}
.leaflet-control-attribution a{color:#5a8faa!important}
@media(max-width:640px){
  .brand{font-size:17px}.legend{padding:6px 8px}.lg-item{width:30px}
  .panel{bottom:170px}
}
</style>
</head>
<body>
<div id="map"></div>

<!-- top -->
<div class="topbar">
  <div>
    <div class="brand">Aus<em>weather</em><small>SEVERE WEATHER · LIVE RADAR</small></div>
  </div>
  <div class="legend">
    <div class="lg-title">RISK SCALE</div>
    <div class="lg-scale">__LEGEND__</div>
  </div>
</div>
<div class="left-stack">
  <button class="icon-btn" id="infoBtn" title="About">&#9432;</button>
  <button class="icon-btn" id="locBtn" title="My location">&#9678;</button>
  <button class="icon-btn" id="layerBtn" title="Layers">&#9783;</button>
</div>

<div class="radar-flag" id="radarFlag"><span class="dot"></span>LIVE RADAR</div>

<!-- layers panel -->
<div class="panel" id="panel">
  <h4>Layers</h4>
  <div class="row"><label>Risk overlay</label><div class="sw on" id="swRisk"></div></div>
  <div class="row"><label>Risk opacity</label>
    <input type="range" class="opac" id="riskOpac" min="20" max="100" value="85"></div>
  <div class="row"><label>Live radar</label><div class="sw on" id="swRadar"></div></div>
  <div class="row"><label>Map labels</label><div class="sw on" id="swLabels"></div></div>
</div>

<!-- bottom HUD -->
<div class="bottom">
  <div class="hazard-bar"><div class="hsel" id="hsel"></div></div>
  <div class="scrub">
    <button class="pbtn" id="playBtn" title="Play">&#9654;</button>
    <button class="sbtn" id="prevBtn" title="Previous hour">&#9664;</button>
    <div class="tl">
      <div class="tl-top">
        <div class="tl-time" id="timeMain">—</div>
        <div class="tl-utc" id="timeUtc"></div>
      </div>
      <div class="range">
        <input type="range" id="scrub" min="0" max="__NFRAMES__" value="0">
      </div>
      <div class="dayticks" id="dayticks"></div>
    </div>
    <button class="sbtn" id="nextBtn" title="Next hour">&#9654;</button>
  </div>
</div>

<!-- info modal -->
<div class="modal" id="modal">
  <div class="card">
    <h2>Aus<em>weather</em></h2>
    <div class="sub">5-DAY SEVERE WEATHER OUTLOOK</div>
    <p>Hour-by-hour severe weather risk for Australia — <b>wind, hail, flood and tornado</b> —
       derived from the NOAA GFS global forecast model, overlaid on live precipitation radar.</p>
    <p>Use the timeline to step through individual forecast hours. Pan and zoom the map (or tap
       the &#9678; button) to focus on your local area. Toggle live radar and layers with the &#9783; button.</p>
    <p style="color:#7e98a8;font-size:11px">Risk levels are computed heuristically from model fields
       and are <b>not official warnings</b>. For authoritative forecasts and warnings visit
       <a href="https://www.bom.gov.au" target="_blank" rel="noopener">bom.gov.au</a>.</p>
    <p style="color:#5a7488;font-size:10px">Data: __RUNLABEL__ · generated __TIMESTAMP__ UTC ·
       radar &copy; RainViewer · map &copy; CARTO/OpenStreetMap</p>
    <button class="close" id="closeBtn">Got it</button>
  </div>
</div>

<script>
const IMAGES = __IMAGES__;     // [frame][hazard] = base64 png
const META   = __META__;       // [{f,utc,local,day}]
const HAZARDS= __HAZARDS__;
const ICONS  = __ICONS__;
const BOUNDS = __BOUNDS__;     // [[S,W],[N,E]]
const N = META.length;

let curFrame = 0, curHazard = HAZARDS[0], playing = false, playTimer = null;

/* ── Map ── */
const map = L.map('map',{zoomControl:false,attributionControl:true,minZoom:3,maxZoom:11,
  zoomSnap:.25,inertia:true}).fitBounds(BOUNDS);

// dedicated pane so radar draws above the basemap but below the risk overlay
map.createPane('radarPane');
map.getPane('radarPane').style.zIndex = 350;
map.getPane('radarPane').style.pointerEvents = 'none';

const baseDark = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png',
  {subdomains:'abcd',maxZoom:19,attribution:'&copy; OpenStreetMap &copy; CARTO'}).addTo(map);
const labels = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}{r}.png',
  {subdomains:'abcd',maxZoom:19,pane:'markerPane'}).addTo(map);

/* ── Risk overlay (two layers for crossfade) ── */
function dataUri(b64){return 'data:image/png;base64,'+b64;}
let ovA = L.imageOverlay(dataUri(IMAGES[0][curHazard]),BOUNDS,
            {opacity:.85,className:'risk-overlay',interactive:false}).addTo(map);
let ovB = L.imageOverlay(dataUri(IMAGES[0][curHazard]),BOUNDS,
            {opacity:0,className:'risk-overlay',interactive:false}).addTo(map);
let ovTop = ovA, riskOn = true, riskOpac = .85;

function showFrame(i,animate){
  curFrame = (i+N)%N;
  const uri = dataUri(IMAGES[curFrame][curHazard]);
  const back = (ovTop===ovA)?ovB:ovA;
  back.setUrl(uri);
  if(animate){
    back.setOpacity(riskOn?riskOpac:0);
    ovTop.setOpacity(0);
    ovTop = back;
  }else{
    ovA.setUrl(uri); ovB.setUrl(uri);
    ovTop.setOpacity(riskOn?riskOpac:0);
    back.setOpacity(0);
  }
  const m = META[curFrame];
  document.getElementById('timeMain').innerHTML = m.local;
  document.getElementById('timeUtc').textContent = m.utc;
  document.getElementById('scrub').value = curFrame;
  updateDayticks();
}

/* ── Hazard selector ── */
const hsel = document.getElementById('hsel');
HAZARDS.forEach(h=>{
  const b=document.createElement('button');
  b.className='hbtn'+(h===curHazard?' active':'');
  b.innerHTML=(ICONS[h]||'')+' '+h.toUpperCase();
  b.onclick=()=>{curHazard=h;
    [...hsel.children].forEach(c=>c.classList.toggle('active',c===b));
    showFrame(curFrame,false);};
  hsel.appendChild(b);
});

/* ── Timeline ── */
const scrub=document.getElementById('scrub');
scrub.max=N-1;
scrub.oninput=()=>{stop();showFrame(parseInt(scrub.value),false);};
document.getElementById('prevBtn').onclick=()=>{stop();showFrame(curFrame-1,true);};
document.getElementById('nextBtn').onclick=()=>{stop();showFrame(curFrame+1,true);};

function buildDayticks(){
  const wrap=document.getElementById('dayticks');
  const seen={};
  META.forEach((m,i)=>{ if(!(m.day in seen)) seen[m.day]=i; });
  const dayName=['TODAY','TOMORROW','DAY 3','DAY 4','DAY 5','DAY 6'];
  wrap.innerHTML='';
  Object.keys(seen).forEach(d=>{
    const s=document.createElement('div');s.className='dt';s.dataset.day=d;
    s.textContent=dayName[d]||('DAY '+(parseInt(d)+1));
    wrap.appendChild(s);
  });
}
function updateDayticks(){
  const d=META[curFrame].day;
  document.querySelectorAll('.dt').forEach(e=>e.classList.toggle('on',e.dataset.day==d));
}
buildDayticks();

/* ── Playback ── */
const playBtn=document.getElementById('playBtn');
function play(){playing=true;playBtn.innerHTML='&#10074;&#10074;';
  playTimer=setInterval(()=>showFrame(curFrame+1,true),650);}
function stop(){playing=false;playBtn.innerHTML='&#9654;';clearInterval(playTimer);}
playBtn.onclick=()=>{playing?stop():play();};

/* ── Layers panel ── */
const panel=document.getElementById('panel');
document.getElementById('layerBtn').onclick=()=>{
  panel.classList.toggle('show');
  document.getElementById('layerBtn').classList.toggle('on',panel.classList.contains('show'));
};
function bindSwitch(id,init,fn){
  const el=document.getElementById(id);
  el.classList.toggle('on',init);
  el.onclick=()=>{el.classList.toggle('on');fn(el.classList.contains('on'));};
}
bindSwitch('swRisk',true,on=>{riskOn=on;ovTop.setOpacity(on?riskOpac:0);});
bindSwitch('swRadar',true,on=>{on?startRadar():stopRadar();});
bindSwitch('swLabels',true,on=>{on?labels.addTo(map):map.removeLayer(labels);});
document.getElementById('riskOpac').oninput=function(){
  riskOpac=this.value/100; if(riskOn)ovTop.setOpacity(riskOpac);};

/* ── Info modal ── */
const modal=document.getElementById('modal');
document.getElementById('infoBtn').onclick=()=>modal.classList.add('show');
document.getElementById('closeBtn').onclick=()=>modal.classList.remove('show');
modal.onclick=e=>{if(e.target===modal)modal.classList.remove('show');};

/* ── Geolocation ── */
document.getElementById('locBtn').onclick=()=>{
  if(!navigator.geolocation)return;
  navigator.geolocation.getCurrentPosition(p=>{
    map.flyTo([p.coords.latitude,p.coords.longitude],8,{duration:1.2});
  },()=>{},{enableHighAccuracy:true,timeout:8000});
};

/* ── Live radar (RainViewer, loads in viewer's browser) ── */
let radarLayers=[], radarTimer=null, radarIdx=0, radarOn=true;
const radarFlag=document.getElementById('radarFlag');
function startRadar(){
  radarOn=true;
  if(radarLayers.length){animateRadar();return;}
  fetch('https://api.rainviewer.com/public/weather-maps.json')
   .then(r=>r.json()).then(d=>{
      const host=d.host;
      const past=(d.radar&&d.radar.past)||[];
      const now=(d.radar&&d.radar.nowcast)||[];
      const frames=past.slice(-10).concat(now.slice(0,3));
      if(!frames.length)return;
      radarLayers=frames.map(fr=>L.tileLayer(
        host+fr.path+'/256/{z}/{x}/{y}/4/1_1.png',
        {opacity:0,pane:'radarPane',maxZoom:12})).map(l=>l.addTo(map));
      animateRadar();
   }).catch(()=>{});
}
function animateRadar(){
  if(!radarOn||!radarLayers.length)return;
  radarFlag.classList.add('show');
  clearInterval(radarTimer);
  radarTimer=setInterval(()=>{
    radarLayers.forEach((l,i)=>l.setOpacity(i===radarIdx?.7:0));
    radarIdx=(radarIdx+1)%radarLayers.length;
  },420);
}
function stopRadar(){
  radarOn=false;clearInterval(radarTimer);
  radarLayers.forEach(l=>l.setOpacity(0));
  radarFlag.classList.remove('show');
}

/* keyboard */
document.addEventListener('keydown',e=>{
  if(e.code==='Space'){e.preventDefault();playing?stop():play();}
  else if(e.code==='ArrowRight'){stop();showFrame(curFrame+1,true);}
  else if(e.code==='ArrowLeft'){stop();showFrame(curFrame-1,true);}
});

/* boot */
showFrame(0,false);
startRadar();
setTimeout(()=>{if(!sessionStorage.getItem('aw_seen')){modal.classList.add('show');
  sessionStorage.setItem('aw_seen','1');}},400);
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

    for n, fhour in enumerate(FRAME_HOURS):
        valid = run_dt + timedelta(hours=fhour)
        print(f"\n  [{n+1}/{len(FRAME_HOURS)}] f{fhour:03d}  {valid:%a %d %b %HZ}", end="  ", flush=True)
        lats, lons, fields = fetch_frame(date_s, run_s, fhour)
        risks = compute_risks(fields)
        imgs = {}
        for hz in HAZARDS:
            imgs[hz] = render_overlay(lats, lons, risks[hz], polys_proj, clip_path)
        day = (valid.date() - first_date).days
        frames.append({
            "f": fhour,
            "utc": valid.strftime("%H:%MZ %d %b"),
            "local": aest_label(valid),
            "day": day,
            "images": imgs,
        })
        print("✓", end="", flush=True)

    print("\n\n[4/4] Writing index.html...")
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    html = make_html(frames, run_label, timestamp)
    with open(out, "w") as f:
        f.write(html)
    size_mb = len(html.encode()) / 1e6
    print(f"      {out}  ({size_mb:.1f} MB)\n")


if __name__ == "__main__":
    main()
