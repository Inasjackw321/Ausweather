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

# Regional boxes used for warning generation (land-masked, so generous
# boxes that spill offshore are harmless).
REGIONS = [
    ("NSW", 140.9, 153.7, -37.6, -28.1),
    ("VIC", 140.9, 150.0, -39.2, -33.9),
    ("QLD", 137.9, 153.7, -29.2, -10.4),
    ("SA",  128.9, 141.1, -38.2, -25.9),
    ("WA",  112.8, 129.1, -35.3, -13.5),
    ("NT",  128.9, 138.1, -25.9, -10.8),
    ("TAS", 143.7, 148.6, -43.8, -39.5),
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


def land_mask(lats, lons, polys_ll):
    """
    Boolean grid, True over Australian land (dilated by one cell so coastal
    grid points survive). Risk fields are masked with this so that offshore
    convection never drives city/region warnings or the peak-risk profile —
    the rendered overlays were already coast-clipped, but the numbers weren't.
    """
    if not polys_ll:
        return np.ones((len(lats), len(lons)), bool)
    GX, GY = np.meshgrid(lons, lats)
    pts = np.column_stack([GX.ravel(), GY.ravel()])
    inside = np.zeros(pts.shape[0], bool)
    for p in polys_ll:
        if len(p) < 3:
            continue
        try:
            inside |= Path(p).contains_points(pts)
        except Exception:
            pass
    m = inside.reshape(GX.shape)
    # dilate by one cell in each direction (keeps coastal cells)
    d = m.copy()
    d[1:, :] |= m[:-1, :]; d[:-1, :] |= m[1:, :]
    d[:, 1:] |= m[:, :-1]; d[:, :-1] |= m[:, 1:]
    return d


def region_risks_for_frame(risks, lats, lons):
    """{region: {hazard: peak_risk_int}} using the land-masked risk grids."""
    out = {}
    for name, lo0, lo1, la0, la1 in REGIONS:
        lm = (lats >= la0) & (lats <= la1)
        om = (lons >= lo0) & (lons <= lo1)
        if not lm.any() or not om.any():
            out[name] = {h: 0 for h in DATA_HAZARDS}
            continue
        sl = np.ix_(lm, om)
        out[name] = {h: int(risks[h][sl].max()) for h in DATA_HAZARDS}
    return out


def city_region(clat, clon):
    for name, lo0, lo1, la0, la1 in REGIONS:
        if lo0 <= clon <= lo1 and la0 <= clat <= la1:
            return name
    return ""


def areas_for_frame(risks, lats, lons):
    """Approximate land area (km²) at risk level >= 3, per hazard."""
    dlat = abs(float(lats[1] - lats[0])) if len(lats) > 1 else 0.25
    dlon = abs(float(lons[1] - lons[0])) if len(lons) > 1 else 0.25
    # cell area varies with latitude
    cell = (dlat * 111.32) * (dlon * 111.32 * np.cos(np.radians(lats)))[:, None]
    return {h: int(round(float((cell * (risks[h] >= 3)).sum()))) for h in DATA_HAZARDS}


def build_warnings(frames):
    """
    Collapse the per-frame regional risk grids into a list of warning events:
    contiguous runs (single-frame gaps bridged) where a region reaches ENH+.
    Each warning carries its peak level, time window, peak frame and the
    capital cities inside that region that are also affected.
    """
    n = len(frames)
    city_by_region = {}
    for cname, clat, clon in CITIES:
        city_by_region.setdefault(city_region(clat, clon), []).append(cname)

    warns = []
    for hz in HAZARDS:
        for rname, *_ in REGIONS:
            series = [frames[i]["region_risks"][rname][hz] for i in range(n)]
            i = 0
            while i < n:
                if series[i] < 3:
                    i += 1
                    continue
                j = i
                gap = 0
                while j + 1 < n and (series[j + 1] >= 3 or (gap == 0 and
                                     j + 2 < n and series[j + 2] >= 3)):
                    # bridge a lone sub-threshold frame, but never two in a row
                    gap = 1 if series[j + 1] < 3 else 0
                    j += 1
                run = range(i, j + 1)
                lvl = max(series[k] for k in run)
                peak = max(run, key=lambda k: series[k])
                cities = [c for c in city_by_region.get(rname, [])
                          if any(frames[k]["city_risks"][c][hz] >= 3 for k in run)]
                warns.append({
                    "h": hz, "lvl": lvl, "reg": rname,
                    "s": i, "e": j, "p": peak, "cities": cities,
                    "area": max(frames[k]["areas"][hz] for k in run),
                })
                i = j + 1
    warns.sort(key=lambda w: (-w["lvl"], w["s"]))
    return warns


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
def make_html(frames, run_label, timestamp, coarse_lats, coarse_lons,
              run_iso="", warnings=None):
    """
    frames: list of dicts: { f, utc, local, day, images:{hazard:b64}, city_risks, coarse }
    """
    images = [fr["images"] for fr in frames]
    meta   = [{"f": fr["f"], "utc": fr["utc"], "local": fr["local"],
               "day": fr["day"], "iso": fr["iso"]}
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
        "__WARNINGS__":     json.dumps(warnings or [], separators=(",", ":")),
        "__REGION_RISKS__": json.dumps([fr["region_risks"] for fr in frames],
                                       separators=(",", ":")),
        "__AREAS__":        json.dumps([fr["areas"] for fr in frames],
                                       separators=(",", ":")),
        "__RUN_ISO__":      run_iso,
        "__REGIONS__":      json.dumps([r[0] for r in REGIONS]),
    }
    for k, v in repl.items():
        tpl = tpl.replace(k, v)
    return tpl


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no,viewport-fit=cover">
<meta name="theme-color" content="#050a10">
<meta name="description" content="Hour-by-hour severe weather risk for Australia with live radar.">
<title>Ausweather · Severe Weather &amp; Live Radar</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
:root{
  --bg:#050a10; --bg2:#08111a;
  --glass:rgba(9,17,26,.82); --glass-solid:rgba(9,17,26,.97);
  --line:#15324a; --line2:#1e405a;
  --txt:#eaf6ff; --txt2:#9fc4d8; --txt3:#5a8faa; --txt4:#3f6377;
  --acc:#36c5e0; --acc2:#5fe0f5;
  --r-sm:8px; --r-md:12px; --r-lg:16px;
  --shadow:0 10px 40px rgba(0,0,0,.55);
  --safe-b:env(safe-area-inset-bottom,0px);
  --ease:cubic-bezier(.22,.8,.36,1);
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;overflow:hidden;background:var(--bg);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
  color:var(--txt);-webkit-tap-highlight-color:transparent;
  -webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility}
#map{position:fixed;inset:0;background:var(--bg);z-index:1}
.leaflet-container{background:var(--bg);font-family:inherit}
.risk-overlay{image-rendering:auto;transition:opacity .35s var(--ease);will-change:opacity}
button{font-family:inherit}
:focus-visible{outline:2px solid var(--acc);outline-offset:2px}
.glass{background:var(--glass);border:1px solid var(--line);
  -webkit-backdrop-filter:blur(10px) saturate(1.2);backdrop-filter:blur(10px) saturate(1.2)}

/* ── Top bar ── */
.topbar{position:fixed;top:0;left:0;right:0;z-index:500;display:flex;align-items:flex-start;
  gap:12px;padding:10px 14px;pointer-events:none;
  background:linear-gradient(180deg,rgba(5,10,16,.94) 55%,rgba(5,10,16,0))}
.brand{font-size:20px;font-weight:800;letter-spacing:2px;text-transform:uppercase;
  text-shadow:0 2px 10px rgba(0,0,0,.6);white-space:nowrap;line-height:1}
.brand em{color:var(--acc);font-style:normal}
.brand small{display:block;font-size:8px;letter-spacing:2.6px;color:var(--txt3);font-weight:600;margin-top:3px}
.runchip{display:inline-flex;align-items:center;gap:5px;margin-top:6px;pointer-events:auto;
  font-size:8.5px;letter-spacing:1.2px;color:var(--txt3);background:rgba(8,16,24,.7);
  border:1px solid var(--line);border-radius:20px;padding:3px 8px;cursor:help}
.runchip .rdot{width:5px;height:5px;border-radius:50%;background:#2fd07a;box-shadow:0 0 6px #2fd07a}
.runchip.stale{color:#ffbf5a;border-color:#5a4420}
.runchip.stale .rdot{background:#ffbf5a;box-shadow:0 0 6px #ffbf5a}
.legbox{margin-left:auto;pointer-events:auto;border-radius:var(--r-md);padding:7px 10px;flex-shrink:0}
.legbox .lg-title{font-size:8px;letter-spacing:2px;color:var(--txt3);margin-bottom:5px}
.lg-scale{display:flex;gap:0}
.lg-item{font-size:8px;letter-spacing:.4px;color:var(--txt2);text-align:center;width:36px}
.lg-sw{display:block;height:7px;border-radius:2px;margin-bottom:3px}
.radar-legend{display:none}
.radar-legend.show{display:block}
.radar-grad{height:7px;width:190px;border-radius:2px;margin-bottom:3px}
.radar-grad-labels{display:flex;justify-content:space-between;font-size:8px;color:var(--txt2);letter-spacing:.4px}

/* ── Alert banner ── */
.alert{position:fixed;top:56px;left:50%;transform:translateX(-50%) translateY(-6px);z-index:480;
  display:none;align-items:center;gap:8px;font-size:11px;font-weight:700;letter-spacing:.9px;
  padding:7px 15px;border-radius:22px;border:1px solid;cursor:pointer;
  -webkit-backdrop-filter:blur(10px);backdrop-filter:blur(10px);white-space:nowrap;
  max-width:92vw;overflow:hidden;text-overflow:ellipsis;opacity:0;
  transition:opacity .3s var(--ease),transform .3s var(--ease)}
.alert.show{display:flex;opacity:1;transform:translateX(-50%) translateY(0)}
.alert-3{background:rgba(255,226,59,.13);border-color:rgba(255,226,59,.45);color:#ffe23b}
.alert-4{background:rgba(255,106,31,.16);border-color:rgba(255,106,31,.55);color:#ff9040}
.alert-5{background:rgba(192,38,211,.2);border-color:rgba(192,38,211,.6);color:#e07df0;
  animation:sev 2.4s ease-in-out infinite}
@keyframes sev{0%,100%{box-shadow:0 0 0 0 rgba(192,38,211,0)}50%{box-shadow:0 0 18px 2px rgba(192,38,211,.35)}}
.alert .dot{width:6px;height:6px;border-radius:50%;background:currentColor;animation:blink 1.2s infinite;flex:0 0 auto}
.alert .more{font-size:9px;font-weight:800;letter-spacing:.6px;opacity:.75;
  border:1px solid currentColor;border-radius:9px;padding:0 5px;flex:0 0 auto}
@keyframes blink{0%,100%{opacity:.3}50%{opacity:1}}

/* ── Left icon buttons ── */
.icon-btn{position:relative;pointer-events:auto;width:38px;height:38px;border-radius:var(--r-md);
  border:1px solid var(--line);background:var(--glass);color:var(--txt2);font-size:15px;cursor:pointer;
  display:flex;align-items:center;justify-content:center;
  -webkit-backdrop-filter:blur(10px);backdrop-filter:blur(10px);
  transition:border-color .15s,color .15s,background .15s,transform .12s var(--ease);user-select:none}
.icon-btn:hover{border-color:var(--acc);color:#cdeefb}
.icon-btn:active{transform:scale(.93)}
.icon-btn.on{border-color:var(--acc);color:var(--acc);background:rgba(0,40,60,.7)}
.icon-btn .badge{position:absolute;top:-5px;right:-5px;min-width:16px;height:16px;padding:0 4px;
  border-radius:9px;background:#ff6a1f;color:#150800;font-size:9px;font-weight:800;line-height:16px;
  text-align:center;border:1.5px solid var(--bg);display:none}
.icon-btn .badge.show{display:block}
.icon-btn .badge.sev{background:#d43bd2;color:#fff;animation:pulse 1.8s infinite}
@keyframes pulse{0%,100%{transform:scale(1)}50%{transform:scale(1.16)}}
.left-stack{position:fixed;top:96px;left:14px;z-index:500;display:flex;flex-direction:column;gap:8px}
.zoom-pair{display:flex;flex-direction:column}
.zoom-pair .icon-btn:first-child{border-radius:var(--r-md) var(--r-md) 4px 4px}
.zoom-pair .icon-btn:last-child{border-radius:4px 4px var(--r-md) var(--r-md);margin-top:-1px}

/* ── Radar badge ── */
.radar-flag{position:fixed;left:50%;transform:translateX(-50%);top:56px;z-index:450;
  display:none;align-items:center;gap:6px;font-size:10px;letter-spacing:1.1px;color:#7fe3ff;
  background:var(--glass);border:1px solid #15506a;border-radius:22px;
  padding:6px 13px;-webkit-backdrop-filter:blur(10px);backdrop-filter:blur(10px);pointer-events:none}
.radar-flag.show{display:flex}
.radar-flag.stacked{top:94px}   /* sit below the alert banner instead of under it */
.radar-flag .dot{width:6px;height:6px;border-radius:50%;background:var(--acc);animation:blink 1.4s infinite}
.radar-flag.err{color:#ff8f8f;border-color:#5a2020}
.radar-flag.err .dot{background:#ff6a6a;animation:none}
.radar-flag .rtime{margin-left:4px;opacity:.8}
.radar-flag .rage{opacity:.55;margin-left:2px}
.spin{width:9px;height:9px;border-radius:50%;border:1.5px solid rgba(127,227,255,.3);
  border-top-color:#7fe3ff;animation:spin .7s linear infinite;display:none}
.spin.show{display:block}
@keyframes spin{to{transform:rotate(360deg)}}

/* ── Side panels ── */
.panel{position:fixed;right:14px;bottom:180px;z-index:600;width:232px;
  background:var(--glass-solid);border:1px solid #163450;border-radius:var(--r-lg);
  padding:14px 14px 11px;-webkit-backdrop-filter:blur(14px);backdrop-filter:blur(14px);
  box-shadow:var(--shadow);opacity:0;transform:translateY(8px) scale(.98);pointer-events:none;
  transition:opacity .2s var(--ease),transform .2s var(--ease)}
.panel.show{opacity:1;transform:none;pointer-events:auto}
.panel h4{font-size:9px;letter-spacing:2px;color:var(--txt3);margin-bottom:11px;text-transform:uppercase}
.prow{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:11px}
.prow:last-child{margin-bottom:0}
.prow label{font-size:12px;color:#cfe6f2}
.sw{position:relative;width:38px;height:21px;border-radius:11px;background:#1a3145;
  cursor:pointer;transition:background .2s;flex:0 0 auto}
.sw.on{background:#0c6d8c}
.sw::after{content:"";position:absolute;top:2px;left:2px;width:17px;height:17px;border-radius:50%;
  background:#cdeefb;transition:left .2s var(--ease),background .2s}
.sw.on::after{left:19px;background:#7fe3ff}
.slider-sm{width:82px;accent-color:var(--acc)}
.pdiv{height:1px;background:#163450;margin:11px 0 12px}
.psel{background:#0e2436;color:#cfe6f2;border:1px solid var(--line);border-radius:7px;
  font-size:11px;padding:4px 6px;cursor:pointer;max-width:112px}
.pnote{font-size:9px;color:var(--txt4);letter-spacing:.3px;line-height:1.5;margin-top:9px}

/* ── Warnings panel ── */
.warnp{position:fixed;right:14px;bottom:180px;z-index:600;width:296px;max-height:min(60vh,440px);
  display:flex;flex-direction:column;
  background:var(--glass-solid);border:1px solid #163450;border-radius:var(--r-lg);
  -webkit-backdrop-filter:blur(14px);backdrop-filter:blur(14px);box-shadow:var(--shadow);
  opacity:0;transform:translateY(8px) scale(.98);pointer-events:none;
  transition:opacity .2s var(--ease),transform .2s var(--ease)}
.warnp.show{opacity:1;transform:none;pointer-events:auto}
.warnp-hd{display:flex;align-items:center;justify-content:space-between;padding:13px 14px 10px;
  border-bottom:1px solid #163450;flex:0 0 auto}
.warnp-hd h4{font-size:9px;letter-spacing:2px;color:var(--txt3);text-transform:uppercase}
.warnp-hd .cnt{font-size:9px;color:var(--txt4);letter-spacing:1px}
.warnp-body{overflow-y:auto;padding:10px;display:flex;flex-direction:column;gap:8px;
  scrollbar-width:thin;scrollbar-color:#1e405a transparent}
.warnp-body::-webkit-scrollbar{width:6px}
.warnp-body::-webkit-scrollbar-thumb{background:#1e405a;border-radius:3px}
.wcard{border:1px solid var(--line);border-left-width:3px;border-radius:10px;padding:9px 11px;
  background:rgba(10,20,30,.7);cursor:pointer;transition:transform .12s var(--ease),background .15s,border-color .15s}
.wcard:hover{background:rgba(16,32,46,.9);transform:translateX(-2px)}
.wcard .wtop{display:flex;align-items:center;gap:7px;margin-bottom:5px}
.wlvl{font-size:9px;font-weight:800;letter-spacing:1px;padding:2px 6px;border-radius:5px;color:#0a1016}
.whz{font-size:12px;font-weight:700;color:var(--txt)}
.wreg{margin-left:auto;font-size:9px;letter-spacing:1.4px;color:var(--txt3);font-weight:700}
.wtime{font-size:10px;color:var(--txt2);letter-spacing:.2px}
.wsub{font-size:9.5px;color:var(--txt4);margin-top:3px;letter-spacing:.2px}
.wempty{text-align:center;color:var(--txt4);font-size:11px;line-height:1.7;padding:26px 14px}
.wempty .big{font-size:22px;display:block;margin-bottom:6px;opacity:.5}

/* ── Bottom HUD ── */
.bottom{position:fixed;left:0;right:0;bottom:0;z-index:500;
  padding:12px 14px calc(10px + var(--safe-b));
  background:linear-gradient(0deg,rgba(5,10,16,.97) 60%,rgba(5,10,16,0));pointer-events:none}
.bottom>*{pointer-events:auto}
.hazard-bar{display:flex;justify-content:center;margin-bottom:10px}
.hsel{display:flex;gap:4px;background:var(--glass);border:1px solid var(--line);border-radius:26px;
  padding:4px;-webkit-backdrop-filter:blur(10px);backdrop-filter:blur(10px);
  max-width:96vw;overflow-x:auto;overscroll-behavior-x:contain;scrollbar-width:none;
  -webkit-overflow-scrolling:touch}
.hsel::-webkit-scrollbar{display:none}
@media(max-width:640px){
  /* fade the edges so it reads as scrollable */
  .hsel{-webkit-mask-image:linear-gradient(90deg,transparent 0,#000 14px,#000 calc(100% - 14px),transparent 100%);
        mask-image:linear-gradient(90deg,transparent 0,#000 14px,#000 calc(100% - 14px),transparent 100%)}
}
.hbtn{position:relative;border:0;background:none;color:#7ba6bd;font-size:12px;font-weight:700;
  letter-spacing:.5px;padding:7px 13px;border-radius:20px;cursor:pointer;white-space:nowrap;flex:0 0 auto;
  transition:color .15s,background .2s var(--ease)}
.hbtn:hover{color:#cdeefb}
.hbtn.active{background:linear-gradient(180deg,#0a4d66,#073549);color:var(--acc2);
  box-shadow:0 0 0 1px #1c6f8c inset}
.hbtn.radar.active{background:linear-gradient(180deg,#0a3a66,#062a49);color:#7fb6ff;
  box-shadow:0 0 0 1px #1c5a8c inset}
.hbtn .pk{position:absolute;top:3px;right:5px;width:5px;height:5px;border-radius:50%;display:none}
.hbtn .pk.show{display:block}

.scrub-row{max-width:1100px;margin:0 auto;display:flex;align-items:center;gap:10px}
.pbtn{flex:0 0 auto;width:42px;height:42px;border-radius:50%;border:1.5px solid rgba(54,197,224,.55);
  background:rgba(54,197,224,.14);color:var(--acc2);font-size:15px;cursor:pointer;display:flex;
  align-items:center;justify-content:center;transition:background .15s,transform .12s var(--ease)}
.pbtn:hover{background:rgba(54,197,224,.3)}
.pbtn:active{transform:scale(.92)}
.sbtn{flex:0 0 auto;width:31px;height:31px;border-radius:50%;border:1px solid #1b3c54;
  background:rgba(8,16,24,.7);color:var(--txt2);cursor:pointer;font-size:11px;display:flex;
  align-items:center;justify-content:center;transition:border-color .15s,color .15s,transform .12s}
.sbtn:hover{border-color:var(--acc);color:#cdeefb}
.sbtn:active{transform:scale(.9)}
.tl{flex:1 1 auto;min-width:0}
.tl-top{display:flex;justify-content:space-between;align-items:baseline;gap:10px;margin-bottom:4px}
.tl-time{font-size:13px;font-weight:700;letter-spacing:.3px;color:var(--txt);
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.tl-time b{color:var(--acc2)}
.tl-time .tag{font-size:9px;font-weight:800;letter-spacing:1px;padding:2px 6px;border-radius:5px;
  margin-right:7px;background:#0c4a63;color:#7fe3ff;vertical-align:middle}
.tl-time .tag.obs{background:#10406a;color:#7fb6ff}
.tl-time .tag.now{background:#0d5a3a;color:#5ff0a8}
.tl-time .sm{font-size:9px;color:var(--txt3);margin-left:5px;letter-spacing:.6px}
.tl-meta{display:flex;gap:10px;align-items:center;flex:0 0 auto}
.tl-utc{font-size:10px;color:var(--txt3);letter-spacing:.4px;white-space:nowrap}
.speed-wrap{display:flex;align-items:center;gap:5px;font-size:9px;color:var(--txt4)}
.speed-wrap input{width:54px;accent-color:var(--acc)}

/* risk profile strip */
.strip{position:relative;display:flex;gap:1px;height:8px;margin:3px 0 1px;border-radius:4px;
  overflow:hidden;cursor:pointer}
.seg{flex:1 1 0;background:#13202f;transition:background .25s,transform .15s var(--ease)}
.seg.cur{transform:scaleY(1.8)}
.seg.rc{background:#12324a}
.seg.rc.nc{background:#2a2350}
.nowline{position:absolute;top:-2px;bottom:-2px;width:2px;background:#5ff0a8;
  box-shadow:0 0 6px rgba(95,240,168,.8);pointer-events:none;display:none;border-radius:1px}
.nowline.show{display:block}

input[type=range]{-webkit-appearance:none;appearance:none;width:100%;height:5px;border-radius:3px;
  background:#0e2436;outline:none;margin:6px 0 4px;cursor:pointer}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;width:15px;height:15px;
  border-radius:50%;background:var(--acc2);border:2px solid #04222e;
  box-shadow:0 0 8px rgba(95,224,245,.65);cursor:pointer}
input[type=range]::-moz-range-thumb{width:15px;height:15px;border-radius:50%;background:var(--acc2);
  border:2px solid #04222e;cursor:pointer}
.dayticks{display:flex;justify-content:space-between;margin-top:2px}
.dayticks.hide{display:none}
.dt{font-size:8px;letter-spacing:1px;color:var(--txt4);cursor:pointer;padding:3px 4px;
  border-radius:6px;text-align:center;transition:color .15s,background .15s}
.dt:hover{color:#8ac0d0;background:rgba(54,197,224,.08)}
.dt.on{color:var(--acc2);font-weight:700}
.dt-label{display:block}
.dh-dots{display:flex;justify-content:center;gap:2px;margin-top:3px}
.dh-dot{width:7px;height:5px;border-radius:1.5px;flex:0 0 auto;opacity:.88}

/* ── Digest panel ── */
.digest{position:fixed;left:50%;transform:translateX(-50%) translateY(8px) scale(.98);bottom:176px;z-index:600;
  background:var(--glass-solid);border:1px solid #163450;border-radius:var(--r-lg);
  padding:13px 15px 11px;-webkit-backdrop-filter:blur(14px);backdrop-filter:blur(14px);
  box-shadow:var(--shadow);min-width:312px;max-width:94vw;opacity:0;pointer-events:none;
  transition:opacity .2s var(--ease),transform .2s var(--ease)}
.digest.show{opacity:1;transform:translateX(-50%);pointer-events:auto}
.digest h4{font-size:9px;letter-spacing:2px;color:var(--txt3);margin-bottom:9px;text-transform:uppercase}
.dg-table{border-collapse:separate;border-spacing:2px;width:100%}
.dg-table td,.dg-table th{padding:3px 6px;font-size:9px;text-align:center;border-radius:4px}
.dg-table th{color:var(--txt3);letter-spacing:1px;font-weight:600;padding-bottom:4px}
.dg-table td.lbl{text-align:left;color:var(--txt2);font-size:10px;padding-right:10px;white-space:nowrap}
.dg-table td.cell{font-weight:800;letter-spacing:.5px;font-size:9px;min-width:40px;cursor:pointer;
  transition:transform .12s var(--ease),box-shadow .15s}
.dg-table td.cell:hover{transform:scale(1.07);box-shadow:0 0 0 1px rgba(255,255,255,.35)}
.dg-foot{font-size:9px;color:var(--txt4);margin-top:8px;letter-spacing:.3px}

/* ── City markers ── */
.city-icon{background:none!important;border:none!important;overflow:visible!important}
.city-pin{display:flex;flex-direction:column;align-items:center;cursor:pointer}
.city-dot{width:9px;height:9px;border-radius:50%;border:2px solid var(--bg);
  background:var(--c,#2a3a48);box-shadow:0 0 7px var(--c,transparent);transition:background .3s,box-shadow .3s}
.city-pin.hot .city-dot{animation:cityPulse 1.8s ease-in-out infinite}
@keyframes cityPulse{0%,100%{box-shadow:0 0 5px var(--c)}50%{box-shadow:0 0 14px 3px var(--c)}}
.city-label{margin-top:2px;text-align:center;background:rgba(5,10,16,.74);
  border:1px solid rgba(255,255,255,.1);border-radius:6px;padding:2px 5px;
  line-height:1.25;white-space:nowrap;-webkit-backdrop-filter:blur(4px);backdrop-filter:blur(4px)}
.city-name{display:block;font-size:8px;font-weight:700;letter-spacing:.5px;color:var(--txt2)}
.city-risk{display:block;font-size:8px;font-weight:800;color:var(--c,var(--txt3))}
.me-icon{background:none!important;border:none!important}
.me-dot{width:14px;height:14px;border-radius:50%;background:#5ff0a8;border:2px solid #04180e;
  box-shadow:0 0 0 4px rgba(95,240,168,.22),0 0 12px rgba(95,240,168,.7)}

/* ── Popup ── */
.leaflet-popup-content-wrapper.risk-popup-wrap{
  background:rgba(9,17,26,.97);border:1px solid var(--line2);border-radius:12px;
  color:var(--txt);padding:0;box-shadow:0 8px 30px rgba(0,0,0,.55)}
.leaflet-popup-tip{background:rgba(9,17,26,.97)!important}
.leaflet-popup-content{margin:0!important}
.leaflet-popup-close-button{color:var(--txt3)!important;top:5px!important;right:5px!important}
.popup-inner{padding:11px 14px;min-width:170px}
.popup-hdr{font-size:9px;letter-spacing:1.5px;color:var(--txt3);margin-bottom:8px}
.popup-hdr b{color:var(--txt2);letter-spacing:1px}
.popup-row{display:flex;justify-content:space-between;align-items:center;gap:14px;
  padding:3px 0;border-bottom:1px solid #142032;font-size:12px}
.popup-row.tot{border-bottom:none;border-top:1px solid var(--line2);margin-top:3px;padding-top:5px}
.popup-row:last-child{border-bottom:none}
.popup-row .ph{color:var(--txt2)}
.popup-row .pv{font-weight:800;font-size:11px}
.popup-when{font-size:9px;color:var(--txt4);margin-top:7px;letter-spacing:.3px;line-height:1.5}

/* ── Toast ── */
.toast{position:fixed;left:50%;bottom:calc(160px + var(--safe-b));transform:translateX(-50%) translateY(10px);
  z-index:900;background:var(--glass-solid);border:1px solid var(--line2);border-radius:22px;
  padding:8px 16px;font-size:11px;letter-spacing:.5px;color:#cdeefb;box-shadow:var(--shadow);
  opacity:0;pointer-events:none;transition:opacity .22s var(--ease),transform .22s var(--ease)}
.toast.show{opacity:1;transform:translateX(-50%)}

/* ── Info modal ── */
.modal{position:fixed;inset:0;z-index:1000;display:none;align-items:center;justify-content:center;
  background:rgba(2,6,10,.74);-webkit-backdrop-filter:blur(5px);backdrop-filter:blur(5px);padding:20px}
.modal.show{display:flex;animation:fadein .2s ease}
@keyframes fadein{from{opacity:0}to{opacity:1}}
.card{max-width:460px;background:#0a1622;border:1px solid #173552;border-radius:18px;padding:22px;
  box-shadow:0 20px 60px rgba(0,0,0,.6);max-height:88vh;overflow-y:auto}
.card h2{font-size:17px;letter-spacing:1px;margin-bottom:3px}
.card h2 em{color:var(--acc);font-style:normal}
.card .sub{font-size:9px;letter-spacing:2.5px;color:var(--txt3);margin-bottom:14px}
.card p{font-size:12px;line-height:1.68;color:#bcd6e6;margin-bottom:10px}
.card a{color:var(--acc2);text-decoration:none}
.card kbd{background:#13283a;border:1px solid var(--line2);border-radius:4px;padding:0 5px;
  font-size:11px;color:#cdeefb;font-family:inherit}
.klist{display:grid;grid-template-columns:auto 1fr;gap:5px 10px;font-size:11px;color:#a9c6d8;margin:2px 0 12px}
.card .close{margin-top:6px;width:100%;padding:10px;border:0;border-radius:10px;
  background:#0c6d8c;color:var(--txt);font-size:13px;font-weight:700;cursor:pointer;transition:background .15s}
.card .close:hover{background:#0d7ea1}
.disc{color:#7e98a8;font-size:11px;border-left:2px solid #234;padding-left:10px}

.leaflet-control-zoom{display:none}
.leaflet-control-attribution{font-size:9px!important;background:rgba(5,10,16,.55)!important;color:#456!important}
.leaflet-control-attribution a{color:var(--txt3)!important}
.leaflet-control-scale-line{background:rgba(5,10,16,.5)!important;border-color:#2a4a60!important;
  color:#8ab!important;font-size:9px!important}

@media(max-width:780px){.legbox{display:none}}
@media(max-width:640px){
  .brand{font-size:16px}
  .left-stack{top:88px;left:10px;gap:7px}
  .icon-btn{width:36px;height:36px}
  .panel,.warnp{right:8px;left:8px;width:auto;bottom:186px}
  .warnp{max-height:52vh}
  .hbtn{padding:7px 11px;font-size:11px}
  .speed-wrap{display:none}
  .digest{bottom:182px;min-width:0;width:calc(100vw - 16px)}
  .dg-table td.lbl{font-size:9px}
}
@media(prefers-reduced-motion:reduce){
  *{animation-duration:.001ms!important;animation-iteration-count:1!important;
    transition-duration:.001ms!important}
}
</style>
</head>
<body>
<div id="map"></div>

<!-- Top bar -->
<div class="topbar">
  <div>
    <div class="brand">Aus<em>weather</em><small>SEVERE WEATHER · LIVE RADAR</small></div>
    <div class="runchip" id="runChip" title="Model run"><span class="rdot"></span><span id="runAge">—</span></div>
  </div>
  <div class="legbox glass">
    <div id="riskLegend">
      <div class="lg-title">RISK SCALE</div>
      <div class="lg-scale">__LEGEND__</div>
    </div>
    <div class="radar-legend" id="radarLegend">
      <div class="lg-title">RADAR · PRECIP INTENSITY</div>
      <div class="radar-grad" id="radarGrad"></div>
      <div class="radar-grad-labels"><span>LIGHT</span><span>MOD</span><span>HEAVY</span><span>INTENSE</span></div>
    </div>
  </div>
</div>

<!-- Alert banner -->
<div class="alert" id="alert" role="button" tabindex="0" title="Open warnings">
  <span class="dot"></span><span id="alertTxt"></span><span class="more" id="alertMore"></span>
</div>

<!-- Left controls -->
<div class="left-stack">
  <button class="icon-btn" id="warnBtn" title="Warnings (W)" aria-label="Warnings">&#9888;<span class="badge" id="warnBadge"></span></button>
  <button class="icon-btn" id="digestBtn" title="5-day digest (D)" aria-label="Digest">&#9783;</button>
  <button class="icon-btn" id="layerBtn" title="Layers (L)" aria-label="Layers">&#11052;</button>
  <button class="icon-btn" id="locBtn" title="My location" aria-label="My location">&#9678;</button>
  <button class="icon-btn" id="shareBtn" title="Copy share link" aria-label="Share">&#128279;</button>
  <button class="icon-btn" id="fsBtn" title="Fullscreen (F)" aria-label="Fullscreen">&#9974;</button>
  <button class="icon-btn" id="infoBtn" title="About (?)" aria-label="About">&#9432;</button>
  <div class="zoom-pair">
    <button class="icon-btn" id="zoomIn" title="Zoom in" aria-label="Zoom in">+</button>
    <button class="icon-btn" id="zoomOut" title="Zoom out" aria-label="Zoom out">&#8722;</button>
  </div>
</div>

<!-- Warnings panel -->
<div class="warnp" id="warnPanel">
  <div class="warnp-hd"><h4>Active Warnings</h4><span class="cnt" id="warnCount"></span></div>
  <div class="warnp-body" id="warnBody"></div>
</div>

<!-- Daily digest panel -->
<div class="digest" id="digest"></div>

<!-- Radar badge -->
<div class="radar-flag" id="radarFlag">
  <span class="dot" id="radarDot"></span>
  <span class="spin" id="radarSpin"></span>
  <span id="radarMode">RADAR</span>
  <span class="rtime" id="radarTime"></span>
  <span class="rage" id="radarAge"></span>
</div>

<!-- Layers panel -->
<div class="panel" id="panel">
  <h4>Layers &amp; Display</h4>
  <div class="prow"><label>Risk overlay</label><div class="sw on" id="swRisk"></div></div>
  <div class="prow"><label>Risk opacity</label><input type="range" class="slider-sm" id="riskOpac" min="20" max="100" value="85"></div>
  <div class="pdiv"></div>
  <div class="prow"><label>Radar underlay</label><div class="sw on" id="swRadar"></div></div>
  <div class="prow"><label>Satellite IR</label><div class="sw" id="swSatIR"></div></div>
  <div class="prow"><label>Radar opacity</label><input type="range" class="slider-sm" id="radarOpac" min="10" max="100" value="70"></div>
  <div class="prow"><label>Radar colours</label>
    <select class="psel" id="radarScheme"></select></div>
  <div class="prow"><label>Smoothing</label><div class="sw on" id="swSmooth"></div></div>
  <div class="pdiv"></div>
  <div class="prow"><label>Satellite map</label><div class="sw" id="swSatMap"></div></div>
  <div class="prow"><label>City markers</label><div class="sw on" id="swCities"></div></div>
  <div class="prow"><label>Map labels</label><div class="sw on" id="swLabels"></div></div>
  <div class="pnote">Settings are saved on this device.</div>
</div>

<!-- Bottom HUD -->
<div class="bottom">
  <div class="hazard-bar"><div class="hsel" id="hsel"></div></div>
  <div class="scrub-row">
    <button class="pbtn" id="playBtn" title="Play / Pause (Space)">&#9654;</button>
    <button class="sbtn" id="prevBtn" title="Previous (&larr;)">&#9664;</button>
    <div class="tl">
      <div class="tl-top">
        <div class="tl-time" id="timeMain">—</div>
        <div class="tl-meta">
          <span class="tl-utc" id="timeUtc"></span>
          <span class="speed-wrap"><span>SPEED</span>
            <input type="range" id="speedCtrl" min="1" max="10" value="5" title="Playback speed"></span>
        </div>
      </div>
      <div class="strip" id="strip"><div class="nowline" id="nowline"></div></div>
      <input type="range" id="scrub" min="0" max="__NFRAMES__" value="0" aria-label="Timeline">
      <div class="dayticks" id="dayticks"></div>
      <div class="dayticks hide" id="radarticks"></div>
    </div>
    <button class="sbtn" id="nextBtn" title="Next (&rarr;)">&#9654;</button>
  </div>
</div>

<!-- Toast -->
<div class="toast" id="toast"></div>

<!-- Info modal -->
<div class="modal" id="modal">
  <div class="card">
    <h2>Aus<em>weather</em></h2>
    <div class="sub">5-DAY SEVERE WEATHER OUTLOOK</div>
    <p>Hour-by-hour severe weather risk for Australia — <b>wind, hail, flood and tornado</b> — derived from the
       NOAA GFS model, plus a live <b>radar</b> loop and infrared satellite.</p>
    <p>Pick a hazard or <b>Max</b> (overall threat), then scrub the timeline. The coloured bar under the
       timeline is the risk profile for the selected hazard — click it or a day label to jump.
       Click anywhere on the map (or a city) for a local breakdown.</p>
    <p>The <b>&#9888;</b> button lists every <b>ENH or higher</b> risk window in the next five days, by region,
       with its peak time. Click a warning to jump straight to it. Risks are computed over
       <b>land only</b>, so offshore convection never raises a false warning.</p>
    <p>Tap <b>📡 RADAR</b> for the live loop: 12 past frames plus 30&nbsp;min of nowcast, auto-refreshing
       every five minutes. Radar colours, smoothing and opacity are all in the <b>&#11052;</b> layers panel.</p>
    <div class="klist">
      <kbd>Space</kbd><span>play / pause</span>
      <kbd>&larr; &rarr;</kbd><span>step one frame</span>
      <kbd>Home</kbd> <span>first frame · <kbd>End</kbd> last frame</span>
      <kbd>1</kbd>–<kbd>5</kbd><span>switch hazard · <kbd>0</kbd> radar</span>
      <kbd>W</kbd><span>warnings · <kbd>D</kbd> digest · <kbd>L</kbd> layers</span>
      <kbd>R</kbd><span>refresh radar · <kbd>F</kbd> fullscreen</span>
    </div>
    <p class="disc">Risk levels are computed from raw model fields and are
       <b>not official warnings</b>. For authoritative forecasts and warnings visit
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
const REGION_RISKS= __REGION_RISKS__;
const REGIONS     = __REGIONS__;
const AREAS       = __AREAS__;
const WARNINGS    = __WARNINGS__;
const COARSE      = __COARSE__;
const COARSE_LATS = __COARSE_LATS__;
const COARSE_LONS = __COARSE_LONS__;
const COARSE_SHAPE= __COARSE_SHAPE__;
const PROFILE     = __PROFILE__;       // {hazard:[peakPerFrame]}
const RUN_ISO     = "__RUN_ISO__";
const N = META.length;
const REAL_HAZARDS = HAZARDS.filter(h=>h!=='Max');
const MODES = HAZARDS.concat(['Radar']);
const LAT0=BOUNDS[0][0], LON0=BOUNDS[0][1], LAT1=BOUNDS[1][0], LON1=BOUNDS[1][1];

/* ── Persisted settings ── */
const DEFAULTS={riskOpac:85,radarOpac:70,scheme:6,smooth:true,risk:true,radar:true,
  satIR:false,satMap:false,cities:true,labels:true,speed:5};
let S=Object.assign({},DEFAULTS);
try{S=Object.assign(S,JSON.parse(localStorage.getItem('aw_settings')||'{}'));}catch(e){}
let saveTimer=null;
function save(){clearTimeout(saveTimer);saveTimer=setTimeout(()=>{
  try{localStorage.setItem('aw_settings',JSON.stringify(S));}catch(e){}},400);}

let mode='forecast', curHazard=HAZARDS[0], curFrame=0, rIdx=0;
let playing=false, playTimer=null, playInterval=1400-S.speed*120;
let riskOn=S.risk, riskOpac=S.riskOpac/100, radarOpacity=S.radarOpac/100;
let radarUnderlay=S.radar, satIROn=S.satIR, citiesOn=S.cities;

/* ── Map ── */
const map=L.map('map',{zoomControl:false,attributionControl:true,
  minZoom:3,maxZoom:14,zoomSnap:.25,wheelPxPerZoomLevel:110,
  inertia:true,preferCanvas:true,fadeAnimation:true}).fitBounds(BOUNDS);
map.createPane('radarPane');
map.getPane('radarPane').style.zIndex=350;
map.getPane('radarPane').style.pointerEvents='none';
L.control.scale({imperial:false,position:'bottomleft',maxWidth:110}).addTo(map);

const tOpt={subdomains:'abcd',maxZoom:19,updateWhenIdle:false,keepBuffer:3};
const baseDark=L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png',
  {attribution:'&copy; OSM &copy; CARTO',...tOpt});
const baseSat=L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
  {attribution:'&copy; Esri',maxZoom:19,pane:'tilePane'});
const labels=L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}{r}.png',
  {pane:'markerPane',...tOpt});
(S.satMap?baseSat:baseDark).addTo(map);
if(S.labels)labels.addTo(map);

/* ── Helpers ── */
function esc(s){return String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
function fmtArea(km2){
  if(!km2)return '';
  if(km2>=1e6)return (km2/1e6).toFixed(1)+'M km²';
  if(km2>=1000)return Math.round(km2/1000)+'k km²';
  return km2+' km²';
}
let toastTimer=null;
function toast(msg){
  const t=document.getElementById('toast');
  t.textContent=msg;t.classList.add('show');
  clearTimeout(toastTimer);toastTimer=setTimeout(()=>t.classList.remove('show'),2200);
}
/* AEST (UTC+10, no DST) short label for an epoch-seconds or Date */
function aestParts(d){
  const t=new Date(d.getTime()+10*3600e3);
  const h=t.getUTCHours(),m=t.getUTCMinutes();
  return {h12:(h%12)||12,ap:h<12?'am':'pm',mm:('0'+m).slice(-2),
    day:['Sun','Mon','Tue','Wed','Thu','Fri','Sat'][t.getUTCDay()],
    dnum:t.getUTCDate(),
    mon:['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][t.getUTCMonth()]};
}
function aestClock(d){const p=aestParts(d);return p.h12+(p.mm==='00'?'':':'+p.mm)+p.ap;}
function aestDayClock(d){const p=aestParts(d);return p.day+' '+p.h12+(p.mm==='00'?'':':'+p.mm)+p.ap;}
function frameDate(i){return new Date(META[i].iso);}
function relLabel(ms){
  const m=Math.round(ms/60000);
  if(Math.abs(m)<1)return 'now';
  if(m<0)return (-m<60?(-m)+' min ago':Math.round(-m/60)+' h ago');
  return (m<60?'in '+m+' min':'in '+Math.round(m/60)+' h');
}

/* ── Model run age ── */
(function runAge(){
  const chip=document.getElementById('runChip'),el=document.getElementById('runAge');
  function tick(){
    if(!RUN_ISO){el.textContent='GFS';return;}
    const age=(Date.now()-new Date(RUN_ISO).getTime())/3600e3;
    const hh=RUN_ISO.slice(11,13);
    el.textContent='GFS '+hh+'Z · '+(age<1?Math.max(0,Math.round(age*60))+' min old':age.toFixed(1)+' h old');
    chip.classList.toggle('stale',age>9);
    chip.title=age>9?'This model run is getting old — a newer GFS run may be available.'
                    :'Model run '+RUN_ISO;
  }
  tick();setInterval(tick,60000);
})();

/* ── Risk overlay crossfade pair ── */
function dataUri(b64){return 'data:image/png;base64,'+b64;}
let ovA=L.imageOverlay(dataUri(IMAGES[0][curHazard]),BOUNDS,{opacity:.85,className:'risk-overlay',interactive:false}).addTo(map);
let ovB=L.imageOverlay(dataUri(IMAGES[0][curHazard]),BOUNDS,{opacity:0,className:'risk-overlay',interactive:false}).addTo(map);
let ovTop=ovA;

/* Idle-preload the selected hazard's frames so scrubbing/playback never stalls */
let preloadQueue=[],preloadToken=0;
const idle=window.requestIdleCallback||(f=>setTimeout(()=>f({timeRemaining:()=>8}),120));
function preloadHazard(h){
  const token=++preloadToken;
  preloadQueue=[];
  for(let i=0;i<N;i++)preloadQueue.push(i);
  const pump=deadline=>{
    if(token!==preloadToken)return;
    while(preloadQueue.length&&(!deadline||deadline.timeRemaining()>3)){
      const i=preloadQueue.shift();
      const im=new Image();im.decoding='async';im.src=dataUri(IMAGES[i][h]);
    }
    if(preloadQueue.length)idle(pump);
  };
  idle(pump);
}

/* ── Forecast frame ── */
function showForecast(i,animate){
  curFrame=(i+N)%N;
  const uri=dataUri(IMAGES[curFrame][curHazard]);
  const back=(ovTop===ovA)?ovB:ovA;
  back.setUrl(uri);
  if(animate){back.setOpacity(riskOn?riskOpac:0);ovTop.setOpacity(0);ovTop=back;}
  else{ovA.setUrl(uri);ovB.setUrl(uri);ovTop.setOpacity(riskOn?riskOpac:0);back.setOpacity(0);}
  const m=META[curFrame];
  const d=frameDate(curFrame),dt=d.getTime()-Date.now();
  const isNow=Math.abs(dt)<45*60000;
  document.getElementById('timeMain').innerHTML=
    '<span class="tag'+(isNow?' now':'')+'">'+(isNow?'NOW':'FORECAST')+'</span>'+m.local+
    '<span class="sm">'+esc(relLabel(dt))+'</span>';
  document.getElementById('timeUtc').textContent=m.utc;
  document.getElementById('scrub').value=curFrame;
  updateDayticks();updateStripCursor();updateCityMarkers();updateMe();updateAlert();
  updateHazardPeaks();syncHash();
}

/* ── Hazard / mode selector ── */
const hsel=document.getElementById('hsel');
MODES.forEach(h=>{
  const b=document.createElement('button');
  b.className='hbtn'+(h==='Radar'?' radar':'')+(h===curHazard?' active':'');
  b.dataset.h=h;
  b.innerHTML=(ICONS[h]||'')+' '+(h==='Max'?'MAX':h.toUpperCase())+
    (h==='Radar'?'':'<span class="pk" data-pk="'+h+'"></span>');
  b.onclick=()=>selectMode(h);
  hsel.appendChild(b);
});
/* dot on each hazard button = that hazard's risk at the current frame */
function updateHazardPeaks(){
  if(mode==='radar')return;
  REAL_HAZARDS.concat(['Max']).forEach(h=>{
    const el=hsel.querySelector('[data-pk="'+h+'"]');if(!el)return;
    const v=(PROFILE[h]&&PROFILE[h][curFrame])||0;
    el.classList.toggle('show',v>=2);
    el.style.background=RISK_COLORS[v];
  });
}
function highlightMode(h){
  [...hsel.children].forEach(c=>c.classList.toggle('active',c.dataset.h===h));
}
function selectMode(h){
  if(h==='Radar'){if(playing)stop();enterRadar();highlightMode('Radar');return;}
  if(mode==='radar'){stop();exitRadar();}
  mode='forecast';
  curHazard=h;highlightMode(h);
  buildStrip();preloadHazard(h);
  document.getElementById('riskLegend').style.display='';
  document.getElementById('radarLegend').classList.remove('show');
  document.getElementById('dayticks').classList.remove('hide');
  document.getElementById('radarticks').classList.add('hide');
  scrub.max=N-1;
  showForecast(curFrame,false);
}

/* ── Timeline ── */
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
const dayLabels=['TODAY','TMW','DAY 3','DAY 4','DAY 5','DAY 6','DAY 7'];
function dayName(d){return dayLabels[parseInt(d)]||('DAY '+(parseInt(d)+1));}
function framesOfDay(d){return META.reduce((a,m,i)=>{if(String(m.day)===String(d))a.push(i);return a;},[]);}
const dayFrameCache={};
Object.keys(dayFirst).forEach(d=>dayFrameCache[d]=framesOfDay(d));
function buildDayticks(){
  const wrap=document.getElementById('dayticks');
  wrap.innerHTML='';
  Object.keys(dayFirst).forEach(d=>{
    const el=document.createElement('div');el.className='dt';el.dataset.day=d;
    const dots=HAZARDS.map(h=>{
      const mx=dayFrameCache[d].reduce((v,i)=>Math.max(v,(PROFILE[h]&&PROFILE[h][i])||0),0);
      return '<span class="dh-dot" style="background:'+RISK_COLORS[mx]+'" title="'+h+': '+RISK_LABELS[mx]+'"></span>';
    }).join('');
    el.innerHTML='<span class="dt-label">'+dayName(d)+'</span><div class="dh-dots">'+dots+'</div>';
    el.onclick=()=>{if(mode==='radar')return;stop();showForecast(dayFirst[d],true);};
    wrap.appendChild(el);
  });
}
function updateDayticks(){const d=META[curFrame].day;
  document.querySelectorAll('#dayticks .dt').forEach(e=>e.classList.toggle('on',e.dataset.day==d));}
buildDayticks();

/* ── Risk profile strip ── */
const stripEl=document.getElementById('strip');
const nowline=document.getElementById('nowline');
function buildStrip(){
  const prof=PROFILE[curHazard]||[];
  const frag=document.createDocumentFragment();
  for(let i=0;i<N;i++){
    const s=document.createElement('div');s.className='seg';
    s.style.background=RISK_COLORS[prof[i]||0];
    s.title=META[i].utc+' · '+RISK_LABELS[prof[i]||0];
    s.onclick=()=>{stop();showForecast(i,true);};
    frag.appendChild(s);
  }
  stripEl.innerHTML='';stripEl.appendChild(nowline);stripEl.appendChild(frag);
  positionNow();
}
function buildRadarStrip(){
  const set=curSet(),n=set.length;
  const frag=document.createDocumentFragment();
  for(let i=0;i<n;i++){
    const s=document.createElement('div');
    s.className='seg rc'+((!satIROn&&i>=radar.past)?' nc':'');
    s.title=fmtTime(set[i].time);
    s.onclick=()=>{stop();showRadar(i);};
    frag.appendChild(s);
  }
  stripEl.innerHTML='';stripEl.appendChild(nowline);stripEl.appendChild(frag);
  nowline.classList.remove('show');
}
function updateStripCursor(){
  const idx=(mode==='radar')?rIdx:curFrame;
  const kids=stripEl.querySelectorAll('.seg');
  kids.forEach((s,i)=>s.classList.toggle('cur',i===idx));
}
function positionNow(){
  if(mode==='radar'||N<2){nowline.classList.remove('show');return;}
  const t0=frameDate(0).getTime(),t1=frameDate(N-1).getTime(),now=Date.now();
  if(now<t0||now>t1){nowline.classList.remove('show');return;}
  nowline.style.left=(100*(now-t0)/(t1-t0))+'%';
  nowline.classList.add('show');nowline.title='Now';
}
setInterval(positionNow,60000);

/* ── Playback (self-correcting, pauses when tab hidden) ── */
const playBtn=document.getElementById('playBtn');
const speedCtrl=document.getElementById('speedCtrl');
speedCtrl.value=S.speed;
speedCtrl.oninput=function(){S.speed=+this.value;save();
  playInterval=Math.round(1400-this.value*120);if(playing){stop();play();}};
function tickPlay(){
  if(!playing)return;
  const last=(mode==='radar')?(rIdx===radarCount()-1):(curFrame===N-1);
  step(1);
  playTimer=setTimeout(tickPlay,last?playInterval+700:playInterval);
}
function play(){if(playing)return;playing=true;playBtn.innerHTML='&#10074;&#10074;';
  playBtn.title='Pause (Space)';playTimer=setTimeout(tickPlay,playInterval);}
function stop(){playing=false;playBtn.innerHTML='&#9654;';
  playBtn.title='Play (Space)';clearTimeout(playTimer);}
playBtn.onclick=()=>{playing?stop():play();};
document.addEventListener('visibilitychange',()=>{if(document.hidden&&playing)stop();});

/* ── Zoom / fullscreen ── */
document.getElementById('zoomIn').onclick=()=>map.zoomIn();
document.getElementById('zoomOut').onclick=()=>map.zoomOut();
function toggleFs(){
  if(!document.fullscreenElement)document.documentElement.requestFullscreen?.();
  else document.exitFullscreen?.();}
document.getElementById('fsBtn').onclick=toggleFs;

/* ── Share ── */
document.getElementById('shareBtn').onclick=()=>{
  const url=location.href;
  const done=()=>toast('Link copied — opens on this exact frame');
  if(navigator.clipboard&&navigator.clipboard.writeText)
    navigator.clipboard.writeText(url).then(done,()=>toast(url));
  else toast(url);
};

/* ── City markers ── */
function cityIcon(name,risk){
  const c=RISK_COLORS[risk]||'#2a3a48';
  return L.divIcon({className:'city-icon',iconSize:[0,0],iconAnchor:[0,0],
    html:'<div class="city-pin'+(risk>=4?' hot':'')+'" style="--c:'+c+'"><div class="city-dot"></div>'+
      '<div class="city-label" style="--c:'+c+'"><span class="city-name">'+name+'</span>'+
      (risk>0?'<span class="city-risk">'+RISK_LABELS[risk]+'</span>':'')+'</div></div>'});
}
const cityLast={};
const cityMarkers=CITIES.map(([name,lat,lon])=>{
  const mk=L.marker([lat,lon],{icon:cityIcon(name,0),zIndexOffset:100});
  cityLast[name]=0;
  mk.on('click',()=>{
    map.flyTo([lat,lon],Math.max(map.getZoom(),7),{duration:.8});
    if(mode!=='radar')openCityPopup(name,lat,lon);
  });
  return mk;
});
if(citiesOn)cityMarkers.forEach(m=>m.addTo(map));
function updateCityMarkers(){
  if(!citiesOn||mode==='radar'||!CITY_RISKS[curFrame])return;
  CITIES.forEach(([name],i)=>{
    const r=(CITY_RISKS[curFrame][name]||{})[curHazard]||0;
    if(cityLast[name]===r)return;          // avoid rebuilding unchanged icons
    cityLast[name]=r;
    cityMarkers[i].setIcon(cityIcon(name,r));
  });
}

/* ── My location ── */
let meMarker=null,mePos=null;
function updateMe(){
  if(!meMarker||mode==='radar')return;
  meMarker.setTooltipContent(meLabel());
}
function meLabel(){
  if(!mePos)return '';
  const iLat=nearest1d(COARSE_LATS,mePos[0]),iLon=nearest1d(COARSE_LONS,mePos[1]);
  const r=coarseGet(curFrame,curHazard,iLat,iLon);
  return '<b>You</b> · '+curHazard+': <span style="color:'+RISK_COLORS[r]+'">'+RISK_LABELS[r]+'</span>';
}
document.getElementById('locBtn').onclick=()=>{
  if(!navigator.geolocation){toast('Geolocation not available');return;}
  toast('Locating…');
  navigator.geolocation.getCurrentPosition(p=>{
    mePos=[p.coords.latitude,p.coords.longitude];
    if(meMarker)meMarker.setLatLng(mePos);
    else{
      meMarker=L.marker(mePos,{icon:L.divIcon({className:'me-icon',html:'<div class="me-dot"></div>',
        iconSize:[14,14],iconAnchor:[7,7]}),zIndexOffset:400}).addTo(map);
      meMarker.bindTooltip(meLabel(),{direction:'top',offset:[0,-8],className:'me-tip'});
      meMarker.on('click',()=>{if(mode!=='radar')openPointPopup(mePos[0],mePos[1],'MY LOCATION');});
    }
    updateMe();
    map.flyTo(mePos,Math.max(map.getZoom(),8),{duration:1.2});
    document.getElementById('locBtn').classList.add('on');
  },err=>toast('Location unavailable'),{enableHighAccuracy:true,timeout:8000,maximumAge:60000});
};

/* ── Warnings ── */
function warnWindow(w){
  const a=frameDate(w.s),b=frameDate(w.e),p=frameDate(w.p);
  const sameDay=aestParts(a).dnum===aestParts(b).dnum;
  return aestDayClock(a)+' → '+(sameDay?aestClock(b):aestDayClock(b))+
         ' · peaks '+aestClock(p);
}
function activeWarnings(){return WARNINGS;}
function buildWarnPanel(){
  const body=document.getElementById('warnBody');
  const ws=activeWarnings();
  document.getElementById('warnCount').textContent=
    ws.length?(ws.length+(ws.length===1?' event':' events')):'';
  if(!ws.length){
    body.innerHTML='<div class="wempty"><span class="big">✓</span>No enhanced or higher risk<br>'+
      'anywhere in Australia for the next five days.<br>'+
      '<span style="color:#2e4a5c">Marginal and slight risks may still be present —<br>'+
      'check the individual hazard layers.</span></div>';
    return;
  }
  body.innerHTML=ws.map((w,i)=>{
    const c=RISK_COLORS[w.lvl];
    const cities=w.cities&&w.cities.length?w.cities.join(' · '):'';
    const area=fmtArea(w.area);
    const sub=[cities,area].filter(Boolean).join('  ·  ');
    return '<div class="wcard" data-w="'+i+'" style="border-left-color:'+c+'">'+
      '<div class="wtop"><span class="wlvl" style="background:'+c+'">'+RISK_LABELS[w.lvl]+'</span>'+
      '<span class="whz">'+(ICONS[w.h]||'')+' '+esc(w.h)+'</span>'+
      '<span class="wreg">'+esc(w.reg)+'</span></div>'+
      '<div class="wtime">'+esc(warnWindow(w))+'</div>'+
      (sub?'<div class="wsub">'+esc(sub)+'</div>':'')+'</div>';
  }).join('');
  [...body.querySelectorAll('.wcard')].forEach(el=>{
    el.onclick=()=>jumpToWarning(ws[+el.dataset.w]);
  });
}
function jumpToWarning(w){
  stop();
  if(mode==='radar'){exitRadar();}
  curHazard=w.h;highlightMode(w.h);buildStrip();preloadHazard(w.h);
  document.getElementById('riskLegend').style.display='';
  document.getElementById('radarLegend').classList.remove('show');
  document.getElementById('dayticks').classList.remove('hide');
  document.getElementById('radarticks').classList.add('hide');
  scrub.max=N-1;
  showForecast(w.p,true);
  closePanels();
  toast(RISK_LABELS[w.lvl]+' '+w.h.toLowerCase()+' · '+w.reg+' · '+aestClock(frameDate(w.p)));
}
function updateWarnBadge(){
  const ws=activeWarnings();
  const badge=document.getElementById('warnBadge');
  const top=ws.reduce((v,w)=>Math.max(v,w.lvl),0);
  badge.textContent=ws.length>99?'99+':ws.length;
  badge.classList.toggle('show',ws.length>0);
  badge.classList.toggle('sev',top>=5);
  if(top>=4)badge.style.background=RISK_COLORS[top];
  document.getElementById('warnBtn').title=ws.length?
    (ws.length+' warning window'+(ws.length===1?'':'s')+' — peak '+RISK_LABELS[top]):'No ENH+ warnings';
}

/* ── Alert banner (current frame) ── */
function updateAlert(){
  const el=document.getElementById('alert');
  if(mode==='radar'||!REGION_RISKS[curFrame]){el.classList.remove('show');layoutTopBadges();return;}
  const rr=REGION_RISKS[curFrame];
  let max=0,best=[];
  REAL_HAZARDS.forEach(h=>REGIONS.forEach(rg=>{
    const v=(rr[rg]||{})[h]||0;
    if(v>max){max=v;best=[{h:h,r:rg}];}
    else if(v===max&&v>=3)best.push({h:h,r:rg});
  }));
  if(max<3){el.classList.remove('show');layoutTopBadges();return;}
  const hz=best[0].h;
  const regs=[...new Set(best.filter(b=>b.h===hz).map(b=>b.r))];
  const area=fmtArea((AREAS[curFrame]||{})[hz]||0);
  el.className='alert show alert-'+Math.min(max,5);
  document.getElementById('alertTxt').textContent=
    RISK_LABELS[max]+' '+hz.toUpperCase()+' — '+regs.slice(0,4).join(', ')+(area?'  ·  '+area:'');
  const others=best.filter(b=>b.h!==hz).length;
  const more=document.getElementById('alertMore');
  more.textContent=others?('+'+others):'';
  more.style.display=others?'':'none';
  layoutTopBadges();
}
/* keep the radar badge clear of the alert banner */
function layoutTopBadges(){
  radarFlag.classList.toggle('stacked',
    document.getElementById('alert').classList.contains('show'));
}
document.getElementById('alert').onclick=()=>toggleWarn(true);

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
  let a=coarseCache[key];
  if(!a){const s=atob(COARSE[frame][hazard]);
    a=new Uint8Array(s.length);for(let i=0;i<s.length;i++)a[i]=s.charCodeAt(i);coarseCache[key]=a;}
  return a[iLat*COARSE_SHAPE[1]+iLon]||0;
}
function nearest1d(arr,v){let b=0,bd=1e9;for(let i=0;i<arr.length;i++){const d=Math.abs(arr[i]-v);if(d<bd){bd=d;b=i;}}return b;}
/* worst upcoming risk at a point, for popup context */
function pointOutlook(iLat,iLon){
  let best=0,bi=curFrame;
  for(let i=curFrame;i<N;i++){
    const v=coarseGet(i,'Max',iLat,iLon);
    if(v>best){best=v;bi=i;}
  }
  return best>0?{lvl:best,i:bi}:null;
}
function popupHtml(title,rows,iLat,iLon){
  let foot='';
  if(iLat!=null){
    const o=pointOutlook(iLat,iLon);
    if(o&&o.i!==curFrame)
      foot='<div class="popup-when">Peak here: <b style="color:'+RISK_COLORS[o.lvl]+'">'+
        RISK_LABELS[o.lvl]+'</b> '+aestDayClock(frameDate(o.i))+'</div>';
  }
  return '<div class="popup-inner"><div class="popup-hdr">'+title+'</div>'+rows+foot+'</div>';
}
function openCityPopup(name,lat,lon){
  const cr=(CITY_RISKS[curFrame]||{})[name]||{};
  const iLat=nearest1d(COARSE_LATS,lat),iLon=nearest1d(COARSE_LONS,lon);
  L.popup({className:'risk-popup-wrap',maxWidth:230,autoPanPadding:L.point(20,96)})
   .setLatLng([lat,lon])
   .setContent(popupHtml('<b>'+esc(name.toUpperCase())+'</b> · '+aestDayClock(frameDate(curFrame)),
     riskRows(h=>cr[h]||0),iLat,iLon)).openOn(map);
}
function openPointPopup(lat,lon,title){
  const iLat=nearest1d(COARSE_LATS,lat),iLon=nearest1d(COARSE_LONS,lon);
  L.popup({className:'risk-popup-wrap',maxWidth:230,autoPanPadding:L.point(20,96)})
   .setLatLng([lat,lon])
   .setContent(popupHtml((title?'<b>'+esc(title)+'</b> · ':'')+aestDayClock(frameDate(curFrame)),
     riskRows(h=>coarseGet(curFrame,h,iLat,iLon)),iLat,iLon)).openOn(map);
}
map.on('click',e=>{
  if(mode==='radar')return;
  const lat=e.latlng.lat,lon=e.latlng.lng;
  if(lat<LAT0||lat>LAT1||lon<LON0||lon>LON1)return;
  openPointPopup(lat,lon,'');
});

/* ── Panels ── */
const panel=document.getElementById('panel');
const warnPanel=document.getElementById('warnPanel');
const digestEl=document.getElementById('digest');
const layerBtn=document.getElementById('layerBtn');
const warnBtn=document.getElementById('warnBtn');
const digestBtn=document.getElementById('digestBtn');
function closePanels(except){
  [[panel,layerBtn],[warnPanel,warnBtn],[digestEl,digestBtn]].forEach(([p,b])=>{
    if(p===except)return;p.classList.remove('show');b.classList.remove('on');});
}
function toggleWarn(force){
  const show=force===true?true:!warnPanel.classList.contains('show');
  closePanels(warnPanel);
  if(show)buildWarnPanel();
  warnPanel.classList.toggle('show',show);warnBtn.classList.toggle('on',show);
}
warnBtn.onclick=()=>toggleWarn();
layerBtn.onclick=()=>{const s=!panel.classList.contains('show');
  closePanels(panel);panel.classList.toggle('show',s);layerBtn.classList.toggle('on',s);};
digestBtn.onclick=()=>{const s=!digestEl.classList.contains('show');
  closePanels(digestEl);
  if(s&&!digestEl.querySelector('table'))buildDigest();
  digestEl.classList.toggle('show',s);digestBtn.classList.toggle('on',s);};

/* ── Layer switches ── */
function bindSw(id,init,fn){const el=document.getElementById(id);el.classList.toggle('on',!!init);
  el.onclick=()=>{el.classList.toggle('on');fn(el.classList.contains('on'));};}
bindSw('swRisk',S.risk,on=>{riskOn=on;S.risk=on;save();if(mode==='forecast')ovTop.setOpacity(on?riskOpac:0);});
bindSw('swLabels',S.labels,on=>{S.labels=on;save();on?labels.addTo(map):map.removeLayer(labels);});
bindSw('swCities',S.cities,on=>{citiesOn=on;S.cities=on;save();
  cityMarkers.forEach(m=>on?m.addTo(map):m.remove());if(on){Object.keys(cityLast).forEach(k=>cityLast[k]=-1);updateCityMarkers();}});
bindSw('swSatMap',S.satMap,on=>{S.satMap=on;save();
  if(on){map.removeLayer(baseDark);baseSat.addTo(map);}else{map.removeLayer(baseSat);baseDark.addTo(map);}});
bindSw('swRadar',S.radar,on=>{radarUnderlay=on;S.radar=on;save();paintRadar();});
bindSw('swSatIR',S.satIR,on=>{satIROn=on;S.satIR=on;save();
  hideAllRadar();
  if(mode==='radar'){rIdx=Math.max(0,Math.min(rIdx,radarCount()-1));buildRadarStrip();
    if(radarCount())showRadar(rIdx);}
  else paintRadar();
  updateRadarLegend();});
bindSw('swSmooth',S.smooth,on=>{S.smooth=on;save();rebuildRadarLayers();});
const riskOpacEl=document.getElementById('riskOpac');riskOpacEl.value=S.riskOpac;
riskOpacEl.oninput=function(){riskOpac=this.value/100;S.riskOpac=+this.value;save();
  if(riskOn&&mode==='forecast')ovTop.setOpacity(riskOpac);};
const radarOpacEl=document.getElementById('radarOpac');radarOpacEl.value=S.radarOpac;
radarOpacEl.oninput=function(){radarOpacity=this.value/100;S.radarOpac=+this.value;save();
  if(mode==='radar'){const f=curSet()[rIdx];if(f&&f.layer)f.layer.setOpacity(radarOpacity);}
  else paintRadar();};

/* ── Radar (RainViewer) ─────────────────────────────────────────────────────
   Tile layers are created lazily around the cursor and pruned behind it, so a
   radar loop costs a handful of tile layers instead of fifteen loading at once. */
const RADAR_SCHEMES=[[2,'Universal blue'],[4,'TWC'],[6,'NEXRAD level III'],
                     [7,'Rainbow SELEX'],[8,'Dark sky'],[1,'Original'],[3,'Titan'],[5,'Meteored']];
const SCHEME_GRADS={
  1:'linear-gradient(90deg,#6ea6ff,#3ad46a,#f2e33c,#ff8a1f,#ff2d2d,#d23bd2)',
  2:'linear-gradient(90deg,#c7e0ff,#6fb2ff,#2d6fe0,#1a3fb0,#8a3fd0,#d23bd2)',
  3:'linear-gradient(90deg,#8fd6ff,#3ad46a,#f2e33c,#ff8a1f,#ff2d2d,#ffffff)',
  4:'linear-gradient(90deg,#00c8c8,#19d36b,#e9e337,#ff8a1f,#ff2d2d,#d23bd2)',
  5:'linear-gradient(90deg,#7ad0ff,#39c46b,#ffe23b,#ff7a1f,#e02020,#b02090)',
  6:'linear-gradient(90deg,#3aa0ff,#19d36b,#e9e337,#ff8a1f,#ff2d2d,#d23bd2)',
  7:'linear-gradient(90deg,#9be7ff,#39d0c8,#4ad04a,#ffe23b,#ff5a1f,#d23bd2)',
  8:'linear-gradient(90deg,#2c3e50,#3f7fa8,#5fb0c8,#9fd8b0,#ffe0a0,#ff9a80)'
};
let radarScheme=+S.scheme||6;
const radar={host:'',rad:[],sat:[],past:0,ready:false,err:false,loadedAt:0,fetching:false};
const radarFlag=document.getElementById('radarFlag');
const radarSpin=document.getElementById('radarSpin');
function curSet(){return satIROn?radar.sat:radar.rad;}
function radarCount(){return curSet().length;}

(function fillSchemeSelect(){
  const sel=document.getElementById('radarScheme');
  RADAR_SCHEMES.forEach(([v,n])=>{
    const o=document.createElement('option');o.value=v;o.textContent=n;
    if(v===radarScheme)o.selected=true;sel.appendChild(o);});
  sel.onchange=function(){radarScheme=+this.value;S.scheme=radarScheme;save();
    rebuildRadarLayers();updateRadarLegend();};
})();
function updateRadarLegend(){
  const g=document.getElementById('radarGrad');
  g.style.background=satIROn?'linear-gradient(90deg,#101820,#3a4a58,#7a8a98,#c8d4dc,#ffffff)'
                            :(SCHEME_GRADS[radarScheme]||SCHEME_GRADS[6]);
  const lbls=document.querySelector('.radar-grad-labels');
  lbls.innerHTML=satIROn?'<span>WARM</span><span>·</span><span>·</span><span>COLD TOPS</span>'
    :'<span>LIGHT</span><span>MOD</span><span>HEAVY</span><span>INTENSE</span>';
  document.querySelector('#radarLegend .lg-title').textContent=
    satIROn?'SATELLITE · CLOUD-TOP TEMP':'RADAR · PRECIP INTENSITY';
}
updateRadarLegend();

function tileUrl(fr){
  const opts=satIROn?'0/0_0':(radarScheme+'/'+(S.smooth?1:0)+'_1');
  return radar.host+fr.path+'/256/{z}/{x}/{y}/'+opts+'.png';
}
function ensureLayer(fr){
  if(fr.layer)return fr.layer;
  fr.layer=L.tileLayer(tileUrl(fr),{opacity:0,pane:'radarPane',
    maxNativeZoom:12,maxZoom:19,tileSize:256,updateWhenIdle:false,keepBuffer:2,
    crossOrigin:true,className:'radar-tiles'});
  fr.layer.on('loading',()=>{if(isCurrent(fr))radarSpin.classList.add('show');});
  fr.layer.on('load',()=>{if(isCurrent(fr))radarSpin.classList.remove('show');});
  fr.layer.addTo(map);
  return fr.layer;
}
function isCurrent(fr){const s=curSet();return s[rIdx]===fr;}
function dropLayer(fr){if(fr.layer){try{map.removeLayer(fr.layer);}catch(e){}fr.layer=null;}}
function pruneLayers(center,keep){
  const s=curSet();
  s.forEach((fr,i)=>{if(Math.abs(i-center)>keep)dropLayer(fr);});
  (satIROn?radar.rad:radar.sat).forEach(dropLayer);
}
function rebuildRadarLayers(){
  radar.rad.forEach(dropLayer);radar.sat.forEach(dropLayer);
  if(mode==='radar'&&radarCount())showRadar(rIdx);else paintRadar();
}
function hideAllRadar(){
  radar.rad.forEach(f=>{if(f.layer)f.layer.setOpacity(0);});
  radar.sat.forEach(f=>{if(f.layer)f.layer.setOpacity(0);});
}
function fmtMins(m){return m<60?m+'m':Math.floor(m/60)+'h'+(m%60?('0'+(m%60)).slice(-2):'');}
function fmtTime(ts){const d=new Date(ts*1000);
  return ('0'+d.getUTCHours()).slice(-2)+':'+('0'+d.getUTCMinutes()).slice(-2)+'Z';}
function fmtLocal(ts){return aestClock(new Date(ts*1000));}

function loadRadar(){
  if(radar.fetching)return;
  radar.fetching=true;
  fetch('https://api.rainviewer.com/public/weather-maps.json',{cache:'no-store'})
   .then(r=>{if(!r.ok)throw new Error('HTTP '+r.status);return r.json();})
   .then(d=>{
     radar.fetching=false;radar.err=false;
     radar.host=d.host;
     const past=(d.radar&&d.radar.past?d.radar.past:[]).slice(-12);
     const now =(d.radar&&d.radar.nowcast?d.radar.nowcast:[]).slice(0,3);
     const ir  =(d.satellite&&d.satellite.infrared?d.satellite.infrared:[]).slice(-12);
     radar.past=past.length;
     radar.rad=mergeFrames(radar.rad,past.concat(now));
     radar.sat=mergeFrames(radar.sat,ir);
     radar.ready=radar.rad.length>0;
     radar.loadedAt=Date.now();
     updateRadarAge();
     radarFlag.classList.remove('err');
     if(mode==='radar'){
       // keep the cursor pinned to the newest observation when it was already there
       const atEnd=rIdx>=radarCount()-1||rIdx===radar.past-1;
       rIdx=Math.max(0,Math.min(rIdx,radarCount()-1));
       if(atEnd&&!satIROn)rIdx=Math.max(0,radar.past-1);
       buildRadarStrip();buildRadarTicks();
       scrub.max=Math.max(0,radarCount()-1);
       if(radarCount())showRadar(rIdx);
     }else paintRadar();
   })
   .catch(e=>{
     radar.fetching=false;radar.err=true;
     console.warn('Radar load failed:',e);
     radarFlag.classList.add('show','err');
     document.getElementById('radarMode').textContent='RADAR UNAVAILABLE';
     document.getElementById('radarTime').textContent='retrying…';
     document.getElementById('radarAge').textContent='';
     if(mode==='radar')
       document.getElementById('timeMain').innerHTML=
         '<span class="tag obs">RADAR</span>feed unavailable — retrying';
     setTimeout(loadRadar,20000);
   });
}
/* reuse existing tile layers for timestamps we already have (no reload flash) */
function mergeFrames(old,incoming){
  const byTime={};old.forEach(f=>byTime[f.time]=f);
  const out=incoming.map(fr=>{
    const ex=byTime[fr.time];
    if(ex){delete byTime[fr.time];ex.path=fr.path;return ex;}
    return {time:fr.time,path:fr.path,layer:null};
  });
  Object.values(byTime).forEach(dropLayer);   // frames that aged out
  return out;
}
function updateRadarAge(){
  const el=document.getElementById('radarAge');
  if(!radar.loadedAt){el.textContent='';return;}
  const s=Math.round((Date.now()-radar.loadedAt)/1000);
  el.textContent=s<90?'· just updated':'· '+Math.round(s/60)+'m ago';
}
setInterval(updateRadarAge,30000);
setInterval(()=>{if(!document.hidden)loadRadar();},300000);
document.addEventListener('visibilitychange',()=>{
  if(!document.hidden&&radar.loadedAt&&Date.now()-radar.loadedAt>300000)loadRadar();});
function reloadRadar(){toast('Refreshing radar…');loadRadar();}

function paintRadar(){
  // forecast mode: pin the most recent observation under the risk overlay
  if(mode!=='forecast')return;
  hideAllRadar();
  if(radarUnderlay&&radar.ready){
    const set=curSet();
    const idx=satIROn?set.length-1:Math.max(0,radar.past-1);
    const fr=set[idx];
    if(fr){ensureLayer(fr).setOpacity(radarOpacity*.85);pruneLayers(idx,0);}
    radarFlag.classList.add('show');radarFlag.classList.remove('err');
    document.getElementById('radarMode').textContent=satIROn?'SAT IR · LATEST':'RADAR · LATEST';
    document.getElementById('radarTime').textContent=fr?fmtLocal(fr.time)+' AEST':'';
  }else radarFlag.classList.remove('show');
}
function buildRadarTicks(){
  const wrap=document.getElementById('radarticks');
  const set=curSet();
  if(!set.length){wrap.innerHTML='';return;}
  const marks=satIROn?[0,Math.floor(set.length/2),set.length-1]
                     :[0,Math.max(0,radar.past-1),set.length-1];
  wrap.innerHTML=[...new Set(marks)].map(i=>{
    const mins=Math.round((set[i].time*1000-Date.now())/60000);
    const lbl=Math.abs(mins)<=3?'NOW':(mins<0?'-'+fmtMins(-mins):'+'+fmtMins(mins));
    return '<div class="dt'+(Math.abs(mins)<=3?' on':'')+'" data-r="'+i+'">'+
      '<span class="dt-label">'+lbl+'</span><span style="font-size:8px;opacity:.6">'+
      fmtLocal(set[i].time)+'</span></div>';
  }).join('');
  [...wrap.children].forEach(el=>el.onclick=()=>{stop();showRadar(+el.dataset.r);});
}
function enterRadar(){
  mode='radar';
  ovA.setOpacity(0);ovB.setOpacity(0);
  document.getElementById('riskLegend').style.display='none';
  document.getElementById('radarLegend').classList.add('show');
  document.getElementById('alert').classList.remove('show');
  document.getElementById('dayticks').classList.add('hide');
  document.getElementById('radarticks').classList.remove('hide');
  nowline.classList.remove('show');
  // forecast risk labels are meaningless over live radar — blank them out
  CITIES.forEach(([name],i)=>{cityLast[name]=0;
    if(citiesOn)cityMarkers[i].setIcon(cityIcon(name,0));});
  updateRadarLegend();
  if(!radar.ready||!radarCount()){
    document.getElementById('timeMain').innerHTML=
      '<span class="tag obs">RADAR</span>'+(radar.err?'feed unavailable — retrying':'loading…');
    document.getElementById('timeUtc').textContent='';
    radarSpin.classList.add('show');
    stripEl.innerHTML='';stripEl.appendChild(nowline);
    loadRadar();
    return;
  }
  radarSpin.classList.remove('show');
  rIdx=Math.max(0,Math.min(rIdx,radarCount()-1));
  if(rIdx===0&&!satIROn)rIdx=Math.max(0,radar.past-1);
  scrub.max=radarCount()-1;
  buildRadarStrip();buildRadarTicks();
  showRadar(rIdx);
  if(!playing)play();
}
function exitRadar(){
  hideAllRadar();
  mode='forecast';
  Object.keys(cityLast).forEach(k=>cityLast[k]=-1);   // force a risk-label refresh
  scrub.max=N-1;
  document.getElementById('dayticks').classList.remove('hide');
  document.getElementById('radarticks').classList.add('hide');
  radarSpin.classList.remove('show');
  buildStrip();
  paintRadar();
}
function showRadar(i){
  const set=curSet(),n=set.length;
  if(!n)return;
  rIdx=(i+n)%n;
  const fr=set[rIdx];
  ensureLayer(fr).setOpacity(radarOpacity);
  set.forEach((f,k)=>{if(k!==rIdx&&f.layer)f.layer.setOpacity(0);});
  // prefetch neighbours so playback is seamless, prune the rest
  [rIdx+1,rIdx+2,rIdx-1].forEach(k=>{if(k>=0&&k<n)ensureLayer(set[k]);});
  pruneLayers(rIdx,3);

  const nc=rIdx>=radar.past&&!satIROn;
  const tag=nc?'NOWCAST':(satIROn?'SAT IR':'OBSERVED');
  const mins=Math.round((fr.time*1000-Date.now())/60000);
  const rel=Math.abs(mins)<=2?'now':(mins<0?(-mins)+' min ago':'+'+mins+' min');
  document.getElementById('timeMain').innerHTML=
    '<span class="tag obs">'+tag+'</span>'+fmtLocal(fr.time)+' AEST'+
    '<span class="sm">'+rel+' · '+(rIdx+1)+'/'+n+'</span>';
  document.getElementById('timeUtc').textContent=
    (satIROn?'Infrared satellite · ':'Precipitation radar · ')+fmtTime(fr.time);
  document.getElementById('radarMode').textContent=satIROn?'SATELLITE IR':(nc?'RADAR · NOWCAST':'RADAR');
  document.getElementById('radarTime').textContent=fmtLocal(fr.time)+' AEST';
  radarFlag.classList.add('show');radarFlag.classList.remove('err');
  scrub.max=n-1;scrub.value=rIdx;
  updateStripCursor();syncHash();
}

/* ── Daily digest ── */
function buildDigest(){
  const days=Object.keys(dayFirst);
  let html='<h4>5-Day Risk Outlook · peak per day</h4><table class="dg-table"><thead><tr><th></th>';
  days.forEach(d=>{html+='<th>'+dayName(d)+'</th>';});
  html+='</tr></thead><tbody>';
  REAL_HAZARDS.concat(['Max']).forEach(h=>{
    html+='<tr><td class="lbl">'+(ICONS[h]||'')+' '+(h==='Max'?'Overall':h)+'</td>';
    days.forEach(d=>{
      let mx=0,mi=dayFirst[d];
      dayFrameCache[d].forEach(i=>{const v=(PROFILE[h]&&PROFILE[h][i])||0;if(v>mx){mx=v;mi=i;}});
      const tc=mx>=3?'#0a1016':mx>0?'#cfe6f2':'#3f6377';
      html+='<td class="cell" data-h="'+h+'" data-i="'+mi+'" title="peak '+
        aestClock(frameDate(mi))+'" style="background:'+RISK_COLORS[mx]+';color:'+tc+'">'+
        RISK_LABELS[mx]+'</td>';
    });
    html+='</tr>';
  });
  html+='</tbody></table><div class="dg-foot">Click a cell to jump to that peak.</div>';
  digestEl.innerHTML=html;
  [...digestEl.querySelectorAll('td.cell')].forEach(td=>{
    td.onclick=()=>{
      if(mode==='radar')exitRadar();
      selectMode(td.dataset.h);
      stop();showForecast(+td.dataset.i,true);
      closePanels();
    };
  });
}

/* ── Info modal ── */
const modal=document.getElementById('modal');
document.getElementById('infoBtn').onclick=()=>modal.classList.add('show');
document.getElementById('closeBtn').onclick=()=>modal.classList.remove('show');
modal.onclick=e=>{if(e.target===modal)modal.classList.remove('show');};

/* ── Keyboard ── */
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT'||e.target.tagName==='SELECT')return;
  const k=e.key;
  if(e.code==='Space'){e.preventDefault();playing?stop():play();}
  else if(e.code==='ArrowRight'){e.preventDefault();stop();step(1);}
  else if(e.code==='ArrowLeft'){e.preventDefault();stop();step(-1);}
  else if(e.code==='Home'){stop();if(mode==='radar')showRadar(0);else showForecast(0,true);}
  else if(e.code==='End'){stop();if(mode==='radar')showRadar(radarCount()-1);else showForecast(N-1,true);}
  else if(k==='Escape'){modal.classList.remove('show');closePanels();map.closePopup();}
  else if(k>='1'&&k<='5'){const h=MODES[+k-1];if(h)selectMode(h);}
  else if(k==='0'){selectMode('Radar');}
  else if(k==='w'||k==='W'){toggleWarn();}
  else if(k==='d'||k==='D'){digestBtn.click();}
  else if(k==='l'||k==='L'){layerBtn.click();}
  else if(k==='f'||k==='F'){toggleFs();}
  else if(k==='?'){modal.classList.toggle('show');}
  else if((k==='r'||k==='R')&&!e.ctrlKey&&!e.metaKey){reloadRadar();}
});

/* ── Share hash ── */
let hashTimer=null;
function syncHash(){
  clearTimeout(hashTimer);
  hashTimer=setTimeout(()=>{
    const v=(mode==='radar')?('Radar/'+rIdx):(curHazard+'/'+curFrame);
    try{history.replaceState(null,'','#'+v);}catch(e){}
  },250);
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
updateWarnBadge();
if(want==='Radar'){highlightMode('Radar');showForecast(curFrame,false);enterRadar();}
else{highlightMode(curHazard);showForecast(curFrame,false);preloadHazard(curHazard);}
loadRadar();
map.on('zoomend',()=>{if(mode==='forecast')positionNow();});
setTimeout(()=>{if(!localStorage.getItem('aw_seen')){modal.classList.add('show');
  try{localStorage.setItem('aw_seen','1');}catch(e){}}},600);
/* nudge new visitors toward the warnings list when something serious is on */
setTimeout(()=>{
  const top=WARNINGS.reduce((v,w)=>Math.max(v,w.lvl),0);
  if(top>=4&&!modal.classList.contains('show'))
    toast(WARNINGS.length+' significant risk window'+(WARNINGS.length===1?'':'s')+' — tap ⚠ for details');
},2600);
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

    lmask = None
    last_lats = last_lons = None
    for n, fhour in enumerate(FRAME_HOURS):
        valid = run_dt + timedelta(hours=fhour)
        print(f"\n  [{n+1}/{len(FRAME_HOURS)}] f{fhour:03d}  {valid:%a %d %b %HZ}", end="  ", flush=True)
        try:
            lats, lons, fields = fetch_frame(date_s, run_s, fhour)
        except Exception as e:
            print(f"![frame:{e}]", end="", flush=True)
            lats = lons = None
            fields = {}
        if lats is None:
            # every field failed for this hour — reuse the previous grid so the
            # frame renders empty instead of taking the whole run down
            lats, lons = last_lats, last_lons
            if lats is None:
                print(" skipped (no grid yet)", end="", flush=True)
                continue
        else:
            last_lats, last_lons = lats, lons
        risks = compute_risks(fields)
        if lmask is None or lmask.shape != risks["Max"].shape:
            lmask = land_mask(lats, lons, polys_ll if geojson else [])
        for hz in DATA_HAZARDS:
            risks[hz] = risks[hz] * lmask
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
            "iso": valid.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "images": imgs,
            "city_risks": city_r,
            "region_risks": region_risks_for_frame(risks, lats, lons),
            "areas": areas_for_frame(risks, lats, lons),
            "coarse": coarse,
            "peak": peak,
        })
        print("✓", end="", flush=True)

    print("\n\n[4/4] Building warnings + writing index.html...")
    warnings = build_warnings(frames)
    print(f"      {len(warnings)} warning events (ENH+)")
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    html = make_html(frames, run_label, timestamp, coarse_lats or [], coarse_lons or [],
                     run_iso=run_dt.strftime("%Y-%m-%dT%H:%M:%SZ"), warnings=warnings)
    with open(out, "w") as f:
        f.write(html)
    size_mb = len(html.encode()) / 1e6
    print(f"      {out}  ({size_mb:.1f} MB)\n")


if __name__ == "__main__":
    main()
