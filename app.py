
import re
import zipfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from scipy import signal, sparse
from scipy.sparse.linalg import spsolve
from scipy.interpolate import CubicSpline
from scipy.spatial.distance import pdist, squareform


st.set_page_config(page_title="VRC / HRV RRi Analyzer Pro v4.1", layout="wide")

PHASES = ["Basal"] + [f"E{i}" for i in range(1, 7)] + [f"R{i}" for i in range(1, 4)]
PHASE_GROUP = {"Basal": "Basal", **{f"E{i}": "Ejercicio" for i in range(1, 7)}, **{f"R{i}": "Recuperación" for i in range(1, 4)}}
PHASE_COLORS = {"Basal": "rgba(0,150,255,0.22)", "Ejercicio": "rgba(255,140,0,0.18)", "Recuperación": "rgba(0,200,100,0.18)"}
FS_INTERP = 4.0
LAMBDA_DEFAULT = 500

PARAM_GROUPS = {
    "Tiempo": ["MeanHR", "MeanRR", "SDNN", "RMSSD", "pNN50", "SD1", "SD2"],
    "Frecuencia": ["VLF", "LF", "HF", "TOTAL", "LF_HF"],
    "Complejidad": ["DFA_alpha1", "DFA_alpha2", "ApEn", "SampEn"],
    "Recurrencia": ["REC", "DET", "Lmean", "Lmax", "ShanEn"],
}

DOMAIN_GROUPS = {
    "Amplitud": ["SDNN", "SD2", "TOTAL"],
    "Vagal": ["RMSSD", "SD1", "HF", "pNN50"],
    "Complejidad": ["DFA_alpha1", "DFA_alpha2", "ApEn", "SampEn"],
    "Recurrencia": ["REC", "DET", "Lmean", "Lmax", "ShanEn"],
}


def sanitize_name(name):
    name = Path(str(name)).stem
    name = re.sub(r"[^A-Za-z0-9_\-]+", "_", name).strip("_")
    return name or "registro"


def read_rri_file(uploaded_file):
    raw = uploaded_file.read()
    text = raw.decode("utf-8", errors="ignore")
    vals = []
    for line in text.replace(";", "\n").replace("\t", "\n").splitlines():
        line = line.strip().replace(",", ".")
        if not line:
            continue
        for p in line.split():
            try:
                vals.append(float(p))
            except Exception:
                pass
    rr = np.asarray(vals, dtype=float)
    rr = rr[np.isfinite(rr)]
    if len(rr) == 0:
        raise ValueError("No se han detectado RRi numéricos.")
    if np.nanmedian(rr) > 10:
        rr = rr / 1000.0
    rr = rr[(rr >= 0.3) & (rr <= 2.0)]
    if len(rr) == 0:
        raise ValueError("Tras el filtrado fisiológico no quedan RRi válidos.")
    return rr


def cumulative_time(rr):
    return np.cumsum(rr)


def sec_to_hms(seconds):
    seconds = int(round(float(seconds)))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def hms_to_sec(s):
    parts = [float(p) for p in str(s).strip().split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0]


def cut_segment(rr, start_s, end_s):
    t = cumulative_time(rr)
    return rr[(t >= start_s) & (t <= end_s)]


def default_windows(t_max):
    t_max = float(t_max)
    if t_max <= 0:
        t_max = 600.0
    if t_max < 600:
        step = max(t_max / 10, 20)
        return {ph: [min(i * step, t_max), min((i + 1) * step, t_max)] for i, ph in enumerate(PHASES)}
    basal = [0.0, min(300.0, t_max)]
    rem_start = basal[1]
    rem = max(0.0, t_max - rem_start)
    step = rem / 9.0 if rem > 0 else 60.0
    w = {"Basal": basal}
    for i in range(1, 7):
        w[f"E{i}"] = [min(rem_start + (i - 1) * step, t_max), min(rem_start + i * step, t_max)]
    for i in range(1, 4):
        j = 6 + i
        w[f"R{i}"] = [min(rem_start + (j - 1) * step, t_max), min(rem_start + j * step, t_max)]
    return w


def smoothness_priors_detrend(y, lam=500):
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 5:
        return y - np.mean(y) if n else y
    I = sparse.eye(n, format="csc")
    e = np.ones(n)
    D2 = sparse.diags([e[:-2], -2 * e[:-2], e[:-2]], [0, 1, 2], shape=(n - 2, n), format="csc")
    trend = spsolve(I + (lam ** 2) * (D2.T @ D2), y)
    return y - trend


def interpolate_rr(rr, fs=FS_INTERP, apply_lambda=False, lam=500):
    t = cumulative_time(rr)
    if len(t) < 5:
        return np.array([]), np.array([])
    t = t - t[0]
    x = rr.copy()
    keep = np.r_[True, np.diff(t) > 0]
    t = t[keep]
    x = x[keep]
    if len(t) < 5:
        return np.array([]), np.array([])
    ti = np.arange(0, t[-1], 1 / fs)
    if len(ti) < 5:
        return np.array([]), np.array([])
    xi = CubicSpline(t, x, bc_type="natural")(ti)
    if apply_lambda:
        xi = smoothness_priors_detrend(xi, lam)
    return ti, xi


def time_metrics(rr):
    rr_ms = rr * 1000.0
    diff = np.diff(rr_ms)
    mean_rr = np.mean(rr_ms)
    sdnn = np.std(rr_ms, ddof=1) if len(rr_ms) > 1 else np.nan
    rmssd = np.sqrt(np.mean(diff ** 2)) if len(diff) else np.nan
    nn50 = int(np.sum(np.abs(diff) > 50)) if len(diff) else 0
    pnn50 = 100 * nn50 / len(diff) if len(diff) else np.nan
    sd1 = np.sqrt(0.5) * np.std(diff, ddof=1) if len(diff) > 1 else np.nan
    sd2 = np.sqrt(max(0, 2 * sdnn ** 2 - sd1 ** 2)) if np.isfinite(sdnn) and np.isfinite(sd1) else np.nan
    return {"N_RRi": len(rr), "Duration_s": float(np.sum(rr)), "MeanRR": mean_rr, "MeanHR": 60000 / mean_rr if mean_rr > 0 else np.nan,
            "SDNN": sdnn, "RMSSD": rmssd, "NN50": nn50, "pNN50": pnn50, "SD1": sd1, "SD2": sd2}


def psd_metrics(rr):
    ti, xi = interpolate_rr(rr, fs=FS_INTERP, apply_lambda=True, lam=LAMBDA_DEFAULT)
    if len(xi) < 32:
        return {"VLF": np.nan, "LF": np.nan, "HF": np.nan, "TOTAL": np.nan, "LF_HF": np.nan}
    xi_ms = xi * 1000
    xi_ms = xi_ms - np.mean(xi_ms)
    nperseg = min(int(256 * FS_INTERP), len(xi_ms))
    noverlap = int(0.5 * nperseg)
    f, pxx = signal.welch(xi_ms, fs=FS_INTERP, window="hann", nperseg=nperseg, noverlap=noverlap, detrend=False, scaling="density")
    def bp(lo, hi):
        mask = (f >= lo) & (f < hi)
        return np.trapezoid(pxx[mask], f[mask]) if np.any(mask) else np.nan
    vlf, lf, hf = bp(0.0033, 0.04), bp(0.04, 0.15), bp(0.15, 0.40)
    total = np.nansum([vlf, lf, hf])
    return {"VLF": vlf, "LF": lf, "HF": hf, "TOTAL": total, "LF_HF": lf / hf if pd.notna(hf) and hf > 0 else np.nan}


def _phi_apen(x, m, r):
    n = len(x)
    if n <= m + 1:
        return np.nan
    pats = np.array([x[i:i + m] for i in range(n - m + 1)])
    vals = []
    for p in pats:
        dist = np.max(np.abs(pats - p), axis=1)
        c = np.mean(dist <= r)
        if c > 0:
            vals.append(np.log(c))
    return np.mean(vals) if vals else np.nan


def apen_calc(x, m=2, r_ratio=0.2):
    x = smoothness_priors_detrend(np.asarray(x, dtype=float), LAMBDA_DEFAULT)
    r = r_ratio * np.std(x, ddof=1)
    if not np.isfinite(r) or r == 0:
        return np.nan
    return _phi_apen(x, m, r) - _phi_apen(x, m + 1, r)


def sampen_calc(x, m=2, r_ratio=0.2):
    x = smoothness_priors_detrend(np.asarray(x, dtype=float), LAMBDA_DEFAULT)
    n = len(x)
    if n <= m + 2:
        return np.nan
    r = r_ratio * np.std(x, ddof=1)
    if not np.isfinite(r) or r == 0:
        return np.nan
    def count(mm):
        pats = np.array([x[i:i + mm] for i in range(n - mm + 1)])
        c = 0
        for i in range(len(pats) - 1):
            dist = np.max(np.abs(pats[i + 1:] - pats[i]), axis=1)
            c += np.sum(dist <= r)
        return c
    b, a = count(m), count(m + 1)
    if a == 0 or b == 0:
        return np.nan
    return -np.log(a / b)


def dfa_calc(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 50:
        return np.nan, np.nan
    y = np.cumsum(x - np.mean(x))
    scales = np.unique(np.floor(np.logspace(np.log10(4), np.log10(max(5, n // 4)), 18)).astype(int))
    ss, ff = [], []
    for s in scales:
        if s < 4 or n // s < 2:
            continue
        rms = []
        for i in range(n // s):
            seg = y[i * s:(i + 1) * s]
            t = np.arange(s)
            co = np.polyfit(t, seg, 1)
            rms.append(np.sqrt(np.mean((seg - np.polyval(co, t)) ** 2)))
        val = np.sqrt(np.mean(np.asarray(rms) ** 2))
        if val > 0:
            ss.append(s)
            ff.append(val)
    ss, ff = np.asarray(ss), np.asarray(ff)
    if len(ss) < 4:
        return np.nan, np.nan
    m1, m2 = (ss >= 4) & (ss <= 16), ss > 16
    return (np.polyfit(np.log(ss[m1]), np.log(ff[m1]), 1)[0] if np.sum(m1) >= 2 else np.nan,
            np.polyfit(np.log(ss[m2]), np.log(ff[m2]), 1)[0] if np.sum(m2) >= 2 else np.nan)


def rqa_calc(x, emb_dim=10, tau=1, l_min=2, max_n=500):
    x = np.asarray(x, dtype=float)
    if len(x) > max_n:
        x = x[np.linspace(0, len(x) - 1, max_n).astype(int)]
    n = len(x) - (emb_dim - 1) * tau
    if n < 20:
        return {"REC": np.nan, "DET": np.nan, "Lmean": np.nan, "Lmax": np.nan, "ShanEn": np.nan}
    X = np.array([x[i:i + emb_dim * tau:tau] for i in range(n)])
    D = squareform(pdist(X))
    radius = np.sqrt(emb_dim) * np.std(x, ddof=1)
    R = (D <= radius).astype(int)
    np.fill_diagonal(R, 0)
    rec = 100 * R.sum() / (n * n - n)
    lens = []
    for k in range(-n + 1, n):
        diag = np.diag(R, k=k)
        c = 0
        for val in diag:
            if val:
                c += 1
            else:
                if c >= l_min:
                    lens.append(c)
                c = 0
        if c >= l_min:
            lens.append(c)
    if not lens:
        return {"REC": rec, "DET": 0, "Lmean": 0, "Lmax": 0, "ShanEn": 0}
    lens = np.asarray(lens)
    det = 100 * lens.sum() / R.sum() if R.sum() > 0 else 0
    vals, counts = np.unique(lens, return_counts=True)
    p = counts / counts.sum()
    return {"REC": rec, "DET": det, "Lmean": np.mean(lens), "Lmax": np.max(lens), "ShanEn": -np.sum(p * np.log(p))}


def calculate_all(rr, include_rqa=True):
    rr_ms = rr * 1000
    out = {}
    out.update(time_metrics(rr))
    out.update(psd_metrics(rr))
    a1, a2 = dfa_calc(rr_ms)
    out["DFA_alpha1"], out["DFA_alpha2"] = a1, a2
    out["ApEn"] = apen_calc(rr_ms)
    out["SampEn"] = sampen_calc(rr_ms)
    if include_rqa:
        out.update(rqa_calc(rr_ms))
    return out


def calculate_record(rr, windows, min_rr, include_rqa):
    rows, segments, valid = [], {}, {}
    for ph in PHASES:
        s, e = windows[ph]
        seg = cut_segment(rr, s, e)
        segments[ph] = seg
        valid[ph] = len(seg) >= min_rr
        if valid[ph]:
            res = calculate_all(seg, include_rqa=include_rqa)
            res["Fase"] = ph
            rows.append(res)
    return (pd.DataFrame(rows).set_index("Fase") if rows else pd.DataFrame()), segments, valid


def build_long(records_results):
    rows = []
    for rec, df in records_results.items():
        if df is None or df.empty:
            continue
        tmp = df.copy()
        tmp.insert(0, "Registro", rec)
        tmp.insert(1, "Fase", tmp.index)
        rows.append(tmp.reset_index(drop=True))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def domain_values(metrics_df, method="median"):
    if metrics_df.empty or "Basal" not in metrics_df.index:
        return pd.DataFrame()
    base = metrics_df.loc["Basal"]
    out = {}
    for dom, vars_ in DOMAIN_GROUPS.items():
        vals_phase = []
        for ph in metrics_df.index:
            vals = []
            for v in vars_:
                if v in metrics_df.columns and pd.notna(base.get(v)) and pd.notna(metrics_df.loc[ph, v]) and base[v] != 0:
                    vals.append(100 * metrics_df.loc[ph, v] / base[v])
            vals_phase.append(np.nanmedian(vals) if vals and method == "median" else (np.nanmean(vals) if vals else np.nan))
        out[dom] = vals_phase
    return pd.DataFrame(out, index=metrics_df.index)


def rr_plot(record_data, windows, view_mode, selected_record):
    fig = go.Figure()
    names = [selected_record] if view_mode == "Registro principal" else list(record_data.keys())
    for name in names:
        rr = record_data[name]["rr"]
        t = cumulative_time(rr) / 60
        fig.add_trace(go.Scatter(x=t, y=rr * 1000, mode="lines", name=name))
    if view_mode == "Registro principal":
        for ph, (s, e) in windows.items():
            group = PHASE_GROUP.get(ph, ph)
            fig.add_vrect(x0=s/60, x1=e/60, fillcolor=PHASE_COLORS.get(group, "rgba(180,180,180,.15)"),
                          line_width=0, annotation_text=ph, annotation_position="top left")
    fig.update_layout(height=480, xaxis_title="Tiempo acumulado (min)", yaxis_title="RRi (ms)", hovermode="x unified", dragmode="select")
    fig.update_xaxes(rangeslider_visible=True)
    return fig


def comparison_fig(pivot, variable):
    """
    Comparación clara:
    - columnas verticales agrupadas por fase y por registro
    - línea de tendencia superpuesta para cada registro
    """
    fig = go.Figure()
    phases = list(pivot.index)

    for rec in pivot.columns:
        y = pivot[rec].astype(float)
        fig.add_trace(go.Bar(
            x=phases,
            y=y,
            name=f"{rec} · barras",
            opacity=0.62
        ))
        fig.add_trace(go.Scatter(
            x=phases,
            y=y,
            mode="lines+markers",
            name=f"{rec} · tendencia",
            line=dict(width=3)
        ))

    fig.update_layout(
        height=500,
        title=f"{variable}: columnas por fase + tendencia por registro",
        xaxis_title="Fase",
        yaxis_title=variable,
        barmode="group",
        bargap=0.22,
        bargroupgap=0.08,
        hovermode="x unified"
    )
    return fig


def multi_param_bar_trend_fig(long_df, phases, params):
    """
    Una figura por múltiples parámetros:
    eje X combinado Fase · Parámetro para evitar que todos los puntos caigan en la misma columna.
    Cada registro tiene barras y línea de tendencia superpuesta.
    """
    df = long_df[long_df["Fase"].isin(phases)].copy()
    if df.empty or not params:
        fig = go.Figure()
        fig.update_layout(height=500, title="Sin datos para comparar")
        return fig

    order_labels = []
    for ph in phases:
        for p in params:
            order_labels.append(f"{ph}<br>{p}")

    fig = go.Figure()

    for rec in df["Registro"].unique():
        y_vals = []
        x_vals = []
        drec = df[df["Registro"] == rec].set_index("Fase")
        for ph in phases:
            for p in params:
                x_vals.append(f"{ph}<br>{p}")
                if ph in drec.index and p in drec.columns:
                    y_vals.append(drec.loc[ph, p])
                else:
                    y_vals.append(np.nan)

        fig.add_trace(go.Bar(
            x=x_vals,
            y=y_vals,
            name=f"{rec} · barras",
            opacity=0.60
        ))
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines+markers",
            name=f"{rec} · tendencia",
            line=dict(width=3)
        ))

    fig.update_layout(
        height=620,
        title="Comparación multiparámetro: columnas verticales + tendencias",
        xaxis_title="Fase · parámetro",
        yaxis_title="Valor",
        barmode="group",
        bargap=0.20,
        bargroupgap=0.06,
        hovermode="x unified"
    )
    return fig


def parameter_grid_bar_trend(long_df, phases, params):
    """
    Devuelve una figura por parámetro, cada una con barras agrupadas por fase y tendencia.
    """
    figs = {}
    for p in params:
        pivot = long_df[long_df["Fase"].isin(phases)].pivot_table(
            index="Fase",
            columns="Registro",
            values=p,
            aggfunc="first"
        ).reindex(phases)
        figs[p] = comparison_fig(pivot, p)
    return figs


def multi_param_compare_fig(long_df, phases, params):
    fig = go.Figure()
    df = long_df[long_df["Fase"].isin(phases)].copy()
    for rec in df["Registro"].unique():
        for p in params:
            d = df[df["Registro"] == rec]
            if p in d.columns:
                fig.add_trace(go.Scatter(x=d["Fase"], y=d[p], mode="lines+markers", name=f"{rec} · {p}"))
    fig.update_layout(height=560, title="Comparación multiparámetro", xaxis_title="Fase", yaxis_title="Valor")
    return fig


def phase_rr_overlay(record_data, windows, phase):
    s, e = windows[phase]
    fig = go.Figure()
    for rec, data in record_data.items():
        seg = cut_segment(data["rr"], s, e)
        if len(seg) < 3:
            continue
        t = cumulative_time(seg)
        t = t - t[0]
        fig.add_trace(go.Scatter(x=t/60, y=seg*1000, mode="lines", name=rec))
    fig.update_layout(height=440, title=f"RRi superpuesto dentro de {phase}", xaxis_title="Tiempo dentro de fase (min)", yaxis_title="RRi (ms)")
    return fig


st.title("VRC / HRV RRi Analyzer Pro v4.1")
st.caption("Selección con ratón, ventanas libres y comparación con columnas verticales + líneas de tendencia por registro.")

with st.sidebar:
    uploaded_files = st.file_uploader("Sube uno o varios CSV/TXT con RRi", type=["csv", "txt"], accept_multiple_files=True)
    min_rr = st.number_input("Mínimo RRi por ventana", min_value=10, max_value=300, value=30, step=5)
    include_rqa = st.checkbox("Calcular RQA", value=False, help="Puede tardar en ventanas largas.")
    domain_method = st.selectbox("Dominios", ["median", "mean"], index=0)

if not uploaded_files:
    st.info("Sube uno o varios registros RRi.")
    st.stop()

record_data = {}
errors = []
for uf in uploaded_files:
    try:
        rr = read_rri_file(uf)
        name = sanitize_name(uf.name)
        base, k = name, 2
        while name in record_data:
            name = f"{base}_{k}"
            k += 1
        record_data[name] = {"rr": rr, "duration": float(np.sum(rr)), "filename": uf.name}
    except Exception as e:
        errors.append(f"{uf.name}: {e}")
if errors:
    st.error("\n".join(errors))
if not record_data:
    st.stop()

records = list(record_data.keys())
selected_record = st.sidebar.selectbox("Registro principal", records)
t_max = record_data[selected_record]["duration"]

if "selected_record_v40" not in st.session_state or st.session_state.selected_record_v40 != selected_record:
    st.session_state.selected_record_v40 = selected_record
    st.session_state.windows = default_windows(t_max)
if "windows" not in st.session_state:
    st.session_state.windows = default_windows(t_max)
if st.sidebar.button("Reiniciar ventanas"):
    st.session_state.windows = default_windows(t_max)
    st.rerun()

tab1, tab2, tab3, tab4, tab5 = st.tabs(["1) Segmentación", "2) HRV", "3) Comparar", "4) Gráficas", "5) Exportar"])

# central calculation
records_results, records_segments, records_valid = {}, {}, {}
for rec, data in record_data.items():
    df, segs, valid = calculate_record(data["rr"], st.session_state.windows, min_rr, include_rqa)
    records_results[rec], records_segments[rec], records_valid[rec] = df, segs, valid
metrics_df = records_results[selected_record]
long_df = build_long(records_results)

with tab1:
    st.subheader("Segmentación con ratón o manual")
    st.write("Selecciona una zona en el gráfico con el ratón; después elige la fase y pulsa aplicar.")

    active_phase = st.selectbox("Fase a modificar con la selección del ratón", PHASES, index=0)
    view_mode = st.radio("Vista", ["Registro principal", "Todos superpuestos"], horizontal=True)
    event = st.plotly_chart(rr_plot(record_data, st.session_state.windows, view_mode, selected_record),
                            use_container_width=True, on_select="rerun", selection_mode=("box", "lasso"))

    if event and getattr(event, "selection", None):
        pts = event.selection.get("points", [])
        xs = [p.get("x") for p in pts if "x" in p]
        if xs:
            s_sel, e_sel = min(xs) * 60, max(xs) * 60
            st.info(f"Selección: {sec_to_hms(s_sel)} - {sec_to_hms(e_sel)}")
            if st.button(f"Aplicar selección a {active_phase}"):
                st.session_state.windows[active_phase] = [s_sel, e_sel]
                st.success(f"{active_phase} actualizada.")
                st.rerun()

    st.markdown("### Editar ventanas manualmente")
    cols = st.columns(5)
    edited = {}
    for idx, ph in enumerate(PHASES):
        with cols[idx % 5]:
            st.markdown(f"**{ph}**")
            s0, e0 = st.session_state.windows[ph]
            ini = st.text_input(f"{ph} inicio", value=sec_to_hms(s0), key=f"{ph}_ini")
            fin = st.text_input(f"{ph} fin", value=sec_to_hms(e0), key=f"{ph}_fin")
            edited[ph] = (ini, fin)

    if st.button("Aplicar ventanas escritas"):
        new_w, ok = {}, True
        for ph, (ini, fin) in edited.items():
            try:
                s, e = hms_to_sec(ini), hms_to_sec(fin)
                if e <= s:
                    st.warning(f"{ph}: final debe ser mayor que inicio.")
                    ok = False
                new_w[ph] = [s, e]
            except Exception:
                st.warning(f"{ph}: formato no válido.")
                ok = False
        if ok:
            st.session_state.windows = new_w
            st.rerun()

    win_table = pd.DataFrame([{
        "Fase": ph, "Inicio": sec_to_hms(st.session_state.windows[ph][0]), "Fin": sec_to_hms(st.session_state.windows[ph][1]),
        "Duración_min": round((st.session_state.windows[ph][1]-st.session_state.windows[ph][0])/60, 2),
        **{f"{rec}_N": len(records_segments[rec][ph]) for rec in records},
        **{f"{rec}_OK": records_valid[rec][ph] for rec in records},
    } for ph in PHASES])
    st.dataframe(win_table, use_container_width=True)

with tab2:
    st.subheader(f"HRV: {selected_record}")
    if metrics_df.empty:
        st.info("No hay ventanas válidas para el registro principal. Baja el mínimo RRi o ajusta ventanas.")
    else:
        for group, cols in PARAM_GROUPS.items():
            present = [c for c in cols if c in metrics_df.columns]
            if present:
                st.markdown(f"### {group}")
                st.dataframe(metrics_df[present], use_container_width=True)

with tab3:
    st.subheader("Comparar registros")
    if len(records) < 2:
        st.info("Sube dos o más registros.")
    elif long_df.empty:
        st.info("No hay datos comparables. Ajusta ventanas o baja mínimo RRi.")
    else:
        valid_summary = pd.DataFrame(records_valid).T.reindex(columns=PHASES)
        st.markdown("### Ventanas válidas")
        st.dataframe(valid_summary, use_container_width=True)

        available_phases = [p for p in PHASES if p in long_df["Fase"].unique()]
        selected_phases = st.multiselect("Fases a comparar", PHASES, default=available_phases)
        numeric_vars = [c for c in long_df.columns if c not in ["Registro", "Fase"] and pd.api.types.is_numeric_dtype(long_df[c])]

        default_var = "RMSSD" if "RMSSD" in numeric_vars else numeric_vars[0]
        variable = st.selectbox("Variable principal", numeric_vars, index=numeric_vars.index(default_var))
        df_sel = long_df[long_df["Fase"].isin(selected_phases)] if selected_phases else long_df
        pivot = df_sel.pivot_table(index="Fase", columns="Registro", values=variable, aggfunc="first").reindex(selected_phases)

        st.markdown(f"### {variable}: columnas verticales + líneas de tendencia")
        st.dataframe(pivot, use_container_width=True)
        st.plotly_chart(comparison_fig(pivot, variable), use_container_width=True)

        st.markdown("### Comparación multiparámetro")
        param_defaults = [p for p in ["RMSSD", "SDNN", "SD1", "SD2", "LF", "HF"] if p in numeric_vars]
        params = st.multiselect("Parámetros a comparar", numeric_vars, default=param_defaults)

        modo_multi = st.radio(
            "Formato multiparámetro",
            ["Una gráfica combinada", "Una gráfica por parámetro"],
            horizontal=True
        )

        if params:
            if modo_multi == "Una gráfica combinada":
                st.plotly_chart(
                    multi_param_bar_trend_fig(long_df, selected_phases or available_phases, params),
                    use_container_width=True
                )
            else:
                figs = parameter_grid_bar_trend(long_df, selected_phases or available_phases, params)
                for p, figp in figs.items():
                    st.plotly_chart(figp, use_container_width=True)

        ph_overlay = st.selectbox("RRi superpuesto por fase", selected_phases or available_phases)
        st.plotly_chart(phase_rr_overlay(record_data, st.session_state.windows, ph_overlay), use_container_width=True)

        st.markdown("### Tabla completa filtrada")
        st.dataframe(df_sel, use_container_width=True)

with tab4:
    st.subheader("Gráficas del registro principal")
    if metrics_df.empty:
        st.info("No hay datos.")
    else:
        numeric_vars = [c for c in metrics_df.columns if pd.api.types.is_numeric_dtype(metrics_df[c])]
        params = st.multiselect("Parámetros a graficar", numeric_vars, default=[p for p in ["RMSSD", "SDNN", "SD1", "SD2", "LF", "HF"] if p in numeric_vars])
        fig = go.Figure()
        for p in params:
            fig.add_trace(go.Scatter(x=metrics_df.index, y=metrics_df[p], mode="lines+markers", name=p))
        fig.update_layout(height=500, title=selected_record, xaxis_title="Fase", yaxis_title="Valor")
        st.plotly_chart(fig, use_container_width=True)

        dom = domain_values(metrics_df, method=domain_method)
        if not dom.empty:
            st.markdown("### Dominios normalizados")
            st.dataframe(dom, use_container_width=True)
            fig2 = go.Figure()
            for c in dom.columns:
                fig2.add_trace(go.Scatter(x=dom.index, y=dom[c], mode="lines+markers", name=c))
            fig2.add_hline(y=100, line_dash="dash")
            fig2.update_layout(height=450, title="Dominios. Basal = 100%", yaxis_title="%")
            st.plotly_chart(fig2, use_container_width=True)

with tab5:
    st.subheader("Exportar")
    if long_df.empty:
        st.info("No hay datos para exportar.")
    else:
        valid_summary = pd.DataFrame(records_valid).T.reindex(columns=PHASES)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            xlsx = tmpdir / "resultados_hrv_comparativa.xlsx"
            csv = tmpdir / "resultados_hrv_comparativa.csv"
            zipf = tmpdir / "resultados_hrv_comparativa.zip"

            long_df.to_csv(csv, index=False)
            with pd.ExcelWriter(xlsx) as writer:
                long_df.to_excel(writer, sheet_name="metricas", index=False)
                valid_summary.to_excel(writer, sheet_name="ventanas_validas")
                pd.DataFrame([{"Fase": ph, "Inicio": sec_to_hms(st.session_state.windows[ph][0]), "Fin": sec_to_hms(st.session_state.windows[ph][1]),
                               "Duracion_min": (st.session_state.windows[ph][1]-st.session_state.windows[ph][0])/60} for ph in PHASES]).to_excel(writer, sheet_name="ventanas", index=False)
            with zipfile.ZipFile(zipf, "w", zipfile.ZIP_DEFLATED) as z:
                z.write(xlsx, arcname=xlsx.name)
                z.write(csv, arcname=csv.name)
            st.download_button("Descargar ZIP", zipf.read_bytes(), file_name="resultados_hrv_comparativa.zip", mime="application/zip")
            st.download_button("Descargar Excel", xlsx.read_bytes(), file_name="resultados_hrv_comparativa.xlsx")
