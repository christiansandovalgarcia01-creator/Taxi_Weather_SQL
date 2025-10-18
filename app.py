import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import plotly.express as px

# -------- Config --------
st.set_page_config(page_title="Sprint 8 — Chicago Taxi & Weather (2017)", layout="wide")
DATA_DIR = Path("data")
URL_WEATHER = "https://practicum-content.s3.us-west-1.amazonaws.com/data-analyst-eng/moved_chicago_weather_2017.html"

# -------- Helpers --------
@st.cache_data(show_spinner=False)
def load_weather_online(url: str):
    tables = pd.read_html(url, attrs={"id": "weather_records"})
    if not tables:
        raise ValueError("No se encontró una tabla con id='weather_records'.")
    return tables[0].copy()

def try_load_csv(path: Path, dtype_map=None, numeric_cols=None, datetime_cols=None):
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if dtype_map:
        for col, dt in dtype_map.items():
            if col in df.columns:
                df[col] = df[col].astype(dt)
    if numeric_cols:
        for col in numeric_cols or []:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    if datetime_cols:
        for col in datetime_cols or []:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def welch_or_permutation(a: np.ndarray, b: np.ndarray, alpha=0.05, seed=42, perms=5000):
    a = a.astype(float); b = b.astype(float)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    try:
        from scipy import stats
        t, p = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
        return {"test": "Welch t-test", "stat": float(t), "p": float(p)}
    except Exception:
        rng = np.random.default_rng(seed)
        obs = abs(a.mean() - b.mean())
        pooled = np.concatenate([a, b])
        na = len(a)
        count = 0
        for _ in range(perms):
            rng.shuffle(pooled)
            diff = abs(pooled[:na].mean() - pooled[na:].mean())
            if diff >= obs:
                count += 1
        p = (count + 1) / (perms + 1)
        return {"test": "Permutation", "stat": float("nan"), "p": float(p)}

# -------- Sidebar --------
st.sidebar.title("Sprint 8 — Dashboard")
st.sidebar.markdown(
"""
**Autor:** Christian Sandoval  
**Objetivo:** Analizar clima y taxis en Chicago (Nov 2017).  
**Stack:** Python · Pandas · Plotly · Streamlit
"""
)
with st.sidebar.expander("Datos esperados"):
    st.write("- `weather_records` desde la web (tabla HTML id='weather_records').")
    st.write("- `data/project_sql_result_01.csv` (compañías vs viajes, Nov 15–16).")
    st.write("- `data/project_sql_result_04.csv` (barrios por promedio, Nov).")
    st.write("- `data/project_sql_result_07.csv` (Loop→O'Hare sábados, Good/Bad).")

# -------- Header --------
st.title("Chicago Taxi & Weather — 2017 (Sprint 8)")
st.caption("La app funciona solo con el clima online. Si agregas los CSV a /data, aparecen pestañas extra automáticamente.")

# -------- Weather (online) --------
weather_records = None
try:
    weather_records = load_weather_online(URL_WEATHER)
    st.success("Clima cargado desde la web (weather_records).")
except Exception as e:
    st.error(f"No se pudo cargar el clima desde la web: {e}")

# -------- CSV opcionales --------
df01 = try_load_csv(Path("data/project_sql_result_01.csv"),
                    dtype_map={"company_name": "string"},
                    numeric_cols=["trips_amount"])
df04 = try_load_csv(Path("data/project_sql_result_04.csv"),
                    dtype_map={"dropoff_location_name": "string"},
                    numeric_cols=["average_trips"])
df07 = try_load_csv(Path("data/project_sql_result_07.csv"),
                    numeric_cols=["duration_seconds"],
                    datetime_cols=["start_ts"])

if df07 is not None:
    if "weather_conditions" in df07.columns:
        df07["weather_conditions"] = df07["weather_conditions"].astype("string").str.strip().str.title()
    df07["duration_min"] = df07["duration_seconds"] / 60.0
    df07["weekday"] = df07["start_ts"].dt.weekday  # Monday=0 ... Saturday=5

# -------- Tabs --------
tabs = ["Clima (web)"]
if df01 is not None: tabs.append("Top compañías")
if df04 is not None: tabs.append("Top barrios")
if df07 is not None: tabs.append("Hipótesis (Loop→O'Hare, sábados)")
t = st.tabs(tabs)

# Tab 0: Clima
with t[0]:
    st.subheader("Tabla de clima — weather_records")
    if weather_records is not None:
        st.dataframe(weather_records.head(30), use_container_width=True)
    else:
        st.info("Si el host bloquea lecturas web, sube un respaldo CSV a /data/weather_records.csv y adaptemos.")

idx = 1
# Tab: Top compañías
if df01 is not None:
    with t[idx]:
        st.subheader("Top compañías por # de viajes (Nov 15–16, 2017)")
        topn = st.slider("Empresas a mostrar:", 5, 30, 20, 1)
        df01_top = df01.sort_values("trips_amount", ascending=False).head(topn)
        st.metric("Total viajes (Top N)", int(df01_top["trips_amount"].sum()))
        st.metric("Compañías en Top N", len(df01_top))
        fig = px.bar(df01_top, x="trips_amount", y="company_name", orientation="h",
                     title="Top compañías por viajes")
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=600)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df01_top, use_container_width=True)
    idx += 1
else:
    st.info("Para ver 'Top compañías', sube data/project_sql_result_01.csv al repo.")

# Tab: Top barrios
if df04 is not None:
    with t[idx]:
        st.subheader("Top barrios por viajes promedio (Nov 2017)")
        k = st.slider("Barrios a mostrar:", 5, 20, 10, 1)
        topk = df04.sort_values("average_trips", ascending=False).head(k)
        st.metric("Media (Top k)", f"{topk['average_trips'].mean():.1f}")
        st.metric("Máximo (Top k)", f"{topk['average_trips'].max():.0f}")
        fig2 = px.bar(topk, x="average_trips", y="dropoff_location_name", orientation="h",
                      title="Barrios por viajes promedio")
        fig2.update_layout(yaxis={"categoryorder": "total ascending"}, height=550)
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(topk, use_container_width=True)
    idx += 1
else:
    st.info("Para ver 'Top barrios', sube data/project_sql_result_04.csv al repo.")

# Tab: Hipótesis
if df07 is not None:
    with t[idx]:
        st.subheader("¿Cambia la duración promedio en sábados lluviosos?")
        alpha = st.slider("α (nivel de significancia)", 0.001, 0.10, 0.05, 0.001)
        sat = df07[df07["weekday"] == 5].copy()
        good = sat.loc[sat["weather_conditions"] == "Good", "duration_min"].dropna().values
        bad  = sat.loc[sat["weather_conditions"] == "Bad",  "duration_min"].dropna().values
        res = welch_or_permutation(bad, good, alpha=alpha)
        decision = "Rechazamos H0" if res["p"] < alpha else "No rechazamos H0"
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("n Good", len(good))
        c2.metric("n Bad",  len(bad))
        c3.metric("Δ medias (Bad–Good)", f"{(np.nanmean(bad) - np.nanmean(good)):.2f} min")
        c4.metric("p-value", f"{res['p']:.4f}")
        st.write(f"**Test:** {res['test']}  |  **Decisión (α={alpha}):** **{decision}**")
        plot_df = pd.DataFrame({
            "duration_min": np.concatenate([good, bad]),
            "weather": (["Good"] * len(good)) + (["Bad"] * len(bad))
        })
        fig3 = px.box(plot_df, x="weather", y="duration_min", points="outliers",
                      title="Duración (min) — Sábados, Loop → O'Hare")
        st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("Para ver la pestaña de hipótesis, sube data/project_sql_result_07.csv al repo.")
