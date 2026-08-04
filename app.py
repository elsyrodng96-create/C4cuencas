import os
from datetime import datetime, date
import ee
import folium
from google.oauth2 import service_account
import streamlit as st
from streamlit_folium import st_folium
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE PÁGINA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SAT Rio David",
    page_icon="🌧️",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
    </style>
""",
    unsafe_allow_html=True,
)

HYBAS_ID = 7090885760

COLORS = {
    "Verde": "#16A34A",
    "Amarilla": "#EAB308",
    "Naranja": "#F97316",
    "Roja": "#DC2626",
}


# -----------------------------------------------------------------------------
# 2. AUTENTICACIÓN Y ADAPTADOR DE FOLIUM PARA GEE
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Conectando con Google Earth Engine...")
def init_earth_engine():
    service_account_info = dict(st.secrets["gee_service_account"])

    if "private_key" in service_account_info:
        pk = service_account_info["private_key"]
        pk = pk.replace("\\n", "\n").strip("'\"")
        service_account_info["private_key"] = pk

    scopes = [
        "https://www.googleapis.com/auth/earthengine",
        "https://www.googleapis.com/auth/devstorage.full_control",
    ]

    credentials = service_account.Credentials.from_service_account_info(
        service_account_info, scopes=scopes
    )

    project_id = service_account_info.get("project_id")
    ee.Initialize(credentials, project=project_id)


def add_ee_layer(self, ee_image_object, vis_params, name):
    map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
    folium.TileLayer(
        tiles=map_id_dict["tile_fetcher"].url_format,
        attr="Google Earth Engine",
        name=name,
        overlay=True,
        control=True,
    ).add_to(self)


folium.Map.add_ee_layer = add_ee_layer


# -----------------------------------------------------------------------------
# 3. PROCESAMIENTO Y DATOS HISTÓRICOS
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Cargando serie temporal histórica CHIRPS...")
def load_historical_data():
    """
    Si subiste el archivo 'chirps_historico.csv' a tu repositorio de GitHub,
    lo leerá directo (mucho más rápido). De lo contrario, se procesará mediante GEE.
    """
    csv_file = "data/chirps_historico.csv"
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        df["fecha"] = pd.to_datetime(df["fecha"])
        return df.sort_values("fecha")
    else:
        # Extracción vía GEE si no existe el CSV
        cuenca = ee.FeatureCollection("WWF/HydroSHEDS/v1/Basins/hybas_9").filter(
            ee.Filter.eq("HYBAS_ID", HYBAS_ID)
        )
        geom = cuenca.geometry()
        
        chirps = (
            ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
            .filterDate("1985-01-01", "2025-12-31")
            .select("precipitation")
        )

        def calc_mean(image):
            media = image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geom,
                scale=5566,
                maxPixels=1e13
            )
            return ee.Feature(None, {
                "fecha": image.date().format("YYYY-MM-dd"),
                "lluvia_mm": media.get("precipitation")
            })

        fc = chirps.map(calc_mean)
        features = fc.getInfo()["features"]
        data = [f["properties"] for f in features]
        df = pd.DataFrame(data)
        df["fecha"] = pd.to_datetime(df["fecha"])
        return df.sort_values("fecha")


@st.cache_resource(show_spinner="Procesando datos de la cuenca...")
def build_layers():
    cuenca = ee.FeatureCollection("WWF/HydroSHEDS/v1/Basins/hybas_9").filter(
        ee.Filter.eq("HYBAS_ID", HYBAS_ID)
    )

    geom = cuenca.geometry()

    dem = ee.Image("USGS/SRTMGL1_003").clip(geom)
    pendiente = ee.Terrain.slope(dem)
    pendiente_baja = ee.Image(1).subtract(pendiente.divide(45).clamp(0, 1))

    flujo = ee.Image("WWF/HydroSHEDS/15ACC").clip(geom)
    flujo_norm = flujo.log10().unitScale(0, 7).clamp(0, 1)

    cobertura = ee.Image("ESA/WorldCover/v200/2021").select("Map").clip(geom)

    uso = (
        cobertura.eq(50)
        .multiply(1.0)
        .add(cobertura.eq(40).multiply(0.7))
        .add(cobertura.eq(10).multiply(0.2))
    )

    riesgo_bruto = (
        flujo_norm.multiply(0.55)
        .add(pendiente_baja.multiply(0.30))
        .add(uso.multiply(0.15))
        .rename("riesgo")
    )

    min_max = riesgo_bruto.reduceRegion(
        reducer=ee.Reducer.minMax(),
        geometry=geom,
        scale=30,
        maxPixels=1e13,
    )

    riesgo_min = ee.Number(min_max.get("riesgo_min"))
    riesgo_max = ee.Number(min_max.get("riesgo_max"))

    riesgo = (
        riesgo_bruto.subtract(riesgo_min)
        .divide(riesgo_max.subtract(riesgo_min))
        .clamp(0, 1)
        .rename("riesgo")
    )

    lluvia_img = (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .select("precipitation")
        .sort("system:time_start", False)
        .first()
        .clip(geom)
    )

    lluvia_mm = ee.Number(
        lluvia_img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=5566,
            maxPixels=1e13,
        ).get("precipitation")
    )

    fecha = ee.Date(lluvia_img.get("system:time_start")).format("YYYY-MM-dd")

    centroid = geom.centroid().coordinates().getInfo()
    lat, lon = centroid[1], centroid[0]

    return cuenca, riesgo, lluvia_mm, fecha, (lat, lon)


@st.cache_data(show_spinner=False)
def current_rainfall():
    _, _, lluvia_mm, fecha, _ = build_layers()
    return float(lluvia_mm.getInfo()), fecha.getInfo()


def alert_level(lluvia_mm, p90, p95, p99):
    if lluvia_mm >= p99:
        return "Roja"
    if lluvia_mm >= p95:
        return "Naranja"
    if lluvia_mm >= p90:
        return "Amarilla"
    return "Verde"


def make_map(view, p90, p95, p99):
    cuenca, riesgo, lluvia_mm, _, center = build_layers()

    mapa = folium.Map(location=center, zoom_start=11)

    if view == "Indice de riesgo":
        mapa.add_ee_layer(
            riesgo,
            {
                "min": 0,
                "max": 1,
                "palette": ["#16A34A", "#FDE047", "#F97316", "#DC2626"],
            },
            "Riesgo de inundación",
        )
    else:
        riesgo_alto = riesgo.gte(0.60)
        alerta = (
            ee.Image.constant(0)
            .where(riesgo_alto.And(ee.Image.constant(lluvia_mm).gte(p90)), 1)
            .where(riesgo_alto.And(ee.Image.constant(lluvia_mm).gte(p95)), 2)
            .where(riesgo_alto.And(ee.Image.constant(lluvia_mm).gte(p99)), 3)
            .clip(cuenca.geometry())
        )

        mapa.add_ee_layer(
            alerta,
            {
                "min": 0,
                "max": 3,
                "palette": [
                    COLORS["Verde"],
                    COLORS["Amarilla"],
                    COLORS["Naranja"],
                    COLORS["Roja"],
                ],
            },
            "Alerta vigente",
        )

    mapa.add_ee_layer(
        cuenca.style(color="#0F172A", fillColor="00000000", width=2),
        {},
        "Límite de cuenca",
    )

    folium.LayerControl().add_to(mapa)
    return mapa


# -----------------------------------------------------------------------------
# 4. APLICACIÓN PRINCIPAL
# -----------------------------------------------------------------------------
def main():
    st.title("🌧️ SAT - Subcuenca del Rio David")
    st.caption(
        "Portal de monitoreo de precipitacion, riesgo y alerta por inundaciones - Chiriqui, Panama"
    )

    # Iniciar Earth Engine y Cargar datos
    try:
        init_earth_engine()
        df_hist = load_historical_data()
        lluvia_mm, fecha_mas_reciente = current_rainfall()
    except Exception as error:
        st.error("No se pudo conectar con Google Earth Engine o cargar datos.")
        st.info("Revisa la clave [gee_service_account] en tus secretos de Streamlit.")
        st.exception(error)
        st.stop()

    # Umbrales históricos (calculados automáticamente, ya no editables en la barra lateral)
    p90 = float(df_hist["lluvia_mm"].quantile(0.90)) if not df_hist.empty else 27.42
    p95 = float(df_hist["lluvia_mm"].quantile(0.95)) if not df_hist.empty else 35.86
    p99 = float(df_hist["lluvia_mm"].quantile(0.99)) if not df_hist.empty else 53.71

    # --- BARRA LATERAL (PANEL DE CONTROL) ---
    with st.sidebar:
        st.header("⚙️ Panel de Control")

        st.subheader("📅 Consulta Histórica de Fecha")
        min_date = df_hist["fecha"].min().date() if not df_hist.empty else date(1985, 1, 1)
        max_date = df_hist["fecha"].max().date() if not df_hist.empty else date(2025, 12, 31)
        fecha_seleccionada = st.date_input(
            "Selecciona un día:",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
        )
        
        # Filtrar precipitación para el día seleccionado en la barra lateral
        lluvia_dia = df_hist[df_hist["fecha"].dt.date == fecha_seleccionada]
        if not lluvia_dia.empty:
            val_lluvia = lluvia_dia["lluvia_mm"].values[0]
            st.metric(
                label=f"Lluvia el {fecha_seleccionada.strftime('%Y-%m-%d')}",
                value=f"{val_lluvia:.2f} mm/día"
            )
            # Nivel de alerta específico para esa fecha seleccionada
            alerta_sel = alert_level(val_lluvia, p90, p95, p99)
            st.markdown(f"**Nivel de Alerta:** :{COLORS[alerta_sel]}[ALERTA {alerta_sel.upper()}]")
        else:
            st.warning("No hay registro para la fecha seleccionada.")

    # Estado de alerta actual
    alerta = alert_level(lluvia_mm, p90, p95, p99)
    st.markdown(
        f"""
        <div style="border-left: 8px solid {COLORS[alerta]}; background: #F8FAFC; padding: 16px 20px; border-radius: 8px;">
            <b style="font-size:1.25rem; color:{COLORS[alerta]}">ALERTA {alerta.upper()}</b><br>
            Lluvia media mas reciente: <b>{lluvia_mm:.2f} mm/dia</b> - Fecha CHIRPS: <b>{fecha_mas_reciente}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    col1, col2, col3, col4 = st.columns(4)
    threshold_name = {"Verde": "Normal", "Amarilla": "P90", "Naranja": "P95", "Roja": "P99"}
    col1.metric("Precipitacion Actual", f"{lluvia_mm:.2f} mm/dia")
    col2.metric("Umbral activo", threshold_name[alerta])
    col3.metric("P90", f"{p90:.2f} mm/dia")
    col4.metric("Riesgo alto", "Indice >= 0.60")

        # Filtrar precipitación para el día seleccionado en la barra lateral
        lluvia_dia = df_hist[df_hist["fecha"].dt.date == fecha_seleccionada]
        if not lluvia_dia.empty:
            val_lluvia = lluvia_dia["lluvia_mm"].values[0]
            st.metric(
                label=f"Lluvia el {fecha_seleccionada.strftime('%Y-%m-%d')}",
                value=f"{val_lluvia:.2f} mm/día"
            )
            # Nivel de alerta específico para esa fecha seleccionada
            alerta_sel = alert_level(val_lluvia, p90, p95, p99)
            st.markdown(f"**Nivel de Alerta:** :{COLORS[alerta_sel]}[ALERTA {alerta_sel.upper()}]")
        else:
            st.warning("No hay registro para la fecha seleccionada.")

    # Estado de alerta actual
    alerta = alert_level(lluvia_mm, p90, p95, p99)

    st.markdown(
        f"""
        <div style="border-left: 8px solid {COLORS[alerta]}; background: #F8FAFC; padding: 16px 20px; border-radius: 8px;">
            <b style="font-size:1.25rem; color:{COLORS[alerta]}">ALERTA {alerta.upper()}</b><br>
            Lluvia media mas reciente: <b>{lluvia_mm:.2f} mm/dia</b> - Fecha CHIRPS: <b>{fecha_mas_reciente}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    col1, col2, col3, col4 = st.columns(4)
    threshold_name = {"Verde": "Normal", "Amarilla": "P90", "Naranja": "P95", "Roja": "P99"}
    col1.metric("Precipitacion Actual", f"{lluvia_mm:.2f} mm/dia")
    col2.metric("Umbral activo", threshold_name[alerta])
    col3.metric("P90", f"{p90:.2f} mm/dia")
    col4.metric("Riesgo alto", "Indice >= 0.60")

    # --- PESTAÑAS ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "Resumen", "Mapas", "Análisis Histórico", "Modelo de riesgo"
    ])

    with tab1:
        st.subheader("Interpretacion operativa")
        messages = {
            "Verde": "Condiciones de lluvia dentro de los valores habituales. Mantener vigilancia rutinaria.",
            "Amarilla": "Lluvia intensa. Incrementar la vigilancia en zonas de riesgo alto.",
            "Naranja": "Lluvia muy intensa. Preparar acciones preventivas y comunicacion con comunidades expuestas.",
            "Roja": "Lluvia extrema. Activar protocolos institucionales de respuesta ante inundaciones.",
        }
        st.info(messages[alerta])
        st.write(f"**Sector seleccionado:** {sector}")

    with tab2:
        view = st.radio("Capa a visualizar", ["Alerta vigente", "Indice de riesgo"], horizontal=True)
        m = make_map(view, p90, p95, p99)
        st_folium(m, width=1200, height=600)

    with tab3:
        st.subheader("📈 Análisis Histórico de Precipitaciones (1985–2024)")

        # BLOQUE 14: Frecuencia anual de eventos extremos (> P99)
        st.markdown("### Frecuencia Anual de Eventos Extremos (> P99)")
        eventos_p99 = df_hist[df_hist["lluvia_mm"] > p99].copy()
        eventos_p99["year"] = eventos_p99["fecha"].dt.year
        frecuencia = (
            eventos_p99.groupby("year")
            .size()
            .reset_index(name="eventos")
        )

        fig1, ax1 = plt.subplots(figsize=(12, 4))
        sns.barplot(data=frecuencia, x="year", y="eventos", color="steelblue", ax=ax1)
        ax1.set_title(f"Frecuencia anual de eventos extremos (> P99 = {p99:.2f} mm)", fontsize=12)
        ax1.set_xlabel("Año")
        ax1.set_ylabel("Número de eventos")
        plt.xticks(rotation=45)
        ax1.grid(axis="y", alpha=0.3)
        st.pyplot(fig1)

        st.markdown("---")

        # BLOQUE 15: Serie Temporal Completa de Precipitaciones Diarias
        st.markdown("### Serie Temporal de Precipitación Diaria y Umbrales")
        fig2, ax2 = plt.subplots(figsize=(14, 5))
        
        ax2.plot(df_hist["fecha"], df_hist["lluvia_mm"], color="steelblue", linewidth=0.5, label="Precipitación diaria")
        ax2.axhline(y=p90, color="gold", linestyle="--", linewidth=1.5, label=f"P90 = {p90:.2f} mm")
        ax2.axhline(y=p95, color="darkorange", linestyle="--", linewidth=1.5, label=f"P95 = {p95:.2f} mm")
        ax2.axhline(y=p99, color="red", linestyle="--", linewidth=1.5, label=f"P99 = {p99:.2f} mm")
        
        # Resaltar en el gráfico la fecha seleccionada en la barra lateral
        if not lluvia_dia.empty:
            fecha_dt = pd.to_datetime(fecha_seleccionada)
            val_ll = lluvia_dia["lluvia_mm"].values[0]
            ax2.scatter([fecha_dt], [val_ll], color="black", s=50, zorder=5, label=f"Seleccionado: {val_ll:.1f} mm")

        ax2.set_title("Serie temporal de precipitación diaria (CHIRPS 1985–2024)", fontsize=14, fontweight="bold")
        ax2.set_xlabel("Año", fontsize=10)
        ax2.set_ylabel("Precipitación (mm/día)", fontsize=10)
        ax2.grid(True, linestyle="--", alpha=0.4)
        ax2.legend()
        plt.tight_layout()
        st.pyplot(fig2)

    with tab4:
        st.subheader("Modelo de riesgo de inundacion")
        st.write("El índice combina acumulación de flujo (55%), pendiente baja (30%) y uso del suelo (15%).")

if __name__ == "__main__":
    main()
