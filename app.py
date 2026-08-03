import json
from datetime import datetime
from io import BytesIO
from pathlib import Path

import ee
import geemap.foliumap as geemap
from google.oauth2 import service_account
import pandas as pd
import plotly.express as px
import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


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
HISTORY_FILE = Path("data/chirps_historico.csv")

COLORS = {
    "Verde": "#16A34A",
    "Amarilla": "#EAB308",
    "Naranja": "#F97316",
    "Roja": "#DC2626",
}


@st.cache_resource(show_spinner="Conectando con Google Earth Engine...")
def init_earth_engine():
  # 1. Copiamos el diccionario desde los secretos
  service_account_info = dict(st.secrets["gee_service_account"])

  # 2. Limpieza de la clave privada (soporta saltos de línea codificados)
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

  # 6. Inicializamos con el proyecto dinámico
  ee.Initialize(credentials, project=project_id)


@st.cache_resource(show_spinner="Procesando mapa de riesgo...")
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

  return cuenca, riesgo, lluvia_mm, fecha


@st.cache_data(show_spinner=False)
def current_rainfall():
  _, _, lluvia_mm, fecha = build_layers()
  return float(lluvia_mm.getInfo()), fecha.getInfo()


@st.cache_data(show_spinner=False)
def load_history():
  if not HISTORY_FILE.exists():
    return pd.DataFrame(columns=["fecha", "lluvia_mm"])

  history = pd.read_csv(HISTORY_FILE)
  history["fecha"] = pd.to_datetime(history["fecha"])
  return history.sort_values("fecha")


def alert_level(lluvia_mm, p90, p95, p99):
  if lluvia_mm >= p99:
    return "Roja"

  if lluvia_mm >= p95:
    return "Naranja"

  if lluvia_mm >= p90:
    return "Amarilla"

  return "Verde"


def make_map(view, p90, p95, p99):
  cuenca, riesgo, lluvia_mm, _ = build_layers()

  mapa = geemap.Map()
  mapa.centerObject(cuenca, 10)

  if view == "Indice de riesgo":
    mapa.addLayer(
        riesgo,
        {
            "min": 0,
            "max": 1,
            "palette": [
                "#16A34A",
                "#FDE047",
                "#F97316",
                "#DC2626",
            ],
        },
        "Riesgo de inundacion",
    )

  else:
    riesgo_alto = riesgo.gte(0.60)

    alerta = (
        ee.Image.constant(0)
        .where(
            riesgo_alto.And(ee.Image.constant(lluvia_mm).gte(p90)),
            1,
        )
        .where(
            riesgo_alto.And(ee.Image.constant(lluvia_mm).gte(p95)),
            2,
        )
        .where(
            riesgo_alto.And(ee.Image.constant(lluvia_mm).gte(p99)),
            3,
        )
        .clip(cuenca.geometry())
    )

    mapa.addLayer(
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

    mapa.add_legend(
        title="Nivel de alerta",
        legend_dict={
            "Verde - normal": COLORS["Verde"],
            "Amarilla - P90": COLORS["Amarilla"],
            "Naranja - P95": COLORS["Naranja"],
            "Roja - P99": COLORS["Roja"],
        },
    )

  mapa.addLayer(
      cuenca.style(
          color="#0F172A",
          fillColor="00000000",
          width=2,
      ),
      {},
      "Limite de cuenca",
  )

  return mapa


def create_pdf(alerta, lluvia_mm, fecha, sector, p90, p95, p99):
  buffer = BytesIO()

  document = SimpleDocTemplate(
      buffer,
      pagesize=letter,
      rightMargin=40,
      leftMargin=40,
      topMargin=40,
      bottomMargin=40,
  )

  styles = getSampleStyleSheet()

  content = [
      Paragraph(
          "Sistema de Alerta Temprana - Subcuenca del rio David",
          styles["Title"],
      ),
      Paragraph(
          f"Reporte generado: {datetime.now():%Y-%m-%d %H:%M}",
          styles["Normal"],
      ),
      Spacer(1, 0.2 * inch),
      Paragraph(
          f"<b>Estado:</b> ALERTA {alerta.upper()}",
          styles["Heading2"],
      ),
      Paragraph(
          f"<b>Sector:</b> {sector}",
          styles["Normal"],
      ),
      Paragraph(
          f"<b>Lluvia CHIRPS:</b> {lluvia_mm:.2f} mm/dia ({fecha})",
          styles["Normal"],
      ),
      Paragraph(
          f"<b>Umbrales:</b> P90={p90:.2f}; "
          f"P95={p95:.2f}; P99={p99:.2f} mm/dia",
          styles["Normal"],
      ),
      Spacer(1, 0.2 * inch),
      Paragraph("Modelo de riesgo", styles["Heading2"]),
      Paragraph(
          "El indice combina acumulacion de flujo (55 %), "
          "pendiente baja (30 %) y uso/cobertura del suelo (15 %).",
          styles["Normal"],
      ),
      Paragraph(
          "Las alertas espaciales se activan donde el riesgo "
          "es igual o superior a 0.60.",
          styles["Normal"],
      ),
      Spacer(1, 0.2 * inch),
      Paragraph(
          "Nota: prototipo academico. Las alertas requieren "
          "validacion institucional antes de su uso operativo.",
          styles["Italic"],
      ),
  ]

  document.build(content)
  return buffer.getvalue()


# -----------------------------------------------------------------------------
# 4. APLICACIÓN PRINCIPAL
# -----------------------------------------------------------------------------
def main():
  st.title("🌧️ SAT - Subcuenca del Rio David")
  st.caption(
      "Portal de monitoreo de precipitacion, riesgo y alerta "
      "por inundaciones - Chiriqui, Panama"
  )

  with st.sidebar:
    st.header("Panel de control")

    sector = st.selectbox(
        "Sector",
        [
            "Toda la subcuenca",
            "Cuenca alta - Boquete",
            "Cuenca media - Dolega",
            "Cuenca baja - David",
        ],
    )

    st.subheader("Umbrales historicos CHIRPS")

    p90 = st.number_input(
        "P90 (mm/dia)",
        min_value=0.0,
        value=27.35,
        step=0.01,
    )

    p95 = st.number_input(
        "P95 (mm/dia)",
        min_value=p90,
        value=35.86,
        step=0.01,
    )

    p99 = st.number_input(
        "P99 (mm/dia)",
        min_value=p95,
        value=53.86,
        step=0.01,
    )

    st.caption("Calculados con CHIRPS diario, 1985-2024.")

  try:
    init_earth_engine()
    lluvia_mm, fecha = current_rainfall()

  except Exception as error:
    st.error("No se pudo conectar con Google Earth Engine.")
    st.info(
        "Revisa la clave [gee_service_account] en tus secretos de Streamlit"
        " (.streamlit/secrets.toml)."
    )
    st.exception(error)
    st.stop()

  alerta = alert_level(lluvia_mm, p90, p95, p99)

  st.markdown(
      f"""
        <div style="
            border-left: 8px solid {COLORS[alerta]};
            background: #F8FAFC;
            padding: 16px 20px;
            border-radius: 8px;
        ">
            <b style="font-size:1.25rem; color:{COLORS[alerta]}">
                ALERTA {alerta.upper()}
            </b><br>
            Lluvia media mas reciente:
            <b>{lluvia_mm:.2f} mm/dia</b>
            - Fecha CHIRPS: <b>{fecha}</b>
        </div>
        """,
      unsafe_allow_html=True,
  )

  threshold_name = {
      "Verde": "Normal",
      "Amarilla": "P90",
      "Naranja": "P95",
      "Roja": "P99",
  }

  col1, col2, col3, col4 = st.columns(4)

  col1.metric("Precipitacion", f"{lluvia_mm:.2f} mm/dia")
  col2.metric("Umbral activo", threshold_name[alerta])
  col3.metric("P90", f"{p90:.2f} mm/dia")
  col4.metric("Riesgo alto", "Indice >= 0.60")

  tab1, tab2, tab3, tab4, tab5 = st.tabs([
      "Resumen",
      "Mapas",
      "Precipitacion",
      "Riesgo y umbrales",
      "Reportes",
  ])

  with tab1:
    st.subheader("Interpretacion operativa")

    messages = {
        "Verde": (
            "Condiciones de lluvia dentro de los valores habituales. "
            "Mantener vigilancia rutinaria."
        ),
        "Amarilla": (
            "Lluvia intensa. Incrementar la vigilancia "
            "en zonas de riesgo alto."
        ),
        "Naranja": (
            "Lluvia muy intensa. Preparar acciones preventivas "
            "y comunicacion con comunidades expuestas."
        ),
        "Roja": (
            "Lluvia extrema. Activar protocolos institucionales "
            "de respuesta ante inundaciones."
        ),
    }

    st.info(messages[alerta])

    st.markdown("**Fuentes del portal**")
    st.write(
        "Precipitacion diaria: CHIRPS. "
        "Riesgo: pendiente baja, acumulacion de flujo "
        "y cobertura del suelo."
    )

  with tab2:
    view = st.radio(
        "Capa a visualizar",
        ["Alerta vigente", "Indice de riesgo"],
        horizontal=True,
    )

    st.caption(
        "La alerta aparece solamente donde coinciden "
        "lluvia critica y riesgo espacial alto."
    )

    make_map(view, p90, p95, p99).to_streamlit(height=610)

  with tab3:
    st.subheader("Precipitacion historica")

    history = load_history()

    if history.empty:
      st.warning(
          "No se encontro data/chirps_historico.csv. "
          "Sube el archivo exportado desde Colab."
      )

    else:
      start, end = st.date_input(
          "Periodo",
          value=(
              history["fecha"].min().date(),
              history["fecha"].max().date(),
          ),
      )

      period = history[
          (history["fecha"].dt.date >= start)
          & (history["fecha"].dt.date <= end)
      ]

      chart = px.line(
          period,
          x="fecha",
          y="lluvia_mm",
          labels={
              "fecha": "Fecha",
              "lluvia_mm": "Precipitacion (mm/dia)",
          },
      )

      chart.add_hline(
          y=p90,
          line_dash="dash",
          line_color=COLORS["Amarilla"],
          annotation_text="P90",
      )

      chart.add_hline(
          y=p95,
          line_dash="dash",
          line_color=COLORS["Naranja"],
          annotation_text="P95",
      )

      chart.add_hline(
          y=p99,
          line_dash="dash",
          line_color=COLORS["Roja"],
          annotation_text="P99",
      )

      chart.update_layout(
          height=420,
          margin=dict(l=10, r=10, t=20, b=10),
      )

      st.plotly_chart(chart, use_container_width=True)

  with tab4:
    st.subheader("Modelo de riesgo de inundacion")

    st.write(
        "El indice se normaliza entre 0 y 1. "
        "Las zonas con indice igual o superior a 0.60 "
        "se consideran de riesgo alto."
    )

    weights = pd.DataFrame({
        "Factor": [
            "Acumulacion de flujo",
            "Pendiente baja",
            "Uso/cobertura del suelo",
        ],
        "Peso": ["55 %", "30 %", "15 %"],
        "Interpretacion": [
            "Proximidad funcional a la red de drenaje",
            "Mayor propension a acumulacion de agua",
            "Sensibilidad urbana y agropecuaria",
        ],
    })

    st.dataframe(weights, use_container_width=True, hide_index=True)

    thresholds = pd.DataFrame({
        "Nivel": ["Amarilla", "Naranja", "Roja"],
        "Regla de lluvia": [
            f">= P90 ({p90:.2f} mm/dia)",
            f">= P95 ({p95:.2f} mm/dia)",
            f">= P99 ({p99:.2f} mm/dia)",
        ],
        "Condicion espacial": ["Riesgo >= 0.60"] * 3,
    })

    st.dataframe(thresholds, use_container_width=True, hide_index=True)

  with tab5:
    st.subheader("Reportes descargables")
    st.write(
        "Descarga un resumen operativo en PDF o los datos "
        "actuales en CSV."
    )

    pdf = create_pdf(
        alerta,
        lluvia_mm,
        fecha,
        sector,
        p90,
        p95,
        p99,
    )

    export = pd.DataFrame([{
        "sector": sector,
        "alerta_actual": alerta,
        "lluvia_chirps_mm_dia": lluvia_mm,
        "fecha_chirps": fecha,
        "p90_mm_dia": p90,
        "p95_mm_dia": p95,
        "p99_mm_dia": p99,
    }])

    col_pdf, col_csv = st.columns(2)

    col_pdf.download_button(
        "Descargar reporte PDF",
        pdf,
        f"reporte_SAT_Rio_David_{datetime.now():%Y%m%d}.pdf",
        "application/pdf",
        use_container_width=True,
    )

    col_csv.download_button(
        "Descargar datos CSV",
        export.to_csv(index=False).encode("utf-8-sig"),
        f"datos_SAT_Rio_David_{datetime.now():%Y%m%d}.csv",
        "text/csv",
        use_container_width=True,
    )

  st.divider()

  st.caption(
      "Prototipo academico. Las alertas requieren validacion "
      "institucional y no sustituyen los avisos oficiales de SINAPROC."
  )


if __name__ == "__main__":
  main()
