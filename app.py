import streamlit as st
import pandas as pd

from optimizer_service import solve_routing
from config_real import (
    PLANT,
    CLIENTS,
    TRUCK_TYPES,
    MAX_CLIENTES_POR_RUTA,
    T_MAX_MIN,
    COSTO_POR_KM,
    COSTO_POR_MIN,
    DEMAND_MIN,
    DEMAND_MAX,
)


st.set_page_config(
    page_title="Optimización de rutas de reparto",
    layout="wide",
)


st.title("🚚 Optimización de rutas de reparto desde planta de cementos")
st.write(
    "Esta herramienta construye rutas de reparto desde la **planta** hacia "
    "los **clientes**, combinando costos fijos por camión y costos variables "
    "por distancia y tiempo. Usa la API de Google Routes para obtener "
    "distancias y tiempos realistas."
)

# --------------------------------------------------------------------
# API key (nunca se muestra, se lee desde st.secrets)
# --------------------------------------------------------------------
api_key = st.secrets.get("GOOGLE_MAPS_API_KEY", "")
if not api_key:
    st.warning(
        "No se encontró `GOOGLE_MAPS_API_KEY` en `st.secrets`. "
        "Configura tu API key de Google Routes en los secretos de la app."
    )

# --------------------------------------------------------------------
# Sidebar: parámetros de planta, flota y rutas
# --------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Parámetros generales")

    plant_address = st.text_input(
        "Dirección de la planta",
        value=PLANT.get("address", ""),
        help="Esta dirección se envía a la API de Google Routes.",
    )

    st.subheader("🚛 Tipos de camión")
    st.caption("Puedes pensar en **tipo 1** como camión de mayor capacidad y "
               "**tipo 2** como camión de menor capacidad.")

    truck_types_input = {}
    for tname, info in TRUCK_TYPES.items():
        st.markdown(f"**Tipo {tname}**")
        cap = st.number_input(
            f"Capacidad tipo {tname} (ton)",
            min_value=0.0,
            value=float(info["capacity_tons"]),
            step=1.0,
            key=f"cap_{tname}",
        )
        fix = st.number_input(
            f"Costo fijo tipo {tname} (S/)",
            min_value=0.0,
            value=float(info["fixed_cost"]),
            step=50.0,
            key=f"fix_{tname}",
        )
        cnt = st.number_input(
            f"Número de camiones tipo {tname}",
            min_value=0,
            value=int(info["count"]),
            step=1,
            key=f"cnt_{tname}",
        )

        truck_types_input[tname] = {
            "capacity_tons": cap,
            "fixed_cost": fix,
            "count": cnt,
        }

    st.subheader("🧭 Parámetros de rutas")
    max_cli = st.slider(
        "Máximo de clientes por ruta",
        min_value=1,
        max_value=5,
        value=MAX_CLIENTES_POR_RUTA,
        step=1,
    )
    t_max = st.number_input(
        "Tiempo máximo por ruta (minutos)",
        min_value=10.0,
        max_value=600.0,
        value=float(T_MAX_MIN),
        step=10.0,
    )
    cost_km = st.number_input(
        "Costo por km (S/)",
        min_value=0.0,
        max_value=100.0,
        value=float(COSTO_POR_KM),
        step=1.0,
    )
    cost_min = st.number_input(
        "Costo por minuto (S/)",
        min_value=0.0,
        max_value=100.0,
        value=float(COSTO_POR_MIN),
        step=1.0,
    )

    st.subheader("📦 Rango de demanda por cliente (validación)")
    dmin = st.number_input(
        "Demanda mínima (ton)",
        min_value=0.0,
        max_value=1000.0,
        value=float(DEMAND_MIN),
        step=0.5,
    )
    dmax = st.number_input(
        "Demanda máxima (ton)",
        min_value=0.0,
        max_value=1000.0,
        value=float(DEMAND_MAX),
        step=0.5,
    )

    route_params = {
        "max_clientes_por_ruta": max_cli,
        "t_max_min": t_max,
        "costo_por_km": cost_km,
        "costo_por_min": cost_min,
        "demand_min": dmin,
        "demand_max": dmax,
    }

# --------------------------------------------------------------------
# Tabla editable de clientes
# --------------------------------------------------------------------
st.header("📍 Clientes y demandas")

default_clients_df = pd.DataFrame(
    [
        {
            "id": c["id"],
            "name": c.get("name", c["id"]),
            "address": c["address"],
            "demand_tons": c["demand_tons"],
        }
        for c in CLIENTS
    ]
)

st.caption(
    "Edita la tabla, agrega o elimina filas según sea necesario. "
    "Solo se considerarán los clientes con dirección no vacía y demanda positiva."
)

clients_df = st.data_editor(
    default_clients_df,
    num_rows="dynamic",
    use_container_width=True,
    key="clients_editor",
)

# --------------------------------------------------------------------
# Botón principal
# --------------------------------------------------------------------
if st.button("🚀 Optimizar rutas"):
    if not api_key:
        st.error("Falta la API key de Google Routes. Configúrala en `st.secrets`.")
    elif not plant_address.strip():
        st.error("Debes ingresar la dirección de la planta.")
    else:
        # Convertimos DataFrame en lista de dicts
        clients_list = []
        for _, row in clients_df.iterrows():
            address = str(row.get("address", "")).strip()
            if not address:
                continue
            cid = str(row.get("id", "")).strip()
            if not cid:
                continue
            name = str(row.get("name", cid)).strip()
            try:
                demand = float(row.get("demand_tons", 0.0))
            except Exception:
                demand = 0.0
            if demand <= 0:
                continue
            clients_list.append(
                {
                    "id": cid,
                    "name": name,
                    "address": address,
                    "demand_tons": demand,
                }
            )

        if not clients_list:
            st.error("Debes ingresar al menos un cliente con dirección y demanda positiva.")
        else:
            with st.spinner("Resolviendo el modelo de optimización..."):
                try:
                    result = solve_routing(
                        plant_address=plant_address,
                        clients=clients_list,
                        truck_types=truck_types_input,
                        route_params=route_params,
                        api_key=api_key,
                    )
                except Exception as e:
                    st.error(str(e))
                else:
                    st.success("Optimización completada.")

                    st.subheader("📊 Resumen global")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Costo total mínimo (S/)",
                            f"{result['objective']:.1f}",
                        )
                    with col2:
                        st.metric(
                            "Demanda total (ton)",
                            f"{result['total_demand']:.1f}",
                        )
                    with col3:
                        st.metric(
                            "Capacidad total (ton)",
                            f"{result['total_capacity']:.1f}",
                        )

                    st.subheader("🧾 Asignación de rutas por camión")
                    if result["assignments"]:
                        df_assign = pd.DataFrame(result["assignments"])
                        st.dataframe(df_assign, use_container_width=True)
                    else:
                        st.info(
                            "El modelo no activó ninguna ruta. Revisa los parámetros de flota y rutas."
                        )
