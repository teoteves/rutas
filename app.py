# app.py
import streamlit as st
import pandas as pd

from optimizer_service import solve_routing

# Tomamos como referencia los datos de config_real.py
try:
    from config_real import PLANT as DEFAULT_PLANT
    from config_real import CLIENTS as DEFAULT_CLIENTS
    from config_real import DEMAND_MIN as DEFAULT_DEMAND_MIN
    from config_real import DEMAND_MAX as DEFAULT_DEMAND_MAX
except Exception:
    # Valores de respaldo por si no existe config_real.py
    DEFAULT_PLANT = {
        "id": "PLANTA",
        "address": "UNACEM, Atocongo 2440, Villa María del Triunfo 15822",
    }
    DEFAULT_CLIENTS = []
    DEFAULT_DEMAND_MIN = 0.0
    DEFAULT_DEMAND_MAX = 1e9


st.set_page_config(
    page_title="Optimización de Rutas de Camiones",
    page_icon="🚚",
    layout="wide",
)


def build_default_clients_df() -> pd.DataFrame:
    if not DEFAULT_CLIENTS:
        data = [
            {
                "id": "C1",
                "name": "Cliente 1",
                "address": "Dirección cliente 1",
                "demand_tons": 10.0,
            }
        ]
    else:
        data = DEFAULT_CLIENTS

    df = pd.DataFrame(data)
    # Nos quedamos solo con las columnas que nos interesan
    cols = ["id", "name", "address", "demand_tons"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]


def sidebar_config():
    st.sidebar.header("Configuración global")

    # --- API KEY (no se muestra si está en secrets) ---
    api_key_secret = st.secrets.get("GOOGLE_ROUTES_API_KEY", None)
    if api_key_secret:
        api_key = api_key_secret
        st.sidebar.success("Google Routes API configurada en el servidor.")
    else:
        api_key = st.sidebar.text_input(
            "Google Routes API Key",
            type="password",
            help="Solo se usa en esta sesión; no se guarda en GitHub.",
        )
        if not api_key:
            st.sidebar.warning("⚠️ Falta la API key de Google Routes.")

    max_clientes = st.sidebar.slider(
        "Máximo de clientes por ruta", min_value=1, max_value=5, value=3, step=1
    )
    t_max = st.sidebar.slider(
        "Tiempo máximo por ruta (minutos)",
        min_value=30,
        max_value=300,
        value=180,
        step=10,
    )
    costo_km = st.sidebar.number_input(
        "Costo por km (S/)", min_value=0.0, value=10.0, step=0.5
    )
    costo_min = st.sidebar.number_input(
        "Costo por minuto (S/)", min_value=0.0, value=5.0, step=0.5
    )

    return api_key, max_clientes, t_max, costo_km, costo_min


def main():
    api_key, max_clientes, t_max, costo_km, costo_min = sidebar_config()

    st.title("🚚 Optimización de Rutas de Camiones")
    st.markdown(
        """
Herramienta interactiva para **programar rutas de reparto** desde una planta a varios
clientes en Lima Metropolitana (o cualquier ciudad), usando:

- Distancias y tiempos reales vía **Google Routes API**  
- Modelo de **generación de columnas** con Pyomo  
- Distinción entre **dos tipos de camión** con distinta capacidad  

Ajusta los parámetros y haz click en **Ejecutar optimización**.
"""
    )

    col1, col2 = st.columns(2)

    # ---------------------------------------------------
    # 1) Planta
    # ---------------------------------------------------
    with col1:
        st.subheader("1️⃣ Planta")

        plant_address = st.text_input(
            "Dirección de la planta",
            value=DEFAULT_PLANT.get(
                "address",
                "UNACEM, Atocongo 2440, Villa María del Triunfo 15822",
            ),
            help="Puedes modificarla si lo deseas.",
        )

    # ---------------------------------------------------
    # 2) Flota de camiones
    # ---------------------------------------------------
    with col2:
        st.subheader("2️⃣ Flota de camiones")

        st.markdown("**Tipo 1: camión de mayor capacidad**")
        T1_capacity = st.number_input(
            "Capacidad tipo 1 (t)", min_value=0.0, value=30.0, step=1.0
        )
        T1_fixed = st.number_input(
            "Costo fijo tipo 1 (S/ por viaje)",
            min_value=0.0,
            value=800.0,
            step=50.0,
        )
        T1_count = st.number_input(
            "Cantidad de camiones tipo 1", min_value=0, value=3, step=1
        )

        st.markdown("---")
        st.markdown("**Tipo 2: camión de menor capacidad**")
        T2_capacity = st.number_input(
            "Capacidad tipo 2 (t)", min_value=0.0, value=10.0, step=1.0
        )
        T2_fixed = st.number_input(
            "Costo fijo tipo 2 (S/ por viaje)",
            min_value=0.0,
            value=400.0,
            step=50.0,
        )
        T2_count = st.number_input(
            "Cantidad de camiones tipo 2", min_value=0, value=4, step=1
        )

    truck_types = {
        "T1": {
            "capacity_tons": T1_capacity,
            "fixed_cost": T1_fixed,
            "count": int(T1_count),
        },
        "T2": {
            "capacity_tons": T2_capacity,
            "fixed_cost": T2_fixed,
            "count": int(T2_count),
        },
    }

    # ---------------------------------------------------
    # 3) Clientes (tabla editable)
    # ---------------------------------------------------
    st.subheader("3️⃣ Clientes")

    st.caption(
        f"Demanda permitida por cliente: entre **{DEFAULT_DEMAND_MIN}** y "
        f"**{DEFAULT_DEMAND_MAX}** toneladas."
    )

    df_clients_init = build_default_clients_df()
    df_clients = st.data_editor(
        df_clients_init,
        num_rows="dynamic",
        use_container_width=True,
        key="clients_editor",
    )

    # ---------------------------------------------------
    # 4) Botón de optimización
    # ---------------------------------------------------
    st.markdown("---")
    st.subheader("4️⃣ Ejecutar modelo de optimización")

    if st.button("🚀 Ejecutar optimización", type="primary"):
        # Construimos lista limpia de clientes
        clients_list = []
        for _, row in df_clients.iterrows():
            cid = str(row.get("id", "")).strip()
            addr = str(row.get("address", "")).strip()
            try:
                demand = float(row.get("demand_tons", 0.0))
            except Exception:
                demand = 0.0

            if not cid or not addr:
                continue
            clients_list.append(
                {
                    "id": cid,
                    "name": str(row.get("name", cid)),
                    "address": addr,
                    "demand_tons": demand,
                }
            )

        if not api_key:
            st.warning("Debes proporcionar una API key de Google Routes.")
            return

        if not clients_list:
            st.warning("Debes definir al menos un cliente con demanda positiva.")
            return

        if (int(T1_count) + int(T2_count)) == 0:
            st.warning("Debes disponer de al menos un camión en la flota.")
            return

        route_params = {
            "max_clientes_por_ruta": max_clientes,
            "t_max_min": t_max,
            "costo_por_km": costo_km,
            "costo_por_min": costo_min,
            "demand_min": DEFAULT_DEMAND_MIN,
            "demand_max": DEFAULT_DEMAND_MAX,
        }

        try:
            result = solve_routing(
                plant_address=plant_address,
                clients=clients_list,
                truck_types=truck_types,
                route_params=route_params,
                api_key=api_key,
            )
        except ValueError as e:
            # Errores "esperados": falta de capacidad, rutas imposibles, etc.
            st.warning(str(e))
        except Exception as e:
            # Errores inesperados
            st.error(f"Ocurrió un error inesperado durante la optimización:\n\n{e}")
        else:
            st.success("Optimización completada ✅")

            st.metric(
                "Costo total mínimo",
                f"S/ {result['objective']:.2f}",
                help="Incluye costos fijos de camiones y costos variables de rutas.",
            )

            assignments = result.get("assignments", [])
            if assignments:
                df_sol = pd.DataFrame(assignments)
                st.subheader("Detalle de rutas seleccionadas")
                st.dataframe(df_sol, use_container_width=True)
            else:
                st.info("El modelo terminó sin seleccionar ninguna ruta (solución vacía).")


if __name__ == "__main__":
    main()
