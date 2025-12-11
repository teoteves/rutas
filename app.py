# app.py
import streamlit as st
import pandas as pd

from optimizer_service import solve_routing

st.set_page_config(
    page_title="Optimización de Rutas de Camiones",
    page_icon="🚚",
    layout="wide",
)

st.title("🚚 Optimización de Rutas de Camiones")
st.markdown(
    """
Herramienta interactiva para **programar rutas de reparto** desde una planta
a varios clientes en Lima Metropolitana (o cualquier ciudad), usando:
- Distancias y tiempos reales vía **Google Routes API**
- Modelo de **generación de columnas** con Pyomo
- Distinción entre camiones de **30 t** y **10 t**

Ajusta los parámetros y haz click en **Ejecutar optimización**.
"""
)

# =======================
# Sidebar: API key y parámetros globales
# =======================
st.sidebar.header("Configuración global")

# 1) Intentar leer la key desde secretos del servidor
api_key = st.secrets.get("GOOGLE_ROUTES_API_KEY", None)

if api_key:
    st.sidebar.success("Google Routes API configurada en el servidor.")
else:
    # Solo si NO hay secreto configurado, permitimos escribirla a mano
    api_key = st.sidebar.text_input(
        "Google Routes API Key",
        type="password",
        help="Solo se usa en esta sesión; no se guarda en GitHub.",
    )
    if not api_key:
        st.sidebar.warning("⚠️ Falta la API key de Google Routes.")

max_clientes_por_ruta = st.sidebar.slider(
    "Máximo de clientes por ruta", min_value=1, max_value=5, value=3, step=1
)
t_max_min = st.sidebar.slider(
    "Tiempo máximo por ruta (minutos)", min_value=30, max_value=300, value=180, step=15
)
costo_por_km = st.sidebar.number_input(
    "Costo por km (S/)", min_value=0.0, value=10.0, step=0.5
)
costo_por_min = st.sidebar.number_input(
    "Costo por minuto (S/)", min_value=0.0, value=5.0, step=0.5
)

# =======================
# Layout principal
# =======================
col_plant, col_trucks = st.columns(2)

with col_plant:
    st.subheader("1️⃣ Planta")
    plant_address = st.text_input(
        "Dirección de la planta",
        value="UNACEM, Atocongo 2440, Villa María del Triunfo 15822",
    )

with col_trucks:
    st.subheader("2️⃣ Flota de camiones")

    st.markdown("**Camiones de 30 toneladas (T30)**")
    T30_capacity = st.number_input("Capacidad T30 (t)", value=30.0, step=1.0)
    T30_fixed = st.number_input("Costo fijo T30 (S/ por viaje)", value=800.0, step=50.0)
    T30_count = st.number_input("Cantidad de camiones T30", value=3, step=1, min_value=0)

    st.markdown("---")
    st.markdown("**Camiones de 10 toneladas (T10)**")
    T10_capacity = st.number_input("Capacidad T10 (t)", value=10.0, step=1.0)
    T10_fixed = st.number_input("Costo fijo T10 (S/ por viaje)", value=400.0, step=50.0)
    T10_count = st.number_input("Cantidad de camiones T10", value=4, step=1, min_value=0)

st.subheader("3️⃣ Clientes")

st.markdown(
    """
Define tus clientes: **ID**, nombre, dirección y demanda en toneladas.
Puedes editar la tabla directamente.
"""
)

# DataFrame inicial de ejemplo
df_clients_init = pd.DataFrame([
    {"id": "C1", "name": "Promart Ate",
     "address": "Promart Ate, Av. la Molina Cdra. 3, Ate 15012",
     "demand_tons": 15.0},
    {"id": "C2", "name": "Promart La Molina",
     "address": "Promart La Molina, Cruce de Av. la Molina con La Molina 15026",
     "demand_tons": 10.0},
    {"id": "C3", "name": "Promart Chorrillos",
     "address": "Promart Chorrillos, Av. Guardia Civil Sur 927, Urb. Chorrillos 15056",
     "demand_tons": 12.0},
])

df_clients = st.data_editor(
    df_clients_init,
    num_rows="dynamic",
    use_container_width=True,
    key="clients_editor",
)

# =======================
# Botón de ejecución
# =======================

st.markdown("----")
run_button = st.button("🧠 Ejecutar optimización", type="primary")

if run_button:
    if not api_key:
        st.error("Debes ingresar la Google Routes API Key.")
    elif df_clients.empty:
        st.error("Debes definir al menos un cliente.")
    else:
        with st.spinner("Calculando rutas óptimas... ⏳"):
            # preparar datos para el servicio
            clients_list = df_clients.to_dict(orient="records")
            truck_types = {
                "T30": {
                    "capacity_tons": T30_capacity,
                    "fixed_cost": T30_fixed,
                    "count": int(T30_count),
                },
                "T10": {
                    "capacity_tons": T10_capacity,
                    "fixed_cost": T10_fixed,
                    "count": int(T10_count),
                },
            }
            route_params = {
                "max_clientes_por_ruta": int(max_clientes_por_ruta),
                "t_max_min": int(t_max_min),
                "costo_por_km": float(costo_por_km),
                "costo_por_min": float(costo_por_min),
            }

            try:
                result = solve_routing(
                    plant_address=plant_address,
                    clients=clients_list,
                    truck_types=truck_types,
                    route_params=route_params,
                    api_key=api_key,
                )
            except Exception as e:
                st.error(f"Ocurrió un error durante la optimización:\n\n{e}")
            else:
                st.success("Optimización completada ✅")

                # =======================
                # Mostrar resultados
                # =======================
                lp_info = result["lp_relaxation"]
                mip = result["mip_solution"]

                st.subheader("4️⃣ Resumen del problema")
                total_demand = df_clients["demand_tons"].sum()
                n_clients = len(df_clients)
                n_trucks = int(T30_count + T10_count)
                st.metric("Número de clientes", n_clients)
                st.metric("Demanda total (t)", total_demand)
                st.metric("Número total de camiones", n_trucks)

                st.subheader("5️⃣ Resultado óptimo (MIP final)")
                col_z, col_iter = st.columns(2)
                with col_z:
                    st.metric("Costo total mínimo (S/)", f"{mip['z_star']:.2f}")
                with col_iter:
                    st.metric("Iteraciones CG (LP)", lp_info["iterations"])

                df_assign = pd.DataFrame(mip["assignments"])
                if not df_assign.empty:
                    # Expandir lista de clientes a string para mostrar
                    df_assign["clientes_str"] = df_assign["clients"].apply(
                        lambda xs: ", ".join(xs)
                    )
                    df_assign_view = df_assign[["truck", "clientes_str", "load", "cost"]]
                    df_assign_view.columns = ["Camión", "Clientes", "Carga (t)", "Costo (S/)"]

                    st.markdown("### Rutas seleccionadas por camión")
                    st.dataframe(df_assign_view, use_container_width=True)

                    # Gráfico de barras de costo por camión
                    st.markdown("### Costo por camión")
                    chart_data = df_assign_view[["Camión", "Costo (S/)"]].set_index("Camión")
                    st.bar_chart(chart_data)

                else:
                    st.warning("No se encontró ninguna asignación de rutas en la solución MIP.")
