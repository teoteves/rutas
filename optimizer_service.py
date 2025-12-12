from __future__ import annotations

from typing import Dict, List, Any, Iterable, Tuple
import math

import pandas as pd
import pyomo.environ as pyo

from matriz_rutas import compute_matrix
from generar_rutas import generar_rutas
from rutas_cg import (
    get_solver,
    all_candidate_routes,
    build_rmp,
    pricing_dual_rc,
)


# ----------------------------------------------------
# Auxiliar: recomendación mínima de camiones por tipo
# ----------------------------------------------------
def recomendar_camiones_minimos(
    demanda_total: float,
    capacidades: Dict[str, float],
) -> Dict[str, int]:
    """
    Dado:
      - demanda_total D
      - capacidades: dict {tipo: Q_t} con Q_t > 0

    Devuelve un dict {tipo: n_t} con el número mínimo de camiones de cada tipo
    para que sum_t Q_t * n_t >= D, minimizando sum_t n_t.

    Implementado para 1 o 2 tipos (caso actual).
    """
    tipos_positivos = [(t, float(Q)) for t, Q in capacidades.items() if Q > 0]

    if not tipos_positivos or demanda_total <= 0:
        return {t: 0 for t in capacidades.keys()}

    # Caso: un solo tipo
    if len(tipos_positivos) == 1:
        t, Q = tipos_positivos[0]
        n = math.ceil(demanda_total / Q)
        return {t: n}

    # Caso: dos tipos
    (t1, Q1), (t2, Q2) = tipos_positivos[:2]
    if Q2 > Q1:
        (t1, Q1), (t2, Q2) = (t2, Q2), (t1, Q1)

    mejor_n1 = mejor_n2 = None
    mejor_total = None
    max_n1 = math.ceil(demanda_total / Q1)

    for n1 in range(max_n1 + 1):
        restante = demanda_total - n1 * Q1
        if restante <= 0:
            n2 = 0
        else:
            n2 = math.ceil(restante / Q2)
        total_camiones = n1 + n2
        if total_camiones == 0 and demanda_total > 0:
            continue
        if (mejor_total is None) or (total_camiones < mejor_total):
            mejor_total = total_camiones
            mejor_n1 = n1
            mejor_n2 = n2

    return {t1: mejor_n1, t2: mejor_n2}


# ----------------------------------------------------
# Construcción de datos desde inputs de la app
# ----------------------------------------------------
def build_data_from_inputs(
    clients: List[Dict[str, Any]],
    truck_types: Dict[str, Dict[str, Any]],
    rutas_file: str,
    demand_min: float = 0.0,
    demand_max: float = 1e9,
):
    """
    Construye I, J, F, Q, d, C_route, route_text_map a partir de:
      - clients: lista de dicts {id, name, address, demand_tons}
      - truck_types: dict {'T1': {...}, 'T2': {...}}
      - rutas_file: CSV generado por generar_rutas.py
    """
    # --- clientes y demandas ---
    J: List[str] = [c["id"] for c in clients]
    d: Dict[str, float] = {}
    for c in clients:
        cid = c["id"]
        dj = float(c["demand_tons"])
        if dj < demand_min or dj > demand_max:
            raise ValueError(
                f"Demanda del cliente {cid} = {dj} fuera de rango "
                f"[{demand_min}, {demand_max}]"
            )
        d[cid] = dj

    # --- flota de camiones ---
    I: List[str] = []
    Q: Dict[str, float] = {}
    F: Dict[str, float] = {}
    for tname, info in truck_types.items():
        cap = float(info["capacity_tons"])
        fix = float(info["fixed_cost"])
        count = int(info["count"])
        for k in range(count):
            i_id = f"{tname}_{k+1}"
            I.append(i_id)
            Q[i_id] = cap
            F[i_id] = fix

    df_routes = pd.read_csv(rutas_file)

    C_route: Dict[frozenset, float] = {}
    route_text_map: Dict[frozenset, str] = {}

    for _, row in df_routes.iterrows():
        clientes_str = str(row["clientes_ids"]).strip()
        if not clientes_str:
            continue
        # Orden de visita de los clientes en la ruta física
        S_ids_order = clientes_str.split(",")
        # Conjunto lógico (ignora el orden)
        S_ids_sorted = tuple(sorted(S_ids_order))
        S = frozenset(S_ids_sorted)
        costo_var = float(row["costo_ruta"])

        total_load = sum(d[j] for j in S_ids_sorted)
        max_cap = max(Q.values())
        if total_load > max_cap:
            continue

        if S not in C_route or costo_var < C_route[S]:
            C_route[S] = costo_var
            # Texto de ruta en IDs (en la app mostramos siempre "Planta" al inicio)
            ruta_ids = " -> ".join(S_ids_order)
            route_text_map[S] = f"Planta -> {ruta_ids}"

    if not C_route:
        raise ValueError(
            "No se generó ninguna ruta factible. "
            "Revisa el tiempo máximo por ruta, el máximo de clientes por ruta "
            "o las matrices de distancias/tiempos."
        )

    return I, J, F, Q, d, C_route, route_text_map


# ----------------------------------------------------
# Flujo principal para la app
# ----------------------------------------------------
def solve_routing(
    plant_address: str,
    clients: List[Dict[str, Any]],
    truck_types: Dict[str, Dict[str, Any]],
    route_params: Dict[str, Any],
    api_key: str,
) -> Dict[str, Any]:
    """
    Orquesta todo el flujo para la app Streamlit.
    """
    if not api_key:
        raise ValueError("Falta la API key de Google Routes.")

    if not clients:
        raise ValueError("Debes ingresar al menos un cliente.")

    # --- chequeo rápido de capacidad total ---
    demanda_total = sum(float(c["demand_tons"]) for c in clients)

    capacidades = {t: float(info["capacity_tons"]) for t, info in truck_types.items()}
    conteo_actual = {t: int(info["count"]) for t, info in truck_types.items()}
    capacidad_total = sum(
        capacidades[t] * conteo_actual[t] for t in capacidades.keys()
    )

    if capacidad_total <= 0:
        raise ValueError("No hay camiones disponibles en la flota.")

    if capacidad_total + 1e-6 < demanda_total:
        recomendados = recomendar_camiones_minimos(demanda_total, capacidades)
        msg_lines = [
            "No se puede satisfacer la **demanda de todos los clientes** con la flota actual.",
            "",
            f"- Demanda total: {demanda_total:.1f} t",
            f"- Capacidad disponible actual: {capacidad_total:.1f} t",
            "",
            "Recomendación mínima de camiones (solo por capacidad, ignorando límites actuales):",
        ]
        for t, n in recomendados.items():
            msg_lines.append(f"  · Tipo {t}: {n} camión(es) de {capacidades[t]:.1f} t")

        msg_lines.append("")
        msg_lines.append("Flota actual declarada:")
        for t, c in conteo_actual.items():
            msg_lines.append(f"  · Tipo {t}: {c} camión(es)")
        raise ValueError("\n".join(msg_lines))

    # --- parámetros de rutas ---
    max_clients = int(route_params.get("max_clientes_por_ruta", 3))
    t_max = float(route_params.get("t_max_min", 180.0))
    costo_km = float(route_params.get("costo_por_km", 10.0))
    costo_min = float(route_params.get("costo_por_min", 5.0))
    dmin = float(route_params.get("demand_min", 0.0))
    dmax = float(route_params.get("demand_max", 1e9))

    # --- direcciones para matriz de distancias/tiempos ---
    addresses = [plant_address] + [c["address"] for c in clients]
    client_ids = [c["id"] for c in clients]

    dist_csv = "distancias_km.csv"
    time_csv = "tiempos_min.csv"
    rutas_csv = "rutas_generadas.csv"

    # 1) Matriz de distancias y tiempos (escribe CSVs)
    compute_matrix(addresses, api_key, output_dist=dist_csv, output_time=time_csv)

    # 2) Enumeración de rutas factibles
    generar_rutas(
        dist_csv,
        time_csv,
        addresses,
        client_ids,
        max_clientes_por_ruta=max_clients,
        t_max_min=t_max,
        costo_por_km=costo_km,
        costo_por_min=costo_min,
        output_file=rutas_csv,
    )

    # 3) Datos para el modelo
    I, J, F, Q, d, C_route, route_text_map = build_data_from_inputs(
        clients=clients,
        truck_types=truck_types,
        rutas_file=rutas_csv,
        demand_min=dmin,
        demand_max=dmax,
    )

    # 3.1) Verificar que cada cliente aparezca en alguna ruta candidata
    Qref = max(Q.values()) if Q else 0.0
    ALL_R = all_candidate_routes(J, C_route, d, max_size=max_clients, Qref=Qref)

    uncovered = [k for k in J if not any(k in S for S in ALL_R)]
    if uncovered:
        raise ValueError(
            "Con la configuración actual NO existe ninguna ruta factible que visite:\n  - "
            + "\n  - ".join(uncovered)
            + "\n\nRevisa el tiempo máximo por ruta, el máximo de clientes por ruta "
            "o las matrices de distancias/tiempos."
        )

    # 4) Generación de columnas (LP)
    solver = get_solver()
    TOL = 1e-9

    current_columns: List[Tuple[str, frozenset]] = []
    for i in I:
        for j in J:
            S = frozenset([j])
            if S in C_route:
                current_columns.append((i, S))

    if not current_columns:
        raise ValueError(
            "No se encontraron rutas unitarias planta→cliente factibles. "
            "Revisa las restricciones de rutas."
        )

    it = 0
    while True:
        rmp = build_rmp(I, J, Q, F, d, C_route, current_columns, route_text_map=route_text_map, binary=False)
        solver.solve(rmp, tee=False)

        beta = {i: rmp.dual[rmp.slot[i]] for i in I}
        alpha = {i: rmp.dual[rmp.cap[i]] for i in I}
        pi_dual = {k: rmp.dual[rmp.cover[k]] for k in J}

        curr_set = set(current_columns)
        rc_list = pricing_dual_rc(I, ALL_R, F, d, C_route, beta, alpha, pi_dual, curr_set)

        nneg = sum(1 for rc, _ in rc_list if rc < -TOL)
        if nneg == 0:
            break

        for rc, (i_col, S_tuple) in rc_list:
            if rc < -TOL:
                current_columns.append((i_col, frozenset(S_tuple)))

        it += 1
        if it > 50:  # por seguridad
            break

    # 5) MIP final (binario)
    mip = build_rmp(I, J, Q, F, d, C_route, current_columns, route_text_map=route_text_map, binary=True)
    solver.solve(mip, tee=False)

    assignments = []
    client_name = {c["id"]: c.get("name", c["id"]) for c in clients}

    for c_idx in mip.COL:
        if pyo.value(mip.x[c_idx]) > 0.5:
            i_col = mip.col_i[c_idx]
            S_tuple = tuple(sorted(mip.col_S[c_idx]))
            load = float(mip.col_load[c_idx])
            total_cost = float(mip.col_cost[c_idx])
            S_fset = frozenset(S_tuple)
            var_cost = float(C_route[S_fset])
            # Fijo según camión
            tipo_camion = i_col.split("_")[0]
            fix_cost = float(F[i_col])

            ruta_ids_txt = "Planta -> " + " -> ".join(S_tuple)
            ruta_txt = ruta_ids_txt
            if route_text_map and S_fset in route_text_map:
                ruta_txt = route_text_map[S_fset]

            assignments.append(
                {
                    "camion": i_col,
                    "tipo_camion": tipo_camion,
                    "ruta_ids": ruta_txt,
                    "clientes_ids": ",".join(S_tuple),
                    "clientes_nombres": ", ".join(client_name[k] for k in S_tuple),
                    "carga_t": load,
                    "costo_fijo_S/": fix_cost,
                    "costo_variable_S/": var_cost,
                    "costo_total_S/": total_cost,
                }
            )

    z_opt = float(pyo.value(mip.OBJ))

    return {
        "objective": z_opt,
        "assignments": assignments,
        "total_demand": demanda_total,
        "total_capacity": capacidad_total,
    }
