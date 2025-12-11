# optimizer_service.py
from __future__ import annotations

from typing import Dict, List, Any

import pyomo.environ as pyo

from matriz_rutas import compute_matrix
from generar_rutas import generar_rutas
from rutas_cg import (
    get_solver,
    all_candidate_routes,
    build_rmp,
    pricing_dual_rc,
)


def build_data_from_inputs(
    clients: List[Dict[str, Any]],
    truck_types: Dict[str, Dict[str, Any]],
    rutas_file: str,
    demand_min: float = 0.0,
    demand_max: float = 1e9,
):
    """
    Construye I, J, F, Q, d, C_route a partir de:
      - clients: lista de dicts {id, name, address, demand_tons}
      - truck_types: dict {'T1': {...}, 'T2': {...}}
      - rutas_file: CSV generado por generar_rutas.py

    d[j]: demanda por cliente
    I:   camiones (tipo_k) por ID individual
    Q[i]: capacidad de camión i
    F[i]: costo fijo de camión i
    C_route[S]: mejor costo variable de una ruta que atiende exactamente S
    """
    import pandas as pd

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
            i_id = f"{tname}_{k + 1}"
            I.append(i_id)
            Q[i_id] = cap
            F[i_id] = fix

    # --- rutas generadas ---
    df_routes = pd.read_csv(rutas_file)
    C_route: Dict[frozenset, float] = {}

    for _, row in df_routes.iterrows():
        ids_str = row.get("clientes_ids", "")
        if not isinstance(ids_str, str) or not ids_str.strip():
            continue
        S = frozenset(ids_str.split(","))
        cost = float(row["costo_ruta"])
        if (S not in C_route) or (cost < C_route[S]):
            C_route[S] = cost

    if not C_route:
        raise ValueError(
            "No se generó ninguna ruta factible. "
            "Revisa el tiempo máximo, el máximo número de clientes por ruta "
            "o las matrices de distancias/tiempos."
        )

    return I, J, F, Q, d, C_route


def solve_routing(
    plant_address: str,
    clients: List[Dict[str, Any]],
    truck_types: Dict[str, Dict[str, Any]],
    route_params: Dict[str, Any],
    api_key: str,
) -> Dict[str, Any]:
    """
    Orquesta todo el flujo:

      1) Chequeo de capacidad total vs demanda total.
      2) Llamada a Google Routes (compute_matrix).
      3) Generación de rutas enumerativas (generar_rutas).
      4) Construcción de datos para el modelo (build_data_from_inputs).
      5) Generación de columnas + MIP final (rutas_cg).
      6) Devuelve un diccionario con el costo óptimo y el detalle de rutas.
    """
    if not api_key:
        raise ValueError("Falta la API key de Google Routes.")

    if not clients:
        raise ValueError("Debes ingresar al menos un cliente.")

    # --- chequeo rápido de capacidad total ---
    total_demand = sum(float(c["demand_tons"]) for c in clients)
    total_capacity = 0.0
    for info in truck_types.values():
        cap = float(info["capacity_tons"])
        cnt = int(info["count"])
        total_capacity += cap * cnt

    if total_capacity <= 0:
        raise ValueError("No hay camiones disponibles en la flota.")

    if total_capacity + 1e-6 < total_demand:
        raise ValueError(
            "No se puede satisfacer la **demanda de todos los clientes** con la flota actual.\n\n"
            f"- Demanda total: {total_demand:.1f} t\n"
            f"- Capacidad disponible: {total_capacity:.1f} t"
        )

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

    # 1) Matriz de distancias y tiempos
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
    I, J, F, Q, d, C_route = build_data_from_inputs(
        clients=clients,
        truck_types=truck_types,
        rutas_file=rutas_csv,
        demand_min=dmin,
        demand_max=dmax,
    )

    # 3.1) Verificar que cada cliente aparezca en al menos una ruta candidata
    Qref = max(Q.values()) if Q else 0.0
    ALL_R = all_candidate_routes(
        J, C_route, d, max_size=max_clients, Qref=Qref
    )

    uncovered = [k for k in J if not any(k in S for S in ALL_R)]
    if uncovered:
        raise ValueError(
            "Con la configuración actual NO existe ninguna ruta factible que visite:\n  - "
            + "\n  - ".join(uncovered)
            + "\n\nRevisa el tiempo máximo por ruta, el máximo de clientes por ruta o las matrices de distancias/tiempos."
        )

    # 4) Generación de columnas + MIP final
    solver = get_solver()
    TOL = 1e-9

    # 4.1) Columnas iniciales: rutas unitarias (cada camión a un solo cliente)
    current_columns = []
    for i in I:
        for j in J:
            S = frozenset([j])
            if S in C_route:
                current_columns.append((i, S))

    if not current_columns:
        raise ValueError(
            "No se encontraron rutas unitarias planta→cliente factibles. "
            "Revisa restricciones de tiempo y distancias."
        )

    it = 0
    while True:
        rmp = build_rmp(I, J, Q, F, d, C_route, current_columns, binary=False)
        solver.solve(rmp, tee=False)

        beta = {i: rmp.dual[rmp.slot[i]] for i in I}
        alpha = {i: rmp.dual[rmp.cap[i]] for i in I}
        pi = {k: rmp.dual[rmp.cover[k]] for k in J}

        curr_set = set(current_columns)
        rc_list = pricing_dual_rc(
            I, ALL_R, F, d, C_route, beta, alpha, pi, curr_set
        )

        nneg = sum(1 for rc, _ in rc_list if rc < -TOL)
        if nneg == 0:
            # Óptimo LP alcanzado (no hay columnas con rc<0)
            break

        for rc, (i, S_tuple) in rc_list:
            if rc < -TOL:
                current_columns.append((i, frozenset(S_tuple)))

        it += 1
        if it > 50:  # tope de seguridad
            break

    # 5) MIP final (x binaria)
    mip = build_rmp(I, J, Q, F, d, C_route, current_columns, binary=True)
    solver.solve(mip, tee=False)

    assignments = []
    client_name = {c["id"]: c.get("name", c["id"]) for c in clients}

    for c_idx in mip.COL:
        if pyo.value(mip.x[c_idx]) > 0.5:
            i = mip.col_i[c_idx]
            S_tuple = tuple(sorted(mip.col_S[c_idx]))
            load = float(mip.col_load[c_idx])
            total_cost = float(mip.col_cost[c_idx])
            S_fset = frozenset(S_tuple)
            var_cost = float(C_route[S_fset])
            fix_cost = float(F[i])

            assignments.append(
                {
                    "camion": i,
                    "tipo_camion": i.split("_")[0],  # T1, T2, ...
                    "clientes_ids": ",".join(S_tuple),
                    "clientes_nombres": ", ".join(
                        client_name[k] for k in S_tuple
                    ),
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
        "total_demand": total_demand,
        "total_capacity": total_capacity,
    }

