# optimizer_service.py

from typing import List, Dict, Any
import pandas as pd
import pyomo.environ as pyo

from matriz_rutas import compute_matrix
from generar_rutas import generar_rutas
from rutas_cg import (
    build_rmp, all_candidate_routes, pricing_dual_rc, get_solver
)


def solve_routing(
    plant_address: str,
    clients: List[Dict[str, Any]],
    truck_types: Dict[str, Dict[str, float]],
    route_params: Dict[str, Any],
    api_key: str,
) -> Dict[str, Any]:
    """
    Ejecuta TODO el pipeline:
      - Matriz de distancias/tiempos (Google Routes API)
      - Generación de rutas (planta -> clientes)
      - Construcción de I,J,F,Q,d,C_route
      - Column Generation + MIP final

    Entradas:
      plant_address: dirección de la planta.
      clients: lista de dicts con:
          { "id": "C1", "name": "...", "address": "...", "demand_tons": 15.0 }
      truck_types: ej:
          {
            "T30": {"capacity_tons": 30.0, "fixed_cost": 800.0, "count": 5},
            "T10": {"capacity_tons": 10.0, "fixed_cost": 300.0, "count": 3},
          }
      route_params:
          {
            "max_clientes_por_ruta": 3,
            "t_max_min": 180,
            "costo_por_km": 10.0,
            "costo_por_min": 5.0,
          }
      api_key: API key de Google

    Devuelve:
      dict con resultado de la relajación LP y del MIP final.
    """

    # 1) Direcciones y IDs de clientes
    addresses = [plant_address] + [c["address"] for c in clients]
    client_ids = [c["id"] for c in clients]

    # 2) Google Routes API -> matrices
    df_D, df_T = compute_matrix(addresses, api_key)
    # compute_matrix normalmente también guarda distancias_km.csv y tiempos_min.csv
    # y generamos rutas usando esos CSV para mantener compatibilidad.

    # 3) Generar rutas físicas (planta -> secuencia de clientes)
    df_routes = generar_rutas(
        dist_file="distancias_km.csv",
        time_file="tiempos_min.csv",
        addresses=addresses,
        client_ids=client_ids,
        max_clientes_por_ruta=route_params["max_clientes_por_ruta"],
        t_max_min=route_params["t_max_min"],
        costo_por_km=route_params["costo_por_km"],
        costo_por_min=route_params["costo_por_min"],
        output_file="rutas_generadas.csv",
    )

    # 4) Construir I,J,F,Q,d,C_route en memoria
    I, J, F, Q, d, C_route = build_data_in_memory(clients, truck_types, df_routes)

    # 5) Column Generation
    result_lp, current_columns = solve_with_column_generation(
        I, J, F, Q, d, C_route,
        max_clientes_por_ruta=route_params["max_clientes_por_ruta"]
    )

    # 6) MIP final
    mip_solution = solve_mip_final(I, J, F, Q, d, C_route, current_columns)

    return {
        "lp_relaxation": result_lp,
        "mip_solution": mip_solution,
    }


def build_data_in_memory(
    clients: List[Dict[str, Any]],
    truck_types: Dict[str, Dict[str, float]],
    df_routes: pd.DataFrame,
):
    """Versión en memoria de build_data_from_real."""
    # ---- clientes y demandas
    J = [c["id"] for c in clients]
    d = {c["id"]: float(c["demand_tons"]) for c in clients}

    # ---- flota de camiones
    I = []
    Q = {}
    F = {}
    for tname, info in truck_types.items():
        cap = float(info["capacity_tons"])
        fix = float(info["fixed_cost"])
        count = int(info["count"])
        for k in range(count):
            i_id = f"{tname}_{k+1}"
            I.append(i_id)
            Q[i_id] = cap
            F[i_id] = fix

    # ---- construir C_route[S] desde df_routes
    C_route: Dict[frozenset, float] = {}
    for _, row in df_routes.iterrows():
        clientes_str = str(row["clientes_ids"]).strip()
        if not clientes_str:
            continue
        S_ids = tuple(sorted(clientes_str.split(",")))
        S = frozenset(S_ids)
        costo_var = float(row["costo_ruta"])

        total_load = sum(d[j] for j in S_ids)
        max_cap = max(Q.values())
        if total_load > max_cap:
            continue

        if S not in C_route or costo_var < C_route[S]:
            C_route[S] = costo_var

    if not C_route:
        raise RuntimeError("No se generó ninguna ruta factible en memoria.")

    return I, J, F, Q, d, C_route


def solve_with_column_generation(
    I, J, F, Q, d, C_route, max_clientes_por_ruta: int
):
    """Lógica de generación de columnas (LP)."""
    Qref = max(Q.values())
    ALL_R = all_candidate_routes(J, C_route, d,
                                 max_size=max_clientes_por_ruta,
                                 Qref=Qref)

    # columnas iniciales: rutas unitarias
    current_columns = []
    for i in I:
        for j in J:
            S = frozenset([j])
            if S in C_route:
                current_columns.append((i, S))

    solver = get_solver()
    it = 0
    TOL = 1e-9

    while True:
        rmp = build_rmp(I, J, Q, F, d, C_route, current_columns, binary=False)
        solver.solve(rmp, tee=False)

        beta  = {i: rmp.dual[rmp.slot[i]]  for i in I}
        alpha = {i: rmp.dual[rmp.cap[i]]   for i in I}
        pi    = {k: rmp.dual[rmp.cover[k]] for k in J}
        zLP   = pyo.value(rmp.OBJ)

        curr_set = set(current_columns)
        rc_list = pricing_dual_rc(I, ALL_R, F, d, C_route,
                                  betas=beta, alphas=alpha, pis=pi,
                                  current_columns_set=curr_set)

        added = False
        for rc, (i, S_tuple) in rc_list:
            if rc < -TOL:
                current_columns.append((i, frozenset(S_tuple)))
                added = True
        if not added:
            break
        it += 1

    return {"zLP": zLP, "iterations": it}, current_columns


def solve_mip_final(I, J, F, Q, d, C_route, current_columns):
    """Resuelve el MIP binario final y devuelve asignación como lista de dicts."""
    solver = get_solver()
    mip = build_rmp(I, J, Q, F, d, C_route, current_columns, binary=True)
    solver.solve(mip, tee=False)

    z_star = pyo.value(mip.OBJ)
    solution_rows = []
    for c in mip.COL:
        if pyo.value(mip.x[c]) > 0.5:
            i = mip.col_i[c]
            S = tuple(sorted(mip.col_S[c]))
            carga = mip.col_load[c]
            costo = mip.col_cost[c]
            solution_rows.append({
                "truck": i,
                "clients": list(S),
                "load": carga,
                "cost": costo,
            })

    return {
        "z_star": z_star,
        "assignments": solution_rows,
    }
