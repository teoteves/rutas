from itertools import combinations
from typing import Dict, Iterable, Tuple, List, Set
import pyomo.environ as pyo

# ============================================================
#  Helpers de impresión (sólo para que el output sea legible)
# ============================================================

def fmt_float(x, nd: int = 3) -> str:
    """Formatea un número flotante con nd decimales."""
    return f"{float(x):.{nd}f}"


def fmt_dict(
    d: Dict[str, float],
    nd: int = 3,
    keys_order: Iterable[str] = None,
    title: str | None = None,
    indent: str = "  ",
) -> str:
    """Devuelve un string bonito de un diccionario numérico."""
    if keys_order is None:
        keys_order = list(d.keys())
    lines: List[str] = []
    if title:
        lines.append(f"{indent}{title}:")
    for k in keys_order:
        v = d[k]
        lines.append(f"{indent}  {k:>4} : {fmt_float(v, nd)}")
    return "\n".join(lines)


def print_header(iter_num: int, z_lp: float) -> None:
    line = "=" * 66
    print("\n" + line)
    print(f" Iteración CG: {iter_num:<3d}    |    RMP (LP)  z = {fmt_float(z_lp, 3)}")
    print(line)


def print_duals(beta: Dict[str, float],
                alpha: Dict[str, float],
                pi: Dict[str, float]) -> None:
    """Imprime los multiplicadores duales de las restricciones."""
    print(fmt_dict(beta, nd=3, title="β (slot por camión)"))
    print(fmt_dict(alpha, nd=3, title="α (capacidad por camión)"))
    print(fmt_dict(pi, nd=3, title="π (cobertura por cliente)"))


def print_rc_candidates(
    rc_items: Iterable[Tuple[float, Tuple[str, Tuple[str, ...]]]],
    F: Dict[str, float],
    C_route: Dict[frozenset, float],
    d: Dict[str, float],
    tol: float = -1e-9,
) -> int:
    """
    Imprime las mejores columnas candidatas según el costo reducido.
    rc_items: lista de (rc, (i, S_tuple_sorted)).
    Devuelve cuántas tienen rc < tol.
    """
    neg: List[Tuple[float, Tuple[str, Tuple[str, ...]]]] = [
        (rc, info) for rc, info in rc_items if rc < tol
    ]
    if not neg:
        print("  No hay columnas con costo reducido negativo.")
        return 0

    print("\n  Columnas candidatas con rc < 0 (las primeras son más prometedoras):")
    print("    rc       camión   ruta      carga   costo(Fi+Cj)")
    print("   " + "-" * 46)

    for rc, (i, S_tuple) in neg[:10]:  # sólo mostramos las 10 mejores
        S = frozenset(S_tuple)
        carga = sum(d[k] for k in S)
        costo = F[i] + C_route[S]
        ruta_str = "{" + ",".join(S_tuple) + "}"
        print(
            f"   {fmt_float(rc,3):>7}  {i:<6}  {ruta_str:<10} "
            f"{fmt_float(carga,1):>6}  {fmt_float(costo,1):>8}"
        )
    if len(neg) > 10:
        print(f"   ... y {len(neg)-10} columnas negativas adicionales.")
    return len(neg)


def print_solution_integer(mip_model: pyo.ConcreteModel) -> None:
    """Imprime la solución entera final del MIP."""
    line = "=" * 66
    print("\n" + line)
    print(" SOLUCIÓN ENTERA FINAL")
    print(line)
    print(f"  z* = {fmt_float(pyo.value(mip_model.OBJ), 3)}")
    print(f"  {'Camión':<8} {'Ruta':<18} {'Carga':>7} {'Costo':>10}")
    print(f"  {'-'*8} {'-'*18} {'-'*7} {'-'*10}")
    for c in mip_model.COL:
        if pyo.value(mip_model.x[c]) > 0.5:
            i = mip_model.col_i[c]
            S = tuple(sorted(mip_model.col_S[c]))
            carga = mip_model.col_load[c]
            costo = mip_model.col_cost[c]
            ruta = "{" + ",".join(S) + "}"
            print(
                f"  {i:<8} {ruta:<18} {fmt_float(carga,1):>7} "
                f"{fmt_float(costo,1):>10}"
            )


# =========================
# Elección de solver
# =========================

def get_solver():
    """
    Intenta usar, en este orden:
      - HiGHS (appsi_highs)
      - GLPK
      - CBC

    Requisitos:
      pip install highspy          # para appsi_highs
      conda install -c conda-forge glpk
      conda install -c conda-forge coincbc
    """
    for name in ["appsi_highs", "glpk", "cbc"]:
        try:
            s = pyo.SolverFactory(name)
            if s is not None and s.available(False):
                print(f"[INFO] Usando solver: {name}")
                return s
        except Exception:
            pass
    raise RuntimeError(
        "No se encontró solver disponible. Instala uno de:\n"
        "  pip install highspy   (y usa appsi_highs)\n"
        "  conda install -c conda-forge glpk   (y usa glpk)\n"
        "  conda install -c conda-forge coincbc (y usa cbc)"
    )



# ============================================================
# Datos desde caso real (rutas_generadas.csv + config_real.py)
# ============================================================

def build_data_from_real(
    config_module: str = "config_real",
    rutas_file: str = "rutas_generadas.csv",
):
    """
    Lee:
      - parámetros de PLANT, CLIENTS, TRUCK_TYPES desde config_real.py
      - rutas generadas desde rutas_generadas.csv

    y construye:
      I: conjunto de camiones (por ID individual)
      J: conjunto de clientes (ID's tipo 'C1', 'C2', ...)
      F[i]: costo fijo de camión i
      Q[i]: capacidad de camión i (toneladas)
      d[j]: demanda del cliente j (toneladas)
      C_route[S]: costo variable de la mejor ruta que atiende exactamente S
                  (S es un frozenset de IDs de clientes)
    """
    import pandas as pd
    cfg = __import__(config_module)

    PLANT = cfg.PLANT          # no se usa directamente aquí, pero queda documentado
    CLIENTS = cfg.CLIENTS
    DEM_MIN = cfg.DEMAND_MIN
    DEM_MAX = cfg.DEMAND_MAX
    TRUCK_TYPES = cfg.TRUCK_TYPES

    # ---- clientes y demandas
    J: List[str] = [c["id"] for c in CLIENTS]
    d: Dict[str, float] = {}
    for c in CLIENTS:
        cid = c["id"]
        dj = float(c["demand_tons"])
        if dj < DEM_MIN or dj > DEM_MAX:
            raise ValueError(
                f"Demanda de {cid} = {dj} fuera de rango "
                f"[{DEM_MIN}, {DEM_MAX}]"
            )
        d[cid] = dj

    # ---- flota de camiones
    I: List[str] = []
    Q: Dict[str, float] = {}
    F: Dict[str, float] = {}
    for tname, info in TRUCK_TYPES.items():
        cap = float(info["capacity_tons"])
        fix = float(info["fixed_cost"])
        count = int(info["count"])
        for k in range(count):
            i_id = f"{tname}_{k+1}"
            I.append(i_id)
            Q[i_id] = cap
            F[i_id] = fix

    # ---- rutas generadas (cada fila: planta -> clientes_ids en cierto orden)
    df_routes = pd.read_csv(rutas_file)

    # Construimos C_route[S] como: para cada conjunto S de clientes,
    # tomamos el *mínimo* costo variable entre todas las rutas
    # que cubren exactamente S (independiente del orden).
    C_route: Dict[frozenset, float] = {}

    for _, row in df_routes.iterrows():
        clientes_str = str(row["clientes_ids"]).strip()
        if not clientes_str:
            continue
        S_ids = tuple(sorted(clientes_str.split(",")))
        S = frozenset(S_ids)
        costo_var = float(row["costo_ruta"])

        # Chequeo rápido de capacidad con la capacidad MÁXIMA de la flota.
        total_load = sum(d[j] for j in S_ids)
        max_cap = max(Q.values())
        if total_load > max_cap:
            # ni siquiera el camión más grande podría llevarlos juntos
            continue

        if S not in C_route or costo_var < C_route[S]:
            C_route[S] = costo_var

    if not C_route:
        raise RuntimeError(
            "No se generó ninguna ruta factible (revisa demandas, "
            "capacidades y rutas_generadas.csv)."
        )

    return I, J, F, Q, d, C_route


# ============================================================
# Generación del catálogo de rutas por subconjunto de clientes
# ============================================================

def feasible_route(S: frozenset, d: Dict[str, float], Qref: float) -> bool:
    """
    Devuelve True si la demanda total de S cumple sum_{k in S} d_k <= Qref.

    Qref NO es la capacidad de cada camión Q_i del modelo matemático.
    Es sólo un filtro previo para construir el catálogo ALL_R.
    """
    return sum(d[k] for k in S) <= Qref


def all_candidate_routes(
    J: Iterable[str],
    C_route: Dict[frozenset, float],
    d: Dict[str, float],
    max_size: int = 3,
    Qref: float = 1e9,
) -> List[frozenset]:
    """
    Construye el conjunto ALL_R de subconjuntos S de clientes que:

      - Tienen tamaño 1 <= |S| <= max_size.
      - Existen en C_route (es decir, conocemos C_S).
      - Cumplen factibilidad básica de capacidad: sum d_k <= Qref.

    ALL_R se usa luego en el pricing para probar columnas (i,S).
    """
    R: List[frozenset] = []
    J_list = list(J)

    for r_size in range(1, max_size + 1):
        for S_tuple in combinations(J_list, r_size):
            S = frozenset(S_tuple)
            if S in C_route and feasible_route(S, d, Qref):
                R.append(S)
    return R


# ============================================================
# Modelo maestro (RMP/MIP) en espacio de columnas
# ============================================================

def column_cost(F: Dict[str, float],
                C_route: Dict[frozenset, float],
                i: str,
                S: frozenset) -> float:
    """Costo total de una columna (i,S): F_i + C_S."""
    return F[i] + C_route[S]


def column_load(d: Dict[str, float], S: frozenset) -> float:
    """Carga total de la ruta S: sum_{k in S} d_k."""
    return sum(d[k] for k in S)


def covers_k(S: frozenset, k: str) -> int:
    """1 si el cliente k está contenido en S, 0 en caso contrario."""
    return 1 if k in S else 0


def build_rmp(
    I: Iterable[str],
    J: Iterable[str],
    Q: Dict[str, float],
    F: Dict[str, float],
    d: Dict[str, float],
    C_route: Dict[frozenset, float],
    current_columns: List[Tuple[str, frozenset]],
    route_text_map=None,
    binary: bool = False,
) -> pyo.ConcreteModel:
    """
    Construye el modelo maestro (RMP/MIP) sobre las columnas actuales.

    Notación del modelo matemático:

      - Índice i  -> camión
      - Índice j  -> ruta (subconjunto S de clientes)
      - Variable x[c]  -> x_{ij} (para la columna c = (i,S))
      - F[i]      -> F_i (costo fijo de camión i)
      - C_route[S]-> C_j (costo variable de la ruta S)
      - Q[i]      -> Q_i (capacidad del camión i)
      - column_load(d,S) -> D_j (consumo de capacidad de la ruta S)
      - col_cover[c,k]   -> δ_{k∈S}  (si la ruta S cubre al cliente k)

    Restricciones implementadas:

      1)  sum_j x_{ij} <= 1                (a lo sumo una ruta por camión)
      2)  sum_j D_j x_{ij} <= Q_i          (capacidad del camión i)
      3)  sum_i sum_j δ_{k∈S} x_{ij} = 1   (cada cliente exactamente una vez)

    Si binary=False construye el RMP continuo (LP).
    Si binary=True construye el MIP final (x ∈ {0,1}).
    """
    m = pyo.ConcreteModel()

    # Conjuntos
    I_list = list(I)
    J_list = list(J)
    m.I = pyo.Set(initialize=I_list)      # camiones
    m.K = pyo.Set(initialize=J_list)      # clientes
    m.COL = pyo.Set(initialize=range(len(current_columns)))  # columnas activas

    # Metadatos por columna
    m.col_i = {c: current_columns[c][0] for c in m.COL}   # camión de la columna c
    m.col_S = {c: current_columns[c][1] for c in m.COL}   # conjunto de clientes S

    m.col_cost = {
        c: column_cost(F, C_route, m.col_i[c], m.col_S[c]) for c in m.COL
    }
    m.col_load = {c: column_load(d, m.col_S[c]) for c in m.COL}
    m.col_cover = {
        (c, k): covers_k(m.col_S[c], k) for c in m.COL for k in m.K
    }

    # Variables x_c
    m.x = pyo.Var(
        m.COL,
        domain=pyo.Binary if binary else pyo.NonNegativeReals,
    )

    # Objetivo: min sum_c (F_i + C_S) x_c
    def obj_rule(mm: pyo.ConcreteModel) -> float:
        return sum(mm.col_cost[c] * mm.x[c] for c in mm.COL)

    m.OBJ = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # Restricción 1: a lo sumo una ruta por camión i
    def slot_rule(mm: pyo.ConcreteModel, i: str):
        return sum(mm.x[c] for c in mm.COL if mm.col_i[c] == i) <= 1.0

    m.slot = pyo.Constraint(m.I, rule=slot_rule)

    # Restricción 2: capacidad de cada camión i
    def cap_rule(mm: pyo.ConcreteModel, i: str):
        return sum(
            mm.col_load[c] * mm.x[c]
            for c in mm.COL
            if mm.col_i[c] == i
        ) <= Q[i]

    m.cap = pyo.Constraint(m.I, rule=cap_rule)

    # Restricción 3: cada cliente k debe ser cubierto exactamente una vez
    def cover_rule(mm: pyo.ConcreteModel, k: str):
        return sum(
            mm.col_cover[c, k] * mm.x[c] for c in mm.COL
        ) == 1.0

    m.cover = pyo.Constraint(m.K, rule=cover_rule)

    # Suffix para leer duales cuando trabajamos en LP
    if not binary:
        m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    return m


# ============================================================
# Pricing (cálculo de costos reducidos)
# ============================================================

def pricing_dual_rc(
    I: Iterable[str],
    ALL_R: Iterable[frozenset],
    F: Dict[str, float],
    d: Dict[str, float],
    C_route: Dict[frozenset, float],
    betas: Dict[str, float],
    alphas: Dict[str, float],
    pis: Dict[str, float],
    current_columns_set: Set[Tuple[str, frozenset]],
) -> List[Tuple[float, Tuple[str, Tuple[str, ...]]]]:
    """
    Calcula el costo reducido para cada columna candidata (i,S).

    Duales:
      - β_i: restricción de "slot" (a lo sumo una ruta por camión i).
      - α_i: restricción de capacidad del camión i.
      - π_k: restricción de cobertura del cliente k.

    Para cada camión i y cada conjunto S en ALL_R tal que (i,S) no esté ya
    en current_columns_set, se calcula:

        rc(i,S) = (F_i + C_S)
                  - β_i
                  - α_i * D_S
                  - sum_{k in S} π_k,

    donde D_S = sum_{k in S} d_k.

    Devuelve una lista ordenada por rc ascendente:
        [(rc, (i, S_ordenado)), ...]
    """
    out: List[Tuple[float, Tuple[str, Tuple[str, ...]]]] = []

    for i in I:
        for S in ALL_R:
            if (i, S) in current_columns_set:
                continue
            carga_S = column_load(d, S)
            rc = (F[i] + C_route[S]) - betas[i] - alphas[i] * carga_S - sum(
                pis[k] for k in S
            )
            out.append((rc, (i, tuple(sorted(S)))))

    out.sort(key=lambda z: z[0])
    return out

