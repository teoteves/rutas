# generar_rutas.py
import pandas as pd
import itertools
import math

def generar_rutas(dist_file, time_file,
                  addresses,
                  client_ids,
                  max_clientes_por_ruta=3,
                  t_max_min=180,
                  costo_por_km=10.0,
                  costo_por_min=5.0,
                  output_file="rutas_generadas.csv"):

    df_D = pd.read_csv(dist_file, index_col=0)
    df_T = pd.read_csv(time_file, index_col=0)

    D = df_D.to_numpy(dtype=float)
    T = df_T.to_numpy(dtype=float)
    n = len(addresses)

    routes = []
    route_id = 0

    clientes_indices = list(range(1, n))  # 1..n-1 (0 es planta)

    for k in range(1, max_clientes_por_ruta + 1):
        for seq in itertools.permutations(clientes_indices, k):
            nodos = [0] + list(seq)  # planta primero
            total_dist = 0.0
            total_time = 0.0
            factible = True
            for i, j in zip(nodos[:-1], nodos[1:]):
                d_ij = D[i, j]
                t_ij = T[i, j]
                if (not math.isfinite(d_ij)) or (not math.isfinite(t_ij)):
                    factible = False
                    break
                total_dist += d_ij
                total_time += t_ij
            if not factible:
                continue
            if (t_max_min is not None) and (total_time > t_max_min):
                continue

            costo_ruta = costo_por_km * total_dist + costo_por_min * total_time

            ruta_indices = " -> ".join(str(idx) for idx in nodos)
            ruta_texto = " -> ".join(addresses[idx] for idx in nodos)

            # IDs de clientes en la ruta (sin la planta)
            cliente_ids_seq = [client_ids[idx-1] for idx in seq]  # idx-1 porque clientes empiezan en 1
            clientes_str = ",".join(cliente_ids_seq)  # ej. "C1,C3"

            routes.append({
                "id_ruta": route_id,
                "n_nodos": len(nodos),
                "n_clientes": k,
                "indices_nodos": ruta_indices,
                "ruta_texto": ruta_texto,
                "clientes_ids": clientes_str,       # importante para el CG
                "distancia_km": total_dist,
                "tiempo_min": total_time,
                "costo_ruta": costo_ruta,
            })
            route_id += 1

    df_routes = pd.DataFrame(routes)
    df_routes.to_csv(output_file, index=False, encoding="utf-8-sig")
    return df_routes

if __name__ == "__main__":
    from config_real import PLANT, CLIENTS, MAX_CLIENTES_POR_RUTA, T_MAX_MIN, COSTO_POR_KM, COSTO_POR_MIN
    addresses = [PLANT["address"]] + [c["address"] for c in CLIENTS]
    client_ids = [c["id"] for c in CLIENTS]
    generar_rutas("distancias_km.csv", "tiempos_min.csv",
                  addresses, client_ids,
                  max_clientes_por_ruta=MAX_CLIENTES_POR_RUTA,
                  t_max_min=T_MAX_MIN,
                  costo_por_km=COSTO_POR_KM,
                  costo_por_min=COSTO_POR_MIN)
