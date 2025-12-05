# matriz_rutas.py
import requests
import pandas as pd
import math
import json

def compute_matrix(addresses, api_key, output_dist="distancias_km.csv", output_time="tiempos_min.csv"):
    n = len(addresses)

    origins = [{"waypoint": {"address": a}} for a in addresses]
    destinations = [{"waypoint": {"address": a}} for a in addresses]

    body = {
        "origins": origins,
        "destinations": destinations,
        "travelMode": "DRIVE",
        "routingPreference": "TRAFFIC_AWARE",
        # opcional: "departureTime": "2025-12-04T18:00:00Z",
    }

    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "originIndex,destinationIndex,status,condition,distanceMeters,duration",
    }

    r = requests.post("https://routes.googleapis.com/distanceMatrix/v2:computeRouteMatrix",
                      headers=headers, json=body)
    r.raise_for_status()

    text = r.text
    if text.startswith(")]}'"):
        text = text[4:]
    elements = json.loads(text)

    D = [[math.inf]*n for _ in range(n)]
    T = [[math.inf]*n for _ in range(n)]

    def parse_duration_seconds(s):
        if isinstance(s, str) and s.endswith("s"):
            return float(s[:-1])
        return float(s)

    for elem in elements:
        oi = elem["originIndex"]
        di = elem["destinationIndex"]
        if oi == di:
            continue
        status = elem.get("status", {})
        condition = elem.get("condition", "")
        if status.get("code") not in (None, 0) or condition != "ROUTE_EXISTS":
            continue
        dist_m = elem["distanceMeters"]
        dur_s = parse_duration_seconds(elem["duration"])
        D[oi][di] = dist_m / 1000.0
        T[oi][di] = dur_s / 60.0

    df_D = pd.DataFrame(D, index=addresses, columns=addresses)
    df_T = pd.DataFrame(T, index=addresses, columns=addresses)

    df_D.to_csv(output_dist, encoding="utf-8-sig")
    df_T.to_csv(output_time, encoding="utf-8-sig")

    return df_D, df_T

if __name__ == "__main__":
    # ejemplo rápido
    from config_real import PLANT, CLIENTS, API_KEY
    addresses = [PLANT["address"]] + [c["address"] for c in CLIENTS]
    compute_matrix(addresses, API_KEY)
