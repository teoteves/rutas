# config_real.py

# 1) Planta de cementos
PLANT = {
    "id": "PLANTA",
    "address": "UNACEM, Atocongo 2440, Villa María del Triunfo 15822"
}

# 2) Clientes: id, address, demand_tons
CLIENTS = [
    {
        "id": "C1",
        "name": "Promart Ate",
        "address": "Promart Ate, Av. la Molina Cdra. 3, Ate 15012",
        "demand_tons": 15.0,
    },
    {
        "id": "C2",
        "name": "Promart La Molina",
        "address": "Promart La Molina, Cruce de Av. la Molina con La Molina 15026",
        "demand_tons": 10.0,
    },
    {
        "id": "C3",
        "name": "Promart Chorrillos",
        "address": "Promart Chorrillos, Av. Guardia Civil Sur 927, Urb. Chorrillos 15056",
        "demand_tons": 8.0,
    },
    {
        "id": "C4",
        "name": "Sodimac San Luis",
        "address": "Sodimac San Luis, Av. Nicolás Ayllón 1680, San Luis 15019",
        "demand_tons": 10.0,
    },
    {
        "id": "C5",
        "name": "Sodimac Maestro Surquillo",
        "address": "Sodimac Maestro, Av. Angamos Este 1353, Surquillo 15048",
        "demand_tons": 6.0,
    },
    {
        "id": "C6",
        "name": "Sodimac Atocongo",
        "address": "Sodimac Atocongo, Circunvalación 1803, San Juan de Miraflores 15803",
        "demand_tons": 10.0,
    },
    {
        "id": "C7",
        "name": "Sodimac Chacarilla",
        "address": "Sodimac Chacarilla, Av. Caminos del Inca 100, Santiago de Surco 15037",
        "demand_tons": 4.0,
    },
    {
        "id": "C8",
        "name": "Sodimac Javier Prado",
        "address": "Sodimac Javier Prado, Av Javier Prado Este 1059, La Victoria 15034",
        "demand_tons": 5.0,
    },
    {
        "id": "C9",
        "name": "Sodimac Centro de Lima",
        "address": "Sodimac Centro de Lima, Cruce con Moquegua, Av. Tacna 640, Lima 15001",
        "demand_tons": 14.0,
    },
    {
        "id": "C10",
        "name": "Sodimac Puruchuco - Ate",
        "address": "Sodimac Puruchuco, Av. Nicolás Ayllón 4770, Ate 15494",
        "demand_tons": 13.0,
    },
    {
        "id": "C11",
        "name": "Sodimac San Juan de Lurigancho",
        "address": "Sodimac San Juan de Lurigancho, Av. Las Lomas 601-649, San Juan de Lurigancho 15427",
        "demand_tons": 16.0,
    },
    {
        "id": "C12",
        "name": "Promart San Miguel",
        "address": "Promart San Miguel, Av. Venezuela 5415, San Miguel 15088",
        "demand_tons": 6.0,
    },
    {
        "id": "C13",
        "name": "MACISA",
        "address": "Macisa, Maríe Curie 123, Ate 15022",
        "demand_tons": 5.0,
    },
    {
        "id": "C14",
        "name": "CEMENSA",
        "address": "CEMENSA, XXHW+Q3X, Av. Próceres de la Independencia, San Juan de Lurigancho 15401",
        "demand_tons": 4.0,
    },
    {
        "id": "C15",
        "name": "Cemento Nacional",
        "address": "Cemento Nacional, Av. Próceres de la Independencia 1800, San Juan de Lurigancho 15431",
        "demand_tons": 6.0,
    },
    {
        "id": "C16",
        "name": "Cementos Sol",
        "address": "Cementos Sol, Jr. Centenario 1407, Breña 15083",
        "demand_tons": 5.0,
    },
    {
        "id": "C17",
        "name": "FADICO SAC",
        "address": "FADICO SAC, Av. El Sol 651, San Juan de Lurigancho 15434",
        "demand_tons": 4.0,
    },
    {
        "id": "C18",
        "name": "CEMFINOR",
        "address": "CEMFINOR, XWQQ+PG2, Av. Alfredo Mendiola, San Martín de Porres 15103",
        "demand_tons": 6.0,
    },
    {
        "id": "C19",
        "name": "Ferretería y Agregados Zeus",
        "address": "Ferretería y Agregados Zeus, Jirón Castrovirreyna 775, Breña 15083",
        "demand_tons": 4.0,
    },
    {
        "id": "C20",
        "name": "PIEDRATEK",
        "address": "PIEDRATEK, Centro Comercial Jessie, Int. Aj - 13, Av. Nicolás Ayllón 3080, Ate 15023",
        "demand_tons": 7.0,
    },
    # ... agrega más clientes ...
]

# 3) Rango de demanda permitido (validación)
DEMAND_MIN = 3.0   # t
DEMAND_MAX = 30.0  # t

# 4) Flota de camiones (dos tipos)
TRUCK_TYPES = {
    "T1": {  # camión de mayor capacidad
        "capacity_tons": 30.0,   # o lo que quieras
        "fixed_cost": 800.0,
        "count": 20
    },
    "T2": {  # camión de menor capacidad
        "capacity_tons": 10.0,
        "fixed_cost": 300.0,
        "count": 20
    }
}

# 5) Parámetros de rutas
MAX_CLIENTES_POR_RUTA = 3
T_MAX_MIN = 180   # minutos

COSTO_POR_KM = 10.0
COSTO_POR_MIN = 5.0


