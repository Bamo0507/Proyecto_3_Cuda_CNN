"""Corre las mediciones de entrenamiento e inferencia CPU/GPU y guarda el entorno.

Ejecutar desde la raiz del repo:
    python experimentos/correr_experimentos.py --prefijo dani --repeticiones 3

Cada entrenamiento es un proceso nuevo que parte de los pesos iniciales de la
semilla de config.py (1 epoca, lote 64). La inferencia usa un unico archivo de
pesos (por defecto el del primer entrenamiento GPU) en ambos backends.
Las salidas que ya existen se saltan, asi se puede reanudar si se interrumpe.
"""

import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys

RAIZ = Path(__file__).resolve().parent.parent
SRC = RAIZ / "src"


def estado_energia():
    """En Windows: BatteryStatus 1 = descargando (bateria), 2 = conectado a corriente."""
    if platform.system() != "Windows":
        return None
    texto = salida_texto([
        "powershell", "-NoProfile", "-Command",
        "Get-CimInstance Win32_Battery | Select-Object BatteryStatus,"
        "EstimatedChargeRemaining | ConvertTo-Json",
    ])
    try:
        datos = json.loads(texto)
    except (TypeError, ValueError):
        return None
    datos["fuente"] = "bateria" if datos.get("BatteryStatus") == 1 else "corriente"
    return datos


def ejecutar(comando, ruta_registro=None):
    if ruta_registro is not None:
        ruta_energia = ruta_registro.with_name(f"energia_{ruta_registro.stem}.json")
        ruta_energia.write_text(json.dumps(estado_energia(), indent=2), encoding="utf-8")
    print(">>", " ".join(str(parte) for parte in comando), flush=True)
    subprocess.run([str(parte) for parte in comando], cwd=RAIZ, check=True)


def salida_texto(comando):
    try:
        return subprocess.check_output(comando, cwd=RAIZ, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def registrar_entorno(ruta):
    nombre_cpu = salida_texto([
        "powershell", "-NoProfile", "-Command",
        "(Get-CimInstance Win32_Processor).Name",
    ]) if platform.system() == "Windows" else platform.processor()
    entorno = {
        "commit": salida_texto(["git", "rev-parse", "HEAD"]),
        "cambios_sin_commit": salida_texto(["git", "status", "--short"]),
        "cpu": nombre_cpu,
        "gpu_nvidia_smi": salida_texto([
            "nvidia-smi", "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ]),
        "sistema": platform.platform(),
        "python": sys.version,
        "paquetes": salida_texto([sys.executable, "-m", "pip", "freeze"]),
    }
    ruta.write_text(json.dumps(entorno, indent=2), encoding="utf-8")
    print("Entorno guardado:", ruta, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefijo", default="dani")
    parser.add_argument("--repeticiones", type=int, default=3)
    parser.add_argument("--solo", choices=("cpu", "gpu", "prediccion"), action="append")
    parser.add_argument("--ruta-modelo-prediccion", type=Path, default=None)
    argumentos = parser.parse_args()
    etapas = set(argumentos.solo or ("cpu", "gpu", "prediccion"))

    directorio = RAIZ / "resultados" / argumentos.prefijo
    directorio_modelos = RAIZ / "models" / argumentos.prefijo
    directorio.mkdir(parents=True, exist_ok=True)
    directorio_modelos.mkdir(parents=True, exist_ok=True)
    registrar_entorno(directorio / "entorno.json")

    for backend in ("gpu", "cpu"):
        if backend not in etapas:
            continue
        for numero in range(1, argumentos.repeticiones + 1):
            nombre = f"{argumentos.prefijo}_{backend}_{numero:02d}"
            ruta_registro = directorio / f"entrenamiento_{nombre}.json"
            if ruta_registro.exists():
                print("Ya existe, se salta:", ruta_registro, flush=True)
                continue
            ejecutar([
                sys.executable, "-u", SRC / f"entrenar_{backend}.py",
                "--epocas", 1, "--tamano-lote", 64,
                "--ruta-modelo", directorio_modelos / f"{nombre}.npz",
                "--ruta-registro", ruta_registro,
            ], ruta_registro)

    if "prediccion" in etapas:
        ruta_modelo = argumentos.ruta_modelo_prediccion or (
            directorio_modelos / f"{argumentos.prefijo}_gpu_01.npz"
        )
        for backend in ("gpu", "cpu"):
            for numero in range(1, argumentos.repeticiones + 1):
                nombre = f"{argumentos.prefijo}_{backend}_{numero:02d}"
                ruta_registro = directorio / f"prediccion_{nombre}.json"
                if ruta_registro.exists():
                    print("Ya existe, se salta:", ruta_registro, flush=True)
                    continue
                ejecutar([
                    sys.executable, "-u", SRC / "medir_prediccion.py",
                    "--backend", backend,
                    "--ruta-modelo", ruta_modelo,
                    "--ruta-registro", ruta_registro,
                ], ruta_registro)


main()
