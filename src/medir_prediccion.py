"""Mide la latencia de prediccion de una imagen en CPU o GPU con pesos guardados.

Ejecutar: python src/medir_prediccion.py --backend gpu --ruta-modelo models/x.npz \
    --ruta-registro resultados/x.json

Se usa el mismo archivo de pesos y las mismas imagenes de validacion en ambos
backends. Las imagenes se eligen con la semilla de config.py. La carga de pesos
(y en GPU la reserva de buffers y copia de pesos) se registra aparte. Primero se
calienta con imagenes distintas a las medidas; despues se mide cada prediccion
individual. En GPU cada tiempo incluye enviar la imagen, forward y recuperar las
probabilidades a RAM.
"""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time

import numba
import numpy as np

import config
from compartido.persistencia import cargar_parametros


def main():
    argumentos_parser = argparse.ArgumentParser(description=__doc__)
    argumentos_parser.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    argumentos_parser.add_argument("--ruta-modelo", type=Path, required=True)
    argumentos_parser.add_argument("--ruta-registro", type=Path, required=True)
    argumentos_parser.add_argument("--ruta-tensores", type=Path, default=config.RUTA_TENSORES)
    argumentos_parser.add_argument("--cantidad-mediciones", type=int, default=100)
    argumentos_parser.add_argument("--cantidad-calentamiento", type=int, default=10)
    argumentos = argumentos_parser.parse_args()

    if argumentos.ruta_registro.exists():
        raise FileExistsError(f"La salida ya existe; elige otra ruta: {argumentos.ruta_registro}")

    with np.load(argumentos.ruta_tensores, allow_pickle=False) as archivo_tensores:
        if not np.array_equal(archivo_tensores["clases"], np.asarray(config.CLASES)):
            raise ValueError("El orden de clases del dataset no coincide con config.py.")
        imagenes_validacion = archivo_tensores["x_val"]
        etiquetas_validacion = archivo_tensores["y_val"]

    total_necesario = argumentos.cantidad_calentamiento + argumentos.cantidad_mediciones
    if total_necesario > len(imagenes_validacion):
        raise ValueError("No hay suficientes imagenes de validacion distintas.")
    generador_aleatorio = np.random.default_rng(config.SEMILLA_ALEATORIA)
    indices = generador_aleatorio.choice(
        len(imagenes_validacion), size=total_necesario, replace=False
    )
    indices_calentamiento = indices[:argumentos.cantidad_calentamiento]
    indices_medicion = indices[argumentos.cantidad_calentamiento:]

    inicio_carga = time.perf_counter()
    parametros = cargar_parametros(argumentos.ruta_modelo)
    tiempo_carga_pesos = time.perf_counter() - inicio_carga

    tiempos_extra = {}
    dispositivo_info = None
    if argumentos.backend == "cpu":
        from cpu.prediccion import predecir

        def predecir_una(imagen):
            return predecir(imagen, parametros)[0]

        sincronizar = lambda: None
    else:
        from numba import cuda
        from gpu.memoria import preparar_memoria
        from gpu.prediccion import predecir

        if cuda.config.ENABLE_CUDASIM or not cuda.is_available():
            raise RuntimeError("Se requiere una GPU NVIDIA real.")
        dispositivo = cuda.get_current_device()
        nombre_dispositivo = dispositivo.name
        if isinstance(nombre_dispositivo, bytes):
            nombre_dispositivo = nombre_dispositivo.decode("utf-8")
        dispositivo_info = {
            "nombre": nombre_dispositivo,
            "capacidad_computo": dispositivo.compute_capability,
        }

        cuda.synchronize()
        inicio_preparacion = time.perf_counter()
        memoria = preparar_memoria(parametros, tamano_lote=1, entrenamiento=False)
        cuda.synchronize()
        tiempos_extra["preparacion_memoria_y_copia_pesos"] = (
            time.perf_counter() - inicio_preparacion
        )

        def predecir_una(imagen):
            return predecir(imagen, memoria)[0]

        sincronizar = cuda.synchronize

    tiempos_calentamiento = []
    for indice_imagen in indices_calentamiento:
        imagen = imagenes_validacion[indice_imagen:indice_imagen + 1]
        sincronizar()
        inicio = time.perf_counter()
        predecir_una(imagen)
        sincronizar()
        tiempos_calentamiento.append(time.perf_counter() - inicio)

    tiempos_prediccion = []
    aciertos = 0
    predicciones = []
    for indice_imagen in indices_medicion:
        imagen = imagenes_validacion[indice_imagen:indice_imagen + 1]
        sincronizar()
        inicio = time.perf_counter()
        prediccion = predecir_una(imagen)
        tiempos_prediccion.append(time.perf_counter() - inicio)
        acierto = prediccion["indice_clase"] == int(etiquetas_validacion[indice_imagen])
        aciertos += int(acierto)
        predicciones.append({
            "indice_validacion": int(indice_imagen),
            "real": int(etiquetas_validacion[indice_imagen]),
            "predicha": prediccion["indice_clase"],
            "probabilidad": prediccion["probabilidad"],
        })

    tiempos_ms = np.asarray(tiempos_prediccion) * 1000
    registro = {
        "backend": argumentos.backend,
        "ruta_modelo": str(argumentos.ruta_modelo.resolve()),
        "semilla": config.SEMILLA_ALEATORIA,
        "cantidad_calentamiento": argumentos.cantidad_calentamiento,
        "cantidad_mediciones": argumentos.cantidad_mediciones,
        "versiones": {
            "python": sys.version,
            "numpy": np.__version__,
            "numba": numba.__version__,
            **({"numba_cuda": version("numba-cuda")} if argumentos.backend == "gpu" else {}),
        },
        "dispositivo": dispositivo_info,
        "tiempos_segundos": {
            "carga_pesos_disco": tiempo_carga_pesos,
            **tiempos_extra,
            "primera_prediccion_con_compilacion": tiempos_calentamiento[0],
        },
        "frontera_medicion": (
            "Una imagen por llamada: forward y seleccion de etiqueta en CPU. "
            "En GPU incluye copia RAM->GPU de la imagen y copia GPU->RAM de las "
            "probabilidades; pesos y buffers ya residentes en GPU."
        ),
        "tiempos_calentamiento_ms": (np.asarray(tiempos_calentamiento) * 1000).tolist(),
        "tiempos_prediccion_ms": tiempos_ms.tolist(),
        "resumen_ms": {
            "promedio": float(tiempos_ms.mean()),
            "desviacion_estandar": float(tiempos_ms.std(ddof=1)),
            "mediana": float(np.median(tiempos_ms)),
            "minimo": float(tiempos_ms.min()),
            "maximo": float(tiempos_ms.max()),
        },
        "accuracy_muestra": aciertos / len(indices_medicion),
        "predicciones": predicciones,
    }

    argumentos.ruta_registro.parent.mkdir(parents=True, exist_ok=True)
    argumentos.ruta_registro.write_text(json.dumps(registro, indent=2), encoding="utf-8")
    print(f"Backend: {argumentos.backend}", flush=True)
    print(f"Carga de pesos: {tiempo_carga_pesos * 1000:.3f} ms", flush=True)
    for nombre, valor in tiempos_extra.items():
        print(f"{nombre}: {valor * 1000:.3f} ms", flush=True)
    print(f"Primera prediccion (compilacion): {tiempos_calentamiento[0]:.4f} s", flush=True)
    print(
        f"Latencia: {tiempos_ms.mean():.4f} +- {tiempos_ms.std(ddof=1):.4f} ms "
        f"(mediana {np.median(tiempos_ms):.4f} ms, n={len(tiempos_ms)})",
        flush=True,
    )
    print(f"Registro guardado: {argumentos.ruta_registro}", flush=True)


main()
