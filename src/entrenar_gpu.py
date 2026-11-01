"""Entrena la CNN CUDA, evalua validacion y guarda pesos e historial.

Ejecutar: python src/entrenar_gpu.py --epocas 1
El entrenamiento caliente incluye lotes y metricas, pero excluye reserva inicial,
compilacion, validacion y recuperacion final de pesos, registradas por separado.
"""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time

import numba
from numba import cuda
import numpy as np

import config
from compartido.datos import cargar_datos
from compartido.parametros import inicializar_parametros
from compartido.persistencia import guardar_parametros
from gpu.entrenamiento import entrenar, paso_entrenamiento
from gpu.memoria import copiar_parametros_a_host, preparar_memoria
from gpu.prediccion import evaluar


def main():
    argumentos_parser = argparse.ArgumentParser(description=__doc__)
    argumentos_parser.add_argument("--epocas", type=int, required=True)
    argumentos_parser.add_argument("--tamano-lote", type=int, default=config.TAMANO_LOTE)
    argumentos_parser.add_argument("--ruta-tensores", type=Path, default=config.RUTA_TENSORES)
    argumentos_parser.add_argument(
        "--ruta-modelo", type=Path,
        default=config.DIRECTORIO_MODELOS / "cnn_gpu.npz",
    )
    argumentos_parser.add_argument(
        "--ruta-registro", type=Path,
        default=config.DIRECTORIO_RESULTADOS / "entrenamiento_gpu.json",
    )
    argumentos = argumentos_parser.parse_args()

    if argumentos.epocas < 1 or argumentos.tamano_lote < 1:
        argumentos_parser.error("Las epocas y el tamano de lote deben ser positivos.")
    if argumentos.ruta_modelo.resolve() == argumentos.ruta_registro.resolve():
        argumentos_parser.error("El modelo y el registro deben tener rutas diferentes.")
    for ruta_salida in (argumentos.ruta_modelo, argumentos.ruta_registro):
        if ruta_salida.exists():
            raise FileExistsError(f"La salida ya existe; elige otra ruta: {ruta_salida}")

    if cuda.config.ENABLE_CUDASIM:
        raise RuntimeError("Este entrenamiento requiere GPU real, no el simulador.")
    if not cuda.is_available():
        raise RuntimeError("CUDA no esta disponible. Selecciona una GPU NVIDIA en Colab.")
    dispositivo = cuda.get_current_device()
    nombre_dispositivo = dispositivo.name
    if isinstance(nombre_dispositivo, bytes):
        nombre_dispositivo = nombre_dispositivo.decode("utf-8")
    print("GPU:", nombre_dispositivo, flush=True)

    datos = cargar_datos(argumentos.ruta_tensores)
    parametros_iniciales = inicializar_parametros()
    print(
        f"Datos: {len(datos['x_train'])} entrenamiento, {len(datos['x_val'])} validacion.",
        flush=True,
    )

    inicio_preparacion_calentamiento = time.perf_counter()
    memoria_calentamiento = preparar_memoria(
        parametros_iniciales, tamano_lote=1, entrenamiento=True
    )
    cuda.synchronize()
    tiempo_preparacion_calentamiento = time.perf_counter() - inicio_preparacion_calentamiento

    print("Calentando y compilando los kernels con parametros descartables...", flush=True)
    inicio_calentamiento = time.perf_counter()
    paso_entrenamiento(datos["x_train"][:1], datos["y_train"][:1], memoria_calentamiento)
    cuda.synchronize()
    tiempo_calentamiento = time.perf_counter() - inicio_calentamiento
    del memoria_calentamiento

    inicio_preparacion = time.perf_counter()
    memoria = preparar_memoria(
        parametros_iniciales, tamano_lote=argumentos.tamano_lote, entrenamiento=True
    )
    cuda.synchronize()
    tiempo_preparacion = time.perf_counter() - inicio_preparacion

    print(
        f"Entrenando {argumentos.epocas} epocas con lotes de {argumentos.tamano_lote}...",
        flush=True,
    )
    cuda.synchronize()
    inicio_entrenamiento = time.perf_counter()
    historial, estado_adam = entrenar(
        datos["x_train"], datos["y_train"], memoria,
        cantidad_epocas=argumentos.epocas,
        tamano_lote=argumentos.tamano_lote,
    )
    cuda.synchronize()
    tiempo_entrenamiento = time.perf_counter() - inicio_entrenamiento

    for metricas_epoca in historial:
        print(
            f"Epoca {metricas_epoca['epoca']}: "
            f"perdida={metricas_epoca['perdida']:.4f}, "
            f"accuracy={metricas_epoca['accuracy']:.4f}",
            flush=True,
        )

    inicio_recuperacion = time.perf_counter()
    parametros_entrenados = copiar_parametros_a_host(memoria)
    cuda.synchronize()
    tiempo_recuperacion = time.perf_counter() - inicio_recuperacion

    print("Evaluando validacion, sin modificar pesos...", flush=True)
    cuda.synchronize()
    inicio_validacion = time.perf_counter()
    metricas_validacion = evaluar(
        datos["x_val"], datos["y_val"], memoria,
        tamano_lote=argumentos.tamano_lote,
    )
    cuda.synchronize()
    tiempo_validacion = time.perf_counter() - inicio_validacion

    registro = {
        "backend": "gpu",
        "ruta_tensores": str(argumentos.ruta_tensores.resolve()),
        "ruta_modelo": str(argumentos.ruta_modelo.resolve()),
        "configuracion": {
            "epocas": argumentos.epocas,
            "tamano_lote": argumentos.tamano_lote,
            "semilla": config.SEMILLA_ALEATORIA,
            "forma_entrada": config.FORMA_ENTRADA,
            "clases": config.CLASES,
            "tipo_datos": config.TIPO_DATOS,
            "filtros_convolucion": config.FILTROS_CONVOLUCION,
            "tamano_filtro": config.TAMANO_FILTRO,
            "paso_convolucion": config.PASO_CONVOLUCION,
            "padding_convolucion": config.PADDING_CONVOLUCION,
            "tamano_pooling": config.TAMANO_POOLING,
            "paso_pooling": config.PASO_POOLING,
            "neuronas_capa_densa": config.NEURONAS_CAPA_DENSA,
            "tasa_aprendizaje": config.TASA_APRENDIZAJE,
            "beta_primer_momento": config.BETA_PRIMER_MOMENTO_ADAM,
            "beta_segundo_momento": config.BETA_SEGUNDO_MOMENTO_ADAM,
            "epsilon_adam": config.EPSILON_ADAM,
        },
        "versiones": {
            "python": sys.version,
            "numpy": np.__version__,
            "numba": numba.__version__,
            "numba_cuda": version("numba-cuda"),
        },
        "dispositivo": {
            "nombre": nombre_dispositivo,
            "capacidad_computo": dispositivo.compute_capability,
        },
        "tiempos_segundos": {
            "preparacion_calentamiento_gpu": tiempo_preparacion_calentamiento,
            "calentamiento_y_compilacion": tiempo_calentamiento,
            "preparacion_gpu": tiempo_preparacion,
            "entrenamiento": tiempo_entrenamiento,
            "recuperacion_parametros": tiempo_recuperacion,
            "entrenamiento_con_preparacion_y_recuperacion": (
                tiempo_preparacion + tiempo_entrenamiento + tiempo_recuperacion
            ),
            "validacion": tiempo_validacion,
        },
        "frontera_entrenamiento": (
            "Lotes en RAM, transferencias de cada lote, kernels y metricas en RAM; "
            "pesos y momentos residentes en GPU."
        ),
        "historial": historial,
        "validacion": metricas_validacion,
        "pasos_entrenamiento": estado_adam["numero_paso"],
        "incluye_estado_adam": False,
    }

    guardar_parametros(parametros_entrenados, argumentos.ruta_modelo)
    argumentos.ruta_registro.parent.mkdir(parents=True, exist_ok=True)
    argumentos.ruta_registro.write_text(json.dumps(registro, indent=2), encoding="utf-8")
    print(f"Preparacion GPU: {tiempo_preparacion:.4f} s", flush=True)
    print(f"Entrenamiento caliente: {tiempo_entrenamiento:.4f} s", flush=True)
    print(f"Recuperacion de pesos: {tiempo_recuperacion:.4f} s", flush=True)
    print(f"Validacion: {metricas_validacion}", flush=True)
    print(f"Modelo guardado: {argumentos.ruta_modelo}", flush=True)
    print(f"Registro guardado: {argumentos.ruta_registro}", flush=True)


main()
