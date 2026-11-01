"""Guardado y carga de pesos y biases, sin el estado del optimizador."""

from pathlib import Path

import numpy as np


def guardar_parametros(parametros, ruta_archivo):
    """Guarda los arreglos en un NPZ comprimido y devuelve la ruta utilizada."""
    ruta_archivo = Path(ruta_archivo)
    ruta_archivo.parent.mkdir(parents=True, exist_ok=True)

    with ruta_archivo.open("wb") as archivo_parametros:
        np.savez_compressed(archivo_parametros, **parametros)

    return ruta_archivo


def cargar_parametros(ruta_archivo):
    """Recupera los arreglos del NPZ sin permitir objetos serializados con pickle."""
    with np.load(ruta_archivo, allow_pickle=False) as archivo_parametros:
        parametros = {
            nombre_parametro: archivo_parametros[nombre_parametro]
            for nombre_parametro in archivo_parametros.files
        }

    return parametros
