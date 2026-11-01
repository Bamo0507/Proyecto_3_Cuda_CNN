"""Carga y valida los tensores preparados en el laboratorio."""

import numpy as np

import config


def cargar_datos(ruta_tensores=config.RUTA_TENSORES):
    """Devuelve imagenes, etiquetas y clases sin modificar las particiones."""
    particiones = ("train", "val", "test")
    nombres_requeridos = {"clases"}
    for particion in particiones:
        nombres_requeridos.update((f"x_{particion}", f"y_{particion}"))

    with np.load(ruta_tensores, allow_pickle=False) as archivo_tensores:
        nombres_faltantes = nombres_requeridos - set(archivo_tensores.files)
        if nombres_faltantes:
            raise ValueError(f"Faltan arreglos en el NPZ: {sorted(nombres_faltantes)}")
        datos = {
            nombre_arreglo: archivo_tensores[nombre_arreglo]
            for nombre_arreglo in nombres_requeridos
        }

    if not np.array_equal(datos["clases"], np.asarray(config.CLASES)):
        raise ValueError("Las clases del NPZ no coinciden con el orden de config.CLASES.")

    for particion in particiones:
        imagenes = datos[f"x_{particion}"]
        etiquetas = datos[f"y_{particion}"]

        if imagenes.ndim != 4 or imagenes.shape[1:] != config.FORMA_ENTRADA:
            raise ValueError(
                f"{particion}: las imagenes deben tener forma (N, {config.FORMA_ENTRADA})."
            )
        if imagenes.dtype != np.dtype(config.TIPO_DATOS):
            raise ValueError(f"{particion}: las imagenes deben ser {config.TIPO_DATOS}.")
        if etiquetas.ndim != 1 or len(imagenes) != len(etiquetas):
            raise ValueError(f"{particion}: debe existir una etiqueta por imagen.")
        if len(imagenes) == 0:
            raise ValueError(f"{particion}: la particion no puede estar vacia.")
        if not np.isfinite(imagenes).all():
            raise ValueError(f"{particion}: las imagenes contienen NaN o infinitos.")
        if imagenes.min() < 0 or imagenes.max() > 1:
            raise ValueError(f"{particion}: las intensidades deben estar entre 0 y 1.")
        if not np.issubdtype(etiquetas.dtype, np.integer):
            raise ValueError(f"{particion}: las etiquetas deben ser enteros.")
        if etiquetas.min() < 0 or etiquetas.max() >= config.CANTIDAD_CLASES:
            raise ValueError(f"{particion}: hay etiquetas fuera del rango de clases.")

    return datos
