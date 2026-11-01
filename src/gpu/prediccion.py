"""Prediccion y evaluacion GPU sin modificar pesos ni estado de Adam."""

import numpy as np

import config
from gpu.cnn import cnn_forward
from gpu.memoria import cargar_lote
from gpu.operaciones.perdida import cross_entropy_forward


def predecir(imagenes, memoria):
    """Copia un lote a GPU y devuelve sus etiquetas y probabilidades en RAM."""
    cargar_lote(memoria, imagenes)
    probabilidades_device = cnn_forward(memoria)
    probabilidades_host = probabilidades_device.copy_to_host()
    predicciones = []

    for indice_imagen in range(len(imagenes)):
        indice_clase = int(probabilidades_host[indice_imagen].argmax())
        predicciones.append({
            "indice_clase": indice_clase,
            "etiqueta": config.CLASES[indice_clase],
            "probabilidad": float(probabilidades_host[indice_imagen, indice_clase]),
        })

    return predicciones


def evaluar(imagenes, etiquetas, memoria, tamano_lote=config.TAMANO_LOTE):
    """Calcula perdida y accuracy por lotes sin ejecutar backward ni Adam."""
    cantidad_imagenes = len(imagenes)
    if cantidad_imagenes == 0:
        raise ValueError("Debe existir al menos una imagen para evaluar.")
    if etiquetas.shape != (cantidad_imagenes,):
        raise ValueError("Debe existir una etiqueta por imagen.")
    if tamano_lote < 1 or tamano_lote > memoria["tamano_lote_maximo"]:
        raise ValueError("El tamano de lote debe estar dentro de la capacidad reservada.")

    buffers = memoria["buffers"]
    perdida_acumulada = 0.0
    cantidad_aciertos = 0

    for inicio_lote in range(0, cantidad_imagenes, tamano_lote):
        imagenes_lote = imagenes[inicio_lote:inicio_lote + tamano_lote]
        etiquetas_lote = etiquetas[inicio_lote:inicio_lote + tamano_lote]
        cantidad_imagenes_lote = len(imagenes_lote)
        cargar_lote(memoria, imagenes_lote, etiquetas_lote)
        probabilidades_device = cnn_forward(memoria)

        cross_entropy_forward(
            buffers["densa2_salida"][:cantidad_imagenes_lote],
            buffers["etiquetas"][:cantidad_imagenes_lote],
            buffers["perdidas_individuales"][:cantidad_imagenes_lote],
            buffers["perdida_promedio"],
        )
        perdida_lote = float(buffers["perdida_promedio"].copy_to_host()[0])
        if not np.isfinite(perdida_lote):
            raise ValueError("La perdida de evaluacion no es finita.")

        probabilidades_host = probabilidades_device.copy_to_host()
        perdida_acumulada += perdida_lote * cantidad_imagenes_lote
        cantidad_aciertos += int(np.count_nonzero(
            probabilidades_host.argmax(axis=1) == etiquetas_lote
        ))

    return {
        "perdida": perdida_acumulada / cantidad_imagenes,
        "accuracy": cantidad_aciertos / cantidad_imagenes,
        "cantidad_imagenes": cantidad_imagenes,
    }
