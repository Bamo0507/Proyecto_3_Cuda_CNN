"""Prediccion y evaluacion de la CNN sin actualizar parametros."""

import numpy as np

import config
from cpu.cnn import cnn_forward
from cpu.perdida import cross_entropy_forward


def predecir(imagenes, parametros):
    """Devuelve la etiqueta mas probable y su probabilidad para cada imagen del lote."""
    probabilidades, cache = cnn_forward(imagenes, parametros, guardar_cache=False)
    predicciones = []

    for indice_imagen in range(len(imagenes)):
        indice_clase = int(probabilidades[indice_imagen].argmax())
        predicciones.append({
            "indice_clase": indice_clase,
            "etiqueta": config.CLASES[indice_clase],
            "probabilidad": float(probabilidades[indice_imagen, indice_clase]),
        })

    return predicciones


def evaluar(imagenes, etiquetas, parametros, tamano_lote=config.TAMANO_LOTE):
    """Calcula perdida desde logits y accuracy por lotes, sin backward ni Adam."""
    cantidad_imagenes = len(imagenes)
    if cantidad_imagenes == 0:
        raise ValueError("Debe existir al menos una imagen para evaluar.")
    if etiquetas.shape != (cantidad_imagenes,):
        raise ValueError("Debe existir una etiqueta por imagen.")
    if tamano_lote < 1:
        raise ValueError("El tamano de lote debe ser positivo.")

    perdida_acumulada = 0.0
    cantidad_aciertos = 0

    for inicio_lote in range(0, cantidad_imagenes, tamano_lote):
        imagenes_lote = imagenes[inicio_lote:inicio_lote + tamano_lote]
        etiquetas_lote = etiquetas[inicio_lote:inicio_lote + tamano_lote]
        probabilidades, cache = cnn_forward(imagenes_lote, parametros)
        perdida_lote = cross_entropy_forward(cache["salida_densa2"], etiquetas_lote)
        if not np.isfinite(perdida_lote):
            raise ValueError("La perdida de evaluacion no es finita.")

        perdida_acumulada += perdida_lote * len(imagenes_lote)
        cantidad_aciertos += int(np.count_nonzero(
            probabilidades.argmax(axis=1) == etiquetas_lote
        ))

    return {
        "perdida": perdida_acumulada / cantidad_imagenes,
        "accuracy": cantidad_aciertos / cantidad_imagenes,
        "cantidad_imagenes": cantidad_imagenes,
    }
