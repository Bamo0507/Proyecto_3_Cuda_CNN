"""Entrenamiento de la CNN con calculos y parametros residentes en GPU."""

import numpy as np
from numba import cuda

import config
from gpu.cnn import cnn_backward, cnn_forward
from gpu.memoria import cargar_lote
from gpu.operaciones.optimizador import adam_actualizar
from gpu.operaciones.perdida import (
    cross_entropy_forward,
    softmax_cross_entropy_backward,
)


def paso_entrenamiento(
    imagenes,
    etiquetas,
    memoria,
    tasa_aprendizaje=config.TASA_APRENDIZAJE,
):
    """Actualiza parametros y momentos GPU; devuelve metricas anteriores al ajuste."""
    if not memoria["entrenamiento"] or memoria["estado_adam"] is None:
        raise ValueError("El paso requiere memoria preparada para entrenamiento.")
    if tasa_aprendizaje <= 0:
        raise ValueError("La tasa de aprendizaje debe ser positiva.")

    cargar_lote(memoria, imagenes, etiquetas)
    cantidad_imagenes = memoria["cantidad_imagenes"]
    buffers = memoria["buffers"]
    probabilidades_device = cnn_forward(memoria)
    etiquetas_device = buffers["etiquetas"][:cantidad_imagenes]

    cross_entropy_forward(
        buffers["densa2_salida"][:cantidad_imagenes],
        etiquetas_device,
        buffers["perdidas_individuales"][:cantidad_imagenes],
        buffers["perdida_promedio"],
    )
    perdida = float(buffers["perdida_promedio"].copy_to_host()[0])
    if not np.isfinite(perdida):
        raise ValueError("La perdida no es finita; no se actualizaron los parametros.")

    gradiente_logits_device = buffers["gradiente_logits"][:cantidad_imagenes]
    softmax_cross_entropy_backward(
        probabilidades_device, etiquetas_device, gradiente_logits_device
    )
    gradientes = cnn_backward(gradiente_logits_device, memoria)
    estado_adam = memoria["estado_adam"]
    numero_paso = estado_adam["numero_paso"] + 1

    for nombre_parametro, valores_parametro in memoria["parametros"].items():
        adam_actualizar(
            valores_parametro,
            gradientes[nombre_parametro],
            estado_adam["primer_momento"][nombre_parametro],
            estado_adam["segundo_momento"][nombre_parametro],
            numero_paso,
            tasa_aprendizaje=tasa_aprendizaje,
        )

    memoria["forward_disponible"] = False
    cuda.synchronize()
    estado_adam["numero_paso"] = numero_paso

    probabilidades_host = probabilidades_device.copy_to_host()
    clases_predichas = probabilidades_host.argmax(axis=1)
    return {
        "perdida": perdida,
        "accuracy": float(np.mean(clases_predichas == etiquetas)),
        "cantidad_imagenes": cantidad_imagenes,
    }


def entrenar(
    imagenes,
    etiquetas,
    memoria,
    cantidad_epocas,
    tamano_lote=config.TAMANO_LOTE,
    tasa_aprendizaje=config.TASA_APRENDIZAJE,
    semilla=config.SEMILLA_ALEATORIA,
):
    """Reutiliza memoria y momentos; devuelve historial y estado de Adam en device.

    Las metricas ponderan las predicciones anteriores al ajuste de cada lote.
    Cada llamada inicia la mezcla con la semilla indicada.
    """
    cantidad_imagenes = len(imagenes)
    if cantidad_imagenes == 0:
        raise ValueError("Debe existir al menos una imagen para entrenar.")
    if etiquetas.shape != (cantidad_imagenes,):
        raise ValueError("Debe existir una etiqueta por imagen.")
    if cantidad_epocas < 1 or tamano_lote < 1:
        raise ValueError("La cantidad de epocas y el tamano de lote deben ser positivos.")
    if tasa_aprendizaje <= 0:
        raise ValueError("La tasa de aprendizaje debe ser positiva.")
    if not memoria["entrenamiento"] or memoria["estado_adam"] is None:
        raise ValueError("El ciclo requiere memoria preparada para entrenamiento.")
    if tamano_lote > memoria["tamano_lote_maximo"]:
        raise ValueError("El tamano de lote supera la capacidad reservada en GPU.")

    generador_aleatorio = np.random.default_rng(semilla)
    historial = []

    for numero_epoca in range(1, cantidad_epocas + 1):
        orden_imagenes = generador_aleatorio.permutation(cantidad_imagenes)
        perdida_acumulada = 0.0
        aciertos_acumulados = 0.0
        cantidad_lotes = 0

        for inicio_lote in range(0, cantidad_imagenes, tamano_lote):
            indices_lote = orden_imagenes[inicio_lote:inicio_lote + tamano_lote]
            metricas_lote = paso_entrenamiento(
                imagenes[indices_lote],
                etiquetas[indices_lote],
                memoria,
                tasa_aprendizaje=tasa_aprendizaje,
            )
            cantidad_imagenes_lote = metricas_lote["cantidad_imagenes"]
            perdida_acumulada += metricas_lote["perdida"] * cantidad_imagenes_lote
            aciertos_acumulados += metricas_lote["accuracy"] * cantidad_imagenes_lote
            cantidad_lotes += 1

        historial.append({
            "epoca": numero_epoca,
            "perdida": perdida_acumulada / cantidad_imagenes,
            "accuracy": aciertos_acumulados / cantidad_imagenes,
            "cantidad_imagenes": cantidad_imagenes,
            "cantidad_lotes": cantidad_lotes,
            "pasos_acumulados": memoria["estado_adam"]["numero_paso"],
        })

    return historial, memoria["estado_adam"]
