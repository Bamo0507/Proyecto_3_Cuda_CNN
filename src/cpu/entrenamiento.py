"""Entrenamiento de la CNN con operaciones CPU secuenciales."""

import numpy as np

import config
from cpu.cnn import cnn_backward, cnn_forward
from cpu.optimizador import adam_actualizar
from cpu.perdida import cross_entropy_forward, softmax_cross_entropy_backward


def inicializar_estado_adam(parametros):
    """Crea momentos independientes para cada parametro y un contador global."""
    return {
        "numero_paso": 0,
        "primer_momento": {
            nombre_parametro: np.zeros_like(valores_parametro)
            for nombre_parametro, valores_parametro in parametros.items()
        },
        "segundo_momento": {
            nombre_parametro: np.zeros_like(valores_parametro)
            for nombre_parametro, valores_parametro in parametros.items()
        },
    }


def paso_entrenamiento(
    imagenes,
    etiquetas,
    parametros,
    estado_adam,
    tasa_aprendizaje=config.TASA_APRENDIZAJE,
):
    """Actualiza parametros y estado en sitio; devuelve metricas anteriores al ajuste."""
    probabilidades, cache = cnn_forward(imagenes, parametros)
    perdida = cross_entropy_forward(cache["salida_densa2"], etiquetas)
    if not np.isfinite(perdida):
        raise ValueError("La perdida no es finita; no se actualizaron los parametros.")

    gradiente_salida = softmax_cross_entropy_backward(probabilidades, etiquetas)
    gradientes = cnn_backward(gradiente_salida, parametros, cache)
    numero_paso = estado_adam["numero_paso"] + 1

    for nombre_parametro, valores_parametro in parametros.items():
        adam_actualizar(
            valores_parametro,
            gradientes[nombre_parametro],
            estado_adam["primer_momento"][nombre_parametro],
            estado_adam["segundo_momento"][nombre_parametro],
            numero_paso,
            tasa_aprendizaje=tasa_aprendizaje,
        )

    estado_adam["numero_paso"] = numero_paso
    clases_predichas = probabilidades.argmax(axis=1)
    return {
        "perdida": float(perdida),
        "accuracy": float(np.mean(clases_predichas == etiquetas)),
        "cantidad_imagenes": len(imagenes),
    }


def entrenar(
    imagenes,
    etiquetas,
    parametros,
    cantidad_epocas,
    tamano_lote=config.TAMANO_LOTE,
    tasa_aprendizaje=config.TASA_APRENDIZAJE,
    semilla=config.SEMILLA_ALEATORIA,
    estado_adam=None,
):
    """Actualiza parametros en sitio y devuelve historial y estado de Adam.

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

    if estado_adam is None:
        estado_adam = inicializar_estado_adam(parametros)

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
                parametros,
                estado_adam,
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
            "pasos_acumulados": estado_adam["numero_paso"],
        })

    return historial, estado_adam
