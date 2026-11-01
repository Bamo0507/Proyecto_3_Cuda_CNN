"""Recorridos forward y backward de la CNN en CPU."""

import numpy as np

import config
from cpu.operaciones.activaciones import relu_backward, relu_forward, softmax_forward
from cpu.operaciones.convolucion import convolucion_backward, convolucion_forward
from cpu.operaciones.densas import densa_backward, densa_forward
from cpu.operaciones.pooling import maxpool_backward, maxpool_forward


def cnn_forward(imagenes, parametros, guardar_cache=True):
    """Devuelve probabilidades y los datos de backward; sin cache devuelve un diccionario vacio."""
    if imagenes.ndim != 4 or imagenes.shape[1:] != config.FORMA_ENTRADA:
        raise ValueError("Las imagenes deben tener forma (N, 64, 64, 1).")
    if len(imagenes) == 0:
        raise ValueError("Debe existir al menos una imagen.")
    if imagenes.dtype != np.dtype(config.TIPO_DATOS):
        raise ValueError(f"Las imagenes deben ser {config.TIPO_DATOS}.")

    activaciones = imagenes
    bloques_convolucionales = []

    for numero_capa in range(1, len(config.FILTROS_CONVOLUCION) + 1):
        entrada_convolucion = activaciones
        salida_convolucion = convolucion_forward(
            entrada_convolucion,
            parametros[f"conv{numero_capa}_pesos"],
            parametros[f"conv{numero_capa}_bias"],
        )
        activaciones_relu = relu_forward(salida_convolucion)
        activaciones, posiciones_maximos = maxpool_forward(activaciones_relu)

        if guardar_cache:
            bloques_convolucionales.append({
                "entrada_convolucion": entrada_convolucion,
                "salida_convolucion": salida_convolucion,
                "forma_entrada_pooling": activaciones_relu.shape,
                "posiciones_maximos": posiciones_maximos,
            })

    entrada_densa1 = activaciones.reshape(len(imagenes), -1)
    salida_densa1 = densa_forward(
        entrada_densa1, parametros["densa1_pesos"], parametros["densa1_bias"]
    )
    entrada_densa2 = relu_forward(salida_densa1)
    salida_densa2 = densa_forward(
        entrada_densa2, parametros["densa2_pesos"], parametros["densa2_bias"]
    )
    probabilidades = softmax_forward(salida_densa2)

    cache = {}
    if guardar_cache:
        cache = {
            "bloques_convolucionales": bloques_convolucionales,
            "forma_salida_pooling": activaciones.shape,
            "entrada_densa1": entrada_densa1,
            "salida_densa1": salida_densa1,
            "entrada_densa2": entrada_densa2,
            "salida_densa2": salida_densa2,
        }

    return probabilidades, cache


def cnn_backward(gradiente_salida, parametros, cache):
    """Recibe el gradiente respecto a logits y devuelve gradientes por parametro."""
    if not cache:
        raise ValueError("Backward requiere ejecutar forward con guardar_cache=True.")

    gradientes = {}
    gradiente_activaciones, gradientes["densa2_pesos"], gradientes["densa2_bias"] = (
        densa_backward(
            gradiente_salida, cache["entrada_densa2"], parametros["densa2_pesos"]
        )
    )
    gradiente_activaciones = relu_backward(
        gradiente_activaciones, cache["salida_densa1"]
    )
    gradiente_activaciones, gradientes["densa1_pesos"], gradientes["densa1_bias"] = (
        densa_backward(
            gradiente_activaciones, cache["entrada_densa1"], parametros["densa1_pesos"]
        )
    )
    gradiente_activaciones = gradiente_activaciones.reshape(cache["forma_salida_pooling"])

    bloques_convolucionales = cache["bloques_convolucionales"]
    for indice_bloque in range(len(bloques_convolucionales) - 1, -1, -1):
        numero_capa = indice_bloque + 1
        bloque_cache = bloques_convolucionales[indice_bloque]
        gradiente_activaciones = maxpool_backward(
            gradiente_activaciones,
            bloque_cache["posiciones_maximos"],
            bloque_cache["forma_entrada_pooling"],
        )
        gradiente_activaciones = relu_backward(
            gradiente_activaciones, bloque_cache["salida_convolucion"]
        )
        nombre_pesos = f"conv{numero_capa}_pesos"
        nombre_bias = f"conv{numero_capa}_bias"
        gradiente_activaciones, gradientes[nombre_pesos], gradientes[nombre_bias] = (
            convolucion_backward(
                gradiente_activaciones,
                bloque_cache["entrada_convolucion"],
                parametros[nombre_pesos],
            )
        )

    return gradientes
