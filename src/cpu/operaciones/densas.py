"""Operaciones de capas densas para la implementacion CPU secuencial."""

import numpy as np
from numba import njit


@njit
def densa_forward(activaciones_entrada, pesos_capa, bias_capa):
    cantidad_imagenes, cantidad_entradas = activaciones_entrada.shape
    cantidad_entradas_pesos, cantidad_neuronas = pesos_capa.shape

    if cantidad_entradas_pesos != cantidad_entradas:
        raise ValueError("Los pesos deben tener una fila por entrada de la capa.")
    if bias_capa.shape != (cantidad_neuronas,):
        raise ValueError("Debe existir un bias por neurona.")

    salida = np.empty((cantidad_imagenes, cantidad_neuronas), dtype=np.float32)

    for indice_imagen in range(cantidad_imagenes):
        for indice_neurona in range(cantidad_neuronas):
            valor_salida = bias_capa[indice_neurona]
            for indice_entrada in range(cantidad_entradas):
                valor_salida += (
                    activaciones_entrada[indice_imagen, indice_entrada]
                    * pesos_capa[indice_entrada, indice_neurona]
                )
            salida[indice_imagen, indice_neurona] = valor_salida

    return salida


@njit
def densa_backward(gradiente_salida, activaciones_entrada, pesos_capa):
    """Calcula gradientes sin normalizarlos nuevamente ni actualizar parametros."""
    cantidad_imagenes, cantidad_entradas = activaciones_entrada.shape
    cantidad_entradas_pesos, cantidad_neuronas = pesos_capa.shape

    if cantidad_entradas_pesos != cantidad_entradas:
        raise ValueError("Los pesos deben tener una fila por entrada de la capa.")
    if gradiente_salida.shape != (cantidad_imagenes, cantidad_neuronas):
        raise ValueError("El gradiente no coincide con la salida de la capa densa.")

    gradiente_entrada = np.zeros_like(activaciones_entrada)
    gradiente_pesos = np.zeros_like(pesos_capa)
    gradiente_bias = np.zeros(cantidad_neuronas, dtype=np.float32)

    for indice_imagen in range(cantidad_imagenes):
        for indice_neurona in range(cantidad_neuronas):
            gradiente_actual = gradiente_salida[indice_imagen, indice_neurona]
            gradiente_bias[indice_neurona] += gradiente_actual

            for indice_entrada in range(cantidad_entradas):
                gradiente_entrada[indice_imagen, indice_entrada] += (
                    pesos_capa[indice_entrada, indice_neurona] * gradiente_actual
                )
                gradiente_pesos[indice_entrada, indice_neurona] += (
                    activaciones_entrada[indice_imagen, indice_entrada]
                    * gradiente_actual
                )

    return gradiente_entrada, gradiente_pesos, gradiente_bias
