"""Funciones de activacion para la implementacion CPU secuencial."""

import numpy as np
from numba import njit


@njit
def relu_forward(valores_entrada):
    return np.maximum(valores_entrada, np.float32(0.0))


@njit
def relu_backward(gradiente_salida, valores_entrada):
    if gradiente_salida.shape != valores_entrada.shape:
        raise ValueError("El gradiente debe tener la misma forma que la entrada de ReLU.")

    return np.where(valores_entrada > 0, gradiente_salida, np.float32(0.0))


@njit
def softmax_forward(valores_salida):
    cantidad_imagenes, cantidad_clases = valores_salida.shape
    if cantidad_imagenes == 0 or cantidad_clases == 0:
        raise ValueError("Softmax necesita al menos una imagen y una clase.")

    probabilidades = np.empty_like(valores_salida)

    for indice_imagen in range(cantidad_imagenes):
        valor_maximo = np.max(valores_salida[indice_imagen])
        suma_exponenciales = np.float32(0.0)

        for indice_clase in range(cantidad_clases):
            valor_exponencial = np.exp(
                valores_salida[indice_imagen, indice_clase] - valor_maximo
            )
            probabilidades[indice_imagen, indice_clase] = valor_exponencial
            suma_exponenciales += valor_exponencial

        for indice_clase in range(cantidad_clases):
            probabilidades[indice_imagen, indice_clase] /= suma_exponenciales

    return probabilidades
