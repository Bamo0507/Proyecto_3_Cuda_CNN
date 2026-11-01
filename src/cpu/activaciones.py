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
