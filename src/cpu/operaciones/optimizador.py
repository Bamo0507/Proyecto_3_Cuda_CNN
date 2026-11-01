"""Actualizacion Adam secuencial con la convencion de epsilon de Keras."""

import numpy as np
from numba import njit

import config


@njit
def adam_actualizar(
    parametros,
    gradientes,
    primer_momento,
    segundo_momento,
    numero_paso,
    tasa_aprendizaje=config.TASA_APRENDIZAJE,
    beta_primer_momento=config.BETA_PRIMER_MOMENTO_ADAM,
    beta_segundo_momento=config.BETA_SEGUNDO_MOMENTO_ADAM,
    epsilon=config.EPSILON_ADAM,
):
    """Modifica parametros y momentos en sitio; usa un mismo paso para todo el batch."""
    if (
        parametros.shape != gradientes.shape
        or parametros.shape != primer_momento.shape
        or parametros.shape != segundo_momento.shape
    ):
        raise ValueError("Parametros, gradientes y momentos deben tener la misma forma.")
    if numero_paso < 1:
        raise ValueError("El contador de pasos debe comenzar en 1.")
    if tasa_aprendizaje <= 0 or epsilon <= 0:
        raise ValueError("La tasa de aprendizaje y epsilon deben ser positivos.")
    if not 0 <= beta_primer_momento < 1 or not 0 <= beta_segundo_momento < 1:
        raise ValueError("Los factores de los momentos deben estar entre 0 y 1, sin incluir 1.")

    tasa_corregida = tasa_aprendizaje * np.sqrt(
        1 - beta_segundo_momento ** numero_paso
    ) / (1 - beta_primer_momento ** numero_paso)

    for indice_parametro in range(parametros.size):
        gradiente_actual = gradientes.flat[indice_parametro]
        primer_momento.flat[indice_parametro] = (
            beta_primer_momento * primer_momento.flat[indice_parametro]
            + (1 - beta_primer_momento) * gradiente_actual
        )
        segundo_momento.flat[indice_parametro] = (
            beta_segundo_momento * segundo_momento.flat[indice_parametro]
            + (1 - beta_segundo_momento) * gradiente_actual * gradiente_actual
        )
        parametros.flat[indice_parametro] -= (
            tasa_corregida * primer_momento.flat[indice_parametro]
            / (np.sqrt(segundo_momento.flat[indice_parametro]) + epsilon)
        )
