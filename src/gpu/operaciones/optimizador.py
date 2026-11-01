"""Adam CUDA: cada hilo actualiza un parametro y sus dos momentos."""

import math

from numba import cuda

import config


@cuda.jit
def adam_actualizar_kernel(
    parametros,
    gradientes,
    primer_momento,
    segundo_momento,
    tasa_corregida,
    beta_primer_momento,
    beta_segundo_momento,
    epsilon,
):
    indice_parametro = cuda.grid(1)
    if indice_parametro >= parametros.size:
        return

    gradiente_actual = gradientes[indice_parametro]
    primer_momento[indice_parametro] = (
        beta_primer_momento * primer_momento[indice_parametro]
        + (1 - beta_primer_momento) * gradiente_actual
    )
    segundo_momento[indice_parametro] = (
        beta_segundo_momento * segundo_momento[indice_parametro]
        + (1 - beta_segundo_momento) * gradiente_actual * gradiente_actual
    )
    parametros[indice_parametro] -= (
        tasa_corregida * primer_momento[indice_parametro]
        / (math.sqrt(segundo_momento[indice_parametro]) + epsilon)
    )


def adam_actualizar(
    parametros_device,
    gradientes_device,
    primer_momento_device,
    segundo_momento_device,
    numero_paso,
    tasa_aprendizaje=config.TASA_APRENDIZAJE,
    beta_primer_momento=config.BETA_PRIMER_MOMENTO_ADAM,
    beta_segundo_momento=config.BETA_SEGUNDO_MOMENTO_ADAM,
    epsilon=config.EPSILON_ADAM,
):
    """Actualiza buffers en sitio, sin transferir datos ni sincronizar."""
    if (
        parametros_device.shape != gradientes_device.shape
        or parametros_device.shape != primer_momento_device.shape
        or parametros_device.shape != segundo_momento_device.shape
    ):
        raise ValueError("Parametros, gradientes y momentos deben tener la misma forma.")
    if numero_paso < 1:
        raise ValueError("El contador de pasos debe comenzar en 1.")
    if tasa_aprendizaje <= 0 or epsilon <= 0:
        raise ValueError("La tasa de aprendizaje y epsilon deben ser positivos.")
    if not 0 <= beta_primer_momento < 1 or not 0 <= beta_segundo_momento < 1:
        raise ValueError("Los factores de los momentos deben estar entre 0 y 1, sin incluir 1.")
    if parametros_device.size == 0:
        return parametros_device

    tasa_corregida = tasa_aprendizaje * math.sqrt(
        1 - beta_segundo_momento ** numero_paso
    ) / (1 - beta_primer_momento ** numero_paso)

    hilos_por_bloque = 128
    cantidad_bloques = (parametros_device.size + hilos_por_bloque - 1) // hilos_por_bloque
    adam_actualizar_kernel[cantidad_bloques, hilos_por_bloque](
        parametros_device.ravel(),
        gradientes_device.ravel(),
        primer_momento_device.ravel(),
        segundo_momento_device.ravel(),
        tasa_corregida,
        beta_primer_momento,
        beta_segundo_momento,
        epsilon,
    )
    return parametros_device
