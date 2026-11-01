"""Convolucion secuencial con paso 1 y padding same para filtros 3x3."""

import numpy as np
from numba import njit


@njit
def convolucion_forward(imagenes, pesos_filtros, bias_filtros):
    cantidad_imagenes, alto_imagen, ancho_imagen, canales_entrada = imagenes.shape
    alto_filtro, ancho_filtro, canales_filtro, cantidad_filtros = pesos_filtros.shape

    if alto_filtro != 3 or ancho_filtro != 3:
        raise ValueError("Los filtros deben tener dimensiones espaciales 3x3.")
    if canales_filtro != canales_entrada:
        raise ValueError("Los filtros deben abarcar todos los canales de entrada.")
    if bias_filtros.shape != (cantidad_filtros,):
        raise ValueError("Debe existir un bias por filtro.")

    salida = np.empty(
        (cantidad_imagenes, alto_imagen, ancho_imagen, cantidad_filtros),
        dtype=np.float32,
    )

    for indice_imagen in range(cantidad_imagenes):
        for fila_salida in range(alto_imagen):
            for columna_salida in range(ancho_imagen):
                for indice_filtro in range(cantidad_filtros):
                    valor_salida = bias_filtros[indice_filtro]

                    for fila_filtro in range(alto_filtro):
                        fila_entrada = fila_salida + fila_filtro - 1
                        if fila_entrada < 0 or fila_entrada >= alto_imagen:
                            continue

                        for columna_filtro in range(ancho_filtro):
                            columna_entrada = columna_salida + columna_filtro - 1
                            if columna_entrada < 0 or columna_entrada >= ancho_imagen:
                                continue

                            for indice_canal in range(canales_entrada):
                                valor_pixel = imagenes[
                                    indice_imagen,
                                    fila_entrada,
                                    columna_entrada,
                                    indice_canal,
                                ]
                                peso_filtro = pesos_filtros[
                                    fila_filtro,
                                    columna_filtro,
                                    indice_canal,
                                    indice_filtro,
                                ]
                                valor_salida += valor_pixel * peso_filtro

                    salida[
                        indice_imagen,
                        fila_salida,
                        columna_salida,
                        indice_filtro,
                    ] = valor_salida

    return salida


@njit
def convolucion_backward(gradiente_salida, imagenes_entrada, pesos_filtros):
    """Calcula gradientes sin normalizarlos nuevamente ni actualizar parametros."""
    cantidad_imagenes, alto_imagen, ancho_imagen, canales_entrada = imagenes_entrada.shape
    alto_filtro, ancho_filtro, canales_filtro, cantidad_filtros = pesos_filtros.shape

    if alto_filtro != 3 or ancho_filtro != 3:
        raise ValueError("Los filtros deben tener dimensiones espaciales 3x3.")
    if canales_filtro != canales_entrada:
        raise ValueError("Los filtros deben abarcar todos los canales de entrada.")
    if gradiente_salida.shape != (
        cantidad_imagenes, alto_imagen, ancho_imagen, cantidad_filtros
    ):
        raise ValueError("El gradiente no coincide con la salida de la convolucion.")

    gradiente_entrada = np.zeros_like(imagenes_entrada)
    gradiente_pesos = np.zeros_like(pesos_filtros)
    gradiente_bias = np.zeros(cantidad_filtros, dtype=np.float32)

    for indice_imagen in range(cantidad_imagenes):
        for fila_salida in range(alto_imagen):
            for columna_salida in range(ancho_imagen):
                for indice_filtro in range(cantidad_filtros):
                    gradiente_actual = gradiente_salida[
                        indice_imagen, fila_salida, columna_salida, indice_filtro
                    ]
                    gradiente_bias[indice_filtro] += gradiente_actual

                    for fila_filtro in range(alto_filtro):
                        fila_entrada = fila_salida + fila_filtro - 1
                        if fila_entrada < 0 or fila_entrada >= alto_imagen:
                            continue

                        for columna_filtro in range(ancho_filtro):
                            columna_entrada = columna_salida + columna_filtro - 1
                            if columna_entrada < 0 or columna_entrada >= ancho_imagen:
                                continue

                            for indice_canal in range(canales_entrada):
                                gradiente_entrada[
                                    indice_imagen,
                                    fila_entrada,
                                    columna_entrada,
                                    indice_canal,
                                ] += pesos_filtros[
                                    fila_filtro,
                                    columna_filtro,
                                    indice_canal,
                                    indice_filtro,
                                ] * gradiente_actual

                                gradiente_pesos[
                                    fila_filtro,
                                    columna_filtro,
                                    indice_canal,
                                    indice_filtro,
                                ] += imagenes_entrada[
                                    indice_imagen,
                                    fila_entrada,
                                    columna_entrada,
                                    indice_canal,
                                ] * gradiente_actual

    return gradiente_entrada, gradiente_pesos, gradiente_bias
