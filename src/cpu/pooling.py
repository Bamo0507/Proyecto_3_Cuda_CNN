"""MaxPooling secuencial con ventanas 2x2 y paso 2, sin padding."""

import numpy as np
from numba import njit


@njit
def maxpool_forward(activaciones_entrada):
    """Devuelve maximos y posiciones locales 0-3, recorridas por filas."""
    cantidad_imagenes, alto_entrada, ancho_entrada, cantidad_canales = (
        activaciones_entrada.shape
    )
    if alto_entrada < 2 or ancho_entrada < 2:
        raise ValueError("La entrada debe permitir al menos una ventana 2x2.")

    alto_salida = alto_entrada // 2
    ancho_salida = ancho_entrada // 2
    forma_salida = (cantidad_imagenes, alto_salida, ancho_salida, cantidad_canales)
    valores_maximos = np.empty(forma_salida, dtype=np.float32)
    posiciones_maximos = np.empty(forma_salida, dtype=np.int32)

    for indice_imagen in range(cantidad_imagenes):
        for fila_salida in range(alto_salida):
            for columna_salida in range(ancho_salida):
                fila_inicio = fila_salida * 2
                columna_inicio = columna_salida * 2

                for indice_canal in range(cantidad_canales):
                    valor_maximo = activaciones_entrada[
                        indice_imagen, fila_inicio, columna_inicio, indice_canal
                    ]
                    posicion_maximo = 0

                    for desplazamiento_fila in range(2):
                        for desplazamiento_columna in range(2):
                            valor_actual = activaciones_entrada[
                                indice_imagen,
                                fila_inicio + desplazamiento_fila,
                                columna_inicio + desplazamiento_columna,
                                indice_canal,
                            ]
                            if valor_actual > valor_maximo:
                                valor_maximo = valor_actual
                                posicion_maximo = (
                                    desplazamiento_fila * 2 + desplazamiento_columna
                                )

                    valores_maximos[
                        indice_imagen, fila_salida, columna_salida, indice_canal
                    ] = valor_maximo
                    posiciones_maximos[
                        indice_imagen, fila_salida, columna_salida, indice_canal
                    ] = posicion_maximo

    return valores_maximos, posiciones_maximos


@njit
def maxpool_backward(gradiente_salida, posiciones_maximos, forma_entrada):
    cantidad_imagenes, alto_entrada, ancho_entrada, cantidad_canales = forma_entrada
    forma_salida = (
        cantidad_imagenes,
        alto_entrada // 2,
        ancho_entrada // 2,
        cantidad_canales,
    )

    if gradiente_salida.shape != forma_salida:
        raise ValueError("El gradiente no coincide con la salida del pooling.")
    if posiciones_maximos.shape != forma_salida:
        raise ValueError("Las posiciones no coinciden con la salida del pooling.")

    gradiente_entrada = np.zeros(forma_entrada, dtype=np.float32)

    for indice_imagen in range(cantidad_imagenes):
        for fila_salida in range(forma_salida[1]):
            for columna_salida in range(forma_salida[2]):
                for indice_canal in range(cantidad_canales):
                    posicion_maximo = posiciones_maximos[
                        indice_imagen, fila_salida, columna_salida, indice_canal
                    ]
                    fila_entrada = fila_salida * 2 + posicion_maximo // 2
                    columna_entrada = columna_salida * 2 + posicion_maximo % 2

                    gradiente_entrada[
                        indice_imagen, fila_entrada, columna_entrada, indice_canal
                    ] = gradiente_salida[
                        indice_imagen, fila_salida, columna_salida, indice_canal
                    ]

    return gradiente_entrada
