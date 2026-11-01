"""Convolucion CUDA 3x3, paso 1 y padding same, con una salida por hilo."""

from numba import cuda, float32


@cuda.jit
def convolucion_forward_kernel(
    imagenes, pesos_filtros, bias_filtros, salida
):
    indice_salida = cuda.grid(1)
    if indice_salida >= salida.size:
        return

    cantidad_filtros = salida.shape[3]
    ancho_salida = salida.shape[2]
    alto_salida = salida.shape[1]

    # Se recuperan las coordenadas NHWC del indice lineal asignado al hilo.
    indice_filtro = indice_salida % cantidad_filtros
    indice_posicion = indice_salida // cantidad_filtros
    columna_salida = indice_posicion % ancho_salida
    indice_posicion //= ancho_salida
    fila_salida = indice_posicion % alto_salida
    indice_imagen = indice_posicion // alto_salida

    valor_salida = bias_filtros[indice_filtro]
    for fila_filtro in range(3):
        fila_entrada = fila_salida + fila_filtro - 1
        if fila_entrada < 0 or fila_entrada >= imagenes.shape[1]:
            continue

        for columna_filtro in range(3):
            columna_entrada = columna_salida + columna_filtro - 1
            if columna_entrada < 0 or columna_entrada >= imagenes.shape[2]:
                continue

            for indice_canal in range(imagenes.shape[3]):
                valor_salida += imagenes[
                    indice_imagen, fila_entrada, columna_entrada, indice_canal
                ] * pesos_filtros[
                    fila_filtro, columna_filtro, indice_canal, indice_filtro
                ]

    salida[indice_imagen, fila_salida, columna_salida, indice_filtro] = valor_salida


def convolucion_forward(
    imagenes_device, pesos_filtros_device, bias_filtros_device, salida_device
):
    """Lanza el kernel sobre buffers device; no transfiere datos ni sincroniza."""
    cantidad_imagenes, alto_imagen, ancho_imagen, canales_entrada = imagenes_device.shape
    alto_filtro, ancho_filtro, canales_filtro, cantidad_filtros = pesos_filtros_device.shape

    if alto_filtro != 3 or ancho_filtro != 3:
        raise ValueError("Los filtros deben tener dimensiones espaciales 3x3.")
    if canales_filtro != canales_entrada:
        raise ValueError("Los filtros deben abarcar todos los canales de entrada.")
    if bias_filtros_device.shape != (cantidad_filtros,):
        raise ValueError("Debe existir un bias por filtro.")
    if salida_device.shape != (
        cantidad_imagenes, alto_imagen, ancho_imagen, cantidad_filtros
    ):
        raise ValueError("El buffer de salida no tiene la forma de la convolucion.")
    if salida_device.size == 0:
        return salida_device

    hilos_por_bloque = 128
    cantidad_bloques = (salida_device.size + hilos_por_bloque - 1) // hilos_por_bloque
    convolucion_forward_kernel[cantidad_bloques, hilos_por_bloque](
        imagenes_device, pesos_filtros_device, bias_filtros_device, salida_device
    )
    return salida_device


@cuda.jit
def convolucion_gradiente_entrada_kernel(
    gradiente_salida, pesos_filtros, gradiente_entrada
):
    indice_entrada = cuda.grid(1)
    if indice_entrada >= gradiente_entrada.size:
        return

    cantidad_canales = gradiente_entrada.shape[3]
    ancho_entrada = gradiente_entrada.shape[2]
    alto_entrada = gradiente_entrada.shape[1]
    indice_canal = indice_entrada % cantidad_canales
    indice_posicion = indice_entrada // cantidad_canales
    columna_entrada = indice_posicion % ancho_entrada
    indice_posicion //= ancho_entrada
    fila_entrada = indice_posicion % alto_entrada
    indice_imagen = indice_posicion // alto_entrada

    valor_gradiente = float32(0.0)
    for fila_filtro in range(3):
        fila_salida = fila_entrada - fila_filtro + 1
        if fila_salida < 0 or fila_salida >= gradiente_salida.shape[1]:
            continue

        for columna_filtro in range(3):
            columna_salida = columna_entrada - columna_filtro + 1
            if columna_salida < 0 or columna_salida >= gradiente_salida.shape[2]:
                continue

            for indice_filtro in range(pesos_filtros.shape[3]):
                valor_gradiente += pesos_filtros[
                    fila_filtro, columna_filtro, indice_canal, indice_filtro
                ] * gradiente_salida[
                    indice_imagen, fila_salida, columna_salida, indice_filtro
                ]

    gradiente_entrada[
        indice_imagen, fila_entrada, columna_entrada, indice_canal
    ] = valor_gradiente


@cuda.jit
def convolucion_gradiente_pesos_kernel(
    gradiente_salida, imagenes_entrada, gradiente_pesos
):
    indice_peso = cuda.grid(1)
    if indice_peso >= gradiente_pesos.size:
        return

    cantidad_filtros = gradiente_pesos.shape[3]
    cantidad_canales = gradiente_pesos.shape[2]
    indice_filtro = indice_peso % cantidad_filtros
    indice_posicion = indice_peso // cantidad_filtros
    indice_canal = indice_posicion % cantidad_canales
    indice_posicion //= cantidad_canales
    columna_filtro = indice_posicion % 3
    fila_filtro = indice_posicion // 3

    valor_gradiente = float32(0.0)
    for indice_imagen in range(imagenes_entrada.shape[0]):
        for fila_salida in range(gradiente_salida.shape[1]):
            fila_entrada = fila_salida + fila_filtro - 1
            if fila_entrada < 0 or fila_entrada >= imagenes_entrada.shape[1]:
                continue

            for columna_salida in range(gradiente_salida.shape[2]):
                columna_entrada = columna_salida + columna_filtro - 1
                if columna_entrada < 0 or columna_entrada >= imagenes_entrada.shape[2]:
                    continue

                valor_gradiente += imagenes_entrada[
                    indice_imagen, fila_entrada, columna_entrada, indice_canal
                ] * gradiente_salida[
                    indice_imagen, fila_salida, columna_salida, indice_filtro
                ]

    gradiente_pesos[
        fila_filtro, columna_filtro, indice_canal, indice_filtro
    ] = valor_gradiente


@cuda.jit
def convolucion_gradiente_bias_kernel(gradiente_salida, gradiente_bias):
    indice_filtro = cuda.grid(1)
    if indice_filtro >= gradiente_bias.size:
        return

    valor_gradiente = float32(0.0)
    for indice_imagen in range(gradiente_salida.shape[0]):
        for fila_salida in range(gradiente_salida.shape[1]):
            for columna_salida in range(gradiente_salida.shape[2]):
                valor_gradiente += gradiente_salida[
                    indice_imagen, fila_salida, columna_salida, indice_filtro
                ]

    gradiente_bias[indice_filtro] = valor_gradiente


def convolucion_backward(
    gradiente_salida_device,
    imagenes_entrada_device,
    pesos_filtros_device,
    gradiente_entrada_device,
    gradiente_pesos_device,
    gradiente_bias_device,
):
    """Escribe los tres gradientes en buffers device, sin modificar imagenes ni pesos."""
    cantidad_imagenes, alto_imagen, ancho_imagen, canales_entrada = (
        imagenes_entrada_device.shape
    )
    alto_filtro, ancho_filtro, canales_filtro, cantidad_filtros = pesos_filtros_device.shape

    if alto_filtro != 3 or ancho_filtro != 3:
        raise ValueError("Los filtros deben tener dimensiones espaciales 3x3.")
    if canales_filtro != canales_entrada:
        raise ValueError("Los filtros deben abarcar todos los canales de entrada.")
    if gradiente_salida_device.shape != (
        cantidad_imagenes, alto_imagen, ancho_imagen, cantidad_filtros
    ):
        raise ValueError("El gradiente no coincide con la salida de la convolucion.")
    if gradiente_entrada_device.shape != imagenes_entrada_device.shape:
        raise ValueError("El gradiente de entrada debe tener la forma de las imagenes.")
    if gradiente_pesos_device.shape != pesos_filtros_device.shape:
        raise ValueError("El gradiente de pesos debe tener la forma de los filtros.")
    if gradiente_bias_device.shape != (cantidad_filtros,):
        raise ValueError("Debe existir un gradiente de bias por filtro.")

    hilos_por_bloque = 128
    if gradiente_entrada_device.size:
        bloques_entrada = (
            gradiente_entrada_device.size + hilos_por_bloque - 1
        ) // hilos_por_bloque
        convolucion_gradiente_entrada_kernel[bloques_entrada, hilos_por_bloque](
            gradiente_salida_device, pesos_filtros_device, gradiente_entrada_device
        )

    if gradiente_pesos_device.size:
        bloques_pesos = (
            gradiente_pesos_device.size + hilos_por_bloque - 1
        ) // hilos_por_bloque
        convolucion_gradiente_pesos_kernel[bloques_pesos, hilos_por_bloque](
            gradiente_salida_device, imagenes_entrada_device, gradiente_pesos_device
        )

    if gradiente_bias_device.size:
        bloques_bias = (
            gradiente_bias_device.size + hilos_por_bloque - 1
        ) // hilos_por_bloque
        convolucion_gradiente_bias_kernel[bloques_bias, hilos_por_bloque](
            gradiente_salida_device, gradiente_bias_device
        )

    return gradiente_entrada_device, gradiente_pesos_device, gradiente_bias_device
