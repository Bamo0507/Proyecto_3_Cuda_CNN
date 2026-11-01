"""MaxPooling CUDA 2x2 con paso 2, sin padding ni escrituras compartidas."""

from numba import cuda


@cuda.jit
def maxpool_forward_kernel(activaciones_entrada, valores_maximos, posiciones_maximos):
    indice_salida = cuda.grid(1)
    if indice_salida >= valores_maximos.size:
        return

    cantidad_canales = valores_maximos.shape[3]
    ancho_salida = valores_maximos.shape[2]
    alto_salida = valores_maximos.shape[1]
    indice_canal = indice_salida % cantidad_canales
    indice_posicion = indice_salida // cantidad_canales
    columna_salida = indice_posicion % ancho_salida
    indice_posicion //= ancho_salida
    fila_salida = indice_posicion % alto_salida
    indice_imagen = indice_posicion // alto_salida

    fila_inicio = fila_salida * 2
    columna_inicio = columna_salida * 2
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
                posicion_maximo = desplazamiento_fila * 2 + desplazamiento_columna

    valores_maximos[
        indice_imagen, fila_salida, columna_salida, indice_canal
    ] = valor_maximo
    posiciones_maximos[
        indice_imagen, fila_salida, columna_salida, indice_canal
    ] = posicion_maximo


@cuda.jit
def maxpool_backward_kernel(gradiente_salida, posiciones_maximos, gradiente_entrada):
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

    fila_salida = fila_entrada // 2
    columna_salida = columna_entrada // 2
    valor_gradiente = 0.0

    # Si una dimension es impar, el borde sin ventana completa recibe gradiente cero.
    if fila_salida < gradiente_salida.shape[1] and columna_salida < gradiente_salida.shape[2]:
        posicion_local = (fila_entrada % 2) * 2 + columna_entrada % 2
        if posicion_local == posiciones_maximos[
            indice_imagen, fila_salida, columna_salida, indice_canal
        ]:
            valor_gradiente = gradiente_salida[
                indice_imagen, fila_salida, columna_salida, indice_canal
            ]

    gradiente_entrada[
        indice_imagen, fila_entrada, columna_entrada, indice_canal
    ] = valor_gradiente


def maxpool_forward(
    activaciones_entrada_device, valores_maximos_device, posiciones_maximos_device
):
    """Lanza forward sobre buffers device; conserva el primer maximo en empates."""
    cantidad_imagenes, alto_entrada, ancho_entrada, cantidad_canales = (
        activaciones_entrada_device.shape
    )
    if alto_entrada < 2 or ancho_entrada < 2:
        raise ValueError("La entrada debe permitir al menos una ventana 2x2.")
    forma_salida = (
        cantidad_imagenes, alto_entrada // 2, ancho_entrada // 2, cantidad_canales
    )
    if valores_maximos_device.shape != forma_salida:
        raise ValueError("El buffer de maximos no tiene la forma de salida del pooling.")
    if posiciones_maximos_device.shape != forma_salida:
        raise ValueError("Las posiciones no coinciden con la salida del pooling.")
    if valores_maximos_device.size == 0:
        return valores_maximos_device, posiciones_maximos_device

    hilos_por_bloque = 128
    cantidad_bloques = (
        valores_maximos_device.size + hilos_por_bloque - 1
    ) // hilos_por_bloque
    maxpool_forward_kernel[cantidad_bloques, hilos_por_bloque](
        activaciones_entrada_device, valores_maximos_device, posiciones_maximos_device
    )
    return valores_maximos_device, posiciones_maximos_device


def maxpool_backward(
    gradiente_salida_device, posiciones_maximos_device, gradiente_entrada_device
):
    """Cada hilo escribe una posicion de entrada, incluido cero donde no gano el maximo."""
    cantidad_imagenes, alto_entrada, ancho_entrada, cantidad_canales = (
        gradiente_entrada_device.shape
    )
    forma_salida = (
        cantidad_imagenes, alto_entrada // 2, ancho_entrada // 2, cantidad_canales
    )
    if gradiente_salida_device.shape != forma_salida:
        raise ValueError("El gradiente no coincide con la salida del pooling.")
    if posiciones_maximos_device.shape != forma_salida:
        raise ValueError("Las posiciones no coinciden con la salida del pooling.")
    if gradiente_entrada_device.size == 0:
        return gradiente_entrada_device

    hilos_por_bloque = 128
    cantidad_bloques = (
        gradiente_entrada_device.size + hilos_por_bloque - 1
    ) // hilos_por_bloque
    maxpool_backward_kernel[cantidad_bloques, hilos_por_bloque](
        gradiente_salida_device, posiciones_maximos_device, gradiente_entrada_device
    )
    return gradiente_entrada_device
