"""Capas densas CUDA con una salida o un gradiente por hilo."""

from numba import cuda, float32


@cuda.jit
def densa_forward_kernel(activaciones_entrada, pesos_capa, bias_capa, salida):
    indice_salida = cuda.grid(1)
    if indice_salida >= salida.size:
        return

    cantidad_neuronas = salida.shape[1]
    indice_imagen = indice_salida // cantidad_neuronas
    indice_neurona = indice_salida % cantidad_neuronas
    valor_salida = bias_capa[indice_neurona]

    for indice_entrada in range(activaciones_entrada.shape[1]):
        valor_salida += (
            activaciones_entrada[indice_imagen, indice_entrada]
            * pesos_capa[indice_entrada, indice_neurona]
        )

    salida[indice_imagen, indice_neurona] = valor_salida


@cuda.jit
def densa_gradiente_entrada_kernel(gradiente_salida, pesos_capa, gradiente_entrada):
    indice_elemento = cuda.grid(1)
    if indice_elemento >= gradiente_entrada.size:
        return

    cantidad_entradas = gradiente_entrada.shape[1]
    indice_imagen = indice_elemento // cantidad_entradas
    indice_entrada = indice_elemento % cantidad_entradas
    valor_gradiente = float32(0.0)

    for indice_neurona in range(pesos_capa.shape[1]):
        valor_gradiente += (
            pesos_capa[indice_entrada, indice_neurona]
            * gradiente_salida[indice_imagen, indice_neurona]
        )

    gradiente_entrada[indice_imagen, indice_entrada] = valor_gradiente


@cuda.jit
def densa_gradiente_pesos_kernel(
    gradiente_salida, activaciones_entrada, gradiente_pesos
):
    indice_peso = cuda.grid(1)
    if indice_peso >= gradiente_pesos.size:
        return

    cantidad_neuronas = gradiente_pesos.shape[1]
    indice_entrada = indice_peso // cantidad_neuronas
    indice_neurona = indice_peso % cantidad_neuronas
    valor_gradiente = float32(0.0)

    for indice_imagen in range(activaciones_entrada.shape[0]):
        valor_gradiente += (
            activaciones_entrada[indice_imagen, indice_entrada]
            * gradiente_salida[indice_imagen, indice_neurona]
        )

    gradiente_pesos[indice_entrada, indice_neurona] = valor_gradiente


@cuda.jit
def densa_gradiente_bias_kernel(gradiente_salida, gradiente_bias):
    indice_neurona = cuda.grid(1)
    if indice_neurona >= gradiente_bias.size:
        return

    valor_gradiente = float32(0.0)
    for indice_imagen in range(gradiente_salida.shape[0]):
        valor_gradiente += gradiente_salida[indice_imagen, indice_neurona]

    gradiente_bias[indice_neurona] = valor_gradiente


def densa_forward(
    activaciones_entrada_device, pesos_capa_device, bias_capa_device, salida_device
):
    """Lanza forward sobre buffers device sin transferencias ni sincronizacion."""
    cantidad_imagenes, cantidad_entradas = activaciones_entrada_device.shape
    cantidad_entradas_pesos, cantidad_neuronas = pesos_capa_device.shape

    if cantidad_entradas_pesos != cantidad_entradas:
        raise ValueError("Los pesos deben tener una fila por entrada de la capa.")
    if bias_capa_device.shape != (cantidad_neuronas,):
        raise ValueError("Debe existir un bias por neurona.")
    if salida_device.shape != (cantidad_imagenes, cantidad_neuronas):
        raise ValueError("El buffer de salida no tiene la forma de la capa densa.")
    if salida_device.size == 0:
        return salida_device

    hilos_por_bloque = 128
    cantidad_bloques = (salida_device.size + hilos_por_bloque - 1) // hilos_por_bloque
    densa_forward_kernel[cantidad_bloques, hilos_por_bloque](
        activaciones_entrada_device, pesos_capa_device, bias_capa_device, salida_device
    )
    return salida_device


def densa_backward(
    gradiente_salida_device,
    activaciones_entrada_device,
    pesos_capa_device,
    gradiente_entrada_device,
    gradiente_pesos_device,
    gradiente_bias_device,
):
    """Escribe los tres gradientes sin actualizar los parametros de la capa."""
    cantidad_imagenes, cantidad_entradas = activaciones_entrada_device.shape
    cantidad_entradas_pesos, cantidad_neuronas = pesos_capa_device.shape

    if cantidad_entradas_pesos != cantidad_entradas:
        raise ValueError("Los pesos deben tener una fila por entrada de la capa.")
    if gradiente_salida_device.shape != (cantidad_imagenes, cantidad_neuronas):
        raise ValueError("El gradiente no coincide con la salida de la capa densa.")
    if gradiente_entrada_device.shape != activaciones_entrada_device.shape:
        raise ValueError("El gradiente de entrada debe tener la forma de las activaciones.")
    if gradiente_pesos_device.shape != pesos_capa_device.shape:
        raise ValueError("El gradiente de pesos debe tener la forma de los pesos de la capa.")
    if gradiente_bias_device.shape != (cantidad_neuronas,):
        raise ValueError("Debe existir un gradiente de bias por neurona.")

    hilos_por_bloque = 128
    if gradiente_entrada_device.size:
        bloques_entrada = (
            gradiente_entrada_device.size + hilos_por_bloque - 1
        ) // hilos_por_bloque
        densa_gradiente_entrada_kernel[bloques_entrada, hilos_por_bloque](
            gradiente_salida_device, pesos_capa_device, gradiente_entrada_device
        )

    if gradiente_pesos_device.size:
        bloques_pesos = (
            gradiente_pesos_device.size + hilos_por_bloque - 1
        ) // hilos_por_bloque
        densa_gradiente_pesos_kernel[bloques_pesos, hilos_por_bloque](
            gradiente_salida_device, activaciones_entrada_device, gradiente_pesos_device
        )

    if gradiente_bias_device.size:
        bloques_bias = (
            gradiente_bias_device.size + hilos_por_bloque - 1
        ) // hilos_por_bloque
        densa_gradiente_bias_kernel[bloques_bias, hilos_por_bloque](
            gradiente_salida_device, gradiente_bias_device
        )

    return gradiente_entrada_device, gradiente_pesos_device, gradiente_bias_device
