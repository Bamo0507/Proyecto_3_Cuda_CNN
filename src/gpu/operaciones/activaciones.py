"""ReLU por elemento y softmax por imagen, usando buffers device."""

import math

from numba import cuda, float32


@cuda.jit
def relu_forward_kernel(valores_entrada, salida):
    indice_elemento = cuda.grid(1)
    if indice_elemento >= salida.size:
        return

    valor_entrada = valores_entrada[indice_elemento]
    if valor_entrada < 0:
        salida[indice_elemento] = 0.0
    else:
        salida[indice_elemento] = valor_entrada


@cuda.jit
def relu_backward_kernel(gradiente_salida, valores_entrada, gradiente_entrada):
    indice_elemento = cuda.grid(1)
    if indice_elemento >= gradiente_entrada.size:
        return

    if valores_entrada[indice_elemento] > 0:
        gradiente_entrada[indice_elemento] = gradiente_salida[indice_elemento]
    else:
        gradiente_entrada[indice_elemento] = 0.0


@cuda.jit
def softmax_forward_kernel(valores_salida, probabilidades):
    indice_imagen = cuda.grid(1)
    if indice_imagen >= valores_salida.shape[0]:
        return

    cantidad_clases = valores_salida.shape[1]
    valor_maximo = valores_salida[indice_imagen, 0]
    for indice_clase in range(1, cantidad_clases):
        valor_actual = valores_salida[indice_imagen, indice_clase]
        if valor_actual > valor_maximo:
            valor_maximo = valor_actual

    suma_exponenciales = float32(0.0)
    for indice_clase in range(cantidad_clases):
        valor_exponencial = float32(math.exp(
            valores_salida[indice_imagen, indice_clase] - valor_maximo
        ))
        probabilidades[indice_imagen, indice_clase] = valor_exponencial
        suma_exponenciales += valor_exponencial

    for indice_clase in range(cantidad_clases):
        probabilidades[indice_imagen, indice_clase] /= suma_exponenciales


def relu_forward(valores_entrada_device, salida_device):
    """Lanza ReLU sin transferir datos ni sincronizar."""
    if valores_entrada_device.shape != salida_device.shape:
        raise ValueError("La salida de ReLU debe tener la misma forma que la entrada.")
    if salida_device.size == 0:
        return salida_device

    hilos_por_bloque = 128
    cantidad_bloques = (salida_device.size + hilos_por_bloque - 1) // hilos_por_bloque
    relu_forward_kernel[cantidad_bloques, hilos_por_bloque](
        valores_entrada_device.ravel(), salida_device.ravel()
    )
    return salida_device


def relu_backward(
    gradiente_salida_device, valores_entrada_device, gradiente_entrada_device
):
    """Lanza el backward usando los valores anteriores a aplicar ReLU."""
    if (
        gradiente_salida_device.shape != valores_entrada_device.shape
        or gradiente_entrada_device.shape != valores_entrada_device.shape
    ):
        raise ValueError("Los gradientes y la entrada de ReLU deben tener la misma forma.")
    if gradiente_entrada_device.size == 0:
        return gradiente_entrada_device

    hilos_por_bloque = 128
    cantidad_bloques = (
        gradiente_entrada_device.size + hilos_por_bloque - 1
    ) // hilos_por_bloque
    relu_backward_kernel[cantidad_bloques, hilos_por_bloque](
        gradiente_salida_device.ravel(),
        valores_entrada_device.ravel(),
        gradiente_entrada_device.ravel(),
    )
    return gradiente_entrada_device


def softmax_forward(valores_salida_device, probabilidades_device):
    """Un hilo procesa todas las clases de una imagen, sin transferencias."""
    cantidad_imagenes, cantidad_clases = valores_salida_device.shape
    if cantidad_imagenes == 0 or cantidad_clases == 0:
        raise ValueError("Softmax necesita al menos una imagen y una clase.")
    if probabilidades_device.shape != valores_salida_device.shape:
        raise ValueError("Las probabilidades deben tener la misma forma que los logits.")

    hilos_por_bloque = 128
    cantidad_bloques = (cantidad_imagenes + hilos_por_bloque - 1) // hilos_por_bloque
    softmax_forward_kernel[cantidad_bloques, hilos_por_bloque](
        valores_salida_device, probabilidades_device
    )
    return probabilidades_device
