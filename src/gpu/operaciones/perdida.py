"""Cross-entropy CUDA estable desde logits y su gradiente combinado con softmax."""

import math

from numba import cuda, float32


@cuda.jit
def cross_entropy_forward_kernel(valores_salida, etiquetas, perdidas_individuales):
    indice_imagen = cuda.grid(1)
    if indice_imagen >= valores_salida.shape[0]:
        return

    cantidad_clases = valores_salida.shape[1]
    clase_correcta = etiquetas[indice_imagen]
    if clase_correcta < 0 or clase_correcta >= cantidad_clases:
        perdidas_individuales[indice_imagen] = float32(math.nan)
        return

    valor_maximo = valores_salida[indice_imagen, 0]
    for indice_clase in range(1, cantidad_clases):
        valor_actual = valores_salida[indice_imagen, indice_clase]
        if valor_actual > valor_maximo:
            valor_maximo = valor_actual

    suma_exponenciales = float32(0.0)
    for indice_clase in range(cantidad_clases):
        suma_exponenciales += float32(math.exp(
            valores_salida[indice_imagen, indice_clase] - valor_maximo
        ))

    perdidas_individuales[indice_imagen] = float32(math.log(suma_exponenciales)) - (
        valores_salida[indice_imagen, clase_correcta] - valor_maximo
    )


@cuda.jit
def promediar_perdidas_kernel(perdidas_individuales, perdida_promedio):
    """Un solo hilo acumula el promedio del lote sin llevar las perdidas a CPU."""
    if cuda.grid(1) != 0:
        return

    perdida_total = float32(0.0)
    for indice_imagen in range(perdidas_individuales.size):
        perdida_total += perdidas_individuales[indice_imagen]
    perdida_promedio[0] = perdida_total / float32(perdidas_individuales.size)


@cuda.jit
def softmax_cross_entropy_backward_kernel(probabilidades, etiquetas, gradiente_salida):
    indice_elemento = cuda.grid(1)
    if indice_elemento >= gradiente_salida.size:
        return

    cantidad_imagenes = probabilidades.shape[0]
    cantidad_clases = probabilidades.shape[1]
    indice_imagen = indice_elemento // cantidad_clases
    indice_clase = indice_elemento % cantidad_clases
    clase_correcta = etiquetas[indice_imagen]
    if clase_correcta < 0 or clase_correcta >= cantidad_clases:
        gradiente_salida[indice_imagen, indice_clase] = float32(math.nan)
        return

    valor_gradiente = probabilidades[indice_imagen, indice_clase]
    if indice_clase == clase_correcta:
        valor_gradiente -= float32(1.0)
    gradiente_salida[indice_imagen, indice_clase] = (
        valor_gradiente / float32(cantidad_imagenes)
    )


def cross_entropy_forward(
    valores_salida_device,
    etiquetas_device,
    perdidas_individuales_device,
    perdida_promedio_device,
):
    """Calcula perdidas por imagen y promedio en GPU, sin transferir ni sincronizar."""
    cantidad_imagenes, cantidad_clases = valores_salida_device.shape
    if cantidad_imagenes == 0 or cantidad_clases == 0:
        raise ValueError("La perdida necesita al menos una imagen y una clase.")
    if etiquetas_device.shape != (cantidad_imagenes,):
        raise ValueError("Debe existir una etiqueta por imagen.")
    if perdidas_individuales_device.shape != (cantidad_imagenes,):
        raise ValueError("Debe existir una perdida por imagen.")
    if perdida_promedio_device.shape != (1,):
        raise ValueError("El buffer de perdida promedio debe tener forma (1,).")

    hilos_por_bloque = 128
    cantidad_bloques = (cantidad_imagenes + hilos_por_bloque - 1) // hilos_por_bloque
    cross_entropy_forward_kernel[cantidad_bloques, hilos_por_bloque](
        valores_salida_device, etiquetas_device, perdidas_individuales_device
    )
    promediar_perdidas_kernel[1, 1](perdidas_individuales_device, perdida_promedio_device)
    return perdida_promedio_device


def softmax_cross_entropy_backward(
    probabilidades_device, etiquetas_device, gradiente_salida_device
):
    """Un hilo calcula un gradiente respecto a logits, normalizado una vez por lote."""
    cantidad_imagenes, cantidad_clases = probabilidades_device.shape
    if cantidad_imagenes == 0 or cantidad_clases == 0:
        raise ValueError("El gradiente necesita al menos una imagen y una clase.")
    if etiquetas_device.shape != (cantidad_imagenes,):
        raise ValueError("Debe existir una etiqueta por imagen.")
    if gradiente_salida_device.shape != probabilidades_device.shape:
        raise ValueError("El gradiente debe tener la misma forma que las probabilidades.")

    hilos_por_bloque = 128
    cantidad_bloques = (
        gradiente_salida_device.size + hilos_por_bloque - 1
    ) // hilos_por_bloque
    softmax_cross_entropy_backward_kernel[cantidad_bloques, hilos_por_bloque](
        probabilidades_device, etiquetas_device, gradiente_salida_device
    )
    return gradiente_salida_device
