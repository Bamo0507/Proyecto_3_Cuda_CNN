"""Cross-entropy promedio y gradiente combinado con softmax en CPU secuencial."""

import numpy as np
from numba import njit


@njit
def cross_entropy_forward(valores_salida, etiquetas):
    """Calcula la perdida desde logits para evitar logaritmos de probabilidades cero."""
    cantidad_imagenes, cantidad_clases = valores_salida.shape
    if cantidad_imagenes == 0 or cantidad_clases == 0:
        raise ValueError("La perdida necesita al menos una imagen y una clase.")
    if etiquetas.shape != (cantidad_imagenes,):
        raise ValueError("Debe existir una etiqueta por imagen.")

    perdida_total = np.float32(0.0)

    for indice_imagen in range(cantidad_imagenes):
        clase_correcta = etiquetas[indice_imagen]
        if clase_correcta < 0 or clase_correcta >= cantidad_clases:
            raise ValueError("Hay etiquetas fuera del rango de clases.")

        valor_maximo = np.max(valores_salida[indice_imagen])
        suma_exponenciales = np.float32(0.0)
        for indice_clase in range(cantidad_clases):
            suma_exponenciales += np.exp(
                valores_salida[indice_imagen, indice_clase] - valor_maximo
            )

        perdida_total += np.log(suma_exponenciales) - (
            valores_salida[indice_imagen, clase_correcta] - valor_maximo
        )

    return perdida_total / cantidad_imagenes


@njit
def softmax_cross_entropy_backward(probabilidades, etiquetas):
    """Devuelve el gradiente respecto a logits, promediado una sola vez por batch."""
    cantidad_imagenes, cantidad_clases = probabilidades.shape
    if cantidad_imagenes == 0 or cantidad_clases == 0:
        raise ValueError("El gradiente necesita al menos una imagen y una clase.")
    if etiquetas.shape != (cantidad_imagenes,):
        raise ValueError("Debe existir una etiqueta por imagen.")

    gradiente_salida = probabilidades.copy()

    for indice_imagen in range(cantidad_imagenes):
        clase_correcta = etiquetas[indice_imagen]
        if clase_correcta < 0 or clase_correcta >= cantidad_clases:
            raise ValueError("Hay etiquetas fuera del rango de clases.")
        gradiente_salida[indice_imagen, clase_correcta] -= np.float32(1.0)

    gradiente_salida /= np.float32(cantidad_imagenes)
    return gradiente_salida
