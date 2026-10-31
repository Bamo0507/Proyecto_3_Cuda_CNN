"""Verifica un kernel propio de suma de vectores en una GPU NVIDIA real."""

import numpy as np
from numba import cuda


@cuda.jit
def sumar_vectores_cuda(primer_vector, segundo_vector, vector_resultado):
    indice_elemento = cuda.grid(1)
    if indice_elemento < vector_resultado.size:
        vector_resultado[indice_elemento] = (
            primer_vector[indice_elemento] + segundo_vector[indice_elemento]
        )


def main():
    if cuda.config.ENABLE_CUDASIM:
        raise RuntimeError("Esta verificacion requiere una GPU real, no el simulador.")
    if not cuda.is_available():
        raise RuntimeError("CUDA no esta disponible. Ejecuta este archivo en Colab con GPU.")

    cantidad_elementos = 1000
    primer_vector = np.arange(cantidad_elementos, dtype=np.float32)
    segundo_vector = -2 * primer_vector
    resultado_esperado = primer_vector + segundo_vector

    primer_vector_device = cuda.to_device(primer_vector)
    segundo_vector_device = cuda.to_device(segundo_vector)
    resultado_device = cuda.device_array_like(primer_vector)

    hilos_por_bloque = 128
    cantidad_bloques = (cantidad_elementos + hilos_por_bloque - 1) // hilos_por_bloque
    sumar_vectores_cuda[cantidad_bloques, hilos_por_bloque](
        primer_vector_device, segundo_vector_device, resultado_device
    )
    cuda.synchronize()
    resultado_obtenido = resultado_device.copy_to_host()

    np.testing.assert_allclose(
        resultado_obtenido, resultado_esperado, rtol=1e-6, atol=1e-6
    )
    print(f"Elementos verificados: {cantidad_elementos}")
    print(f"Bloques: {cantidad_bloques}; hilos por bloque: {hilos_por_bloque}")
    print(f"Primeros resultados: {resultado_obtenido[:5]}")
    print("Verificacion correcta: el kernel CUDA coincide con NumPy.")


main()
