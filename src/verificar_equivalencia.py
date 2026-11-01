"""Comprobacion de integracion CPU/GPU con un mismo lote y pesos identicos."""

import argparse
from pathlib import Path

import numpy as np
from numba import cuda

import config
from compartido.parametros import inicializar_parametros
from compartido.persistencia import cargar_parametros
from cpu.cnn import cnn_backward as cnn_backward_cpu, cnn_forward as cnn_forward_cpu
from cpu.operaciones.activaciones import relu_backward as relu_backward_cpu
from cpu.operaciones.convolucion import convolucion_backward as convolucion_backward_cpu
from cpu.operaciones.densas import densa_backward as densa_backward_cpu
from cpu.operaciones.pooling import maxpool_backward as maxpool_backward_cpu
from cpu.operaciones.perdida import (
    cross_entropy_forward as perdida_cpu,
    softmax_cross_entropy_backward as gradiente_logits_cpu,
)
from gpu.cnn import cnn_backward as cnn_backward_gpu, cnn_forward as cnn_forward_gpu
from gpu.operaciones.convolucion import convolucion_backward as convolucion_backward_gpu
from gpu.memoria import cargar_lote, preparar_memoria
from gpu.operaciones.perdida import (
    cross_entropy_forward as perdida_gpu,
    softmax_cross_entropy_backward as gradiente_logits_gpu,
)


def comparar(nombre, referencia, resultado, tolerancia_absoluta, tolerancia_relativa):
    """Informa el error maximo y devuelve si todos los elementos estan en tolerancia."""
    referencia = np.asarray(referencia)
    resultado = np.asarray(resultado)
    coincide = (
        referencia.shape == resultado.shape
        and np.isfinite(referencia).all()
        and np.isfinite(resultado).all()
        and np.allclose(
            referencia, resultado,
            atol=tolerancia_absoluta, rtol=tolerancia_relativa, equal_nan=False,
        )
    )
    error_maximo = (
        float(np.max(np.abs(referencia - resultado)))
        if referencia.shape == resultado.shape else float("inf")
    )
    print(
        f"{'OK' if coincide else 'REVISAR'} {nombre}: error absoluto maximo={error_maximo:.8g}",
        flush=True,
    )
    return bool(coincide)


def diagnosticar_pooling(cache_cpu, buffers):
    """Muestra cambios de ruta y los valores de las ventanas que los produjeron."""
    for numero_capa, bloque in enumerate(cache_cpu["bloques_convolucionales"], start=1):
        posiciones_cpu = bloque["posiciones_maximos"]
        posiciones_gpu = buffers[f"conv{numero_capa}_posiciones"].copy_to_host()
        diferencias = np.argwhere(posiciones_cpu != posiciones_gpu)
        salida_cpu = bloque["salida_convolucion"]
        salida_gpu = buffers[f"conv{numero_capa}_salida"].copy_to_host()
        cambios_relu = np.count_nonzero((salida_cpu > 0) != (salida_gpu > 0))
        print(
            f"Rutas conv{numero_capa}: {len(diferencias)}/{posiciones_cpu.size} "
            f"posiciones de pooling distintas; {cambios_relu} cambios de mascara ReLU.",
            flush=True,
        )
        for coordenadas in diferencias[:3]:
            indice_imagen, fila_pooling, columna_pooling, indice_canal = coordenadas
            fila_inicio = int(fila_pooling) * 2
            columna_inicio = int(columna_pooling) * 2
            ventana_cpu = np.maximum(salida_cpu[
                indice_imagen, fila_inicio:fila_inicio + 2,
                columna_inicio:columna_inicio + 2, indice_canal,
            ], 0).ravel()
            ventana_gpu = np.maximum(salida_gpu[
                indice_imagen, fila_inicio:fila_inicio + 2,
                columna_inicio:columna_inicio + 2, indice_canal,
            ], 0).ravel()
            print(f"  Ventana NHWC {coordenadas.tolist()}:", flush=True)
            print("    CPU:", ventana_cpu.tolist(), "posicion:", int(posiciones_cpu[tuple(coordenadas)]))
            print("    GPU:", ventana_gpu.tolist(), "posicion:", int(posiciones_gpu[tuple(coordenadas)]))


def diagnosticar_backward(gradiente_logits, parametros, cache_cpu, memoria, argumentos):
    """Localiza divergencias y aisla cada convolucion con operandos CPU identicos."""
    buffers = memoria["buffers"]
    tolerancias = (argumentos.tolerancia_absoluta, argumentos.tolerancia_relativa)
    comparar(
        "gradiente logits", gradiente_logits,
        buffers["gradiente_logits"].copy_to_host(), *tolerancias,
    )
    gradiente_densa2 = densa_backward_cpu(
        gradiente_logits, cache_cpu["entrada_densa2"], parametros["densa2_pesos"]
    )[0]
    comparar(
        "backward densa2 -> entrada", gradiente_densa2,
        buffers["densa2_gradiente_entrada"].copy_to_host(), *tolerancias,
    )
    gradiente_relu = relu_backward_cpu(gradiente_densa2, cache_cpu["salida_densa1"])
    comparar(
        "backward ReLU densa1", gradiente_relu,
        buffers["densa1_gradiente_relu"].copy_to_host(), *tolerancias,
    )
    gradiente_activaciones = densa_backward_cpu(
        gradiente_relu, cache_cpu["entrada_densa1"], parametros["densa1_pesos"]
    )[0]
    comparar(
        "backward densa1 -> entrada", gradiente_activaciones,
        buffers["densa1_gradiente_entrada"].copy_to_host(), *tolerancias,
    )
    gradiente_activaciones = gradiente_activaciones.reshape(cache_cpu["forma_salida_pooling"])

    for indice_bloque in range(len(cache_cpu["bloques_convolucionales"]) - 1, -1, -1):
        numero_capa = indice_bloque + 1
        nombre_capa = f"conv{numero_capa}"
        bloque = cache_cpu["bloques_convolucionales"][indice_bloque]
        gradiente_pooling = maxpool_backward_cpu(
            gradiente_activaciones, bloque["posiciones_maximos"], bloque["forma_entrada_pooling"]
        )
        comparar(
            f"backward pooling{numero_capa}", gradiente_pooling,
            buffers[f"{nombre_capa}_gradiente_pooling"].copy_to_host(), *tolerancias,
        )
        gradiente_convolucion = relu_backward_cpu(gradiente_pooling, bloque["salida_convolucion"])
        comparar(
            f"backward ReLU {nombre_capa}", gradiente_convolucion,
            buffers[f"{nombre_capa}_gradiente_relu"].copy_to_host(), *tolerancias,
        )
        entrada = bloque["entrada_convolucion"]
        pesos = parametros[f"{nombre_capa}_pesos"]
        gradiente_activaciones, gradiente_pesos, gradiente_bias = convolucion_backward_cpu(
            gradiente_convolucion, entrada, pesos
        )
        comparar(
            f"backward {nombre_capa} -> entrada", gradiente_activaciones,
            buffers[f"{nombre_capa}_gradiente_entrada"].copy_to_host(), *tolerancias,
        )

        # Se usan buffers aparte para conservar intacto el backward original de GPU.
        entrada_device = cuda.to_device(np.ascontiguousarray(entrada))
        pesos_device = cuda.to_device(np.ascontiguousarray(pesos))
        gradiente_salida_device = cuda.to_device(np.ascontiguousarray(gradiente_convolucion))
        gradiente_entrada_device = cuda.device_array_like(entrada_device)
        gradiente_pesos_device = cuda.device_array_like(pesos_device)
        gradiente_bias_device = cuda.device_array(gradiente_bias.shape, dtype=gradiente_bias.dtype)
        convolucion_backward_gpu(
            gradiente_salida_device, entrada_device, pesos_device,
            gradiente_entrada_device, gradiente_pesos_device, gradiente_bias_device,
        )
        for nombre_gradiente, referencia, resultado_device in (
            ("entrada", gradiente_activaciones, gradiente_entrada_device),
            ("pesos", gradiente_pesos, gradiente_pesos_device),
            ("bias", gradiente_bias, gradiente_bias_device),
        ):
            comparar(
                f"AISLADO {nombre_capa} {nombre_gradiente}", referencia,
                resultado_device.copy_to_host(), *tolerancias,
            )


def main():
    argumentos_parser = argparse.ArgumentParser(description=__doc__)
    argumentos_parser.add_argument("--ruta-tensores", type=Path, default=config.RUTA_TENSORES)
    argumentos_parser.add_argument("--ruta-modelo", type=Path)
    argumentos_parser.add_argument("--cantidad-imagenes", type=int, default=2)
    argumentos_parser.add_argument("--solo-forward", action="store_true")
    argumentos_parser.add_argument("--tolerancia-absoluta", type=float, default=1e-5)
    argumentos_parser.add_argument("--tolerancia-relativa", type=float, default=1e-3)
    argumentos = argumentos_parser.parse_args()
    if argumentos.cantidad_imagenes < 1:
        argumentos_parser.error("La cantidad de imagenes debe ser positiva.")
    if argumentos.tolerancia_absoluta < 0 or argumentos.tolerancia_relativa < 0:
        argumentos_parser.error("Las tolerancias no pueden ser negativas.")
    if cuda.config.ENABLE_CUDASIM or not cuda.is_available():
        raise RuntimeError("La comprobacion requiere una GPU CUDA real.")

    with np.load(argumentos.ruta_tensores, allow_pickle=False) as archivo_tensores:
        if not np.array_equal(archivo_tensores["clases"], np.asarray(config.CLASES)):
            raise ValueError("El orden de clases no coincide con config.py.")
        imagenes_validacion = archivo_tensores["x_val"]
        etiquetas_validacion = archivo_tensores["y_val"]
    if argumentos.cantidad_imagenes > len(imagenes_validacion):
        argumentos_parser.error("La cantidad solicitada supera la validacion disponible.")
    generador_aleatorio = np.random.default_rng(config.SEMILLA_ALEATORIA)
    indices = generador_aleatorio.choice(
        len(imagenes_validacion), size=argumentos.cantidad_imagenes, replace=False
    )
    imagenes = np.ascontiguousarray(imagenes_validacion[indices])
    etiquetas = etiquetas_validacion[indices]
    parametros = (
        cargar_parametros(argumentos.ruta_modelo)
        if argumentos.ruta_modelo else inicializar_parametros()
    )
    print("Pesos:", argumentos.ruta_modelo or "inicializacion compartida", flush=True)
    print("Indices de validacion:", indices.tolist(), flush=True)
    print(
        f"Criterio: abs(GPU-CPU) <= {argumentos.tolerancia_absoluta} "
        f"+ {argumentos.tolerancia_relativa} * abs(CPU)",
        flush=True,
    )
    memoria = preparar_memoria(
        parametros, tamano_lote=len(imagenes), entrenamiento=not argumentos.solo_forward
    )
    cargar_lote(memoria, imagenes, etiquetas)
    probabilidades_cpu, cache_cpu = cnn_forward_cpu(imagenes, parametros)
    probabilidades_gpu = cnn_forward_gpu(memoria).copy_to_host()
    buffers = memoria["buffers"]
    comprobaciones = []

    for numero_capa, bloque in enumerate(cache_cpu["bloques_convolucionales"], start=1):
        comprobaciones.append(comparar(
            f"conv{numero_capa}", bloque["salida_convolucion"],
            buffers[f"conv{numero_capa}_salida"].copy_to_host(),
            argumentos.tolerancia_absoluta, argumentos.tolerancia_relativa,
        ))
        if numero_capa < len(config.FILTROS_CONVOLUCION):
            pooling_cpu = cache_cpu["bloques_convolucionales"][numero_capa]["entrada_convolucion"]
        else:
            pooling_cpu = cache_cpu["entrada_densa1"].reshape(cache_cpu["forma_salida_pooling"])
        comprobaciones.append(comparar(
            f"pooling{numero_capa}", pooling_cpu,
            buffers[f"conv{numero_capa}_pooling"].copy_to_host(),
            argumentos.tolerancia_absoluta, argumentos.tolerancia_relativa,
        ))
    for nombre, referencia, resultado in (
        ("densa1", cache_cpu["salida_densa1"], buffers["densa1_salida"].copy_to_host()),
        ("logits", cache_cpu["salida_densa2"], buffers["densa2_salida"].copy_to_host()),
        ("probabilidades", probabilidades_cpu, probabilidades_gpu),
    ):
        comprobaciones.append(comparar(
            nombre, referencia, resultado,
            argumentos.tolerancia_absoluta, argumentos.tolerancia_relativa,
        ))

    perdida_referencia = perdida_cpu(cache_cpu["salida_densa2"], etiquetas)
    perdida_gpu(
        buffers["densa2_salida"], buffers["etiquetas"],
        buffers["perdidas_individuales"], buffers["perdida_promedio"],
    )
    comprobaciones.append(comparar(
        "perdida", perdida_referencia, buffers["perdida_promedio"].copy_to_host()[0],
        argumentos.tolerancia_absoluta, argumentos.tolerancia_relativa,
    ))
    clases_cpu = probabilidades_cpu.argmax(axis=1)
    clases_gpu = probabilidades_gpu.argmax(axis=1)
    print("Clases CPU:", clases_cpu.tolist(), flush=True)
    print("Clases GPU:", clases_gpu.tolist(), flush=True)
    print("Coincidencias de etiqueta:", int(np.sum(clases_cpu == clases_gpu)), "/", len(imagenes))

    if not argumentos.solo_forward:
        gradiente_cpu = gradiente_logits_cpu(probabilidades_cpu, etiquetas)
        gradiente_logits_gpu(
            buffers["probabilidades"], buffers["etiquetas"], buffers["gradiente_logits"]
        )
        gradientes_cpu = cnn_backward_cpu(gradiente_cpu, parametros, cache_cpu)
        gradientes_gpu = cnn_backward_gpu(buffers["gradiente_logits"], memoria)
        for nombre_parametro, referencia in gradientes_cpu.items():
            comprobaciones.append(comparar(
                f"gradiente {nombre_parametro}", referencia,
                gradientes_gpu[nombre_parametro].copy_to_host(),
                argumentos.tolerancia_absoluta, argumentos.tolerancia_relativa,
            ))

    if not all(comprobaciones):
        print("\nDIAGNOSTICO: rutas del forward", flush=True)
        diagnosticar_pooling(cache_cpu, buffers)
        if not argumentos.solo_forward:
            print("\nDIAGNOSTICO: backward completo y convoluciones aisladas", flush=True)
            diagnosticar_backward(gradiente_cpu, parametros, cache_cpu, memoria, argumentos)
        raise AssertionError(
            "Hay diferencias fuera de tolerancia. Revisar las capas indicadas; "
            "no ampliar tolerancias sin analizar la causa."
        )
    print("Comprobacion correcta para este lote; no sustituye las pruebas independientes.")
    print("No se actualizaron ni guardaron pesos.")


main()
