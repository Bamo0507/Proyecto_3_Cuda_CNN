"""Recorridos forward y backward de la CNN con buffers residentes en GPU."""

import config
from gpu.operaciones.activaciones import relu_backward, relu_forward, softmax_forward
from gpu.operaciones.convolucion import convolucion_backward, convolucion_forward
from gpu.operaciones.densas import densa_backward, densa_forward
from gpu.operaciones.pooling import maxpool_backward, maxpool_forward


def cnn_forward(memoria):
    """Lanza las capas y devuelve las probabilidades en device, sin sincronizar."""
    cantidad_imagenes = memoria["cantidad_imagenes"]
    if cantidad_imagenes < 1:
        raise ValueError("Debe cargarse un lote antes de ejecutar forward.")

    buffers = memoria["buffers"]
    parametros = memoria["parametros"]
    activaciones_device = buffers["imagenes"][:cantidad_imagenes]

    for numero_capa in range(1, len(config.FILTROS_CONVOLUCION) + 1):
        nombre_capa = f"conv{numero_capa}"
        salida_convolucion_device = buffers[f"{nombre_capa}_salida"][:cantidad_imagenes]
        salida_relu_device = buffers[f"{nombre_capa}_relu"][:cantidad_imagenes]
        salida_pooling_device = buffers[f"{nombre_capa}_pooling"][:cantidad_imagenes]
        posiciones_maximos_device = buffers[f"{nombre_capa}_posiciones"][:cantidad_imagenes]

        convolucion_forward(
            activaciones_device,
            parametros[f"{nombre_capa}_pesos"],
            parametros[f"{nombre_capa}_bias"],
            salida_convolucion_device,
        )
        relu_forward(salida_convolucion_device, salida_relu_device)
        maxpool_forward(
            salida_relu_device, salida_pooling_device, posiciones_maximos_device
        )
        activaciones_device = salida_pooling_device

    entrada_densa1_device = activaciones_device.reshape(cantidad_imagenes, -1)
    salida_densa1_device = buffers["densa1_salida"][:cantidad_imagenes]
    entrada_densa2_device = buffers["densa1_relu"][:cantidad_imagenes]
    salida_densa2_device = buffers["densa2_salida"][:cantidad_imagenes]
    probabilidades_device = buffers["probabilidades"][:cantidad_imagenes]

    densa_forward(
        entrada_densa1_device,
        parametros["densa1_pesos"], parametros["densa1_bias"],
        salida_densa1_device,
    )
    relu_forward(salida_densa1_device, entrada_densa2_device)
    densa_forward(
        entrada_densa2_device,
        parametros["densa2_pesos"], parametros["densa2_bias"],
        salida_densa2_device,
    )
    softmax_forward(salida_densa2_device, probabilidades_device)
    memoria["forward_disponible"] = True
    return probabilidades_device


def cnn_backward(gradiente_salida_device, memoria):
    """Escribe todos los gradientes en device, sin actualizar pesos ni copiar a RAM."""
    if not memoria["entrenamiento"]:
        raise ValueError("Backward requiere memoria preparada para entrenamiento.")
    if not memoria["forward_disponible"]:
        raise ValueError("Debe ejecutarse forward sobre el lote actual antes de backward.")

    cantidad_imagenes = memoria["cantidad_imagenes"]
    if gradiente_salida_device.shape != (cantidad_imagenes, config.CANTIDAD_CLASES):
        raise ValueError("El gradiente debe tener la forma de los logits del lote actual.")

    buffers = memoria["buffers"]
    parametros = memoria["parametros"]
    gradientes = memoria["gradientes"]
    gradiente_densa2_device = buffers["densa2_gradiente_entrada"][:cantidad_imagenes]
    gradiente_relu_densa1_device = buffers["densa1_gradiente_relu"][:cantidad_imagenes]
    gradiente_densa1_device = buffers["densa1_gradiente_entrada"][:cantidad_imagenes]

    densa_backward(
        gradiente_salida_device,
        buffers["densa1_relu"][:cantidad_imagenes],
        parametros["densa2_pesos"],
        gradiente_densa2_device,
        gradientes["densa2_pesos"], gradientes["densa2_bias"],
    )
    relu_backward(
        gradiente_densa2_device,
        buffers["densa1_salida"][:cantidad_imagenes],
        gradiente_relu_densa1_device,
    )

    nombre_ultimo_pooling = f"conv{len(config.FILTROS_CONVOLUCION)}_pooling"
    salida_ultimo_pooling_device = buffers[nombre_ultimo_pooling][:cantidad_imagenes]
    densa_backward(
        gradiente_relu_densa1_device,
        salida_ultimo_pooling_device.reshape(cantidad_imagenes, -1),
        parametros["densa1_pesos"],
        gradiente_densa1_device,
        gradientes["densa1_pesos"], gradientes["densa1_bias"],
    )
    gradiente_activaciones_device = gradiente_densa1_device.reshape(
        salida_ultimo_pooling_device.shape
    )

    for numero_capa in range(len(config.FILTROS_CONVOLUCION), 0, -1):
        nombre_capa = f"conv{numero_capa}"
        gradiente_pooling_device = buffers[f"{nombre_capa}_gradiente_pooling"][:cantidad_imagenes]
        gradiente_relu_device = buffers[f"{nombre_capa}_gradiente_relu"][:cantidad_imagenes]
        gradiente_entrada_device = buffers[f"{nombre_capa}_gradiente_entrada"][:cantidad_imagenes]

        maxpool_backward(
            gradiente_activaciones_device,
            buffers[f"{nombre_capa}_posiciones"][:cantidad_imagenes],
            gradiente_pooling_device,
        )
        relu_backward(
            gradiente_pooling_device,
            buffers[f"{nombre_capa}_salida"][:cantidad_imagenes],
            gradiente_relu_device,
        )
        if numero_capa == 1:
            entrada_convolucion_device = buffers["imagenes"][:cantidad_imagenes]
        else:
            nombre_pooling_anterior = f"conv{numero_capa - 1}_pooling"
            entrada_convolucion_device = buffers[nombre_pooling_anterior][:cantidad_imagenes]

        convolucion_backward(
            gradiente_relu_device,
            entrada_convolucion_device,
            parametros[f"{nombre_capa}_pesos"],
            gradiente_entrada_device,
            gradientes[f"{nombre_capa}_pesos"], gradientes[f"{nombre_capa}_bias"],
        )
        gradiente_activaciones_device = gradiente_entrada_device

    return gradientes
