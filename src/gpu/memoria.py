"""Reserva de buffers y transferencias entre RAM y memoria GPU para la CNN."""

import numpy as np
from numba import cuda

import config


def preparar_memoria(parametros, tamano_lote=config.TAMANO_LOTE, entrenamiento=True):
    """Copia parametros y reserva buffers reutilizables para un lote maximo."""
    if tamano_lote < 1:
        raise ValueError("El tamano de lote debe ser positivo.")
    if not cuda.is_available():
        raise RuntimeError("CUDA no esta disponible para reservar memoria GPU.")

    for nombre_parametro, valores_parametro in parametros.items():
        if valores_parametro.dtype != np.dtype(config.TIPO_DATOS):
            raise ValueError(f"{nombre_parametro} debe ser {config.TIPO_DATOS}.")

    parametros_device = {
        nombre_parametro: cuda.to_device(np.ascontiguousarray(valores_parametro))
        for nombre_parametro, valores_parametro in parametros.items()
    }
    formas_activaciones = {"imagenes": (tamano_lote, *config.FORMA_ENTRADA)}
    formas_enteros = {}
    alto_entrada, ancho_entrada, canales_entrada = config.FORMA_ENTRADA

    for numero_capa, cantidad_filtros in enumerate(config.FILTROS_CONVOLUCION, start=1):
        nombre_capa = f"conv{numero_capa}"
        forma_entrada = (tamano_lote, alto_entrada, ancho_entrada, canales_entrada)
        forma_convolucion = (tamano_lote, alto_entrada, ancho_entrada, cantidad_filtros)
        forma_pooling = (
            tamano_lote, alto_entrada // 2, ancho_entrada // 2, cantidad_filtros
        )
        formas_activaciones[f"{nombre_capa}_salida"] = forma_convolucion
        formas_activaciones[f"{nombre_capa}_relu"] = forma_convolucion
        formas_activaciones[f"{nombre_capa}_pooling"] = forma_pooling
        formas_enteros[f"{nombre_capa}_posiciones"] = forma_pooling

        if entrenamiento:
            formas_activaciones[f"{nombre_capa}_gradiente_pooling"] = forma_convolucion
            formas_activaciones[f"{nombre_capa}_gradiente_relu"] = forma_convolucion
            formas_activaciones[f"{nombre_capa}_gradiente_entrada"] = forma_entrada

        alto_entrada //= 2
        ancho_entrada //= 2
        canales_entrada = cantidad_filtros

    cantidad_entradas_densas = alto_entrada * ancho_entrada * canales_entrada
    formas_activaciones.update({
        "densa1_salida": (tamano_lote, config.NEURONAS_CAPA_DENSA),
        "densa1_relu": (tamano_lote, config.NEURONAS_CAPA_DENSA),
        "densa2_salida": (tamano_lote, config.CANTIDAD_CLASES),
        "probabilidades": (tamano_lote, config.CANTIDAD_CLASES),
    })

    if entrenamiento:
        formas_enteros["etiquetas"] = (tamano_lote,)
        formas_activaciones.update({
            "gradiente_logits": (tamano_lote, config.CANTIDAD_CLASES),
            "densa2_gradiente_entrada": (tamano_lote, config.NEURONAS_CAPA_DENSA),
            "densa1_gradiente_relu": (tamano_lote, config.NEURONAS_CAPA_DENSA),
            "densa1_gradiente_entrada": (tamano_lote, cantidad_entradas_densas),
            "perdidas_individuales": (tamano_lote,),
            "perdida_promedio": (1,),
        })

    buffers = {
        nombre_buffer: cuda.device_array(forma_buffer, dtype=config.TIPO_DATOS)
        for nombre_buffer, forma_buffer in formas_activaciones.items()
    }
    buffers.update({
        nombre_buffer: cuda.device_array(forma_buffer, dtype=np.int32)
        for nombre_buffer, forma_buffer in formas_enteros.items()
    })

    gradientes_device = {}
    estado_adam = None
    if entrenamiento:
        gradientes_device = {
            nombre_parametro: cuda.device_array_like(valores_parametro)
            for nombre_parametro, valores_parametro in parametros_device.items()
        }
        estado_adam = {
            "numero_paso": 0,
            "primer_momento": {
                nombre_parametro: cuda.to_device(np.zeros_like(valores_parametro))
                for nombre_parametro, valores_parametro in parametros.items()
            },
            "segundo_momento": {
                nombre_parametro: cuda.to_device(np.zeros_like(valores_parametro))
                for nombre_parametro, valores_parametro in parametros.items()
            },
        }

    return {
        "parametros": parametros_device,
        "gradientes": gradientes_device,
        "estado_adam": estado_adam,
        "buffers": buffers,
        "tamano_lote_maximo": tamano_lote,
        "cantidad_imagenes": 0,
        "entrenamiento": entrenamiento,
        "etiquetas_cargadas": False,
        "forward_disponible": False,
    }


def cargar_lote(memoria, imagenes, etiquetas=None):
    """Copia el lote a buffers existentes; acepta un ultimo lote mas pequeno."""
    if imagenes.ndim != 4 or imagenes.shape[1:] != config.FORMA_ENTRADA:
        raise ValueError("Las imagenes deben tener forma (N, 64, 64, 1).")
    if imagenes.dtype != np.dtype(config.TIPO_DATOS):
        raise ValueError(f"Las imagenes deben ser {config.TIPO_DATOS}.")

    cantidad_imagenes = len(imagenes)
    if cantidad_imagenes < 1 or cantidad_imagenes > memoria["tamano_lote_maximo"]:
        raise ValueError("El lote debe tener entre 1 imagen y la capacidad reservada.")

    if etiquetas is not None:
        if not memoria["entrenamiento"]:
            raise ValueError("La memoria de prediccion no incluye etiquetas de entrenamiento.")
        if etiquetas.shape != (cantidad_imagenes,):
            raise ValueError("Debe existir una etiqueta por imagen.")
        if not np.issubdtype(etiquetas.dtype, np.integer):
            raise ValueError("Las etiquetas deben ser enteros.")
        if etiquetas.min() < 0 or etiquetas.max() >= config.CANTIDAD_CLASES:
            raise ValueError("Hay etiquetas fuera del rango de clases.")

    memoria["forward_disponible"] = False
    memoria["buffers"]["imagenes"][:cantidad_imagenes].copy_to_device(
        np.ascontiguousarray(imagenes)
    )
    if etiquetas is not None:
        memoria["buffers"]["etiquetas"][:cantidad_imagenes].copy_to_device(
            np.ascontiguousarray(etiquetas, dtype=np.int32)
        )

    memoria["cantidad_imagenes"] = cantidad_imagenes
    memoria["etiquetas_cargadas"] = etiquetas is not None


def copiar_parametros_a_host(memoria):
    """Recupera los pesos y biases para guardarlos con la persistencia compartida."""
    return {
        nombre_parametro: valores_parametro.copy_to_host()
        for nombre_parametro, valores_parametro in memoria["parametros"].items()
    }
