"""Inicializacion comun de pesos y biases para CPU y GPU."""

import numpy as np

import config


def inicializar_parametros(semilla=config.SEMILLA_ALEATORIA):
    """Devuelve pesos Glorot uniformes y biases cero para las cinco capas."""
    generador_aleatorio = np.random.default_rng(semilla)
    alto_activaciones, ancho_activaciones, canales_entrada = config.FORMA_ENTRADA
    alto_filtro, ancho_filtro = config.TAMANO_FILTRO
    formas_pesos = {}

    for numero_capa, cantidad_filtros in enumerate(config.FILTROS_CONVOLUCION, start=1):
        formas_pesos[f"conv{numero_capa}"] = (
            alto_filtro, ancho_filtro, canales_entrada, cantidad_filtros
        )
        alto_activaciones = (
            alto_activaciones - config.TAMANO_POOLING[0]
        ) // config.PASO_POOLING + 1
        ancho_activaciones = (
            ancho_activaciones - config.TAMANO_POOLING[1]
        ) // config.PASO_POOLING + 1
        canales_entrada = cantidad_filtros

    cantidad_entradas_densas = alto_activaciones * ancho_activaciones * canales_entrada
    formas_pesos["densa1"] = (cantidad_entradas_densas, config.NEURONAS_CAPA_DENSA)
    formas_pesos["densa2"] = (config.NEURONAS_CAPA_DENSA, config.CANTIDAD_CLASES)

    parametros = {}
    for nombre_capa, forma_pesos in formas_pesos.items():
        cantidad_entradas = forma_pesos[-2]
        cantidad_salidas = forma_pesos[-1]
        if len(forma_pesos) == 4:
            area_filtro = forma_pesos[0] * forma_pesos[1]
            cantidad_entradas *= area_filtro
            cantidad_salidas *= area_filtro

        limite_glorot = np.sqrt(6.0 / (cantidad_entradas + cantidad_salidas))
        parametros[f"{nombre_capa}_pesos"] = generador_aleatorio.uniform(
            -limite_glorot, limite_glorot, size=forma_pesos
        ).astype(config.TIPO_DATOS)
        parametros[f"{nombre_capa}_bias"] = np.zeros(
            forma_pesos[-1], dtype=config.TIPO_DATOS
        )

    return parametros
