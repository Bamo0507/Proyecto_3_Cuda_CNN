# Análisis CPU vs GPU — corridas `dani`

Tablas completas en `resumen.md`, datos por corrida en `entrenamiento_individual.csv`
y `prediccion_individual_{cpu,gpu}.csv`, gráficas en `graficas/`.

## Entorno (para reproducir)

| | |
|---|---|
| Commit | `54ccb42` + scripts nuevos sin commit (`src/medir_prediccion.py`, `experimentos/`) |
| CPU | AMD Ryzen 9 8945HS (CPU secuencial: Numba `@njit`, 1 hilo) |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU, 8 GB, driver 596.36, CC 8.9 |
| Software | Windows 11, Python 3.11, numpy 2.4.6, numba 0.68.1, numba-cuda 0.30.4, cuda-core 1.1.1 (CUDA 12.9 vía pip) |
| Configuración | 1 época, lote 64, semilla 123, 12 180 imágenes de entrenamiento, 2 610 de validación |
| Pesos para inferencia | `gpu_01.npz` original del Drive, igual en CPU y GPU |

Cada entrenamiento fue un proceso nuevo que partió de los mismos pesos iniciales
(Glorot, semilla 123). Las 3 repeticiones de cada backend dieron exactamente la misma
pérdida y accuracy, así que el entrenamiento es determinista dentro de cada backend.

## ⚠ Condición de energía (leer antes de usar los números de CPU)

La laptop se apagaba con el cargador conectado, así que parte de las corridas se hizo en batería:

| Corrida | Energía | Entrenamiento |
|---|---|---|
| GPU 01–03 | cargador | 32.9 / 34.2 / 32.8 s |
| CPU 01 | cargador | 588.6 s (9.8 min) |
| CPU 02 | batería (53 %) | 722.1 s (12.0 min) |
| CPU 03 | batería (38 %) | 951.1 s (15.9 min) |
| Inferencia CPU y GPU (todas) | batería (19 %) | — |

En batería Windows reduce el rendimiento de la CPU. Por eso la desviación de CPU es alta
(±183 s) y su promedio (754 s) es pesimista. **La comparación justa en entrenamiento es
CPU 01 contra GPU, todas con cargador: 588.6 s / 33.3 s ≈ 17.7x.** Con el promedio
mezclado sale 22.6x, pero ese valor está inflado. En inferencia CPU y GPU se midieron
en las mismas condiciones (batería), así que esa comparación sí es consistente.

## Resultados principales

### Entrenamiento (1 época)

| Etapa | CPU | GPU | Speedup |
|---|---|---|---|
| Compilación / calentamiento | 8.5 ± 1.0 s | 1.5 ± 0.2 s | 5.6x |
| Preparación GPU (reserva + copia de pesos) | — | 16 ± 4 ms | — |
| **Entrenamiento** | 588.6 s (con cargador) · 753.9 ± 183.3 s (las 3) | 33.3 ± 0.8 s | **17.7x** con cargador (22.6x mezclado) |
| Recuperar pesos GPU→RAM | — | 0.9 ms | — |
| Validación (2 610 imágenes) | 64.3 ± 23.6 s | 0.57 ± 0.02 s | 113.7x |

Comparado con la referencia del equipo en Colab (CPU ~22 min, Tesla T4 49 s), la RTX 4070
Laptop entrena ~1.5x más rápido que la T4.

### Inferencia de una imagen (300 mediciones = 3 corridas × 100 imágenes distintas)

| | CPU | GPU |
|---|---|---|
| Latencia promedio ± desv | 34.44 ± 1.39 ms | 2.92 ± 0.52 ms |
| Mediana (mín–máx) | 33.98 ms (32.9–42.4) | 2.79 ms (2.36–5.17) |
| Carga de pesos desde disco | 17 ms | 32 ms |
| Reserva de buffers + copia de pesos a GPU | — | 7 ms |
| Primera llamada (incluye compilación) | 3.7 s | 1.2 s |
| **Speedup** | | **11.8x** (promedio), 12.2x (mediana) |

La latencia GPU incluye copiar la imagen a la GPU, el forward y traer las probabilidades
de vuelta a RAM.

## ¿Cuándo vale la pena usar CUDA?

1. **Entrenamiento: siempre, en este proyecto.** El costo fijo de la GPU (compilar
   kernels ~1.5 s, preparar memoria ~16 ms, recuperar pesos ~1 ms) es menor que el de la
   CPU (compilar ~8.5 s). Desde el primer lote la GPU ya va ganando. Contando todo
   (compilación + preparación + entrenamiento + recuperación), una época tarda ~35 s en
   GPU contra ~10 min en CPU. Las transferencias de cada minibatch están incluidas en esos
   33 s y no anulan la ventaja: cada lote de 64 imágenes de 64×64 son ~1 MB, poco frente
   al cómputo de las convoluciones.

2. **Validación / inferencia por lotes: es donde más gana (~114x).** Sin backward ni
   Adam, cada lote es casi solo convoluciones, que se paralelizan muy bien, y con 64
   imágenes la GPU está mucho mejor ocupada.

3. **Inferencia de una sola imagen: gana, pero menos (~12x).** Con una imagen hay muy
   poco trabajo por kernel. Numba lo advierte (*grid size 12/50 will likely result in GPU
   under-utilization*), así que la latencia de ~2.8 ms está dominada por lanzar unos 15
   kernels y por las dos copias RAM↔GPU, no por el cálculo. Aun así es bastante más
   rápida que la CPU (~34 ms).

4. **Cuándo NO conviene.** Para predecir una sola imagen en un proceso nuevo, que carga
   pesos y compila, el costo fijo domina: GPU ≈ 32 + 7 + 1 200 ms ≈ 1.2 s y CPU ≈ 17 +
   3 700 ms ≈ 3.7 s. La GPU sigue ganando aquí solo porque su JIT compila más rápido; con
   la caché de Numba activada esa diferencia se reduce. CUDA conviene cuando el modelo
   queda residente en la GPU y se hacen muchas predicciones (un servicio o predicción
   por lotes), o cuando se entrena. Para muy pocas predicciones esporádicas, o en una
   máquina sin GPU NVIDIA, la CPU es suficiente.

## Observaciones

- **Accuracy distinta entre backends:** CPU da 0.4487 de validación y GPU 0.5034 (con
  los mismos pesos iniciales y la misma semilla). Cada backend es determinista consigo
  mismo. Esto coincide con el caso de redondeo del pooling que el equipo ya documentó y
  no se investigó más, como se acordó.
- **Pendiente para mediciones limpias:** repetir CPU 02–03 con cargador, o las 3 de GPU en
  batería, para que todo el entrenamiento se haya medido en las mismas condiciones.
