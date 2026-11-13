# Proyecto 3 CNN con CUDA

Comparamos dos implementaciones de una red neuronal convolucional (CNN) que
clasifica imágenes del alfabeto del Lenguaje de Señas Americano (ASL). Una
ejecuta las operaciones en CPU de forma secuencial con Numba y la otra usa
kernels propios de CUDA en una GPU NVIDIA. Ambas reutilizan las imágenes
preparadas en el Laboratorio 3 de Data Science, la misma arquitectura y la misma
configuración de entrenamiento.

## Organización del código

| Ruta | Qué contiene |
|---|---|
| `src/config.py` | Configuración común de la red, clases, rutas y semilla. |
| `src/compartido/` | Carga y validación de los tensores, inicialización de los parámetros y guardado y carga de pesos. |
| `src/cpu/` | Implementación secuencial de la CNN, entrenamiento y predicción. Los cálculos de cada capa están en `src/cpu/operaciones/`. |
| `src/gpu/` | Implementación paralela de la misma CNN. Los kernels están en `src/gpu/operaciones/` y `src/gpu/memoria.py` reserva los arreglos y transfiere datos entre CPU y GPU. |
| `src/entrenar_cpu.py` y `src/entrenar_gpu.py` | Entrenan, evalúan en validación y guardan los pesos y tiempos de cada corrida. |
| `src/medir_prediccion.py` | Mide cuánto tarda en clasificar una imagen con pesos guardados. |
| `src/verificar_equivalencia.py` | Compara salidas y gradientes con los mismos pesos e imágenes. Reporta diferencias de redondeo que pueden cambiar la posición elegida por max pooling. |
| `experimentos/` | Ejecuta corridas repetidas y genera tablas y gráficas a partir de los registros. |
| `notebooks/` | Cuadernos de entrenamiento e inferencia para Google Colab. |
| `models/dani/` y `resultados/dani/` | Pesos, registros, tablas y gráficas de las corridas en Windows. |

El flujo empieza en la CPU, donde se lee el archivo de tensores y se forman los
lotes. En la versión CUDA se copian las imágenes y etiquetas de cada lote a la
GPU. Los pesos permanecen en la GPU entre lotes y los kernels calculan las
salidas, la pérdida y los gradientes y actualizan los parámetros. La inferencia
usa pesos ya guardados y devuelve las probabilidades a la CPU para seleccionar
la clase predicha.

## Preparar los datos

Colocá el archivo `02_tensores.npz` del laboratorio en
`data/processed/02_tensores.npz`. Se puede obtener en la
[carpeta compartida del proyecto](https://drive.google.com/drive/folders/1nAVs5gy2fb9hBYm0fmC4BasqGQjtQoEv).

El archivo contiene las imágenes de 64 * 64 píxeles en escala de grises y sus
etiquetas, distribuidas en tres particiones. Se usan 12,180 imágenes para entrenar
y 2,610 para validar. La partición de prueba queda reservada.

## Ejecutar en local

Ejecutá los comandos desde la raíz del repositorio, con tu entorno virtual de
Python activado.

- Para CPU, instalá las dependencias con `python -m pip install -r requirements.txt`.
- Para GPU en Windows, usá un equipo con GPU NVIDIA y su controlador instalado.
  El entorno probado utilizó Python 3.11.8. Instalá las dependencias con
  `python -m pip install -r requirements_windows.txt`.

### Medir la predicción con pesos guardados

Estos comandos usan el mismo archivo de pesos y seleccionan las mismas 100
imágenes de validación mediante la semilla fija. El segundo requiere una GPU
NVIDIA. Cada comando guarda un registro JSON distinto.

```bash
python src/medir_prediccion.py --backend cpu --ruta-modelo models/dani/dani_gpu_01.npz --ruta-registro resultados/demo/prediccion_cpu.json
python src/medir_prediccion.py --backend gpu --ruta-modelo models/dani/dani_gpu_01.npz --ruta-registro resultados/demo/prediccion_gpu.json
```

### Entrenar o repetir las mediciones

```bash
python src/entrenar_cpu.py --epocas 1
python src/entrenar_gpu.py --epocas 1
```

En Windows con GPU NVIDIA podés ejecutar el conjunto de experimentos y generar
el resumen con los siguientes comandos.

```bash
python experimentos/correr_experimentos.py --prefijo nueva_corrida --repeticiones 3
python experimentos/analizar_resultados.py --prefijo nueva_corrida
```

El ejecutor guarda los pesos en `models/nueva_corrida/` y los registros en
`resultados/nueva_corrida/`. Para inferencia usa los pesos de la primera corrida
GPU, salvo que se indique otro archivo con `--ruta-modelo-prediccion`. Los
scripts individuales requieren rutas de salida nuevas para cada corrida. Si
repetís el experimento completo, usá un prefijo nuevo.

## Ejecutar en Colab

| Cuaderno | Uso |
|---|---|
| [00_colab.ipynb](notebooks/00_colab.ipynb) | Preparar el entorno, comprobar CUDA y entrenar en CPU. |
| [01_prediccion_cpu.ipynb](notebooks/01_prediccion_cpu.ipynb) | Predecir en CPU con pesos guardados. |
| [02_entrenamiento_gpu.ipynb](notebooks/02_entrenamiento_gpu.ipynb) | Entrenar con CUDA y respaldar los resultados en Drive. |
| [03_prediccion_gpu.ipynb](notebooks/03_prediccion_gpu.ipynb) | Predecir con CUDA usando pesos guardados. |

Seleccioná un entorno con GPU NVIDIA para los cuadernos que usan CUDA. Estos
instalan las dependencias de `requirements_colab.txt`. Los cuadernos buscan
`proyecto_senias` en Mi unidad de Drive. Colocá ahí la carpeta o un acceso
directo a ella. Para entrenar de nuevo, elegí un nombre de corrida distinto.

El cuaderno de predicción GPU incluye una comprobación de equivalencia. Si se
detiene por la diferencia de redondeo de max pooling documentada en el informe,
podés ejecutar las celdas de predicción por separado.

## Resultados

Las mediciones se realizaron en una laptop Windows con un AMD Ryzen 9 8945HS y
una NVIDIA RTX 4070 Laptop GPU.

| Medición | CPU | GPU | Aceleración |
|---|---:|---:|---:|
| Entrenamiento de una época con cargador | 588.6 s | 33.3 s | 17.7x |
| Predicción de una imagen en batería, con pesos cargados | 34.44 ms | 2.92 ms | 11.8x |

El tiempo de GPU incluye las transferencias necesarias durante la operación
medida. Para entrenamiento se compara una corrida CPU con el promedio de tres
corridas GPU con cargador. Las otras dos corridas CPU se hicieron en batería.
Para inferencia se usaron tres corridas por versión con las mismas 100 imágenes
de validación. Se reportan los tiempos medios de las 300 mediciones por versión.

Consultá el [análisis de las corridas](resultados/dani/analisis.md) y la
[gráfica de latencia](resultados/dani/graficas/prediccion_latencia.png) para ver
el detalle de las mediciones.

## Referencias

- [ASL Alphabet en Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- [Kernels con Numba CUDA](https://nvidia.github.io/numba-cuda/user/kernels.html)
