# Proyecto 3 CNN con CUDA

Comparacion de entrenamiento e inferencia de una CNN ASL en CPU secuencial y
GPU NVIDIA con kernels propios. El proyecto esta en configuracion inicial;
todavia no implementa la CNN ni tiene mediciones de rendimiento.

## Desarrollo local

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

La base local usa NumPy y Numba para la referencia CPU. No necesita instalar un
toolkit CUDA en el Mac. `requirements_colab.txt` se usa unicamente en Colab con
GPU NVIDIA. Las versiones aun no estan fijadas: se cerraran tras verificar ambos
entornos; guardar un freeze de la sesion no equivale a validarlas en otro sistema.

## Comprobacion en Colab

Abrir `notebooks/00_colab.ipynb`, seleccionar un entorno con GPU y ejecutar sus
celdas en orden. El notebook descarga una revision explicita de GitHub, instala
dependencias y verifica que Numba pueda detectar la GPU. No entrena ni mide aun.
La configuracion inicial de dependencias CUDA usa la familia 12 y debe verificarse
en la maquina asignada antes de fijar versiones o ejecutar experimentos.

El repositorio debe tener un primer commit publicado antes de clonarlo desde
Colab. Si es privado, el acceso requiere autenticacion; no poner tokens en las
celdas ni en URLs guardadas. El notebook se detiene si falla la descarga.

## Flujo de cambios

Editar codigo localmente, subir cambios a GitHub y abrir una copia limpia del
commit elegido en Colab. Registrar el SHA efectivo en cada ejecucion. Git no
sincroniza archivos automaticamente; los modulos Python ya importados tampoco
se actualizan solo por descargar una nueva revision.

## Referencias

- [Instalacion de Numba-CUDA](https://nvidia.github.io/numba-cuda/user/installation.html)
- [Kernels propios con Numba-CUDA](https://nvidia.github.io/numba-cuda/user/kernels.html)
