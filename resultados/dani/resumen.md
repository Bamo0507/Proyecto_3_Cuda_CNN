# Resultados `dani`

## Entorno

- Commit: `54ccb42ada6fa6ba1883df225a0c6d32e0d44ffe`
- Cambios locales sin commit: `?? experimentos/
?? models/
?? requirements_windows.txt
?? resultados/
?? src/medir_prediccion.py`
- CPU: AMD Ryzen 9 8945HS w/ Radeon 780M Graphics
- GPU: NVIDIA GeForce RTX 4070 Laptop GPU, 596.36, 8188 MiB
- Sistema: Windows-10-10.0.26200-SP0

## Entrenamiento (segundos)

| Etapa | CPU promedio ± desv (n) | GPU promedio ± desv (n) | Speedup CPU/GPU |
|---|---|---|---|
| Compilacion | 8.526 ± 1.022 (3) | 1.512 ± 0.184 (3) | 5.6x |
| Preparacion GPU | — | 0.016 ± 0.004 (3) | — |
| Entrenamiento (1 epoca) | 753.922 ± 183.325 (3) | 33.300 ± 0.778 (3) | 22.6x |
| Recuperar pesos | — | 0.001 ± 0.000 (3) | — |
| Validacion | 64.326 ± 23.564 (3) | 0.566 ± 0.021 (3) | 113.7x |
| **Total extremo a extremo** (compilacion + preparacion + entrenamiento + recuperacion) | 762.448 ± 183.501 | 34.839 ± 0.755 | 21.9x |

Accuracy por corrida (entrenamiento / validacion):

- entrenamiento_dani_cpu_01.json: 0.2407 / 0.4487
- entrenamiento_dani_cpu_02.json: 0.2407 / 0.4487
- entrenamiento_dani_cpu_03.json: 0.2407 / 0.4487
- entrenamiento_dani_gpu_01.json: 0.2484 / 0.5034
- entrenamiento_dani_gpu_02.json: 0.2484 / 0.5034
- entrenamiento_dani_gpu_03.json: 0.2484 / 0.5034

## Prediccion de una imagen (milisegundos)

| Backend | Latencia promedio ± desv | Mediana | Min | Max | n | Carga de pesos (ms) | Preparacion GPU (ms) | Primera llamada (s) |
|---|---|---|---|---|---|---|---|---|
| CPU | 34.4361 ± 1.3897 | 33.9770 | 32.9443 | 42.4069 | 300 (3 corridas) | 17.21 | — | 3.738 |
| GPU | 2.9205 ± 0.5217 | 2.7910 | 2.3639 | 5.1654 | 300 (3 corridas) | 32.39 | 6.93 | 1.206 |

Speedup de inferencia CPU/GPU: 11.79x (promedio), 12.17x (mediana).

