"""Resume las corridas: tablas (CSV y Markdown), speedup y graficas PNG.

Ejecutar desde la raiz del repo:
    python experimentos/analizar_resultados.py --prefijo dani
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
COLOR = {"cpu": "#2a78d6", "gpu": "#eb6834"}
TEXTO = "#0b0b0b"
TEXTO_SECUNDARIO = "#52514e"
GRILLA = "#e4e3df"
SUPERFICIE = "#fcfcfb"

ETAPAS_ENTRENAMIENTO = [
    ("calentamiento_y_compilacion", "Compilacion"),
    ("preparacion_gpu", "Preparacion GPU"),
    ("entrenamiento", "Entrenamiento (1 epoca)"),
    ("recuperacion_parametros", "Recuperar pesos"),
    ("validacion", "Validacion"),
]


def cargar(directorio, patron):
    return [json.loads(ruta.read_text(encoding="utf-8")) | {"_archivo": ruta.name}
            for ruta in sorted(directorio.glob(patron))]


def estadisticos(valores):
    valores = np.asarray(valores, dtype=float)
    return {
        "n": len(valores),
        "promedio": valores.mean(),
        "desv": valores.std(ddof=1) if len(valores) > 1 else 0.0,
        "min": valores.min(),
        "max": valores.max(),
    }


def estilo_ejes(eje):
    eje.set_facecolor(SUPERFICIE)
    for lado in ("top", "right"):
        eje.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        eje.spines[lado].set_color(GRILLA)
    eje.tick_params(colors=TEXTO_SECUNDARIO)
    eje.yaxis.grid(True, color=GRILLA, linewidth=0.8)
    eje.set_axisbelow(True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefijo", default="dani")
    argumentos = parser.parse_args()
    directorio = RAIZ / "resultados" / argumentos.prefijo
    directorio_graficas = directorio / "graficas"
    directorio_graficas.mkdir(exist_ok=True)

    entrenamiento = {b: cargar(directorio, f"entrenamiento_*_{b}_*.json") for b in ("cpu", "gpu")}
    prediccion = {b: cargar(directorio, f"prediccion_*_{b}_*.json") for b in ("cpu", "gpu")}
    lineas_md = [f"# Resultados `{argumentos.prefijo}`", ""]

    entorno_ruta = directorio / "entorno.json"
    if entorno_ruta.exists():
        entorno = json.loads(entorno_ruta.read_text(encoding="utf-8"))
        lineas_md += [
            "## Entorno", "",
            f"- Commit: `{entorno.get('commit')}`",
            f"- Cambios locales sin commit: `{entorno.get('cambios_sin_commit') or 'ninguno'}`",
            f"- CPU: {entorno.get('cpu')}",
            f"- GPU: {entorno.get('gpu_nvidia_smi')}",
            f"- Sistema: {entorno.get('sistema')}", "",
        ]

    # ---- Entrenamiento: corridas individuales ----
    filas_individuales = []
    for backend, registros in entrenamiento.items():
        for registro in registros:
            fila = {"backend": backend, "archivo": registro["_archivo"]}
            fila.update({clave: registro["tiempos_segundos"].get(clave, "")
                         for clave, _ in ETAPAS_ENTRENAMIENTO})
            fila["accuracy_entrenamiento"] = registro["historial"][-1]["accuracy"]
            fila["accuracy_validacion"] = registro["validacion"]["accuracy"]
            filas_individuales.append(fila)
    if filas_individuales:
        with (directorio / "entrenamiento_individual.csv").open("w", newline="", encoding="utf-8") as f:
            escritor = csv.DictWriter(f, fieldnames=list(filas_individuales[0]))
            escritor.writeheader()
            escritor.writerows(filas_individuales)

    # ---- Entrenamiento: resumen y speedup ----
    resumen_entrenamiento = {}
    lineas_md += ["## Entrenamiento (segundos)", "",
                  "| Etapa | CPU promedio ± desv (n) | GPU promedio ± desv (n) | Speedup CPU/GPU |",
                  "|---|---|---|---|"]
    for clave, nombre in ETAPAS_ENTRENAMIENTO:
        celdas = {}
        for backend, registros in entrenamiento.items():
            valores = [r["tiempos_segundos"][clave] for r in registros if clave in r["tiempos_segundos"]]
            if valores:
                resumen_entrenamiento[(backend, clave)] = estadisticos(valores)
        cpu = resumen_entrenamiento.get(("cpu", clave))
        gpu = resumen_entrenamiento.get(("gpu", clave))
        texto = lambda s: f"{s['promedio']:.3f} ± {s['desv']:.3f} ({s['n']})" if s else "—"
        speedup = f"{cpu['promedio'] / gpu['promedio']:.1f}x" if cpu and gpu else "—"
        lineas_md.append(f"| {nombre} | {texto(cpu)} | {texto(gpu)} | {speedup} |")

    total_cpu = [sum(r["tiempos_segundos"][c] for c in ("calentamiento_y_compilacion", "entrenamiento"))
                 for r in entrenamiento["cpu"]]
    total_gpu = [sum(r["tiempos_segundos"][c] for c in (
        "preparacion_calentamiento_gpu", "calentamiento_y_compilacion",
        "preparacion_gpu", "entrenamiento", "recuperacion_parametros"))
        for r in entrenamiento["gpu"]]
    if total_cpu and total_gpu:
        s_cpu, s_gpu = estadisticos(total_cpu), estadisticos(total_gpu)
        lineas_md += [
            f"| **Total extremo a extremo** (compilacion + preparacion + entrenamiento + recuperacion) "
            f"| {s_cpu['promedio']:.3f} ± {s_cpu['desv']:.3f} | {s_gpu['promedio']:.3f} ± {s_gpu['desv']:.3f} "
            f"| {s_cpu['promedio'] / s_gpu['promedio']:.1f}x |",
        ]
    lineas_md.append("")
    if filas_individuales:
        lineas_md += ["Accuracy por corrida (entrenamiento / validacion):", ""]
        for fila in filas_individuales:
            lineas_md.append(
                f"- {fila['archivo']}: {fila['accuracy_entrenamiento']:.4f} / {fila['accuracy_validacion']:.4f}")
        lineas_md.append("")

    # ---- Prediccion ----
    lineas_md += ["## Prediccion de una imagen (milisegundos)", "",
                  "| Backend | Latencia promedio ± desv | Mediana | Min | Max | n | Carga de pesos (ms) | Preparacion GPU (ms) | Primera llamada (s) |",
                  "|---|---|---|---|---|---|---|---|---|"]
    latencias = {}
    for backend, registros in prediccion.items():
        if not registros:
            continue
        todas = np.concatenate([r["tiempos_prediccion_ms"] for r in registros])
        latencias[backend] = todas
        promedios_corrida = [np.mean(r["tiempos_prediccion_ms"]) for r in registros]
        carga = estadisticos([r["tiempos_segundos"]["carga_pesos_disco"] * 1000 for r in registros])
        prep = [r["tiempos_segundos"].get("preparacion_memoria_y_copia_pesos") for r in registros]
        prep = estadisticos([p * 1000 for p in prep]) if prep[0] is not None else None
        primera = estadisticos([r["tiempos_segundos"]["primera_prediccion_con_compilacion"] for r in registros])
        texto_prep = f"{prep['promedio']:.2f}" if prep else "—"
        lineas_md.append(
            f"| {backend.upper()} | {todas.mean():.4f} ± {todas.std(ddof=1):.4f} | {np.median(todas):.4f} "
            f"| {todas.min():.4f} | {todas.max():.4f} | {len(todas)} ({len(registros)} corridas) "
            f"| {carga['promedio']:.2f} | {texto_prep} | {primera['promedio']:.3f} |"
        )
        with (directorio / f"prediccion_individual_{backend}.csv").open("w", newline="", encoding="utf-8") as f:
            escritor = csv.writer(f)
            escritor.writerow(["archivo", "medicion", "indice_validacion", "latencia_ms"])
            for r in registros:
                for i, (t, p) in enumerate(zip(r["tiempos_prediccion_ms"], r["predicciones"])):
                    escritor.writerow([r["_archivo"], i, p["indice_validacion"], t])
        latencias[backend + "_promedios_corrida"] = promedios_corrida
    if "cpu" in latencias and "gpu" in latencias:
        speedup_mediana = np.median(latencias["cpu"]) / np.median(latencias["gpu"])
        speedup_promedio = latencias["cpu"].mean() / latencias["gpu"].mean()
        lineas_md += ["", f"Speedup de inferencia CPU/GPU: {speedup_promedio:.2f}x (promedio), "
                          f"{speedup_mediana:.2f}x (mediana).", ""]

    (directorio / "resumen.md").write_text("\n".join(lineas_md) + "\n", encoding="utf-8")
    print("\n".join(lineas_md))

    plt.rcParams.update({"font.size": 10, "text.color": TEXTO, "axes.labelcolor": TEXTO_SECUNDARIO,
                         "figure.facecolor": SUPERFICIE})

    # ---- Grafica 1: etapas de entrenamiento ----
    etapas = [(c, n) for c, n in ETAPAS_ENTRENAMIENTO
              if ("cpu", c) in resumen_entrenamiento or ("gpu", c) in resumen_entrenamiento]
    if etapas:
        figura, eje = plt.subplots(figsize=(9, 4.5))
        estilo_ejes(eje)
        x = np.arange(len(etapas))
        ancho = 0.36
        for desplazamiento, backend in ((-ancho / 2 - 0.01, "cpu"), (ancho / 2 + 0.01, "gpu")):
            proms = [resumen_entrenamiento.get((backend, c), {}).get("promedio", np.nan) for c, _ in etapas]
            desvs = [resumen_entrenamiento.get((backend, c), {}).get("desv", 0) for c, _ in etapas]
            barras = eje.bar(x + desplazamiento, proms, ancho, yerr=desvs, color=COLOR[backend],
                             label=backend.upper(), capsize=3, error_kw={"ecolor": TEXTO_SECUNDARIO, "lw": 1})
            for barra, valor in zip(barras, proms):
                if np.isfinite(valor):
                    etiqueta = (f"{valor * 1000:.1f} ms" if valor < 0.1
                                else f"{valor:.2f} s" if valor < 100 else f"{valor / 60:.1f} min")
                    eje.annotate(etiqueta,
                                 (barra.get_x() + barra.get_width() / 2, valor),
                                 xytext=(0, 4), textcoords="offset points", ha="center",
                                 fontsize=8, color=TEXTO_SECUNDARIO)
        eje.set_yscale("log")
        eje.set_xticks(x, [n for _, n in etapas])
        eje.set_ylabel("Tiempo (s, escala log)")
        eje.set_title("Tiempo por etapa de entrenamiento: promedio ± desv. estandar", loc="left", color=TEXTO)
        eje.legend(frameon=False)
        figura.tight_layout()
        figura.savefig(directorio_graficas / "entrenamiento_etapas.png", dpi=150)
        plt.close(figura)

    # ---- Grafica 2: speedup ----
    speedups = [(n, resumen_entrenamiento[("cpu", c)]["promedio"] / resumen_entrenamiento[("gpu", c)]["promedio"])
                for c, n in ETAPAS_ENTRENAMIENTO
                if ("cpu", c) in resumen_entrenamiento and ("gpu", c) in resumen_entrenamiento]
    if total_cpu and total_gpu:
        speedups.append(("Total extremo a extremo", np.mean(total_cpu) / np.mean(total_gpu)))
    if "cpu" in latencias and "gpu" in latencias:
        speedups.append(("Inferencia 1 imagen", latencias["cpu"].mean() / latencias["gpu"].mean()))
    if speedups:
        figura, eje = plt.subplots(figsize=(8, 0.6 * len(speedups) + 1.2))
        estilo_ejes(eje)
        eje.yaxis.grid(False)
        eje.xaxis.grid(True, color=GRILLA, linewidth=0.8)
        nombres = [n for n, _ in speedups][::-1]
        valores = [v for _, v in speedups][::-1]
        barras = eje.barh(nombres, valores, color=COLOR["gpu"], height=0.55)
        eje.axvline(1, color=TEXTO_SECUNDARIO, lw=1, ls="--")
        for barra, valor in zip(barras, valores):
            eje.annotate(f"{valor:.1f}x", (valor, barra.get_y() + barra.get_height() / 2),
                         xytext=(4, 0), textcoords="offset points", va="center", fontsize=9, color=TEXTO)
        eje.set_xlabel("Speedup = tiempo CPU / tiempo GPU (linea punteada = 1x, sin mejora)")
        eje.set_title("Speedup de la GPU por etapa", loc="left", color=TEXTO)
        figura.tight_layout()
        figura.savefig(directorio_graficas / "speedup.png", dpi=150)
        plt.close(figura)

    # ---- Grafica 3: latencia de inferencia ----
    if "cpu" in latencias or "gpu" in latencias:
        figura, eje = plt.subplots(figsize=(6, 4.5))
        estilo_ejes(eje)
        backends = [b for b in ("cpu", "gpu") if b in latencias]
        generador = np.random.default_rng(0)
        for posicion, backend in enumerate(backends):
            datos = latencias[backend]
            jitter = generador.uniform(-0.15, 0.15, len(datos))
            eje.scatter(posicion + jitter, datos, s=10, color=COLOR[backend], alpha=0.45, linewidths=0)
            mediana = np.median(datos)
            eje.hlines(mediana, posicion - 0.28, posicion + 0.28, color=TEXTO, lw=2)
            eje.annotate(f"mediana {mediana:.3f} ms", (posicion + 0.3, mediana), va="center",
                         fontsize=8, color=TEXTO_SECUNDARIO)
        eje.set_yscale("log")
        eje.set_xticks(range(len(backends)), [b.upper() for b in backends])
        eje.set_xlim(-0.5, len(backends) - 0.5 + 0.6)
        eje.set_ylabel("Latencia por imagen (ms, escala log)")
        eje.set_title("Prediccion de una imagen (cada punto = 1 medicion)", loc="left", color=TEXTO)
        figura.tight_layout()
        figura.savefig(directorio_graficas / "prediccion_latencia.png", dpi=150)
        plt.close(figura)

    print("Graficas en:", directorio_graficas)


main()
