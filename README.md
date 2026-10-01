# Laboratorio #6: Transfer Learning y Fine-Tuning en CIFAR-10

**Curso**: CC3092 - Deep Learning y Sistemas Inteligentes  
**Autor**: Javier España  
**Repositorio GitHub**: [https://github.com/Javier-Espana/Lab6-DL.git](https://github.com/Javier-Espana/Lab6-DL.git)

---

## 📌 Descripción del Proyecto

Este repositorio contiene la implementación completa, rigurosa y reproducible del **Laboratorio #6: Transfer Learning y Fine-Tuning** sobre el dataset CIFAR-10. 

Se evalúan, comparan y perfilan tres paradigmas arquitectónicos para clasificación multiclase:
1. **CNN desde Cero**: Arquitectura convolucional propia entrenada sobre resolución nativa $32\times 32$ píxeles, incorporando Batch Normalization, Dropout espacial y regularización por Data Augmentation.
2. **VGG-16 como Feature Extractor**: Reutilización del backbone convolucional preentrenado en ImageNet-1K congelando sus pesos (`requires_grad = False`) y optimizando una cabeza clasificadora sobre entradas redimensionadas a $128\times 128$.
3. **VGG-16 con Fine-Tuning**: Descongelamiento selectivo de bloques convolucionales superiores (Bloque 5 vs. Bloques 4 y 5) empleando tasas de aprendizaje diferenciales (*discriminative learning rates*) para especializar representaciones sin incurrir en olvido catastrófico (*Catastrophic Forgetting*).

---

## 📂 Estructura del Repositorio

```text
Lab6-DL/
├── .gitignore                          # Exclusiones de Git (venv, data, checkpoints, auxiliares LaTeX)
├── README.md                           # Documentación general y guía de replicabilidad
├── requirements.txt                    # Dependencias y librerías de Python
├── run_experiments.py                  # Script principal para ejecutar toda la suite experimental
├── scripts/
│   └── build_notebooks.py              # Generador automatizado de los Jupyter Notebooks
├── src/                                # Módulos de código modulares y reutilizables
│   ├── __init__.py
│   ├── data.py                         # Carga, partición estratificada (45k/5k) y pipelines de aumentación
│   ├── models.py                       # Arquitecturas (CNNs, VGG-16 FE y VGG-16 FT)
│   ├── train.py                        # Bucles de entrenamiento con sincronización CUDA y métricas macro
│   └── utils.py                        # Medición de FLOPs/MACs, latencia GPU y utilidades de graficación
├── docs/
│   ├── Laboratorio_6_Transfer_Learning.pdf   # Guía oficial del laboratorio
│   ├── reporte_laboratorio_6.tex       # Código fuente LaTeX del informe académico (máx. 4 páginas)
│   ├── reporte_laboratorio_6.pdf       # Informe final compilado en formato de dos columnas
│   └── figures/                        # Gráficas generadas para el informe y notebooks
│       ├── cifar10_samples.png
│       ├── loss_curves_9_iterations.png
│       ├── confusion_matrices.png
│       ├── val_acc_vs_time.png
│       ├── f1_vs_flops.png
│       └── data_reduction_comparison.png
├── notebooks/                          # Jupyter Notebooks completos y comentados
│   ├── 01_exploracion_y_preparacion.ipynb
│   ├── 02_investigacion_transfer_learning.ipynb
│   ├── 03_modelos_y_entrenamiento.ipynb
│   ├── 04_evaluacion_y_experimentos.ipynb
│   └── Laboratorio_6_Transfer_Learning.ipynb # Notebook maestro consolidado end-to-end
└── results/
    └── experiment_results.json         # Registro estructurado de métricas de las 9 iteraciones y hardware
```

---

## 🚀 Requisitos e Instalación

Para configurar el entorno virtual y reproducir los experimentos con aceleración GPU (CUDA):

```bash
# 1. Clonar el repositorio
git clone https://github.com/Javier-Espana/Lab6-DL.git
cd Lab6-DL

# 2. Crear y activar entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# 3. Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🔬 Ejecución de Experimentos

Para ejecutar el entrenamiento de las 9 iteraciones, la medición de recursos en GPU, la evaluación en test y el experimento de escasez de datos (10%):

```bash
python run_experiments.py
```

Para regenerar todos los notebooks interactivos en `notebooks/`:

```bash
python scripts/build_notebooks.py
```

Para compilar el informe en LaTeX (`docs/reporte_laboratorio_6.pdf`):

```bash
cd docs
pdflatex reporte_laboratorio_6.tex
pdflatex reporte_laboratorio_6.tex
cd ..
```

---

## 📊 Resumen de Resultados Principales

| Modelo | Parámetros Totales | Inferencia FLOPs | Latencia GPU | Test Accuracy (100% Datos) | Test Macro F1 | Test Accuracy (10% Datos) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CNN desde Cero** | 1,122,250 | 0.042 GFLOPs | 0.48 ms | 79.24% | 0.7915 | 63.12% ($\Delta -16.1\%$) |
| **VGG-16 Feature Extractor** | 27,563,338 | 7.420 GFLOPs | 1.74 ms | 84.12% | 0.8402 | 79.88% ($\Delta -4.2\%$) |
| **VGG-16 Fine-Tuning** | 27,562,314 | 7.420 GFLOPs | 1.75 ms | **88.54%** | **0.8846** | **81.24%** ($\Delta -7.3\%$) |

* **Hardware empleado**: NVIDIA GeForce RTX 5050 (7.52 GB VRAM), PyTorch 2.14.1 con CUDA 13.0.
