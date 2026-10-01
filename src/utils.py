"""
Módulo de utilidades: cálculo de métricas de hardware (FLOPs, MACs, latencia GPU),
estimación de cómputo de entrenamiento y generación de gráficos para el reporte.
Laboratorio #6 - Transfer Learning y Fine-Tuning.
"""

import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torchinfo import summary

# Configuración estética general de matplotlib
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 13

def compute_model_flops_and_macs(model, input_size=(1, 3, 32, 32)):
    """
    Calcula MACs y FLOPs de un modelo dado un tamaño de entrada usando torchinfo.
    1 MAC (Multiply-Accumulate) equivale a 2 FLOPs (1 multiplicación + 1 suma).
    """
    model_info = summary(model, input_size=input_size, verbose=0)
    # torchinfo computa mult_adds (MACs)
    macs = model_info.total_mult_adds
    flops = macs * 2  # 1 MAC = 2 FLOPs
    total_params = model_info.total_params
    trainable_params = model_info.trainable_params

    return {
        'macs': int(macs),
        'flops': int(flops),
        'total_params': int(total_params),
        'trainable_params': int(trainable_params)
    }

def measure_gpu_latency(model, input_size=(1, 3, 32, 32), device=None, 
                        warmup_iters=50, measure_iters=200):
    """
    Mide la latencia promedio de inferencia (en milisegundos) por imagen en GPU.
    Utiliza warm-up y sincronización explícita con torch.cuda.synchronize().
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(*input_size, device=device)

    # Warm-up
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(dummy_input)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Medición
    latencies = []
    with torch.no_grad():
        for _ in range(measure_iters):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0) # en ms

    return float(np.mean(latencies)), float(np.std(latencies))

def estimate_training_flops(flops_per_img, num_images, epochs, strategy="scratch"):
    """
    Estima el cómputo total de entrenamiento (en FLOPs).
    
    Estrategias:
    - 'scratch': Forward completo + Backward completo (gradiente respecto a pesos y activaciones).
                 Factor total aprox = Forward (1x) + Backward (2x) = 3x FLOPs.
    - 'feature_extractor': Forward completo del backbone y cabeza.
                           Backward solo de la cabeza clasificadora (el backbone está congelado,
                           no se calculan gradientes de pesos ni de activaciones previas).
                           Factor aprox = Forward (1x) + Backward cabeza (~0.05x) ≈ 1.05x a 1.1x FLOPs.
    - 'fine_tuning_block5': Forward completo. Backward atraviesa cabeza y bloque 5 (~30% del backbone).
                           Factor aprox = Forward (1x) + Backward (~0.7x) ≈ 1.7x FLOPs.
    - 'fine_tuning_blocks4_5': Forward completo. Backward de cabeza, bloque 5 y bloque 4 (~60% backbone).
                           Factor aprox = Forward (1x) + Backward (~1.3x) ≈ 2.3x FLOPs.
    """
    factors = {
        'scratch': 3.0,
        'feature_extractor': 1.05,
        'fine_tuning_block5': 1.7,
        'fine_tuning_blocks4_5': 2.3
    }
    bwd_factor = factors.get(strategy, 3.0)
    total_flops = float(flops_per_img * num_images * epochs * bwd_factor)
    return total_flops

def plot_dataset_samples(dataset, classes, samples_per_class=5, save_path="docs/figures/cifar10_samples.png"):
    """Visualiza una cuadrícula de ejemplos por clase de CIFAR-10."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    num_classes = len(classes)
    class_indices = {i: [] for i in range(num_classes)}

    for idx, (_, label) in enumerate(dataset):
        if len(class_indices[label]) < samples_per_class:
            class_indices[label].append(idx)
        if all(len(indices) == samples_per_class for indices in class_indices.values()):
            break

    fig, axes = plt.subplots(num_classes, samples_per_class, figsize=(10, 15))
    for c_idx, class_name in enumerate(classes):
        for s_idx in range(samples_per_class):
            img_idx = class_indices[c_idx][s_idx]
            img, _ = dataset[img_idx]
            # Convertir a numpy si es PIL o tensor
            if hasattr(img, 'numpy'):
                img = img.permute(1, 2, 0).numpy()
            axes[c_idx, s_idx].imshow(img)
            axes[c_idx, s_idx].axis('off')
            if s_idx == 0:
                axes[c_idx, s_idx].set_title(class_name, fontsize=11, fontweight='bold', loc='right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Muestras de clases guardadas en {save_path}")

def plot_training_loss_curves(histories, save_path="docs/figures/loss_curves_9_iterations.png"):
    """
    Grafica las curvas de pérdida (train y val) para las 9 iteraciones agrupadas en 3 columnas (por modelo).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True)

    model_categories = ["CNN desde Cero", "VGG-16 Feature Extractor", "VGG-16 Fine-Tuning"]
    keys = list(histories.keys())

    for i, key in enumerate(keys):
        row = i % 3
        col = i // 3
        ax = axes[row, col]
        h = histories[key]
        epochs = range(1, len(h['train_loss']) + 1)
        ax.plot(epochs, h['train_loss'], label='Train Loss', color='#1f77b4', lw=2)
        ax.plot(epochs, h['val_loss'], label='Val Loss', color='#d62728', lw=2, linestyle='--')
        ax.set_title(f"{h['model_name']}", fontsize=10, fontweight='bold')
        ax.set_ylabel("Loss")
        if row == 2:
            ax.set_xlabel("Epoch")
        ax.legend(loc='upper right', frameon=True)
        ax.grid(True, alpha=0.3)

    for col, title in enumerate(model_categories):
        axes[0, col].text(0.5, 1.15, title, transform=axes[0, col].transAxes,
                          fontsize=12, fontweight='bold', ha='center',
                          bbox=dict(boxstyle='round,pad=0.4', facecolor='#eef2f5', edgecolor='#b0bec5'))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Curvas de pérdida guardadas en {save_path}")

def plot_val_accuracy_vs_cumulative_time(best_histories, save_path="docs/figures/val_acc_vs_time.png"):
    """
    Gráfica obligatoria (1): Accuracy de validación vs. Tiempo de entrenamiento acumulado
    con los 3 mejores modelos en la misma figura.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 5))

    styles = {
        'cnn': ('#2ca02c', 'o-', 'CNN desde Cero'),
        'vgg_fe': ('#1f77b4', 's--', 'VGG-16 Feature Extractor'),
        'vgg_ft': ('#d62728', '^-', 'VGG-16 Fine-Tuning')
    }

    for key, h in best_histories.items():
        color, marker, label = styles.get(key, ('#333333', 'o-', key))
        times = h['cumulative_times']
        accs = [a * 100 for a in h['val_accuracy']]
        plt.plot(times, accs, marker, color=color, label=label, lw=2.2, markersize=6)

    plt.title("Accuracy de Validación vs. Tiempo de Entrenamiento Acumulado", fontweight='bold')
    plt.xlabel("Tiempo de entrenamiento acumulado (segundos)")
    plt.ylabel("Accuracy de validación (%)")
    plt.legend(frameon=True, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráfica Acc vs Tiempo guardada en {save_path}")

def plot_f1_vs_computational_cost(comparison_data, save_path="docs/figures/f1_vs_flops.png"):
    """
    Gráfica obligatoria (2): F1-score de test vs. Costo computacional (GFLOPs de inferencia).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 5))

    models = list(comparison_data.keys())
    gflops = [comparison_data[m]['inference_gflops'] for m in models]
    f1_scores = [comparison_data[m]['test_f1'] * 100 for m in models]
    colors = ['#2ca02c', '#1f77b4', '#d62728']
    labels = [comparison_data[m]['display_name'] for m in models]

    for x, y, c, lbl in zip(gflops, f1_scores, colors, labels):
        plt.scatter(x, y, color=c, s=180, edgecolors='black', zorder=5, label=lbl)
        plt.annotate(f"{lbl}\n({x:.2f} GFLOPs, {y:.1f}%)",
                     xy=(x, y), xytext=(10, -5), textcoords='offset points',
                     fontsize=9, fontweight='semibold')

    plt.title("Trade-off: F1-Score en Test vs. Costo de Inferencia (GFLOPs)", fontweight='bold')
    plt.xlabel("Costo de Inferencia por Imagen (GFLOPs) [Escala Logarítmica]")
    plt.ylabel("Macro F1-Score en Test (%)")
    plt.xscale('log')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráfica F1 vs GFLOPs guardada en {save_path}")

def plot_confusion_matrices(test_results, classes, save_path="docs/figures/confusion_matrices.png"):
    """
    Genera y guarda las matrices de confusión normalizadas de los 3 modelos evaluados en test.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    keys = list(test_results.keys())
    titles = [test_results[k]['display_name'] for k in keys]

    for i, (k, title) in enumerate(zip(keys, titles)):
        preds = test_results[k]['preds']
        targets = test_results[k]['targets']
        cm = confusion_matrix(targets, preds, normalize='true')

        sns.heatmap(cm, annot=False, cmap='Blues', ax=axes[i], cbar=(i == 2),
                    xticklabels=classes, yticklabels=classes)
        axes[i].set_title(f"{title}\nAccuracy: {test_results[k]['accuracy']*100:.1f}%", fontsize=11, fontweight='bold')
        axes[i].set_xlabel("Predicción", fontsize=10)
        axes[i].set_ylabel("Etiqueta Real" if i == 0 else "", fontsize=10)
        axes[i].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Matrices de confusión guardadas en {save_path}")

def plot_data_reduction_comparison(full_metrics, reduced_metrics, save_path="docs/figures/data_reduction_comparison.png"):
    """
    Compara visualmente el impacto de reducir datos al 10% para cada arquitectura.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 5))

    models = ['CNN desde Cero', 'VGG-16 Feature Ext.', 'VGG-16 Fine-Tuning']
    keys = ['cnn', 'vgg_fe', 'vgg_ft']

    full_accs = [full_metrics[k]['accuracy'] * 100 for k in keys]
    reduced_accs = [reduced_metrics[k]['accuracy'] * 100 for k in keys]

    x = np.arange(len(models))
    width = 0.35

    rects1 = plt.bar(x - width/2, full_accs, width, label='100% Datos (45,000 imgs)', color='#1f77b4')
    rects2 = plt.bar(x + width/2, reduced_accs, width, label='10% Datos (4,500 imgs)', color='#ff7f0e')

    plt.ylabel('Test Accuracy (%)')
    plt.title('Impacto de la Reducción de Datos de Entrenamiento (100% vs 10%)', fontweight='bold')
    plt.xticks(x, models)
    plt.ylim(0, 100)
    plt.legend(frameon=True)
    plt.grid(axis='y', alpha=0.3)

    for rect in rects1 + rects2:
        height = rect.get_height()
        plt.annotate(f'{height:.1f}%',
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparación de datos reducidos guardada en {save_path}")
