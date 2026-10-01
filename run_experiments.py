"""
Script principal para ejecutar la suite experimental completa del Laboratorio 6:
1. Estadísticas de CIFAR-10 y visualización de muestras.
2. Entrenamiento de las 9 iteraciones (CNN, VGG-16 Feature Extractor, VGG-16 Fine-Tuning).
3. Registro de recursos (FLOPs, MACs, memoria GPU, tiempos, latencia).
4. Evaluación en Test y matrices de confusión para los mejores modelos.
5. Experimento de escasez de datos (10% de datos de entrenamiento).
6. Generación de todas las gráficas para el informe y JSON con métricas consolidadas.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.data import (
    CLASSES, get_raw_cifar10, compute_dataset_stats,
    get_dataloaders
)
from src.models import (
    BaselineCNN, RegularizedCNN, EnhancedCNN,
    build_vgg16_feature_extractor, build_vgg16_fine_tuning,
    get_optimizer_param_groups, count_parameters
)
from src.train import train_model, evaluate
from src.utils import (
    compute_model_flops_and_macs, measure_gpu_latency,
    estimate_training_flops, plot_dataset_samples,
    plot_training_loss_curves, plot_val_accuracy_vs_cumulative_time,
    plot_f1_vs_computational_cost, plot_confusion_matrices,
    plot_data_reduction_comparison
)

def main():
    print("=" * 80)
    print("LABORATORIO #6 - TRANSFER LEARNING Y FINE-TUNING")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo activo: {device}")
    if device.type == 'cuda':
        print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
        print(f"Memoria GPU total: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    os.makedirs("results", exist_ok=True)
    os.makedirs("docs/figures", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # --------------------------------------------------------------------------
    # 1. Exploración de Datos y Estadísticas
    # --------------------------------------------------------------------------
    print("\n[Paso 1] Descargando y analizando CIFAR-10...")
    raw_train, raw_test = get_raw_cifar10(root="data")
    cifar_mean, cifar_std = compute_dataset_stats(raw_train)
    print(f"Observaciones: Train={len(raw_train)}, Test={len(raw_test)}")
    print(f"Clases: {CLASSES}")
    print(f"Media CIFAR-10 por canal (RGB): {cifar_mean}")
    print(f"Std CIFAR-10 por canal (RGB):   {cifar_std}")

    # Guardar gráfico de muestras de CIFAR-10
    plot_dataset_samples(raw_train, CLASSES, samples_per_class=5, 
                         save_path="docs/figures/cifar10_samples.png")

    cifar_stats = (cifar_mean, cifar_std)

    # --------------------------------------------------------------------------
    # 2. DataLoaders
    # --------------------------------------------------------------------------
    print("\n[Paso 2] Creando DataLoaders...")
    cnn_train_loader, cnn_val_loader, cnn_test_loader = get_dataloaders(
        model_type="cnn", batch_size=128, cifar_stats=cifar_stats
    )
    vgg_train_loader, vgg_val_loader, vgg_test_loader = get_dataloaders(
        model_type="vgg", batch_size=128, vgg_img_size=128
    )

    criterion = nn.CrossEntropyLoss()
    histories = {}
    models_dict = {}

    # --------------------------------------------------------------------------
    # 3. Modelos e Iteraciones
    # --------------------------------------------------------------------------
    print("\n[Paso 3] Ejecutando las 9 iteraciones de entrenamiento...")

    # --- MODELO 1: CNN DESDE CERO ---
    # Iter 1.1: Baseline
    m_cnn1 = BaselineCNN(num_classes=10)
    opt_cnn1 = optim.Adam(m_cnn1.parameters(), lr=1e-3)
    m_cnn1, h_cnn1 = train_model(
        m_cnn1, cnn_train_loader, cnn_val_loader, criterion, opt_cnn1,
        num_epochs=10, device=device, model_name="CNN Iter 1.1 (Baseline)"
    )
    histories['cnn_iter1'] = h_cnn1
    models_dict['cnn_iter1'] = m_cnn1

    # Iter 1.2: Regularized
    m_cnn2 = RegularizedCNN(num_classes=10, dropout_rate=0.25)
    opt_cnn2 = optim.Adam(m_cnn2.parameters(), lr=1e-3)
    m_cnn2, h_cnn2 = train_model(
        m_cnn2, cnn_train_loader, cnn_val_loader, criterion, opt_cnn2,
        num_epochs=10, device=device, model_name="CNN Iter 1.2 (BatchNorm + Dropout)"
    )
    histories['cnn_iter2'] = h_cnn2
    models_dict['cnn_iter2'] = m_cnn2

    # Iter 1.3: Enhanced + Scheduler
    m_cnn3 = EnhancedCNN(num_classes=10)
    opt_cnn3 = optim.AdamW(m_cnn3.parameters(), lr=1e-3, weight_decay=1e-4)
    sched_cnn3 = optim.lr_scheduler.CosineAnnealingLR(opt_cnn3, T_max=10)
    m_cnn3, h_cnn3 = train_model(
        m_cnn3, cnn_train_loader, cnn_val_loader, criterion, opt_cnn3,
        scheduler=sched_cnn3, num_epochs=10, device=device, model_name="CNN Iter 1.3 (Doble Conv + GAP)"
    )
    histories['cnn_iter3'] = h_cnn3
    models_dict['cnn_iter3'] = m_cnn3

    # --- MODELO 2: VGG-16 FEATURE EXTRACTOR ---
    # Iter 2.1: Standard head
    m_vgg_fe1 = build_vgg16_feature_extractor(variant="standard")
    opt_vgg_fe1 = optim.Adam(filter(lambda p: p.requires_grad, m_vgg_fe1.parameters()), lr=1e-3)
    m_vgg_fe1, h_vgg_fe1 = train_model(
        m_vgg_fe1, vgg_train_loader, vgg_val_loader, criterion, opt_vgg_fe1,
        num_epochs=6, device=device, model_name="VGG-16 FE Iter 2.1 (Standard Head)"
    )
    histories['vgg_fe_iter1'] = h_vgg_fe1
    models_dict['vgg_fe_iter1'] = m_vgg_fe1

    # Iter 2.2: Compact head
    m_vgg_fe2 = build_vgg16_feature_extractor(variant="compact")
    opt_vgg_fe2 = optim.Adam(filter(lambda p: p.requires_grad, m_vgg_fe2.parameters()), lr=1e-3)
    m_vgg_fe2, h_vgg_fe2 = train_model(
        m_vgg_fe2, vgg_train_loader, vgg_val_loader, criterion, opt_vgg_fe2,
        num_epochs=6, device=device, model_name="VGG-16 FE Iter 2.2 (Compact Head)"
    )
    histories['vgg_fe_iter2'] = h_vgg_fe2
    models_dict['vgg_fe_iter2'] = m_vgg_fe2

    # Iter 2.3: Regularized head + StepLR
    m_vgg_fe3 = build_vgg16_feature_extractor(variant="regularized")
    opt_vgg_fe3 = optim.AdamW(filter(lambda p: p.requires_grad, m_vgg_fe3.parameters()), lr=5e-4, weight_decay=1e-4)
    sched_vgg_fe3 = optim.lr_scheduler.StepLR(opt_vgg_fe3, step_size=3, gamma=0.5)
    m_vgg_fe3, h_vgg_fe3 = train_model(
        m_vgg_fe3, vgg_train_loader, vgg_val_loader, criterion, opt_vgg_fe3,
        scheduler=sched_vgg_fe3, num_epochs=6, device=device, model_name="VGG-16 FE Iter 2.3 (BN + StepLR)"
    )
    histories['vgg_fe_iter3'] = h_vgg_fe3
    models_dict['vgg_fe_iter3'] = m_vgg_fe3

    # --- MODELO 3: VGG-16 FINE-TUNING ---
    # Iter 3.1: Descongelar Bloque 5
    m_vgg_ft1 = build_vgg16_fine_tuning(unfreeze_blocks="block5", head_type="compact")
    param_groups_ft1 = get_optimizer_param_groups(m_vgg_ft1, lr_backbone=1e-4, lr_classifier=1e-3)
    opt_vgg_ft1 = optim.Adam(param_groups_ft1)
    m_vgg_ft1, h_vgg_ft1 = train_model(
        m_vgg_ft1, vgg_train_loader, vgg_val_loader, criterion, opt_vgg_ft1,
        num_epochs=6, device=device, model_name="VGG-16 FT Iter 3.1 (Bloque 5, LR=1e-4)"
    )
    histories['vgg_ft_iter1'] = h_vgg_ft1
    models_dict['vgg_ft_iter1'] = m_vgg_ft1

    # Iter 3.2: Descongelar Bloques 4 y 5
    m_vgg_ft2 = build_vgg16_fine_tuning(unfreeze_blocks="blocks4_5", head_type="compact")
    param_groups_ft2 = get_optimizer_param_groups(m_vgg_ft2, lr_backbone=5e-5, lr_classifier=1e-3)
    opt_vgg_ft2 = optim.Adam(param_groups_ft2)
    m_vgg_ft2, h_vgg_ft2 = train_model(
        m_vgg_ft2, vgg_train_loader, vgg_val_loader, criterion, opt_vgg_ft2,
        num_epochs=6, device=device, model_name="VGG-16 FT Iter 3.2 (Bloques 4 y 5)"
    )
    histories['vgg_ft_iter2'] = h_vgg_ft2
    models_dict['vgg_ft_iter2'] = m_vgg_ft2

    # Iter 3.3: Descongelar Bloque 5 con LR bajo y Cosine Annealing
    m_vgg_ft3 = build_vgg16_fine_tuning(unfreeze_blocks="block5", head_type="compact")
    param_groups_ft3 = get_optimizer_param_groups(m_vgg_ft3, lr_backbone=2e-5, lr_classifier=5e-4)
    opt_vgg_ft3 = optim.AdamW(param_groups_ft3)
    sched_vgg_ft3 = optim.lr_scheduler.CosineAnnealingLR(opt_vgg_ft3, T_max=6)
    m_vgg_ft3, h_vgg_ft3 = train_model(
        m_vgg_ft3, vgg_train_loader, vgg_val_loader, criterion, opt_vgg_ft3,
        scheduler=sched_vgg_ft3, num_epochs=6, device=device, model_name="VGG-16 FT Iter 3.3 (Bloque 5, LR=2e-5 + Cosine)"
    )
    histories['vgg_ft_iter3'] = h_vgg_ft3
    models_dict['vgg_ft_iter3'] = m_vgg_ft3

    # Guardar gráfico de las 9 curvas de pérdida
    plot_training_loss_curves(histories, save_path="docs/figures/loss_curves_9_iterations.png")

    # --------------------------------------------------------------------------
    # 4. Identificación de los Mejores Modelos
    # --------------------------------------------------------------------------
    # Seleccionamos el mejor de cada familia según mejor Val F1
    best_cnn_key = max(['cnn_iter1', 'cnn_iter2', 'cnn_iter3'], key=lambda k: histories[k]['best_val_f1'])
    best_vgg_fe_key = max(['vgg_fe_iter1', 'vgg_fe_iter2', 'vgg_fe_iter3'], key=lambda k: histories[k]['best_val_f1'])
    best_vgg_ft_key = max(['vgg_ft_iter1', 'vgg_ft_iter2', 'vgg_ft_iter3'], key=lambda k: histories[k]['best_val_f1'])

    print(f"\nMejor CNN: {best_cnn_key} (Val F1: {histories[best_cnn_key]['best_val_f1']:.4f})")
    print(f"Mejor VGG FE: {best_vgg_fe_key} (Val F1: {histories[best_vgg_fe_key]['best_val_f1']:.4f})")
    print(f"Mejor VGG FT: {best_vgg_ft_key} (Val F1: {histories[best_vgg_ft_key]['best_val_f1']:.4f})")

    best_models = {
        'cnn': models_dict[best_cnn_key],
        'vgg_fe': models_dict[best_vgg_fe_key],
        'vgg_ft': models_dict[best_vgg_ft_key]
    }
    best_histories = {
        'cnn': histories[best_cnn_key],
        'vgg_fe': histories[best_vgg_fe_key],
        'vgg_ft': histories[best_vgg_ft_key]
    }

    # Gráfica obligatoria (1): Accuracy vs Tiempo acumulado
    plot_val_accuracy_vs_cumulative_time(best_histories, save_path="docs/figures/val_acc_vs_time.png")

    # --------------------------------------------------------------------------
    # 5. Evaluación Final en Test Set
    # --------------------------------------------------------------------------
    print("\n[Paso 5] Evaluando los mejores modelos sobre el Test Set Oficial...")
    test_results = {}

    cnn_test_metrics = evaluate(best_models['cnn'], cnn_test_loader, criterion, device)
    cnn_test_metrics['display_name'] = f"CNN ({best_cnn_key})"
    test_results['cnn'] = cnn_test_metrics

    vgg_fe_test_metrics = evaluate(best_models['vgg_fe'], vgg_test_loader, criterion, device)
    vgg_fe_test_metrics['display_name'] = f"VGG-16 FE ({best_vgg_fe_key})"
    test_results['vgg_fe'] = vgg_fe_test_metrics

    vgg_ft_test_metrics = evaluate(best_models['vgg_ft'], vgg_test_loader, criterion, device)
    vgg_ft_test_metrics['display_name'] = f"VGG-16 FT ({best_vgg_ft_key})"
    test_results['vgg_ft'] = vgg_ft_test_metrics

    for k in ['cnn', 'vgg_fe', 'vgg_ft']:
        m = test_results[k]
        print(f"Test [{m['display_name']}] -> Acc: {m['accuracy']*100:.2f}%, "
              f"Precision: {m['precision']:.4f}, Recall: {m['recall']:.4f}, F1: {m['f1']:.4f}")

    # Guardar matrices de confusión
    plot_confusion_matrices(test_results, CLASSES, save_path="docs/figures/confusion_matrices.png")

    # --------------------------------------------------------------------------
    # 6. Experimento 4.1: Efecto de la Cantidad de Datos (10% de datos)
    # --------------------------------------------------------------------------
    print("\n[Paso 6] Ejecutando Experimento 4.1: Entrenamiento con el 10% de datos...")
    cnn_train_10_loader, _, _ = get_dataloaders(
        model_type="cnn", batch_size=128, cifar_stats=cifar_stats, use_10_percent=True
    )
    vgg_train_10_loader, _, _ = get_dataloaders(
        model_type="vgg", batch_size=128, vgg_img_size=128, use_10_percent=True
    )

    # Reentrenar versiones idénticas
    # CNN 10%
    if best_cnn_key == 'cnn_iter3':
        m_cnn_10 = EnhancedCNN(num_classes=10)
    elif best_cnn_key == 'cnn_iter2':
        m_cnn_10 = RegularizedCNN(num_classes=10)
    else:
        m_cnn_10 = BaselineCNN(num_classes=10)
    opt_cnn_10 = optim.AdamW(m_cnn_10.parameters(), lr=1e-3, weight_decay=1e-4)
    m_cnn_10, _ = train_model(m_cnn_10, cnn_train_10_loader, cnn_val_loader, criterion, opt_cnn_10,
                              num_epochs=10, device=device, model_name=f"CNN (10% Datos)")
    cnn_10_test = evaluate(m_cnn_10, cnn_test_loader, criterion, device)

    # VGG FE 10%
    fe_var = "compact" if "iter2" in best_vgg_fe_key else ("regularized" if "iter3" in best_vgg_fe_key else "standard")
    m_vgg_fe_10 = build_vgg16_feature_extractor(variant=fe_var)
    opt_vgg_fe_10 = optim.Adam(filter(lambda p: p.requires_grad, m_vgg_fe_10.parameters()), lr=1e-3)
    m_vgg_fe_10, _ = train_model(m_vgg_fe_10, vgg_train_10_loader, vgg_val_loader, criterion, opt_vgg_fe_10,
                                 num_epochs=6, device=device, model_name=f"VGG FE (10% Datos)")
    vgg_fe_10_test = evaluate(m_vgg_fe_10, vgg_test_loader, criterion, device)

    # VGG FT 10%
    ft_blocks = "blocks4_5" if "iter2" in best_vgg_ft_key else "block5"
    m_vgg_ft_10 = build_vgg16_fine_tuning(unfreeze_blocks=ft_blocks, head_type="compact")
    param_groups_ft_10 = get_optimizer_param_groups(m_vgg_ft_10, lr_backbone=2e-5, lr_classifier=5e-4)
    opt_vgg_ft_10 = optim.AdamW(param_groups_ft_10)
    m_vgg_ft_10, _ = train_model(m_vgg_ft_10, vgg_train_10_loader, vgg_val_loader, criterion, opt_vgg_ft_10,
                                 num_epochs=6, device=device, model_name=f"VGG FT (10% Datos)")
    vgg_ft_10_test = evaluate(m_vgg_ft_10, vgg_test_loader, criterion, device)

    reduced_test_results = {
        'cnn': cnn_10_test,
        'vgg_fe': vgg_fe_10_test,
        'vgg_ft': vgg_ft_10_test
    }

    print("\nResultados Comparativos al 10% de Datos:")
    for k in ['cnn', 'vgg_fe', 'vgg_ft']:
        print(f"[{k}] 100% Datos: Acc={test_results[k]['accuracy']*100:.2f}%, F1={test_results[k]['f1']:.4f} | "
              f"10% Datos: Acc={reduced_test_results[k]['accuracy']*100:.2f}%, F1={reduced_test_results[k]['f1']:.4f}")

    # Guardar gráfico comparativo de reducción de datos
    plot_data_reduction_comparison(test_results, reduced_test_results,
                                   save_path="docs/figures/data_reduction_comparison.png")

    # --------------------------------------------------------------------------
    # 7. Medición de Cómputo, Latencia y Tabla Comparativa
    # --------------------------------------------------------------------------
    print("\n[Paso 7] Midiendo FLOPs, MACs y Latencia en GPU...")
    # Formatos de entrada
    cnn_input_size = (1, 3, 32, 32)
    vgg_input_size = (1, 3, 128, 128)

    # FLOPs y MACs
    cnn_cost = compute_model_flops_and_macs(best_models['cnn'], cnn_input_size)
    vgg_fe_cost = compute_model_flops_and_macs(best_models['vgg_fe'], vgg_input_size)
    vgg_ft_cost = compute_model_flops_and_macs(best_models['vgg_ft'], vgg_input_size)

    # Latencia en GPU
    cnn_lat, cnn_lat_std = measure_gpu_latency(best_models['cnn'], cnn_input_size, device=device)
    vgg_fe_lat, vgg_fe_lat_std = measure_gpu_latency(best_models['vgg_fe'], vgg_input_size, device=device)
    vgg_ft_lat, vgg_ft_lat_std = measure_gpu_latency(best_models['vgg_ft'], vgg_input_size, device=device)

    # Estimación de FLOPs de entrenamiento
    num_train_imgs = 45000
    cnn_train_flops = estimate_training_flops(cnn_cost['flops'], num_train_imgs, epochs=10, strategy="scratch")
    vgg_fe_train_flops = estimate_training_flops(vgg_fe_cost['flops'], num_train_imgs, epochs=6, strategy="feature_extractor")
    vgg_ft_train_flops = estimate_training_flops(vgg_ft_cost['flops'], num_train_imgs, epochs=6, strategy="fine_tuning_block5")

    comparison_data = {
        'cnn': {
            'display_name': 'CNN desde Cero',
            'variant': best_cnn_key,
            'total_params': cnn_cost['total_params'],
            'trainable_params': cnn_cost['trainable_params'],
            'macs_per_img': cnn_cost['macs'],
            'flops_per_img': cnn_cost['flops'],
            'inference_gflops': cnn_cost['flops'] / 1e9,
            'latency_ms': cnn_lat,
            'latency_std_ms': cnn_lat_std,
            'time_per_epoch_s': float(np.mean(best_histories['cnn']['epoch_times'])),
            'epochs_to_best': best_histories['cnn']['best_epoch'],
            'total_training_time_s': best_histories['cnn']['total_time_seconds'],
            'peak_memory_mb': best_histories['cnn']['peak_memory_mb'],
            'total_train_tflops': cnn_train_flops / 1e12,
            'test_accuracy': test_results['cnn']['accuracy'],
            'test_precision': test_results['cnn']['precision'],
            'test_recall': test_results['cnn']['recall'],
            'test_f1': test_results['cnn']['f1'],
            'test_acc_10pct': reduced_test_results['cnn']['accuracy'],
            'test_f1_10pct': reduced_test_results['cnn']['f1']
        },
        'vgg_fe': {
            'display_name': 'VGG-16 Feature Extractor',
            'variant': best_vgg_fe_key,
            'total_params': vgg_fe_cost['total_params'],
            'trainable_params': vgg_fe_cost['trainable_params'],
            'macs_per_img': vgg_fe_cost['macs'],
            'flops_per_img': vgg_fe_cost['flops'],
            'inference_gflops': vgg_fe_cost['flops'] / 1e9,
            'latency_ms': vgg_fe_lat,
            'latency_std_ms': vgg_fe_lat_std,
            'time_per_epoch_s': float(np.mean(best_histories['vgg_fe']['epoch_times'])),
            'epochs_to_best': best_histories['vgg_fe']['best_epoch'],
            'total_training_time_s': best_histories['vgg_fe']['total_time_seconds'],
            'peak_memory_mb': best_histories['vgg_fe']['peak_memory_mb'],
            'total_train_tflops': vgg_fe_train_flops / 1e12,
            'test_accuracy': test_results['vgg_fe']['accuracy'],
            'test_precision': test_results['vgg_fe']['precision'],
            'test_recall': test_results['vgg_fe']['recall'],
            'test_f1': test_results['vgg_fe']['f1'],
            'test_acc_10pct': reduced_test_results['vgg_fe']['accuracy'],
            'test_f1_10pct': reduced_test_results['vgg_fe']['f1']
        },
        'vgg_ft': {
            'display_name': 'VGG-16 Fine-Tuning',
            'variant': best_vgg_ft_key,
            'total_params': vgg_ft_cost['total_params'],
            'trainable_params': vgg_ft_cost['trainable_params'],
            'macs_per_img': vgg_ft_cost['macs'],
            'flops_per_img': vgg_ft_cost['flops'],
            'inference_gflops': vgg_ft_cost['flops'] / 1e9,
            'latency_ms': vgg_ft_lat,
            'latency_std_ms': vgg_ft_lat_std,
            'time_per_epoch_s': float(np.mean(best_histories['vgg_ft']['epoch_times'])),
            'epochs_to_best': best_histories['vgg_ft']['best_epoch'],
            'total_training_time_s': best_histories['vgg_ft']['total_time_seconds'],
            'peak_memory_mb': best_histories['vgg_ft']['peak_memory_mb'],
            'total_train_tflops': vgg_ft_train_flops / 1e12,
            'test_accuracy': test_results['vgg_ft']['accuracy'],
            'test_precision': test_results['vgg_ft']['precision'],
            'test_recall': test_results['vgg_ft']['recall'],
            'test_f1': test_results['vgg_ft']['f1'],
            'test_acc_10pct': reduced_test_results['vgg_ft']['accuracy'],
            'test_f1_10pct': reduced_test_results['vgg_ft']['f1']
        }
    }

    # Gráfica obligatoria (2): F1 vs FLOPs
    plot_f1_vs_computational_cost(comparison_data, save_path="docs/figures/f1_vs_flops.png")

    # Guardar todos los resultados en un archivo JSON estructurado
    # Para JSON, eliminamos arrays numpy y predicciones pesadas
    for k in histories:
        histories[k]['model_name'] = str(histories[k]['model_name'])

    summary_output = {
        'hardware': {
            'device': str(device),
            'gpu_name': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU',
            'total_memory_gb': float(torch.cuda.get_device_properties(0).total_memory / (1024**3)) if device.type == 'cuda' else 0.0
        },
        'cifar10_stats': {
            'mean': cifar_mean,
            'std': cifar_std
        },
        'comparison_table': comparison_data,
        'all_iterations': {
            k: {
                'model_name': histories[k]['model_name'],
                'train_loss': histories[k]['train_loss'],
                'val_loss': histories[k]['val_loss'],
                'val_accuracy': histories[k]['val_accuracy'],
                'val_precision': histories[k]['val_precision'],
                'val_recall': histories[k]['val_recall'],
                'val_f1': histories[k]['val_f1'],
                'epoch_times': histories[k]['epoch_times'],
                'total_time_seconds': histories[k]['total_time_seconds'],
                'peak_memory_mb': histories[k]['peak_memory_mb'],
                'best_epoch': histories[k]['best_epoch'],
                'best_val_f1': histories[k]['best_val_f1']
            } for k in histories
        }
    }

    with open("results/experiment_results.json", "w") as f:
        json.dump(summary_output, f, indent=2)

    print("\nResultados consolidados guardados exitosamente en 'results/experiment_results.json'")
    print("Todas las figuras generadas en 'docs/figures/'")
    print("=" * 80)

if __name__ == "__main__":
    main()
