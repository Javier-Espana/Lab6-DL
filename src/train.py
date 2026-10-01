"""
Módulo de entrenamiento, validación y benchmarking de modelos en PyTorch.
Laboratorio #6 - Transfer Learning y Fine-Tuning.
"""

import time
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def evaluate(model, data_loader, criterion, device):
    """
    Evalúa el modelo sobre un DataLoader calculando loss y métricas multiclase macro.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    mean_loss = total_loss / len(data_loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='macro', zero_division=0
    )

    return {
        'loss': mean_loss,
        'accuracy': float(acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'preds': all_preds,
        'targets': all_targets
    }

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                scheduler=None, num_epochs=10, device=None, model_name="model"):
    """
    Ejecuta el bucle de entrenamiento completo midiendo con precisión:
    - Tiempo por epoch y tiempo total (usando torch.cuda.synchronize).
    - Memoria pico de GPU (usando torch.cuda.max_memory_allocated).
    - Métricas de validación por epoch.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    # Reiniciar métricas de memoria si estamos en CUDA
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    history = {
        'model_name': model_name,
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'epoch_times': [],
        'cumulative_times': [],
        'peak_memory_mb': 0.0,
        'total_time_seconds': 0.0,
        'best_epoch': 0,
        'best_val_f1': 0.0
    }

    best_val_f1 = -1.0
    best_model_state = None
    cum_time = 0.0

    print(f"\n--- Iniciando entrenamiento de [{model_name}] por {num_epochs} epochs ---")

    for epoch in range(1, num_epochs + 1):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = time.time()

        # Fase de Entrenamiento
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        if scheduler is not None:
            scheduler.step()

        if device.type == 'cuda':
            torch.cuda.synchronize()
        epoch_time = time.time() - start_time
        cum_time += epoch_time

        train_loss = running_loss / len(train_loader.dataset)

        # Fase de Validación
        val_metrics = evaluate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1'])
        history['epoch_times'].append(epoch_time)
        history['cumulative_times'].append(cum_time)

        print(f"Epoch {epoch:02d}/{num_epochs:02d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val Acc: {val_metrics['accuracy']*100:.2f}% | "
              f"Val F1: {val_metrics['f1']:.4f} | "
              f"Tiempo: {epoch_time:.2f}s")

        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            history['best_epoch'] = epoch
            history['best_val_f1'] = best_val_f1
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # Memoria pico al finalizar
    if device.type == 'cuda':
        history['peak_memory_mb'] = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
    else:
        history['peak_memory_mb'] = 0.0

    history['total_time_seconds'] = cum_time

    # Restaurar los mejores pesos para evaluación posterior
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    print(f"-> Entrenamiento completado. Tiempo total: {cum_time:.2f}s. "
          f"Pico de memoria GPU: {history['peak_memory_mb']:.2f} MB. "
          f"Mejor Val F1: {history['best_val_f1']:.4f} (Epoch {history['best_epoch']})")

    return model, history
