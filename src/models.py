"""
Definición de arquitecturas de modelos:
1. CNN desde cero (3 variantes)
2. VGG-16 Feature Extractor (3 variantes)
3. VGG-16 Fine-Tuning (3 variantes con distintos bloques descongelados y LRs)
"""

import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

# ==============================================================================
# 1. MODELO: CNN DESDE CERO
# ==============================================================================

class BaselineCNN(nn.Module):
    """
    Iteración 1.1: CNN básica de 3 bloques convolucionales sin BN ni Dropout.
    Resolución de entrada: 32x32x3.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # Bloque 1: 32x32 -> 16x16
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Bloque 2: 16x16 -> 8x8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Bloque 3: 8x8 -> 4x4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class RegularizedCNN(nn.Module):
    """
    Iteración 1.2: CNN de 4 bloques con Batch Normalization y Dropout moderado.
    """
    def __init__(self, num_classes=10, dropout_rate=0.25):
        super().__init__()
        self.features = nn.Sequential(
            # Bloque 1: 32x32 -> 16x16
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate),

            # Bloque 2: 16x16 -> 8x8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate),

            # Bloque 3: 8x8 -> 4x4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate),

            # Bloque 4: 4x4 -> 2x2
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class EnhancedCNN(nn.Module):
    """
    Iteración 1.3: CNN profunda con doble convolución por bloque y Global Average Pooling.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(0.2)
            )

        self.block1 = conv_block(3, 64)    # 32x32 -> 16x16
        self.block2 = conv_block(64, 128)  # 16x16 -> 8x8
        self.block3 = conv_block(128, 256) # 8x8 -> 4x4
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        return self.classifier(x)


# ==============================================================================
# 2. MODELO: VGG-16 FEATURE EXTRACTOR
# ==============================================================================

def build_vgg16_feature_extractor(variant="standard", num_classes=10):
    """
    Construye VGG-16 como Feature Extractor:
    - Carga pesos preentrenados de ImageNet (VGG16_Weights.DEFAULT).
    - Congela todos los parámetros del backbone convolucional (requires_grad = False).
    - Adapta la cabeza clasificadora para 10 clases.
    
    Variantes:
    - 'standard' (Iter 2.1): Reemplaza classifier[6] = Linear(4096, 10).
    - 'compact' (Iter 2.2): Clasificador ligero de 2 capas: Linear(512*4*4, 512) -> ReLU -> Dropout -> Linear(512, 10).
    - 'regularized' (Iter 2.3): Clasificador con Dropout(0.4) y BatchNorm.
    """
    model = vgg16(weights=VGG16_Weights.DEFAULT)

    # Congelar todas las características convolucionales
    for param in model.features.parameters():
        param.requires_grad = False

    # Para resolución 128x128, AdaptiveAvgPool2d((7, 7)) produce mapas de 7x7.
    # El aplanado es 512 * 7 * 7 = 25088.
    if variant == "standard":
        # Congelar también primeras capas del clasificador original
        for param in model.classifier.parameters():
            param.requires_grad = False
        model.classifier[6] = nn.Linear(4096, num_classes)
        for param in model.classifier[6].parameters():
            param.requires_grad = True

    elif variant == "compact":
        model.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )
        for param in model.classifier.parameters():
            param.requires_grad = True

    elif variant == "regularized":
        model.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(512, num_classes)
        )
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Variante desconocida: {variant}")

    return model


# ==============================================================================
# 3. MODELO: VGG-16 FINE-TUNING
# ==============================================================================

def build_vgg16_fine_tuning(unfreeze_blocks="block5", head_type="compact", num_classes=10):
    """
    Construye VGG-16 para Fine-Tuning:
    - Carga pesos preentrenados de ImageNet.
    - Congela bloques iniciales y descongela bloques superiores específicos:
      * 'block5' (Iter 3.1 & 3.3): features[24:] descongelado (Bloque 5).
      * 'blocks4_5' (Iter 3.2): features[17:] descongelado (Bloques 4 y 5).
    - Clasificador totalmente entrenable.
    """
    model = vgg16(weights=VGG16_Weights.DEFAULT)

    # Congelar inicialmente todo el backbone
    for param in model.features.parameters():
        param.requires_grad = False

    # Configurar el clasificador
    if head_type == "compact":
        model.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )
    elif head_type == "standard":
        model.classifier[6] = nn.Linear(4096, num_classes)

    # Descongelar clasificador
    for param in model.classifier.parameters():
        param.requires_grad = True

    # Descongelar bloques según especificación
    if unfreeze_blocks == "block5":
        # Bloque 5 comprende índices 24 a 30 en model.features
        for layer in model.features[24:]:
            for param in layer.parameters():
                param.requires_grad = True
    elif unfreeze_blocks == "blocks4_5":
        # Bloque 4 comprende 17 a 23; Bloque 5 comprende 24 a 30
        for layer in model.features[17:]:
            for param in layer.parameters():
                param.requires_grad = True
    else:
        raise ValueError(f"unfreeze_blocks no soportado: {unfreeze_blocks}")

    return model

def get_optimizer_param_groups(model, lr_backbone=1e-4, lr_classifier=1e-3, weight_decay=1e-4):
    """
    Crea grupos de parámetros para el optimizador con tasas de aprendizaje diferenciales:
    - lr_backbone para las capas convolucionales descongeladas.
    - lr_classifier para la cabeza de clasificación.
    """
    backbone_params = []
    classifier_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "features" in name:
            backbone_params.append(param)
        else:
            classifier_params.append(param)

    param_groups = []
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': lr_backbone, 'weight_decay': weight_decay})
    if classifier_params:
        param_groups.append({'params': classifier_params, 'lr': lr_classifier, 'weight_decay': weight_decay})

    return param_groups

def count_parameters(model):
    """Retorna (parámetros totales, parámetros entrenables)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
