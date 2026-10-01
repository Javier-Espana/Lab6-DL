"""
Módulo de preparación, exploración y carga de datos para CIFAR-10.
Laboratorio #6 - Transfer Learning y Fine-Tuning.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedShuffleSplit

CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_raw_cifar10(root="data"):
    """Descarga o carga los datasets crudos de CIFAR-10 sin transformaciones complejas."""
    train_raw = datasets.CIFAR10(root=root, train=True, download=True)
    test_raw = datasets.CIFAR10(root=root, train=False, download=True)
    return train_raw, test_raw

def compute_dataset_stats(train_raw):
    """Calcula la media y desviación estándar por canal del conjunto de entrenamiento."""
    # train_raw.data es uint8 de shape (50000, 32, 32, 3)
    data = train_raw.data.astype(np.float32) / 255.0
    mean = np.mean(data, axis=(0, 1, 2)).tolist()
    std = np.std(data, axis=(0, 1, 2)).tolist()
    return mean, std

class TransformedSubset(torch.utils.data.Dataset):
    """Aplica una transformación específica a un subconjunto de datos."""
    def __init__(self, base_dataset, indices, transform=None):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.base_dataset[self.indices[idx]]
        # Si el base_dataset no tiene transform, img es PIL Image
        if self.transform:
            img = self.transform(img)
        return img, label

def get_stratified_split_indices(targets, train_size=45000, val_size=5000, seed=42):
    """Genera índices estratificados para dividir el entrenamiento en train y val."""
    targets = np.array(targets)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, train_size=train_size, random_state=seed)
    train_idx, val_idx = next(sss.split(np.zeros(len(targets)), targets))
    return train_idx, val_idx

def get_subset_10percent_indices(train_indices, targets, seed=42):
    """Extrae un subconjunto del 10% (4,500 imágenes) preservando el balance estratificado."""
    sub_targets = np.array(targets)[train_indices]
    sss = StratifiedShuffleSplit(n_splits=1, train_size=0.10, random_state=seed)
    sampled_rel_idx, _ = next(sss.split(np.zeros(len(sub_targets)), sub_targets))
    sampled_idx = np.array(train_indices)[sampled_rel_idx]
    return sampled_idx

def get_transforms(model_type="cnn", cifar_stats=None, vgg_img_size=128):
    """
    Retorna pipelines de transformación para entrenamiento y validación/test.
    
    Args:
        model_type: 'cnn' o 'vgg'
        cifar_stats: tupla (mean, std) para CIFAR-10
        vgg_img_size: resolución de entrada para VGG (ej. 128)
    """
    if cifar_stats is None:
        cifar_stats = ([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
    cifar_mean, cifar_std = cifar_stats

    if model_type == "cnn":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar_mean, std=cifar_std)
        ])
        val_test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar_mean, std=cifar_std)
        ])
    elif model_type == "vgg":
        train_transform = transforms.Compose([
            transforms.Resize((vgg_img_size, vgg_img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        val_test_transform = transforms.Compose([
            transforms.Resize((vgg_img_size, vgg_img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        raise ValueError(f"model_type desconocido: {model_type}")

    return train_transform, val_test_transform

def get_dataloaders(root="data", model_type="cnn", batch_size=128, num_workers=4, 
                    cifar_stats=None, vgg_img_size=128, use_10_percent=False, seed=42):
    """
    Construye y retorna los DataLoaders de entrenamiento, validación y test.
    """
    # Cargamos el dataset crudo sin transformaciones fijas para poder usar TransformedSubset
    raw_train = datasets.CIFAR10(root=root, train=True, download=True)
    raw_test = datasets.CIFAR10(root=root, train=False, download=True)

    train_idx, val_idx = get_stratified_split_indices(raw_train.targets, seed=seed)
    if use_10_percent:
        train_idx = get_subset_10percent_indices(train_idx, raw_train.targets, seed=seed)

    train_tf, val_tf = get_transforms(model_type=model_type, cifar_stats=cifar_stats, vgg_img_size=vgg_img_size)

    train_dataset = TransformedSubset(raw_train, train_idx, transform=train_tf)
    val_dataset = TransformedSubset(raw_train, val_idx, transform=val_tf)
    test_dataset = TransformedSubset(raw_test, list(range(len(raw_test))), transform=val_tf)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
