# loader.py
import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import config

def get_loaders(data_dir=None):
    """
    Build train/val/test DataLoaders from an ImageFolder-style directory
    (must contain train/, val/, test/ subfolders, one folder per class).

    data_dir: path to a specific dataset split (e.g. one of the 5 MS_Dataset
              splits). Defaults to config.DATA_DIR for backward compatibility.
    """
    data_dir = data_dir or config.DATA_DIR

    transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])



    train_set = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    val_set   = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)
    test_set  = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)

    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=config.BATCH_SIZE)
    test_loader  = DataLoader(test_set, batch_size=config.BATCH_SIZE)

    return train_loader, val_loader, test_loader, train_set.classes
