from torch.utils import data
from torchvision.datasets import STL10
import torchvision.transforms as transforms

data_transform = transforms.Compose([
    transforms.ToTensor(),
])

# configs
train_validate_split = 0.8
train = STL10(root="./stl10", split="unlabeled", transform=data_transform, download=True)


def get_loaders(batch_size=128, shuffle=True, num_workers=0):
    train_size = int(train_validate_split * len(train))
    validation_size = len(train) - train_size
    train_unlabeled, validation_unlabeled = data.random_split(train, [train_size, validation_size])

    train_loader = data.DataLoader(train_unlabeled, batch_size=batch_size, shuffle=shuffle,
                                   num_workers=num_workers)
    validation_loader = data.DataLoader(validation_unlabeled,
                                        batch_size=batch_size,
                                        shuffle=shuffle,
                                        num_workers=num_workers)
    return train_loader, validation_loader
