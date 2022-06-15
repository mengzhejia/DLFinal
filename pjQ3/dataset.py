import torchvision
import torchvision.transforms as transforms
import torch
import numpy as np


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes=1, length=8):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

def cifar100_dataset(CIFAR_PATH, train_batch_size :int = 128, test_batch_size: int = 128, num_workers : int = 2, cutout :bool =  False):
    mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
    std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    if cutout:
        transform_train.transforms.append(Cutout(length = 8))

    transform_test = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])

    cifar100_training = torchvision.datasets.CIFAR100(root=CIFAR_PATH, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(cifar100_training, batch_size = train_batch_size, shuffle=True, num_workers=num_workers)

    cifar100_testing = torchvision.datasets.CIFAR100(root=CIFAR_PATH, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(cifar100_testing, batch_size = test_batch_size, shuffle=False, num_workers=num_workers)
    return  trainloader,testloader
