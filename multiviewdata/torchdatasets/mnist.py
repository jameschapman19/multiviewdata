import numpy as np
import torch
import torch.utils.data
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms


class SplitMNIST(Dataset):
    """
    Class to generate paired noisy mnist data
    """

    def __init__(
            self,
            root: str,
            mnist_type: str = "MNIST",
            train: bool = True,
            flatten: bool = True,
            download=False,
    ):
        """
        :param root: Root directory of dataset
        :param mnist_type: "MNIST", "FashionMNIST" or "KMNIST"
        :param train: whether this is train or test
        :param flatten: whether to flatten the data into array or use 2d images
        """

        self.dataset = load_mnist(mnist_type, train, root, download)
        self.flatten = flatten

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x_a, label = self.dataset[idx]
        x_b = x_a[:, :, 14:]
        x_a = x_a[:, :, :14]
        if self.flatten:
            x_a = torch.flatten(x_a)
            x_b = torch.flatten(x_b)
        return {"views": (x_a, x_b), "label": label, "index": idx}


class NoisyMNIST(Dataset):
    """
    Class to generate paired noisy mnist data
    """

    def __init__(
            self,
            root: str,
            mnist_type: str = "MNIST",
            train: bool = True,
            flatten: bool = True,
            download=False,
    ):
        """
        :param root: Root directory of dataset
        :param mnist_type: "MNIST", "FashionMNIST" or "KMNIST"
        :param train: whether this is train or test
        :param download: If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        :param flatten: whether to flatten the data into array or use 2d images
        """
        self.dataset = load_mnist(mnist_type, train, root, download)
        self.a_transform = torchvision.transforms.RandomRotation((-45, 45))
        self.b_transform = transforms.Compose(
            [
                transforms.Lambda(_add_mnist_noise),
                transforms.Lambda(self.__threshold_func__),
            ]
        )
        self.targets = self.dataset.targets
        self.filtered_classes = []
        self.filtered_nums = []
        for i in range(10):
            self.filtered_nums.append(np.where(self.targets == i)[0])
        self.flatten = flatten

    def __threshold_func__(self, x):
        x[x > 1] = 1
        return x

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x_a, label = self.dataset[index]
        x_a = self.a_transform(x_a)
        # get random index of image with same class
        random_index = np.random.choice(self.filtered_nums[label])
        x_b = self.b_transform(self.dataset[random_index][0])
        if self.flatten:
            x_a = torch.flatten(x_a)
            x_b = torch.flatten(x_b)
        return {"views": (x_a.float(), x_b.float()), "label": label, "index": index}


class TangledMNIST(Dataset):
    """
    Class to generate paired tangled MNIST dataset
    """

    def __init__(
            self,
            root: str,
            mnist_type: str = "MNIST",
            train: bool = True,
            flatten: bool = True,
            download=False,
    ):
        """
        :param root: Root directory of dataset
        :param mnist_type: "MNIST", "FashionMNIST" or "KMNIST"
        :param train: whether this is train or test
        :param flatten: whether to flatten the data into array or use 2d images
        """
        self.dataset = load_mnist(mnist_type, train, root, download)
        self.transform = torchvision.transforms.RandomRotation((-45, 45))
        self.targets = self.dataset.targets
        self.filtered_classes = []
        self.filtered_nums = []
        for i in range(10):
            self.filtered_nums.append(np.where(self.targets == i)[0])
        self.flatten = flatten

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x_a, label = self.dataset[idx]
        x_a = self.transform(x_a)
        # get random index of image with same class
        random_index = np.random.choice(self.filtered_nums[label])
        x_b = self.transform(self.dataset[random_index][0])
        if self.flatten:
            x_a = torch.flatten(x_a)
            x_b = torch.flatten(x_b)
        return {"views": (x_a, x_b), "label": label, "index": idx}


def _add_mnist_noise(x):
    x = x + torch.rand(28, 28)
    return x


def load_mnist(mnist_type, train, root, download):
    if mnist_type == "MNIST":
        dataset = datasets.MNIST(
            root,
            train=train,
            download=download,
            transform=torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor()]
            ),
        )

    elif mnist_type == "FashionMNIST":
        dataset = datasets.FashionMNIST(
            root,
            train=train,
            download=download,
            transform=torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor()]
            ),
        )
    elif mnist_type == "KMNIST":
        dataset = datasets.KMNIST(
            root,
            train=train,
            download=download,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                ]
            ),
        )
    else:
        raise ValueError
    return dataset


class MNISTSVHN(Dataset):
    """
    We construct a dataset of pairs of MNIST and SVHN such that each pair depicts the same digit class. Each instance of a digit class in either dataset is randomly paired with 20 instances of the same digit class from the other dataset.
    """

    def __init__(
            self,
            root: str,
            max_d: int = 10000,
            dm: int = 10,
            train: bool = True,
            download=False,
            flatten: bool = True,
    ):
        """
        :param root: Root directory of dataset
        :param train: whether this is train or test
        :param flatten: whether to flatten the data into array or use 2d images
        """
        self.flatten = flatten
        self.mnist = load_mnist("MNIST", train, root, download)
        self.svhn = self.load_svhn(train, root, download)
        self.mnist_index, self.svhn_index = self.rand_match_on_idx(max_d=max_d, dm=dm)

    def __len__(self):
        return len(self.mnist_index)

    def __getitem__(self, idx):
        x, label = self.mnist[self.mnist_index[idx]]
        y, label_ = self.svhn[self.svhn_index[idx]]
        assert label == label_, "labels not the same so something is wrong with the dataset"
        if self.flatten:
            x = torch.flatten(x)
            y = torch.flatten(y)
        return {"views": (x, y), "label": label, "index": idx}

    def load_svhn(self, train, root, download):
        if train:
            svhn = datasets.SVHN(root, split='train', download=download,transform=torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor()]
            ),)
        else:
            svhn = datasets.SVHN(root, split='test', download=download,transform=torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor()]
            ),)
        svhn.labels = torch.LongTensor(svhn.labels.squeeze().astype(int)) % 10
        return svhn

    def rand_match_on_idx(self, max_d=10000, dm=10):
        """
        l*: sorted labels
        idx*: indices of sorted labels in original list
        """
        mnist_l, mnist_li = self.mnist.targets.sort()
        svhn_l, svhn_li = self.svhn.labels.sort()
        _idx1, _idx2 = [], []
        for l in mnist_l.unique():  # assuming both have same idxs
            l_idx1, l_idx2 = mnist_li[mnist_l == l], svhn_li[svhn_l == l]
            n = min(l_idx1.size(0), l_idx2.size(0), max_d)
            l_idx1, l_idx2 = l_idx1[:n], l_idx2[:n]
            for _ in range(dm):
                _idx1.append(l_idx1[torch.randperm(n)])
                _idx2.append(l_idx2[torch.randperm(n)])
        return torch.cat(_idx1), torch.cat(_idx2)
