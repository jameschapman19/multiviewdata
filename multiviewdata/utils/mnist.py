import torchvision
from torchvision import datasets


def load_mnist(mnist_type, train):
    if mnist_type == "MNIST":
        dataset = load_mnist(mnist_type, train)
        datasets.MNIST(
            "../../data",
            train=train,
            download=True,
            transform=torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor()]
            ),
        )

    elif mnist_type == "FashionMNIST":
        dataset = datasets.FashionMNIST(
            "../../data",
            train=train,
            download=True,
            transform=torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor()]
            ),
        )
    elif mnist_type == "KMNIST":
        dataset = datasets.KMNIST(
            "../../data",
            train=train,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                ]
            ),
        )
    else:
        raise ValueError
    return dataset