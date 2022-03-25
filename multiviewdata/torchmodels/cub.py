
# Constants
import torch
from torch import nn
import torch.nn.functional as F
imgChans = 3
fBase = 64

class ImageEncoder(nn.Module):
    """ Generate latent parameters for CUB image data. """

    def __init__(self, latentDim, eta=1e-6):
        super(ImageEncoder, self).__init__()
        self.eta=eta
        modules = [
            # input size: 3 x 128 x 128
            nn.Conv2d(imgChans, fBase, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # input size: 1 x 64 x 64
            nn.Conv2d(fBase, fBase * 2, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 2) x 32 x 32
            nn.Conv2d(fBase * 2, fBase * 4, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 4) x 16 x 16
            nn.Conv2d(fBase * 4, fBase * 8, 4, 2, 1, bias=True),
            nn.ReLU(True)]
        # size: (fBase * 8) x 4 x 4

        self.enc = nn.Sequential(*modules)
        self.c1 = nn.Conv2d(fBase * 8, latentDim, 4, 1, 0, bias=True)
        self.c2 = nn.Conv2d(fBase * 8, latentDim, 4, 1, 0, bias=True)
        # c1, c2 size: latentDim x 1 x 1

    def forward(self, x):
        e = self.enc(x)
        return self.c1(e).squeeze(), F.softplus(self.c2(e)).squeeze() + self.eta


class ImageDecoder(nn.Module):
    """ Generate an image given a sample from the latent space. """

    def __init__(self, latentDim):
        super(ImageDecoder, self).__init__()
        modules = [nn.ConvTranspose2d(latentDim, fBase * 8, 4, 1, 0, bias=True),
                   nn.ReLU(True), ]

        modules.extend([
            nn.ConvTranspose2d(fBase * 8, fBase * 4, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 4) x 16 x 16
            nn.ConvTranspose2d(fBase * 4, fBase * 2, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 2) x 32 x 32
            nn.ConvTranspose2d(fBase * 2, fBase, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase) x 64 x 64
            nn.ConvTranspose2d(fBase, imgChans, 4, 2, 1, bias=True),
            nn.Sigmoid()
            # Output size: 3 x 128 x 128
        ])
        self.dec = nn.Sequential(*modules)

    def forward(self, z):
        z = z.unsqueeze(-1).unsqueeze(-1)  # fit deconv layers
        out = self.dec(z.view(-1, *z.size()[-3:]))
        out = out.view(*z.size()[:-3], *out.size()[1:])
        return out, torch.tensor(0.01).to(z.device)

# Constants
maxSentLen = 32  # max length of any description for birds dataset
minOccur = 3
embeddingDim = 128
lenWindow = 3
fBase = 32
vocabSize = 1590
vocab_path = '../data/cub/oc:{}_sl:{}_s:{}_w:{}/cub.vocab'.format(minOccur, maxSentLen, 300, lenWindow)


# Classes
class SentenceEncoder(nn.Module):
    """ Generate latent parameters for sentence data. """

    def __init__(self, latentDim, eta=1e-6):
        super(SentenceEncoder, self).__init__()
        self.eta=eta
        self.embedding = nn.Embedding(vocabSize, embeddingDim, padding_idx=0)
        self.enc = nn.Sequential(
            # input size: 1 x 32 x 128
            nn.Conv2d(1, fBase, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase),
            nn.ReLU(True),
            # size: (fBase) x 16 x 64
            nn.Conv2d(fBase, fBase * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase * 2),
            nn.ReLU(True),
            # size: (fBase * 2) x 8 x 32
            nn.Conv2d(fBase * 2, fBase * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase * 4),
            nn.ReLU(True),
            # # size: (fBase * 4) x 4 x 16
            nn.Conv2d(fBase * 4, fBase * 4, (1, 4), (1, 2), (0, 1), bias=False),
            nn.BatchNorm2d(fBase * 4),
            nn.ReLU(True),
            # size: (fBase * 8) x 4 x 8
            nn.Conv2d(fBase * 4, fBase * 4, (1, 4), (1, 2), (0, 1), bias=False),
            nn.BatchNorm2d(fBase * 4),
            nn.ReLU(True),
            # size: (fBase * 8) x 4 x 4
        )
        self.c1 = nn.Conv2d(fBase * 4, latentDim, 4, 1, 0, bias=False)
        self.c2 = nn.Conv2d(fBase * 4, latentDim, 4, 1, 0, bias=False)
        # c1, c2 size: latentDim x 1 x 1

    def forward(self, x):
        e = self.enc(self.embedding(x.long()).unsqueeze(1))
        mu, logvar = self.c1(e).squeeze(), self.c2(e).squeeze()
        return mu, F.softplus(logvar) + self.eta


class SentenceDecoder(nn.Module):
    """ Generate a sentence given a sample from the latent space. """

    def __init__(self, latentDim):
        super(SentenceDecoder, self).__init__()
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(latentDim, fBase * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(fBase * 4),
            nn.ReLU(True),
            # size: (fBase * 8) x 4 x 4
            nn.ConvTranspose2d(fBase * 4, fBase * 4, (1, 4), (1, 2), (0, 1), bias=False),
            nn.BatchNorm2d(fBase * 4),
            nn.ReLU(True),
            # size: (fBase * 8) x 4 x 8
            nn.ConvTranspose2d(fBase * 4, fBase * 4, (1, 4), (1, 2), (0, 1), bias=False),
            nn.BatchNorm2d(fBase * 4),
            nn.ReLU(True),
            # size: (fBase * 4) x 8 x 32
            nn.ConvTranspose2d(fBase * 4, fBase * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase * 2),
            nn.ReLU(True),
            # size: (fBase * 2) x 16 x 64
            nn.ConvTranspose2d(fBase * 2, fBase, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fBase),
            nn.ReLU(True),
            # size: (fBase) x 32 x 128
            nn.ConvTranspose2d(fBase, 1, 4, 2, 1, bias=False),
            nn.ReLU(True)
            # Output size: 1 x 64 x 256
        )
        # inverts the 'embedding' module upto one-hotness
        self.toVocabSize = nn.Linear(embeddingDim, vocabSize)

    def forward(self, z):
        z = z.unsqueeze(-1).unsqueeze(-1)  # fit deconv layers
        out = self.dec(z.view(-1, *z.size()[-3:])).view(-1, embeddingDim)

        return self.toVocabSize(out).view(*z.size()[:-3], maxSentLen, vocabSize),
