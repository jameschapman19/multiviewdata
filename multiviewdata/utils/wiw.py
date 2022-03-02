import numpy as np


def Generate_Simlex_batch(languages, folder):
    batch = [[], []]
    for i, language in enumerate(languages):
        batch[i] = np.genfromtxt(
            "./data_sample/simlex_data/"
            + folder
            + "/simlex_"
            + language
            + "_batch_vectors.csv",
            delimiter=",",
            dtype=np.float32,
        )
    return batch
