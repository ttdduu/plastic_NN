import numpy as np
import matplotlib.pyplot as plt

path = "/home/tomasdu/repos/trained_models/02yo20f3/gabor_tuning_model_stages_0_0_dwconv_results.npz"
ch, y, x = 0, 100, 140
with np.load(path) as results:
    patch = results[f"patch_{ch}_{y}_{x}"]
    plt.imshow(patch, cmap="RdBu_r")
    plt.colorbar()
    plt.show()
