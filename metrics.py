import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr

def calculate_accuracy(output, gt):
    return np.mean(np.array([psnr(gt[:, seq], output[:, seq], data_range=255) for seq in range(output.shape[1])]))