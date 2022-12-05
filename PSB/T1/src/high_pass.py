import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.core.umath import (
    pi, cos, sin,
    ) 


def window_hamming_high_pass(signal, passband_edge_frequency, transition_width, sampling_frequency, duration):
    delta_f = transition_width / sampling_frequency
    M = int(sampling_frequency * duration)
    N = round(3.3/delta_f)
    n = round((N-1)/2 + 1)
    h_d = np.zeros(n)
    w = np.zeros(n)
    h = np.zeros(n)
    fc_norm = (passband_edge_frequency + transition_width/2) / sampling_frequency

    # Calculating the low pass filter kernel
    for i in range(0, n):
        if i == 0:
            h_d = 1 - 2 * fc_norm
            w =  0.54 + 0.46*cos(2*pi*i/(N))
        else:
            h_d = - 2 * fc_norm * sin(2*pi*i*fc_norm)/(2*pi*i*fc_norm)
            w =  0.54 + 0.46*cos(2*pi*i/(N))
        h[i] =  h_d * w
    
    # To make the causal filter
    h_list = h.tolist()
    h_rev = h.tolist()
    h_rev.reverse()
    h_rev.extend(h_list[1:])
    kernel = h_rev
    
    # Convolve the input signal and filter the kernel
    y = np.zeros(M)
    for j in range(N, M):
        for i in range(0, N):
            y[j] += signal[j-i]*kernel[i]
    return y