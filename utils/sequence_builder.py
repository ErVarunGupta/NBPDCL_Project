import numpy as np

def create_sequences(data, seq_len):
    return np.array([
        data[i:i+seq_len]
        for i in range(len(data) - seq_len)
    ])
