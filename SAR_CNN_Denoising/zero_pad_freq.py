import numpy as np

def zero_pad_freq(sublook, target_size):
    """
    Zero-pad a sublook in the frequency domain to match the target size.
    
    :param sublook: 2D numpy array representing the sublook in frequency domain.
    :param target_size: tuple of desired (rows, columns).
    
    :return: Zero-padded sublook in the frequency domain.
    """
    
    padded_sublook = np.zeros(target_size, dtype=sublook.dtype)
    
    rows_diff = target_size[0] - sublook.shape[0]
    cols_diff = target_size[1] - sublook.shape[1]
    
    start_row = rows_diff // 2
    start_col = cols_diff // 2
    
    padded_sublook[start_row:start_row+sublook.shape[0], start_col:start_col+sublook.shape[1]] = sublook
    
    return padded_sublook