import numpy as np

def custom_bytescale(data, cmin=None, cmax=None, high=255, low=0):
    if data.dtype == np.uint8:
        return data

    if high > 255:
        raise ValueError("`high` should be less than or equal to 255.")
    if low < 0:
        raise ValueError("`low` should be greater than or equal to 0.")
    if high < low:
        raise ValueError("`high` should be greater than or equal to `low`.")

    dmin = data.min() if cmin is None else cmin
    dmax = data.max() if cmax is None else cmax

    cscale = dmax - dmin
    scale = float(high - low) / cscale
    bytedata = (data - dmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype(np.uint8)