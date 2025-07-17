import numpy as np

def softclip(x):
    """
    As in Vasist+23
    """
    scaled_flux = x/(1+abs(x/100))

    return scaled_flux

def log(x):
    
    return np.log10(x)

