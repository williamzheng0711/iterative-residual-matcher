import numpy as np

def pdf_Rayleigh(scale, x):
    return x*np.exp(-x**2 /(2*scale**2) ) / scale**2
