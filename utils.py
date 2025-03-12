import numpy as np
from scipy.ndimage import map_coordinates

def rotate_arr(N, arr, angle):
    cen = N // 2
    qh, qk = np.indices((N, N))
    qh -= cen
    qk -= cen
    qh_rot = np.cos(angle) * qh - np.sin(angle) * qk
    qk_rot = np.sin(angle) * qh + np.cos(angle) * qk
    coords = np.array([qh_rot + cen, qk_rot + cen])
    rotated_arr = map_coordinates(arr, coords, order=1, mode='nearest')
    return rotated_arr

def do_fft(obj):
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(obj)))

def do_ifft(ftobj):
    return np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(ftobj)))

def get_vals(array, cen, h, k):
    qh = h + cen
    qk = k + cen
    return array[qh, qk]

def init_tobj(N, pixels):
    obj = np.zeros((N, N))
    cen = (N // 2, N // 2)
    y, x = np.ogrid[:N, :N]
    dcen = np.sqrt((x - cen[1])**2 + (y - cen[0])**2)
    idx = np.argsort(dcen.ravel())[:pixels]
    obj.ravel()[idx] = np.random.rand(pixels)
    return obj
