import cupy as cp
from cupyx.scipy import ndimage
from cupyx.scipy.ndimage import map_coordinates

def rotate_arr(N, arr, angle):
    cen = N // 2
    qh, qk = cp.indices((N, N))
    qh -= cen
    qk -= cen
    qh_rot = cp.cos(angle) * qh - cp.sin(angle) * qk
    qk_rot = cp.sin(angle) * qh + cp.cos(angle) * qk
    coords = cp.array([qh_rot + cen, qk_rot + cen])
    rotated_arr = map_coordinates(arr, coords, order=1, mode='nearest')
    return rotated_arr

def do_fft(obj):
    return cp.fft.fftshift(cp.fft.fftn(cp.fft.ifftshift(obj)))

def do_ifft(ftobj):
    return cp.fft.fftshift(cp.fft.ifftn(cp.fft.ifftshift(ftobj)))

def shrinkwrap(N, ftobj, pixels, sig):
    invsuppmask = cp.ones((N,) * 2, dtype=cp.bool_)
    amodel = cp.real(do_ifft(ftobj.reshape((N,) * 2)))
    samodel = ndimage.gaussian_filter(amodel, sig)
    thresh = cp.quantile(samodel, (samodel.size - pixels) / samodel.size)
    invsuppmask = samodel < thresh
    amodel[invsuppmask] = 0
    return do_fft(amodel)
