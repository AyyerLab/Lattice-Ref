import random
from scipy.ndimage import map_coordinates
from scipy import ndimage
from utils import do_fft, do_ifft

num_frames = 100
N = 101
cen = N // 2
qh, qk = np.indices((N, N))
qk -= cen
qh -= cen

def target_obj(N):
     mask = np.zeros((N, N), dtype='f8')
     with mrcfile.open('/scratch/mallabhi/lattice_ref/data/PS1_map.mrc', permissive=True) as mrc:
         ps1 = mrc.data
     ps1_proj= ps1.sum(0)
     pad_h = N - ps1_proj.shape[0]
     pad_w = N - ps1_proj.shape[1]
     pad_top = pad_h // 2
     pad_bottom = pad_h - pad_top
     pad_left = pad_w // 2
     pad_right = pad_w - pad_left
     padded_ps1_proj = np.pad(ps1_proj, ((pad_top, pad_bottom),
                                         (pad_left, pad_right)),
                                         mode='constant', constant_values=0)
     mask = padded_ps1_proj
     return mask, do_fft(mask)

def unit_cell(N):
    ind = np.arange(N) - N//2
    rad = np.sqrt(ind[:,None]**2 + ind[None,:]**2)
    np.random.seed(42)
    random_ = np.random.rand(N, N)
    random_ = random_ > 0.7
    unitc = ndimage.gaussian_filter(random_.astype(float), sigma=(0.7, 0.7), mode='wrap') * (rad<30)
    return unitc, 100 * do_fft(unitc)


def rotate_ft(N, ftobj, angle):
    cen = N // 2
    qh, qk = np.indices((N, N))
    qk -= cen
    qh -= cen
    qh_rot = np.cos(angle) * qh - np.sin(angle) * qk
    qk_rot = np.sin(angle) * qh + np.cos(angle) * qk
    coords = np.array([qh_rot + cen, qk_rot + cen])
    rotated_ft = map_coordinates(ftobj, coords, order=0, mode='nearest')
    return rotated_ft

def phase_ramp(shiftx, shifty):
    return np.exp(1j * 2.0 * np.pi * (qh * shiftx + qk * shifty))

def _getvals(array, h, k):
    qh = h + cen
    qk = k + cen
    return array[qh, qk]

tobj, ftobj = target_obj(N)
unitc, funitc = unit_cell(N)

np.random.seed(42)
dx_vals = np.random.uniform(0.01,1, size=num_frames)
dy_vals = np.random.uniform(0.01,1, size=num_frames)
fluence_vals = np.random.uniform(0.01,1, size=num_frames)
angle_vals = np.random.uniform(0,1, size=num_frames) * 2 * np.pi

intensities = np.zeros((num_frames, N, N))
for i in range(num_frames):
    pramp = phase_ramp(dx_vals[i], dy_vals[i])
    intens =  np.abs(funitc + fluence_vals[i] * rotate_ft(N, ftobj, angle_vals[i]) * pramp)**2
    intensities[i] = intens
Iobs = intensities

# Algorithm
#Qh, Qk = (1,1)

#ftobj_true_val = _getvals(ftobj, Qh, Qk)
#print("True Value : ", ftobj_true_val)
#funitc_vals = np.zeros(num_frames, dtype=complex)
#intens_vals = np.zeros(num_frames, dtype=float)
#phases = np.zeros(num_frames, dtype=float)

#for i in range(num_frames):
#    angle = angle_vals[i]
#    qh_rot = np.cos(angle) * Qh - np.sin(angle) * Qk
#    qk_rot = np.sin(angle) * Qh + np.cos(angle) * Qk
#    funitc_vals[i] = _getvals(funitc, round(qh_rot), round(qk_rot))
#    intens_vals[i] = _getvals(Iobs[i], round(qh_rot), round(qk_rot))
#    phases[i] = 2 * np.pi * (round(qh_rot) * dx_vals[i] + round(qk_rot) * dy_vals[i])
#pramp = np.exp(1j * phases)


# Algorithm
Qh, Qk = (1,1)

ftobj_true_val = _getvals(ftobj, Qh, Qk)
print("True Value : ", ftobj_true_val)
funitc_vals = np.zeros(num_frames, dtype=complex)
intens_vals = np.zeros((num_frames,4), dtype=float)
phases = np.zeros(num_frames, dtype=float)

for i in range(num_frames):
    angle = angle_vals[i]
    qh_rot = np.cos(angle) * Qh - np.sin(angle) * Qk
    qk_rot = np.sin(angle) * Qh + np.cos(angle) * Qk

    qh_floor, qk_floor = np.floor([qh_rot, qk_rot])
    qh_ceil, qk_ceil = np.ceil([qh_rot, qk_rot])
    frac_qh, frac_qk = qh_rot - qh_floor, qk_rot - qk_floor
    qhks = [(qh_floor, qk_floor),
            (qh_floor, qk_ceil),
            (qh_ceil, qk_floor),
            (qh_ceil, qk_ceil)]
    funitc_vals[i] = [_getvals(funitc, int(qh), int(qk)) for qh, qk in qhks]
    intens_vals[i] = [_getvals(Iobs[i], int(qh), int(qk)) for qh, qk in qhks]
    phases[i] = [2 * np.pi * (int(qh) * dx_vals[i] + int(qk) * dy_vals[i]) for qh, qk in qhks]
pramp = np.exp(1j * phases)


def compute_error_grid(real_range, imag_range, num_frames, funitc_vals, fluence_vals, pramp, intens_vals):
    R = len(real_range)
    I = len(imag_range)
    err_grid = np.zeros((R, I), dtype=float)

    for rr in range(R):
        for cc in range(I):
            ftobj_guess = real_range[rr] + 1j * imag_range[cc]
            model_frames = np.zeros(num_frames, dtype=float)

            for i in range(num_frames):
                model = np.abs(funitc_vals[i] + fluence_vals[i] * ftobj_guess * pramp[i])**2
                model_frames[i] = model

            err = (model_frames - intens_vals)**2
            err_grid[rr, cc] = err.sum()

    return err_grid



#def compute_error_grid(real_range, imag_range):
#    real_grid, imag_grid = np.meshgrid(real_range, imag_range, indexing='ij')
#    ftobj_guess_grid = real_grid + 1j * imag_grid

#    funitc_vals_ = funitc_vals[:, np.newaxis, np.newaxis]
#    fluence_vals_ = fluence_vals[:, np.newaxis, np.newaxis]
#    pramp_ = pramp[:, np.newaxis, np.newaxis]       
#    intens_vals_ = intens_vals[:, np.newaxis, np.newaxis]
    
#    ftobj_guess_grid_ = ftobj_guess_grid[np.newaxis, :, :]
    
#    res = funitc_vals_ + fluence_vals_ * ftobj_guess_grid_ * pramp_
#    Icalc = np.abs(res)**2
#    err = (Icalc - intens_vals_)**2
#    err_grid = err.sum(axis=0)
#    return err_grid

# Coarse search
coarse_size = 100
coarse_real_range = np.linspace(-3000, 3000, coarse_size)
coarse_imag_range = np.linspace(-3000, 3000, coarse_size)

err_grid_coarse = compute_error_grid(coarse_real_range, coarse_imag_range)
min_index_coarse = np.unravel_index(np.argmin(err_grid_coarse), err_grid_coarse.shape)
coarse_best_real = coarse_real_range[min_index_coarse[0]]
coarse_best_imag = coarse_imag_range[min_index_coarse[1]]
coarse_best_guess = coarse_best_real + 1j * coarse_best_imag
print("Initial Coarse best guess:", coarse_best_guess)

# Adaptive refinement
max_refinement_steps = 100
grid_size = 40
initial_range = 20.0
range_decay = 0.5
convergence_threshold = 1e-4

current_best_real = coarse_best_real
current_best_imag = coarse_best_imag
current_range = initial_range
prev_fitted_value = coarse_best_guess

for step in range(max_refinement_steps):
    fine_real_range = np.linspace(current_best_real - current_range, current_best_real + current_range, grid_size)
    fine_imag_range = np.linspace(current_best_imag - current_range, current_best_imag + current_range, grid_size)

    err_grid_fine = compute_error_grid(fine_real_range, fine_imag_range)
    min_index_fine = np.unravel_index(np.argmin(err_grid_fine), err_grid_fine.shape)
    fine_best_real = fine_real_range[min_index_fine[0]]
    fine_best_imag = fine_imag_range[min_index_fine[1]]

    current_best_real = fine_best_real
    current_best_imag = fine_best_imag
    fitted_value = current_best_real + 1j * current_best_imag
    current_range *= range_decay

    improvement =  np.abs(fitted_value - prev_fitted_value)
    print(f"Refinement step {step+1}/{max_refinement_steps}, Best guess: {fitted_value}, Improvement: {improvement}")
    # Check convergence
    if improvement < convergence_threshold:
        print(f"Converged at step {step+1}, improvement below {convergence_threshold}")
        break
    prev_fitted_value = fitted_value

print("True Value :", ftobj_true_val)
print("Fitted value (after adaptive refinement):", fitted_value)



