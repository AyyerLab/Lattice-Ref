# Plotting Radial averages : intesn , ftobj, flattice, background

def rad_avg(intens):
    y, x = np.indices(intens.shape)
    r = np.sqrt((x - intens.shape[1]//2)**2 + (y - intens.shape[0]//2)**2).astype(np.int32)
    return np.bincount(r.ravel(), intens.ravel()) / np.maximum(np.bincount(r.ravel()), 1)

def plot_radial_intensity(run_number):
     with h5py.File(f'/scratch/mallabhi/lattice_ref/data/K/dataset_run{run_number:03d}.h5', 'r') as f:
         intens, ftobj, funitc, background = f['intens'][:], f['ftobj'][:], f['funitc'][:], f['background'][()]
     fig, ax = plt.subplots(figsize=(5,4))
     data_list = [intens[0], np.abs(ftobj)**2, np.abs(funitc)**2, background**2]
     labels = [r'$I_{obs}$', r'$|F_{obj}|^2$', r'$|F_{Lattice}|^2$', r'$B^2$']
     styles = ['-', '-', 'None', '-']
     markers = [None, None, 'D', None]

     for data, label, style, marker in zip(data_list, labels, styles, markers):
         ax.plot(np.arange(1, 72), rad_avg(data)[:72], label=label, linestyle=style, marker=marker, markersize=4)

     ax.set_xlabel("Rad")
     ax.set_ylabel("Vals (Arb. Units)")
     ax.set_xlim(0, 50)
     ax.set_yscale("log")
     ax.legend()
     return fig

plot_radial_intensity(21)

# Plot FRC
def compute_rvals_fvals(run_number, iteration_number):
    data_file = f'/scratch/mallabhi/lattice_ref/data/K/dataset_run{run_number:03d}.h5'
    output_file = f'/scratch/mallabhi/lattice_ref/output/output_run{run_number:03d}_{iteration_number}.h5'

    with h5py.File(data_file, 'r') as f:
        ftobj = f['ftobj'][:]
    with h5py.File(output_file, 'r') as f:
        fitted_ftobj = f['fitted_ftobj'][:]

    aligner = align_obj.Align(fitted_ftobj, ftobj, 'config.ini')
    fitted_tobj_aligned, tobj_aligned, _ = aligner.align()

    frc = calc_frc.FRC(obj1=np.abs(fitted_tobj_aligned), obj2=np.abs(tobj_aligned), verbose=False)
    rvals, fvals, _ = frc.calc_rot(binsize=1.0, num_rot=3600, do_abs=True)

    return rvals, fvals

rvals, fvals = compute_rvals_fvals(21,100)


# Plot FRC (Rad=40) vs B**2

Bg_sqr = []
frc_vals = []

for run_number in [i for i in range(21, 30) if i != 26]:
    filename = f'/scratch/mallabhi/lattice_ref/data/K/dataset_run{run_number:03d}.h5'
    with h5py.File(filename, 'r') as f:
        bg = f['background'][:].mean()**2
        Bg_sqr.append(bg)

    rvals, fvals = compute_rvals_fvals(run_number, 100)

    idx_rad40 = np.argmin(np.abs(rvals - 40))
    frc_vals.append(fvals[idx_rad40])

Bg_sqr, frc_vals = zip(*sorted(zip(Bg_sqr, frc_vals)))

plt.figure(figsize=(8,6))
plt.plot(Bg_sqr, frc_vals, 'o-', linewidth=2, markersize=8)

plt.xscale('log')
plt.xlabel('$B^2$', fontsize=12)
plt.ylabel('FRC (Rad = 40)', fontsize=12)
plt.grid(True, which='both', ls='--')
plt.tight_layout()
plt.show()


# Plot Bragg Peaks

N = 256
spacing = 16
lattice = np.zeros((N, N))
for i in range(0, N, spacing):
    for j in range(0, N, spacing):
        lattice[i, j] = 1.0

fft_lattice = np.fft.fft2(lattice)
fft_lattice_shifted = np.fft.fftshift(fft_lattice)
intensity = np.abs(fft_lattice_shifted)**2

x = np.linspace(-N//2, N//2, N)
y = np.linspace(-N//2, N//2, N)
X, Y = np.meshgrid(x, y)
sizes = (intensity.flatten() / intensity.max()) * 10

plt.figure(figsize=(4,4), dpi=200)
plt.scatter(X.flatten(), Y.flatten(), s=sizes, c='black', marker='s')
plt.gca().set_facecolor('white')
plt.axis('off')
plt.show()




# Plot FRC with B**2

B_sqr_vals = [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4,1e5]
nruns = 8
start_run = 38

fmean_vals = []
fstd_vals = []

for B_sqr in B_sqr_vals:
    fgroup = []
    for i in range(nruns):
        run_number = start_run + i
        filename = f'/scratch/mallabhi/lattice_ref/data/K/dataset_run{run_number:03d}.h5'

        with h5py.File(filename, 'r') as f:
             bg = f['background'][:].mean()**2
             print(f"Run {run_number}: computed B^2 = {bg}")

        rvals, fvals = compute_rvals_fvals(run_number, 100)
        idx_rad40 = np.argmin(np.abs(rvals - 40))
        frc_group.append(fvals[idx_rad40])

    fmean_vals.append(np.mean(fgroup))
    fstd_vals.append(np.std(fgroup))
    start_run += nruns

plt.figure(figsize=(8,6))
plt.errorbar(B_sqr_vals, fmean_vals, yerr=fstd_vals, fmt='o-', linewidth=2, markersize=8)
plt.xscale('log')
plt.xlabel('$B^2$', fontsize=12)
plt.ylabel('FRC (Rad = 40)', fontsize=12)
plt.grid(True, which='both', ls='--')
plt.tight_layout()
plt.show()

