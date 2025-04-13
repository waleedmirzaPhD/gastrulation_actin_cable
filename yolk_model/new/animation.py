import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Rectangle, Circle
from scipy.interpolate import interp1d

plt.style.use('dark_background')

# Parameters
L = 1.0
gamma = 1.0
eta = 1.0
F_applied = 0.0
l = 1
xi = eta / l**2

# Velocity profile
x_vals_local = np.linspace(0, L, 300)
coshL = np.cosh(L / l)
tanhL = np.tanh(L / l)
v0 = (gamma * (1 - 1 / coshL) - F_applied) * (l / eta) / tanhL
vx_local = (1 / coshL) * (v0 * np.cosh((x_vals_local - L) / l) - (gamma * l / eta) * np.sinh(x_vals_local / l))
v_interp_local = interp1d(x_vals_local, vx_local, bounds_error=False, fill_value=0.0)

# Time settings
dt = 0.03
n_frames = 100
spawn_prob = 0.2
remove_prob = 0.2

# Filament parameters
N_initial_filaments = 200
filament_length = 0.05
filaments = []

np.random.seed(42)
for _ in range(N_initial_filaments):
    x = np.random.uniform(0, L)
    angle = np.random.uniform(-np.pi / 2, np.pi / 2)
    filaments.append({'x': x, 'angle': angle})

# Cell network parameters
n_cells = 60
cell_width = 0.08
cell_y = 0.0
cells = []

# Set up figure
fig, ax = plt.subplots(figsize=(10, 1))
ax.set_ylim(-0.125, 0.125)
ax.set_xlim(-L, 3 * L + 0.05)
ax.axis('off')

# Create cell patches
for i in range(n_cells):
    rect = Rectangle((0, cell_y), cell_width, cell_width, linewidth=1, edgecolor='white', facecolor='none')
    circ = Circle((0, cell_y + cell_width / 2), cell_width * 0.3, facecolor="#67a9cf", edgecolor='none')
    ax.add_patch(rect)
    ax.add_patch(circ)
    cells.append((rect, circ))

# Filament lines
filament_lines = [ax.plot([], [], '-', c="#ef8a62", lw=1)[0] for _ in range(200)]

# Velocity arrows
arrow_y = -0.03
n_arrows = 8
arrow_x = np.linspace(0, L, n_arrows)
dummy_u = np.zeros_like(arrow_x)
dummy_v = np.zeros_like(arrow_x)
velocity_arrows = ax.quiver(arrow_x, [arrow_y] * n_arrows, dummy_u, dummy_v,
                            angles='xy', scale_units='xy', scale=5, color='white', width=0.004)

def init():
    for line in filament_lines:
        line.set_data([], [])
    velocity_arrows.set_UVC(dummy_u, dummy_v)
    return filament_lines + [velocity_arrows] + [patch for pair in cells for patch in pair]

def animate(frame):
    t = frame * dt
    left_margin_pos = v0 * t

    new_filaments = []
    for f in filaments:
        x_local = f['x'] - left_margin_pos
        if np.random.rand() < remove_prob:
            continue
        v = v_interp_local(x_local)
        f['x'] += v * dt
        if left_margin_pos <= f['x'] <= left_margin_pos + L:
            new_filaments.append(f)
    filaments[:] = new_filaments

    n_new = np.random.poisson(spawn_prob * 100)
    for _ in range(n_new):
        x = np.random.uniform(left_margin_pos, left_margin_pos + L)
        angle = np.random.uniform(-np.pi / 2, np.pi / 2)
        filaments.append({'x': x, 'angle': angle})

    for i, line in enumerate(filament_lines):
        if i < len(filaments):
            f = filaments[i]
            dx = 0.5 * filament_length * np.cos(f['angle'])
            dy = 0.5 * filament_length * np.sin(f['angle'])
            line.set_data([f['x'] - dx, f['x'] + dx], [-dy, dy])
        else:
            line.set_data([], [])

    for i, (rect, circ) in enumerate(cells):
        cx = left_margin_pos - (n_cells - i) * cell_width * 1.1
        rect.set_xy((cx, cell_y))
        circ.set_center((cx + cell_width / 2, cell_y + cell_width / 2))

    arrow_x_new = np.linspace(left_margin_pos, left_margin_pos + L, n_arrows)
    arrow_u = v_interp_local(arrow_x_new - left_margin_pos)
    arrow_u/=np.max(arrow_u)
    velocity_arrows.set_offsets(np.c_[arrow_x_new, [arrow_y] * n_arrows])
    velocity_arrows.set_UVC(arrow_u, np.zeros_like(arrow_u))

    return filament_lines + [velocity_arrows] + [patch for pair in cells for patch in pair]

# Create animation
anim = FuncAnimation(fig, animate, init_func=init, frames=n_frames, interval=100, blit=True)

# Save the animation to MP4
output_path = "active_gel_with_cells.mp4"
writer = FFMpegWriter(fps=10, metadata=dict(artist='Your Name'), bitrate=1800)
anim.save(output_path, writer=writer)

print(f"Animation saved as {output_path}")