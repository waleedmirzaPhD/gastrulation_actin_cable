import numpy as np
import imageio
from vedo import Plotter, Mesh, Text2D
from vedo.utils import buildPolyData

data = np.loadtxt("data.txt")
frame = 0

def spherical_cap_cross_section(a, h, resolution=500):
    R = (a**2 + h**2) / (2 * h)
    theta = np.arccos(1 - h / R)
    thetas = np.linspace(0, theta, resolution)
    x = R * np.sin(thetas)
    z = h - R + R * np.cos(thetas)
    return np.c_[x, z]

def revolve_profile(profile, n_around=500):
    x, z = profile[:, 0], profile[:, 1]
    theta = np.linspace(0, 2*np.pi, n_around)
    X, T = np.meshgrid(x, theta)
    Z = np.tile(z, (n_around, 1))
    Y = X * np.sin(T)
    X = X * np.cos(T)
    pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

    faces = []
    res = len(x)
    for i in range(n_around - 1):
        for j in range(res - 1):
            p0 = i * res + j
            p1 = p0 + 1
            p2 = p0 + res
            p3 = p2 + 1
            faces.append([p0, p1, p3])
            faces.append([p0, p3, p2])

    poly = buildPolyData(pts, faces)
    return Mesh(poly)

# Setup plotter
plt = Plotter(size=(2160, 2160), bg='black', axes=1, offscreen=True)

# Initial shapes
h0, H0, a0 = data[0, 1:4]
cross1 = spherical_cap_cross_section(a0, h0)  # upper cap
cross2 = spherical_cap_cross_section(a0, H0)  # lower cap

# Positioning
z_center = 0.5 * (np.max(cross1[:, 1]) - np.max(cross2[:, 1]))
cross1[:, 1] -= z_center
cross2[:, 1] += z_center
cross2[:, 1] *= -1  # Flip lower cap

# Split lower cap (cross2) into belt + body
z_max = np.max(cross2[:, 1])
belt_cut = z_max - (1 / 7)

belt_profile = cross2[cross2[:, 1] >= belt_cut]
body_profile = cross2[cross2[:, 1] <= belt_cut]

surf_upper = revolve_profile(cross1).c('#b3cde3').alpha(1)      # Upper cap
surf_belt = revolve_profile(belt_profile).c('#fbb4ae').alpha(1)  # Belt (top of lower cap)
surf_body = revolve_profile(body_profile).c('#d3d3d3').alpha(1)  # Rest of lower cap

time_label = Text2D("t = 0.00", pos='top-left', c='black', s=1.2)
plt += [surf_upper, surf_belt, surf_body, time_label]

# Camera
plt.camera.SetPosition([0, -5, 0])
plt.camera.SetFocalPoint([0, 0, 0])
plt.camera.SetViewUp([0, 0, 1])

# Video writer
writer = imageio.get_writer("output_vedo.mp4", fps=20, format='ffmpeg')

# Animation loop
t_vals = np.linspace(0, data[-1, 0], int(100*data[-1, 0]/3))

for t in t_vals:
    idx = (np.abs(data[:, 0] - t)).argmin()
    h, H, a = data[idx, 1:4]

    # Generate cross-sections
    cross1 = spherical_cap_cross_section(a, h)
    cross2 = spherical_cap_cross_section(a, H)

    # Align in Z
    z_center = 0.5 * (np.max(cross1[:, 1]) - np.max(cross2[:, 1]))
    cross1[:, 1] -= z_center
    cross2[:, 1] += z_center
    cross2[:, 1] *= -1

    # Split lower cap into belt + body
    z_max = np.max(cross2[:, 1])
    belt_cut = z_max - (1 / 7)

    belt_profile = cross2[cross2[:, 1] >= belt_cut]
    body_profile = cross2[cross2[:, 1] <= belt_cut+0.01]

    # Remove old surfaces from the scene
    plt.remove([surf_upper, surf_belt, surf_body])

    # Create new surfaces
    surf_upper = revolve_profile(cross1).c('#b3cde3').alpha(1)
    surf_belt  = revolve_profile(belt_profile).c('#fbb4ae').alpha(1)
    surf_body  = revolve_profile(body_profile).c('#d3d3d3').alpha(1)

    plt += [surf_upper, surf_belt, surf_body]

    # Update time label
    time_label.text(f"t = {t:.2f}")

    # Render and write frame
    plt.render()
    frame = plt.screenshot(asarray=True)
    writer.append_data(frame)

writer.close()
plt.close()