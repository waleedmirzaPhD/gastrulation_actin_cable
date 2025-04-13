import numpy as np
import imageio
from vedo import Plotter, Mesh, Text2D
from vedo.utils import buildPolyData

data = np.loadtxt("data.txt")
frame = 0

def spherical_cap_cross_section(a, h, resolution=200):
    R = (a**2 + h**2) / (2 * h)
    theta = np.arccos(1 - h / R)
    thetas = np.linspace(-theta, theta, resolution)
    x = R * np.sin(thetas)
    z = h - R + R * np.cos(thetas)
    return np.c_[x, z]

def revolve_profile(profile, n_around=200):
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
cross1 = spherical_cap_cross_section(a0, h0)
cross2 = spherical_cap_cross_section(a0, H0)
z_center = 0.5 * (np.max(cross1[:, 1]) - np.max(cross2[:, 1]))
cross1[:, 1] -= z_center
cross2[:, 1] += z_center
cross2[:, 1] *= -1

surf1 = revolve_profile(cross1).c('#fbb4ae').alpha(1)
surf2 = revolve_profile(cross2).c('#b3cde3').alpha(1)
time_label = Text2D("t = 0.00", pos='top-left', c='black', s=1.2)

plt += [surf1, surf2, time_label]
# plt.camera.Elevation(90)
# plt.camera.Dolly(100)

plt.camera.SetPosition([0, -5, 0])  # [x, y, z]
plt.camera.SetFocalPoint([0, 0, 0])
plt.camera.SetViewUp([0, 0, 1])      # Z axis pointing up

# Set up video writer
writer = imageio.get_writer("output_vedo.mp4", fps=20, format='ffmpeg')

# Animation loop
n_frames = 100
t_vals = np.linspace(0, data[-1, 0], n_frames)

for t in t_vals:
    idx = (np.abs(data[:, 0] - t)).argmin()
    h, H, a = data[idx, 1:4]

    cross1 = spherical_cap_cross_section(a, h)
    cross2 = spherical_cap_cross_section(a, H)

    z_center = 0.5 * (np.max(cross1[:, 1]) - np.max(cross2[:, 1]))
    cross1[:, 1] -= z_center
    cross2[:, 1] += z_center
    cross2[:, 1] *= -1

    new_surf1 = revolve_profile(cross1)
    new_surf2 = revolve_profile(cross2)

    surf1.points = new_surf1.points
    surf2.points = new_surf2.points
    time_label.text(f"t = {t:.2f}")

    plt.render()
    frame = plt.screenshot(asarray=True)
    writer.append_data(frame)

writer.close()
plt.close()