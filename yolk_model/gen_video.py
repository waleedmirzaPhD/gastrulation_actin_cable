import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as animation

data = np.loadtxt("data.txt")

xlim = (-3,3)
ylim = (-3,3)
def spherical_cap_cross_section(a, h, resolution=100):
    R = (a**2 + h**2) / (2 * h)
    theta = np.arccos(1 - h / R)
    thetas = np.linspace(-theta, theta, resolution, endpoint=True)
    x = R * np.sin(thetas)
    z = h - R + R * np.cos(thetas)
    return x, z

def update(t):
    idx = (np.abs(data[:,0] - t)).argmin()
    h,H,a = data[idx,1:4]
    x1, z1 = spherical_cap_cross_section(a, h)
    x2, z2 = spherical_cap_cross_section(a, H)

    # Center the curves at z = 0
    z_center = 0.5 * (np.max(z1)-np.max(z2))
    z1 -= z_center
    z2 += z_center


    plt1.set_data(x1, z1)
    plt2.set_data(x2, -z2)
    # plt.title(f'g = {g:.3f}, time step = {t}')
    # plt.close()

fig = plt.figure(figsize=(8, 6))
plt.axis('equal')
plt.xlim(xlim)
plt.ylim(ylim)
plt1, = plt.plot([], [], color='b', label='EVL', animated=True)
plt2, = plt.plot([], [], color='r', label='Yolk', animated=True)
plt.legend()

ani = animation.FuncAnimation(fig, update, frames=np.linspace(0,data[-1,0],100))
FFwriter = animation.FFMpegWriter(fps=20)
ani.save('output.mp4', writer=FFwriter)