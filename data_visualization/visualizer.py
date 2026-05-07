import h5py
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Circle


# =========================
# CONFIG
# =========================
H5_PATH = "dataset/Training-validation set (11:36 and 12:43)/frog_11-36_12-43_train_val.h5"

PLAY_SPEED = 0.03
SPEED_MULTIPLIER = 2


# =========================
# LOAD DATA
# =========================
with h5py.File(H5_PATH, "r") as f:
    scans = f["scans"][:]
    circles = f["circles"][:]
    circle_idx = f["circle_idx"][:]
    circle_num = f["circle_num"][:]

    # Try to load odometry if it exists
    odom_x = f["x"][:] if "x" in f else None
    odom_y = f["y"][:] if "y" in f else None


N = scans.shape[0]
angles = np.linspace(-np.pi, np.pi, scans.shape[1])


# =========================
# FIGURE SETUP
# =========================
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)

ax.set_title("LiDAR + People + Robot Trajectory")
ax.set_aspect("equal")
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)


# LiDAR
scan_plot, = ax.plot([], [], "k.", markersize=2)

# robot
robot_plot, = ax.plot([], [], "bo", markersize=6)

# trajectory line
traj_plot, = ax.plot([], [], "b-", linewidth=1)

trajectory_x = []
trajectory_y = []

circle_patches = []


# =========================
# DRAW FRAME
# =========================
def draw_frame(i):
    global circle_patches, trajectory_x, trajectory_y

    scan = scans[i]

    # LiDAR → Cartesian
    x = scan * np.cos(angles)
    y = scan * np.sin(angles)

    scan_plot.set_data(x, y)

    # remove old circles
    for c in circle_patches:
        c.remove()
    circle_patches = []

    start = circle_idx[i]
    count = circle_num[i]

    for j in range(start, start + count):
        cx, cy, r = circles[j, 0], circles[j, 1], circles[j, 2]
        c = Circle((cx, cy), r, fill=False, color="red", linewidth=2)
        ax.add_patch(c)
        circle_patches.append(c)

    # =========================
    # ROBOT POSITION (if available)
    # =========================
    if odom_x is not None and odom_y is not None:
        rx = odom_x[i]
        ry = odom_y[i]

        robot_plot.set_data([rx], [ry])

        trajectory_x.append(rx)
        trajectory_y.append(ry)

        traj_plot.set_data(trajectory_x, trajectory_y)

    fig.canvas.draw_idle()


# =========================
# SLIDER
# =========================
ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
slider = Slider(ax_slider, "Frame", 0, N - 1, valinit=0, valstep=1)


def update(val):
    draw_frame(int(slider.val))


slider.on_changed(update)


# =========================
# PLAY / PAUSE
# =========================
playing = False


def toggle_play(event):
    global playing
    playing = not playing
    if playing:
        play()


def play():
    global playing

    i = int(slider.val)

    while playing and i < N:
        slider.set_val(i)
        plt.pause(PLAY_SPEED / SPEED_MULTIPLIER)
        i += 1

    playing = False


# =========================
# SPEED CONTROL
# =========================
def speed_up(event):
    global SPEED_MULTIPLIER
    SPEED_MULTIPLIER *= 2
    if SPEED_MULTIPLIER > 8:
        SPEED_MULTIPLIER = 1
    print("Speed:", SPEED_MULTIPLIER, "x")


ax_button = plt.axes([0.72, 0.02, 0.1, 0.05])
button = Button(ax_button, "Play")
button.on_clicked(toggle_play)

ax_speed = plt.axes([0.83, 0.02, 0.1, 0.05])
speed_button = Button(ax_speed, "Speed")
speed_button.on_clicked(speed_up)


# =========================
# INIT
# =========================
draw_frame(0)
plt.show()