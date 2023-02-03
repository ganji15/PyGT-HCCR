import sys
import numpy
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_points(traj, info='k-o', off_axes=True, show=True):
    out = numpy.max(traj, axis=0) - numpy.min(traj, axis=0)
    if out.shape[0] == 3:
        width, height, _ = out
    else:
        width, height = out
    nw = width / height * 2
    plt.figure(figsize=(nw, 1))
    plt.plot(traj[:, 0], traj[:, 1], info)
    if off_axes:
        plt.axis('off')
    if show:
        plt.show()


def plot_points_3d(traj, info='b', simple=True):

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    if simple:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        # plt.axis('off')
        ax.w_xaxis.line.set_color("skyblue")
        ax.w_yaxis.line.set_color("skyblue")
        ax.w_zaxis.line.set_color("skyblue")
        ax.w_yaxis.set_pane_color((0.8, .8, .8))
        ax.w_xaxis.set_pane_color((0.7, .7, .7))
        ax.w_zaxis.set_pane_color((0.9, .9, .9))

    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], info)
    plt.show()


def plot_strokes(strokes, info):
    # plt.axis('off')
    for stroke in strokes:
        plt.plot(stroke[:, 0], stroke[:, 1], info, markersize=3)
