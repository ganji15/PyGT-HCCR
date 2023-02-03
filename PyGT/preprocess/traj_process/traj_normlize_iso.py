# -*- encoding: utf-8 -*-

import math
import numpy as np
from numpy import linalg as LA
import copy

"""
All normalization steps are based on the following two papers:

Liwicki, M. ; Bunke, H.:  HMM-based on-line recognition  of  handwritten  whiteboard  notes.   
In: Tenth  International Workshop on Frontiers in Handwriting Recognition,
Suvisoft, 2006

Jaeger, S. ; Manke, S. ; Waibel, A.: Npen++: An On-Line Handwriting Recognition System.  
In: 7th International Workshop on Frontiers in Handwriting Recognition,
2000, S.249–260
"""

IMAGE_HEIGHT = 100
SAMPLE_DIST = 1


def normalize_trajectory(traj, thres=0.02):
    """
    Applies given normalization steps in args to trajectory of points in traj.
    Valid normalizations are "flip", "slope", "origin", "resample", "slant", "height",
    "smooth" and "delayed". Note that with application of "delayed" there will be
    two objects returned, the trajectory and the list of delayed strokes.

    The object that "traj" points to WILL BE CHANGED!
    """
    # traj = correct_slope(traj)

    # traj = remove_aboundents(traj)
    traj = remove_extra_point(traj, thres)

    traj = coord_normlize(traj)

    # storkes = traj2storkes(traj)
    # traj = np.concatenate([resampling(stroke) for stroke in storkes], axis=0)
    return traj


def normalize_sketch(traj, char_size, thres=0.05):
    traj = remove_extra_point(traj, thres)
    traj = normalize_size_coord(traj, char_size)
    return traj


def normalize_size_coord(traj, size=64., margin=2):
    font_sz = size - margin * 2
    min_x, min_y, _ = np.min(traj, axis=0)
    max_x, max_y, _ = np.max(traj, axis=0)

    if (max_x - min_x) >= (max_y - min_y):
        old_width = max_x - min_x
        scale_factor = font_sz * 1.0 / float(old_width)
        traj[:, :2] *= scale_factor
        min_x, min_y, _ = np.min(traj, axis=0)
        max_x, max_y, _ = np.max(traj, axis=0)
        shift = [0 - min_x + margin, (size - max_y - min_y) // 2 + margin, 0]
    else:
        old_height = max_y - min_y
        scale_factor = font_sz * 1.0 / float(old_height)
        traj[:, :2] *= scale_factor
        min_x, min_y, _ = np.min(traj, axis=0)
        max_x, max_y, _ = np.max(traj, axis=0)
        shift = [(size - max_x - min_x) // 2 + margin, -min_y + margin, 0]

    traj += np.array(shift)[np.newaxis,:]
    return traj


def traj_size(traj):
    len_traj = len(traj)
    min_x, min_y = np.min(traj[:, :2], axis=0)
    max_x, max_y = traj[:, :2].max(axis=0)
    print('traj size: #%d, (sx: %.1f, lx: %.1f, sy: %.1f, ly: %.1f)'%(len_traj, min_x, max_x, min_y, max_y))


def remove_extra_point(traj, thres=0.02):
    traj = traj.astype('float32')
    height, width = np.max(traj[:, :2], axis=0) - np.min(traj[:, :2], axis=0)
    t_dist  = max(height, width) * thres

    storkes = traj2storkes(traj)
    new_strokes = []
    for storke in storkes:
        dists = LA.norm(storke[: -1, :] - storke[1:, :], 2, axis=1)
        idx = np.where(dists > t_dist)[0]
        idx = [0] + list(idx + 1)
        n_storke = storke[idx]
        n_storke[-1, 2] = 1
        new_strokes.append(n_storke)
    traj = np.concatenate(new_strokes)
    return traj


def remove_from_stroke(stroke, t_dist):
    dists = LA.norm(stroke[: -1, :] - stroke[1:, :], 2, axis=1)
    idx = np.where(dists > t_dist)[0]
    idx = [0] + list(idx + 1)
    storke = stroke[idx]
    storke[-1, 2] = 1

    # t_cos  = 0.9
    # t_dist2 = max(height, width) * 0.1
    # d_xy = traj[: -1, :] - traj[1:, :]
    # dx1 = d_xy[:-1, 0]
    # dx2 = d_xy[1:, 0]
    # dy1 = d_xy[:-1, 1]
    # dy2 = d_xy[1:, 1]
    #
    # frac_up =  dx1 * dx2 + dy1 *  dy2
    # frac_down = np.sqrt(dx1 * dx1 + dy1 * dy1) * \
    #             np.sqrt(dx2 * dx2 + dy2 * dy2)
    # cos_dist = frac_up / frac_down
    # idx = np.where(cos_dist <= t_cos)[0]
    # idx = [0] + list(idx + 1) + [len(traj) - 1]
    # traj = traj[idx]
    # traj[-1, 2] = 1
    return stroke


def traj2storkes(traj):
    assert traj.shape[1] == 3
    splits = np.where(traj[:, 2] > 0)[0] + 1
    splits = [0] + list(splits)
    storkes = []

    if len(splits) == 1:
        return [traj]

    for i in range(len(splits) - 1):
        stroke = traj[splits[i] : splits[i + 1]]
        storkes.append(stroke)

    return storkes


def move_to_origin(traj):
    """
    Move trajectory so that the lower left corner
    of its bounding box is the origin afterwards.
    """
    origin = np.min(traj, axis=0)
    return traj - origin


def flip_vertically(traj):
    """
    Rotates trajectory by 180 degrees.
    """
    max_y = max(traj[:, 1])
    return np.array([[x, max_y - y] for [x, y] in traj])


def resampling(traj, step_size=SAMPLE_DIST):
    """
    Replaces given trajectory by a recalculated sequence of equidistant points.
    """
    t = []
    t.append(traj[0, :])
    i = 0
    length = 0
    current_length = 0
    old_length = 0
    curr, last = 0, None
    len_traj = traj.shape[0]
    while i < len_traj:
        current_length += step_size
        while length <= current_length and i < len_traj:
            i += 1
            if i < len_traj:
                last = curr
                curr = i
                old_length = length
                length += math.sqrt((traj[curr, 0] - traj[last, 0])**2) + math.sqrt((traj[curr, 1] - traj[last, 1])**2)
        if i < len_traj:
            c = (current_length - old_length) / float(length-old_length)
            x = traj[last, 0] + (traj[curr, 0] - traj[last, 0]) * c
            y = traj[last, 1] + (traj[curr, 1] - traj[last, 1]) * c
            if traj.shape[1] == 3:
                p = traj[last, 2]
                t.append([x, y, p])
            else:
                t.append([x, y])
    t.append(traj[-1, :])
    return np.array(t)


def normalize_height(traj, new_height=IMAGE_HEIGHT):
    """
    Returns scaled trajectory whose height will be new_height.
    TODO: try to scale core height instead
    """
    min_y = min(traj[:, 1])
    max_y = max(traj[:, 1])
    old_height = max_y - min_y
    if old_height < 1e-5:
        return normalize_width(traj, new_height)
    scale_factor = new_height * 1.0 / float(old_height)
    traj[:, :2] *= scale_factor
    # traj_size(traj)
    return traj


def normalize_width(traj, new_width=IMAGE_HEIGHT):
    """
    Returns scaled trajectory whose height will be new_height.
    TODO: try to scale core height instead
    """
    min_x = min(traj[:, 0])
    max_x = max(traj[:, 0])
    old_width = max_x - min_x
    scale_factor = new_width * 1.0 / float(old_width)
    traj[:, :2] *= scale_factor
    # traj_size(traj)
    return traj


def smoothing(traj):
    """
    Applies gaussian smoothing to the trajectory with a (0.25, 0.5, 0.25) sliding
    window. Smoothing point p(t) uses un-smoothed points p(t-1) and p(t+1).
    """
    s = lambda p, c, n: 0.25 * p + 0.5 * c + 0.25 * n
    smoothed = np.array([s(traj[i-1], traj[i], traj[i+1]) for i in range(1, traj.shape[0]-1)])
    # the code above also changes penups, so we just copy them again
    if traj.shape[1] == 3:
        smoothed[:, 2] = traj[1:-1, 2]
        # we deleted the unsmoothed first and last points,
        # so the last penup needs to be moved to the second to last point
        smoothed[-1, 2] = 1
    return smoothed


def cut_long_trajs(trajs, max_length):
    while trajs.shape[0] > max_length:
        scale = max_length * 1.0 / trajs.shape[0]
        trajs[:, 0] *= scale
        trajs = resampling(trajs)

    return trajs
    

def coord_normlize(org_traj):
    traj = org_traj[:, :2]
    dists = LA.norm(traj[: -1, :] - traj[1:, :], 2, axis=1)
    weight_sum = (traj[: -1, :] + traj[1:, :]) * dists[:, None] / 2
    ux, uy = weight_sum.sum(axis=0) / dists.sum()
    dy = np.power(traj[: -1, 1] - uy, 2) + \
         np.power(traj[1 :, 1] - uy, 2) + \
         (traj[1:, 1] - uy) * (traj[: -1, 1] - uy)
    dy = dy * dists / 3
    vy = np.sqrt(dy.sum() / dists.sum())

    dx = np.power(traj[: -1, 0] - ux, 2) + \
         np.power(traj[1 :, 0] - ux, 2) + \
         (traj[1:, 0] - ux) * (traj[: -1, 0] - ux)
    dx = dx * dists / 3
    vx = np.sqrt(dx.sum() / dists.sum())

    vxy = max(vx, vy)
    org_traj[:, 0] = (traj[:, 0] - ux) / vxy
    org_traj[:, 1] = (traj[:, 1] - uy) / vxy
    return org_traj


def coord_normlize2(org_traj):
    traj = org_traj[:, :2]
    dists = LA.norm(traj[: -1, :] - traj[1:, :], 2, axis=1)
    weight_sum = (traj[: -1, :] + traj[1:, :]) * dists[:, None] / 2
    ux, uy = weight_sum.sum(axis=0) / dists.sum()
    dy = np.power(traj[: -1, 1] - uy, 2) + \
         np.power(traj[1 :, 1] - uy, 2) + \
         (traj[1:, 1] - uy) * (traj[: -1, 1] - uy)
    dy = dy * dists / 3
    vy = np.sqrt(dy.sum() / dists.sum())

    dx = np.power(traj[: -1, 0] - ux, 2) + \
         np.power(traj[1 :, 0] - ux, 2) + \
         (traj[1:, 0] - ux) * (traj[: -1, 0] - ux)
    dx = dx * dists / 3
    vx = np.sqrt(dx.sum() / dists.sum())

    vy = max(vy, vx)

    org_traj[:, 0] = (traj[:, 0] - ux) / vy
    org_traj[:, 1] = (traj[:, 1] - uy) / vy
    return org_traj


def traj_to_strokes(traj):
    splits = np.where(traj[:, 2])[0]
    strokes = []
    start = 0
    for id in splits:
        stroke = traj[start : id + 1]
        strokes.append(stroke)
        start = id + 1
    return strokes


def resample_strokes(strokes, step_size=0.85):
    new_strokes = [resampling(stroke, step_size) for stroke in strokes]
    return new_strokes


def normalize_strokes(traj, sample_step=0.85):
    traj = coord_normlize(traj)
    strokes = traj_to_strokes(traj)
    strokes = resample_strokes(strokes, step_size=sample_step)
    traj = np.concatenate(strokes, axis=0)
    return traj


def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def spatial_cluster(points, d):
    clusters = []
    n = len(points)
    taken = [False] * n
    for i in range(n):
        if not taken[i]:
            cluster = []
            cluster.append(i)
            taken[i] = True
            for j in range(i+1, n):
                if dist(points[i], points[j]) < d:
                    cluster.append(j)
                    taken[j] = True
            clusters.append(cluster)

    return clusters


def spatial_fuse(points, d):
    clusters = spatial_cluster(points, d)
    ret = copy.copy(points).astype(np.float32)
    for cluster in clusters:
        avg_coord = points[cluster].mean(0)
        ret[cluster] = avg_coord
    return np.array(ret)
