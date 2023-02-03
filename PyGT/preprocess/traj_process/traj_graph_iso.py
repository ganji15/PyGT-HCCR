import math
import networkx as nx
import matplotlib.pyplot as plt
from preprocess.traj_process.traj_normlize_iso import spatial_cluster
import copy
import numpy as np


class Line(object):
    def __init__(self, pt1, pt2, fake=False):
        self.pts = [pt1, pt2]
        self.fake = fake

    def __eq__(self, other):
        return (other.pts[0] == self.pts[0] and other.pts[1] == self.pts[1]) or\
               (other.pts[0] == self.pts[1] and other.pts[0] == self.pts[1])

    def add_point(self, pt):
        if pt not in self.pts:
            self.pts.append(pt)

    def remove_point(self, id):
        if id in self.pts:
            self.pts.remove(id)

    def calc_edges(self, nodes, include_fake=False):
        if not include_fake and self.fake:
            return []

        all_pts = sorted(self.pts, key=lambda pt: (nodes[pt][0], nodes[pt][1]))
        edges = []
        if all_pts[0] == self.pts[0]:
            for i in range(len(all_pts) - 1):
                edges.append([all_pts[i], all_pts[i + 1]])
        else:
            for i in range(len(all_pts) - 1):
                edges.append([all_pts[i + 1], all_pts[i]])
        return edges


def cross_exp(p1, p2, p3):#跨立实验
    x1 = p2[0] - p1[0]
    y1 = p2[1] - p1[1]
    x2 = p3[0] - p1[0]
    y2 = p3[1] - p1[1]
    return x1 * y2 - x2 * y1


def cross_point(p1, p2, p3, p4):  # 计算交点函数
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    x3, y3 = p3[0], p3[1]
    x4, y4 = p4[0], p4[1]

    k1 = (y2 - y1) * 1.0 / (x2 - x1 + 1e-7)  # 计算k1,由于点均为整数，需要进行浮点数转化
    b1 = y1 * 1.0 - x1 * k1 * 1.0  # 整型转浮点型是关键
    if math.fabs(x4 - x3) < 1e-7:  # L2直线斜率不存在操作
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)  # 斜率存在操作
        b2 = y3 * 1.0 - x3 * k2 * 1.0

    if k2 is None:
        x = x3
    elif k1 == k2:
        return None
    else:
        x = (b2 - b1) * 1.0 / (k1 - k2)

    y = k1 * x * 1.0 + b1 * 1.0
    return [x, y]


def IsIntersect(p1, p2, p3, p4): #判断两线段是否相交
    #快速排斥，以l1、l2为对角线的矩形必相交，否则两线段不相交
    if(max(p1[0], p2[0]) >= min(p3[0], p4[0])    #矩形1最右端大于矩形2最左端
    and max(p3[0], p4[0]) >= min(p1[0], p2[0])   #矩形2最右端大于矩形最左端
    and max(p1[1], p2[1]) >= min(p3[1], p4[1])   #矩形1最高端大于矩形最低端
    and max(p3[1], p4[1]) >= min(p1[1], p2[1])): #矩形2最高端大于矩形最低端

    #若通过快速排斥则进行跨立实验
        if(cross_exp(p1, p2, p3) * cross_exp(p1, p2, p4) <= 0
           and cross_exp(p3, p4, p1) * cross_exp(p3, p4, p2) <= 0):
            return True

    return False


def calculate_joint_point(p1, p2, p3, p4):
    if IsIntersect(p1, p2, p3, p4):
        return cross_point(p1, p2, p3, p4)
    else:
        return None


def merge_close_nodes(nodes, edge_index, merge_thres, len_traj):
    nodes = np.array(nodes).astype(np.float32)
    clusters = spatial_cluster(nodes, merge_thres)
    map_id = {}
    new_nodes = []
    for i, cluster in enumerate(clusters):
        cross_id = None
        for id in cluster:
            map_id[id] = i
            if cross_id is None and id >= len_traj:
                cross_id = id
        # new_nodes.append(nodes[cluster].mean(0))
        if cross_id is None:
            new_nodes.append(nodes[cluster].mean(0))
        else:
            new_nodes.append(nodes[cross_id])

    new_edge_index = []
    for edge in edge_index:
        id0 = map_id[edge[0]]
        id1 = map_id[edge[1]]
        if id0 == id1:
            continue

        if [id0, id1] not in new_edge_index:
            new_edge_index.append([id0, id1])

    return new_nodes, new_edge_index


def construct_graph(traj, merge_thres=0., with_cross_pt=False, with_visual_connection=False, sort_by_geomtric=True):
    nodes = []
    lines = []
    stroke_id = 0
    pt_tag = 1
    for i, pt in enumerate(traj):
        node = (pt[0], pt[1], pt_tag)
        nodes.append(node)
        if i < len(traj) - 1:
            if traj[i, 2] == 1 and traj[i + 1, 2] == 0:
                stroke_id += 1
                pt_tag = 1
                if with_visual_connection:
                    line = Line(i, i + 1, -1)
                    lines.append(line)
            else:
                line = Line(i, i + 1, stroke_id)
                lines.append(line)
                pt_tag = traj[i + 1, 2]

    num_lines = len(lines)
    if with_cross_pt:
        for i in range(num_lines - 1):
            for j in range(i + 1, num_lines):
                if lines[i].stroke_id == lines[j].stroke_id:
                    continue

                if lines[i].stroke_id == -1 or lines[j].stroke_id == -1:
                    continue

                cross_pt = calculate_joint_point(nodes[lines[i].pts[0]], nodes[lines[i].pts[1]],
                                                 nodes[lines[j].pts[0]], nodes[lines[j].pts[1]])

                if cross_pt is not None:
                    lines[i].add_point(len(nodes))
                    lines[j].add_point(len(nodes))
                    nodes.append([cross_pt[0], cross_pt[1], 2])

    edge_index = []
    for line in lines:
        edge_index += line.calc_edges(nodes, merge_thres)
    edge_index = np.array(edge_index)

    if sort_by_geomtric:
        nodes = np.array(nodes, dtype=[('x', '<f8'), ('y', '<f8'), ('z', '<i4')])
        dict_id = {}
        ind = np.argsort(nodes, order=('x', 'y', 'z'))
        # print(ind)
        new_nodes = []
        for i, id in enumerate(ind):
            dict_id[id] = i
            new_nodes.append(list(nodes[id]))
        nodes = new_nodes

        # print(edge_index.shape)
        for i in range(edge_index.shape[0]):
            for j in range(edge_index.shape[1]):
                edge_index[i, j] = dict_id[edge_index[i, j]]

    return np.array(nodes), edge_index


def plot_graph_attention(nodes, edge_index, node_weights=None, node_size=60, cmap='GnBu',
                         node_shape='o', edge_width=1.0, edge_alpha=0.5):
    if node_weights is None:
        node_weights = [1] * len(nodes)

    G = nx.Graph()
    for i, (pos, weight) in enumerate(zip(nodes, node_weights)):
        G.add_node(i, pos=(pos[0], pos[1]))

    for edge in edge_index:
        G.add_edge(*edge)

    g_nodes = nx.draw_networkx_nodes(G, nodes, cmap=plt.get_cmap(cmap), node_color=node_weights,
                                     node_size=node_size, node_shape=node_shape)
    g_nodes.set_edgecolor('k')
    nx.draw_networkx_edges(G, nodes, alpha=edge_alpha, width=edge_width)
    # plt.axis('on')


def plot_graph2(nodes, edges, node_weights, cmap='Blues'):
    plt.scatter(nodes[:, 0], nodes[:, 1], cmap=plt.get_cmap(cmap), c=int(node_weights))
    for id1, id2 in edges:
        pts = np.stack([nodes[id1], nodes[id2]])
        plt.plot(pts[:, 0], pts[:, 1])


def plot_graph(nodes, edge_index, with_labels=False, label_size=50, node_size=80, font_size=10, directed=True):
    G = nx.DiGraph() if directed else nx.Graph()
    for i, node in enumerate(nodes):
        G.add_node(i, pos=(node[0], node[1]))

    for edge in edge_index:
        G.add_edge(*edge)

    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, with_labels=with_labels,
            # label_size=label_size,
            node_size=node_size,
            font_size=font_size)
    ax = plt.gca()  # to get the current axis
    ax.collections[0].set_color("#11EE96")
    ax.collections[0].set_edgecolor("#0099FF")
    plt.axis('on')



if __name__ == '__main__':
    pass
