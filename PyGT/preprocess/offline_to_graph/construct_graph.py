import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import copy
import math
import cv2
from imutils import resize
from skimage.morphology import skeletonize
import PIL


def plot_graph_attention(nodes, edge_index, node_weights=None, node_size=60, cmap='GnBu', node_shape='o', edge_width=2.0,
                         edge_alpha=1.):
    if node_weights is None:
        node_weights = [0.5] * len(nodes)

    G = nx.Graph()
    for i, (pos, weight) in enumerate(zip(nodes, node_weights)):
        G.add_node(i, pos=(pos[0], pos[1]))

    for edge in edge_index:
        G.add_edge(*edge)

    g_nodes = nx.draw_networkx_nodes(G, nodes, cmap=plt.get_cmap(cmap), node_color=node_weights,
                                     node_size=node_size, node_shape=node_shape)
    g_nodes.set_edgecolor('k')
    nx.draw_networkx_edges(G, nodes, alpha=edge_alpha, width=edge_width)
    plt.axis('on')


def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def spatial_cluster(points, d, cluster_with_center=False):
    clusters = []
    n = len(points)
    taken = [False] * n
    for i in range(n):
        if not taken[i]:
            cluster = [i]
            cluster_center = np.array(points[i]).astype(np.float)
            taken[i] = True
            for j in range(i+1, n):
                if cluster_with_center:
                    base_point = cluster_center
                else:
                    base_point = points[i]
                if dist(base_point, points[j]) < d:
                    cluster_center = (cluster_center * len(cluster) + np.array(points[j])) / (len(cluster) + 1)
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


def _cal_edge(id1, id2):
    return [id1, id2] if id1 < id2 else [id2, id1]


def merge_close_nodes(nodes, edge_index, node_vals, merge_thres, merge_with_center):
    nodes = np.array(nodes).astype(np.float32)
    clusters = spatial_cluster(nodes, merge_thres, merge_with_center)
    map_id = {}
    new_nodes = []
    new_node_vals = []

    for i, cluster in enumerate(clusters):
        for id in cluster:
            map_id[id] = i

        new_nodes.append(nodes[cluster].mean(0))
        new_node_vals.append(node_vals[cluster].mean())

    new_edge_index = []
    for edge in edge_index:
        id0 = map_id[edge[0]]
        id1 = map_id[edge[1]]

        if id0 == id1:
            continue

        edge = _cal_edge(id0, id1)
        if edge not in new_edge_index:
            new_edge_index.append(edge)

    return np.array(new_nodes), new_edge_index, np.array(new_node_vals)


def construct_init_dense_graph(map, img, half_win_size=1):
    pts = np.stack(np.where(map > 0)).transpose().astype(np.int)

    # dmap = np.zeros_like(map)
    # for pt in pts:
    #     dmap[pt[0], pt[1]] = 255.
    # plt.subplot(121)
    # plt.imshow(dmap, cmap='gray')
    # plt.title('dmap')
    # plt.subplot(122)
    # plt.imshow(map, cmap='gray')
    # plt.title('map')
    # plt.show()
    def _valid_pos(val):
        return max(min(int(abs(val)), map.shape[0] - 1), 0)

    node_vals = [img[_valid_pos(pt[0] - half_win_size): _valid_pos(pt[0] + half_win_size + 1),
                 _valid_pos(pt[1] - half_win_size): _valid_pos(pt[1] + half_win_size + 1)].mean()
                 for pt in pts]

    node_vals = np.array(node_vals, dtype=np.float32)

    def hash(pt):
        return pt[0] * map.shape[0] + pt[1]

    node_id_dict = {}
    for i, pt in enumerate(pts):
        node_id_dict[hash(pt)] = i

    def _calc_pid(pt):
        return node_id_dict[hash(pt)]

    def _is_valid_pt(pt):
        if 0 <= pt[0] < map.shape[0] and \
                0 <= pt[1] < map.shape[1] and \
                map[pt[0], pt[1]] > 0:
            return True
        return False

    edges = []
    for pt in pts:
        pt_id = _calc_pid(pt)
        for i, j in [(0, 1), [1, 0], [0, -1], [-1, 0]]:
            new_pt = [pt[0] + i, pt[1] + j]
            if _is_valid_pt(new_pt):
                edges.append(_cal_edge(pt_id, _calc_pid(new_pt)))

    for pt in pts:
        pt_id = _calc_pid(pt)
        for i, j in [(1, 1), [1, -1], [-1, -1], [-1, 1]]:
            add_cx_edge = True
            cx_nb_pt = [pt[0] + i, pt[1] + j]
            if _is_valid_pt(cx_nb_pt):
                candi_pt1 = [pt[0] + i, pt[1]]
                candi_pt2 = [pt[0], pt[1] + j]
                for candi_pt in [candi_pt1, candi_pt2]:
                    if _is_valid_pt(candi_pt):
                        add_cx_edge = False
                        break

                if add_cx_edge:
                    edges.append(_cal_edge(pt_id, _calc_pid(cx_nb_pt)))

    return pts.astype(np.float), edges, node_vals


def _get_neibors(pid, bi_edges):
    eid_of_pid = np.where(bi_edges[:, 0] == pid)[0]
    return bi_edges[eid_of_pid, 1]


def calc_k_order_edges(bi_edges, bi_k_1_edges, num_nodes=None):
    '''edge_kth = connect(edge_k_1, edge_1)'''
    if num_nodes is None:
        num_nodes = bi_edges.max() + 1

    # 宽度优先
    edges_kth_orders = []
    for pid in range(num_nodes):
        neibors_k_1_order = _get_neibors(pid, bi_k_1_edges)
        for nb_k_1_pid in neibors_k_1_order:
            neibors_k_order = _get_neibors(nb_k_1_pid, bi_edges)
            for nb_kth_pid in neibors_k_order:
                if nb_kth_pid != pid:
                    new_edge = _cal_edge(nb_kth_pid, pid)
                    edges_kth_orders.append(new_edge)

    return edges_kth_orders


def _bidirection_edges(edges):
    edges_ndarr = np.array(edges).astype(np.int)
    bi_edges = np.concatenate([edges_ndarr, edges_ndarr[:, [1, 0]]])
    return bi_edges


def remove_small_node_cluster(pts, edges, node_vals):
    '''to be done'''
    pt_idxs = set(range(len(node_vals)))
    bi_edges = _bidirection_edges(edges)
    clusters = []

    while(len(pt_idxs) > 0):
        pid = pt_idxs.pop()

        cluster = [pid]
        neibors_k_1_order = _get_neibors(pid, bi_edges)
        cluster += neibors_k_1_order
        break


def construct_graph(map, img, half_win_size=1, merge_dist_thres=0., isolate_num_thresh=2, edge_order=1, merge_with_center=False, sort_by_geomtric=True):
    pts, edges, node_vals = construct_init_dense_graph(map, img, half_win_size)

    if merge_dist_thres > 0.99:
        pts, edges, node_vals = merge_close_nodes(pts, edges, node_vals, merge_dist_thres, merge_with_center)

    node_vals /= 255.
    pts = pts[:, [1, 0]]
    pts[:, 1] = map.shape[0] - pts[:, 1] - 1
    pts = pts * 2 / map.shape[0] - 1

    if sort_by_geomtric:
        pts = [tuple(pt) for pt in pts]
        pts = np.array(pts, dtype=[('x', '<f8'), ('y', '<f8')])
        dict_id = {}
        ind = np.argsort(pts, order=('x', 'y'))
        # print(ind)
        new_pts, new_node_vals = [], []
        for i, id in enumerate(ind):
            dict_id[id] = i
            new_pts.append(list(pts[id]))
            new_node_vals.append(node_vals[id])
        pts =  np.array(new_pts, dtype=np.float32)
        node_vals = np.array(new_node_vals, dtype=np.float32)

        # print(edge_index.shape)
        edges = np.array(edges, dtype=int)
        for i in range(edges.shape[0]):
            for j in range(edges.shape[1]):
                edges[i, j] = dict_id[edges[i, j]]

    return pts, edges, node_vals



def image2graph(image, width, ref_img=None, with_processing=False, debug=False, **kwargs):
    if isinstance(image, str):
        img = cv2.imread(image, -1)
    elif isinstance(image, PIL.Image.Image):
        img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif isinstance(image, np.ndarray):
        img = image

    if width != img.shape[1]:
        img = resize(img, width=width)

    if ref_img is None:
        org_img = img
    else:
        org_img = ref_img

    if with_processing:
        pass

    img = cv2.equalizeHist(img)
    ret, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thin_img = skeletonize(binary / 255.)

    nodes, edges, node_vals = construct_graph(thin_img, org_img, **kwargs)
    # construct_2nd_order_graph

    if debug:
        print(thin_img.shape)
        plt.figure(figsize=(5, 5))
        plt.subplot(221)
        plt.imshow(org_img, cmap='gray')

        plt.subplot(222)
        plt.imshow(binary, cmap='gray')

        plt.subplot(223)
        plt.imshow(thin_img, cmap='gray')

        plt.subplot(224)
        plot_graph_attention(nodes, edges, node_vals * 255.)
        print('node_vals:', node_vals)
        print('#nodes: %d  #edges: %d' % (len(nodes), len(edges)))
        print('val min: %.2f  max: %.2f'%(node_vals.min(), node_vals.max()))

        plt.xlim(-1, 1)
        plt.ylim(-1, 1)

        plt.tight_layout()
        plt.show()

    return nodes, edges, node_vals


def plot_graph_img(nodes, edges, image_size=64):
    nodes = (nodes * image_size + image_size) / 2
    nodes = nodes.astype(np.int).clip(0, image_size - 1)
    nodes[:, 1] = image_size - nodes[:, 1]
    img = np.zeros((image_size, image_size))
    for id1, id2 in edges:
        cv2.line(img, tuple(nodes[id1]), tuple(nodes[id2]), 255, 1)
    return img
