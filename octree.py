# octree的具体实现，包括构建和查找

import random
import math
import numpy as np
import time

from result_set import KNNResultSet, RadiusNNResultSet


# 节点，构成OCtree的基本元素
class Octant:
    def __init__(self, children, center, extent, point_indices, is_leaf):
        self.children = children
        self.center = center
        self.extent = extent
        self.point_indices = point_indices
        self.is_leaf = is_leaf

    def __str__(self):
        output = ''
        output += 'center: [%.2f, %.2f, %.2f], ' % (self.center[0], self.center[1], self.center[2])
        output += 'extent: %.2f, ' % self.extent
        output += 'is_leaf: %d, ' % self.is_leaf
        output += 'children: ' + str([x is not None for x in self.children]) + ", "
        output += 'point_indices: ' + str(self.point_indices)
        return output


# 功能：翻转octree
# 输入：
#     root: 构建好的octree
#     depth: 当前深度
#     max_depth：最大深度
def traverse_octree(root: Octant, depth, max_depth):
    depth[0] += 1
    if max_depth[0] < depth[0]:
        max_depth[0] = depth[0]

    if root is None:
        pass
    elif root.is_leaf:
        print(root)
    else:
        for child in root.children:
            traverse_octree(child, depth, max_depth)
    depth[0] -= 1


# 功能：通过递归的方式构建octree
# 输入：
#     root：根节点
#     db：原始数据
#     center: 中心
#     extent: 当前分割区间
#     point_indices: 点的key
#     leaf_size: scale
#     min_extent: 最小分割区间
def octree_recursive_build(root, db, center, extent, point_indices, leaf_size, min_extent):
    if len(point_indices) == 0:
        return None

    if root is None:
        root = Octant([None for i in range(8)], center, extent, point_indices, is_leaf=True)

    # determine whether to split this octant
    if len(point_indices) <= leaf_size or extent <= min_extent:
        root.is_leaf = True
    else:
        # 作业4
        # 屏蔽开始
        root.is_leaf = False
        children_point_indices = [[] for i in range(8)]
        for point_index in point_indices:
            point_coor = db[point_index]
            octant_index = 0
            if point_coor[0] > center[0]:
                octant_index = octant_index | 1
            if point_coor[1] > center[1]:
                octant_index = octant_index | 2
            if point_coor[2] > center[2]:
                octant_index = octant_index | 4
            # 这里按位 或 运算 可以做到 每增加一个维度,那么索引 增加 2倍, 所以最终结果就是[0-7],
            # 也就是8个子节点
            children_point_indices[octant_index].append(point_index)

        # creat children node
        factor = [-0.5, 0.5]
        for i in range(8):
            # 既然上面是按x y z顺序确定的索引,这里就按这个顺序去计算就行了
            # 二进制的第一位决定x轴坐标,第二位决定y轴坐标,第三位决定z轴坐标
            child_center_x = center[0] + factor[(i & 1) > 0] * extent
            child_center_y = center[1] + factor[(i & 2) > 0] * extent
            child_center_z = center[2] + factor[(i & 4) > 0] * extent
            child_extent = 0.5 * extent
            child_center = np.asarray([child_center_x, child_center_y, child_center_z])
            root.children[i] = octree_recursive_build(root, db, child_center, child_extent,
                                                      children_point_indices[i], leaf_size, min_extent)

        # 屏蔽结束
    return root


# 功能：判断当前query区间是否在octant内
# 输入：
#     query: 索引信息
#     radius：索引半径
#     octant：octree
# 输出：
#     判断结果，即True/False
def inside(query: np.ndarray, radius: float, octant: Octant):
    """
    Determines if the query ball is inside the octant
    :param query:
    :param radius:
    :param octant:
    :return:
    """
    query_offset = query - octant.center
    query_offset_abs = np.fabs(query_offset)
    possible_space = query_offset_abs + radius
    return np.all(possible_space < octant.extent)


# 功能：判断当前query区间是否和octant有重叠部分
# 输入：
#     query: 索引信息
#     radius：索引半径
#     octant：octree
# 输出：
#     判断结果，即True/False
def overlaps(query: np.ndarray, radius: float, octant: Octant):
    """
    Determines if the query ball overlaps with the octant
    :param query:
    :param radius:
    :param octant:
    :return:
    """
    query_offset = query - octant.center
    query_offset_abs = np.fabs(query_offset)

    # completely outside, since query is outside the relevant area
    max_dist = radius + octant.extent
    if np.any(query_offset_abs > max_dist):
        return False

    # if pass the above check, consider the case that the ball is contacting the face of the octant
    if np.sum((query_offset_abs < octant.extent).astype(np.int)) >= 2:
        return True

    # conside the case that the ball is contacting the edge or corner of the octant
    # since the case of the ball center (query) inside octant has been considered,
    # we only consider the ball center (query) outside octant
    x_diff = max(query_offset_abs[0] - octant.extent, 0)
    y_diff = max(query_offset_abs[1] - octant.extent, 0)
    z_diff = max(query_offset_abs[2] - octant.extent, 0)

    return x_diff * x_diff + y_diff * y_diff + z_diff * z_diff < radius * radius


# 功能：判断当前query是否包含octant
# 输入：
#     query: 索引信息
#     radius：索引半径
#     octant：octree
# 输出：
#     判断结果，即True/False
def contains(query: np.ndarray, radius: float, octant: Octant):
    """
    Determine if the query ball contains the octant
    :param query:
    :param radius:
    :param octant:
    :return:
    """
    query_offset = query - octant.center
    query_offset_abs = np.fabs(query_offset)

    query_offset_to_farthest_corner = query_offset_abs + octant.extent
    return np.linalg.norm(query_offset_to_farthest_corner) < radius


# 功能：在octree中查找信息
# 输入：
#    root: octree
#    db：原始数据
#    result_set: 索引结果
#    query：索引信息
def octree_radius_search_fast(root: Octant, db: np.ndarray, result_set: RadiusNNResultSet, query: np.ndarray):
    if root is None:
        return False

    # 作业5
    # 提示：尽量利用上面的inside、overlaps、contains等函数
    # 屏蔽开始

    # 先判断如果当前 查询球 包含了 当前节点,那么就不需要在查询当前节点的子节点了
    # 直接比较当前节点包含的所有 points

    if contains(query, result_set.worstDist(), root):
        all_points_in = db[root.point_indices]
        diff = np.linalg.norm(np.expand_dims(query, 0) - all_points_in, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        return False  # 包含当前节点,并不意味着不包含其他节点,所以返回 False

    # 如果是叶子节点,那么 直接比较内部所有points

    if root.is_leaf and len(root.point_indices) > 0:
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        return inside(query, result_set.worstDist(), root)

    # 当前root 不是叶子, 也没有被 查询球包含,那么就需要查询其每个子节点
    for i, current_octant in enumerate(root.children):
        if current_octant is None:
            continue
        # 如果当前子节点与 查询球没有交集,那么就忽略这个子节点
        if not overlaps(query, result_set.worst_dist(), current_octant):
            continue
        if octree_radius_search(current_octant, db, result_set, query):
            return True
    # 屏蔽结束

    return inside(query, result_set.worstDist(), root)


# 功能：在octree中查找radius范围内的近邻
# 输入：
#     root: octree
#     db: 原始数据
#     result_set: 搜索结果
#     query: 搜索信息
def octree_radius_search(root: Octant, db: np.ndarray, result_set: RadiusNNResultSet, query: np.ndarray):
    if root is None:
        return False

    if root.is_leaf and len(root.point_indices) > 0:
        # compare the contents of a leaf
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        # check whether we can stop search now
        return inside(query, result_set.worstDist(), root)

    # 作业6
    # 屏蔽开始
    # 半径固定的情况下查询最近邻点, 和 knn是一样的策略, 只是此时这个最坏距离是固定的
    # 既然不是叶子节点,那么我们就应该进入下一级子节点,
    # 方法是判断当前查询点,属于哪个子节点, 肯定是查询当前查询点所在的子节点内的点 得到 最近邻点的概率大
    # 所以先要判断当前查询点谓语哪个子节点呢
    octant_index = 0
    if query[0] > root.center[0]:
        octant_index = octant_index | 1
    if query[0] > root.center[1]:
        octant_index = octant_index | 2
    if query[0] > root.center[2]:
        octant_index = octant_index | 4

    if octree_radius_search(root.children[octant_index], db, result_set, query):
        return True  # 如果条件为真,以为着搜索可以终止

    # 如果当前查询点 和 最坏距离所构成的球体,不完全被某个子节点所包围,那么,就应该查看和当前子节点 平级的
    # 其他子节点

    for i, current_octant in enumerate(root.children):
        # 我们需要的是当前 octant_index 平级的子节点
        # 同时 子节点有肯能是根本没有被构建的,也就是none
        if i == octant_index or current_octant is None:
            continue
        # 如果当前子节点与 查询球没有交集,那么就忽略这个子节点
        if not overlaps(query, result_set.worst_dist(), current_octant):
            continue
        if octree_radius_search(current_octant, db, result_set, query):
            return True
    # 屏蔽结束

    # final check of if we can stop search
    return inside(query, result_set.worstDist(), root)


# 功能：在octree中查找最近的k个近邻
# 输入：
#     root: octree
#     db: 原始数据
#     result_set: 搜索结果
#     query: 搜索信息
def octree_knn_search(root: Octant, db: np.ndarray, result_set: KNNResultSet, query: np.ndarray):
    if root is None:
        return False

    if root.is_leaf and len(root.point_indices) > 0:
        # compare the contents of a leaf
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        # check whether we can stop search now
        return inside(query, result_set.worstDist(), root)

    # 作业7
    # 屏蔽开始 ****
    # 既然不是叶子节点,那么我们就应该进入下一级子节点,
    # 方法是判断当前查询点,属于哪个子节点, 肯定是查询当前查询点所在的子节点内的点 得到 最近邻点的概率大
    # 所以先要判断当前查询点谓语哪个子节点呢
    octant_index = 0
    if query[0] > root.center[0]:
        octant_index = octant_index | 1
    if query[0] > root.center[1]:
        octant_index = octant_index | 2
    if query[0] > root.center[2]:
        octant_index = octant_index | 4

    if octree_knn_search(root.children[octant_index], db, result_set, query):
        return True  # 如果条件为真,以为着搜索可以终止

    # 如果当前查询点 和 最坏距离所构成的球体,不完全被某个子节点所包围,那么,就应该查看和当前子节点 平级的
    # 其他子节点

    for i, current_octant in enumerate(root.children):
        # 我们需要的是当前 octant_index 平级的子节点
        # 同时 子节点有肯能是根本没有被构建的,也就是none
        if i == octant_index or current_octant is None:
            continue
        # 如果当前子节点与 查询球没有交集,那么就忽略这个子节点
        if not overlaps(query, result_set.worst_dist(), current_octant):
            continue
        if octree_knn_search(current_octant, db, result_set, query):
            return True
    # 屏蔽结束*****

    # final check of if we can stop search
    return inside(query, result_set.worstDist(), root)


# 功能：构建octree，即通过调用octree_recursive_build函数实现对外接口
# 输入：
#    dp_np: 原始数据
#    leaf_size：scale
#    min_extent：最小划分区间
def octree_construction(db_np, leaf_size, min_extent):
    N, dim = db_np.shape[0], db_np.shape[1]
    db_np_min = np.amin(db_np, axis=0)
    db_np_max = np.amax(db_np, axis=0)
    db_extent = np.max(db_np_max - db_np_min) * 0.5
    db_center = np.mean(db_np, axis=0)

    root = None
    root = octree_recursive_build(root, db_np, db_center, db_extent, list(range(N)),
                                  leaf_size, min_extent)

    return root


def main():
    # configuration
    db_size = 64000
    dim = 3
    leaf_size = 4
    min_extent = 0.0001
    k = 8

    db_np = np.random.rand(db_size, dim)

    root = octree_construction(db_np, leaf_size, min_extent)

    # depth = [0]
    # max_depth = [0]
    # traverse_octree(root, depth, max_depth)
    # print("tree max depth: %d" % max_depth[0])

    # query = np.asarray([0, 0, 0])
    # result_set = KNNResultSet(capacity=k)
    # octree_knn_search(root, db_np, result_set, query)
    # print(result_set)
    #
    # diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
    # nn_idx = np.argsort(diff)
    # nn_dist = diff[nn_idx]
    # print(nn_idx[0:k])
    # print(nn_dist[0:k])

    begin_t = time.time()
    print("Radius search normal:")
    for i in range(100):
        query = np.random.rand(3)
        result_set = RadiusNNResultSet(radius=0.5)
        octree_radius_search(root, db_np, result_set, query)
    # print(result_set)
    print("Search takes %.3fms\n" % ((time.time() - begin_t) * 1000))

    begin_t = time.time()
    print("Radius search fast:")
    for i in range(100):
        query = np.random.rand(3)
        result_set = RadiusNNResultSet(radius=0.5)
        octree_radius_search_fast(root, db_np, result_set, query)
    # print(result_set)
    print("Search takes %.3fms\n" % ((time.time() - begin_t) * 1000))


if __name__ == '__main__':
    main()
