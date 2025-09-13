import numpy as np


class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    Story data with its priority in the tree.
    """
    # 用于跟踪下一个要插入的数据位置，初始化为0
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        # 完全二叉树的节点数为2n-1，n为叶子节点数
        # 叶子节点存储优先级，非叶子节点存储子节点的优先级之和
        # 所有节点都存放在一个数组中，数组的大小为2n-1，通过索引可以模拟出完全二叉树
        # 对于任何一个父节点i，它的左子节点的索引是2 * i + 1，右子节点的索引是2 * i + 2。
        # 对于任何一个子节点j，它的父节点的索引是(j - 1) // 2
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        # 存储所有经验数据即(s, a, r, s_, done)，大小为capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    # 向求和数中添加新的优先级值
    def add(self, p, data):
        # 计算要插入数据的索引
        # 指针为0时是第一个叶子节点
        tree_idx = self.data_pointer + self.capacity - 1
        # 添加数据
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        # 计算新的优先级值与旧的优先级值的差值
        change = p - self.tree[tree_idx]
        # 然后将新优先级更新到新位置上
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            # 只要不是根节点就一直循环
            # 计算父节点的索引更新父节点的优先级值并一直向上更新直至更新完整棵树
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    # 根据给定的优先级v，在树中找到对应的叶子节点
    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            # 从根节点开始搜索，找到左右子节点的索引
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        # 根据v值，返回叶子节点的索引，叶子节点的优先级值，叶子节点存储的数据
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    # 将经验误差换变为优先级
    alpha = 0.7  # [0~1] convert the importance of TD error to priority
    beta = 0.5  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    # 防止经验值过大
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.sample_count = 0

    def __len__(self):
        return self.sample_count

    # 将新的经验（transition）存储到优先级经验回放（Prioritized Experience Replay）的内存中
    def store(self, transition):
        # 找到所有经验中优先级最大的值
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            # 避免优先级为0
            max_p = self.abs_err_upper
            # 新的经验和优先级添加到树中
        self.tree.add(max_p, transition)   # set the max p for new p
        # 维护经验计数
        self.sample_count = min(self.sample_count + 1, self.tree.capacity)

    # 得到经验的索引、经验和IS权重
    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty((n, 1))
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        a = self.tree.tree[-self.tree.capacity:]
        min_prob = np.min(a[a != 0]) / self.tree.total_p     # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data

        return b_idx, b_memory, ISWeights

    # 更新存储在优先级经验回放（Prioritized Experience Replay）内存中的经验的优先级。
    # 更新树中的优先级
    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)
