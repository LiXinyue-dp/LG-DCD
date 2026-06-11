import numpy as np
import networkx as nx
import secrets
import os
import heapq
import itertools
from scipy import special


def get_mat(data_path,node_num,time_index):
    # Initialize the adjacency matrix
    mat0_index=np.zeros((node_num, node_num))
    # Read the network snapshot of current time slice
    f = open('%s_%d.txt' % (data_path, time_index))
    for line in f:
        s = line.strip().split('\n')
        temp = s[0].split()
        temp = [int(t) for t in temp]
        src = temp[0]
        tar = temp[1]
        # Update the adjacency matrix
        if mat0_index[src-1, tar-1] != 0:
            mat0_index[src-1, tar-1] += 1
            mat0_index[tar-1, src-1] += 1
        else:
            mat0_index[src-1, tar-1] = 1
            mat0_index[tar-1, src-1] = 1
    f.close()

    mid = []
    for i in range(node_num):
        mid.append(i)
    return mat0_index,mid

def geometric(value, sensitivity, epsilon):
    scale = - epsilon / sensitivity
    rng = secrets.SystemRandom()
    unif_rv = rng.random() - 0.5
    unif_rv *= 1 + np.exp(scale)
    sgn = -1 if unif_rv < 0 else 1
    # Use formula for geometric distribution, with ratio of exp(-epsilon/sensitivity)
    return int(np.round(value + sgn * np.floor(np.log(sgn * unif_rv) / scale)))


def adjust_element(vector):
    non_zero_indices = np.nonzero(vector)[0]
    index = np.random.choice(non_zero_indices)
    if np.random.rand() < 0.5:
        vector[index] += 1
    else:
        vector[index] -= 1
    return vector


def SW(ori_samples,eps,h,l=0):
    ee = np.exp(eps)
    w = ((eps * ee) - ee + 1) / (2 * ee * (ee - 1 - eps)) * 2  # w = 2b
    p = ee / (w * ee + 1)
    q = 1 / (w * ee + 1)

    samples = (ori_samples - l) / (h - l)
    randoms = np.random.uniform(0, 1, len(samples))

    noisy_samples = np.zeros_like(samples)

    # report
    index = randoms <= (q * samples)
    noisy_samples[index] = randoms[index] / q - w / 2
    index = randoms > (q * samples)
    noisy_samples[index] = (randoms[index] - q * samples[index]) / p + samples[index] - w / 2
    index = randoms > q * samples + p * w  # v'\in [v+b,1+b]=[v-b,1+b]-[v-b,v+b]=1-q*v-2b*p=1-(q*v+2b*p)
    noisy_samples[index] = (randoms[index] - q * samples[index] - p * w) / q + samples[index] + w / 2
    return noisy_samples


def transform_matrix(domain_bins,randomized_bins,eps):
    ee = np.exp(eps)
    w = ((eps * ee) - ee + 1) / (2 * ee * (ee - 1 - eps)) * 2  # w = 2b
    p = ee / (w * ee + 1)
    q = 1 / (w * ee + 1)
    m = randomized_bins
    n = domain_bins
    m_cell = (1 + w) / m
    n_cell = 1 / n
    transform = np.ones((m, n)) * q * m_cell
    for i in range(n):
        left_most_v = (i * n_cell)
        right_most_v = ((i + 1) * n_cell)

        ll_bound = int(left_most_v / m_cell)
        lr_bound = int((left_most_v + w) / m_cell)
        rl_bound = int(right_most_v / m_cell)
        rr_bound = int((right_most_v + w) / m_cell)

        ll_v = left_most_v - w / 2
        rl_v = right_most_v - w / 2
        l_p = ((ll_bound + 1) * m_cell - w / 2 - ll_v) * (p - q) + q * m_cell
        r_p = ((rl_bound + 1) * m_cell - w / 2 - rl_v) * (p - q) + q * m_cell
        if rl_bound > ll_bound:
            transform[ll_bound, i] = (l_p - q * m_cell) * ((ll_bound + 1) * m_cell - w / 2 - ll_v) / n_cell * 0.5 + q * m_cell
            transform[ll_bound + 1, i] = p * m_cell - (p * m_cell - r_p) * (rl_v - ((ll_bound + 1) * m_cell - w / 2)) / n_cell * 0.5
        else:
            transform[ll_bound, i] = (l_p + r_p) / 2
            transform[ll_bound + 1, i] = p * m_cell

        lr_v = left_most_v + w / 2
        rr_v = right_most_v + w / 2
        r_p = (rr_v - (rr_bound * m_cell - w / 2)) * (p - q) + q * m_cell
        l_p = (lr_v - (lr_bound * m_cell - w / 2)) * (p - q) + q * m_cell
        if rr_bound > lr_bound:
            if rr_bound < m:
                transform[rr_bound, i] = (r_p - q * m_cell) * (rr_v - (rr_bound * m_cell - w / 2)) / n_cell * 0.5 + q * m_cell

            transform[rr_bound - 1, i] = p * m_cell - (p * m_cell - l_p) * ((rr_bound * m_cell - w / 2) - lr_v) / n_cell * 0.5

        else:
            transform[rr_bound, i] = (l_p + r_p) / 2
            transform[rr_bound - 1, i] = p * m_cell

        if rr_bound - 1 > ll_bound + 2:
            transform[ll_bound + 2: rr_bound - 1, i] = p * m_cell

    return transform


def EMS(domain_bins, norm_hist, transform):
    if sum(norm_hist) == 0:
        theta = np.zeros(domain_bins)
        return theta
    else:
        max_iteration = 10000
        loglikelihood_threshold = 1e-3
        # smoothing matrix
        smoothing_factor = 2
        binomial_tmp = [special.binom(smoothing_factor, k) for k in range(smoothing_factor + 1)]
        smoothing_matrix = np.zeros((domain_bins, domain_bins))
        central_idx = int(len(binomial_tmp) / 2)
        for i in range(int(smoothing_factor / 2)):
            smoothing_matrix[i, : central_idx + i + 1] = binomial_tmp[central_idx - i:]
        for i in range(int(smoothing_factor / 2), domain_bins - int(smoothing_factor / 2)):
            smoothing_matrix[i, i - central_idx: i + central_idx + 1] = binomial_tmp
        for i in range(domain_bins - int(smoothing_factor / 2), domain_bins):
            remain = domain_bins - i - 1
            smoothing_matrix[i, i - central_idx + 1:] = binomial_tmp[: central_idx + remain]
        row_sum = np.sum(smoothing_matrix, axis=1)
        smoothing_matrix = (smoothing_matrix.T / row_sum).T

        # EMS
        theta = np.ones(domain_bins) / float(domain_bins)
        theta_old = np.zeros(domain_bins)
        r = 0
        sample_size = sum(norm_hist)
        old_logliklihood = 0

        while np.linalg.norm(theta_old - theta, ord=1) > 1 / sample_size and r < max_iteration:
            theta_old = np.copy(theta)
            X_condition = np.matmul(transform, theta_old)
            TMP = transform.T / X_condition
            P = np.copy(np.matmul(TMP, norm_hist))
            P = P * theta_old
            theta = np.copy(P / sum(P))
            # Smoothing step
            theta = np.matmul(smoothing_matrix, theta)
            theta = theta / sum(theta)
            logliklihood = np.inner(norm_hist, np.log(np.matmul(transform, theta)))
            imporve = logliklihood - old_logliklihood

            if r > 1 and abs(imporve) < loglikelihood_threshold:
                # print("stop when", imporve / old_logliklihood, loglikelihood_threshold)
                break

            old_logliklihood = logliklihood
            r += 1
        return theta


def get_bin_index(x, min_val, max_val, k):
    bin_width = (max_val - min_val) / k
    index = (x - min_val) / bin_width
    index = max(0, min(index, k - 1))
    return int(index)


def divide_interval(min_val, max_val, k):
    bin_width = (max_val - min_val) / k
    bins = np.arange(min_val, max_val + bin_width, bin_width)
    return bins


# Post processing
def FO_pp_sec23(data_noise):
    data = norm_sub_deal(data_noise)
    return data


def norm_sub_deal(data):
    if (len(data) == 0):
        return np.array([])
    data = np.array(data, dtype=np.int32)
    data_min = np.min(data)
    data_sum = np.sum(data)
    delta_m = 1 - data_min

    if delta_m > 0:
        dm = 100000000
        data_seq = np.zeros([len(data)], dtype=np.int32)
        for i in range(0, delta_m):
            data_t = data - i
            data_t[data_t < 0] = 0
            data_t_s = np.sum(data_t)
            dt = np.abs(data_t_s - data_sum)
            if dt < dm:
                dm = dt
                data_seq = data_t
                if dt == 0:
                    break
    else:
        data_seq = data

    data_seq = np.round(data_seq).astype(int)
    return data_seq


def neq_elem_number(array_now, array_before):
    neq_count = np.sum([1 for a, b in zip(array_now, array_before) if a != b])
    return neq_count


def sort_node(presample_node):
    sorted_items = sorted(presample_node, key=presample_node.get, reverse=True)
    return sorted_items


# calculate the diameter
def cal_diam(mat):
    mat_graph = nx.from_numpy_array(mat,create_using=nx.Graph)
    max_diam = 0
    for com in nx.connected_components(mat_graph):
        com_list = list(com)
        mat_sub = mat[np.ix_(com_list,com_list)]
        sub_g = nx.from_numpy_array(mat_sub,create_using=nx.Graph)
        diam = nx.diameter(sub_g)
        if diam > max_diam:
            max_diam = diam
    return max_diam


# calculate the KL divergence
def cal_kl(A,B):
    p = A / sum(A)
    if sum(B) != 0:
        q = B / sum(B)
    else:
        q = B
    if A.shape[0] > B.shape[0]:
        q = np.pad(q,(0,p.shape[0]-q.shape[0]),'constant',constant_values=(0,0))
    elif A.shape[0] < B.shape[0]:
        p = np.pad(p,(0,q.shape[0]-p.shape[0]),'constant',constant_values=(0,0))
    kl = p * np.log((p+np.finfo(np.float64).eps)/(q+np.finfo(np.float64).eps))
    kl = np.sum(kl)
    return kl


# calculate the RE
def cal_rel(A,B):
    eps = 0.000000000000001
    A = np.float64(A)
    B = np.float64(B)
    res = abs((A-B)/(A+eps))
    return res

# calculate the RMSE
def cal_RMSE(A,B):
    A = np.array(A)
    B = np.array(B)
    rmse = np.sqrt(np.mean((A - B) ** 2))
    return rmse


class PriorityQueue(object):
    def __init__(self):
        self.pq = []
        self.entry_finder = {}
        self.REMOVED = '<removed-task>'
        self.counter = itertools.count()

    def add_task(self, task, priority=0):
        if task in self.entry_finder:
            self.remove_task(task)
        count = next(self.counter)
        entry = [priority, count, task]
        self.entry_finder[task] = entry
        heapq.heappush(self.pq, entry)

    def remove_task(self, task):
        entry = self.entry_finder.pop(task)
        entry[-1] = self.REMOVED

    def pop_item(self):
        while self.pq:
            priority, count, task = heapq.heappop(self.pq)
            if task is not self.REMOVED:
                del self.entry_finder[task]
                return task, priority
        raise KeyError('pop from an empty priority queue')

    def is_empty(self):
        return not bool(self.entry_finder)


def write_edge_txt(mat0, mid, file_name):
    rows, cols = np.where(mat0 != 0)
    if os.path.exists(file_name):
        raise FileExistsError(f"File '{file_name}' already exists, refuse to overwrite！")
    else:
        with open(file_name, 'w') as f:
            for i in range(len(rows)):
                u = mid[rows[i]]
                v = mid[cols[i]]
                weight = mat0[rows[i], cols[i]]
                if u <= v:
                    f.write(f"{u}\t{v}\t{weight:.6f}\n")

