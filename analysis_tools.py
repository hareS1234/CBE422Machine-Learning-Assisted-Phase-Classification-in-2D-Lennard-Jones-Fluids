# analysis_tools.py
import numpy as np

def periodic_distance(x1, x2, box_length):
    # x1: shape (2,)
    # x2: shape (M,2)
    # This calculates the minimal periodic distance between a single point x1 and multiple points x2
    delta = x1 - x2
    delta -= np.round(delta/box_length)*box_length
    return np.sqrt((delta**2).sum(axis=-1))

def compute_neighbors(positions, box_length, cutoff=1.5):
    N = positions.shape[0]
    neighbors = [[] for _ in range(N)]
    for i in range(N):
        rij = periodic_distance(positions[i], positions[i+1:], box_length)
        idx = np.where(rij < cutoff)[0] + i + 1
        for j in idx:
            neighbors[i].append(j)
            neighbors[j].append(i)
    return neighbors

def compute_Q6(positions, box_length, cutoff=1.5):
    # positions: Nx2 array of particle coordinates.
    neighbors = compute_neighbors(positions, box_length, cutoff)
    N = positions.shape[0]
    psi6_values = []
    for i in range(N):
        neighs = neighbors[i]
        if len(neighs) < 1:
            psi6_values.append(0.0+0.0j)
            continue
        angles = []
        ref_pos = positions[i]
        for n in neighs:
            dx, dy = positions[n] - ref_pos
            dx -= np.round(dx/box_length)*box_length
            dy -= np.round(dy/box_length)*box_length
            angle = np.arctan2(dy, dx)
            angles.append(angle)
        angles = np.array(angles)
        psi6 = np.mean(np.exp(1j * 6 * angles))
        psi6_values.append(psi6)
    psi6_values = np.array(psi6_values)
    Q6 = np.abs(np.mean(psi6_values))
    return Q6, np.abs(psi6_values)

def compute_rdf(positions_list, box_length, r_max, dr=0.1):
    # This is for particle coordinates. Not used in the current lattice project.
    # positions_list: a list of Nx2 arrays
    bins = np.arange(0, r_max+dr, dr)
    hist = np.zeros(len(bins)-1)
    count = 0
    N = positions_list[0].shape[0]
    density = N/(box_length**2)

    for pos in positions_list:
        for i in range(N):
            rij = periodic_distance(pos[i], pos[i+1:], box_length)
            hist += np.histogram(rij, bins=bins)[0]
        count += 1

    hist /= count
    shell_areas = 2*np.pi*((bins[1:]+bins[:-1])/2)*dr
    ideal_counts = density * shell_areas * N/2
    g_r = hist/ideal_counts
    r = (bins[:-1] + bins[1:])/2
    return r, g_r

def cluster_analysis(positions, box_length, cutoff=1.5):
    N = positions.shape[0]
    visited = np.zeros(N, dtype=bool)
    adj_list = [[] for _ in range(N)]

    for i in range(N):
        rij = periodic_distance(positions[i], positions[i+1:], box_length)
        idx = np.where(rij < cutoff)[0] + i+1
        for n in idx:
            adj_list[i].append(n)
            adj_list[n].append(i)

    def dfs(start):
        stack = [start]
        cluster = []
        while stack:
            node = stack.pop()
            if not visited[node]:
                visited[node] = True
                cluster.append(node)
                for neigh in adj_list[node]:
                    if not visited[neigh]:
                        stack.append(neigh)
        return cluster

    largest_cluster = 0
    for i in range(N):
        if not visited[i]:
            c = dfs(i)
            if len(c) > largest_cluster:
                largest_cluster = len(c)

    return largest_cluster / N
