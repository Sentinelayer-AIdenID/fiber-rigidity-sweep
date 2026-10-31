"""Periodic triangular networks with contracted phantom bypasses.

Lengths are in units of the original triangular lattice spacing. At each
lattice site one randomly selected fiber bypasses the crosslink: its two
segments are contracted into a longer straight segment. The other two fibers
share the crosslink. Thus the undiluted phantom network has coordination four.
"""

from dataclasses import dataclass
from collections import deque

import numpy as np


@dataclass
class Network:
    positions: np.ndarray
    box: np.ndarray
    edges: np.ndarray
    images: np.ndarray
    triplets: np.ndarray
    triplet_images: np.ndarray
    rest_lengths: np.ndarray
    bend_lengths: np.ndarray
    seed: int = 0
    z_target: float = 6.0

    @property
    def z_actual(self):
        return 2.0 * len(self.edges) / len(self.positions) if len(self.positions) else 0.0

    @property
    def l_c(self):
        return float(np.mean(self.rest_lengths)) if len(self.rest_lengths) else 1.0


def _dimensions(nx, ny):
    if isinstance(nx, bool) or isinstance(ny, bool) or int(nx) != nx or int(ny) != ny:
        raise ValueError("nx and ny must be integers")
    nx, ny = int(nx), int(ny)
    if nx < 3 or ny < 4 or ny % 2:
        raise ValueError("nx >= 3 and even ny >= 4 are required")
    return nx, ny


def _lattice(nx, ny, bypass=None, seed=0, z_target=6.0):
    nx, ny = _dimensions(nx, ny)
    n = nx * ny
    j, i = np.divmod(np.arange(n), nx)
    positions = np.column_stack((i + 0.5 * (j % 2), np.sqrt(3.0) / 2.0 * j))
    box = np.diag([float(nx), np.sqrt(3.0) / 2.0 * ny])
    edges, images, lengths, triplets, triplet_images, bend_lengths = [], [], [], [], [], []
    for direction in range(3):
        ni = i + (1 if direction == 0 else j % 2 - (direction == 2))
        nj = j + (direction != 0)
        neighbor = (nj % ny) * nx + ni % nx
        offset = np.column_stack((ni // nx, nj // ny)).astype(np.int64)
        start = np.arange(n) if bypass is None else np.flatnonzero(bypass != direction)
        end = neighbor[start].copy()
        shift = offset[start].copy()
        length = np.ones(len(start), dtype=np.float64)
        # Contract straight bypass sites without adding fictitious crosslinks.
        if bypass is not None:
            skip = bypass[end] == direction
            while np.any(skip):
                shift[skip] += offset[end[skip]]
                end[skip] = neighbor[end[skip]]
                length[skip] += 1.0
                skip = bypass[end] == direction
        edges.append(np.column_stack((start, end)))
        images.append(shift)
        lengths.append(length)
        # Every active site has exactly one preceding edge along this fiber.
        predecessor = np.full(n, -1, dtype=np.int64)
        predecessor[end] = np.arange(len(start))
        previous = predecessor[start]
        triplets.append(np.column_stack((start[previous], start, end)))
        triplet_images.append(np.stack((shift[previous], shift), axis=1))
        bend_lengths.append(0.5 * (length[previous] + length))
    return Network(positions, box, np.concatenate(edges), np.concatenate(images),
                   np.concatenate(triplets), np.concatenate(triplet_images),
                   np.concatenate(lengths), np.concatenate(bend_lengths), seed, z_target)


def triangular_lattice(nx, ny):
    """Undiluted, nonphantom triangular reference with unit bonds and z=6."""
    return _lattice(nx, ny)


def _core_mask(n, edges, mask):
    """The graph 2-core; degree-two bending hinges are intentionally retained."""
    mask = np.asarray(mask, dtype=bool).copy()
    while True:
        degree = np.bincount(edges[mask].ravel(), minlength=n)
        updated = mask & (degree[edges[:, 0]] >= 2) & (degree[edges[:, 1]] >= 2)
        if np.array_equal(mask, updated):
            return mask
        mask = updated


def _load_bearing_edges(n, edges, images):
    """Keep winding blocks and paths joining them in the block-cut forest.

    Iterative Tarjan traversal finds biconnected edge blocks in linear time.
    Integer DFS image potentials distinguish genuine winding cycles from finite
    cycles crossing a seam. Finite leaf blocks are then peeled, removing loops
    attached through a single articulation or bridge. This is a graph-topology
    cleanup, not a rigidity-matrix analysis of the remaining internal modes.
    """
    if not len(edges):
        return np.zeros(0, dtype=bool)
    m = len(edges)
    row = np.concatenate((edges[:, 0], edges[:, 1]))
    col = np.concatenate((edges[:, 1], edges[:, 0]))
    # Keep parallel edges distinct; they can themselves witness a winding cycle.
    order = np.argsort(row, kind="stable")
    neighbors = col[order]
    shifts = np.concatenate((images, -images))[order]
    edge_ids = np.tile(np.arange(m), 2)[order]
    pointers = np.r_[0, np.cumsum(np.bincount(row, minlength=n))]
    discovery = np.full(n, -1, dtype=np.int64)
    low = np.zeros(n, dtype=np.int64)
    parent = np.full(n, -1, dtype=np.int64)
    parent_edge = np.full(n, -1, dtype=np.int64)
    cursor = pointers[:-1].copy()
    potentials = np.zeros((n, 2), dtype=np.int64)
    edge_winds = np.zeros(m, dtype=bool)
    block = np.full(m, -1, dtype=np.int64)
    block_winds = []
    edge_stack = []
    clock = 0
    for root in np.flatnonzero(np.diff(pointers)):
        if discovery[root] >= 0:
            continue
        discovery[root] = low[root] = clock
        clock += 1
        stack = [int(root)]
        while stack:
            node = stack[-1]
            index = cursor[node]
            if index < pointers[node + 1]:
                cursor[node] += 1
                other = neighbors[index]
                edge = edge_ids[index]
                if edge == parent_edge[node]:
                    continue
                if other == node:
                    if block[edge] < 0:
                        block[edge] = len(block_winds)
                        block_winds.append(bool(np.any(shifts[index])))
                elif discovery[other] < 0:
                    parent[other], parent_edge[other] = node, edge
                    potentials[other] = potentials[node] + shifts[index]
                    discovery[other] = low[other] = clock
                    clock += 1
                    edge_stack.append(int(edge))
                    stack.append(int(other))
                elif discovery[other] < discovery[node]:
                    low[node] = min(low[node], discovery[other])
                    edge_stack.append(int(edge))
                    edge_winds[edge] = np.any(potentials[node] + shifts[index] != potentials[other])
            else:
                stack.pop()
                previous = parent[node]
                if previous >= 0:
                    low[previous] = min(low[previous], low[node])
                    if low[node] >= discovery[previous]:
                        winds = False
                        while edge_stack:
                            edge = edge_stack.pop()
                            block[edge] = len(block_winds)
                            winds |= bool(edge_winds[edge])
                            if edge == parent_edge[node]:
                                break
                        block_winds.append(winds)
    if np.any(block < 0):
        raise RuntimeError("incomplete biconnected decomposition")
    protected_blocks = np.asarray(block_winds, dtype=bool)
    if protected_blocks.all() or not protected_blocks.any():
        return protected_blocks[block]

    # Each block is joined to its articulation nodes. Nonarticulation nodes
    # are omitted because they are leaves with no role in connecting blocks.
    membership = np.unique(np.repeat(block, 2) * n + edges.ravel())
    member_nodes, member_blocks = membership % n, membership // n
    articulation = np.bincount(member_nodes, minlength=n) > 1
    use = articulation[member_nodes]
    left, right = member_nodes[use], n + member_blocks[use]
    size = n + len(block_winds)
    forest_row = np.r_[left, right]
    forest_col = np.r_[right, left]
    forest_order = np.argsort(forest_row, kind="stable")
    forest_neighbors = forest_col[forest_order]
    degree = np.bincount(forest_row, minlength=size)
    forest_ptr = np.r_[0, np.cumsum(degree)]
    protected = np.r_[np.zeros(n, dtype=bool), protected_blocks]
    active = np.ones(size, dtype=bool)
    # Unused original nodes need not enter the queue.
    eligible = np.r_[articulation, np.ones(len(block_winds), dtype=bool)]
    queue = deque(np.flatnonzero(eligible & ~protected & (degree <= 1)).tolist())
    while queue:
        node = queue.popleft()
        if not active[node]:
            continue
        active[node] = False
        for index in range(forest_ptr[node], forest_ptr[node + 1]):
            other = forest_neighbors[index]
            if active[other]:
                degree[other] -= 1
                if degree[other] <= 1 and not protected[other]:
                    queue.append(int(other))
    return active[n + block]


def _subset(network, edge_mask):
    edges = network.edges[edge_mask]
    nodes = np.unique(edges)
    remap = np.full(len(network.positions), -1, dtype=np.int64)
    remap[nodes] = np.arange(len(nodes))
    # A triplet survives only if both its original oriented segments survive.
    # Include images in keys, since distinct wrapping edges may share endpoints.
    dtype = np.dtype([(name, np.int64) for name in ("i", "j", "ix", "iy")])
    def keys(pair, offsets):
        raw = np.ascontiguousarray(np.column_stack((pair, offsets)), dtype=np.int64)
        return raw.view(dtype).ravel()
    surviving = keys(edges, network.images[edge_mask])
    left = keys(network.triplets[:, :2], network.triplet_images[:, 0])
    right = keys(network.triplets[:, 1:], network.triplet_images[:, 1])
    tmask = np.isin(left, surviving) & np.isin(right, surviving)
    return Network(network.positions[nodes].copy(), network.box.copy(), remap[edges],
                   network.images[edge_mask].copy(), remap[network.triplets[tmask]],
                   network.triplet_images[tmask].copy(), network.rest_lengths[edge_mask].copy(),
                   network.bend_lengths[tmask].copy(), network.seed, network.z_target)


def cleanup(network, edge_mask=None):
    """Remove dangling bonds, finite components, and singly attached rattlers.

    Retain biconnected blocks with periodic winding and paths connecting those
    blocks. Degree-two hinges remain physical bending degrees of freedom;
    cleanup does not remove internal floppy modes of the wrapping core.
    """
    if edge_mask is None:
        edge_mask = np.ones(len(network.edges), dtype=bool)
    mask = _core_mask(len(network.positions), network.edges, edge_mask)
    retained = _load_bearing_edges(len(network.positions), network.edges[mask], network.images[mask])
    mask[np.flatnonzero(mask)] = retained
    return _subset(network, mask)


def generate_network(nx, ny, z, seed):
    """Deterministically dilute a contracted phantom triangular network.

    A seeded random bond permutation determines nested uniform deletions. A
    binary search selects retention using coordination *after* dangling-bond
    pruning; exact component cleanup is then applied. The actual coordination
    is returned and can differ from the target on small finite networks.
    """
    nx, ny = _dimensions(nx, ny)
    if not np.isfinite(z) or not 2.0 < z <= 4.0:
        raise ValueError("phantom target z must satisfy 2 < z <= 4")
    if isinstance(seed, bool) or int(seed) != seed or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    network = _lattice(nx, ny, rng.integers(0, 3, nx * ny), int(seed), float(z))
    rank = np.empty(len(network.edges), dtype=np.int64)
    rank[rng.permutation(len(rank))] = np.arange(len(rank))
    low, high = 0, len(rank)
    best_error, best_mask = np.inf, None
    while low <= high:
        retained = (low + high) // 2
        mask = _core_mask(len(network.positions), network.edges, rank < retained)
        nodes = np.unique(network.edges[mask])
        actual = 2.0 * np.count_nonzero(mask) / len(nodes) if len(nodes) else 0.0
        if abs(actual - z) < best_error:
            best_error, best_mask = abs(actual - z), mask
        if actual < z:
            low = retained + 1
        else:
            high = retained - 1
    result = cleanup(network, best_mask)
    # Usually the wrapping core dominates and this adjustment is unnecessary.
    # Small samples can lose a material fraction of sites to finite decorations;
    # in that case search using the complete cleanup, not the pre-cleanup count.
    if abs(result.z_actual - z) > 0.02:
        low, high = 0, len(rank)
        error = abs(result.z_actual - z)
        while low <= high:
            retained = (low + high) // 2
            candidate = cleanup(network, rank < retained)
            actual = candidate.z_actual
            if abs(actual - z) < error:
                result, error = candidate, abs(actual - z)
            if actual < z:
                low = retained + 1
            else:
                high = retained - 1
    if not len(result.edges):
        raise ValueError("no wrapping network remains; increase lattice size or target z")
    return result
