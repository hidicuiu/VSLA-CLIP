from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import random
import numpy as np

class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list) #dict with list value
        #{783: [0, 5, 116, 876, 1554, 2041],...,}
        for index, (_, pid, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length


class RandomIdentitySampler_Video2(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Args:
    - data_source (Dataset): dataset to sample from.
    - num_instances (int): number of instances per identity.
    """
    def __init__(self, data_source, num_instances=4):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)

        for index, (_, pid, _) in enumerate(data_source):
            self.index_dic[pid].append(index)

        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)

        # compute number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        list_container = []

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    list_container.append(batch_idxs)
                    batch_idxs = []

        random.shuffle(list_container)

        ret = []
        for batch_idxs in list_container:
            ret.extend(batch_idxs)

        return iter(ret)

    def __len__(self):
        return self.length


class RandomIdentitySampler_Video(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Args:
    - data_source (Dataset): dataset to sample from.
    - num_instances (int): number of instances per identity.
    """
    def __init__(self, data_source, num_instances=4):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)

        for index, (_, pid, _) in enumerate(data_source):
            self.index_dic[pid].append(index)

        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)

        # compute number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        list_container = []

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    list_container.append(batch_idxs)
                    batch_idxs = []

        random.shuffle(list_container)

        ret = []
        for batch_idxs in list_container:
            ret.extend(batch_idxs)

        return iter(ret)

    def __len__(self):
        return self.length


class IdentityBatchSamplerVideo(Sampler):
    """PK sampler with optional RHP tracklets and DFGS PID batches.

    DFGS changes which identities share a batch. Batch-hard triplet loss still
    selects the hardest positive and negative tracklets inside that batch.
    """

    def __init__(self, data_source, batch_size, num_instances=2,
                 neighbor_graph=None, reliable_positive_graph=None,
                 samples_per_pid=0,
                 hard_negative_probability=1.0,
                 baseline_replacement_sampling=False):
        if batch_size % num_instances != 0:
            raise ValueError("batch_size must be divisible by num_instances")
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // num_instances
        self.neighbor_graph = neighbor_graph
        self.low_candidate_ids = set()
        self.low_candidate_weight = 1.0
        self.soft_sampling_stats = {}
        self.reliable_positive_graph = reliable_positive_graph
        self.hard_negative_probability = hard_negative_probability
        self.baseline_replacement_sampling = baseline_replacement_sampling
        self.samples_per_pid = samples_per_pid
        self.index_dic = defaultdict(list)

        for index, (_, pid, _) in enumerate(data_source):
            pid = int(pid)
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        self.num_chunks = {}
        for pid, idxs in self.index_dic.items():
            target_samples = self.samples_per_pid if self.samples_per_pid > 0 else len(idxs)
            self.num_chunks[pid] = max(1, target_samples // self.num_instances)
        self.length = (sum(self.num_chunks.values()) // self.num_pids_per_batch) * self.batch_size

    def update_neighbor_graph(self, neighbor_graph, low_candidate_ids=None,
                              low_candidate_weight=1.0):
        """Replace the identity graph used by subsequent DataLoader epochs."""
        if neighbor_graph is None:
            raise ValueError("neighbor_graph must not be None")
        known = set(self.pids)
        normalized = {}
        for raw_pid, raw_neighbors in neighbor_graph.items():
            pid = int(raw_pid)
            if pid not in known:
                raise ValueError("DFGS graph contains unknown identity {}".format(pid))
            neighbors = [int(candidate) for candidate in raw_neighbors]
            invalid = [candidate for candidate in neighbors
                       if candidate not in known or candidate == pid]
            if invalid:
                raise ValueError(
                    "DFGS graph for identity {} has invalid neighbors {}"
                    .format(pid, invalid[:5]))
            normalized[pid] = neighbors
        missing = known.difference(normalized)
        if missing:
            raise ValueError(
                "DFGS graph is missing {} identities, including {}"
                .format(len(missing), sorted(missing)[:5]))
        self.neighbor_graph = normalized
        if not 0.0 < low_candidate_weight <= 1.0:
            raise ValueError("low_candidate_weight must be in (0, 1]")
        low_ids = set() if low_candidate_ids is None else {
            int(pid) for pid in low_candidate_ids}
        if not low_ids.issubset(known):
            raise ValueError("low_candidate_ids contains unknown identities")
        self.low_candidate_ids = low_ids
        self.low_candidate_weight = low_candidate_weight

    def _neighbor_order(self, neighbors):
        remaining = list(neighbors)
        if not self.low_candidate_ids:
            random.shuffle(remaining)
            return remaining
        ordered = []
        while remaining:
            weights = [self.low_candidate_weight if pid in self.low_candidate_ids
                       else 1.0 for pid in remaining]
            chosen = random.choices(remaining, weights=weights, k=1)[0]
            remaining.remove(chosen)
            ordered.append(chosen)
        return ordered

    def _random_chunks(self, pid):
        idxs = copy.deepcopy(self.index_dic[pid])
        needed = self.num_chunks[pid] * self.num_instances
        if len(idxs) < needed:
            if self.baseline_replacement_sampling:
                # Match RandomIdentitySampler_Video exactly: when an identity
                # has fewer than K tracklets, draw all K entries with replacement.
                idxs = np.random.choice(idxs, size=needed, replace=True).tolist()
            else:
                # Preserve every real tracklet, then fill the missing positions.
                idxs.extend(np.random.choice(idxs, size=needed - len(idxs), replace=True).tolist())
        random.shuffle(idxs)
        return [idxs[i:i + self.num_instances] for i in range(0, needed, self.num_instances)]

    def _reliable_positive_chunks(self, pid):
        """Choose a random anchor and its best valid reliable hard positive."""
        indices = self.index_dic[pid]
        if self.num_instances < 2 or len(indices) < 2:
            return self._random_chunks(pid)
        chunks = []
        valid_indices = set(indices)
        for _ in range(self.num_chunks[pid]):
            anchor = random.choice(indices)
            ranked = self.reliable_positive_graph.get(anchor, ())
            positive = next((candidate for candidate in ranked
                             if candidate in valid_indices and candidate != anchor), None)
            if positive is None:
                positive = random.choice([index for index in indices if index != anchor])
            chunk = [anchor, positive]
            remaining = self.num_instances - 2
            if remaining > 0:
                chunk.extend(np.random.choice(
                    indices, size=remaining, replace=True).tolist())
            random.shuffle(chunk)
            chunks.append(chunk)
        return chunks

    def _make_chunks(self):
        return {
            pid: (self._reliable_positive_chunks(pid)
                  if self.reliable_positive_graph is not None
                  else self._random_chunks(pid))
            for pid in self.pids
        }

    def _random_pid_batch(self, available):
        return random.sample(list(available), self.num_pids_per_batch)

    def _dfgs_pid_batch(self, available):
        start = random.choice(tuple(available))
        stack, visited, selected = [start], set(), []
        while stack and len(selected) < self.num_pids_per_batch:
            pid = stack.pop()
            if pid in visited or pid not in available:
                continue
            visited.add(pid)
            selected.append(pid)
            neighbors = self._neighbor_order(self.neighbor_graph.get(pid, ()))
            stack.extend(reversed([n for n in neighbors if n in available and n not in visited]))
        if len(selected) < self.num_pids_per_batch:
            remaining = list(available.difference(selected))
            selected.extend(random.sample(remaining, self.num_pids_per_batch - len(selected)))
        return selected

    def __iter__(self):
        self.soft_sampling_stats = {
            "graph_batches": 0, "low_neighbor_batches": 0,
            "random_batches": 0,
        }
        chunks = self._make_chunks()
        for pid_chunks in chunks.values():
            random.shuffle(pid_chunks)
        available = {pid for pid, pid_chunks in chunks.items() if pid_chunks}
        final_idxs = []
        while len(available) >= self.num_pids_per_batch:
            if (self.neighbor_graph is None or
                    random.random() >= self.hard_negative_probability):
                selected = self._random_pid_batch(available)
                self.soft_sampling_stats["random_batches"] += 1
            else:
                selected = self._dfgs_pid_batch(available)
                self.soft_sampling_stats["graph_batches"] += 1
                if (len(selected) == 2 and selected[1] in self.low_candidate_ids
                        and selected[1] in self.neighbor_graph.get(selected[0], ())):
                    self.soft_sampling_stats["low_neighbor_batches"] += 1
            for pid in selected:
                final_idxs.extend(chunks[pid].pop())
                if not chunks[pid]:
                    available.remove(pid)
        return iter(final_idxs)

    def __len__(self):
        return self.length
