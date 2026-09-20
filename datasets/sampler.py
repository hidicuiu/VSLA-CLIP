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
    """PK sampler with optional cross-camera positives and DFGS PID batches.

    DFGS changes which identities share a batch. Batch-hard triplet loss still
    selects the hardest positive and negative tracklets inside that batch.
    """

    def __init__(self, data_source, batch_size, num_instances=2,
                 cross_camera=False, neighbor_graph=None, samples_per_pid=0,
                 hard_negative_probability=1.0,
                 baseline_replacement_sampling=False):
        if batch_size % num_instances != 0:
            raise ValueError("batch_size must be divisible by num_instances")
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // num_instances
        self.cross_camera = cross_camera
        self.neighbor_graph = neighbor_graph
        self.hard_negative_probability = hard_negative_probability
        self.baseline_replacement_sampling = baseline_replacement_sampling
        self.samples_per_pid = samples_per_pid
        self.index_dic = defaultdict(list)
        self.camera_dic = defaultdict(lambda: defaultdict(list))

        for index, (_, pid, camid) in enumerate(data_source):
            pid, camid = int(pid), int(camid)
            self.index_dic[pid].append(index)
            self.camera_dic[pid][camid].append(index)
        self.pids = list(self.index_dic.keys())

        self.num_chunks = {}
        for pid, idxs in self.index_dic.items():
            target_samples = self.samples_per_pid if self.samples_per_pid > 0 else len(idxs)
            self.num_chunks[pid] = max(1, target_samples // self.num_instances)
        self.length = (sum(self.num_chunks.values()) // self.num_pids_per_batch) * self.batch_size

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

    def _cross_camera_chunks(self, pid):
        # For the B2/B4 setting K=2, every possible chunk contains one
        # tracklet from each camera. Fall back to random PK for single-camera IDs.
        cameras = [cam for cam, idxs in self.camera_dic[pid].items() if idxs]
        if self.num_instances != 2 or len(cameras) < 2:
            return self._random_chunks(pid)
        pools = {cam: copy.deepcopy(self.camera_dic[pid][cam]) for cam in cameras}
        for values in pools.values():
            random.shuffle(values)
        cursors = defaultdict(int)
        chunks = []
        for chunk_idx in range(self.num_chunks[pid]):
            cam_a = cameras[chunk_idx % len(cameras)]
            cam_b = cameras[(chunk_idx + 1) % len(cameras)]
            if cam_a == cam_b:
                cam_b = cameras[(chunk_idx + 1) % len(cameras)]
            a_values, b_values = pools[cam_a], pools[cam_b]
            a = a_values[cursors[cam_a] % len(a_values)]
            b = b_values[cursors[cam_b] % len(b_values)]
            cursors[cam_a] += 1
            cursors[cam_b] += 1
            chunks.append([a, b])
        return chunks

    def _make_chunks(self):
        return {
            pid: (self._cross_camera_chunks(pid) if self.cross_camera else self._random_chunks(pid))
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
            neighbors = list(self.neighbor_graph.get(pid, ()))
            random.shuffle(neighbors)
            stack.extend(reversed([n for n in neighbors if n in available and n not in visited]))
        if len(selected) < self.num_pids_per_batch:
            remaining = list(available.difference(selected))
            selected.extend(random.sample(remaining, self.num_pids_per_batch - len(selected)))
        return selected

    def __iter__(self):
        chunks = self._make_chunks()
        for pid_chunks in chunks.values():
            random.shuffle(pid_chunks)
        available = {pid for pid, pid_chunks in chunks.items() if pid_chunks}
        final_idxs = []
        while len(available) >= self.num_pids_per_batch:
            if (self.neighbor_graph is None or
                    random.random() >= self.hard_negative_probability):
                selected = self._random_pid_batch(available)
            else:
                selected = self._dfgs_pid_batch(available)
            for pid in selected:
                final_idxs.extend(chunks[pid].pop())
                if not chunks[pid]:
                    available.remove(pid)
        return iter(final_idxs)

    def __len__(self):
        return self.length
