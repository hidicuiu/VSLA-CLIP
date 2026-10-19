import json
import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from scipy.stats import spearmanr

from datasets import data_manager


root = Path('../output/innovation')
graph_dir = root / 'g2a_D3_reliable_dynamic_DFGS_V_q025_p2k4_seed1234_lr2e4' / 'dynamic_dfgs_graphs'
d1_dir = root / 'g2a_D1_dynamic_DFGS_V_q025_p2k4_seed1234_lr2e4' / 'dynamic_dfgs_graphs'
dataset = data_manager.init_dataset(name='g2a', root='../datas/G2A-VReID')
by_id = defaultdict(list)
for index, (paths, pid, camid) in enumerate(dataset.train):
    by_id[int(pid)].append((index, int(camid), len(paths), paths[0]))


def load(epoch, directory=graph_dir):
    with (directory / 'epoch_{:03d}.json'.format(epoch)).open(encoding='utf-8') as f:
        return json.load(f)


def excluded(payload):
    q = payload['identity_reliability_min']
    threshold = payload['filter_diagnostics']['quality_threshold']
    return {pid for pid, value in enumerate(q) if value < threshold}


def describe(values):
    values = sorted(values)
    return {'min': values[0], 'p25': values[len(values)//4],
            'median': median(values), 'p75': values[3*len(values)//4],
            'mean': round(mean(values), 2), 'max': values[-1]}


for epoch in (1, 11, 51, 81, 111):
    payload = load(epoch)
    low = excluded(payload)
    print('EPOCH', epoch, 'excluded', len(low), 'threshold',
          round(payload['filter_diagnostics']['quality_threshold'], 5))
    for label, ids in (('low', low), ('other', set(by_id)-low)):
        tracklets = [item for pid in ids for item in by_id[pid]]
        cams = Counter(cam for _, cam, _, _ in tracklets)
        lens = [n for _, _, n, _ in tracklets]
        print(label, 'IDs', len(ids), 'cams', dict(cams),
              'lengths', describe(lens),
              'frames_lt2', sum(n < 2 for n in lens),
              'frames_lt8', sum(n < 8 for n in lens))
    q = payload['tracklet_temporal_reliability']
    by_camera = defaultdict(list)
    low_camera = defaultdict(list)
    for index, (_, pid, camid) in enumerate(dataset.train):
        by_camera[int(camid)].append(q[index])
        if int(pid) in low:
            low_camera[int(camid)].append(q[index])
    print('quality_by_camera', {cam: round(mean(values), 5)
                                for cam, values in by_camera.items()},
          'low_quality_by_camera', {cam: round(mean(values), 5)
                                   for cam, values in low_camera.items()})

all_low = [excluded(load(epoch)) for epoch in range(1, 112, 10)]
after_warmup = all_low[1:]
print('LOW_SET_OVERLAP_001_TO_111', len(all_low[0] & all_low[-1]))
print('LOW_SET_OVERLAP_011_TO_111', len(all_low[1] & all_low[-1]))
print('ALWAYS_LOW_AFTER_EPOCH_011', len(set.intersection(*after_warmup)))
print('EVER_LOW_AFTER_EPOCH_011', len(set.union(*after_warmup)))

last = load(111)
low = excluded(last)
old = load(111, d1_dir)['graph']
new = last['graph']
old_low_edges = sum(int(n) in low for neighbors in old.values() for n in neighbors)
new_low_edges = sum(int(n) in low for neighbors in new.values() for n in neighbors)
print('LOW_ID_NEIGHBOR_EDGES_D1_D3', old_low_edges, new_low_edges,
      'out_of', sum(len(v) for v in old.values()))

q = last['identity_reliability_min']
tq = last['tracklet_temporal_reliability']
print('LOWEST_12_IDS_EPOCH_111')
for pid in sorted(low, key=lambda key: q[key])[:12]:
    items = [(index, 'C{}'.format(camid + 1), length,
              round(tq[index], 5), Path(path).name)
             for index, camid, length, path in by_id[pid]]
    print(pid, 'min_q', round(q[pid], 5), 'tracks', items)

threshold = last['filter_diagnostics']['quality_threshold']
driver = Counter()
both_below = 0
one_below = 0
for pid in low:
    tracklets = by_id[pid]
    ranked_tracks = sorted(tracklets, key=lambda item: tq[item[0]])
    driver['C{}'.format(ranked_tracks[0][1] + 1)] += 1
    n_below = sum(tq[index] < threshold for index, _, _, _ in tracklets)
    both_below += (n_below == 2)
    one_below += (n_below == 1)
print('LOW_SCORE_DRIVER', dict(driver), 'ONE_BELOW', one_below,
      'BOTH_BELOW', both_below)
for camid in (0, 1):
    cam_indices = [index for index, (_, _, cam) in enumerate(dataset.train)
                   if int(cam) == camid]
    corr, pvalue = spearmanr(
        [len(dataset.train[index][0]) for index in cam_indices],
        [tq[index] for index in cam_indices])
    print('CAMERA_C{}_SPEARMAN_LENGTH_QUALITY'.format(camid + 1),
          round(float(corr), 4), 'p', float(pvalue))
for label, ids in (('low', low), ('other', set(by_id)-low)):
    for camid in (0, 1):
        lengths = [length for pid in ids for _, cam, length, _ in by_id[pid]
                   if cam == camid]
        print(label, 'C{}'.format(camid + 1), 'length', describe(lengths))

audit_path = Path('scripts/excluded_identity_audit_epoch111.csv')
with audit_path.open('w', newline='', encoding='utf-8-sig') as audit_file:
    writer = csv.writer(audit_file)
    writer.writerow([
        'train_pid', 'source_pid', 'c1_frames', 'c2_frames',
        'c1_reliability', 'c2_reliability', 'min_reliability',
        'epoch111_threshold', 'excluded_refresh_count',
        'c1_first_image', 'c2_first_image'])
    for pid in sorted(low, key=lambda key: (q[key], key)):
        tracks = {cam: (index, length, path)
                  for index, cam, length, path in by_id[pid]}
        c1_index, c1_length, c1_path = tracks[0]
        c2_index, c2_length, c2_path = tracks[1]
        source_pid = Path(c1_path).name[:4]
        writer.writerow([
            pid, source_pid, c1_length, c2_length,
            tq[c1_index], tq[c2_index], q[pid], threshold,
            sum(pid in group for group in all_low),
            str(Path(c1_path).resolve()), str(Path(c2_path).resolve())])
print('AUDIT_CSV', audit_path.resolve(), 'ROWS', len(low))
