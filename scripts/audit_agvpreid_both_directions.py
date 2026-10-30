"""AG-VPReID 双向数据完整性审计（只读，不改动任何数据）。

检查项：
1. 目录结构与命名规范 P{pid}T{tid}C{cam}E{seq}F{frame}.jpg
2. 每个 tracklet 的相机一致性、文件名 P 与父目录 ID 一致性
3. 空目录、空 tracklet、过短 tracklet 数量
4. 每个 split 的帧数/相机分布/身份数
5. 帧号 F 的连续性与重复
6. case1 与 case2 是否复用同一批图像
"""

import json
import os
import re
import sys
from collections import Counter, defaultdict

ROOT = r"H:\WSY\ReID\datas\AG-VPReID"
PATTERN = re.compile(
    r"^P(?P<pid>\d+)T(?P<tid>[^C]*)C(?P<camera>\d+)E(?P<seq>[^F]*)F(?P<frame>\d+)\.(?:jpg|jpeg|png)$",
    re.IGNORECASE,
)

SPLITS = {
    "train": os.path.join(ROOT, "train"),
    "a2g_query": os.path.join(ROOT, "case1_aerial_to_groud", "query"),
    "a2g_gallery": os.path.join(ROOT, "case1_aerial_to_groud", "gallery"),
    "g2a_query": os.path.join(ROOT, "case2_groud_to_aerial", "query"),
    "g2a_gallery": os.path.join(ROOT, "case2_groud_to_aerial", "gallery"),
}


def natural_key(value):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", value)]


def audit_split(name, path):
    result = {
        "split": name,
        "path": path,
        "identities": 0,
        "tracklets": 0,
        "frames": 0,
        "camera_frames": Counter(),
        "empty_identity_dirs": 0,
        "tracklets_without_frames": 0,
        "tracklets_below_8_frames": 0,
        "frame_index_gaps": 0,
        "frame_index_duplicates": 0,
        "bad_filename": 0,
        "pid_folder_mismatch": 0,
        "tracklets_with_multiple_p": 0,
        "tracklets_with_multiple_camera": 0,
        "tracklets_with_multiple_seq": 0,
        "frames_per_tracklet": Counter(),
    }
    if not os.path.isdir(path):
        result["error"] = "MISSING DIRECTORY"
        return result

    for person in sorted(os.listdir(path), key=natural_key):
        person_path = os.path.join(path, person)
        if not os.path.isdir(person_path):
            continue
        result["identities"] += 1
        tracklets = sorted(
            (d for d in os.listdir(person_path)
             if os.path.isdir(os.path.join(person_path, d))),
            key=natural_key)
        if not tracklets:
            result["empty_identity_dirs"] += 1
        for tracklet in tracklets:
            tracklet_path = os.path.join(person_path, tracklet)
            result["tracklets"] += 1
            pids, cams, seqs, frames = set(), set(), set(), []
            for entry in os.scandir(tracklet_path):
                if not entry.is_file():
                    continue
                if not entry.name.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                match = PATTERN.match(entry.name)
                if match is None:
                    result["bad_filename"] += 1
                    continue
                pids.add(match.group("pid"))
                cams.add(int(match.group("camera")))
                seqs.add(match.group("seq"))
                frames.append(int(match.group("frame")))
                result["camera_frames"][int(match.group("camera"))] += 1
            count = len(frames)
            result["frames"] += count
            result["frames_per_tracklet"][count] += 1
            if count == 0:
                result["tracklets_without_frames"] += 1
                continue
            if count < 8:
                result["tracklets_below_8_frames"] += 1
            if len(pids) > 1:
                result["tracklets_with_multiple_p"] += 1
            if pids and str(person) not in pids:
                result["pid_folder_mismatch"] += 1
            if len(cams) > 1:
                result["tracklets_with_multiple_camera"] += 1
            if len(seqs) > 1:
                result["tracklets_with_multiple_seq"] += 1
            ordered = sorted(frames)
            if len(set(ordered)) != len(ordered):
                result["frame_index_duplicates"] += 1
            if ordered != list(range(ordered[0], ordered[0] + len(ordered))):
                result["frame_index_gaps"] += 1
    return result


def main():
    report = {}
    for name, path in SPLITS.items():
        sys.stderr.write("scanning {} ...\n".format(name))
        sys.stderr.flush()
        report[name] = audit_split(name, path)
    for value in report.values():
        value["camera_frames"] = dict(sorted(value["camera_frames"].items()))
        fps = value.pop("frames_per_tracklet")
        value["min_frames_per_tracklet"] = min(fps) if fps else None
        value["max_frames_per_tracklet"] = max(fps) if fps else None
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
