"""Save original-slice detection overlays and side-by-side ground-truth views."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import cv2
import numpy as np


COLORS = {1: (0, 220, 0), 2: (0, 0, 255)}  # BGR: handle green, tip red


def draw(image, detections, ground_truth=False):
    canvas = image.copy()
    for detection in detections:
        category = detection['category_id']
        color = COLORS[category]
        if ground_truth:
            x, y = detection['circle_center']
            radius = detection['circle_radius']
            dx, dy = detection['direction']
            angle = math.atan2(-dy, dx)
            valid = detection.get('angle_valid', True)
        else:
            x, y = detection['keypoint_center']
            radius = detection['keypoint_radius']
            angle = detection['keypoint_angle']
            valid = True
        if not all(math.isfinite(float(v)) for v in (x, y, radius, angle)):
            continue
        if radius < 0 or not (0 <= x < canvas.shape[1] and 0 <= y < canvas.shape[0]):
            continue
        center = (int(round(x)), int(round(y)))
        cv2.circle(canvas, center, max(1, int(round(radius))), color, 1, cv2.LINE_AA)
        if valid:
            end = (int(round(x+25*math.cos(angle))), int(round(y-25*math.sin(angle))))
            cv2.arrowedLine(canvas, center, end, color, 1, cv2.LINE_AA, tipLength=.25)
        label = str(detection['needle_id']) if ground_truth else '{:.2f}'.format(detection['score'])
        if ground_truth and not valid:
            label += '?'
        cv2.putText(canvas, label, (center[0]+4, center[1]-4),
                    cv2.FONT_HERSHEY_SIMPLEX, .32, color, 1, cv2.LINE_AA)
    return canvas


def panel(image, title):
    header = np.zeros((44, image.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, title, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, .45, (255,255,255), 1)
    cv2.putText(header, 'tip: red | handle: green', (8, 35),
                cv2.FONT_HERSHEY_SIMPLEX, .4, (255,255,255), 1)
    return np.concatenate([header, image])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--score_thresh', type=float, default=.2)
    args = parser.parse_args()
    data = json.loads(args.annotations.read_text())
    predictions, ground_truth = defaultdict(list), defaultdict(list)
    for detection in json.loads(args.results.read_text()):
        if detection['score'] >= args.score_thresh:
            predictions[detection['image_id']].append(detection)
    for ann in data['annotations']:
        ground_truth[ann['image_id']].append(ann)
    thumbnails = {}
    for im in data['images']:
        image = cv2.imread(str(args.images/im['file_name']))
        if image is None:
            raise FileNotFoundError(args.images/im['file_name'])
        pred = draw(image, predictions[im['id']])
        gt = draw(image, ground_truth[im['id']], ground_truth=True)
        title = '{} z={:03d}'.format(im['case_id'], im['slice_index'])
        comparison = np.concatenate([panel(gt, title+' | ground truth (IDs)'),
                                     panel(pred, title+' | prediction (scores)')], axis=1)
        for kind, pixels in [('predictions', pred), ('comparisons', comparison)]:
            path = args.output/kind/im['case_id']/Path(im['file_name']).with_suffix('.jpg')
            path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(path), pixels, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise OSError('Could not write {}'.format(path))
        if ground_truth[im['id']]:
            count = len(ground_truth[im['id']])
            current = thumbnails.get(im['case_id'])
            if current is None or count > current[0]:
                thumbnails[im['case_id']] = (count, im['case_id'], comparison)
    # Select one annotated slice per case, then show up to twelve cases.
    chosen, seen = [], set()
    for count, case, pixels in sorted(thumbnails.values(), key=lambda item: -item[0]):
        if case in seen:
            continue
        seen.add(case)
        chosen.append(cv2.resize(pixels, (768, 417)))
        if len(chosen) == 12:
            break
    if chosen:
        while len(chosen) % 2:
            chosen.append(np.zeros_like(chosen[0]))
        sheet = np.concatenate([np.concatenate(chosen[i:i+2], axis=1)
                                for i in range(0, len(chosen), 2)])
        cv2.imwrite(str(args.output/'overview.jpg'), sheet)
    summary = {'images': len(data['images']), 'score_threshold': args.score_thresh,
               'predictions_above_threshold': sum(map(len, predictions.values())),
               'ground_truth_annotations': len(data['annotations']),
               'cases': len({im['case_id'] for im in data['images']})}
    (args.output/'visualization_info.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
