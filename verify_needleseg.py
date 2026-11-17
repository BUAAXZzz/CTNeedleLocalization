"""Validate generated NeedleSeg files and render a small annotation preview."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


def main():
    root = Path(__file__).resolve().parent / '2d_detection/data/needleseg'
    manifest = json.loads((root / 'manifest.json').read_text())
    reports = {r['case_id']: r for r in manifest['cases']}
    groups = {s: {r['patient_group'] for r in reports.values() if r['split'] == s}
              for s in ['train', 'val']}
    assert not groups['train'] & groups['val'], 'Patient overlap'
    assert all(r['split'] == 'val' for r in reports.values() if not r['complete'])
    assert all(r['complete'] for r in reports.values() if r['split'] == 'train')
    image_ids, annotation_ids, panels = set(), set(), []
    for split in ['train', 'val']:
        dataset = json.loads((root / 'NeedleSeg_{}2025.json'.format(split)).read_text())
        by_image = defaultdict(list)
        for ann in dataset['annotations']:
            assert ann['id'] not in annotation_ids
            annotation_ids.add(ann['id'])
            by_image[ann['image_id']].append(ann)
        images = {im['id']: im for im in dataset['images']}
        assert set(by_image) <= set(images)
        assert {p.name for p in (root / split).glob('*.png')} == {im['file_name'] for im in images.values()}
        for im in images.values():
            assert im['id'] not in image_ids
            image_ids.add(im['id'])
            report = reports[im['case_id']]
            assert report['split'] == split
            path = root / split / im['file_name']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == im['sha256']
            pixels = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            assert pixels.dtype == np.uint8 and pixels.shape == (im['height'], im['width'])
            needles = {n['needle_id']: n for n in report['needles']}
            for ann in by_image[im['id']]:
                x, y = ann['circle_center']
                assert 0 <= x < im['width'] and 0 <= y < im['height']
                assert ann['circle_radius'] > 0
                needle = needles[ann['needle_id']]
                if ann['angle_valid']:
                    assert not needle['missing']
                    category = ann['category_id']
                    expected = (np.array(needle['centers_xyz'][str(3-category)])
                                - np.array(needle['centers_xyz'][str(category)]))
                    assert np.allclose(ann['direction_xyz'], expected)
                else:
                    assert split == 'val' or not needle['missing']
                if needle['missing']:
                    assert not ann['angle_valid']
        # Show one complete and one incomplete case where available.
        for complete in [True, False]:
            candidates = [im for im in images.values()
                          if reports[im['case_id']]['complete'] == complete and by_image[im['id']]]
            if not candidates:
                continue
            im = max(candidates, key=lambda im: len(by_image[im['id']]))
            canvas = cv2.imread(str(root / split / im['file_name']))
            for ann in by_image[im['id']]:
                center = tuple(np.rint(ann['circle_center']).astype(int))
                color = (0, 0, 255) if ann['category_id'] == 2 else (0, 255, 0)
                cv2.circle(canvas, center, int(np.ceil(ann['circle_radius'])), color, 1)
                d = np.array(ann['direction'], dtype=float)
                if ann['angle_valid']:
                    end = tuple(np.rint(np.array(center) + 20*d/np.linalg.norm(d)).astype(int))
                    cv2.arrowedLine(canvas, center, end, color, 1, tipLength=0.25)
                else:
                    cv2.putText(canvas, '?', center, cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1)
            canvas = cv2.resize(canvas, (512, 512))
            title = '{} {} z={} complete={}'.format(split, im['case_id'], im['slice_index'], complete)
            header = np.zeros((36, 512, 3), dtype=np.uint8)
            cv2.putText(header, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)
            panels.append(np.concatenate([header, canvas]))
        print(split, 'cases=', sum(r['split'] == split for r in reports.values()),
              'images=', len(images), 'annotations=', len(dataset['annotations']),
              'classes=', dict(Counter(a['category_id'] for a in dataset['annotations'])))
    assert cv2.imwrite(str(root / 'annotation_preview.jpg'), np.concatenate(panels, axis=1))
    print('PASS: all PNG checksums/shapes, JSON IDs, patient separation, completeness and directions')
    print('Incomplete cases:', sum(not r['complete'] for r in reports.values()))
    print('Mixed-body components:', sum(len(r['warnings']) for r in reports.values()))
    print('Max annotations per slice:', manifest['max_objects_per_slice'])


if __name__ == '__main__':
    main()
