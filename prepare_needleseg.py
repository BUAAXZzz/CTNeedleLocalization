"""Build patient-separated 2D NeedleSeg data from CT, instance endpoints and body masks.

Run with a Python environment containing SimpleITK, NumPy and SciPy.
Coordinates stay in the original CT voxel grid; z is a zero-based slice index.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def read_aligned(path, reference):
    image = sitk.ReadImage(str(path))
    aligned = (image.GetSize() == reference.GetSize()
               and np.allclose(image.GetSpacing(), reference.GetSpacing())
               and np.allclose(image.GetOrigin(), reference.GetOrigin())
               and np.allclose(image.GetDirection(), reference.GetDirection()))
    if not aligned:
        raise ValueError('CT/mask geometry mismatch: {}'.format(path))
    return sitk.GetArrayFromImage(image)


def classify_endpoints(labels, body):
    """Classify each 26-connected endpoint by majority body occupancy.

    A component that crosses the body boundary remains one endpoint. Components
    of the same needle and class are pooled for the 3D centroid and slice targets.
    """
    endpoints, records, warnings = {}, [], []
    for value in np.unique(labels):
        if value == 0:
            continue
        points = np.argwhere(labels == value)
        low, high = points.min(0), points.max(0) + 1
        roi = tuple(slice(int(a), int(b)) for a, b in zip(low, high))
        components, count = ndimage.label(labels[roi] == value,
                                          structure=np.ones((3, 3, 3)))
        classes = {1: [], 2: []}  # category IDs: handle=1, tip=2
        component_records = []
        for component in range(1, count + 1):
            local = np.argwhere(components == component)
            xyz = local + low
            fraction = float((body[tuple(xyz.T)] > 0).mean())
            category = 2 if fraction > 0.5 else 1
            classes[category].append(xyz)
            component_records.append({'voxel_count': len(xyz),
                                      'inside_body_fraction': fraction,
                                      'category_id': category})
            if 0 < fraction < 1:
                warnings.append({'needle_id': int(value), 'inside_body_fraction': fraction,
                                 'reason': 'endpoint crosses body boundary; majority vote used'})
        endpoints[int(value)] = {
            k: np.concatenate(v) for k, v in classes.items() if v
        }
        missing = [name for category, name in [(1, 'handle'), (2, 'tip')]
                   if category not in endpoints[int(value)]]
        records.append({'needle_id': int(value), 'missing': missing,
                        'components': component_records,
                        'centers_xyz': {str(k): p.mean(0)[::-1].tolist()
                                        for k, p in endpoints[int(value)].items()}})
    return endpoints, records, warnings


def patient_key(name):
    # These source directory names end in a patient identifier. Conservatively
    # group repeat studies by that identifier, including suffixes such as '-3D'.
    return name.rsplit('@', 1)[-1].split('-', 1)[0].strip().lower()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parent / 'data_nii')
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parent / '2d_detection/data/needleseg')
    parser.add_argument('--window-level', type=float, default=-100)
    parser.add_argument('--window-width', type=float, default=1700)
    parser.add_argument('--seed', type=int, default=317)
    parser.add_argument('--complete-val-fraction', type=float, default=0.2,
                        help='Fraction of remaining complete patient groups held out for val')
    args = parser.parse_args()
    if args.window_width <= 0 or not 0 <= args.complete_val_fraction < 1:
        parser.error('Window width must be positive; val fraction must be in [0,1)')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output must be empty to avoid mixing dataset versions')
    cases = sorted(p for p in args.source.iterdir() if p.is_dir())
    if not cases:
        parser.error('No case directories found')
    # Verify all inputs before producing output.
    for case in cases:
        for name in ['ct.nii.gz', 'label.nii.gz', 'body.nii.gz']:
            if not (case / name).is_file():
                raise FileNotFoundError(case / name)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'train').mkdir()
    (args.output / 'val').mkdir()
    # Cache masks/CT only one case at a time. First pass records completeness.
    reports = []
    for index, case in enumerate(cases, 1):
        ct = sitk.ReadImage(str(case / 'ct.nii.gz'))
        labels = read_aligned(case / 'label.nii.gz', ct)
        body = read_aligned(case / 'body.nii.gz', ct)
        if not np.isfinite(labels).all() or not np.equal(labels, np.round(labels)).all():
            raise ValueError('Instance labels must be finite integers: {}'.format(case))
        if not (body > 0).any() or not (labels > 0).any():
            raise ValueError('Empty body or instance labels: {}'.format(case))
        _, needles, warnings = classify_endpoints(labels, body)
        report = {'case_id': 'case_{:03d}'.format(index), 'source_case': case.name,
                  'patient_group': patient_key(case.name),
                  'complete': all(not n['missing'] for n in needles),
                  'needles': needles, 'warnings': warnings,
                  'size_xyz': list(ct.GetSize()), 'spacing_xyz': list(ct.GetSpacing()),
                  'origin_xyz': list(ct.GetOrigin()), 'direction': list(ct.GetDirection())}
        reports.append(report)
        print('[scan {}/{}] {} complete={}'.format(index, len(cases), report['case_id'],
                                                   report['complete']), flush=True)
    forced_val = {r['patient_group'] for r in reports if not r['complete']}
    eligible = sorted({r['patient_group'] for r in reports} - forced_val)
    rng = np.random.RandomState(args.seed)
    rng.shuffle(eligible)
    held_out = set(eligible[:int(np.ceil(len(eligible) * args.complete_val_fraction))])
    val_groups = forced_val | held_out
    datasets = {split: {'info': {'description': 'NeedleSeg from original windowed CT'},
                        'images': [], 'annotations': [],
                        'categories': [{'id': 1, 'name': 'handle'}, {'id': 2, 'name': 'tip'}]}
                for split in ['train', 'val']}
    image_id, annotation_id = 0, 0
    max_objects = 0
    lower = args.window_level - args.window_width / 2
    upper = args.window_level + args.window_width / 2
    for case, report in zip(cases, reports):
        split = 'val' if report['patient_group'] in val_groups else 'train'
        report['split'] = split
        report['split_reason'] = ('incomplete_patient_group' if report['patient_group'] in forced_val
                                  else 'complete_holdout' if split == 'val' else 'complete_training')
        ct = sitk.ReadImage(str(case / 'ct.nii.gz'))
        volume = sitk.GetArrayFromImage(ct).astype(np.float32)
        labels = read_aligned(case / 'label.nii.gz', ct)
        body = read_aligned(case / 'body.nii.gz', ct)
        endpoints, _, _ = classify_endpoints(labels, body)
        targets = {z: [] for z in range(volume.shape[0])}
        for needle_id, classes in endpoints.items():
            complete = len(classes) == 2
            for category, points in classes.items():
                direction = np.zeros(3)
                if complete:
                    # Both endpoint arrows point toward the opposite endpoint.
                    direction = classes[3 - category].mean(0)[::-1] - points.mean(0)[::-1]
                angle_valid = bool(complete and np.linalg.norm(direction[:2]) > 1e-8)
                for z in np.unique(points[:, 0]):
                    xy = points[points[:, 0] == z, 1:][:, ::-1]
                    center = xy.mean(0)
                    radius = max(1., float(np.linalg.norm(xy - center, axis=1).max()) + 0.5)
                    low, high = xy.min(0), xy.max(0) + 1
                    targets[int(z)].append({
                        'category_id': category, 'needle_id': needle_id,
                        'circle_center': center.tolist(), 'circle_radius': radius,
                        'direction': direction[:2].tolist(), 'direction_xyz': direction.tolist(),
                        'angle_valid': angle_valid,
                        'bbox': [int(low[0]), int(low[1]), int(high[0]-low[0]), int(high[1]-low[1])],
                        'area': int(len(xy)), 'iscrowd': 0})
        for z, image in enumerate(volume):
            image_id += 1
            filename = '{}_z{:03d}.png'.format(report['case_id'], z)
            pixels = np.rint(np.clip((image - lower) / (upper - lower), 0, 1) * 255).astype(np.uint8)
            sitk.WriteImage(sitk.GetImageFromArray(pixels), str(args.output / split / filename))
            datasets[split]['images'].append({
                'id': image_id, 'file_name': filename, 'height': image.shape[0], 'width': image.shape[1],
                'case_id': report['case_id'], 'slice_index': z,
                'spacing_xyz': list(ct.GetSpacing()),
                'sha256': hashlib.sha256((args.output / split / filename).read_bytes()).hexdigest()})
            max_objects = max(max_objects, len(targets[z]))
            for annotation in targets[z]:
                annotation_id += 1
                annotation.update({'id': annotation_id, 'image_id': image_id})
                datasets[split]['annotations'].append(annotation)
        print('[write] {} -> {}'.format(report['case_id'], split), flush=True)
    summary = {'source': str(args.source.resolve()), 'seed': args.seed,
               'window_level': args.window_level, 'window_width': args.window_width,
               'window_range_hu': [lower, upper], 'uses_tophat': False,
               'complete_val_fraction': args.complete_val_fraction,
               'max_objects_per_slice': max_objects, 'cases': reports,
               'counts': {split: {'cases': sum(r['split'] == split for r in reports),
                                  'images': len(d['images']), 'annotations': len(d['annotations']),
                                  'angles_invalid': sum(not a['angle_valid'] for a in d['annotations'])}
                          for split, d in datasets.items()}}
    for split, dataset in datasets.items():
        write_json(args.output / 'NeedleSeg_{}2025.json'.format(split), dataset)
    write_json(args.output / 'manifest.json', summary)
    print(json.dumps(summary['counts'], indent=2), flush=True)
    print('max_objects_per_slice={}'.format(max_objects), flush=True)


if __name__ == '__main__':
    main()
