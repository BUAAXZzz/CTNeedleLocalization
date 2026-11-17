"""Convert post-processed keypoint detections into case-level matching inputs."""
import json
import math
from pathlib import Path


def export_pred2d(results, images, output_dir, manifest=None, score_thresh=0.2,
                  ct_root=None):
    """Coordinates must already be in the original CT slice pixel grid.

    ``case_id`` and zero-based ``slice_index`` come from the annotation image
    records, not from image IDs or filename parsing. Angles are radians, with
    x right and y up (atan2(-dy, dx)). CTs are referenced without being copied.
    """
    if not math.isfinite(score_thresh) or not 0 <= score_thresh <= 1:
        raise ValueError('score_thresh must be in [0, 1]')
    manifest = manifest or {}
    reports = {r['case_id']: r for r in manifest.get('cases', [])}
    by_image, cases = {}, {}
    for im in images:
        image_id = int(im['id'])
        if image_id in by_image:
            raise ValueError('Duplicate image_id: {}'.format(image_id))
        case_id = str(im['case_id'])
        if Path(case_id).name != case_id or case_id in ('.', '..'):
            raise ValueError('Unsafe case_id: {}'.format(case_id))
        z = int(im['slice_index'])
        if z < 0 or z != im['slice_index']:
            raise ValueError('slice_index must be a zero-based integer')
        case = cases.setdefault(case_id, {
            'case_id': case_id, 'coordinate_system': 'CT continuous voxel indices',
            'angle_convention': 'radians; atan2(-dy, dx); x right, y up',
            'score_threshold': score_thresh, 'slices': {}})
        if z in case['slices']:
            raise ValueError('Duplicate case/slice: {}/{}'.format(case_id, z))
        case['slices'][z] = {'z': z, 'image_id': image_id, 'dets': []}
        report = reports.get(case_id, {})
        if report:
            case['source_case'] = report['source_case']
            for key in ['spacing_xyz', 'origin_xyz', 'direction', 'size_xyz']:
                if key in report:
                    case[key] = report[key]
        source_root = ct_root or manifest.get('source')
        if source_root and report:
            case['ct_path'] = str((Path(source_root) / report['source_case'] / 'ct.nii.gz').resolve())
        elif im.get('ct_path'):
            case['ct_path'] = str(Path(im['ct_path']).resolve())
        by_image[image_id] = (im, case['slices'][z])
    for detection in results:
        image_id = int(detection['image_id'])
        if image_id not in by_image:
            raise ValueError('Prediction image_id {} absent from annotation JSON'.format(image_id))
        category = detection['category_id']
        if category not in (1, 2):
            raise ValueError('Expected category_id 1=handle or 2=tip')
        x, y = map(float, detection['keypoint_center'])
        angle = float(detection['keypoint_angle'])
        radius = float(detection['keypoint_radius'])
        score = float(detection['score'])
        if not all(math.isfinite(v) for v in (x, y, angle, radius, score)):
            raise ValueError('Non-finite detection: {}'.format(detection))
        im, target = by_image[image_id]
        # Cropping/decoding can produce out-of-frame peaks. Do not clamp them
        # onto CT edges, where they would produce spurious HU path scores.
        if score < score_thresh or radius < 0 or not (0 <= x < im['width'] and 0 <= y < im['height']):
            continue
        target['dets'].append({'x0': x, 'y0': y, 'z': target['z'],
                               'angle': (angle + math.pi) % (2 * math.pi) - math.pi,
                               'cls': category - 1, 'score': score, 'radius': radius,
                               'image_id': image_id})
    output_dir = Path(output_dir)
    paths = []
    # Include cases/slices with no detections so empty cases remain reviewable.
    for case_id, case in sorted(cases.items()):
        case['slices'] = [case['slices'][z] for z in sorted(case['slices'])]
        destination = output_dir / case_id / 'pred_2d.json'
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(case, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        paths.append(destination)
    return paths
