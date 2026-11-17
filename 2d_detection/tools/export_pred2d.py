"""Bridge results_keypoints.json to the per-case input of match3d_batch.py."""
import argparse
import json
from pathlib import Path

import _init_paths
from utils.pred2d import export_pred2d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True, type=Path)
    parser.add_argument('--annotations', required=True, type=Path,
                        help='The exact NeedleSeg split JSON used for inference')
    parser.add_argument('--manifest', type=Path,
                        help='Defaults to manifest.json beside the annotation file')
    parser.add_argument('--ct_root', type=Path, help='Override original NIfTI case root')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--score_thresh', type=float, default=0.2)
    args = parser.parse_args()
    manifest_path = args.manifest or args.annotations.parent / 'manifest.json'
    if not manifest_path.is_file():
        parser.error('Manifest not found: {}'.format(manifest_path))
    paths = export_pred2d(json.loads(args.results.read_text()),
                          json.loads(args.annotations.read_text())['images'], args.output,
                          json.loads(manifest_path.read_text()), args.score_thresh, args.ct_root)
    print('Exported {} cases to {}'.format(len(paths), args.output))


if __name__ == '__main__':
    main()
