import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '3d_matching'))
sys.path.insert(0, str(ROOT / '2d_detection/lib'))
from utils.pred2d import export_pred2d
from match3d_batch import GMMConfig, run_gmm_one_case
from match3d_utils import (cal_angle_diff_deg, greedy_match,
                           greedy_match_wo_nocross, merge_duplicate_needles,
                           PairingConfig, compute_pair_score, NEG_INF,
                           merge_detections_across_slices)


class MatchingPipelineTests(unittest.TestCase):
    def test_periodic_opposite_endpoint_angles(self):
        self.assertAlmostEqual(cal_angle_diff_deg(-math.pi, math.pi/2), 90)
        self.assertAlmostEqual(cal_angle_diff_deg(0, math.pi), 0)
        self.assertAlmostEqual(cal_angle_diff_deg(.2, .2+8*math.pi, opposite=False), 0)
        for a in [-9*math.pi, -.2, 0, 7*math.pi]:
            for b in [-6*math.pi, .5, 13*math.pi]:
                d = cal_angle_diff_deg(a, b)
                self.assertTrue(0 <= d <= 180)
                self.assertAlmostEqual(d, cal_angle_diff_deg(a+12*math.pi, b-4*math.pi))

    def test_empty_and_invalid_score_matrices(self):
        ct = sitk.Image([3, 3, 3], sitk.sitkFloat32)
        for shape in [(0, 0), (0, 1), (1, 0)]:
            self.assertEqual(greedy_match(np.empty(shape), [], [], ct, 1), [])
            self.assertEqual(greedy_match_wo_nocross(np.empty(shape), [], [], 1), [])
        self.assertEqual(greedy_match(np.array([[np.nan]]), [[0]*4], [[0]*4], ct, 1), [])

    def test_merge_preserves_representative_and_members(self):
        pairs = [dict(Tip=np.array([10, y, 1, 0.]), Pin=np.array([210, y, 1, math.pi]),
                      score=score, tip_idx=i, pin_idx=i+3)
                 for i, (y, score) in enumerate([(5, 10), (6, 9)])]
        merged = merge_duplicate_needles(pairs, (1, 1, 1))
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]['tip_idx'], 0)
        self.assertEqual(merged[0]['pin_idx'], 3)
        self.assertEqual(merged[0]['tip_indices'], [0, 1])
        self.assertEqual(merged[0]['pin_indices'], [3, 4])
        np.testing.assert_allclose(merged[0]['Tip'][:3], [10, 5.5, 1])

    def test_export_to_nifti_matching_and_empty_cases(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'source' / 'original_case'
            source.mkdir(parents=True)
            ct = sitk.GetImageFromArray(np.full((3, 20, 230), 100, dtype=np.float32))
            ct.SetOrigin((30, -40, 50))
            sitk.WriteImage(ct, str(source / 'ct.nii.gz'))
            manifest = {'source': str(root / 'source'),
                        'cases': [{'case_id': name, 'source_case': 'original_case'}
                                  for name in ['case_a', 'case_empty']]}
            images = [dict(id=i, case_id=case, slice_index=1, width=230, height=20)
                      for i, case in [(8, 'case_a'), (17, 'case_empty')]]
            def detection(category, x, score=.9):
                return dict(image_id=8, category_id=category, score=score,
                            keypoint_center=[x, 5.125], keypoint_radius=2.5,
                            keypoint_angle=0.123456 if category == 2 else math.pi+0.123456)
            results = [detection(2, 10.25), detection(1, 210.25), detection(1, 215, .01)]
            export_pred2d(results, images, root/'pred', manifest, .2)
            pred = json.loads((root/'pred/case_a/pred_2d.json').read_text())
            dets = pred['slices'][0]['dets']
            self.assertEqual([d['cls'] for d in dets], [1, 0])
            self.assertEqual(dets[0]['x0'], 10.25)
            self.assertEqual(dets[0]['z'], 1)
            self.assertAlmostEqual(dets[0]['angle'], .123456)
            for disable in [True, False]:
                cfg = GMMConfig(n_prior=1, disable_nocross=disable)
                out = run_gmm_one_case(str(root/'pred/case_a'), cfg)
                self.assertEqual(out['counts']['pairs_final'], 1)
                self.assertEqual(out['pairs'][0]['tip_idx'], 0)
                self.assertEqual(out['pairs'][0]['handle_indices'], [0])
                self.assertEqual(out['origin_xyz'], [30, -40, 50])
                empty = run_gmm_one_case(str(root/'pred/case_empty'), cfg)
                self.assertEqual(empty['pairs'], [])
            # Test local CT auto-detection independent of exported absolute paths.
            for name in ['ct.nii.gz', 'ct.nii', 'ct.mha']:
                local = root/name.replace('.', '_')
                local.mkdir()
                sitk.WriteImage(ct, str(local/name))
                (local/'pred_2d.json').write_text('{"slices": []}')
                self.assertEqual(run_gmm_one_case(str(local), GMMConfig())['pairs'], [])

    def test_line_direction_length_and_quality_gates(self):
        ct = sitk.GetImageFromArray(np.full((3, 30, 230), 100, dtype=np.float32))
        tip, handle = [10, 10, 1, 0], [210, 10, 1, math.pi]
        self.assertGreater(compute_pair_score(ct, tip, handle, (1, 1, 1)), NEG_INF)
        # Opposite arrows alone are insufficient: both point perpendicular to line.
        self.assertEqual(compute_pair_score(ct, [10, 10, 1, math.pi/2],
                         [210, 10, 1, -math.pi/2], (1, 1, 1)), NEG_INF)
        short = [110, 10, 1, math.pi]
        self.assertEqual(compute_pair_score(ct, tip, short, (1, 1, 1)), NEG_INF)
        self.assertGreater(compute_pair_score(ct, tip, short, (1, 1, 1),
                           PairingConfig(length_prior_mm=100, length_tol_mm=5)), NEG_INF)
        air = sitk.GetImageFromArray(np.full((3, 30, 230), -800, dtype=np.float32))
        self.assertEqual(compute_pair_score(air, tip, handle, (1, 1, 1),
                         PairingConfig(min_pair_score=0)), NEG_INF)
        with self.assertRaises(ValueError):
            PairingConfig(length_tol_mm=-1)

    def test_cross_slice_complete_link_and_circular_mean(self):
        def det(x, z, angle, score=.9):
            return dict(x0=x, y0=5, z=z, angle=math.radians(angle), score=score,
                        x_mm=x, y_mm=5, z_mm=z)
        points, info = merge_detections_across_slices(
            [det(0, 0, 179), det(0, 1, -179)], 2.5)
        self.assertEqual(len(points), 1)
        self.assertAlmostEqual(abs(points[0][3]), math.pi)
        # A-B and B-C close, A-C too far: no transitive overmerge.
        points, info = merge_detections_across_slices(
            [det(0, 0, 0), det(2, 1, 0), det(4, 2, 0)], 2.5)
        self.assertEqual(len(points), 2)
        points, info = merge_detections_across_slices(
            [det(0, 0, 0), det(0, 0, 0), det(0, 1, 0)], 2.5)
        self.assertEqual(len(points), 2)

    def test_consolidation_preserves_raw_indices_and_rejection_reasons(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ct = sitk.GetImageFromArray(np.full((4, 20, 230), 100, dtype=np.float32))
            sitk.WriteImage(ct, str(root / 'ct.nii.gz'))
            dets = [dict(x0=x, y0=5, z=z, cls=cls, angle=angle, score=score)
                    for x, z, cls, angle, score in [
                        (10, 0, 1, 0, .1), (10, 1, 1, 0, .9),
                        (10, 2, 1, 0, .8), (210, 1, 0, math.pi, .9),
                        (210, 2, 0, math.pi, .8), (100, 1, 1, 0, .9)]]
            (root / 'pred_2d.json').write_text(json.dumps({'slices': [{'dets': dets}]}))
            for disable in (True, False):
                out = run_gmm_one_case(str(root), GMMConfig(n_prior=5, disable_nocross=disable))
                self.assertEqual(out['counts']['pairs_final'], 1)
                self.assertEqual(out['pairs'][0]['tip_indices'], [1, 2])
                self.assertEqual(out['pairs'][0]['handle_indices'], [0, 1])
                self.assertEqual(out['unmatched']['low_confidence_tip_raw_indices'], [0])
                self.assertEqual(out['unmatched']['tip'][0]['raw_indices'], [3])
                self.assertEqual(out['unmatched']['tip'][0]['reason'], 'no_eligible_candidate')
            rejected = run_gmm_one_case(str(root), GMMConfig(n_prior=5, min_pair_confidence=.95))
            self.assertEqual(rejected['pairs'], [])
            self.assertGreater(rejected['candidate_diagnostics']['low_pair_confidence'], 0)

    def test_export_rejects_wrong_split_and_keeps_empty_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            image = dict(id=1, case_id='case_a', slice_index=0, width=10, height=10)
            paths = export_pred2d([], [image], temporary)
            self.assertEqual(json.loads(paths[0].read_text())['slices'][0]['dets'], [])
            with self.assertRaisesRegex(ValueError, 'absent'):
                export_pred2d([{'image_id': 99}], [image], temporary)


if __name__ == '__main__':
    unittest.main()
