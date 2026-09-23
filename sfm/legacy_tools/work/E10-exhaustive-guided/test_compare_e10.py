#!/usr/bin/env python3
"""Focused self-tests for E10 final-only comparison and preservation guards."""
from pathlib import Path
from types import SimpleNamespace
import contextlib,copy,importlib.util,io,json,tempfile,unittest
from unittest import mock
import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('e10_comparison',HERE/'compare_e10.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)

class ComparisonTests(unittest.TestCase):
    def test_final_paths_do_not_use_internal_or_raw_E10(self):
        for subject in c.SUBJECTS:
            first,second=c.source_paths(subject)
            self.assertEqual(second,c.ROOT/'reconstructions/experiments/E10_quality_exhaustive_guided_consensus90'/subject)
            self.assertNotIn('/work/',str(second))
            self.assertNotEqual(first,second)
            self.assertEqual(c.source_paths(subject,True),(first,first))

    def test_scope_requires_only_light_and_rejects_dark(self):
        args=c.arguments([])
        self.assertEqual(args.subjects,['light_shirt'])
        self.assertEqual(c.SUBJECTS,('light_shirt',))
        with self.assertRaises(ValueError):c.source_paths('black_shirt_crutches')
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):c.arguments(['--subjects','black_shirt_crutches'])

    def test_cleanup_policy_must_be_same_pooled_90_percent(self):
        policy={'foreground_agreement':.9,'minimum_distinct_track_views':3,'minimum_positive_views':25}
        self.assertTrue(c.check_cleanup_policy(policy,dict(policy))['passed'])
        for changed in ({**policy,'foreground_agreement':.97},{**policy,'minimum_positive_views':24}):
            with self.assertRaises(ValueError):c.check_cleanup_policy(policy,changed)
        with self.assertRaises(ValueError):c.check_cleanup_policy(None,policy)

    def test_common_gate_rejects_repeated_view_and_error_angle_failures(self):
        # Five synthetic points: valid, same-image repeat, high error, low angle, outside fixed bounds.
        values={'xyz':np.array([[0,0,0],[.1,0,0],[.2,0,0],[.3,0,0],[3,0,0]],float),
            'tracks':np.array([3,5,3,3,3]),'distinct_views':np.array([3,2,3,3,3]),
            'angles':np.array([2,2,2,1,2.]),'native':np.array([2,2,4,2,2.]),
            'normalized':np.array([2,2,3.1,2,2.]),'all_observations_valid':np.ones(5,dtype=bool),
            'stored_native':np.array([2,2,4,2,2.]),'invalid_observations':0,'observations_without_baseline_image':0}
        model=SimpleNamespace(points3D={i:SimpleNamespace(color=np.array([i,10,20])) for i in range(5)},num_reg_images=lambda:4)
        args=SimpleNamespace(min_track=3,max_error=3.,min_angle=1.5)
        common=(model,{},np.eye(3),np.zeros(3),np.full(3,-1.),np.full(3,1.),args)
        with mock.patch('compare_models.point_measurements',return_value=values),mock.patch('plot_comparison.point_measurements',return_value=values):
            metrics,grids=c.summarize(*common);plot=c.comparable_points(*common)
        self.assertEqual(metrics['retained_comparable_points'],1)
        self.assertEqual(metrics['points_with_track_at_least_minimum'],5)
        self.assertEqual(metrics['points_with_distinct_views_at_least_minimum'],4)
        self.assertEqual(metrics['quality_points_outside_fixed_baseline_bounds'],1)
        self.assertTrue(np.array_equal(plot['xyz'],[[0,0,0]]))
        for key,value in plot['summary'].items():self.assertEqual(metrics[key],value)

    def test_repeated_input_check_catches_geometry_or_metric_changes(self):
        first={'xyz':np.array([[1.,2,3]]),'rgb':np.array([[.1,.2,.3]])}
        metrics={'points':1};grids={'grid':{(1,2,3)}}
        self.assertTrue(all(c.check_repeated(first,copy.deepcopy(first),metrics,dict(metrics),grids,copy.deepcopy(grids)).values()))
        second=copy.deepcopy(first);second['xyz'][0,0]+=.001
        with self.assertRaises(AssertionError):c.check_repeated(first,second,metrics,metrics,grids,grids)
        with self.assertRaises(AssertionError):c.check_repeated(first,first,metrics,{'points':2},grids,grids)

    def test_preflight_preserves_existing_output(self):
        parent=HERE/'self_tests';parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as tmp:
            target=Path(tmp)/'existing';target.mkdir();sentinel=target/'keep.txt';sentinel.write_text('unchanged')
            with self.assertRaises(FileExistsError):c.preflight(SimpleNamespace(output_dir=target,subjects=[]))
            self.assertEqual(sentinel.read_text(),'unchanged')

    def test_missing_final_input_does_not_create_output(self):
        parent=HERE/'self_tests';parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as tmp:
            root=Path(tmp);e1=root/'reconstructions/light_shirt/subject_preview';e4=root/'e4';missing=root/'missing_final'
            for p in (e1,e4):p.mkdir(parents=True);(p/'analysis.json').write_text('{}')
            target=root/'comparison'
            with mock.patch.object(c,'ROOT',root),mock.patch.object(c,'source_paths',return_value=(e4,missing)):
                with self.assertRaisesRegex(FileNotFoundError,'missing_final'):
                    c.preflight(SimpleNamespace(output_dir=target,subjects=['light_shirt'],self_test=False))
            self.assertFalse(target.exists())

    def test_cli_preserves_fixed_quality_gates_and_rejects_nan(self):
        args=c.arguments(['--self-test'])
        self.assertEqual((args.min_track,args.max_error,args.min_angle),(3,3.,1.5))
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):c.arguments(['--point-size','nan'])
            with self.assertRaises(SystemExit):c.arguments(['--output-dir',str(c.ROOT/'reconstructions')])

if __name__=='__main__':unittest.main(verbosity=2)
