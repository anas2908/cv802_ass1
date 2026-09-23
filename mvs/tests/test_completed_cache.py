"""Real tiny staged artifacts exercise read-only completed-run loading."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image, ImageDraw

from cv802_mvs.config import MVSConfig
from cv802_mvs.errors import InputValidationError, StageConflictError
from cv802_mvs.io_utils import atomic_json
from cv802_mvs.paths import PathPolicy
from cv802_mvs.runner import ExecutionMetrics, MVSRunner
from cv802_mvs.staging import stage_inputs
from test_paths_config_commands import raw_config

TEST_TEMP = Path('/l/users/anas.khan/cv_802_ass1/mvs/runtime/unit-tests')


def snapshot(root: Path) -> dict:
    return {str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns,
             hashlib.sha256(path.read_bytes()).hexdigest())
            for path in sorted(root.rglob('*')) if path.is_file()}


class CompletedCacheTests(unittest.TestCase):
    def setUp(self):
        TEST_TEMP.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=TEST_TEMP)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.policy = PathPolicy.for_tests(self.root)
        source = self.root / 'sfm'
        for name in ('images', 'masks', 'model'):
            (source / name).mkdir(parents=True)
        records = {}
        for name in ('a.png', 'b.png'):
            Image.new('RGB', (10, 20), (30, 60, 90)).save(source / 'images' / name)
            mask = Image.new('L', (10, 20), 0)
            ImageDraw.Draw(mask).rectangle((2, 5, 7, 14), fill=255)
            mask_path = source / 'masks' / name
            mask.save(mask_path)
            records[name] = {'usable': True, 'status': 'valid', 'mask_relative_path': 'masks/' + name,
                             'mask_sha256': hashlib.sha256(mask_path.read_bytes()).hexdigest()}
        model = source / 'model'
        (model / 'cameras.txt').write_text('1 PINHOLE 10 20 8 8 5 10\n')
        (model / 'images.txt').write_text('1 1 0 0 0 0 0 0 1 a.png\n\n2 1 0 0 0 1 0 0 1 b.png\n\n')
        (model / 'points3D.txt').write_text('# empty staging model\n')
        atomic_json(source / 'mask_manifest.json', {'complete': True, 'images': records})
        stage_inputs(policy=self.policy, experiment='pilot', images_source=source / 'images',
                     model_source=model, masks_source=source / 'masks',
                     mask_manifest_source=source / 'mask_manifest.json')
        raw = raw_config(self.root)
        raw['masking'] = {'mode': 'black_background', 'dilation_pixels': 0}
        self.config = MVSConfig.parse(raw, self.policy)
        self.runner = MVSRunner(self.config, self.policy, command_executor=self.execute)
        self.paths = self.runner.paths
        with patch('cv802_mvs.runner.validate_runtime', return_value=({'test_runtime': True}, {})):
            self.original = self.runner.run()
        self.executor = Mock(side_effect=AssertionError('completed cache must never execute COLMAP'))
        self.runner.command_executor = self.executor

    def execute(self, command, **kwargs):
        if command.name == 'undistort':
            shutil.copytree(self.paths.masked_images, self.paths.workspace / 'images')
            shutil.copytree(self.paths.model, self.paths.workspace / 'sparse')
            stereo = self.paths.workspace / 'stereo'
            stereo.mkdir()
            (stereo / 'patch-match.cfg').write_text('a.png\n__auto__, 20\nb.png\n__auto__, 20\n')
        elif command.name == 'patch_match':
            for kind in ('depth_maps', 'normal_maps'):
                directory = self.paths.workspace / 'stereo' / kind
                directory.mkdir()
                for name in ('a.png', 'b.png'):
                    (directory / (name + '.geometric.bin')).write_bytes(b'mock COLMAP map')
        elif command.name == 'fusion':
            self.paths.fused.write_text('ply\nformat ascii 1.0\nelement vertex 2\n'
                'property float x\nproperty float y\nproperty float z\n'
                'property uchar red\nproperty uchar green\nproperty uchar blue\n'
                'end_header\n0 1 2 10 20 30\n3 4 5 40 50 60\n')
        else:
            raise AssertionError(command.name)
        return ExecutionMetrics(0, 1.25, 123, 456, 78)

    def assert_readonly_rejects(self, error_type):
        before = snapshot(self.paths.root)
        with patch('cv802_mvs.runner.validate_runtime', side_effect=AssertionError('no CUDA probe')) as runtime:
            with self.assertRaises(error_type):
                self.runner.run(resume=True)
            runtime.assert_not_called()
        self.executor.assert_not_called()
        self.assertEqual(before, snapshot(self.paths.root))

    def test_completed_cache_validates_every_stage_without_writes_or_runtime(self):
        before = snapshot(self.paths.root)
        with patch('cv802_mvs.runner.validate_runtime', side_effect=AssertionError('no CUDA probe')) as runtime, \
             patch('cv802_mvs.runner.atomic_json', side_effect=AssertionError('no metadata rewrite')), \
             patch.object(self.runner, '_completed_receipt', wraps=self.runner._completed_receipt) as receipts:
            result = self.runner.run(resume=True)
        self.assertEqual(result, {**self.original, 'cache_hit': True})
        self.assertEqual([call.args[0] for call in receipts.call_args_list], ['mask_inputs', 'undistort', 'patch_match', 'fusion'])
        runtime.assert_not_called()
        self.executor.assert_not_called()
        self.assertEqual(before, snapshot(self.paths.root))

    def test_corrupted_ply_fails_without_gpu_or_recomputation(self):
        self.paths.fused.write_text(self.paths.fused.read_text().replace('0 1 2 10', 'nan 1 2 10'))
        self.assert_readonly_rejects(InputValidationError)

    def test_valid_but_changed_ply_fails_hash_check(self):
        self.paths.fused.write_text(self.paths.fused.read_text().replace('0 1 2 10', '9 1 2 10'))
        self.assert_readonly_rejects(StageConflictError)

    def test_missing_mask_receipt_never_regenerates_masked_images(self):
        (self.paths.receipts / 'mask_inputs.json').unlink()
        self.assert_readonly_rejects(StageConflictError)

    def test_missing_final_receipt_does_not_rewrite_a_complete_run(self):
        (self.paths.manifests / 'final_validation.json').unlink()
        self.assert_readonly_rejects(StageConflictError)

    def test_changed_input_provenance_identity_is_rejected(self):
        path = self.paths.manifests / 'run_manifest.json'
        manifest = json.loads(path.read_text())
        manifest['input_validation']['input_provenance_digest'] = 'different inputs'
        atomic_json(path, manifest)
        self.assert_readonly_rejects(StageConflictError)

    def test_changed_registered_image_count_is_rejected(self):
        path = self.paths.manifests / 'final_validation.json'
        final = json.loads(path.read_text())
        final['registered_images'] = 999
        atomic_json(path, final)
        self.assert_readonly_rejects(StageConflictError)

    def test_incomplete_resume_still_enters_existing_runtime_path(self):
        self.runner._write_state('failed', 'simulated interrupted publication')
        with patch('cv802_mvs.runner.validate_runtime', side_effect=RuntimeError('expected runtime path')) as runtime:
            with self.assertRaisesRegex(RuntimeError, 'expected runtime path'):
                self.runner.run(resume=True)
        runtime.assert_called_once()


if __name__ == '__main__':
    unittest.main()
