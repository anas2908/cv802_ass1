"""Synthetic crutch-protection checks; no reconstruction or mask files are read."""

import copy
import types
import unittest

import numpy as np

from filter_person_crutches import combine_selections, validate_protection
from crutch_protection.build_protection import capsule_gate, confirmation_gate


def _approved_protection():
    return {
        "ready_for_filter": True,
        "review": {"status": "approved_for_experiment"},
        "selection": {
            "minimum_annotated_views": 3,
            "minimum_distinct_track_views": 3,
            "minimum_pairwise_camera_ray_separation_degrees_for_confirming_triple": 15.,
            "max_mean_baseline_pixel_error": 3.,
            "minimum_triangulation_angle_degrees": 1.5,
        },
        "annotated_image_corridors": {f"synthetic_{i}.jpg": {"shaft": {
            "polylines": [[[.5, .25], [.5, .75]]],
            "half_width_fraction_of_image_width": .02,
        }} for i in (1, 2, 3)},
        "regions": {label: {"capsules": [{
            "start_world": [x, 0., -1.], "end_world": [x, 0., 1.], "radius": .1,
        }]} for label, x in (("A", 0.), ("B", 1.))},
    }


def _synthetic_views(angles):
    """Pinhole cameras on an arc, all aimed at the origin and its thin ROI."""
    camera = types.SimpleNamespace(
        width=100,
        height=100,
        img_from_cam=lambda xyz: np.asarray(xyz)[:, :2] / np.asarray(xyz)[:, 2, None] * 30 + 50,
    )
    images, definitions = {}, {}
    for image_id, angle in enumerate(angles, 1):
        radians = np.radians(angle)
        center = np.array([5 * np.sin(radians), 0., -5 * np.cos(radians)])
        forward = -center / np.linalg.norm(center)
        right = np.cross([0., 1., 0.], forward)
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)
        rotation = np.stack([right, up, forward])
        pose = types.SimpleNamespace(
            rotation=types.SimpleNamespace(matrix=lambda rotation=rotation: rotation.copy()),
            translation=-rotation @ center,
        )
        name = f"synthetic_{image_id}.jpg"
        images[image_id] = types.SimpleNamespace(
            name=name, camera_id=1,
            cam_from_world=lambda pose=pose: pose,
            projection_center=lambda center=center: center.copy(),
        )
        definitions[name] = {"shaft": {
            "polylines": [[[.5, .25], [.5, .75]]],
            "half_width_fraction_of_image_width": .02,
        }}
    return types.SimpleNamespace(images=images, cameras={1: camera}), definitions


class CombinedSelectionTest(unittest.TestCase):
    def test_body_and_protected_union_both_require_distinct_track_views(self):
        result = combine_selections(
            body=np.array([True, False, False, True, False, False]),
            protected=np.array([False, True, False, True, True, False]),
            distinct_views=np.array([2, 3, 4, 3, 2, 4]),
        )

        np.testing.assert_array_equal(result, [False, True, False, True, False, False])
        self.assertEqual(result.dtype, np.dtype(bool))

    def test_custom_stronger_track_gate_applies_to_protected_points_too(self):
        result = combine_selections(
            body=np.array([True, False, True, False]),
            protected=np.array([False, True, False, True]),
            distinct_views=np.array([3, 3, 4, 4]), min_track_views=4,
        )

        np.testing.assert_array_equal(result, [False, False, True, True])


class ProtectionApprovalTest(unittest.TestCase):
    def test_approved_protection_meeting_all_thresholds_is_accepted(self):
        validate_protection(_approved_protection())

    def test_missing_or_unapproved_review_cannot_enable_protection(self):
        cases = []
        not_ready = _approved_protection()
        not_ready["ready_for_filter"] = False
        cases.append(not_ready)
        pending = _approved_protection()
        pending["review"]["status"] = "pending"
        cases.append(pending)
        no_review = _approved_protection()
        del no_review["review"]
        cases.append(no_review)
        for protection in cases:
            with self.subTest(protection=protection):
                with self.assertRaises(ValueError):
                    validate_protection(protection)

    def test_each_weakened_geometric_requirement_is_rejected(self):
        weakened = {
            "minimum_annotated_views": 2,
            "minimum_pairwise_camera_ray_separation_degrees_for_confirming_triple": 14.99,
            "max_mean_baseline_pixel_error": 3.01,
            "minimum_triangulation_angle_degrees": 1.49,
        }
        for key, value in weakened.items():
            protection = _approved_protection()
            protection["selection"][key] = value
            with self.subTest(requirement=key):
                with self.assertRaises(ValueError):
                    validate_protection(protection)


class CrutchConfirmationTest(unittest.TestCase):
    def test_capsules_are_bounded_and_do_not_supply_image_confirmation(self):
        xyz = np.array([[0., 0., 0.], [.05, 0., 1.04], [.2, 0., 0.], [0., 0., 1.2]])
        capsules = [{"start_world": [0., 0., -1.], "end_world": [0., 0., 1.], "radius": .1}]
        contained = capsule_gate(xyz, capsules)
        np.testing.assert_array_equal(contained, [True, True, False, False])

        # Three nearby cameras see the synthetic shaft, but do not provide
        # three rays separated by 15 degrees. Capsule membership is insufficient.
        model, definitions = _synthetic_views([0., 5., 10.])
        votes, confirmed, _ = confirmation_gate(model, xyz[:1], definitions)
        np.testing.assert_array_equal(votes, [3])
        np.testing.assert_array_equal(confirmed, [False])
        np.testing.assert_array_equal(contained[:1] & confirmed, [False])

    def test_three_separated_annotated_views_confirm_an_existing_point(self):
        model, definitions = _synthetic_views([-30., 0., 30.])

        votes, confirmed, per_view = confirmation_gate(model, np.array([[0., 0., 0.]]), definitions)

        np.testing.assert_array_equal(votes, [3])
        np.testing.assert_array_equal(confirmed, [True])
        self.assertEqual(len(per_view), 3)

    def test_every_ray_pair_in_the_confirming_triple_must_be_separated(self):
        model, definitions = _synthetic_views([-30., -25., 30.])

        votes, confirmed, _ = confirmation_gate(model, np.array([[0., 0., 0.]]), definitions)

        np.testing.assert_array_equal(votes, [3])
        np.testing.assert_array_equal(confirmed, [False])

    def test_a_missing_exposed_corridor_vote_prevents_confirmation(self):
        model, definitions = _synthetic_views([-30., 0., 30.])
        definitions = copy.deepcopy(definitions)
        definitions["synthetic_3.jpg"]["shaft"]["polylines"] = [[[.1, .25], [.1, .75]]]

        votes, confirmed, _ = confirmation_gate(model, np.array([[0., 0., 0.]]), definitions)

        np.testing.assert_array_equal(votes, [2])
        np.testing.assert_array_equal(confirmed, [False])


if __name__ == "__main__":
    unittest.main()
