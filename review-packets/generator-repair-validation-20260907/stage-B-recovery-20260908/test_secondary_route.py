"""CPU checks for routing defaults and completion-before-results policy."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest
import secondary_route as route


class SecondaryRouteTests(unittest.TestCase):
    def test_inventory_bound_old_default_is_never_used(self):
        old = Path('old-recovery')
        calls = []
        def inventory(plan, recovery_dir=old):
            calls.append((plan, recovery_dir))
            return {'ok': True}
        secondary = SimpleNamespace(inventory=inventory, R=old)
        route.bind_route(secondary)
        self.assertEqual(secondary.inventory({'x': 1}), {'ok': True})
        self.assertEqual(calls, [({'x': 1}, route.R)])
        with self.assertRaises(AssertionError):
            secondary.inventory({}, recovery_dir=old)

    def test_binding_twice_does_not_stack_or_recurse(self):
        original = Mock(return_value={})
        secondary = SimpleNamespace(inventory=original)
        route.bind_route(secondary)
        route.bind_route(secondary)
        secondary.inventory({})
        original.assert_called_once_with({}, recovery_dir=route.R)

    def test_premature_results_do_not_call_original_reader(self):
        secondary = SimpleNamespace(R=route.R, completed_results=Mock())
        with patch.object(Path, 'is_file', return_value=False):
            with self.assertRaises(AssertionError):
                route.completed_results(secondary, {})
        secondary.completed_results.assert_not_called()

    def test_all_three_markers_checked_before_frozen_result_reader(self):
        calls = []
        secondary = SimpleNamespace(R=route.R,
            completed_results=lambda path, plan: calls.append(('results', path)) or {'original': True})
        def exists(path):
            calls.append(('marker', path))
            return True
        with patch.object(Path, 'is_file', exists):
            self.assertEqual(route.completed_results(secondary, {}), {'original': True})
        self.assertEqual([kind for kind, _ in calls], ['marker', 'marker', 'marker', 'results'])
        self.assertTrue(all(path.is_relative_to(route.R) for _, path in calls))

    def test_results_cannot_route_to_r1(self):
        secondary = SimpleNamespace(R=route.R, completed_results=Mock())
        with self.assertRaises(AssertionError):
            route.completed_results(secondary, {}, route.R.parent / 'stage-B-recovery-20260907/analysis-recovery/results.json')
        secondary.completed_results.assert_not_called()

    def test_workload_uses_same_routed_module_without_replacing_functions(self):
        secondary = SimpleNamespace(R=route.R)
        audit = Mock(return_value={'workload': True})
        workload = SimpleNamespace(secondary=secondary, audit=audit)
        with patch.object(route, 'load_module', return_value=workload):
            self.assertEqual(route.workload_report(secondary), {'workload': True})
        audit.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
