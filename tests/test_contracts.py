import unittest

import numpy as np

from kpm_bridge.contracts import (
    KPMContract,
    ContractError,
    compile_contract_mapping,
    compile_transform_plan,
    convert_to_canonical,
)


class ContractTests(unittest.TestCase):
    def _contract(self, **changes):
        base = dict(
            name="DRB.UEThpDl",
            quantity="downlink_throughput",
            unit="kbit/s",
            entity_scope="ue",
            aggregation="mean",
            window_ms=1000,
            clock="e2-node",
            counter_semantics="gauge",
            schema_version="kpm-v3",
            provenance="profile-a",
        )
        base.update(changes)
        return KPMContract(**base)

    def test_unit_conversion(self):
        np.testing.assert_allclose(convert_to_canonical([1000, 2500], "kbit/s"), [1.0, 2.5])
        np.testing.assert_allclose(convert_to_canonical([10, 95], "%"), [0.10, 0.95])

    def test_type_compatibility_rejects_scope_change(self):
        source = self._contract()
        target = self._contract(unit="Mbit/s")
        wrong_scope = self._contract(unit="Mbit/s", entity_scope="cell")
        self.assertTrue(source.type_compatible(target))
        self.assertFalse(source.type_compatible(wrong_scope))
        self.assertTrue(np.isinf(source.semantic_cost(wrong_scope)))

    def test_invalid_window_fails_closed(self):
        with self.assertRaises(ContractError):
            self._contract(window_ms=0)

    def test_compiler_ignores_incompatible_decoy(self):
        target = self._contract(unit="Mbit/s", provenance="canonical")
        valid = self._contract(unit="kbit/s", provenance="profile")
        decoy = self._contract(
            name="RRU.PrbUsedDl",
            quantity="prb_utilisation",
            unit="%",
            provenance="profile",
        )
        mapping = compile_contract_mapping([decoy, valid], [target])
        self.assertEqual(mapping[0].source_index, 1)

    def test_compiler_rejects_missing_quantity(self):
        target = self._contract(unit="Mbit/s", provenance="canonical")
        wrong = self._contract(quantity="uplink_throughput")
        with self.assertRaises(ContractError):
            compile_contract_mapping([wrong], [target])

    def test_executable_plan_converts_units(self):
        source = self._contract(unit="kbit/s")
        target = self._contract(unit="bit/s", provenance="canonical")
        plan = compile_transform_plan([source], [target])
        result = plan.apply(
            np.array([[100.0], [250.0]]),
            reset_mask=np.zeros((2, 1), dtype=bool),
            dt_ms=250.0,
        )
        np.testing.assert_allclose(result[:, 0], [100_000.0, 250_000.0])

    def test_counter_reset_requires_explicit_metadata(self):
        source = self._contract(unit="kbit", counter_semantics="cumulative")
        target = self._contract(unit="bit/s", provenance="canonical")
        plan = compile_transform_plan([source], [target])
        counter = np.array([[100.0], [200.0], [50.0], [100.0]])
        without_reset = plan.apply(
            counter, reset_mask=np.zeros(counter.shape, dtype=bool), dt_ms=250.0
        )
        self.assertTrue(np.isnan(without_reset[2, 0]))
        reset = np.zeros(counter.shape, dtype=bool)
        reset[2, 0] = True
        with_reset = plan.apply(counter, reset_mask=reset, dt_ms=250.0)
        np.testing.assert_allclose(
            with_reset[1:, 0], [400_000.0, 200_000.0, 200_000.0]
        )

    def test_compiled_output_is_source_order_invariant(self):
        source_a = self._contract(name="a", quantity="downlink_throughput")
        source_b = self._contract(
            name="b", quantity="uplink_throughput", unit="Mbit/s"
        )
        target_a = self._contract(
            name="ca", quantity="downlink_throughput", unit="bit/s"
        )
        target_b = self._contract(
            name="cb", quantity="uplink_throughput", unit="bit/s"
        )
        reference = compile_transform_plan([source_a, source_b], [target_a, target_b])
        permuted = compile_transform_plan([source_b, source_a], [target_a, target_b])
        reset = np.zeros((2, 2), dtype=bool)
        expected = reference.apply(
            np.array([[1.0, 2.0], [3.0, 4.0]]), reset_mask=reset, dt_ms=250
        )
        observed = permuted.apply(
            np.array([[2.0, 1.0], [4.0, 3.0]]), reset_mask=reset, dt_ms=250
        )
        np.testing.assert_allclose(observed, expected)


if __name__ == "__main__":
    unittest.main()
