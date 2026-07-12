import unittest

import numpy as np

from kpm_bridge.contracts import (
    KPMContract,
    ContractError,
    compile_contract_mapping,
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


if __name__ == "__main__":
    unittest.main()
