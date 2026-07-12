"""Core research kernel for KPM-Bridge."""

from .calibration import QualityBudget, admit, split_conformal_radius
from .certificates import TelemetryCertificate
from .contracts import KPMContract, ContractError, compile_contract_mapping, convert_to_canonical

__version__ = "1.0.0"

__all__ = [
    "ContractError",
    "KPMContract",
    "QualityBudget",
    "TelemetryCertificate",
    "__version__",
    "admit",
    "compile_contract_mapping",
    "convert_to_canonical",
    "split_conformal_radius",
]
