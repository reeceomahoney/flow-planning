"""Bidirectional safety contract: controller-owned evasion, policy-owned recovery."""

__all__ = ["RecoveryBankConfig", "RecoveryBankController", "build_recovery_bank"]


def __getattr__(name):
    # Keep ``python -m bidirectional_contract.recovery_data`` free of runpy's
    # double-import warning while preserving the small top-level API.
    if name == "RecoveryBankController":
        from .controller import RecoveryBankController

        return RecoveryBankController
    if name in {"RecoveryBankConfig", "build_recovery_bank"}:
        from .recovery_data import RecoveryBankConfig, build_recovery_bank

        return {
            "RecoveryBankConfig": RecoveryBankConfig,
            "build_recovery_bank": build_recovery_bank,
        }[name]
    raise AttributeError(name)
