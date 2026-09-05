"""Regression test for the SysAdmin Portal import/mapper-configuration bug.

`SysRequestLog`, `SysRequestStats`, and `SysAuditLog` live in
`app.models.sysadmin_log`, not `app.models.sysadmin_user`. Several call
sites previously imported them from the wrong module, causing ImportError
at runtime (e.g. POST /sysadmin/auth/login 500ing because record_audit()
did the bad import). Separately, `SysAdminUser.audit_logs` is a
string-based relationship to "SysAuditLog", which only resolves if both
model modules have been imported into the same registry before mapper
configuration runs.
"""
from __future__ import annotations

import importlib

import pytest
from sqlalchemy.orm import configure_mappers


SYSADMIN_MODULES = [
    "app.models.sysadmin_user",
    "app.models.sysadmin_log",
    "app.sysadmin.router",
    "app.sysadmin.deps",
    "app.sysadmin.security",
    "app.sysadmin.audit_service",
    "app.sysadmin.auth.router",
    "app.sysadmin.audit.router",
    "app.sysadmin.tenants.router",
    "app.sysadmin.costs.router",
    "app.sysadmin.stats.router",
    "app.sysadmin.stats.service",
]


@pytest.mark.parametrize("module_name", SYSADMIN_MODULES)
def test_sysadmin_module_imports_cleanly(module_name):
    importlib.import_module(module_name)


def test_sysadmin_mappers_configure_without_error():
    """Would have caught the SysAuditLog string-relationship resolution bug."""
    for module_name in SYSADMIN_MODULES:
        importlib.import_module(module_name)
    configure_mappers()


def test_log_model_classes_importable_from_correct_module():
    from app.models.sysadmin_log import SysRequestLog, SysRequestStats, SysAuditLog

    assert SysRequestLog.__tablename__ == "sysrequestlog"
    assert SysRequestStats.__tablename__ == "sysrequeststats"
    assert SysAuditLog.__tablename__ == "sysauditlog"


def test_log_model_classes_not_importable_from_user_module():
    """Guards against the exact regression that caused the production 500."""
    import app.models.sysadmin_user as sysadmin_user

    for name in ("SysRequestLog", "SysRequestStats", "SysAuditLog"):
        assert not hasattr(sysadmin_user, name), (
            f"{name} should not be defined/re-exported from app.models.sysadmin_user; "
            "it belongs in app.models.sysadmin_log"
        )
