"""Unit tests for SysAdmin stats service helpers."""
from __future__ import annotations

from app.sysadmin.stats.service import _csv_safe, _month_bounds


def test_month_bounds_start_end():
    start, end = _month_bounds("2026-02")
    assert start.year == 2026 and start.month == 2 and start.day == 1
    assert end.year == 2026 and end.month == 2 and end.day == 28


def test_month_bounds_leap_year():
    start, end = _month_bounds("2024-02")
    assert end.day == 29


def test_csv_safe_neutralises_formula_prefix():
    assert _csv_safe("=SUM(A1)").startswith("\t")
    assert _csv_safe("+1").startswith("\t")
    assert _csv_safe("-1").startswith("\t")
    assert _csv_safe("@foo").startswith("\t")


def test_csv_safe_passes_normal_values():
    assert _csv_safe("normal text") == "normal text"
    assert _csv_safe("") == ""
    assert _csv_safe(None) == ""
    assert _csv_safe(200) == "200"
