"""Tests for safe deterministic ToolRegistry and Date/Time tools."""

from datetime import datetime
import pytest

from nova.tools.registry import ToolRegistry
from nova.tools.date_time import get_current_time, get_current_date, get_kolkata_timezone


def mock_clock_time() -> datetime:
    tz = get_kolkata_timezone()
    # Mock fixed time: 2026-10-12 10:30:00 IST
    return datetime(2026, 10, 12, 10, 30, 0, tzinfo=tz)


def test_get_current_time_mock():
    res = get_current_time(now_fn=mock_clock_time)
    assert res["tool_name"] == "get_current_time"
    assert res["timezone"] == "Asia/Kolkata"
    assert res["time"] == "10:30:00"
    assert res["formatted_value"] == "10:30 AM IST"


def test_get_current_date_mock():
    res = get_current_date(now_fn=mock_clock_time)
    assert res["tool_name"] == "get_current_date"
    assert res["timezone"] == "Asia/Kolkata"
    assert res["date"] == "2026-10-12"
    assert res["weekday"] == "Monday"
    assert res["formatted_value"] == "Monday, 12 October 2026"


def test_tool_registry_allowlist():
    reg = ToolRegistry()
    reg.register("get_current_time", get_current_time)

    # Allowed tool
    res = reg.execute("get_current_time", now_fn=mock_clock_time)
    assert res["success"] is True
    assert res["result"]["formatted_value"] == "10:30 AM IST"

    # Disallowed tool
    blocked = reg.execute("unauthorized_system_command")
    assert blocked["success"] is False
    assert "not in the allowlisted" in blocked["error"]


def test_tool_registry_error_handling():
    reg = ToolRegistry()

    def buggy_tool():
        raise RuntimeError("Sensor failure")

    reg.register("buggy", buggy_tool)
    res = reg.execute("buggy")
    assert res["success"] is False
    assert "Sensor failure" in res["error"]
