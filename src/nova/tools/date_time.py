"""Deterministic Date and Time tools configured for Asia/Kolkata."""

from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Dict, Optional
import zoneinfo


def get_kolkata_timezone() -> Any:
    """Returns Asia/Kolkata timezone object, falling back to fixed UTC+5:30 offset if tzdata is missing."""
    try:
        return zoneinfo.ZoneInfo("Asia/Kolkata")
    except Exception:
        # Fallback to fixed offset UTC+5:30 (Indian Standard Time)
        return timezone(timedelta(hours=5, minutes=30), name="IST")


def get_current_time(
    now_fn: Optional[Callable[[], datetime]] = None,
    timezone_name: str = "Asia/Kolkata",
) -> Dict[str, Any]:
    """Retrieves current time in Asia/Kolkata timezone.
    
    Args:
        now_fn: Optional callable returning a datetime object for deterministic testing.
        timezone_name: Target timezone string (default: Asia/Kolkata).
        
    Returns:
        Structured dictionary with time components and formatted string.
    """
    tz = get_kolkata_timezone()
    dt = now_fn() if now_fn is not None else datetime.now(tz)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    else:
        dt = dt.astimezone(tz)

    # Format 12-hour time e.g. "10:30 AM IST"
    # Format hours without leading zero if single digit (%I -> %-I on Unix, %#I on Windows)
    hour = dt.strftime("%I").lstrip("0")
    minute = dt.strftime("%M")
    am_pm = dt.strftime("%p")
    formatted_value = f"{hour}:{minute} {am_pm} IST"

    return {
        "tool_name": "get_current_time",
        "timezone": timezone_name,
        "time": dt.strftime("%H:%M:%S"),
        "formatted_value": formatted_value,
        "iso": dt.isoformat(),
    }


def get_current_date(
    now_fn: Optional[Callable[[], datetime]] = None,
    timezone_name: str = "Asia/Kolkata",
) -> Dict[str, Any]:
    """Retrieves current date and weekday in Asia/Kolkata timezone.
    
    Args:
        now_fn: Optional callable returning a datetime object for deterministic testing.
        timezone_name: Target timezone string (default: Asia/Kolkata).
        
    Returns:
        Structured dictionary with date components, weekday, and formatted string.
    """
    tz = get_kolkata_timezone()
    dt = now_fn() if now_fn is not None else datetime.now(tz)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    else:
        dt = dt.astimezone(tz)

    weekday = dt.strftime("%A")
    day = dt.strftime("%d").lstrip("0")
    month = dt.strftime("%B")
    year = dt.strftime("%Y")
    formatted_value = f"{weekday}, {day} {month} {year}"

    return {
        "tool_name": "get_current_date",
        "timezone": timezone_name,
        "date": dt.strftime("%Y-%m-%d"),
        "weekday": weekday,
        "formatted_value": formatted_value,
        "iso": dt.isoformat(),
    }
