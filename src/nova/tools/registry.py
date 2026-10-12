"""Allowlisted Tool Registry for safe deterministic execution."""

from typing import Any, Callable, Dict, List, Optional
from .date_time import get_current_date, get_current_time


class ToolRegistry:
    """Secure, allowlist-only tool execution registry.
    
    Prevents arbitrary code execution and validates tool invocations.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Callable[..., Dict[str, Any]]] = {}
        self._descriptions: Dict[str, str] = {}
        self._execution_history: List[Dict[str, Any]] = []

    def register(
        self,
        name: str,
        func: Callable[..., Dict[str, Any]],
        description: str = "",
    ) -> None:
        """Registers an authorized Python callable under a strict name."""
        if not callable(func):
            raise TypeError(f"Tool {name} must be a callable.")
        self._tools[name] = func
        self._descriptions[name] = description

    def is_registered(self, name: str) -> bool:
        return name in self._tools

    def list_tools(self) -> Dict[str, str]:
        return dict(self._descriptions)

    def execute(
        self,
        name: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Safely executes an authorized tool by name.
        
        Args:
            name: Exact name of the tool in the registry
            kwargs: Validated arguments passed to the tool
            
        Returns:
            Dictionary containing success status, result data, or error message
        """
        if name not in self._tools:
            return {
                "success": False,
                "error": f"Tool '{name}' is not in the allowlisted tool registry.",
                "tool_name": name,
            }

        try:
            result = self._tools[name](**kwargs)
            record = {
                "success": True,
                "tool_name": name,
                "result": result,
            }
            self._execution_history.append(record)
            return record
        except Exception as e:
            err_record = {
                "success": False,
                "error": f"Tool '{name}' failed with error: {str(e)}",
                "tool_name": name,
            }
            self._execution_history.append(err_record)
            return err_record

    @property
    def history(self) -> List[Dict[str, Any]]:
        return list(self._execution_history)


# Create and populate standard registry
default_registry = ToolRegistry()
default_registry.register(
    "get_current_time",
    get_current_time,
    description="Returns the current time formatted in Asia/Kolkata timezone.",
)
default_registry.register(
    "get_current_date",
    get_current_date,
    description="Returns the current date and day of week formatted in Asia/Kolkata timezone.",
)
