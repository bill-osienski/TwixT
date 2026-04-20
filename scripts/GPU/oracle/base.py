"""Base oracle class for JS subprocess communication."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


class JSOracle:
    """Interface to JS oracle scripts via Node.js subprocess.

    Each oracle script reads JSON from stdin and writes JSON to stdout.
    The input format varies by function, but output always includes:
    - result: The function return value
    - error: null or error message string
    """

    def __init__(self, script_path: Path):
        """Initialize oracle with path to JS script.

        Args:
            script_path: Path to the JS oracle script

        Raises:
            FileNotFoundError: If script doesn't exist
            RuntimeError: If Node.js is not available
        """
        self.script_path = Path(script_path)
        if not self.script_path.exists():
            raise FileNotFoundError(f"JS oracle not found: {self.script_path}")

        # Verify Node.js is available
        self._check_node()

    def _check_node(self) -> None:
        """Verify Node.js is installed and accessible."""
        try:
            result = subprocess.run(
                ["node", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                raise RuntimeError("Node.js check failed")
        except FileNotFoundError:
            raise RuntimeError("Node.js not found in PATH. Please install Node.js.")

    def call(
        self,
        function_name: str,
        input_data: Dict[str, Any],
        timeout: int = 30
    ) -> Any:
        """Call a function in the JS oracle.

        Args:
            function_name: Name of the function to call
            input_data: Dictionary of input parameters
            timeout: Timeout in seconds

        Returns:
            The function result (type varies by function)

        Raises:
            RuntimeError: On oracle errors or timeout
        """
        # Add function name to input
        request = {
            "function": function_name,
            **input_data
        }

        try:
            result = subprocess.run(
                ["node", str(self.script_path)],
                input=json.dumps(request),
                capture_output=True,
                text=True,
                timeout=timeout
            )

            if result.returncode != 0:
                raise RuntimeError(f"JS oracle error: {result.stderr}")

            output = json.loads(result.stdout)
            if output.get("error"):
                raise RuntimeError(f"JS oracle returned error: {output['error']}")

            return output.get("result")

        except subprocess.TimeoutExpired:
            raise RuntimeError(f"JS oracle timed out after {timeout}s")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"JS oracle output parse error: {e}\nstdout: {result.stdout}")

    def call_raw(self, input_data: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
        """Call oracle with raw input (for backward compatibility).

        Args:
            input_data: Raw JSON input for the oracle
            timeout: Timeout in seconds

        Returns:
            Full response dict with 'result' or legacy fields
        """
        try:
            result = subprocess.run(
                ["node", str(self.script_path)],
                input=json.dumps(input_data),
                capture_output=True,
                text=True,
                timeout=timeout
            )

            if result.returncode != 0:
                return {"error": result.stderr}

            return json.loads(result.stdout)

        except subprocess.TimeoutExpired:
            return {"error": f"Timeout after {timeout}s"}
        except json.JSONDecodeError as e:
            return {"error": f"Parse error: {e}"}


def state_to_json(state: "GameState", player: str) -> Dict[str, Any]:
    """Convert Python GameState to JSON format for JS oracle.

    Args:
        state: Python GameState object
        player: Current player ("red" or "black")

    Returns:
        Dict suitable for JSON serialization
    """
    pegs = []
    for (row, col), p in state.pegs.items():
        pegs.append({"row": row, "col": col, "player": p})

    bridges = []
    for (r1, c1), (r2, c2) in state.bridges:
        bridges.append({"r1": r1, "c1": c1, "r2": r2, "c2": c2})

    return {
        "boardSize": state.board_size,
        "pegs": pegs,
        "bridges": bridges,
        "player": player,
        "toMove": state.to_move,
        "moveCount": len(state.move_history)
    }


def metrics_to_json(metrics: Dict[str, Any], player: str) -> Dict[str, Any]:
    """Convert Python component metrics to JSON format.

    Args:
        metrics: Dict from component_metrics()
        player: Current player

    Returns:
        Dict suitable for JSON serialization
    """
    component = []
    for row, col in metrics.get("largest_component", []):
        component.append({"row": row, "col": col})

    return {
        "component": component,
        "touchesTop": metrics.get("touches_top", False),
        "touchesBottom": metrics.get("touches_bottom", False),
        "touchesLeft": metrics.get("touches_left", False),
        "touchesRight": metrics.get("touches_right", False),
        "maxRowSpan": metrics.get("max_row_span", 0),
        "maxColSpan": metrics.get("max_col_span", 0),
        "minRow": metrics.get("min_row"),
        "maxRow": metrics.get("max_row"),
        "minCol": metrics.get("min_col"),
        "maxCol": metrics.get("max_col"),
    }
