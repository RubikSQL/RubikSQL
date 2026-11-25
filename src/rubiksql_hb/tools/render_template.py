"""SQL template rendering: render_sql_template."""
import re
from typing import Any, Dict


def render_sql_template(template: str, params: Dict[str, Any]) -> str:
    """
    Render a SQL template by substituting {param_name} placeholders.

    Supports any Python-format-string style placeholder: {table}, {value},
    {limit}, etc.  Raises KeyError if a placeholder has no matching param.

    Args:
        template: SQL string with {name} placeholders.
        params:   Dict mapping placeholder names to their values.

    Returns:
        Rendered SQL string.

    Example:
        render_sql_template(
            "SELECT * FROM {table} WHERE status = '{status}' LIMIT {limit}",
            {"table": "orders", "status": "paid", "limit": 10},
        )
        # → "SELECT * FROM orders WHERE status = 'paid' LIMIT 10"
    """
    # Find all placeholder names in the template
    placeholders = re.findall(r'\{(\w+)\}', template)
    missing = [p for p in placeholders if p not in params]
    if missing:
        raise KeyError(f"Missing template parameters: {missing}")

    return template.format_map({k: params[k] for k in placeholders})


def render_template(template: str, params: dict) -> str:
    """Render a SQL template by substituting {param_name} placeholders."""
    try:
        return render_sql_template(template, params)
    except KeyError as e:
        return f"ERROR: {e}"
