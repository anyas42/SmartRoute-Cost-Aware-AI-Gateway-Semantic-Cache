"""
Monitoring and metrics computation for SmartRoute.
"""
from typing import Dict, Any, List
import numpy as np


def compute_percentiles(values: List[float], percentiles: List[int] = [50, 90, 95, 99]) -> Dict[str, float]:
    """Calculate arbitrary percentiles over a numeric sequence."""
    if not values:
        return {f"p{p}": 0.0 for p in percentiles}
    return {f"p{p}": round(float(np.percentile(values, p)), 2) for p in percentiles}
