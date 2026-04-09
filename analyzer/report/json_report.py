"""
JSON Report Generator
======================
Assembles all analysis results into a single, structured report dict
and serialises it to a JSON file.

Report structure
----------------
{
  "meta": {
      "analyzer_version": "1.0.0",
      "generated_at": "<ISO datetime>",
      "log_file": "<filename>",
      "drone_profile": "<profile id>",
      "drone_name": "<profile name>",
  },
  "flight_summary": {
      "overall_score": float,
      "grade": str,
      "verdict": str,
      "action": str,
      "issue_summary": {critical: N, warning: N, info: N, total: N},
  },
  "module_results": {
      "flight_overview": { score, grade, metrics, issues, summary },
      "battery":         { ... },
      "motors":          { ... },
  },
  "all_issues": [ {severity, code, module, message, ...} ],
  "recommendations": [ {priority, code, module, recommendation} ],
  "raw_metrics": {
      "flight_overview": { ... },
      "battery": { ... },
      "motors": { ... },
  }
}
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from analyzer import __version__

logger = logging.getLogger(__name__)


def generate(
    parse_result_meta: Dict[str, Any],
    module_results: Dict[str, Dict[str, Any]],
    scoring_result: Dict[str, Any],
    profile_info: Dict[str, str],
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build the full report dict and optionally write it to a JSON file.

    Parameters
    ----------
    parse_result_meta : dict with keys: filepath, message_count, available_types, duration_s
    module_results    : {module_name: result_dict} from each analysis module
    scoring_result    : output from scoring.engine.compute()
    profile_info      : {"id": ..., "name": ..., "type": ...}
    output_path       : if provided, write JSON to this path

    Returns
    -------
    The complete report as a Python dict.
    """
    now = datetime.now(timezone.utc).isoformat()
    log_path = parse_result_meta.get("filepath", "unknown")

    report: Dict[str, Any] = {
        "meta": {
            "analyzer_version": __version__,
            "generated_at": now,
            "log_file": Path(log_path).name,
            "log_file_path": log_path,
            "log_duration_s": parse_result_meta.get("duration_s", 0.0),
            "log_start_time": parse_result_meta.get("log_start_time", ""),
            "log_message_count": parse_result_meta.get("message_count", 0),
            "log_message_types": parse_result_meta.get("available_types", []),
            "drone_profile_id": profile_info.get("id", "unknown"),
            "drone_name": profile_info.get("name", "unknown"),
            "drone_type": profile_info.get("type", "unknown"),
        },
        "flight_summary": {
            "overall_score": scoring_result["overall_score"],
            "grade": scoring_result["grade"],
            "verdict": scoring_result["verdict"],
            "action": scoring_result["action"],
            "issue_summary": scoring_result["issue_summary"],
            "module_scores": scoring_result["module_scores"],
        },
        "module_results": {
            name: {
                "score": res.get("score"),
                "grade": res.get("grade"),
                "available": res.get("available", True),
                "summary": res.get("summary", ""),
                "issues": res.get("issues", []),
                "metrics": _sanitise_metrics(res.get("metrics", {})),
            }
            for name, res in module_results.items()
        },
        "all_issues": scoring_result.get("all_issues", []),
        "recommendations": scoring_result.get("recommendations", []),
    }

    if output_path:
        _write_json(report, output_path)

    return report


def _write_json(report: Dict, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=_json_default)
    logger.info("Report written to: %s", path)


def _sanitise_metrics(metrics: Dict) -> Dict:
    """
    Recursively replace non-JSON-serialisable values (NaN, inf, numpy types).
    """
    import math
    import numpy as np

    def _clean(v: Any) -> Any:
        if isinstance(v, dict):
            return {k: _clean(val) for k, val in v.items()}
        if isinstance(v, list):
            return [_clean(item) for item in v]
        if isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                return None
            return v
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            val = float(v)
            return None if (math.isnan(val) or math.isinf(val)) else val
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    return _clean(metrics)


def _json_default(obj: Any) -> Any:
    """Fallback serialiser for json.dump."""
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")
