from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Mapping

from config_utils import load_json, write_json_atomic


INDEX_VERSION = 1
SUMMARY_RELATIVE_PATH = Path("summer") / "summer_result.json"


def _summary_text(section: Mapping[str, Any]) -> str:
    values: list[str] = []
    for collection, field in (
        ("problems", "problem"),
        ("useful_directions", "direction"),
        ("ineffective_directions", "direction"),
    ):
        rows = section.get(collection)
        if not isinstance(rows, list):
            continue
        values.extend(
            str(row[field])
            for row in rows
            if isinstance(row, Mapping) and isinstance(row.get(field), str)
        )
    return " ".join(values)


def _has_stage_content(section: Any) -> bool:
    if not isinstance(section, Mapping):
        return False
    return any(
        isinstance(section.get(name), list) and bool(section[name])
        for name in ("problems", "useful_directions", "ineffective_directions")
    )


def build_summary_index(output_root: str | Path) -> dict[str, Any]:
    """Build a derived catalog from authoritative per-run Summer results."""
    root = Path(output_root).expanduser().resolve()
    entries: list[dict[str, Any]] = []
    if not root.is_dir():
        return {"version": INDEX_VERSION, "entries": entries}
    for result_path in sorted(root.glob(f"*/{SUMMARY_RELATIVE_PATH.as_posix()}")):
        try:
            result = load_json(result_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        run_context = result.get("run_context")
        if not isinstance(run_context, Mapping):
            continue
        stages = [
            stage
            for stage in ("hardware", "stability")
            if _has_stage_content(result.get(stage))
        ]
        entries.append(
            {
                "run_id": result_path.parents[1].name,
                "result_path": str(result_path.relative_to(root)),
                "run_context": copy.deepcopy(dict(run_context)),
                "stages": stages,
            }
        )
    return {"version": INDEX_VERSION, "entries": entries}


def rebuild_summary_index(output_root: str | Path) -> Path:
    root = Path(output_root).expanduser().resolve()
    target = root / "summer_index.json"
    write_json_atomic(target, build_summary_index(root))
    return target


def _tokens(query: str) -> list[str]:
    return [token for token in re.split(r"\s+", query.casefold().strip()) if token]


def _current_identity(context: Mapping[str, Any]) -> dict[str, Any]:
    immutable = context.get("immutable_context")
    immutable = immutable if isinstance(immutable, Mapping) else {}
    model = immutable.get("model")
    model = model if isinstance(model, Mapping) else {}
    hardware = immutable.get("hardware")
    hardware = hardware if isinstance(hardware, Mapping) else {}
    workload = immutable.get("workload")
    workload = workload if isinstance(workload, Mapping) else {}
    model_path = model.get("model_path")
    if isinstance(model_path, str) and model_path.startswith("/"):
        model_path = Path(model_path).name
    train_dataset = workload.get("train_dataset")
    if isinstance(train_dataset, str) and "/data/" in train_dataset.replace("\\", "/"):
        train_dataset = train_dataset.replace("\\", "/").rsplit("/data/", 1)[1]
    evaluation_dataset = workload.get("evaluation_dataset")
    if isinstance(evaluation_dataset, str) and "/data/" in evaluation_dataset.replace("\\", "/"):
        evaluation_dataset = evaluation_dataset.replace("\\", "/").rsplit("/data/", 1)[1]
    return {
        "algorithm": workload.get("algorithm"),
        "model": model_path,
        "platform": hardware.get("platform"),
        "train_dataset": train_dataset,
        "evaluation_dataset": evaluation_dataset,
    }


def query_summaries(
    output_root: str | Path,
    *,
    stage: str,
    query: str = "",
    max_results: int = 5,
    current_context: Mapping[str, Any] | None = None,
    exclude_run_id: str | None = None,
) -> dict[str, Any]:
    """Return bounded prior-run summaries, ranked by query and context similarity."""
    if stage not in {"hardware", "stability"}:
        raise ValueError("stage must be hardware or stability")
    if (
        not isinstance(max_results, int)
        or isinstance(max_results, bool)
        or not 1 <= max_results <= 8
    ):
        raise ValueError("max_results must be an integer from 1 to 8")
    root = Path(output_root).expanduser().resolve()
    index_path = root / "summer_index.json"
    if not index_path.is_file():
        return {
            "found": False,
            "stage": stage,
            "query": query,
            "results": [],
        }
    try:
        index = load_json(index_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return {
            "found": False,
            "stage": stage,
            "query": query,
            "results": [],
        }
    entries = index.get("entries")
    entries = entries if isinstance(entries, list) else []
    query_tokens = _tokens(query)
    identity = _current_identity(current_context or {})
    ranked: list[tuple[int, str, dict[str, Any]]] = []

    for entry in entries:
        if not isinstance(entry, Mapping) or entry.get("run_id") == exclude_run_id:
            continue
        if stage not in entry.get("stages", []):
            continue
        relative = entry.get("result_path")
        if not isinstance(relative, str):
            continue
        result_path = (root / relative).resolve()
        try:
            result_path.relative_to(root)
        except ValueError:
            continue
        if not result_path.is_file():
            continue
        try:
            result = load_json(result_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        section = result.get(stage)
        if not isinstance(section, Mapping):
            continue
        haystack = (
            _summary_text(section)
            + " "
            + json.dumps(result.get("run_context", {}), ensure_ascii=False)
        ).casefold()
        matches = sum(token in haystack for token in query_tokens)
        if query_tokens and matches == 0:
            continue
        run_context = result.get("run_context")
        run_context = run_context if isinstance(run_context, Mapping) else {}
        context_score = sum(
            1
            for key in (
                "algorithm",
                "model",
                "platform",
                "train_dataset",
                "evaluation_dataset",
            )
            if identity.get(key) is not None and identity.get(key) == run_context.get(key)
        )
        ranked.append(
            (
                matches * 10 + context_score,
                str(entry.get("run_id", "")),
                {
                    "run_context": copy.deepcopy(dict(run_context)),
                    "problems": copy.deepcopy(section.get("problems", [])),
                    "useful_directions": copy.deepcopy(
                        section.get("useful_directions", [])
                    ),
                    "ineffective_directions": copy.deepcopy(
                        section.get("ineffective_directions", [])
                    ),
                    "source": relative,
                },
            )
        )

    ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
    results = [row[2] for row in ranked[:max_results]]
    response = {
        "found": bool(results),
        "stage": stage,
        "query": query,
        "results": results,
    }
    if results:
        response["interpretation"] = (
            "Prior-run summaries are hypothesis-level evidence only. Their trial IDs are "
            "local to the source run and must never be used as current-run reference_trial_id values."
        )
    return response
