from dataclasses import replace

import pytest
from conftest import query

from clir_benchmark.adapters import query_from_row, select_candidates
from clir_benchmark.errors import ValidationError


def test_exported_winner_is_honored():
    first = query(id="first", total_score=20)
    winner = query(id="winner", is_best=True, candidate_rank=2, total_score=19)
    selected, audit = select_candidates([first, winner])
    assert selected == [winner]
    assert audit[0]["method"] == "exported_is_best"


def test_grader_score_then_rank_then_id():
    queries = [
        query(id="a", total_score=10),
        query(id="z", total_score=20, candidate_rank=2),
        query(id="c", total_score=20),
        query(id="b", total_score=20),
    ]
    selected, audit = select_candidates(queries)
    assert selected[0].id == "b"
    assert audit[0]["method"] == "grader_total_score"
    assert len(select_candidates(queries, "all")[0]) == 4


def test_independent_tasks_and_generators():
    queries = [
        query(id="a"),
        query(id="b", generator_model_id="generator-b"),
        query(id="c", task_id="task-c"),
    ]
    assert len(select_candidates(queries)[0]) == 3


@pytest.mark.parametrize(
    "queries,match",
    [
        ([query(is_best=True), query(id="q2", is_best=True)], "Multiple"),
        ([query(task_id="")], "identity"),
        ([query(total_score=None)], "total_score"),
        ([query(), query()], "Duplicate"),
    ],
)
def test_invalid_selection_is_actionable(queries, match):
    with pytest.raises(ValidationError, match=match):
        select_candidates(queries)


def test_all_mode_does_not_require_scores():
    assert select_candidates([query(total_score=None)], "all")[0][0].total_score is None


def test_rank_one_is_not_assumed_best():
    low = query(candidate_rank=1, total_score=10, is_best=False)
    high = replace(low, id="high", candidate_rank=3, total_score=30)
    assert select_candidates([low, high])[0] == [high]


def test_adapter_generation_identity():
    import json

    row = {
        "_id": "q",
        "question": "question",
        "question_language": "en",
        "source_unit_id": "a:en",
        "generator_model_id": "g",
        "provenance_json": json.dumps({"file": "run.sqlite", "task": "task"}),
        "metadata_json": "{}",
    }
    legal = query_from_row(row, "un")
    assert "run.sqlite" in legal.task_id and "task" in legal.task_id
    row["metadata_json"] = json.dumps({"decider_routing_task": '["gp","patent","en"]'})
    chemistry = query_from_row(row, "chemistry")
    assert "patent" in chemistry.task_id
    row["provenance_json"] = "{}"
    with pytest.raises(ValidationError, match="identity"):
        query_from_row(row, "chemistry")
