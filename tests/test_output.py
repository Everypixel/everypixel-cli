from everypixel_cli.output import apply_jq, emit_human


def test_simple_jq_selector_without_optional_jq_package():
    assert apply_jq({"task_id": "abc", "status": "PENDING"}, ".task_id") == "abc"


def test_keywords_human_output_uses_table(capsys):
    emit_human(
        {"status": "ok", "keywords": [{"keyword": "cat", "score": 0.98765}]},
        no_color=True,
    )

    output = capsys.readouterr().out

    assert "Keywords" in output
    assert "cat" in output
    assert "0.9877" in output


def test_faces_human_output_uses_table(capsys):
    emit_human(
        {
            "status": "ok",
            "faces": [
                {"score": 0.91, "bbox": [1, 2, 3, 4], "age": 32, "gender": "male"}
            ],
        },
        no_color=True,
    )

    output = capsys.readouterr().out

    assert "Faces" in output
    assert "[1, 2, 3, 4]" in output
    assert "male" in output


def test_quality_human_output_uses_table(capsys):
    emit_human(
        {"status": "ok", "quality": {"score": 0.383275, "class": "high"}}, no_color=True
    )

    output = capsys.readouterr().out

    assert "Quality" in output
    assert "score" in output
    assert "0.3833" in output
    assert "high" in output


def test_generation_cost_matches_task_table_style(capsys, monkeypatch):
    from rich.text import Text

    monkeypatch.setenv("FORCE_COLOR", "1")
    emit_human({"task_id": "abc", "status": "SUCCESS", "billed_cost": "0.076"})
    output = capsys.readouterr().out
    assert "\x1b[36mtask_id" in output
    assert "\x1b[36mbilled_cost" in output
    rows = Text.from_ansi(output).plain.splitlines()
    task_row = next(row for row in rows if "task_id" in row)
    cost_row = next(row for row in rows if "billed_cost" in row)
    assert task_row.index("abc") == cost_row.index("$0.076")
