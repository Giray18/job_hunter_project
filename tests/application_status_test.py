from datetime import UTC, datetime

from job_hunter import application_status, html_export
from job_hunter.records import ApplyType, StatusValue

NOW = datetime(2026, 10, 21, 9, 0, tzinfo=UTC)


def test_make_statuses_dedupes_and_strips():
    rows = application_status.make_statuses([" 1", "2", "1", "", "3 "], StatusValue.CLOSED, "pipeline", "n", NOW)
    assert [r.job_id for r in rows] == ["1", "2", "3"]
    assert all(r.status is StatusValue.CLOSED and r.source == "pipeline" and r.updated_at == NOW for r in rows)


def test_applied_is_never_overwritten_by_a_later_status():
    # MERGE updates an existing row only when it is not 'applied', or when the new status is 'applied'.
    assert application_status.UPDATE_WHEN == "t.status <> 'applied' OR s.status = 'applied'"


def test_every_apply_type_has_page_instructions():
    for kind in ApplyType:
        assert html_export.apply_instructions(kind.value)
