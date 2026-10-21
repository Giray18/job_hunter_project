from job_hunter import ranking
from job_hunter.ranking import RankedJob
from job_hunter.records import Priority


def _job(job_id: str, priority: Priority, fit: int, similarity: float = 0.5) -> RankedJob:
    return RankedJob(job_id=job_id, title_en="t", company="c", country_code=None, url="u", priority=priority,
                     fit_score=fit, similarity=similarity, reasons="r")


def test_tiers_croatia_then_remote_then_unclear():
    jobs = [
        _job("remote-90", Priority.REMOTE, 90),
        _job("hr-60", Priority.CROATIA, 60),
        _job("hr-59", Priority.CROATIA, 59),
        _job("remote-70", Priority.REMOTE, 70),
        _job("unclear-95", Priority.REMOTE_UNCLEAR, 95),
        _job("excluded-99", Priority.EXCLUDED, 99),
    ]
    # Weak Croatian jobs (< 60) compete with unclear jobs on fit score in tier 3.
    assert [j.job_id for j in ranking.rank(jobs, croatia_min_fit=60)] == [
        "hr-60", "remote-90", "remote-70", "unclear-95", "hr-59",
    ]


def test_weak_croatian_job_ranks_above_weaker_unclear_job():
    jobs = [_job("unclear-40", Priority.REMOTE_UNCLEAR, 40), _job("hr-55", Priority.CROATIA, 55)]
    assert [j.job_id for j in ranking.rank(jobs, croatia_min_fit=60)] == ["hr-55", "unclear-40"]


def test_unclear_jobs_fill_remaining_slots():
    jobs = [_job("hr-80", Priority.CROATIA, 80), _job("unclear-70", Priority.REMOTE_UNCLEAR, 70)]
    assert [j.job_id for j in ranking.top_jobs(jobs, croatia_min_fit=60, n=10)] == ["hr-80", "unclear-70"]


def test_similarity_breaks_ties_and_top_n_limits():
    jobs = [_job("a", Priority.REMOTE, 80, 0.4), _job("b", Priority.REMOTE, 80, 0.9), _job("c", Priority.REMOTE, 50)]
    assert [j.job_id for j in ranking.top_jobs(jobs, croatia_min_fit=60, n=2)] == ["b", "a"]


def test_fewer_than_n_jobs_returns_all_eligible():
    jobs = [_job("a", Priority.CROATIA, 10), _job("b", Priority.EXCLUDED, 99)]
    assert [j.job_id for j in ranking.top_jobs(jobs, croatia_min_fit=60)] == ["a"]
