from job_hunter import linkedin_jobs

PAGE = """
<h2 class="top-card-layout__title font-sans topcard__title">Data Engineer (f/m/x)</h2>
<a class="topcard__org-name-link topcard__flavor--black-link" href="https://example.com">
  NETCONOMY
</a>
<span class="topcard__flavor topcard__flavor--bullet">
  Graz, Styria, Austria
</span>
<div class="description__text description__text--rich">
  <section class="show-more-less-html" data-max-lines="5">
    <div class="show-more-less-html__markup show-more-less-html__markup--clamp-after-5
        relative overflow-hidden">
      We build data platforms.<br><br><strong>Your Job</strong><ul><li>Build ETL &amp; ELT</li><li>Use GCP</li></ul>
    </div>
    <button class="show-more-less-html__button">Show more</button>
  </section>
</div>
<ul class="description__job-criteria-list">
  <li class="description__job-criteria-item">
    <h3 class="description__job-criteria-subheader">Seniority level</h3>
    <span class="description__job-criteria-text description__job-criteria-text--criteria">
      Mid-Senior level
    </span>
  </li>
  <li class="description__job-criteria-item">
    <h3 class="description__job-criteria-subheader">Employment type</h3>
    <span class="description__job-criteria-text description__job-criteria-text--criteria">Full-time</span>
  </li>
</ul>
"""


def test_parse_job_posting():
    posting = linkedin_jobs.parse_job_posting(PAGE)
    assert posting["title"] == "Data Engineer (f/m/x)"
    assert posting["company"] == "NETCONOMY"
    assert posting["location"] == "Graz, Styria, Austria"
    assert posting["description"] == "We build data platforms.\n\nYour Job\n- Build ETL & ELT\n\n- Use GCP"
    assert posting["criteria"] == {"Seniority level": "Mid-Senior level", "Employment type": "Full-time"}


def test_parse_job_posting_without_description():
    posting = linkedin_jobs.parse_job_posting("<html>login wall</html>")
    assert posting["description"] is None
    assert posting["criteria"] == {}
