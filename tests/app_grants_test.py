from job_hunter.app_grants import Requirement, effective_privileges, is_missing, required_privileges

APP = {
    "service_principal_client_id": "sp-1",
    "resources": [
        {"name": "sql-warehouse", "sql_warehouse": {"id": "w1", "permission": "CAN_USE"}},
        {"name": "jobs-index", "uc_securable": {"securable_full_name": "cat.sch.jobs_index",
                                                "securable_type": "TABLE", "permission": "SELECT"}},
        {"name": "cv-volume", "uc_securable": {"securable_full_name": "cat.sch.cv",
                                               "securable_type": "VOLUME", "permission": "READ_VOLUME"}},
    ],
}


def test_required_privileges_include_parents_once():
    assert required_privileges(APP) == [
        Requirement("catalog", "cat", "USE_CATALOG"),
        Requirement("schema", "cat.sch", "USE_SCHEMA"),
        Requirement("table", "cat.sch.jobs_index", "SELECT"),
        Requirement("volume", "cat.sch.cv", "READ_VOLUME"),
    ]
    assert required_privileges({"resources": None}) == []


def test_effective_privileges_count_inherited_ones_for_the_principal_only():
    response = {"privilege_assignments": [
        {"principal": "sp-1", "privileges": [{"privilege": "SELECT", "inherited_from_type": "SCHEMA"}]},
        {"principal": "someone", "privileges": [{"privilege": "READ_VOLUME"}]},
    ]}
    held = effective_privileges(response, "sp-1")
    assert held == {"SELECT"}
    assert not is_missing(Requirement("table", "cat.sch.t", "SELECT"), held)
    assert is_missing(Requirement("volume", "cat.sch.cv", "READ_VOLUME"), held)
    assert not is_missing(Requirement("volume", "cat.sch.cv", "READ_VOLUME"), {"ALL_PRIVILEGES"})
    assert effective_privileges({}, "sp-1") == set()
