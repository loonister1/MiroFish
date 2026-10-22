from app.services.report_agent import Report, ReportManager, ReportStatus


def test_simulation_lookup_returns_the_newest_report_after_a_regeneration(tmp_path, monkeypatch):
    monkeypatch.setattr(ReportManager, "REPORTS_DIR", str(tmp_path))
    # Folder names sort the old report last, so a first-match lookup would pick the newer one;
    # the reverse order is what put the user on a stale report.
    for report_id, created_at in (
        ("report_b_old", "2026-10-14T16:29:22"),
        ("report_a_new", "2026-10-14T16:37:15"),
        ("report_z_old", "2026-10-14T16:00:00"),
    ):
        ReportManager.save_report(Report(
            report_id=report_id, simulation_id="sim-1", graph_id="g", simulation_requirement="r",
            status=ReportStatus.COMPLETED, created_at=created_at,
        ))

    assert ReportManager.get_report_by_simulation("sim-1").report_id == "report_a_new"
    assert ReportManager.get_report_by_simulation("sim-2") is None
