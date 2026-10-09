from adapdomain.runtime.nightshift import MigrationCandidate, NightShift


def test_nightshift_returns_topk_by_delta_ceff_under_budget():
    candidates = [
        MigrationCandidate(replica_id=f"r{i}", src_device=0, dst_device=1, delta_ceff=d, risk=0.1)
        for i, d in enumerate([0.05, 0.20, 0.15, 0.30])
    ]
    plan = NightShift(migration_budget=2).select(candidates)
    assert [c.replica_id for c in plan] == ["r3", "r1"]


def test_nightshift_skips_high_risk():
    candidates = [
        MigrationCandidate(replica_id="r0", src_device=0, dst_device=1, delta_ceff=0.50, risk=0.9),
        MigrationCandidate(replica_id="r1", src_device=0, dst_device=1, delta_ceff=0.30, risk=0.1),
    ]
    plan = NightShift(migration_budget=2, risk_max=0.5).select(candidates)
    assert [c.replica_id for c in plan] == ["r1"]
