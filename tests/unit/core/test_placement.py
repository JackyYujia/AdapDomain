from datetime import datetime, timezone

from adapdomain.core.placement import PlacementPlan, ReplicaPlacement, ScoreBreakdown


def test_placement_plan_basic():
    plan = PlacementPlan(
        plan_id="p0",
        created_at=datetime.now(timezone.utc),
        input_replicas=["r0", "r1"],
        placements={"r0": 0},
        rejected={"r1": "no feasible device"},
        score_breakdown={
            ("r0", 0): ScoreBreakdown(ceff=0.5, x=0.1, h=0.0, q=0.0, f=0.0, u=0.0, final=0.4),
        },
        frontier_versions={"bert-base": 1},
    )
    assert plan.placements["r0"] == 0
    assert plan.rejected["r1"] == "no feasible device"
    assert plan.score_breakdown[("r0", 0)].final == 0.4


def test_replica_placement_dataclass():
    rp = ReplicaPlacement(replica_id="r0", device_id=0, frontier_point_id="pt-3")
    assert rp.replica_id == "r0"
