"""HarmonicFit greedy planner (paper Section 4.3)."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone

from adapdomain.core.frontier import FrontierSnapshot
from adapdomain.core.interference import InterferenceTable
from adapdomain.core.placement import PlacementPlan, ScoreBreakdown
from adapdomain.core.shape import TenantShape
from adapdomain.core.types import (
    EPSILON_DOMAIN_FIT,
    DeviceId,
    FrontierVersion,
    ModelId,
    ReplicaId,
)

from .feasibility import FeasibilityFilter
from .fragmentation import fragmentation_risk
from .harmonic_fit import edge_rho
from .hbm_risk import hbm_risk
from .interference_risk import interference_risk
from .queue_risk import queue_risk
from .residual_state import ClusterState
from .scoring import RiskTerms, ScoreWeights, compute_score


class HarmonicFitPlanner:
    def __init__(
        self,
        *,
        feasibility: FeasibilityFilter,
        weights: ScoreWeights,
        eta: float,
        hbm_high: float,
        queue_high: float,
        epsilon: float = EPSILON_DOMAIN_FIT,
    ) -> None:
        self._feasibility = feasibility
        self._weights = weights
        self._eta = eta
        self._hbm_high = hbm_high
        self._queue_high = queue_high
        self._eps = epsilon

    def plan(
        self,
        replicas: Sequence[TenantShape],
        *,
        state: ClusterState,
        snapshots: Mapping[str, FrontierSnapshot],
        interference: InterferenceTable,
    ) -> PlacementPlan:
        edges: list[tuple[float, TenantShape, int, ScoreBreakdown]] = []
        rejected: dict[ReplicaId, str] = {}

        for replica in replicas:
            profile_key = replica.profile_key or replica.model_name
            snap = snapshots.get(profile_key)
            if snap is None:
                rejected[replica.replica_id] = "no_frontier_snapshot"
                continue
            for did in state.device_ids():
                residual = state.residual_of(did)
                ok, reason = self._feasibility.check(replica, snap, residual)
                if not ok:
                    rejected.setdefault(replica.replica_id, reason.value if reason else "infeasible")
                    continue
                domain_names = [name for name in residual.residual if name in replica.domains]
                if not domain_names:
                    rejected.setdefault(replica.replica_id, "no_scored_domains")
                    continue
                rho = edge_rho(replica, residual, domains=domain_names, eps=self._eps)
                risks = RiskTerms(
                    x=interference_risk(
                        candidate=profile_key,
                        colocated=state.resident_profile_keys_of(did),
                        bucket="mid",
                        table=interference,
                        eta=self._eta,
                    ),
                    h=hbm_risk(residual, high_threshold=self._hbm_high),
                    q=queue_risk(residual, high_threshold=self._queue_high),
                    f=fragmentation_risk(residual, replica),
                    u=0.0 if residual.placed_replicas else 0.5,
                )
                breakdown = compute_score(rho=rho.rho_per_domain, weights=self._weights, risks=risks, eps=self._eps)
                edges.append((breakdown.final, replica, did, breakdown))

        edges.sort(key=lambda x: x[0], reverse=True)
        placements: dict[ReplicaId, DeviceId] = {}
        breakdowns: dict[tuple[ReplicaId, DeviceId], ScoreBreakdown] = {}
        for score, replica, did, breakdown in edges:
            if replica.replica_id in placements:
                continue
            # Re-check feasibility now that earlier picks may have used residual.
            residual = state.residual_of(did)
            key = replica.profile_key or replica.model_name
            ok, _reason = self._feasibility.check(replica, snapshots[key], residual)
            if not ok:
                continue
            placements[replica.replica_id] = DeviceId(did)
            breakdowns[(replica.replica_id, DeviceId(did))] = breakdown
            state.apply_placement(replica=replica, device_id=did)

        for r in replicas:
            if r.replica_id not in placements and r.replica_id not in rejected:
                rejected[r.replica_id] = "no_feasible_device_after_packing"

        return PlacementPlan(
            plan_id=f"plan-{int(datetime.now(timezone.utc).timestamp())}",
            created_at=datetime.now(timezone.utc),
            input_replicas=[r.replica_id for r in replicas],
            placements=placements,
            rejected=rejected,
            score_breakdown=breakdowns,
            frontier_versions={ModelId(key): FrontierVersion(snapshots[key].version)
                               for r in replicas if (key := r.profile_key or r.model_name) in snapshots},
        )
