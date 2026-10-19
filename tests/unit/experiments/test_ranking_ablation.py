import math
import unittest

from scripts.replay_ranking_ablation import (
    Arrival,
    Occupant,
    Profile,
    RiskTerms,
    feasible,
    ranking_score,
    replay,
    score_with_shared_risks,
    shape_value,
)


class RankingAblationTests(unittest.TestCase):
    def test_harmonic_and_dot_product_can_rank_residuals_differently(self):
        demand = (0.24, 0.45, 0.15)
        close_fit = (0.25, 0.45, 0.15)
        extra_hbm = (0.25, 0.45, 0.50)
        self.assertGreater(ranking_score("harmonic", demand, close_fit), ranking_score("harmonic", demand, extra_hbm))
        self.assertLess(ranking_score("tetris", demand, close_fit), ranking_score("tetris", demand, extra_hbm))


    def test_feasibility_checks_capacity_and_both_phase_latency_guards(self):
        profiles = {
            "prefill": Profile("prefill", 1, (0.7, 0.2, 0.2), 150, 500, 1),
            "decode": Profile("decode", 16, (0.1, 0.3, 0.7), 30, 40, 128),
        }
        interference = {
            ("prefill", "decode"): (1.1, 1.2),
            ("decode", "prefill"): (1.1, 1.5),
        }
        occupant = Occupant("prefill", 100, profiles["prefill"].demand)
        self.assertFalse(feasible(profiles["decode"], [occupant], interference, profiles)[0])
        interference[("decode", "prefill")] = (1.1, 1.2)
        self.assertTrue(feasible(profiles["decode"], [occupant], interference, profiles)[0])


    def test_replay_uses_the_same_trace_and_is_seeded(self):
        profiles = {
            "prefill": Profile("prefill", 1, (0.7, 0.2, 0.2), 100, 500, 1),
            "decode": Profile("decode", 2, (0.1, 0.3, 0.7), 20, 40, 10),
        }
        interference = {
            ("prefill", "decode"): (1.1, 1.2),
            ("decode", "prefill"): (1.1, 1.2),
        }
        trace = [Arrival("decode", 0, 3), Arrival("prefill", 10, 3)]
        results = [replay(policy, profiles, interference, trace, cards=2, seed=3) for policy in ("harmonic", "tetris", "random")]
        self.assertEqual({result["offered_trace_units"] for result in results}, {6})
        self.assertTrue(all(0 <= result["admitted_trace_units"] <= 6 for result in results))
        self.assertEqual(results[-1], replay("random", profiles, interference, trace, cards=2, seed=3))

    def test_risk_aware_variants_share_exactly_the_same_penalty(self):
        demand = (0.24, 0.45, 0.15)
        residual = (0.25, 0.45, 0.50)
        risks = RiskTerms(0.2, 0.3, 0.1, 0.4, 0.5)
        for mode in ("harmonic", "tetris", "cosine", "arithmetic", "minimum", "geometric", "no_shape"):
            score = score_with_shared_risks(mode, demand, residual, risks)
            penalty = 0.40 * shape_value(mode, demand, residual) - score
            self.assertAlmostEqual(penalty, 0.25 * 0.2 + 0.15 * 0.3 + 0.10 * 0.1 + 0.07 * 0.4 + 0.05 * 0.5)

    def test_shape_variants_support_two_and_three_dimensions(self):
        for dimensions in (2, 3):
            demand = (0.24, 0.45, 0.15)[:dimensions]
            residual = (0.25, 0.45, 0.50)[:dimensions]
            for mode in ("harmonic", "tetris", "cosine", "arithmetic", "minimum", "geometric"):
                self.assertTrue(math.isfinite(shape_value(mode, demand, residual)))


if __name__ == "__main__":
    unittest.main()
