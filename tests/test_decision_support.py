import unittest

from decision_support import (
    compute_data_quality_score,
    compute_moneyline_probability,
    infer_decision,
    recommendation_snapshot,
    weighted_injury_count,
)


class DecisionSupportTests(unittest.TestCase):
    def test_weighted_injury_count_respects_status(self):
        report = {
            "injuries": [
                {"status": "Out"},
                {"status": "Questionable"},
                {"status": "Probable"},
            ]
        }
        self.assertAlmostEqual(weighted_injury_count(report), 1.55, places=2)

    def test_moneyline_probability_adjusts_from_base(self):
        result = compute_moneyline_probability(
            base_probability=0.55,
            selected_rest_days=2,
            opponent_rest_days=0,
            selected_weighted_injuries=0.0,
            opponent_weighted_injuries=1.5,
            selected_ticket_pct=45.0,
            selected_money_pct=61.0,
            opponent_ticket_pct=55.0,
            opponent_money_pct=39.0,
            selected_ml_move=-15,
            opponent_ml_move=15,
        )
        self.assertGreater(result["adjusted_probability"], 0.55)

    def test_data_quality_penalizes_unsupported_and_conflicting_inputs(self):
        quality = compute_data_quality_score(
            market_supported=False,
            current_odds_found=True,
            selected_side_found=False,
            opposite_side_found=False,
            elo_source="not_initialized",
            rest_data_complete=False,
            injury_data_complete=False,
            roster_data_complete=False,
            public_data_complete=False,
            line_snapshots=0,
            submitted_vs_market_delta=25,
            conflicting_signals=True,
        )
        self.assertLess(quality["score"], 40)
        self.assertTrue(quality["penalties"])

    def test_infer_decision_bet_for_strong_supported_edge(self):
        decision = infer_decision(
            market_supported=True,
            model_probability=0.60,
            fair_probability_no_vig=0.50,
            edge_pct=10.0,
            expected_value_per_unit=0.14,
            data_quality_score=80,
            llm_edge_pct=12.0,
        )
        self.assertEqual(decision, "BET")

    def test_infer_decision_lean_for_unsupported_market(self):
        decision = infer_decision(
            market_supported=False,
            model_probability=None,
            fair_probability_no_vig=None,
            edge_pct=None,
            expected_value_per_unit=None,
            data_quality_score=70,
            llm_edge_pct=6.0,
        )
        self.assertEqual(decision, "LEAN")

    def test_recommendation_snapshot_sets_stake_for_bets_only(self):
        bet = recommendation_snapshot(
            odds=150,
            bankroll=1000,
            market_supported=True,
            model_probability=0.50,
            market_implied_probability=0.40,
            fair_probability_no_vig=0.40,
            data_quality_score=80,
            llm_edge_pct=5.0,
        )
        self.assertEqual(bet["decision"], "BET")
        self.assertGreater(bet["stake_amount"], 0.0)

        lean = recommendation_snapshot(
            odds=150,
            bankroll=1000,
            market_supported=False,
            model_probability=None,
            market_implied_probability=0.40,
            fair_probability_no_vig=0.39,
            data_quality_score=70,
            llm_edge_pct=5.0,
        )
        self.assertEqual(lean["decision"], "LEAN")
        self.assertEqual(lean["stake_amount"], 0.0)


if __name__ == "__main__":
    unittest.main()
