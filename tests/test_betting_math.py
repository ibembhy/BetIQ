import unittest

from betting_math import (
    american_odds_to_decimal,
    american_odds_to_implied_probability,
    expected_value,
    kelly_fraction,
    kelly_stake,
    no_vig_probabilities,
    no_vig_probabilities_from_odds,
)


class BettingMathTests(unittest.TestCase):
    def test_implied_probability_positive_american_odds(self):
        self.assertAlmostEqual(american_odds_to_implied_probability(150), 0.4, places=6)

    def test_implied_probability_negative_american_odds(self):
        self.assertAlmostEqual(
            american_odds_to_implied_probability(-110),
            110 / 210,
            places=6,
        )

    def test_decimal_conversion(self):
        self.assertAlmostEqual(american_odds_to_decimal(-110), 1.9090909, places=6)
        self.assertAlmostEqual(american_odds_to_decimal(150), 2.5, places=6)

    def test_no_vig_probabilities_from_odds_normalizes_market(self):
        home, away = no_vig_probabilities_from_odds(-110, -110)
        self.assertAlmostEqual(home, 0.5, places=6)
        self.assertAlmostEqual(away, 0.5, places=6)

    def test_no_vig_probabilities_accepts_raw_implied_inputs(self):
        probs = no_vig_probabilities([0.55, 0.50])
        self.assertAlmostEqual(sum(probs), 1.0, places=6)
        self.assertAlmostEqual(probs[0], 0.55 / 1.05, places=6)

    def test_expected_value_positive_edge(self):
        ev = expected_value(win_probability=0.55, odds=-110, stake=100)
        self.assertAlmostEqual(ev, 5.0, places=4)

    def test_expected_value_negative_edge(self):
        ev = expected_value(win_probability=0.40, odds=150, stake=100)
        self.assertAlmostEqual(ev, 0.0, places=4)

    def test_kelly_fraction_full_and_fractional(self):
        full = kelly_fraction(win_probability=0.55, odds=-110, fractional_kelly=1.0)
        half = kelly_fraction(win_probability=0.55, odds=-110, fractional_kelly=0.5)
        self.assertGreater(full, 0.0)
        self.assertAlmostEqual(half, full * 0.5, places=6)

    def test_kelly_fraction_caps_at_max_fraction(self):
        fraction = kelly_fraction(
            win_probability=0.80,
            odds=200,
            fractional_kelly=1.0,
            max_fraction=0.12,
        )
        self.assertEqual(fraction, 0.12)

    def test_kelly_stake_returns_amount_and_fraction(self):
        stake, fraction = kelly_stake(
            bankroll=1000,
            win_probability=0.55,
            odds=-110,
            fractional_kelly=0.5,
            max_fraction=0.12,
        )
        self.assertAlmostEqual(stake, 1000 * fraction, places=6)
        self.assertGreater(stake, 0.0)

    def test_zero_odds_is_invalid(self):
        with self.assertRaises(ValueError):
            american_odds_to_implied_probability(0)


if __name__ == "__main__":
    unittest.main()
