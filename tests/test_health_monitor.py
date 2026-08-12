from __future__ import annotations

import unittest

from health_monitor import OnlineHealthMonitor, parse_online_step


class OnlineHealthMonitorTest(unittest.TestCase):
    def test_parse_verl_step_summary(self) -> None:
        parsed = parse_online_step(
            "step:7 - actor/kl_loss:0.01 - critic/rewards/mean:-0.25 - actor/entropy:0.3"
        )
        self.assertIsNotNone(parsed)
        step, metrics = parsed or (0, {})
        self.assertEqual(step, 7)
        self.assertEqual(metrics["critic/rewards/mean"], -0.25)

    def test_kl_sudden_change_requires_relative_and_absolute_change(self) -> None:
        monitor = OnlineHealthMonitor(cooldown_updates=5)
        self.assertIsNone(
            monitor.add_step(
                1,
                {"actor/kl_loss": 0.01, "critic/rewards/mean": 1.0},
            )
        )
        self.assertIsNone(
            monitor.add_step(
                2,
                {"actor/kl_loss": 0.015, "critic/rewards/mean": 1.0},
            )
        )
        trigger = monitor.add_step(
            3,
            {"actor/kl_loss": 0.04, "critic/rewards/mean": 1.0},
        )
        self.assertIsNotNone(trigger)
        rules = {row["name"] for row in (trigger or {}).get("rules", [])}
        self.assertEqual(rules, {"kl_sudden_change"})
        parameters = (trigger or {})["operational_parameters"]
        self.assertEqual(parameters["kl_change_ratio_threshold"], 0.5)
        self.assertEqual(parameters["kl_change_absolute_threshold"], 0.02)

    def test_flat_reward_without_prior_drawdown_does_not_trigger(self) -> None:
        monitor = OnlineHealthMonitor(cooldown_updates=5)
        triggers = []
        for step in range(1, 8):
            triggers.append(monitor.add_step(
                step,
                {"actor/kl_loss": 1.0, "critic/rewards/mean": 0.0},
            ))
        self.assertTrue(all(trigger is None for trigger in triggers))

    def test_gradual_kl_growth_does_not_trigger_sudden_change(self) -> None:
        monitor = OnlineHealthMonitor(cooldown_updates=5)
        values = [1.0, 1.2, 1.44, 1.40, 1.68, 2.016, 2.4192]
        triggers = [
            monitor.add_step(
                step,
                {"actor/kl_loss": value, "critic/rewards/mean": 1.0},
            )
            for step, value in enumerate(values, 1)
        ]
        self.assertTrue(all(trigger is None for trigger in triggers))

    def test_entropy_relative_collapse_triggers_without_absolute_floor(self) -> None:
        monitor = OnlineHealthMonitor(cooldown_updates=1)
        self.assertIsNone(
            monitor.add_step(
                1,
                {
                    "actor/kl_loss": 0.01,
                    "critic/rewards/mean": 0.2,
                    "actor/entropy": 0.5,
                },
            )
        )
        trigger = monitor.add_step(
            2,
            {
                "actor/kl_loss": 0.01,
                "critic/rewards/mean": 0.2,
                "actor/entropy": 0.2,
            },
        )
        rules = (trigger or {}).get("rules", [])
        self.assertEqual({row["name"] for row in rules}, {"entropy_sudden_collapse"})
        self.assertAlmostEqual(rules[0]["observed_drop_ratio"], 0.6)

    def test_five_step_window_trend_detects_reward_decline(self) -> None:
        monitor = OnlineHealthMonitor(
            cooldown_updates=1,
            reward_window_size=3,
            reward_trend_min_drawdown=0.1,
            reward_trend_tolerance=0.0,
        )
        rewards = [0.5, 0.5, 0.5, 0.4, 0.2]
        trigger = None
        for step, reward in enumerate(rewards, 1):
            trigger = monitor.add_step(
                step,
                {"actor/kl_loss": 0.01, "critic/rewards/mean": reward},
            )
        self.assertIsNotNone(trigger)
        names = {row["name"] for row in (trigger or {}).get("rules", [])}
        self.assertEqual(names, {"reward_trend_degradation"})
        self.assertEqual((trigger or {})["severity"], "high")

    def test_dynamic_low_plateau_triggers_without_a_fixed_floor(self) -> None:
        monitor = OnlineHealthMonitor(
            cooldown_updates=1,
            reward_window_size=3,
            reward_trend_min_drawdown=0.1,
            reward_trend_tolerance=0.01,
        )
        trigger = None
        for step, reward in enumerate(
            [0.5, 0.5, 0.5, -0.4, -0.5, -0.5, -0.5, -0.5, -0.5],
            1,
        ):
            trigger = monitor.add_step(
                step,
                {"actor/kl_loss": 0.01, "critic/rewards/mean": reward},
            )
        rules = (trigger or {}).get("rules", [])
        self.assertEqual({row["name"] for row in rules}, {"reward_trend_degradation"})
        self.assertEqual(rules[0]["trend_kind"], "low_plateau")

    def test_reward_recovery_breaks_the_trend(self) -> None:
        monitor = OnlineHealthMonitor(
            cooldown_updates=1,
            reward_window_size=3,
            reward_trend_min_drawdown=0.1,
            reward_trend_tolerance=0.01,
        )
        triggers = []
        for step, reward in enumerate([0.5, 0.2, 0.4, 0.1, 0.3], 1):
            triggers.append(monitor.add_step(
                step,
                {
                    "actor/kl_loss": 0.01,
                    "critic/rewards/mean": reward,
                },
            ))
        self.assertTrue(all(trigger is None for trigger in triggers))

    def test_absolute_kl_and_entropy_do_not_create_extra_rules(self) -> None:
        monitor = OnlineHealthMonitor(cooldown_updates=1)
        triggers = []
        for step in range(1, 7):
            triggers.append(monitor.add_step(
                step,
                {
                    "actor/kl_loss": 0.4,
                    "critic/rewards/mean": -0.2,
                    "actor/entropy": 10.0,
                },
            ))
        self.assertTrue(all(trigger is None for trigger in triggers))

    def test_step_parser_does_not_treat_time_per_step_as_global_step(self) -> None:
        self.assertIsNone(
            parse_online_step("perf/time_per_step:1268.48 - actor/entropy:0.3")
        )


if __name__ == "__main__":
    unittest.main()
