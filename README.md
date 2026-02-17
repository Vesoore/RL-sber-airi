# DDPG + Hindsight Experience Replay (HER)

Implementation of **Deep Deterministic Policy Gradient (DDPG)** with **Hindsight Experience Replay (HER)** for goal-conditioned robotic control.

**Environment:** `FetchReach-v4` (Gymnasium-Robotics) — a robotic arm must reach a randomly sampled target position in 3D space.

## References

- Lillicrap et al., [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971), 2015
- Andrychowicz et al., [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495), 2017

## Results

| Experiment | HER | Goal strategy | 100% Eval SR at episode | Early stop at episode |
|---|---|---|---|---|
| DDPG (baseline) | ✗ | — | ~2700 | 3700 |
| DDPG + HER | ✓ | `final` | ~200 | 1200 |
| DDPG + HER | ✓ | `episode` | ~100 | 1100 |
| DDPG + HER | ✓ | `future` | ~200 | 1200 |

All experiments use early stopping with `patience=10` (no eval improvement for 10 consecutive evaluations, every 100 episodes).

## Key Takeaways

1. **HER dramatically accelerates learning.** All three HER variants reach 100% eval success rate within 100–200 episodes, while vanilla DDPG needs ~2700 episodes — a roughly **15× speedup**.

2. **Vanilla DDPG does solve the task, but slowly and unstably.** It gradually climbs from 0% to 100% over thousands of episodes with noticeable oscillation (eval SR drops to 0% even after periods of 50% success).

3. **All HER strategies perform similarly on FetchReach.** The `episode`, `final`, and `future` strategies converge at roughly the same rate. This is expected — FetchReach is a relatively simple task with short episodes (50 steps), so the choice of hindsight goal matters less. On harder tasks (FetchPush, FetchSlide) differences between strategies would become more pronounced.

4. **`future` is the recommended default.** Despite similar performance here, the original HER paper shows `future` to be the most robust strategy across diverse tasks.


## Architecture

- **Actor:** MLP (256 → 256 → 256) with ReLU activations and sigmoid output scaled to action bounds
- **Critic:** MLP (256 → 256 → 256) with ReLU activations and linear output
- **Target networks:** Polyak averaging (τ = 0.95)
- **Replay buffer:** trajectory-based, 1M transitions, with optional HER relabelling (50% probability)
