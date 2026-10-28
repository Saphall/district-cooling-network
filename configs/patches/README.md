# dcngen patches

`0001-swim-fsed-extensions.patch` applies on upstream dcngen commit `2641335` (see `configs/dcngen.lock.yaml`).
It adds what the five FSED experiments need: `StuckValveInjection`, `inject_stuck_valve`,
`PoorBalancingInjection` / `inject_poor_balancing`, `ExcessControlMarginInjection` / `inject_excess_control_margin`
in `faults/injector.py`, and in `orchestrate/scenario.py` the `stuck_valve` / `poor_balancing` hooks, the
`desired_flow` result channel and the "actual delivered vs demanded" `unmet` definition.

## Provenance

The edits were made in place on the Drive copy of dcngen from the Colab notebook. This patch was
**reconstructed** by replaying those notebook cells on the pinned commit, then compared with the working
copy `docs/archive/dcngen/scenario.py`. That file differs from the patched result only by two
latent defects of the in-place edits, which are intentionally **not** reproduced:

- `run_fixed_dt_scenario` also got a `desired_flow=` line referring to an undefined name (the fixed-dt runner
  raises `NameError`; the eps-NTU runner used by every experiment is unaffected);
- a duplicated `desired_flow_rec[k, col] = desired[j]` assignment (no effect).

`injector.py` of the working copy was not available, so its part of the patch comes from the notebook cells only.
**To make the patch authoritative**, run once where the modified clone lives (read-only on it):

```bash
uv run dcn extract-patch /content/drive/MyDrive/DCN_Thesis/code/dcngen
```

This regenerates the patch from the real files and updates the lock checksum. Changing the patch changes the
spec hash of every dataset pair, so existing pairs show as `stale` (kept, not overwritten) until you decide.

## Known behaviour of the patched simulator

With the patched `unmet` definition, healthy runs report `unmet=True` on many steps (31 of 48 in the notebook's
Experiment 01 healthy run, ~0.71 healthy rate in the pilot report) and several upstream dcngen tests that assert
"no unmet load in normal operation" fail. The generator reproduces this as-is; whether the
`1e-6` tolerance in `q_now < q_step * (1 - 1e-6)` should relate to the solver's `flow_tol_rel` (1e-3) is an open modelling question.
