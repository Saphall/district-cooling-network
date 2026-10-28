# Archive (not used by the pipeline)

Kept for traceability only; nothing in `src/` imports it.

| Folder | Content |
|---|---|
| `colab/` | The original Colab notebook; Experiments 01-05 (and the superseded `04_clogged_filter.py`) recovered by replaying its file-writing cells, Colab paths unchanged; and `06_generate_pilot_dataset.py` as found (1 seed, 5 pairs). Maintained versions: `src/swim_project/validation/` and `swim generate` with `configs/datasets/pilot_v1.yaml` (agrees with 06 to ~1e-12 relative on a synthetic network). |
| `cloud/` | Cloud-worker prototypes (`cloud_worker`, `scenario_generator`, `scenario_inputs`, `fault_runner`, `network_exporter`, `two_worker_pilot`): same pair physics and network exporter, now in `src/swim_project/pipeline/`. `scenario_worker.py` belongs to the earlier `rq1_experiments` generation and imports modules that were not supplied (`parallel_launcher`, `cloud_config`, `controlled_runner`, `diagnostics.writer_with_desired_flow`); it is not ported. |
| `dcngen/` | Loose copies of dcngen files. `scenario.py` is the as-run modified version (basis of `patches/`); the two `scenario_before_*` files are earlier states. The other files are byte-identical to upstream commit `2641335` and can be deleted. |
