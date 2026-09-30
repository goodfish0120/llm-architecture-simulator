# LLM Architecture Simulator

A stochastic architecture sandbox for LLM and Mixture-of-Experts systems.

Use it to ask architecture questions such as:

- Where does multi-node throughput scaling begin to bend?
- How many replicas of hot experts are useful?
- How much bounded layer asynchrony helps before batching fragments?
- When is another full model replica better than further distribution?
- What happens on measured, unusual, or hypothetical hardware and networks?

```bash
python run_simulator.py
```

Run the published full-size model sweep for one through seven 512 GB M5 Ultra
Mac Studios connected as an RDMA-over-Thunderbolt-5 full mesh:

```bash
python run_mac_studio_m5_ultra_scaling.py
```

The sweep independently increases active agents for every model/node combination,
records aggregate throughput and mean/p95 single-token latency, and writes one SVG
pressure curve per combination under `results/pressure_curves/`.

The sweep writes a CSV plus a JSON assumptions manifest under `results/`. It
uses official checkpoint sizes and published architecture fields, but its M5
kernel throughput remains an explicit assumption rather than a benchmark.

Screen cold-expert requests for two-to-four-round deferred batching before
adding a scheduling policy to the event simulator:

```bash
python optimize_deferred_expert_batching.py
```

The optimizer reports the memory-bound throughput ceiling against accumulated
layer-round waiting and marks the non-dominated candidates in
`results/deferred_expert_batching.csv`.

Model fork-heavy workloads with a mostly shared 20K system prefix, a partially
shared 20K-60K family context, and an independent suffix:

```bash
python run_prefix_fork_scenarios.py
python plot_prefix_fork_scenarios.py
```

This is an optimistic prefix-aware batching model: storage uses copy-on-write,
and co-scheduled queries reuse identical KV-prefix reads within their fork group.

Current default hardware values are synthetic. The simulator is currently useful for mechanism experiments and controlled comparisons; calibrated hardware profiles will replace synthetic timings over time.

## Diagnostic replay viewer

Generate a small non-collapsed, synthetic heterogeneous run (dense-only layers 0/2 and MoE layers 1/3) from the real event simulation:

```bash
python run_diagnostic_demo.py
```

Open `diagnostic_viewer/index.html` in a browser and load the output JSON. The
default destination is `results/diagnostic_demo.json`. The viewer replays the exported event trace: use
seek/play/step, expand hierarchy rows, drag/resize panels, duplicate/pin a panel
while comparing it side-by-side with another. Layout changes are in-memory for the
open page only (they are not persisted) and never modify simulation topology or
configuration; duplicated panels are separate
views over the same records, not a second simulated workload.

For a local HTTP launch with the demo preloaded (useful when a browser blocks
file-to-file loading), run `python -m http.server 8000` from the repository and
open `http://localhost:8000/diagnostic_viewer/?trace=../results/diagnostic_demo.json`.

In the restricted validation environment used for this change, Python cannot create
new files beneath this worktree's `results/` directory. The equivalent supported
command is:

```powershell
python run_diagnostic_demo.py --output C:\Users\User\AppData\Local\Temp\diagnostic_demo.json
```

Then open `diagnostic_viewer/index.html` and choose that Temp JSON with the file
picker. This is a sandbox limitation, not a claim that a demo JSON is already in
`results/`; the script read-backs its generated output in normal writable setups.

The trace is optional (`DiagnosticTrace` passed to `StochasticMoeArchitectureSimulator`),
append-only, and does not schedule events or consume routing RNG. It records stable
event ordering, token/agent/layer identities, routing branches and joins, batch
membership/queue depth, actual modeled transfer reservations, and resource intervals.
It is a replay of a completed simulator run—not a browser approximation.

Interpretation limits: resource busy/productive labels are model-defined reservation
intervals, not GPU utilization or MFU; modeled compute productive time includes
startup and batch efficiency. KV is modeled as costs/capacity, not physical page
movement. Each layer has a shared stage, so that must not be confused with a
dense-only layer. Observed temporal correlation is evidence for investigation, not
proof of a bottleneck cause.

## AI development disclosure

All source code currently in this repository was generated and modified by **GPT-5.6 Sol** using **High reasoning effort**. The human project owner defines the problems, architecture direction, constraints, experiments, and evaluation, and has not manually written or edited the source code.

## License

MIT
