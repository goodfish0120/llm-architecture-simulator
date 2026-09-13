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

Current default hardware values are synthetic. The simulator is currently useful for mechanism experiments and controlled comparisons; calibrated hardware profiles will replace synthetic timings over time.

## AI development disclosure

All source code currently in this repository was generated and modified by **GPT-5.6 Sol** using **High reasoning effort**. The human project owner defines the problems, architecture direction, constraints, experiments, and evaluation, and has not manually written or edited the source code.

## License

MIT
