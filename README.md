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

Current default hardware values are synthetic. The simulator is currently useful for mechanism experiments and controlled comparisons; calibrated hardware profiles will replace synthetic timings over time.

## AI development disclosure

All source code currently in this repository was generated and modified by **GPT-5.6 Sol** using **High reasoning effort**. The human project owner defines the problems, architecture direction, constraints, experiments, and evaluation, and has not manually written or edited the source code.

## License

MIT
