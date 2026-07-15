"""Experiment C — multi-objective Bayesian optimization and Separation of Concerns.

Two scientific questions, one shared multi-objective pipeline:

- ``fixed_scalarization``: at a fixed preference lambda, is it better to model
  each objective separately and scalarize the *acquisition* (RIGHT), or to
  transform the observations and model the scalarized function directly
  (WRONG)?
- ``pareto_front``: how much is gained by sharing a common posterior over the
  objective map ``F`` across several preferences (RIGHT Algorithm 1) versus
  solving several independent scalarized problems (Algorithm 2)?

See ``README.md`` for the full scientific protocol and run commands.
"""
