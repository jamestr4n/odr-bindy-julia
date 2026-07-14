# odr-bindy-julia

Julia implementation of ODR-BINDy — a personal learning project where I'm working through implementing ODR-BINDy (Orthogonal Distance Regression with Bayesian Inference for Nonlinear Dynamics) in Julia, as part of my ongoing coursework/research under my supervisor.

## Purpose

This repository documents my process of learning Julia while re-implementing the ODR-BINDy method. It is primarily a learning log rather than a finished software package, so code and structure will evolve as I progress.

## Status

🚧 Work in progress — currently working through: (fill in current topic, e.g. "basic ODR fitting examples in Julia")

See [NOTES.md](NOTES.md) for a running log of what I've covered so far and what I'm learning in each session.

## Prerequisites

- Julia (version X.X or later — check with `julia --version`)
- This project uses Julia's built-in package manager (Pkg) for dependency management

## Setup

Clone the repository and activate the project environment:

```bash
git clone https://github.com/jamestr4n/odr-bindy-julia.git
cd odr-bindy-julia
julia
```

Then inside the Julia REPL:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

This installs all dependencies listed in `Project.toml`.

## Repository structure

```
odr-bindy-julia/
├── README.md          # This file
├── NOTES.md           # Running log of learning progress
├── Project.toml       # Julia project dependencies
├── .gitignore
├── src/               # Source code / core implementation
├── learning/          # Exercises and worked examples while learning Julia
└── test/              # Unit tests (if/when added)
```

## Background / references

- ODR-BINDy paper or resource: (add link here)
- Julia documentation: https://docs.julialang.org/

## Author

James (@jamestr4n) — Learning project, supervised progress check-ins.
