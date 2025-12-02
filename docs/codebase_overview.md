# Codebase Overview

This repository implements the Artificial Cambrian Intelligence (ACI) framework for simulating co-evolution of visual systems and neural controllers in embodied agents. The core package is `cambrian`, which exposes configurable training and evaluation loops built on Hydra, MuJoCo, Gymnasium, and Stable Baselines3.

## High-level purpose
- The project explores how task demands shape visual evolution by co-evolving eye morphologies and neural processing in simulated environments, providing a toolkit for studying biological vision and designing task-specific artificial systems.【F:README.md†L1-L74】

## Primary entrypoint
- `cambrian/main.py` defines the CLI used for both training and evaluation. It builds an argument parser with mutually exclusive `--train` and `--eval` flags, then delegates to Hydra to instantiate a `MjCambrianTrainer` configured through `cambrian/configs` and returns a fitness score for sweeps.【F:cambrian/main.py†L1-L35】

## Configuration core
- `cambrian/config.py` declares the `MjCambrianConfig` container that ties together experiment paths, seeds, trainer settings, evolutionary knobs, and environment definitions. It also registers Hydra resolvers to locate the installed package, derive clean override strings, count CPUs, and surface unresolved config fragments for reuse.【F:cambrian/config.py†L14-L76】【F:cambrian/config.py†L79-L121】

## Environment layer
- `cambrian/envs/env.py` implements `MjCambrianEnv`, a PettingZoo-compatible Gymnasium environment tailored for MuJoCo. The accompanying `MjCambrianEnvConfig` describes XML scene composition, step/termination/reward hooks, renderer options, and agent configs for multi-agent simulations.【F:cambrian/envs/env.py†L42-L110】
- Environment initialization composes agent XML fragments, compiles them into a MuJoCo specification, configures a custom renderer (including overlays), and maintains per-episode bookkeeping such as step counters and timing windows.【F:cambrian/envs/env.py†L113-L175】

## Agent abstraction
- `cambrian/agents/agent.py` defines `MjCambrianAgentConfig` for individual bodies, covering trainability, overlay styling, XML templates, initial poses, and embedded eyes.【F:cambrian/agents/agent.py†L27-L106】
- `MjCambrianAgent` loads its XML, parses geometry/actuator layout to derive observation and action spaces, attaches configurable eyes, and caches initialization details (positions, quaternions, overlays) for use inside environments.【F:cambrian/agents/agent.py†L108-L185】

## Training and evaluation pipeline
- `cambrian/ml/trainer.py` provides `MjCambrianTrainerConfig` to specify timesteps, parallel environment counts, model factory, callbacks, wrappers, pruning logic, and fitness calculation.【F:cambrian/ml/trainer.py†L25-L61】
- `MjCambrianTrainer` orchestrates runs: it prepares experiment directories, builds wrapped environments for training/evaluation, compiles and saves the environment XML, executes learning with Stable Baselines3, persists policies/metrics, and computes fitness for both training and eval flows.【F:cambrian/ml/trainer.py†L63-L172】
