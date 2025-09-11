# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research codebase for Federated Constrained Soft Actor-Critic (Fed-CSAC) reinforcement learning applied to integrated energy systems (IES) and microgrid management (MMG). The project implements multi-agent federated learning for energy optimization with carbon emission constraints.

## Key Architecture Components

### Core Modules
- **`env/`** - Energy environment simulation
  - `env_ies.py` - Main `CombinedEnergyEnv` gym environment for integrated energy systems
  - `config.py` - Configuration management with shared parameters and scenario-specific settings
  - `trade.py` - Energy market clearing and pricing functions (`da_market_clearing`, `get_market_prices_car`)
  - `carbon.py` - Carbon quota calculations and emissions tracking

- **`network/` & `network2/`** - Neural network architectures
  - Contains SAC (Soft Actor-Critic) network implementations
  - `network2/` contains updated versions with shared Q-networks for federated learning
  - `Prioritized_Replay.py` - Experience replay buffer with prioritization

### Main Training Scripts
- **`Fed_CSAC_shared_q.py`** - Main federated learning script with shared Q-networks
- **`Fed_train.py`** - Alternative federated training implementation with logging/export
- **`single_ies_shared_q.py`** - Contains `C_SAC_` class (Constrained SAC agent implementation)

### Key Classes
- `CombinedEnergyEnv` (env/env_ies.py) - Main environment for energy system simulation
- `C_SAC_` (single_ies_shared_q.py) - Constrained Soft Actor-Critic agent
- `Config` (env/config.py) - Configuration management with `get_shared()` and `get_scenario()` methods

## Running the Code

### Prerequisites
- Python 3.11+ (tested with 3.11.7)
- TensorFlow/Keras for neural networks
- Standard ML libraries: numpy, pandas, matplotlib, gym

### Main Execution
Run federated training:
```bash
python Fed_CSAC_shared_q.py
```

Run single-agent training:
```bash
python single_ies_shared_q.py
```

### Architecture Notes
- The code uses a federated learning approach where agents share shallow network layers (Actor_Shared_*, Critic_Shared_*) but maintain personalized deeper layers
- Carbon emission constraints are integrated into the reward structure
- Environment supports multiple IES scenarios configured through `Config` class
- Uses prioritized experience replay for improved sample efficiency

### Data
- **`data/`** - Contains input data files for energy loads, prices, and system parameters  
- **`training_trace.xlsx`** - Training progress and results tracking
- **`image_result/`** - Generated plots and visualization outputs
- **`存档/`** - Archive directory with older implementations and test files