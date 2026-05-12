# HMACE

**HMACE** is a hybrid evolutionary framework that integrates **Cooperative Coevolution** with **Multi-Agent Systems (MAS)** and **Large Language Models (LLMs)**. It is designed to solve complex influence blocking maximization problems in large-scale social networks.

By wrapping the traditional PCMCC algorithm in an agent-based environment, HMACE allows intelligent agents to dynamically adjust evolutionary parameters, propose candidate solutions, and guide community merging based on global insights.

---

## 🏗️ System Architecture

The project follows a strict 4-layer modular architecture to decouple mathematical logic from state management and decision-making.

```text
HMACE/
├── core/               # Layer 1: Mathematical Engine (Pure Logic)
│   ├── evaluator.py    # DPADV Calculation, Negative/Positive Scoring
│   ├── evolution.py    # Genetic Operations (Crossover, Mutation, Local Search)
│   ├── graph_ops.py    # Graph Partitioning (Leiden Algorithm)
│   └── merger.py       # Community Merging Logic
│
├── environment/        # Layer 2: State Management (State Container)
│   ├── env.py          # Global Environment (PCMCCEnvironment)
│   └── community.py    # Local Community State & Observation Generator
│
├── agents/             # Layer 3: Decision Making (Brain)
│   ├── base.py         # Abstract Agent Interface
│   ├── local_agent.py   # Local Optimizer (LLM-based)
│   └── global_agent.py  # Global Controller (LLM-based)
│
├── utils/              # Layer 4: Utilities & Protocol
│   ├── types.py        # Data Protocols (Observation/Action Dataclasses)
│   └── llm_client.py   # LLM API Client (Mock/OpenAI)
│
└── run.py              # Main Entry Point
```

---

## 🧩 Module Details

### 1. Core Layer (`core/`)
This layer contains stateless functions that perform the heavy lifting.
- **`evaluator.py`**: Implements the **DPADV (Dynamic Propagation-Activation-Degree Value)** metric.
    - `calculate_negative_probability`: Computes negative influence propagation.
    - `calculate_fitness`: Evaluates a seed set's blocking capability.
- **`evolution.py`**: The evolutionary engine.
    - `evolve_community`: Orchestrates Crossover, Mutation, and **Local Search (Delta-Score based)**.
    - Implements **Subpopulation Communication** (Ring Topology) for parallel islands.
- **`merger.py`**: Handles the physical merging of communities (graph nodes, populations, history).

### 2. Environment Layer (`environment/`)
- **`PCMCCEnvironment`**: The "God Object" that holds the Graph and Global State.
    - `step()`: Advances the evolution by one generation.
    - `apply_community_action()`: **Try-Evaluate-Revert** logic. When an agent proposes a new seed set, the environment tentatively applies it, calculates the global DPADV, and reverts if performance degrades.
- **`Community`**: Represents a single community.
    - Automatically calculates **Diversity Score** (Jaccard Distance) and **Boundary Risk**.
    - Generates `CommunityObservation` JSON for agents.

### 3. Agent Layer (`agents/`)
- **`LocalAgent`**: Assigned to each community.
    - **Mode A (Parameter Tuning)**: Dynamically adjusts `cr1`, `cr2`, `beta`, `alpha`.
    - **Mode B (Candidate Generation)**: Proposes specific seed sets to jump out of local optima.
- **`GlobalAgent`**: Single global controller.
    - Monitors global convergence.
    - Suggests **Global Baselines** and **Community Merges**.

### 4. Utils Layer (`utils/`)
- **`types.py`**: Defines the strict "Communication Protocol" between Env and Agents using Python Dataclasses (`CommunityObservation`, `GlobalAction`, etc.).
- **`llm_client.py`**: A unified interface for LLM calls. Supports a `mock` mode for testing and an `openai` mode for production.

---

## 🚀 Installation & Usage

### Prerequisites
- Python 3.8+
- Required Libraries:
  ```bash
  pip install networkx python-igraph leidenalg numpy
  ```

### Running the Algorithm
You can start the system using the provided shell script or Python directly.

**Using Shell Script (Recommended):**
```bash
# Sets PYTHONPATH automatically
bash run.sh
```

**Using Python:**
```bash
# Make sure you are in the parent directory (e:\PCMCC\PCMCC\COICM)
python HMACE/run.py --graphs facebook --total_budget 50 --num_communities 4
```

**Arguments:**
- `--graph_name`: Name of the graph file (e.g., `facebook`, `BA3000`).
- `--total_budget`: Total number of seeds to select ($k$).
- `--num_communities`: Initial number of communities.
- `--max_gen`: Maximum generations.
- `--t_comm`: Interval (generations) for Agent interaction.

---

## 🤖 LLM Integration Guide

The system is pre-configured with a **Mock LLM** to ensure it runs out-of-the-box without API keys. To enable real LLM intelligence:

1.  **Open** `HMACE/utils/llm_client.py`.
2.  **Locate** the `LLMClient` class.
3.  **Change** the default provider in `__init__`:
    ```python
    def __init__(self, provider: str = "openai", ...): # Change "mock" to "openai"
    ```
4.  **Set Environment Variable**:
    ```bash
    export OPENAI_API_KEY="<your-api-key>"
    # Optional for OpenRouter or other OpenAI-compatible providers:
    export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
    ```
5.  **Customize Prompts**:
    - Modify `HMACE/agents/local_agent.py` to change how the Local Agent perceives its state.
    - Modify `HMACE/agents/global_agent.py` to adjust the Global Agent's global strategy.

---

## 🔄 Algorithm Workflow

1.  **Initialization**:
    - Graph is loaded and partitioned into $m$ communities.
    - Subpopulations are initialized.
2.  **Main Loop** (Generations $1$ to $MaxGen$):
    - **Evolution Step**: All communities evolve in parallel (Crossover + Local Search).
    - **Agent Check** (Every $T_{comm}$ generations):
        - **Observe**: Agents receive JSON observations (History, Diversity, Top-K Nodes).
        - **Decide**: Agents call LLM to get Actions (Tune Params or Propose Seeds).
        - **Apply**: Environment executes actions (with validation).
    - **Merge Check**: Global Agent evaluates if communities should merge.
3.  **Termination**:
    - Returns the best global seed set $S^*$ found.

---

## 🛠️ Key Features for Researchers

- **Modular DPADV**: The core evaluation logic is isolated in `core/evaluator.py`, making it easy to swap with other influence metrics (e.g., IC/LT models).
- **Strict Evaluation**: The `apply_community_action` method ensures that LLM "hallucinations" (bad seeds) are never accepted, preserving the integrity of the evolutionary process.
- **Traceability**: All agent decisions can be logged to analyze *why* a parameter was changed or a seed was selected.
