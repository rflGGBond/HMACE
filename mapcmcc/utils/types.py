from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

# --- Community Agent Types ---

@dataclass
class CommunityObservation:
    community_id: int
    current_generation: int
    global_stage: str
    budget: int
    current_dpadv: float
    dpadv_history: List[float]
    diversity_score: float
    top_k_score_nodes: List[Dict[str, Any]]
    current_seed_set: List[int]
    boundary_info: Dict[str, Any]
    parameters: Dict[str, float]
    global_dpadv: float
    stagnation_count: int = 0
    solution_history: List[Dict[str, Any]] = field(default_factory=list) # [{'seed': [...], 'score': float}, ...]

@dataclass
class CommunityAction:
    # Mode A: Parameter Adjustment
    parameters: Optional[Dict[str, float]] = None # {'cr1': 0.x, 'cr2': 0.x, 'beta': x, 'alpha': x}
    
    # Mode B: Candidate Generation
    candidate_seed_set: Optional[List[int]] = None
    
# --- Meta Agent Types ---

@dataclass
class CommunitySummary:
    community_id: int
    budget: int
    best_dpadv: float
    improvement_rate: float
    diversity: float
    boundary_risk: float
    closeness_info: Dict[int, float] # neighbor_id -> closeness
    danger_score: float = 0.0 # Gamma * Stagnation

@dataclass
class MetaObservation:
    current_generation: int
    current_global_dpadv: float
    global_dpadv_history: List[float]
    community_summaries: List[CommunitySummary]
    merge_history: List[Any]
    parameter_history: List[Dict[str, Any]] = field(default_factory=list) # [{'params': {...}, 'global_score': float}]

@dataclass
class MetaAction:
    # 1. Global Baselines
    global_baselines: Optional[Dict[str, float]] = None
    
    # 2. Budget Redistribution
    budget_adjustments: Optional[Dict[int, int]] = None # community_id -> new_budget
    
    # 3. Merge Suggestions
    merge_suggestions: Optional[List[Tuple[int, int]]] = None # List of (id1, id2) to suggest merging
