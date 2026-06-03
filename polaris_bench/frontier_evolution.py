"""
POLARIS v4 -- Frontier Module 4: Evolutionary Population Play
==============================================================
MAP-Elites + Genetic Governor: Evolve populations of governance
councils through selection, crossover, and mutation. Discovers
emergent stability structures that humans haven't designed.
"""
import sys, io, os, json, time, random, copy, statistics
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

from server.policy_environment import PolicyEnvironment
from server.config import CORE_ACTIONS, STATE_BOUNDS
from server.tasks import grade_trajectory


N_STATE = len(STATE_BOUNDS)
N_ACTIONS = len(CORE_ACTIONS)


class GovernanceGenome(nn.Module):
    """
    A single governance policy (genome).
    Small MLP that maps state -> action probabilities.
    Can be mutated, crossed over, and evaluated.
    """
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_STATE, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, N_ACTIONS),
        )
        self.fitness = 0.0
        self.age = 0
        self.lineage = []  # track ancestry
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    def act(self, state_vec: torch.Tensor) -> int:
        with torch.no_grad():
            logits = self.forward(state_vec.unsqueeze(0))
            return torch.distributions.Categorical(logits=logits).sample().item()
    
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def mutate(self, sigma: float = 0.02) -> 'GovernanceGenome':
        """Gaussian mutation of all weights."""
        child = GovernanceGenome()
        child.load_state_dict(copy.deepcopy(self.state_dict()))
        with torch.no_grad():
            for p in child.parameters():
                p.add_(torch.randn_like(p) * sigma)
        child.lineage = self.lineage + ["mutate"]
        return child
    
    @staticmethod
    def crossover(parent1: 'GovernanceGenome', 
                  parent2: 'GovernanceGenome') -> 'GovernanceGenome':
        """Uniform crossover: randomly pick weights from each parent."""
        child = GovernanceGenome()
        sd1 = parent1.state_dict()
        sd2 = parent2.state_dict()
        child_sd = {}
        for key in sd1:
            mask = torch.rand_like(sd1[key]) > 0.5
            child_sd[key] = torch.where(mask, sd1[key], sd2[key])
        child.load_state_dict(child_sd)
        child.lineage = parent1.lineage[-2:] + ["cross"] + parent2.lineage[-2:]
        return child


def state_to_vec(metadata: Dict) -> torch.Tensor:
    vec = []
    for key in STATE_BOUNDS:
        val = metadata.get(key, 0.0)
        lo, hi = STATE_BOUNDS[key]
        norm = (val - lo) / (hi - lo) if hi > lo else 0.0
        vec.append(max(0.0, min(1.0, norm)))
    return torch.tensor(vec, dtype=torch.float32)


def evaluate_genome(genome: GovernanceGenome, task_id: str, 
                    seed: int, max_steps: int = 50) -> Dict:
    """Evaluate a single genome on the environment."""
    env = PolicyEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)
    
    actions = []
    total_reward = 0.0
    step = 0
    
    while not obs.done and step < max_steps:
        step += 1
        state_vec = state_to_vec(obs.metadata)
        action_idx = genome.act(state_vec)
        action_name = CORE_ACTIONS[action_idx]
        
        action_data = {"action": action_name, "coalition_target": [],
                       "veto_prediction": [], "stance": "cooperative"}
        obs = env.step(action_data)
        total_reward += obs.reward
        actions.append(action_name)
    
    score = grade_trajectory(task_id, env.get_trajectory())
    collapsed = obs.metadata.get("collapsed", step < max_steps)
    
    return {
        "score": score,
        "reward": total_reward,
        "collapsed": collapsed,
        "steps": step,
        "unique_actions": len(set(actions)),
        "diversity": len(set(actions)) / max(len(actions), 1),
    }


class MAPElites:
    """
    MAP-Elites Quality-Diversity algorithm for governance.
    
    Archive: 2D grid indexed by (behavior_1, behavior_2)
      - behavior_1: action diversity (0-1)
      - behavior_2: survival ratio (0-1)
    
    Each cell stores the BEST genome for that behavioral niche.
    This finds diverse, high-quality governance strategies.
    """
    
    def __init__(self, grid_size: int = 10, pop_size: int = 50):
        self.grid_size = grid_size
        self.pop_size = pop_size
        self.archive: Dict[Tuple[int, int], GovernanceGenome] = {}
        self.archive_fitness: Dict[Tuple[int, int], float] = {}
        self.generation = 0
        self.history = []
    
    def _behavior_to_cell(self, diversity: float, survival: float) -> Tuple[int, int]:
        x = min(self.grid_size - 1, int(diversity * self.grid_size))
        y = min(self.grid_size - 1, int(survival * self.grid_size))
        return (x, y)
    
    def _evaluate(self, genome: GovernanceGenome, 
                  seeds: List[int] = [42, 123, 777]) -> Dict:
        results = []
        for seed in seeds:
            for task in ["environmental_recovery", "negotiation_arena"]:
                r = evaluate_genome(genome, task, seed)
                results.append(r)
        
        avg_score = statistics.mean(r["score"] for r in results)
        collapse_rate = sum(1 for r in results if r["collapsed"]) / len(results)
        avg_diversity = statistics.mean(r["diversity"] for r in results)
        survival = 1.0 - collapse_rate
        
        genome.fitness = avg_score
        return {
            "score": avg_score,
            "collapse_rate": collapse_rate,
            "diversity": avg_diversity,
            "survival": survival,
        }
    
    def initialize(self, n: int = 20):
        """Seed archive with random genomes."""
        print(f"  Initializing {n} random genomes...")
        for i in range(n):
            genome = GovernanceGenome()
            result = self._evaluate(genome)
            cell = self._behavior_to_cell(result["diversity"], result["survival"])
            
            if cell not in self.archive or result["score"] > self.archive_fitness[cell]:
                self.archive[cell] = genome
                self.archive_fitness[cell] = result["score"]
        
        print(f"  Archive: {len(self.archive)} cells filled out of {self.grid_size**2}")
    
    def evolve(self, generations: int = 30):
        """Run MAP-Elites evolution."""
        for gen in range(generations):
            self.generation += 1
            
            if not self.archive:
                self.initialize()
                continue
            
            parents = list(self.archive.values())
            new_genomes = []
            
            # Generate offspring
            for _ in range(self.pop_size):
                r = random.random()
                if r < 0.4:
                    # Mutation
                    parent = random.choice(parents)
                    child = parent.mutate(sigma=0.03)
                elif r < 0.7:
                    # Crossover + mutation
                    p1, p2 = random.sample(parents, min(2, len(parents)))
                    child = GovernanceGenome.crossover(p1, p2)
                    child = child.mutate(sigma=0.01)
                else:
                    # Fresh random
                    child = GovernanceGenome()
                
                new_genomes.append(child)
            
            # Evaluate and insert
            improvements = 0
            for genome in new_genomes:
                result = self._evaluate(genome)
                cell = self._behavior_to_cell(result["diversity"], result["survival"])
                
                if cell not in self.archive or result["score"] > self.archive_fitness[cell]:
                    self.archive[cell] = genome
                    self.archive_fitness[cell] = result["score"]
                    improvements += 1
            
            # Stats
            scores = list(self.archive_fitness.values())
            best = max(scores) if scores else 0
            mean = statistics.mean(scores) if scores else 0
            
            self.history.append({
                "generation": self.generation,
                "archive_size": len(self.archive),
                "best_fitness": round(best, 4),
                "mean_fitness": round(mean, 4),
                "improvements": improvements,
            })
            
            if gen % 5 == 0:
                print(f"  Gen {self.generation}: archive={len(self.archive)} "
                      f"best={best:.4f} mean={mean:.4f} improved={improvements}")
    
    def get_champion(self) -> Tuple[GovernanceGenome, float]:
        """Return the best genome in the archive."""
        if not self.archive:
            return GovernanceGenome(), 0.0
        best_cell = max(self.archive_fitness, key=self.archive_fitness.get)
        return self.archive[best_cell], self.archive_fitness[best_cell]
    
    def archive_heatmap(self) -> List[List[float]]:
        """Return fitness heatmap of the archive."""
        grid = [[0.0] * self.grid_size for _ in range(self.grid_size)]
        for (x, y), fitness in self.archive_fitness.items():
            grid[y][x] = fitness
        return grid
    
    def save(self, path: str):
        data = {
            "generation": self.generation,
            "archive_size": len(self.archive),
            "history": self.history,
            "heatmap": self.archive_heatmap(),
        }
        if self.archive:
            champion, best = self.get_champion()
            data["champion_fitness"] = best
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


def validate_frontier4():
    print("="*60)
    print("  FRONTIER MODULE 4 VALIDATION")
    print("="*60)
    
    # Single genome
    g = GovernanceGenome()
    print(f"  [Genome] params={g.param_count()} ")
    
    # Evaluate
    r = evaluate_genome(g, "environmental_recovery", seed=42)
    print(f"  [Eval] score={r['score']:.4f} collapsed={r['collapsed']} "
          f"diversity={r['diversity']:.2f}")
    
    # Mutation
    child = g.mutate(sigma=0.05)
    r2 = evaluate_genome(child, "environmental_recovery", seed=42)
    print(f"  [Mutant] score={r2['score']:.4f} collapsed={r2['collapsed']}")
    
    # Crossover
    p2 = GovernanceGenome()
    cross = GovernanceGenome.crossover(g, p2)
    r3 = evaluate_genome(cross, "environmental_recovery", seed=42)
    print(f"  [Crossover] score={r3['score']:.4f}")
    
    # Quick MAP-Elites (small)
    print("\n  Running mini MAP-Elites (5 gen, 10 pop)...")
    me = MAPElites(grid_size=5, pop_size=10)
    me.initialize(n=5)
    me.evolve(generations=5)
    
    champ, best_fit = me.get_champion()
    print(f"\n  Champion fitness: {best_fit:.4f}")
    print(f"  Archive coverage: {len(me.archive)}/{me.grid_size**2}")
    
    print("  FRONTIER 4 VALIDATED OK")
    print("="*60)


if __name__ == "__main__":
    validate_frontier4()
