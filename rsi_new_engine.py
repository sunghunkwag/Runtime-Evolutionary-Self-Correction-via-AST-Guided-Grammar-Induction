from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------------------------------------
# Persistent state directory
# -----------------------------------------------------------

STATE_DIR = Path(".rsi_new")
RUNS_DIR = STATE_DIR / "runs"


def ensure_state_dirs() -> None:
    STATE_DIR.mkdir(exist_ok=True)
    RUNS_DIR.mkdir(exist_ok=True)


# -----------------------------------------------------------
# Expression node (used for tasks and tree-based programs)
# -----------------------------------------------------------


@dataclass
class ExprNode:
    """Simple expression tree for tasks and tree-based programs."""

    kind: str  # 'var', 'const', 'op'
    name: str  # variable name / op name / 'const'
    children: List["ExprNode"] = field(default_factory=list)
    value: float = 0.0  # for constants


def eval_expr(node: ExprNode, inputs: List[float]) -> float:
    """Evaluate an ExprNode on a given input vector."""
    if node.kind == "var":
        try:
            idx = int(node.name[1:])
            return inputs[idx]
        except Exception:
            return 0.0
    if node.kind == "const":
        return node.value
    if node.kind == "op":
        op = node.name
        if op in ("add", "sub", "mul"):
            a = eval_expr(node.children[0], inputs)
            b = eval_expr(node.children[1], inputs)
            if op == "add":
                return a + b
            if op == "sub":
                return a - b
            if op == "mul":
                return a * b
        elif op == "safe_div":
            a = eval_expr(node.children[0], inputs)
            b = eval_expr(node.children[1], inputs)
            if abs(b) < 1e-6:
                return a
            return a / b
        elif op == "tanh":
            a = eval_expr(node.children[0], inputs)
            return math.tanh(a)
        elif op == "relu":
            a = eval_expr(node.children[0], inputs)
            return a if a > 0.0 else 0.0
        elif op == "sigmoid":
            a = eval_expr(node.children[0], inputs)
            return 1.0 / (1.0 + math.exp(-a))
        elif op == "sin":
            a = eval_expr(node.children[0], inputs)
            return math.sin(a)
        elif op == "cos":
            a = eval_expr(node.children[0], inputs)
            return math.cos(a)
    return 0.0


def clone_expr(node: ExprNode) -> ExprNode:
    return ExprNode(
        kind=node.kind,
        name=node.name,
        children=[clone_expr(c) for c in node.children],
        value=node.value,
    )


def random_expr(
    num_inputs: int,
    max_depth: int,
    rng: random.Random,
    nonlin_prob: float,
) -> ExprNode:
    """Generate a random expression for tasks or tree programs."""
    if max_depth <= 0 or rng.random() < 0.2:
        # leaf
        if rng.random() < 0.7:
            idx = rng.randrange(num_inputs)
            return ExprNode(kind="var", name=f"x{idx}")
        return ExprNode(kind="const", name="const", value=rng.uniform(-2.0, 2.0))
    # internal op
    if rng.random() < nonlin_prob:
        op = rng.choice(["tanh", "relu", "sigmoid", "sin", "cos"])
        child = random_expr(num_inputs, max_depth - 1, rng, nonlin_prob)
        return ExprNode(kind="op", name=op, children=[child])
    op = rng.choice(["add", "sub", "mul", "safe_div"])
    left = random_expr(num_inputs, max_depth - 1, rng, nonlin_prob)
    right = random_expr(num_inputs, max_depth - 1, rng, nonlin_prob)
    return ExprNode(kind="op", name=op, children=[left, right])


# -----------------------------------------------------------
# Task / Benchmark generation
# -----------------------------------------------------------


@dataclass
class TaskSpec:
    name: str
    expr: ExprNode
    num_inputs: int
    output_dim: int = 1


@dataclass
class TaskGenConfig:
    num_inputs: int = 2
    output_dim: int = 1
    max_depth: int = 3
    num_tasks: int = 3
    nonlin_prob: float = 0.5


def generate_tasks(cfg: TaskGenConfig, rng: random.Random) -> List[TaskSpec]:
    tasks: List[TaskSpec] = []
    for i in range(cfg.num_tasks):
        expr = random_expr(cfg.num_inputs, cfg.max_depth, rng, cfg.nonlin_prob)
        tasks.append(
            TaskSpec(
                name=f"auto_task_{i}",
                expr=expr,
                num_inputs=cfg.num_inputs,
                output_dim=cfg.output_dim,
            )
        )
    return tasks


# -----------------------------------------------------------
# Representation configs and implementations
# -----------------------------------------------------------


@dataclass
class RepresentationConfig:
    kind: str = "linear_gp"  # 'linear_gp' or 'tree_prog'
    max_size: int = 32  # max instructions or nodes
    num_registers: int = 8
    max_depth: int = 4  # for trees
    complexity_penalty: float = 1e-3


@dataclass
class UpdateRuleConfig:
    name: str = "evo_combo"
    selection: str = "tournament"  # 'tournament', 'roulette', 'rank'
    reproduction: str = "sexual"  # 'sexual', 'asexual'
    mutation_rate: float = 0.4
    crossover_rate: float = 0.6
    elitism: int = 2


@dataclass
class SystemConfig:
    repr_cfg: RepresentationConfig
    upd_cfg: UpdateRuleConfig
    task_cfg: TaskGenConfig
    population_size: int = 64
    generations: int = 40


@dataclass
class MetaMetaConfig:
    """Controls how aggressive the meta-search is."""

    config_step_scale: float = 0.5
    eval_repeats: int = 2
    accept_temp: float = 0.2


# -----------------------------------------------------------
# Linear GP representation
# -----------------------------------------------------------


@dataclass
class Instruction:
    op: str
    dst: int
    src1: int
    src2: int


@dataclass
class LinearProgram:
    instructions: List[Instruction]


class RepresentationBase:
    """Interface for program representations."""

    def __init__(self, cfg: RepresentationConfig, num_inputs: int, output_dim: int):
        self.cfg = cfg
        self.num_inputs = num_inputs
        self.output_dim = output_dim

    def random_program(self, rng: random.Random) -> Any:
        raise NotImplementedError

    def mutate(self, prog: Any, rng: random.Random) -> Any:
        raise NotImplementedError

    def crossover(self, a: Any, b: Any, rng: random.Random) -> Any:
        raise NotImplementedError

    def execute(self, prog: Any, inputs: List[float]) -> List[float]:
        raise NotImplementedError

    def complexity(self, prog: Any) -> float:
        raise NotImplementedError


class LinearGPRepresentation(RepresentationBase):
    """Linear register-based GP representation."""

    def _random_instruction(self, rng: random.Random) -> Instruction:
        ops = ["add", "sub", "mul", "safe_div", "tanh", "relu", "sigmoid"]
        op = rng.choice(ops)
        dst = rng.randrange(self.cfg.num_registers)
        src1 = rng.randrange(self.cfg.num_registers)
        src2 = rng.randrange(self.cfg.num_registers)
        return Instruction(op=op, dst=dst, src1=src1, src2=src2)

    def random_program(self, rng: random.Random) -> LinearProgram:
        n = rng.randint(1, self.cfg.max_size)
        return LinearProgram(
            instructions=[self._random_instruction(rng) for _ in range(n)]
        )

    def mutate(self, prog: LinearProgram, rng: random.Random) -> LinearProgram:
        new_insts = [Instruction(i.op, i.dst, i.src1, i.src2) for i in prog.instructions]
        # deletion
        if new_insts and rng.random() < 0.3:
            idx = rng.randrange(len(new_insts))
            del new_insts[idx]
        # insertion
        if rng.random() < 0.5 and len(new_insts) < self.cfg.max_size:
            idx = rng.randrange(len(new_insts) + 1)
            new_insts.insert(idx, self._random_instruction(rng))
        # modification
        for inst in new_insts:
            if rng.random() < 0.2:
                inst.dst = rng.randrange(self.cfg.num_registers)
            if rng.random() < 0.2:
                inst.src1 = rng.randrange(self.cfg.num_registers)
            if rng.random() < 0.2:
                inst.src2 = rng.randrange(self.cfg.num_registers)
            if rng.random() < 0.1:
                inst.op = self._random_instruction(rng).op
        return LinearProgram(new_insts)

    def crossover(
        self, a: LinearProgram, b: LinearProgram, rng: random.Random
    ) -> LinearProgram:
        if not a.instructions or not b.instructions:
            return LinearProgram(
                [Instruction(i.op, i.dst, i.src1, i.src2) for i in a.instructions]
            )
        cut_a = rng.randrange(len(a.instructions))
        cut_b = rng.randrange(len(b.instructions))
        new_insts = a.instructions[:cut_a] + b.instructions[cut_b:]
        if len(new_insts) > self.cfg.max_size:
            new_insts = new_insts[: self.cfg.max_size]
        return LinearProgram(
            [Instruction(i.op, i.dst, i.src1, i.src2) for i in new_insts]
        )

    def execute(self, prog: LinearProgram, inputs: List[float]) -> List[float]:
        regs = [0.0 for _ in range(self.cfg.num_registers)]
        for i in range(min(len(inputs), self.cfg.num_registers)):
            regs[i] = inputs[i]
        for inst in prog.instructions:
            op = inst.op
            dst = inst.dst % self.cfg.num_registers
            s1 = inst.src1 % self.cfg.num_registers
            s2 = inst.src2 % self.cfg.num_registers
            try:
                if op in ("add", "sub", "mul", "safe_div"):
                    a = regs[s1]
                    b = regs[s2]
                    if op == "add":
                        regs[dst] = a + b
                    elif op == "sub":
                        regs[dst] = a - b
                    elif op == "mul":
                        regs[dst] = a * b
                    elif op == "safe_div":
                        if abs(b) < 1e-6:
                            regs[dst] = a
                        else:
                            regs[dst] = a / b
                else:
                    a = regs[s1]
                    if op == "tanh":
                        regs[dst] = math.tanh(a)
                    elif op == "relu":
                        regs[dst] = a if a > 0.0 else 0.0
                    elif op == "sigmoid":
                        regs[dst] = 1.0 / (1.0 + math.exp(-a))
            except Exception:
                # ignore runtime errors
                pass
        out = []
        for i in range(self.output_dim):
            out.append(regs[i % self.cfg.num_registers])
        return out

    def complexity(self, prog: LinearProgram) -> float:
        return float(len(prog.instructions))


class TreeProgram:
    def __init__(self, root: ExprNode):
        self.root = root


class TreeRepresentation(RepresentationBase):
    """Tree-based program representation using the same ExprNode DSL."""

    def random_program(self, rng: random.Random) -> TreeProgram:
        root = random_expr(self.num_inputs, self.cfg.max_depth, rng, nonlin_prob=0.6)
        return TreeProgram(root)

    def mutate(self, prog: TreeProgram, rng: random.Random) -> TreeProgram:
        root = clone_expr(prog.root)
        # collect nodes
        nodes: List[ExprNode] = []

        def collect(n: ExprNode) -> None:
            nodes.append(n)
            for c in n.children:
                collect(c)

        collect(root)
        if not nodes:
            return TreeProgram(root)
        target = rng.choice(nodes)
        # mutate in place
        if target.kind == "const":
            target.value += rng.uniform(-1.0, 1.0)
        elif target.kind == "var":
            idx = rng.randrange(self.num_inputs)
            target.name = f"x{idx}"
        elif target.kind == "op":
            # change op or replace subtree
            if rng.random() < 0.5:
                # change op type
                if len(target.children) == 1:
                    target.name = rng.choice(["tanh", "relu", "sigmoid", "sin", "cos"])
                else:
                    target.name = rng.choice(["add", "sub", "mul", "safe_div"])
            else:
                # replace subtree
                new_sub = random_expr(
                    self.num_inputs,
                    self.cfg.max_depth // 2 + 1,
                    rng,
                    nonlin_prob=0.6,
                )
                target.kind = new_sub.kind
                target.name = new_sub.name
                target.children = new_sub.children
                target.value = new_sub.value
        return TreeProgram(root)

    def crossover(
        self, a: TreeProgram, b: TreeProgram, rng: random.Random
    ) -> TreeProgram:
        # very simple: swap entire subtrees at root with some probability
        if rng.random() < 0.5:
            return TreeProgram(clone_expr(a.root))
        return TreeProgram(clone_expr(b.root))

    def execute(self, prog: TreeProgram, inputs: List[float]) -> List[float]:
        v = eval_expr(prog.root, inputs)
        return [v for _ in range(self.output_dim)]

    def complexity(self, prog: TreeProgram) -> float:
        count = 0

        def count_nodes(n: ExprNode) -> None:
            nonlocal count
            count += 1
            for c in n.children:
                count_nodes(c)

        count_nodes(prog.root)
        return float(count)


def build_representation(
    cfg: RepresentationConfig, num_inputs: int, output_dim: int
) -> RepresentationBase:
    if cfg.kind == "tree_prog":
        return TreeRepresentation(cfg, num_inputs, output_dim)
    return LinearGPRepresentation(cfg, num_inputs, output_dim)


# -----------------------------------------------------------
# Fitness evaluation on a set of tasks
# -----------------------------------------------------------


@dataclass
class EpisodeSummary:
    best_loss: float
    avg_loss: float
    best_complexity: float
    system_config: Dict[str, Any]


def evaluate_population(
    repr_obj: RepresentationBase,
    population: List[Any],
    tasks: List[TaskSpec],
    cfg: SystemConfig,
    rng: random.Random,
) -> Tuple[List[float], float, float]:
    """Compute fitness (loss) for each program and return losses, best_loss, best_complexity."""
    losses: List[float] = []
    best_loss = float("inf")
    best_complexity = 0.0

    for prog in population:
        total_err = 0.0
        samples = 0
        for task in tasks:
            for _ in range(16):
                x = [rng.uniform(-2.0, 2.0) for _ in range(task.num_inputs)]
                target = eval_expr(task.expr, x)
                out = repr_obj.execute(prog, x)
                pred = out[0] if out else 0.0
                diff = target - pred
                total_err += diff * diff
                samples += 1
        mse = total_err / max(1, samples)
        comp = repr_obj.complexity(prog)
        loss = mse + cfg.repr_cfg.complexity_penalty * comp
        losses.append(loss)
        if loss < best_loss:
            best_loss = loss
            best_complexity = comp
    avg_loss = sum(losses) / max(1, len(losses))
    return losses, best_loss, best_complexity


# -----------------------------------------------------------
# Update rule (search algorithm) based on UpdateRuleConfig
# -----------------------------------------------------------


def _select_index(selection: str, losses: List[float], rng: random.Random) -> int:
    n = len(losses)
    if n == 0:
        return 0
    if selection == "roulette":
        # lower loss => higher weight
        inv = [1.0 / (l + 1e-9) for l in losses]
        total = sum(inv)
        r = rng.uniform(0.0, total)
        acc = 0.0
        for i, w in enumerate(inv):
            acc += w
            if r <= acc:
                return i
        return n - 1
    if selection == "rank":
        # rank-based selection
        indices = list(range(n))
        indices.sort(key=lambda i: losses[i])
        weights = [1.0 / (i + 1) for i in range(n)]
        total = sum(weights)
        r = rng.uniform(0.0, total)
        acc = 0.0
        for idx, w in zip(indices, weights):
            acc += w
            if r <= acc:
                return idx
        return indices[-1]
    # tournament
    k = min(3, n)
    cand = rng.sample(range(n), k=k)
    best_i = cand[0]
    best_l = losses[best_i]
    for i in cand[1:]:
        if losses[i] < best_l:
            best_i = i
            best_l = losses[i]
    return best_i


def update_population(
    repr_obj: RepresentationBase,
    population: List[Any],
    losses: List[float],
    cfg: SystemConfig,
    rng: random.Random,
) -> List[Any]:
    """Single evolutionary update step controlled by UpdateRuleConfig."""
    n = len(population)
    if n == 0:
        return population
    upd = cfg.upd_cfg
    # sort by fitness
    indices = list(range(n))
    indices.sort(key=lambda i: losses[i])
    new_pop: List[Any] = []
    # elitism
    for i in indices[: max(0, upd.elitism)]:
        new_pop.append(population[i])
    # fill rest
    while len(new_pop) < cfg.population_size:
        p1_idx = _select_index(upd.selection, losses, rng)
        parent1 = population[p1_idx]
        if upd.reproduction == "sexual" and rng.random() < upd.crossover_rate:
            p2_idx = _select_index(upd.selection, losses, rng)
            parent2 = population[p2_idx]
            child = repr_obj.crossover(parent1, parent2, rng)
        else:
            child = parent1
        if rng.random() < upd.mutation_rate:
            child = repr_obj.mutate(child, rng)
        new_pop.append(child)
    return new_pop[: cfg.population_size]


# -----------------------------------------------------------
# Level 0: run a single system episode on a generated benchmark
# -----------------------------------------------------------


def run_system_episode(sys_cfg: SystemConfig, seed: int) -> EpisodeSummary:
    rng = random.Random(seed)
    tasks = generate_tasks(sys_cfg.task_cfg, rng)
    repr_obj = build_representation(
        sys_cfg.repr_cfg,
        num_inputs=sys_cfg.task_cfg.num_inputs,
        output_dim=sys_cfg.task_cfg.output_dim,
    )
    # initialize population
    population: List[Any] = [
        repr_obj.random_program(rng) for _ in range(sys_cfg.population_size)
    ]
    best_loss_overall = float("inf")
    best_complexity_overall = 0.0
    losses: List[float] = []

    for _ in range(sys_cfg.generations):
        losses, best_loss_gen, best_comp_gen = evaluate_population(
            repr_obj, population, tasks, sys_cfg, rng
        )
        if best_loss_gen < best_loss_overall:
            best_loss_overall = best_loss_gen
            best_complexity_overall = best_comp_gen
        population = update_population(repr_obj, population, losses, sys_cfg, rng)

    avg_loss = sum(losses) / max(1, len(losses))
    summary = EpisodeSummary(
        best_loss=best_loss_overall,
        avg_loss=avg_loss,
        best_complexity=best_complexity_overall,
        system_config=_system_config_to_dict(sys_cfg),
    )
    return summary


# -----------------------------------------------------------
# Meta-level search over SystemConfig (representation + rule + task generator)
# -----------------------------------------------------------


def _jitter_int(x: int, scale: float, lo: int, hi: int, rng: random.Random) -> int:
    if x <= 0:
        x = 1
    factor = math.exp(rng.uniform(-scale, scale))
    v = int(round(x * factor))
    return max(lo, min(hi, v))


def _jitter_float(
    x: float, scale: float, lo: float, hi: float, rng: random.Random
) -> float:
    factor = math.exp(rng.uniform(-scale, scale))
    v = x * factor
    return max(lo, min(hi, v))


def mutate_system_config(
    cfg: SystemConfig, scale: float, rng: random.Random
) -> SystemConfig:
    """Produce a nearby SystemConfig variant (new representation / rule / task space)."""
    # representation
    repr_cfg = RepresentationConfig(
        kind=cfg.repr_cfg.kind,
        max_size=_jitter_int(cfg.repr_cfg.max_size, scale, 4, 128, rng),
        num_registers=_jitter_int(cfg.repr_cfg.num_registers, scale, 2, 32, rng),
        max_depth=_jitter_int(cfg.repr_cfg.max_depth, scale, 2, 8, rng),
        complexity_penalty=_jitter_float(
            cfg.repr_cfg.complexity_penalty, scale, 1e-5, 1e-1, rng
        ),
    )
    if rng.random() < 0.2:
        repr_cfg.kind = "tree_prog" if cfg.repr_cfg.kind == "linear_gp" else "linear_gp"

    # update rule
    selection_choices = ["tournament", "roulette", "rank"]
    reproduction_choices = ["sexual", "asexual"]
    upd_cfg = UpdateRuleConfig(
        name="evo_combo",
        selection=cfg.upd_cfg.selection,
        reproduction=cfg.upd_cfg.reproduction,
        mutation_rate=_jitter_float(cfg.upd_cfg.mutation_rate, scale, 0.05, 0.9, rng),
        crossover_rate=_jitter_float(
            cfg.upd_cfg.crossover_rate, scale, 0.0, 1.0, rng
        ),
        elitism=_jitter_int(cfg.upd_cfg.elitism, scale, 0, 10, rng),
    )
    if rng.random() < 0.2:
        upd_cfg.selection = rng.choice(selection_choices)
    if rng.random() < 0.2:
        upd_cfg.reproduction = rng.choice(reproduction_choices)

    # task generator
    task_cfg = TaskGenConfig(
        num_inputs=_jitter_int(cfg.task_cfg.num_inputs, scale, 1, 5, rng),
        output_dim=1,
        max_depth=_jitter_int(cfg.task_cfg.max_depth, scale, 2, 6, rng),
        num_tasks=_jitter_int(cfg.task_cfg.num_tasks, scale, 1, 6, rng),
        nonlin_prob=_jitter_float(cfg.task_cfg.nonlin_prob, scale, 0.1, 0.9, rng),
    )

    population_size = _jitter_int(cfg.population_size, scale, 8, 256, rng)
    generations = _jitter_int(cfg.generations, scale, 5, 80, rng)

    return SystemConfig(
        repr_cfg=repr_cfg,
        upd_cfg=upd_cfg,
        task_cfg=task_cfg,
        population_size=population_size,
        generations=generations,
    )


def _system_config_to_dict(cfg: SystemConfig) -> Dict[str, Any]:
    return {
        "repr_cfg": asdict(cfg.repr_cfg),
        "upd_cfg": asdict(cfg.upd_cfg),
        "task_cfg": asdict(cfg.task_cfg),
        "population_size": cfg.population_size,
        "generations": cfg.generations,
    }


def _dict_to_system_config(data: Dict[str, Any]) -> SystemConfig:
    return SystemConfig(
        repr_cfg=RepresentationConfig(**data["repr_cfg"]),
        upd_cfg=UpdateRuleConfig(**data["upd_cfg"]),
        task_cfg=TaskGenConfig(**data["task_cfg"]),
        population_size=int(data.get("population_size", 64)),
        generations=int(data.get("generations", 40)),
    )


def default_system_config() -> SystemConfig:
    return SystemConfig(
        repr_cfg=RepresentationConfig(),
        upd_cfg=UpdateRuleConfig(),
        task_cfg=TaskGenConfig(),
        population_size=64,
        generations=40,
    )


def load_system_config() -> SystemConfig:
    ensure_state_dirs()
    path = STATE_DIR / "system_config.json"
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return _dict_to_system_config(data)
        except Exception:
            pass
    return default_system_config()


def save_system_config(cfg: SystemConfig) -> None:
    ensure_state_dirs()
    path = STATE_DIR / "system_config.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(_system_config_to_dict(cfg), f, indent=2, sort_keys=True)


def evaluate_system_config(cfg: SystemConfig, repeats: int, seed: int) -> float:
    """Evaluate a SystemConfig by running multiple episodes and averaging best_loss."""
    best_losses: List[float] = []
    for i in range(repeats):
        summary = run_system_episode(cfg, seed + i * 97)
        best_losses.append(summary.best_loss)
    return sum(best_losses) / max(1, len(best_losses))


def accept_new_score(
    old_score: float, new_score: float, temp: float, rng: random.Random
) -> bool:
    """Metropolis-style acceptance for meta-search."""
    if new_score < old_score:
        return True
    if temp <= 0.0:
        return False
    diff = new_score - old_score
    denom = abs(old_score) * temp + 1e-9
    prob = math.exp(-diff / denom)
    return rng.random() < prob


def meta_search(
    base_cfg: SystemConfig, mm_cfg: MetaMetaConfig, rounds: int, seed: int
) -> Tuple[SystemConfig, float]:
    """Level 1: search over SystemConfig space (representation + update rule + task generator)."""
    rng = random.Random(seed)
    best_cfg = base_cfg
    best_score = evaluate_system_config(best_cfg, mm_cfg.eval_repeats, seed + 1)

    ensure_state_dirs()
    history_path = STATE_DIR / "meta_history.jsonl"
    with history_path.open("a", encoding="utf-8") as f_hist:
        for r in range(rounds):
            cand_cfg = mutate_system_config(best_cfg, mm_cfg.config_step_scale, rng)
            cand_score = evaluate_system_config(
                cand_cfg, mm_cfg.eval_repeats, seed + 1000 + r
            )
            accepted = accept_new_score(best_score, cand_score, mm_cfg.accept_temp, rng)
            rec = {
                "round": r,
                "base_score": best_score,
                "cand_score": cand_score,
                "accepted": accepted,
                "cand_cfg": _system_config_to_dict(cand_cfg),
            }
            f_hist.write(json.dumps(rec) + "\n")
            if accepted:
                best_cfg = cand_cfg
                best_score = cand_score
                save_system_config(best_cfg)
    return best_cfg, best_score


# -----------------------------------------------------------
# Meta-meta level: re-design the meta-search itself
# -----------------------------------------------------------


def default_meta_meta_config() -> MetaMetaConfig:
    return MetaMetaConfig(config_step_scale=0.5, eval_repeats=2, accept_temp=0.2)


def load_meta_meta_config() -> MetaMetaConfig:
    ensure_state_dirs()
    path = STATE_DIR / "meta_meta_config.json"
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return MetaMetaConfig(
                config_step_scale=float(data.get("config_step_scale", 0.5)),
                eval_repeats=int(data.get("eval_repeats", 2)),
                accept_temp=float(data.get("accept_temp", 0.2)),
            )
        except Exception:
            pass
    return default_meta_meta_config()


def save_meta_meta_config(cfg: MetaMetaConfig) -> None:
    ensure_state_dirs()
    path = STATE_DIR / "meta_meta_config.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, sort_keys=True)


def mutate_meta_meta_config(cfg: MetaMetaConfig, rng: random.Random) -> MetaMetaConfig:
    return MetaMetaConfig(
        config_step_scale=_jitter_float(cfg.config_step_scale, 0.4, 0.1, 2.0, rng),
        eval_repeats=max(
            1, min(5, int(_jitter_int(cfg.eval_repeats, 0.3, 1, 5, rng)))
        ),
        accept_temp=_jitter_float(cfg.accept_temp, 0.5, 0.01, 1.0, rng),
    )


def meta_meta_search(
    base_sys_cfg: SystemConfig,
    base_mm_cfg: MetaMetaConfig,
    rounds: int,
    seed: int,
) -> Tuple[MetaMetaConfig, float]:
    """Level 2: redesign the meta-search process by searching over MetaMetaConfig."""
    rng = random.Random(seed)
    best_mm = base_mm_cfg
    # baseline: run a short meta-search with current meta-meta config
    _, best_score = meta_search(base_sys_cfg, best_mm, rounds=3, seed=seed + 1)

    ensure_state_dirs()
    history_path = STATE_DIR / "meta_meta_history.jsonl"
    with history_path.open("a", encoding="utf-8") as f_hist:
        for r in range(rounds):
            cand_mm = mutate_meta_meta_config(best_mm, rng)
            _, cand_score = meta_search(
                base_sys_cfg, cand_mm, rounds=3, seed=seed + 1000 + r
            )
            accepted = accept_new_score(best_score, cand_score, best_mm.accept_temp, rng)
            rec = {
                "round": r,
                "base_score": best_score,
                "cand_score": cand_score,
                "accepted": accepted,
                "cand_mm": asdict(cand_mm),
            }
            f_hist.write(json.dumps(rec) + "\n")
            if accepted:
                best_mm = cand_mm
                best_score = cand_score
                save_meta_meta_config(best_mm)
    return best_mm, best_score


# -----------------------------------------------------------
# CLI
# -----------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Multi-level meta-RSI engine: representation, update rule, task generator, "
            "and meta-meta redesign."
        ),
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s_episode = sub.add_parser(
        "run-episode",
        help="Run a single Level-0 episode with current SystemConfig.",
    )
    s_episode.add_argument("--seed", type=int, default=0)

    s_meta = sub.add_parser("run-meta", help="Run Level-1 meta search over SystemConfig.")
    s_meta.add_argument("--rounds", type=int, default=5)
    s_meta.add_argument("--seed", type=int, default=1)

    s_meta2 = sub.add_parser("run-meta2", help="Run Level-2 meta-meta search over MetaMetaConfig.")
    s_meta2.add_argument("--rounds", type=int, default=5)
    s_meta2.add_argument("--seed", type=int, default=2)

    s_show = sub.add_parser(
        "show-config", help="Print current SystemConfig and MetaMetaConfig."
    )

    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.cmd == "run-episode":
        cfg = load_system_config()
        summary = run_system_episode(cfg, seed=args.seed)
        ensure_state_dirs()
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        path = RUNS_DIR / f"episode_{ts}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary": asdict(summary),
                    "timestamp": ts,
                },
                f,
                indent=2,
                sort_keys=True,
            )
        print(
            f"Episode done: best_loss={summary.best_loss:.6f}, "
            f"avg_loss={summary.avg_loss:.6f}, complexity={summary.best_complexity:.1f}"
        )

    elif args.cmd == "run-meta":
        sys_cfg = load_system_config()
        mm_cfg = load_meta_meta_config()
        best_cfg, best_score = meta_search(
            sys_cfg, mm_cfg, rounds=args.rounds, seed=args.seed
        )
        print("Meta-search complete.")
        print("Best score:", best_score)
        print(
            "Best SystemConfig:",
            json.dumps(_system_config_to_dict(best_cfg), indent=2, sort_keys=True),
        )

    elif args.cmd == "run-meta2":
        sys_cfg = load_system_config()
        mm_cfg = load_meta_meta_config()
        best_mm, best_score = meta_meta_search(
            sys_cfg, mm_cfg, rounds=args.rounds, seed=args.seed
        )
        print("Meta-meta-search complete.")
        print(
            "Best meta-meta config:",
            json.dumps(asdict(best_mm), indent=2, sort_keys=True),
        )
        print("Resulting best score estimate:", best_score)

    elif args.cmd == "show-config":
        sys_cfg = load_system_config()
        mm_cfg = load_meta_meta_config()
        print("SystemConfig:")
        print(json.dumps(_system_config_to_dict(sys_cfg), indent=2, sort_keys=True))
        print("\nMetaMetaConfig:")
        print(json.dumps(asdict(mm_cfg), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
