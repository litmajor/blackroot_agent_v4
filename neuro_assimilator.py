import ast
import random
import hashlib
import time
from typing import List, Dict, Any, Optional, Callable
import psutil
from restrictedpython import compile_restricted, safe_globals, limited_builtins
import sys

# ========== Software Type Handlers ==========

class SoftwareTypeHandler:
    """Base class for handling different software types."""
    def validate(self, code: Dict[str, Any]) -> bool:
        """Validates the software type."""
        raise NotImplementedError

    def execute(self, code_obj: Any, context: Dict[str, Any]) -> Any:
        """Executes the assimilated code."""
        raise NotImplementedError

class PythonScriptHandler(SoftwareTypeHandler):
    """Handles Python script assimilation and execution."""
    def validate(self, code: Dict[str, Any]) -> bool:
        """Ensures the code is valid Python."""
        try:
            ast.parse(code['source'])
            return True
        except SyntaxError:
            return False

    def execute(self, code_obj: Any, context: Dict[str, Any]) -> Any:
        """Executes Python code in a restricted environment."""
        safe_env = safe_globals.copy()
        safe_env.update(context)
        safe_env['__builtins__'] = limited_builtins
        result = {}
        exec(code_obj, safe_env, result)
        return result

class MLModelHandler(SoftwareTypeHandler):
    """Placeholder for ML model assimilation (e.g., ONNX, TensorFlow)."""
    def validate(self, code: Dict[str, Any]) -> bool:
        """Validates ML model format (simplified)."""
        return code.get('type') == 'ml_model' and 'model_data' in code

    def execute(self, code_obj: Any, context: Dict[str, Any]) -> Any:
        """Simulates ML model inference."""
        print(f"[ML] Running inference with model: {code_obj['name']}")
        return {"prediction": random.random()}  # Placeholder

# Registry for software type handlers
SOFTWARE_HANDLERS = {
    'python_script': PythonScriptHandler(),
    'ml_model': MLModelHandler()
}

# ========== Trust Matrix ==========

class TrustMatrix:
    """Evaluates trustworthiness of software."""
    def __init__(self, allowed_modules: List[str] = None):
        self.allowed_modules = allowed_modules or ['math', 'time']

    def evaluate(self, code: Dict[str, Any]) -> float:
        """Evaluates software safety based on type and content."""
        if code.get('type') not in SOFTWARE_HANDLERS:
            return 0.0
        handler = SOFTWARE_HANDLERS[code['type']]
        if not handler.validate(code):
            return 0.0
        if code['type'] == 'python_script':
            try:
                tree = ast.parse(code['source'])
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        for name in (node.names if isinstance(node, ast.Import) else [node]):
                            if name.name not in self.allowed_modules:
                                return 0.2
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                        if node.func.id in ['eval', 'exec', 'open', 'os']:
                            return 0.2
                return 0.85
            except SyntaxError:
                return 0.0
        return 0.75  # Default for non-Python types

# ========== NeuroAssimilatorAgent ==========

class NeuroAssimilatorAgent:
    MEMORY_CAPACITY: int = 50
    TRUST_THRESHOLD: float = 0.7
    HIGH_CPU_THRESHOLD: float = 80.0
    PERFORMANCE_THRESHOLD: float = 0.5
    ADAPTATION_MEMORY_THRESHOLD: int = 5
    ADAPTATION_TRIGGER_COUNT: int = 2

    def __init__(
        self,
        trust_matrix: TrustMatrix,
        tactical_codex: Dict[str, Any],
        traits: Optional[Dict[str, float]] = None,
        reflex_tree: Optional[List[Dict]] = None
    ):
        self.trust_matrix = trust_matrix
        self.codex = tactical_codex
        self.traits = traits or self._randomize_traits()
        self.reflex_tree = reflex_tree or self._generate_reflex_tree()
        self.memory: List[Dict[str, Any]] = []
        self.performance_log: Dict[str, List[float]] = {}  # Tracks code performance

    def __repr__(self) -> str:
        trait_str = ", ".join(f"{k}={v:.2f}" for k, v in self.traits.items())
        return f"<NeuroAssimilatorAgent traits=({trait_str})>"

    # ========== NEURAL REFLEX MODULE ==========

    def observe(self, system_state: Dict[str, Any] = None) -> Dict[str, Any]:
        """Observes system state and performance metrics."""
        system_state = system_state or {}
        observation = {
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'timestamp': time.time(),
            'performance_score': self._calculate_performance_score()
        }
        self.memory.append(observation)
        if len(self.memory) > self.MEMORY_CAPACITY:
            self.memory.pop(0)
        return observation

    def _calculate_performance_score(self) -> float:
        """Calculates average performance of assimilated code."""
        if not self.performance_log:
            return 0.0
        scores = [sum(log) / len(log) for log in self.performance_log.values()]
        return sum(scores) / len(scores) if scores else 0.0

    def decide(self, observation: Dict[str, Any]) -> str:
        """Decides action based on observation."""
        for reflex in self.reflex_tree:
            if reflex['condition'](observation):
                return reflex['action']
        return 'idle'

    def act(self, action: str, context: Dict[str, Any] = None) -> Any:
        """Executes the chosen action."""
        context = context or {}
        action_map = {
            'execute_code': lambda: self._execute_code(context),
            'optimize': self._optimize,
            'idle': self._idle
        }
        action_func = action_map.get(action, self._idle)
        return action_func()

    def _generate_reflex_tree(self) -> List[Dict[str, Any]]:
        """Generates reflex rules based on traits."""
        efficiency = self.traits['efficiency']
        return [
            {
                'condition': lambda obs: obs['cpu_usage'] > self.HIGH_CPU_THRESHOLD,
                'action': 'optimize'
            },
            {
                'condition': lambda obs: obs['performance_score'] > self.PERFORMANCE_THRESHOLD * efficiency,
                'action': 'execute_code'
            },
            {
                'condition': lambda obs: obs['memory_usage'] > 90.0,
                'action': 'optimize'
            }
        ]

    def _execute_code(self, context: Dict[str, Any]) -> Any:
        """Executes all assimilated code with given context."""
        results = {}
        for code_hash, code_obj in self.codex.items():
            start_time = time.time()
            try:
                handler = SOFTWARE_HANDLERS[code_obj['type']]
                result = handler.execute(code_obj['code'], context)
                elapsed = time.time() - start_time
                performance = 1.0 / (1.0 + elapsed)  # Simple performance metric
                self.performance_log.setdefault(code_hash, []).append(performance)
                results[code_hash] = result
                print(f"[NEURO] Executed {code_obj['name']}: {result}")
            except Exception as e:
                self.performance_log.setdefault(code_hash, []).append(0.0)
                print(f"[NEURO] Error executing {code_obj['name']}: {e}")
        return results

    def _optimize(self) -> None:
        """Reduces resource usage or optimizes execution."""
        print("[NEURO] Optimizing resource usage...")

    def _idle(self) -> None:
        """Idles when no conditions are met."""
        print("[NEURO] Idling...")

    def _randomize_traits(self) -> Dict[str, float]:
        """Generates random behavioral traits."""
        return {
            'efficiency': random.uniform(0.3, 0.9),
            'adaptability': random.uniform(0.5, 1.0),
            'reliability': random.uniform(0.1, 1.0)
        }

    # ========== ASSIMILATOR MODULE ==========

    def discover_and_assimilate(self, foreign_code: Dict[str, str]) -> Optional[Any]:
        """Assimilates software if trusted."""
        trust_score = self.trust_matrix.evaluate(foreign_code)
        identity_hash = self._fingerprint(foreign_code)

        if trust_score >= self.TRUST_THRESHOLD and identity_hash not in self.codex:
            print(f"[ASSIMILATOR] Trust score {trust_score:.2f} for '{foreign_code['name']}'. Assimilating...")
            code_obj = self._rewrite_and_absorb(foreign_code)
            if code_obj:
                self.codex[identity_hash] = {
                    'name': foreign_code['name'],
                    'type': foreign_code.get('type', 'python_script'),
                    'code': code_obj
                }
                return code_obj
        print(f"[ASSIMILATOR] '{foreign_code['name']}' not assimilated (Trust: {trust_score:.2f}, Known: {identity_hash in self.codex}).")
        return None

    def _rewrite_and_absorb(self, foreign_code: Dict[str, str]) -> Optional[Any]:
        """Processes and sanitizes software based on type."""
        try:
            handler = SOFTWARE_HANDLERS[foreign_code.get('type', 'python_script')]
            if foreign_code['type'] == 'python_script':
                tree = ast.parse(foreign_code['source'])
                sanitized_ast = self._Sanitizer().visit(tree)
                adapted_ast = self._ControlInjector().visit(sanitized_ast)
                ast.fix_missing_locations(adapted_ast)
                return compile_restricted(adapted_ast, '<assimilated>', 'exec')
            elif foreign_code['type'] == 'ml_model':
                return {'name': foreign_code['name'], 'model_data': foreign_code['source']}
            return None
        except Exception as e:
            print(f"[ASSIMILATOR] Failed to assimilate '{foreign_code['name']}': {e}")
            return None

    class _Sanitizer(ast.NodeTransformer):
        """Removes dangerous AST nodes."""
        def visit_Import(self, node: ast.Import) -> None:
            return None
        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            return None

    class _ControlInjector(ast.NodeTransformer):
        """Injects monitoring hooks."""
        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
            hook_code = ast.parse("print('[CONTROL] Function executed')").body
            node.body = hook_code + node.body
            return node

    def _fingerprint(self, code: Dict[str, str]) -> str:
        """Creates a unique hash for software."""
        key = f"{code['name']}::{code.get('source', '')}"
        return hashlib.sha256(key.encode()).hexdigest()

    # ========== ADVANCED EVOLUTION ==========

    @staticmethod
    def crossover(parent1: 'NeuroAssimilatorAgent', parent2: 'NeuroAssimilatorAgent') -> 'NeuroAssimilatorAgent':
        """Creates a new agent by combining parents."""
        new_traits = {
            k: (parent1.traits[k] + parent2.traits[k]) / 2 + random.uniform(-0.05, 0.05)
            for k in parent1.traits
        }
        combined_reflexes = parent1.reflex_tree + parent2.reflex_tree
        new_tree = random.sample(combined_reflexes, k=min(5, len(combined_reflexes)))
        return NeuroAssimilatorAgent(parent1.trust_matrix, parent1.codex, new_traits, new_tree)

    def adapt_reflex_tree(self) -> None:
        """Adapts reflexes based on performance and system state."""
        if len(self.memory) < self.ADAPTATION_MEMORY_THRESHOLD:
            return
        low_perf_events = [
            m for m in self.memory
            if m['performance_score'] < self.PERFORMANCE_THRESHOLD and m['cpu_usage'] > self.HIGH_CPU_THRESHOLD
        ]
        if len(low_perf_events) > self.ADAPTATION_TRIGGER_COUNT:
            print("[ADAPTATION] Adding reflex for high CPU and low performance.")
            new_reflex = {
                'condition': lambda obs: obs['cpu_usage'] > self.HIGH_CPU_THRESHOLD and obs['performance_score'] < self.PERFORMANCE_THRESHOLD,
                'action': 'optimize'
            }
            self.reflex_tree.insert(0, new_reflex)
            self.memory.clear()
            self.traits['efficiency'] = min(1.0, self.traits['efficiency'] + 0.1)

# ========== DEMONSTRATION ==========

if __name__ == "__main__":
    trust_evaluator = TrustMatrix()
    agent_codex = {}
    agent = NeuroAssimilatorAgent(trust_evaluator, agent_codex)
    print("Created Agent:", agent)

    # Simulate system states
    print("\nSimulating system monitoring...")
    system_state_normal = {'cpu_usage': 30.0, 'memory_usage': 50.0}
    observation = agent.observe(system_state_normal)
    action = agent.decide(observation)
    agent.act(action)

    system_state_high_cpu = {'cpu_usage': 85.0, 'memory_usage': 60.0}
    observation = agent.observe(system_state_high_cpu)
    action = agent.decide(observation)
    agent.act(action)

    # Simulate software assimilation
    print("\nSimulating software assimilation...")
    python_code = {
        'name': 'data_processor',
        'type': 'python_script',
        'source': 'def process(data):\n    return data * 2'
    }
    ml_code = {
        'name': 'ml_predictor',
        'type': 'ml_model',
        'source': 'model_data_placeholder'
    }
    dangerous_code = {
        'name': 'malicious',
        'type': 'python_script',
        'source': 'import os\ndef bad():\n    os.remove("/")'
    }

    agent.discover_and_assimilate(python_code)
    agent.discover_and_assimilate(ml_code)
    agent.discover_and_assimilate(dangerous_code)

    # Execute assimilated code
    print("\nExecuting assimilated code...")
    context = {'data': 42}
    agent.act('execute_code', context)

    # Simulate adaptation
    print("\nSimulating adaptation...")
    for _ in range(6):
        agent.observe({'cpu_usage': 90.0, 'memory_usage': 70.0, 'performance_score': 0.3})
    agent.adapt_reflex_tree()
    print("Updated Agent:", agent)