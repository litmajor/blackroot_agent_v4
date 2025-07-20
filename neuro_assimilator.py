import ast
import random
import hashlib
import time

class NeuroAssimilatorAgent:
    def __init__(self, trust_matrix, tactical_codex, traits=None, reflex_tree=None):
        self.trust_matrix = trust_matrix
        self.codex = tactical_codex
        self.traits = traits or self._randomize_traits()
        self.reflex_tree = reflex_tree or self._generate_reflex_tree()
        self.memory = []

    # ========== NEURAL REFLEX MODULE ==========
    def observe(self, system_state):
        observation = {
            'scan_detected': 'antivirus' in system_state.get('processes', []),
            'latency': system_state.get('network_latency', 0),
            'cpu_usage': system_state.get('cpu', 0),
            'time': time.time()
        }
        self.memory.append(observation)
        if len(self.memory) > 50:
            self.memory.pop(0)
        return observation

    def decide(self, observation):
        for reflex in self.reflex_tree:
            if reflex['condition'](observation):
                return reflex['action']
        return 'idle'

    def act(self, action):
        if action == 'hide':
            self._go_dormant()
        elif action == 'relocate':
            self._migrate()
        elif action == 'attack':
            self._execute_payload()
        else:
            self._idle()

    def _generate_reflex_tree(self):
        stealth = self.traits['stealth']
        aggressiveness = self.traits['aggressiveness']
        return [
            {'condition': lambda obs: obs['scan_detected'], 'action': 'hide'},
            {'condition': lambda obs: obs['cpu_usage'] > (80 * (1 - stealth)), 'action': 'relocate'},
            {'condition': lambda obs: obs['latency'] < (50 * aggressiveness), 'action': 'attack'}
        ]

    def _go_dormant(self):
        print("[NEURO] Entering stealth mode… (simulated pause)")

    def _migrate(self):
        new_ip = f"192.168.1.{random.randint(10, 250)}"
        print(f"[NEURO] Relocating to node: {new_ip}")

    def _execute_payload(self):
        print("[NEURO] Deploying offensive payload to target system…")

    def _idle(self):
        print("[NEURO] No trigger conditions met. Idling...")

    def _randomize_traits(self):
        return {
            'aggressiveness': random.uniform(0.3, 0.9),
            'stealth': random.uniform(0.5, 1.0),
            'reaction_time': random.uniform(0.1, 1.0)
        }

    # ========== ASSIMILATOR MODULE ==========
    def discover_and_assimilate(self, foreign_code):
        trust_score = self.trust_matrix.evaluate(foreign_code)
        identity_hash = self._fingerprint(foreign_code)

        if trust_score >= 0.65 and identity_hash not in self.codex:
            payload = self._rewrite_and_absorb(foreign_code)
            self.codex[identity_hash] = payload
            return payload
        return None

    def _rewrite_and_absorb(self, foreign_code):
        try:
            parsed_ast = ast.parse(foreign_code['source'])
            sanitized = self._sanitize(parsed_ast)
            adapted = self._inject_control(sanitized)
            compiled = compile(adapted, filename="<assimilated>", mode="exec")
            return compiled
        except Exception as e:
            return f"[ASSIMILATION FAILED]: {e}"

    def _sanitize(self, ast_tree):
        class Sanitizer(ast.NodeTransformer):
            def visit_Import(self, node): return None
            def visit_ImportFrom(self, node): return None
        return Sanitizer().visit(ast_tree)

    def _inject_control(self, ast_tree):
        class ControlInjector(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                control_code = ast.parse("print('[SHOGUN CONTROL HOOKED]')").body
                node.body = control_code + node.body
                return node
        return ast.fix_missing_locations(ControlInjector().visit(ast_tree))

    def _fingerprint(self, code):
        key = f"{code['name']}::{code['source']}"
        return hashlib.sha256(key.encode()).hexdigest()

    # ========== ADVANCED EVOLUTION ==========
    @staticmethod
    def crossover(parent1, parent2):
        new_traits = {
            k: (parent1.traits[k] + parent2.traits[k]) / 2 + random.uniform(-0.05, 0.05)
            for k in parent1.traits
        }
        new_tree = random.sample(
            parent1.reflex_tree + parent2.reflex_tree,
            k=min(5, len(parent1.reflex_tree + parent2.reflex_tree))
        )
        return NeuroAssimilatorAgent(parent1.trust_matrix, parent1.codex, new_traits, new_tree)

    def adapt_reflex_tree(self):
        # Optional: Learn from failures, generate new reflexes dynamically
        if len(self.memory) < 5: return
        scan_heavy = [m for m in self.memory if m['scan_detected'] and m['cpu_usage'] > 85]
        if len(scan_heavy) > 2:
            self.reflex_tree.append({
                'condition': lambda obs: obs['cpu_usage'] > 85,
                'action': 'hide'
            })
