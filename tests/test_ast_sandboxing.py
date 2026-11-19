#!/usr/bin/env python3
"""
Unit and Integration Tests for Option D: AST Sandboxing & Module Validation Harness
Validates:
1. ASTPolicyValidator detecting forbidden calls, attributes, dunders, and module imports.
2. MutationVerificationHarness verifying syntax, divergence novelty, security, and semantic invariance.
3. NeuroAssimilatorAgent end-to-end integration with validation, mutation, and vault storage.
"""

import sys
import ast
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from assimilate.src.neuro_assimilator import (
    ASTPolicyValidator,
    MutationVerificationHarness,
    SecurityManager,
    NeuroAssimilatorAgent
)


def test_ast_policy_validator_safe_code():
    validator = ASTPolicyValidator()
    safe_code = """
def calculate_metrics(values):
    total = sum(values)
    count = len(values)
    return total / max(1, count)

result = calculate_metrics([10, 20, 30, 40])
"""
    ok, violations = validator.validate(safe_code)
    assert ok is True, f"Safe code failed validation: {violations}"
    assert len(violations) == 0


def test_ast_policy_validator_forbidden_calls():
    validator = ASTPolicyValidator()
    
    # eval call
    ok, violations = validator.validate("x = eval('2 + 2')")
    assert ok is False
    assert any("eval" in v for v in violations)

    # __import__ call
    ok, violations = validator.validate("mod = __import__('os')")
    assert ok is False
    assert any("__import__" in v for v in violations)

    # open call
    ok, violations = validator.validate("f = open('/etc/passwd')")
    assert ok is False
    assert any("open" in v for v in violations)


def test_ast_policy_validator_forbidden_attributes():
    validator = ASTPolicyValidator()
    
    # os.system
    ok, violations = validator.validate("import os\nos.system('whoami')")
    assert ok is False
    assert any("system" in v for v in violations)

    # subprocess.Popen
    ok, violations = validator.validate("import subprocess\nsubprocess.Popen(['calc'])")
    assert ok is False
    assert any("Popen" in v for v in violations)


def test_ast_policy_validator_sandbox_escape_dunders():
    validator = ASTPolicyValidator()
    
    # __subclasses__
    ok, violations = validator.validate("subs = ().__class__.__bases__[0].__subclasses__()")
    assert ok is False
    assert any("__subclasses__" in v or "__bases__" in v for v in violations)

    # __globals__
    ok, violations = validator.validate("g = (lambda: None).__globals__")
    assert ok is False
    assert any("__globals__" in v for v in violations)

    # __builtins__
    ok, violations = validator.validate("b = getattr(func, '__builtins__')")
    assert ok is False
    assert any("__builtins__" in v for v in violations)


def test_ast_policy_validator_forbidden_imports():
    validator = ASTPolicyValidator()
    
    # ctypes import
    ok, violations = validator.validate("import ctypes\nlibc = ctypes.CDLL(None)")
    assert ok is False
    assert any("ctypes" in v for v in violations)

    # win32api from import
    ok, violations = validator.validate("from win32api import GetCurrentProcess")
    assert ok is False
    assert any("win32api" in v for v in violations)


def test_mutation_verification_harness_semantic_invariance():
    sm = SecurityManager()
    harness = MutationVerificationHarness(security_manager=sm)
    
    original = """
def add_numbers(a, b):
    return a + b
result = add_numbers(15, 25)
"""
    mutated = """
def _fn_renamed(arg_x, arg_y):
    return arg_x + arg_y
result = _fn_renamed(15, 25)
"""
    verified, report = harness.verify_transformation(original, mutated)
    assert verified is True, f"Verification failed: {report}"
    assert report["valid_syntax"] is True
    assert report["security_passed"] is True
    assert report["semantic_match"] is True
    assert report["divergence_score"] > 0.0


def test_mutation_verification_harness_catches_invalid_syntax():
    harness = MutationVerificationHarness()
    original = "x = 42\nresult = x"
    broken_mutated = "x = (42 +\nresult = x"
    verified, report = harness.verify_transformation(original, broken_mutated)
    assert verified is False
    assert report["valid_syntax"] is False


def test_mutation_verification_harness_catches_injected_hazards():
    harness = MutationVerificationHarness()
    original = "x = 42\nresult = x"
    malicious_mutated = "x = eval('42')\nresult = x"
    verified, report = harness.verify_transformation(original, malicious_mutated)
    assert verified is False
    assert report["security_passed"] is False


def test_neuro_assimilator_agent_with_ast_validation():
    agent = NeuroAssimilatorAgent(vault_password="test_ast_pass")
    
    # 1. Test validate_code_ast
    safe_ok, _ = agent.validate_code_ast("a = 10\nb = 20\nresult = a + b")
    assert safe_ok is True

    unsafe_ok, violations = agent.validate_code_ast("import ctypes\nx = ctypes.c_int(5)")
    assert unsafe_ok is False
    assert len(violations) > 0

    # 2. Test assimilate_with_mutation on safe snippet
    safe_code = """
def compute_signature(val):
    return val * 3

result = compute_signature(7)
"""
    compiled = agent.assimilate_with_mutation(safe_code, name="safe_ast_test")
    assert compiled is not None, "assimilate_with_mutation returned None for safe code"

    # 3. Test assimilate_with_mutation rejects unsafe snippet
    unsafe_code = """
import os
os.system('calc')
result = 1
"""
    rejected = agent.assimilate_with_mutation(unsafe_code, name="unsafe_ast_test")
    assert rejected is None, "assimilate_with_mutation should reject unsafe code containing os.system"
