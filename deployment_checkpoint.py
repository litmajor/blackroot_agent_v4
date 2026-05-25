#!/usr/bin/env python3
"""
DEPLOYMENT CHECKPOINT & INTEGRATION VERIFICATION
Blackroot Agent v4 - Intelligent Operations Integration
===========================================================

This script verifies that all operational managers are properly integrated
into the BlackrootKernel and ready for full deployment.
"""

import os
import sys
import json
from pathlib import Path

# Add workspace to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def verify_module_files():
    """Verify all required module files exist."""
    print("\n[1] Verifying Module Files...")
    required_files = [
        "unified_integration.py",
        "unified_causal_reasoner_v2.py",
        "intelligent_credential_extraction_manager.py",
        "intelligent_data_exfiltration_manager.py",
        "intelligent_persistence_manager.py",
        "intelligent_escalation_manager.py",
        "intelligent_lateral_movement_manager.py",
        "intelligent_c2_callback.py",
        "agents/intelligent_operations.py",
        "agents/__init__.py",
        "config.json",
    ]
    
    missing = []
    for file in required_files:
        if not Path(file).exists():
            missing.append(file)
            print(f"  ✗ {file}")
        else:
            print(f"  ✓ {file}")
    
    return len(missing) == 0


def verify_config():
    """Verify config.json has intelligent_operations section."""
    print("\n[2] Verifying Configuration...")
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
        
        if "intelligent_operations" in config:
            print("  ✓ intelligent_operations configuration found")
            io_config = config["intelligent_operations"]
            
            sections = [
                "credential_extraction",
                "data_exfiltration", 
                "persistence",
                "escalation",
                "lateral_movement",
                "c2_callback",
            ]
            
            for section in sections:
                if section in io_config:
                    print(f"    ✓ {section} configured")
                else:
                    print(f"    ✗ {section} missing")
                    return False
            
            return True
        else:
            print("  ✗ intelligent_operations section missing from config.json")
            return False
    except Exception as e:
        print(f"  ✗ Error reading config: {e}")
        return False


def verify_imports():
    """Verify all modules can be imported."""
    print("\n[3] Verifying Module Imports...")
    modules_to_import = [
        ("unified_integration", "create_integration"),
        ("unified_causal_reasoner_v2", "UnifiedCausalReasonerV2"),
        ("intelligent_credential_extraction_manager", "IntelligentCredentialExtractionManager"),
        ("intelligent_data_exfiltration_manager", "IntelligentDataExfiltrationManager"),
        ("intelligent_persistence_manager", "IntelligentPersistenceManager"),
        ("intelligent_escalation_manager", "IntelligentEscalationManager"),
        ("intelligent_lateral_movement_manager", "IntelligentLateralMovementManager"),
        ("intelligent_c2_callback", "IntelligentC2Callback"),
    ]
    
    all_ok = True
    for module_name, class_name in modules_to_import:
        try:
            module = __import__(module_name)
            if hasattr(module, class_name):
                print(f"  ✓ {module_name}.{class_name}")
            else:
                print(f"  ✗ {module_name} missing {class_name}")
                all_ok = False
        except Exception as e:
            print(f"  ✗ {module_name}: {e}")
            all_ok = False
    
    return all_ok


def verify_agent_registration():
    """Verify intelligent operations agent can be registered."""
    print("\n[4] Verifying Agent Registration...")
    try:
        from agents.intelligent_operations import IntelligentOperationsAgent
        agent = IntelligentOperationsAgent()
        
        print(f"  ✓ IntelligentOperationsAgent instantiated")
        print(f"    - Codename: {agent.codename}")
        print(f"    - Priority: {agent.priority}")
        print(f"    - Managers: 6 operational managers initialized")
        
        return True
    except Exception as e:
        print(f"  ✗ Agent registration failed: {e}")
        return False


def verify_learning_persistence():
    """Verify learning persistence layer."""
    print("\n[5] Verifying Learning Persistence Layer...")
    try:
        from unified_causal_reasoner_v2 import UnifiedCausalReasonerV2
        
        reasoner = UnifiedCausalReasonerV2()
        
        # Check for persistence methods
        methods = ["_save_learning_state", "_load_learning_state", "_calculate_success_rates"]
        for method_name in methods:
            if hasattr(reasoner, method_name):
                print(f"  ✓ {method_name} present")
            else:
                print(f"  ✗ {method_name} missing")
                return False
        
        print("  ✓ Learning persistence layer ready")
        return True
    except Exception as e:
        print(f"  ✗ Learning persistence verification failed: {e}")
        return False


def verify_unified_integration():
    """Verify unified integration layer."""
    print("\n[6] Verifying Unified Integration Layer...")
    try:
        from unified_integration import create_integration
        
        integration = create_integration("deployment_test")
        
        # Check for key methods
        methods = ["decide", "record_outcome", "get_decision_context"]
        for method_name in methods:
            if hasattr(integration, method_name):
                print(f"  ✓ {method_name} present")
            else:
                print(f"  ✗ {method_name} missing")
                return False
        
        print("  ✓ Unified integration layer ready")
        return True
    except Exception as e:
        print(f"  ✗ Unified integration verification failed: {e}")
        return False


def generate_deployment_report():
    """Generate final deployment report."""
    print("\n" + "="*70)
    print("  DEPLOYMENT CHECKPOINT REPORT")
    print("="*70)
    
    checks = [
        ("Module Files", verify_module_files),
        ("Configuration", verify_config),
        ("Module Imports", verify_imports),
        ("Agent Registration", verify_agent_registration),
        ("Learning Persistence", verify_learning_persistence),
        ("Unified Integration", verify_unified_integration),
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"\n✗ {check_name} check failed with exception: {e}")
            results.append((check_name, False))
    
    # Summary
    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for check_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {check_name}")
    
    print(f"\n  Overall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n  ✓✓✓ READY FOR DEPLOYMENT ✓✓✓")
        print("\n  Next steps:")
        print("    1. Run: python main.py")
        print("    2. Kernel will initialize all agents including IntelligentOperationsAgent")
        print("    3. Operations will execute through unified causal reasoner")
        print("    4. Learning persistence automatic via learning_state.json")
        return True
    else:
        print("\n  ✗✗✗ DEPLOYMENT BLOCKED ✗✗✗")
        print("  Fix failed checks before deploying")
        return False


if __name__ == "__main__":
    success = generate_deployment_report()
    sys.exit(0 if success else 1)
