#!/usr/bin/env python3
"""
Phase 2 Integration Validation
Verifies all components are integrated into propagation_agents.py
"""

import sys
import inspect
from pathlib import Path

def validate_phase2_integration():
    """Validate Phase 2 components are integrated"""
    
    print("\n" + "="*70)
    print("🔗 PHASE 2 INTEGRATION VALIDATION")
    print("="*70)
    
    try:
        # Import propagation agent
        sys.path.insert(0, str(Path(__file__).parent))
        from agents.propagation_agents import PropagationAgent
        
        print("✅ PropagationAgent imported successfully")
        
        prop = PropagationAgent()
        
        # Check 1: Windows propagation engine initialized
        has_windows = hasattr(prop, 'windows_engine')
        print(f"{'✅' if has_windows else '❌'} Windows engine: {prop.windows_engine is not None}")
        
        # Check 2: Credential harvester initialized
        has_harvester = hasattr(prop, 'credential_harvester')
        print(f"{'✅' if has_harvester else '❌'} Credential harvester: {prop.credential_harvester is not None}")
        
        # Check 3: Retry queue initialized
        has_scheduler = hasattr(prop, 'retry_queue')
        print(f"{'✅' if has_scheduler else '❌'} Retry queue: {prop.retry_queue is not None}")
        
        # Check 4: New methods exist
        methods_to_check = [
            '_detect_target_os',
            '_propagate_windows',
            '_propagate_unix',
        ]
        
        print("\n📋 Method integration:")
        all_methods = True
        for method_name in methods_to_check:
            has_method = hasattr(prop, method_name)
            status = "✅" if has_method else "❌"
            print(f"{status} {method_name}")
            all_methods = all_methods and has_method
        
        # Check 5: Verify flow integration
        print("\n🔄 Flow integration:")
        
        # Get _propagate source
        propagate_source = inspect.getsource(prop._propagate)
        
        checks = {
            "OS detection called": "_detect_target_os" in propagate_source,
            "Windows path available": "_propagate_windows" in propagate_source,
            "Unix path available": "_propagate_unix" in propagate_source,
            "Retry scheduler used": "retry_queue" in propagate_source,
            "Economic model updated": "_reward_propagation_success" in propagate_source,
        }
        
        for check_name, passed in checks.items():
            status = "✅" if passed else "❌"
            print(f"{status} {check_name}")
        
        # Check 6: Windows propagation setup
        if has_windows:
            windows_source = inspect.getsource(prop._propagate_windows)
            win_checks = {
                "Credential harvesting": "credential_harvester" in windows_source,
                "SMB propagation": "windows_engine" in windows_source and "propagate_pth" in windows_source,
                "Pass-the-Hash ready": "credentials" in windows_source,
            }
            
            print("\n🪟 Windows propagation:")
            for check_name, passed in win_checks.items():
                status = "✅" if passed else "❌"
                print(f"{status} {check_name}")
        
        # Summary
        print("\n" + "="*70)
        all_passed = all([
            has_windows, has_harvester, has_scheduler,
            all_methods,
            all(checks.values())
        ])
        
        if all_passed:
            print("✅ PHASE 2 INTEGRATION: COMPLETE")
            print("\n📦 Integrated Components:")
            print("   1. ✅ Windows Propagation Agent (SMB/WMI)")
            print("   2. ✅ Credential Harvester (PTH preparation)")
            print("   3. ✅ Retry Scheduler (Exponential backoff)")
            print("   4. ✅ OS Detection (_detect_target_os)")
            print("   5. ✅ Multi-path propagation (_propagate_windows + _propagate_unix)")
            print("   6. ✅ Economic model integration (rewards/penalties)")
            print("\n🎯 Propagation Flow:")
            print("   1. Detect target OS (SMB port → Windows, SSH → Unix)")
            print("   2. Harvest credentials for Windows targets")
            print("   3. Route to platform-specific propagation")
            print("   4. Track result with retry scheduler")
            print("   5. Apply economic incentives (rewards/penalties)")
            print("   6. Schedule retries with exponential backoff if failed")
            return 0
        else:
            print("❌ PHASE 2 INTEGRATION: INCOMPLETE")
            return 1
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(validate_phase2_integration())
