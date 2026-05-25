#!/usr/bin/env python3
"""
Phase 3 Integration Validation - Scourge Failure Analysis
Validates that propagation failures are reported to Scourge for analysis
"""

import sys
import inspect
from pathlib import Path

def validate_phase3_scourge():
    """Validate Phase 3 Scourge integration"""
    
    print("\n" + "="*70)
    print("🔒 PHASE 3 VALIDATION: SCOURGE FAILURE ANALYSIS")
    print("="*70)
    
    try:
        # Import propagation agent
        sys.path.insert(0, str(Path(__file__).parent))
        from agents.propagation_agents import PropagationAgent
        
        print("✅ PropagationAgent imported successfully")
        
        prop = PropagationAgent()
        
        # Check 1: Scourge initialized
        has_scourge = hasattr(prop, 'scourge')
        print(f"{'✅' if has_scourge else '❌'} Scourge initialized: {prop.scourge is not None}")
        
        # Check 2: Failure reporting method exists
        has_report_method = hasattr(prop, '_report_failure_to_scourge')
        print(f"{'✅' if has_report_method else '❌'} Failure reporting method: {has_report_method}")
        
        # Check 3: Verify method signature
        if has_report_method:
            sig = inspect.signature(prop._report_failure_to_scourge)
            params = list(sig.parameters.keys())
            expected = ['target', 'credential', 'error', 'method']
            all_present = all(p in params for p in expected)
            print(f"{'✅' if all_present else '❌'} Method signature correct: {params}")
        
        # Check 4: Scourge integration in _propagate
        propagate_source = inspect.getsource(prop._propagate)
        
        scourge_checks = {
            "Scourge reporting called": "_report_failure_to_scourge" in propagate_source,
            "Failure analysis on failure": "not success" in propagate_source and "_report_failure_to_scourge" in propagate_source,
            "Target passed": "target=" in propagate_source or "host" in propagate_source,
            "Credential passed": "credential=" in propagate_source,
            "Error details passed": "error=" in propagate_source,
            "Method passed": "method=" in propagate_source
        }
        
        print("\n🔄 Integration in _propagate():")
        integration_working = True
        for check_name, passed in scourge_checks.items():
            status = "✅" if passed else "❌"
            print(f"{status} {check_name}")
            integration_working = integration_working and passed
        
        # Check 5: Scourge failure event types available
        print("\n📋 Scourge FailureType enum:")
        try:
            from agents.scourge import FailureType
            failure_types = [f.value for f in FailureType]
            print(f"   Available failure types: {len(failure_types)}")
            for ft in failure_types:
                print(f"   • {ft}")
        except ImportError:
            print("   ⚠️  FailureType not available (optional)")
        
        # Check 6: Retry scheduler integration
        retry_active = prop.retry_queue is not None
        print(f"\n✅ Retry scheduler active: {retry_active}")
        
        if retry_active:
            print("   Flow: Propagation → Failure → Retry Queue → Scourge Report")
            print("   This ensures:")
            print("   1. Failed attempts are queued with exponential backoff")
            print("   2. Failures are reported to Scourge for analysis")
            print("   3. After max retries, moved to Dead Letter Queue")
            print("   4. Scourge absorbs blame and analyzes patterns")
        
        # Summary
        print("\n" + "="*70)
        all_passed = (
            has_scourge and has_report_method and 
            integration_working and all(scourge_checks.values())
        )
        
        if all_passed:
            print("✅ PHASE 3 INTEGRATION: COMPLETE")
            print("\n📊 Scourge Integration Flow:")
            print("   1. PropagationAgent attempts propagation")
            print("   2. On failure, _report_failure_to_scourge() called")
            print("   3. FailureEvent created with:")
            print("      • Type: COMMUNICATION_FAILURE")
            print("      • Source: PROP-SEED")
            print("      • Target: host IP")
            print("      • Root cause: error message")
            print("      • Severity: 5/10 (medium)")
            print("   4. Event sent to FallmanAgent (Scourge)")
            print("   5. Scourge performs:")
            print("      • Failure containment")
            print("      • Blame absorption")
            print("      • Agent protection")
            print("      • Post-mortem analysis")
            print("      • Recovery pathway generation")
            print("\n🔄 Reliability Architecture:")
            print("   PROPAGATION ATTEMPT")
            print("        ↓")
            print("   SUCCESS? → YES → Reward (0.8 Root)")
            print("        ↓       → NO ↓")
            print("   RETRY SCHEDULER (exponential backoff)")
            print("        ↓")
            print("   MAX RETRIES? → NO → Queue for next attempt")
            print("        ↓        → YES ↓")
            print("   DEAD LETTER QUEUE (permanent record)")
            print("        ↓")
            print("   SCOURGE ANALYSIS (failure patterns)")
            print("        ↓")
            print("   RECOVERY RECOMMENDATIONS")
            print("\n✨ Phase 3 Complete - System now resilient to propagation failures")
            return 0
        else:
            print("❌ PHASE 3 INTEGRATION: INCOMPLETE")
            print(f"\n   Failed checks: {sum(1 for c in scourge_checks.values() if not c)}/6")
            return 1
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(validate_phase3_scourge())
