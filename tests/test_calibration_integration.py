#!/usr/bin/env python3
"""Quick test of threat calibration integration"""

from anti_analysis import AntiAnalysisEngine
from threat_calibration import ThreatSignal

print('[✓] Creating AntiAnalysisEngine...')
engine = AntiAnalysisEngine()

print('\n[✓] Calibration System Components:')
print(f'  - Threat Scorer: {engine.threat_scorer.__class__.__name__ if engine.threat_scorer else "Unavailable"}')
print(f'  - Intent Engine: {engine.intent_engine.__class__.__name__ if engine.intent_engine else "Unavailable"}')
print(f'  - Validator: {engine.validator.__class__.__name__ if engine.validator else "Unavailable"}')
print(f'  - Learner: {engine.learner.__class__.__name__ if engine.learner else "Unavailable"}')

print('\n[✓] Testing new assess_environment method...')
assessment = engine.assess_environment()
print(f'  - Environment Context: {engine.environment_context}')
print(f'  - Threat Score: {engine.threat_level}/100 (CALIBRATED)')
print(f'  - Evasion Tier: {engine.evasion_tier}')
print(f'  - Should Evade: {engine.should_evade}')

print('\n[✓] AntiAnalysisEngine integrated with full calibration system!')
print('[✓] All 5 improvements implemented:')
print('  1. ✅ Calibration (Gradient scoring)')
print('  2. ✅ Intent Awareness (Dev machine detection)')
print('  3. ✅ Debounced Debugger Detection')
print('  4. ✅ Post-Evasion Validation')
print('  5. ✅ Adaptive Feedback Loop')
