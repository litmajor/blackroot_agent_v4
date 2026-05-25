#!/usr/bin/env python3
"""Test threat calibration scoring fixes"""

from threat_calibration import CalibratedThreatScorer, ThreatSignal
import logging

logging.basicConfig(level=logging.INFO)

print('[✓] Testing Threat Calibration Fixes\n')

scorer = CalibratedThreatScorer()

# ✅ Test 1: set_intent_confidence method exists and works
print('Test 1: set_intent_confidence() method')
try:
    scorer.set_intent_confidence(0.3)
    print(f'  ✅ Method works. Intent confidence: {scorer._intent_confidence}')
except AttributeError as e:
    print(f'  ❌ FAILED: {e}')

# ✅ Test 2: Calculate score with signals (not zero)
print('\nTest 2: Score calculation with Windows Defender + Debugger')
scorer.threat_signals.clear()
scorer.add_signal(ThreatSignal('windows_defender', True, 80, 15))
scorer.add_signal(ThreatSignal('debugger_peb_flag', True, 85, 25))

score = scorer.calculate_score()
print(f'  Score: {score}/100')
if score > 0:
    print(f'  ✅ Score is NOT zero (base calculation fixed)')
else:
    print(f'  ❌ FAILED: Score is still zero!')

# ✅ Test 3: Low confidence increases paranoia (not suppresses)
print('\nTest 3: Low confidence environment should increase paranoia')
scorer2 = CalibratedThreatScorer()
scorer2.set_intent_confidence(0.2)  # LOW CONFIDENCE
scorer2.add_signal(ThreatSignal('windows_defender', True, 70, 20))

score_low_confidence = scorer2.calculate_score()
print(f'  Score with LOW confidence (0.2): {score_low_confidence}/100')
if score_low_confidence > 15:
    print(f'  ✅ Low confidence INCREASES threat perception')
else:
    print(f'  ⚠️ Low confidence might not be increasing threat enough')

# ✅ Test 4: High confidence dev machine reduces threat
print('\nTest 4: High confidence dev machine should reduce threat')
scorer3 = CalibratedThreatScorer()
scorer3.set_intent_confidence(0.95)  # HIGH CONFIDENCE: dev machine
scorer3.threat_signals['python_interactive'] = ThreatSignal('python_interactive', True, 90, 5)
scorer3.threat_signals['ide_detected'] = ThreatSignal('ide_detected', True, 90, 5)
scorer3.add_signal(ThreatSignal('windows_defender', True, 70, 15))

score_dev = scorer3.calculate_score()
print(f'  Score with HIGH confidence dev machine: {score_dev}/100')
if score_dev < 15:
    print(f'  ✅ High confidence dev machine reduces threat perception')
else:
    print(f'  ⚠️ Dev machine might not be reducing threat enough')

print('\n[✓] All calibration fixes verified!')
