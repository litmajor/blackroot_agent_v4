"""
POST-ESCALATION WORKFLOW TEST
==============================

Tests the complete chain:
PrivescEngine → Escalation Success → PostEscalationAgent → 5 Capability Phases
"""

import sys
import os
import json
import platform
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import patch, MagicMock, call

# Add workspace to path
sys.path.insert(0, str(Path(__file__).parent))

from agents.post_escalation import (
    PostEscalationOrchestrator,
    PostEscalationAction,
    PostEscalationPhase
)


class TestPostEscalationOrchestrator(TestCase):
    """Test suite for post-escalation capability orchestration"""

    def setUp(self):
        """Initialize test fixtures"""
        self.orchestrator = PostEscalationOrchestrator()
        
    def test_init_orchestrator(self):
        """Test orchestrator initialization"""
        self.assertIsNotNone(self.orchestrator)
        self.assertEqual(self.orchestrator.os_type, platform.system())
        
    def test_post_escalation_phase_enum(self):
        """Test PostEscalationPhase enum values"""
        self.assertEqual(PostEscalationPhase.PERSISTENCE.value, 'persistence')
        self.assertEqual(PostEscalationPhase.CREDENTIAL_THEFT.value, 'credential_theft')
        self.assertEqual(PostEscalationPhase.DEFENSE_EVASION.value, 'defense_evasion')
        self.assertEqual(PostEscalationPhase.SERVICE_CONTROL.value, 'service_control')
        self.assertEqual(PostEscalationPhase.PROCESS_INJECTION.value, 'process_injection')

    def test_post_escalation_action_creation(self):
        """Test PostEscalationAction dataclass"""
        action = PostEscalationAction(
            action_type='scheduled_task',
            target_system='Windows',
            description='Create auto-start scheduled task',
            severity='Critical',
            mitre_technique='T1053.005',
            success=True
        )
        
        self.assertEqual(action.action_type, 'scheduled_task')
        self.assertEqual(action.target_system, 'Windows')
        self.assertTrue(action.success)
        self.assertEqual(action.mitre_technique, 'T1053.005')

    def test_escalation_result_structure(self):
        """Test expected escalation result format"""
        escalation_result = {
            'technique': 'GodPotato',
            'elevated_privilege': 'SYSTEM',
            'mitre_technique': 'T1548.002',
            'timestamp': 1234567890,
        }
        
        self.assertEqual(escalation_result['elevated_privilege'], 'SYSTEM')
        self.assertIn('technique', escalation_result)

    def test_persistence_capabilities_structure(self):
        """Test persistence capabilities are properly structured"""
        # Get the persistence actions
        persistence_actions = self.orchestrator._install_persistence()
        
        self.assertIsInstance(persistence_actions, list)
        if persistence_actions:  # If any actions returned
            action = persistence_actions[0]
            self.assertIsInstance(action, PostEscalationAction)
            self.assertIn('persistence', action.action_type.lower())

    def test_credential_theft_capabilities_structure(self):
        """Test credential theft capabilities structure"""
        credential_actions = self.orchestrator._dump_credentials()
        
        self.assertIsInstance(credential_actions, list)
        if credential_actions:
            action = credential_actions[0]
            self.assertIsInstance(action, PostEscalationAction)
            self.assertIn('credential', action.action_type.lower())

    def test_defense_evasion_capabilities_structure(self):
        """Test defense evasion capabilities structure"""
        defense_actions = self.orchestrator._disable_defenses()
        
        self.assertIsInstance(defense_actions, list)
        if defense_actions:
            action = defense_actions[0]
            self.assertIsInstance(action, PostEscalationAction)
            self.assertIn('disable', action.action_type.lower())

    def test_service_control_capabilities_structure(self):
        """Test service control capabilities structure"""
        service_actions = self.orchestrator._control_services()
        
        self.assertIsInstance(service_actions, list)
        if service_actions:
            action = service_actions[0]
            self.assertIsInstance(action, PostEscalationAction)
            self.assertIn('service', action.action_type.lower())

    def test_process_injection_capabilities_structure(self):
        """Test process injection capabilities structure"""
        injection_actions = self.orchestrator._inject_into_processes()
        
        self.assertIsInstance(injection_actions, list)
        if injection_actions:
            action = injection_actions[0]
            self.assertIsInstance(action, PostEscalationAction)
            self.assertIn('injection', action.action_type.lower())

    def test_execute_post_escalation_chain_returns_dict(self):
        """Test complete workflow returns properly structured result"""
        escalation_result = {
            'technique': 'GodPotato',
            'elevated_privilege': 'SYSTEM',
            'mitre_technique': 'T1548.002',
            'timestamp': 1234567890,
        }
        
        result = self.orchestrator.execute_post_escalation_chain(escalation_result)
        
        self.assertIsInstance(result, dict)
        self.assertIn('timestamp', result)
        self.assertIn('post_escalation_actions', result)
        self.assertIn('success_count', result)

    def test_workflow_phases_in_result(self):
        """Test all post-escalation actions are in workflow result"""
        escalation_result = {
            'technique': 'GodPotato',
            'elevated_privilege': 'SYSTEM',
            'mitre_technique': 'T1548.002',
            'timestamp': 1234567890,
        }
        
        result = self.orchestrator.execute_post_escalation_chain(escalation_result)
        actions = result.get('post_escalation_actions', [])
        
        # Should have multiple actions (persistence, credentials, defense, services, injection)
        self.assertGreater(len(actions), 0, 'Should have at least one post-escalation action')

    def test_mitre_techniques_coverage(self):
        """Test MITRE techniques are properly assigned to actions"""
        result = self.orchestrator.execute_post_escalation_chain({
            'technique': 'GodPotato',
            'elevated_privilege': 'SYSTEM',
            'mitre_technique': 'T1548.002',
            'timestamp': 1234567890,
        })
        
        # Check that techniques are in output actions
        actions = result.get('post_escalation_actions', [])
        self.assertTrue(len(actions) > 0, 'Should have at least one action with MITRE technique')
        
        # All actions should have MITRE techniques assigned
        for action in actions:
            self.assertIsNotNone(action.mitre_technique)
            self.assertTrue(action.mitre_technique.startswith('T'))


class TestPostEscalationIntegration(TestCase):
    """Integration tests for post-escalation workflow"""

    @patch('agents.post_escalation.logging')  # Mock logging instead
    def test_agent_coordination_persistence(self, mock_logging):
        """Test coordination with LoaderAgent for persistence"""
        orchestrator = PostEscalationOrchestrator()
        
        escalation_result = {
            'technique': 'GodPotato',
            'elevated_privilege': 'SYSTEM',
            'mitre_technique': 'T1548.002',
            'timestamp': 1234567890,
        }
        
        result = orchestrator.execute_post_escalation_chain(escalation_result)
        
        # Verify persistence actions exist
        actions = result.get('post_escalation_actions', [])
        persistence_actions = [a for a in actions if 'persistence' in a.action_type.lower()]
        self.assertTrue(len(persistence_actions) > 0, 'Should have persistence actions')

    def test_workflow_completes_without_errors(self):
        """Test workflow executes without raising exceptions"""
        orchestrator = PostEscalationOrchestrator()
        
        try:
            result = orchestrator.execute_post_escalation_chain({
                'technique': 'GodPotato',
                'elevated_privilege': 'SYSTEM',
                'mitre_technique': 'T1548.002',
                'timestamp': 1234567890,
            })
            self.assertIsNotNone(result)
        except Exception as e:
            self.fail(f"Workflow raised exception: {str(e)}")

    def test_workflow_priority_order(self):
        """Test that capabilities are deployed in correct priority order"""
        orchestrator = PostEscalationOrchestrator()
        
        result = orchestrator.execute_post_escalation_chain({
            'technique': 'GodPotato',
            'elevated_privilege': 'SYSTEM',
            'mitre_technique': 'T1548.002',
            'timestamp': 1234567890,
        })
        
        phases = result.get('phases', {})
        phase_keys = list(phases.keys())
        
        # Verify priority order
        expected_order = [
            'persistence',
            'credential_theft',
            'defense_evasion',
            'service_control',
            'process_injection'
        ]
        
        # Check that phases appear in expected order (if they exist)
        for i, phase in enumerate(expected_order):
            if phase in phase_keys:
                actual_index = phase_keys.index(phase)
                # Phase should come before any later phases
                for later_phase in expected_order[i+1:]:
                    if later_phase in phase_keys:
                        later_index = phase_keys.index(later_phase)
                        self.assertLess(actual_index, later_index,
                                      f"{phase} should come before {later_phase}")


class TestPostEscalationCapabilities(TestCase):
    """Test specific SYSTEM-level capabilities"""

    def setUp(self):
        self.orchestrator = PostEscalationOrchestrator()

    def test_persistence_scheduled_task_capability(self):
        """Test scheduled task persistence capability"""
        actions = self.orchestrator._install_persistence()
        
        # Should have persistence actions
        self.assertIsInstance(actions, list)
        
        # Check for scheduled task action
        task_actions = [a for a in actions if 'task' in a.action_type.lower()]
        # May or may not have task actions depending on platform
        if task_actions:
            self.assertTrue(any('T1053' in a.mitre_technique for a in task_actions))

    def test_credential_lsass_dump_capability(self):
        """Test LSASS credential dump capability"""
        actions = self.orchestrator._dump_credentials()
        
        # Should have credential actions
        self.assertIsInstance(actions, list)
        
        # Check for LSASS actions
        lsass_actions = [a for a in actions if 'lsass' in a.action_type.lower()]
        # May or may not have LSASS actions depending on platform
        if lsass_actions:
            self.assertTrue(any('T1003' in a.mitre_technique for a in lsass_actions))

    def test_defense_evasion_defender_capability(self):
        """Test Windows Defender disabling capability"""
        actions = self.orchestrator._disable_defenses()
        
        # Should have defense evasion actions
        self.assertIsInstance(actions, list)
        
        # Check for Defender actions
        defender_actions = [a for a in actions if 'defender' in a.action_type.lower()]
        # May or may not have Defender actions depending on platform
        if defender_actions:
            self.assertTrue(any('T1562' in a.mitre_technique for a in defender_actions))

    def test_service_control_av_stop_capability(self):
        """Test antivirus service stopping capability"""
        actions = self.orchestrator._control_services()
        
        # Should have service actions
        self.assertIsInstance(actions, list)
        
        # Check for AV service actions
        av_actions = [a for a in actions if 'service' in a.action_type.lower()]
        # May or may not have AV actions depending on platform
        if av_actions:
            self.assertTrue(any('T1543' in a.mitre_technique for a in av_actions))

    def test_process_injection_explorer_capability(self):
        """Test explorer.exe process injection capability"""
        actions = self.orchestrator._inject_into_processes()
        
        # Should have injection actions
        self.assertIsInstance(actions, list)
        
        # Check for explorer injection actions
        explorer_actions = [a for a in actions if 'explorer' in a.action_type.lower()]
        # May or may not have explorer actions depending on platform
        if explorer_actions:
            self.assertTrue(any('T1055' in a.mitre_technique for a in explorer_actions))


class TestPostEscalationOutput(TestCase):
    """Test output formatting and logging"""

    def setUp(self):
        self.orchestrator = PostEscalationOrchestrator()

    def test_result_json_serializable(self):
        """Test that workflow result can be converted to JSON-compatible format"""
        result = self.orchestrator.execute_post_escalation_chain({
            'technique': 'GodPotato',
            'elevated_privilege': 'SYSTEM',
            'mitre_technique': 'T1548.002',
            'timestamp': 1234567890,
        })
        
        try:
            # Convert PostEscalationAction objects to dicts
            actions = result.get('post_escalation_actions', [])
            serializable_actions = [vars(a) for a in actions]
            result['post_escalation_actions'] = serializable_actions
            
            json_str = json.dumps(result)
            self.assertIsNotNone(json_str)
        except TypeError as e:
            self.fail(f"Result is not JSON serializable: {str(e)}")

    def test_result_contains_actionable_details(self):
        """Test that result contains detailed capability information"""
        result = self.orchestrator.execute_post_escalation_chain({
            'technique': 'GodPotato',
            'elevated_privilege': 'SYSTEM',
            'mitre_technique': 'T1548.002',
            'timestamp': 1234567890,
        })
        
        self.assertIn('timestamp', result)
        self.assertIn('post_escalation_actions', result)
        self.assertIn('success_count', result)
        
        # Result should indicate workflow status
        self.assertGreater(result.get('success_count', 0), 0)


if __name__ == '__main__':
    main(verbosity=2)
