# src/types/index.py

from typing import Any, Dict, List, Union

# Define a type for software code objects
SoftwareCode = Dict[str, Union[str, bytes]]

# Define a type for the context passed during execution
ExecutionContext = Dict[str, Any]

# Define a type for the result of executing a software handler
ExecutionResult = Dict[str, Any]

# Define a type for the trust evaluation score
TrustScore = float

# Define a type for the software handler interface
class SoftwareHandler:
    def validate(self, code: SoftwareCode) -> bool:
        pass

    def execute(self, code_obj: SoftwareCode, context: ExecutionContext) -> ExecutionResult:
        pass

# Define a type for the reflex tree structure
ReflexTree = List[Dict[str, Any]]