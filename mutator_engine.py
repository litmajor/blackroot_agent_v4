import ast
import random
import string
import keyword
import hashlib
import time
import base64
import zlib
from datetime import datetime, timedelta
import re
import string

# Re-define MutatorEngine to include the AST-based reorderer
class MutatorEngine: # Renamed from Mutator for clarity
    """A class to create mutated, polymorphic copies of a source file."""
    def __init__(self, source_path):
        self.source_path = source_path
        self.obfuscation_chance = 0.3 # Probability for applying certain obfuscations

        # Initialize the AST-based renamer and add its excluded names
        self.ast_renamer = AdvancedIdentifierRenamer(self.obfuscation_chance)


    def mutate_and_replicate(self, output_path):
        """Reads the source, applies transformations, and writes a new variant."""
        with open(self.source_path, 'r', encoding='utf-8') as f:
            code = f.read()
        mutated = self._polymorphic_transform(code)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(mutated)
        print(f"[🧬] Mutated variant saved to {output_path}")

    def _generate_random_string(self, length=8):
        """Generates a random alphanumeric string."""
        chars = string.ascii_letters + string.digits
        return ''.join(random.choice(chars) for _ in range(length))

    def _inject_dummy_code(self, code: str) -> str:
        """Injects random, non-functional code snippets."""
        dummy_functions = [
            "def {name}():\n\tpass\n",
            "def {name}(_val):\n\treturn _val * random.random()\n",
            "class {name}:\n\tdef __init__(self):\n\t\tself.a = 0\n",
            "__dummy_var_{name}__ = {value}\n"
        ]
        dummy_values = [random.randint(0, 1000), f"'{self._generate_random_string()}'", "True", "False", "None"]

        lines = code.splitlines()
        # Find potential injection points (e.g., end of a block, before a function)
        injection_points = [i for i, line in enumerate(lines) if not line.strip().startswith((' ', '\t', '#')) and line.strip() != ""]
        
        if not injection_points: # If no suitable points, append to end
            injection_points = [len(lines)]

        num_injections = random.randint(1, 3) # Inject 1 to 3 dummy elements
        for _ in range(num_injections):
            if random.random() < self.obfuscation_chance:
                # Choose a random line to inject before
                insert_idx = random.choice(injection_points)
                
                # Indent based on the line a bove (simple heuristic)
                indent = re.match(r"^\s*", lines[min(insert_idx, len(lines)-1)]).group(0) if lines else ""

                dummy_choice = random.choice(dummy_functions)
                
                # For dummy variables/functions, avoid clashes with real names
                dummy_name = self._generate_random_string(random.randint(5, 12))
                
                if "{value}" in dummy_choice:
                    dummy_code = dummy_choice.format(name=dummy_name, value=random.choice(dummy_values))
                else:
                    dummy_code = dummy_choice.format(name=dummy_name)

                # Add indentation and ensure newlines
                dummy_code_indented = "\n".join([indent + line for line in dummy_code.splitlines()]) + "\n\n"
                
                lines.insert(insert_idx, dummy_code_indented)
                # Update injection points after insertion
                # This logic is a bit flawed for multiple insertions, but for small N, it's okay.
                # A more robust solution would re-calculate injection_points or use AST for insertion.
                injection_points = [p + dummy_code_indented.count('\n') for p in injection_points if p >= insert_idx]


        return "\n".join(lines)

    def _rename_identifiers(self, code: str) -> str:
        """Renames local variables and simple function names using AST."""
        # Use the AST-based renamer here
        return self.ast_renamer.rename_identifiers(code)

    def _string_obfuscation(self, code: str) -> str:
        """Obfuscates simple strings by encoding them (e.g., base64) and decoding at runtime."""
        if random.random() >= self.obfuscation_chance:
            return code

        # Regex to find simple quoted strings that are not part of imports or definitions (heuristic)
        # This is a basic pattern and would need refinement for production use to avoid breaking code.
        string_pattern = r"(?<!(?:import|from)\s+)(?<![a-zA-Z_]\s*=)\s*?(['\"])(.*?)\1"
        
        def replace_string(match):
            original_string = match.group(2)
            quote = match.group(1)

            # Avoid obfuscating very short strings or numeric strings
            if len(original_string) < 5 or original_string.isdigit():
                return match.group(0) # Return original if too short/numeric

            # Choose an obfuscation method
            obf_method = random.choice(["base64", "zlib_b64"])

            if obf_method == "base64":
                encoded_bytes = base64.b64encode(original_string.encode()).decode()
                # Inject runtime decoding
                return f"base64.b64decode('{encoded_bytes}'.encode()).decode()"
            elif obf_method == "zlib_b64":
                compressed_bytes = zlib.compress(original_string.encode())
                encoded_bytes = base64.b64encode(compressed_bytes).decode()
                # Inject runtime decoding
                return f"zlib.decompress(base64.b64decode('{encoded_bytes}'.encode())).decode()"
            else:
                return match.group(0) # Fallback

        mutated_code = re.sub(string_pattern, replace_string, code)
        return mutated_code

    def _add_random_comments_and_newlines(self, code: str) -> str:
        """Adds random comments and blank lines to change file structure."""
        lines = code.splitlines()
        new_lines = []
        for line in lines:
            new_lines.append(line)
            if random.random() < 0.15: # 15% chance to add a blank line
                new_lines.append("")
            if random.random() < 0.10: # 10% chance to add a random comment
                new_lines.append(f"# {self._generate_random_string(random.randint(10, 30))}")
        return "\n".join(new_lines)

    # === NEW: AST-based Class Member Reorderer ===
    def _reorder_class_members(self, code: str) -> str:
        """
        Reorders methods and attributes within classes using AST.
        This provides structural polymorphism without breaking functionality.
        """
        if random.random() >= self.obfuscation_chance:
            return code

        try:
            tree = ast.parse(code)
            
            class ClassMemberReorderer(ast.NodeTransformer):
                def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
                    """
                    Visits each class definition and reorders its body elements.
                    """
                    # Filter out non-reorderable nodes (e.g., docstrings if they're not explicitly ast.Expr)
                    # For simplicity, we assume all top-level statements within a class body can be reordered
                    # if they are not special methods like __init__ or __new__ that might have dependencies.
                    # A more advanced reorderer would perform dependency analysis.

                    reorderable_body = []
                    fixed_position_nodes                               = [] # e.g., docstrings, certain __dunder__ methods

                    # Separate nodes into reorderable and fixed
                    for item in node.body:
                        # Simple rule: __init__ should probably stay first for class setup
                        # And docstrings (ast.Expr with a string value) should typically stay at the very beginning
                        if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                            fixed_position_nodes.append(item)
                        elif isinstance(item, ast.Expr) and isinstance(item.value, ast.Str):
                            # Docstrings often come first, so we might want to keep them fixed or move them carefully.
                            # For now, let's keep them fixed if they are the very first node.
                            if not reorderable_body and not fixed_position_nodes: # If it's the first thing
                                fixed_position_nodes.append(item)
                            else:
                                reorderable_body.append(item) # Treat as reorderable if not initial docstring
                        else:
                            reorderable_body.append(item)
                    
                    # Shuffle the reorderable parts
                    random.shuffle(reorderable_body)
                    
                    # Reconstruct the body
                    node.body = fixed_position_nodes + reorderable_body

                    # Continue visiting children nodes to apply other transformations recursively
                    self.generic_visit(node)
                    return node

            transformer = ClassMemberReorderer()
            new_tree = transformer.visit(tree)
            
            # Use ast.fix_missing_locations to update line numbers and column offsets
            # This is important after modifying the AST to ensure unparsing works correctly.
            ast.fix_missing_locations(new_tree)

            return ast.unparse(new_tree) # Requires Python 3.9+
        except SyntaxError as e:
            print(f"Error parsing code for class member reordering: {e}")
            return code # Return original code if parsing fails
        except AttributeError:
            print("Warning: ast.unparse not available (requires Python 3.9+). Skipping class member reordering.")
            return code # Fallback for older Python versions

    def _polymorphic_transform(self, code):
        """Applies a series of polymorphic transformations."""
        
        # 1. Inject header with random values (original functionality)
        lines = code.splitlines()
        injected_header = [
            "# POLYMORPHIC HEADER",
            f"__MUTANT_ID__ = '{hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]}'",
            f"__COMPRESSION_MODE__ = '{random.choice(['zlib', 'none', 'gzip', 'brotli'])}' # Dummy flag, no actual compression applied by mutator",
            f"__PADDING_SCHEME__ = '{random.choice(['PKCS7', 'NoPadding'])}' # Dummy flag, no actual padding logic changed",
            f"__RANDOM_SEED__ = {random.randint(0, 99999999)}",
            f"__TIMESTAMP__ = '{datetime.utcnow().isoformat()}'"
        ]

        if '__main__' in code:
            # Inject before the __main__ block
            main_block_index = -1
            for i, line in enumerate(lines):
                if '__main__' in line and line.strip().startswith('if'):
                    main_block_index = i
                    break
            
            if main_block_index != -1:
                code_parts = lines[:main_block_index] + injected_header + [""] + lines[main_block_index:]
                code = "\n".join(code_parts)
            else: # Fallback if __main__ is not found as expected
                code = "\n".join(injected_header + [""] + lines)
        else:
            code = "\n".join(injected_header + [""] + lines)

        # 2. Add random comments and blank lines
        code = self._add_random_comments_and_newlines(code)

        # 3. Inject dummy code (functions, classes, variables)
        code = self._inject_dummy_code(code)

        # 4. Obfuscate strings (requires runtime import of base64/zlib, handled by the code itself)
        code = self._string_obfuscation(code)

        # 5. Rename identifiers (variables, functions). NOW USING AST.
        code = self._rename_identifiers(code) # This now calls the AST-based renamer

        # 6. Reorder class members (NEW AST-BASED FUNCTION)
        code = self._reorder_class_members(code)

        return code


# === ADVANCED IDENTIFIER RENAMER (AST-based) 
# This needs to be defined outside or integrated carefully. 
# For simplicity, I'm putting it here as a separate class that MutatorEngine can use.

class AdvancedIdentifierRenamer(ast.NodeTransformer):
    """
    Renames local variables, function names, and class names using AST manipulation.
    This version is more robust than regex-based renaming as it understands code structure.
    """
    def __init__(self, obfuscation_chance: float = 1.0):
        self.obfuscation_chance = obfuscation_chance
        self.renaming_map = {}
        self.seen_identifiers = set() # To track all identifiers in the code to avoid clashes
        
        # Add all Python keywords and built-ins to excluded names
        self.excluded_names = set(keyword.kwlist)
        self.excluded_names.update(dir(__builtins__))
        
        # Add specific project-related exclusions
        self.excluded_names.update([
            'self', 'super', 'cls', 'os', 'json', 'hashlib', 'base64', 'time', 'datetime', 'timedelta',
            'AES', 'get_random_bytes', 'pad', 'unpad', 'hmac', 'threading', 'socket', 'zlib', 'inspect',
            'random', 'string', 're', 'VAULT_DIR', 'AES_BLOCK_SIZE', 'BlackVault', 'GhostLayer',
            'blackroot_core', 'VaultSyncServer', 'VaultSyncClient', 'MutatorEngine',
            '__init__', '__main__', '__file__', '__name__', '__loader__', '__spec__', '__doc__',
            '__package__', '__cached__', '__path__', # Standard dunder attributes/methods
            # Add variables from the polymorphic header if they should never be renamed
            '__MUTANT_ID__', '__COMPRESSION_MODE__', '__PADDING_SCHEME__', '__RANDOM_SEED__', '__TIMESTAMP__'
        ])


    def _generate_random_string(self, length: int) -> str:
        """Generates a random string suitable for an identifier."""
        first_char = random.choice(string.ascii_lowercase + '_')
        rest_chars = ''.join(random.choice(string.ascii_lowercase + string.digits + '_') for _ in range(length - 1))
        new_name = first_char + rest_chars
        # Ensure new name isn't a keyword, excluded, or already used
        while new_name in self.excluded_names or new_name in self.seen_identifiers or new_name in self.renaming_map.values():
            first_char = random.choice(string.ascii_lowercase + '_')
            rest_chars = ''.join(random.choice(string.ascii_lowercase + string.digits + '_') for _ in range(length - 1))
            new_name = first_char + rest_chars
        return new_name

    def _rename_node_name(self, old_name: str) -> str:
        """Helper to get or create a new name for an identifier."""
        if old_name in self.renaming_map:
            return self.renaming_map[old_name]
        
        # Decide if we want to rename this specific identifier based on chance and exclusions
        if old_name not in self.excluded_names and not old_name.isupper() and random.random() < 0.7:
            new_name = self._generate_random_string(random.randint(8, 15))
            self.renaming_map[old_name] = new_name
            # print(f"Renamed '{old_name}' to '{new_name}'") # Uncomment for debugging
            return new_name
        return old_name # Don't rename if excluded or chance not met

    def visit_Name(self, node: ast.Name) -> ast.Name:
        """
        Visits Name nodes (variables, function calls, etc.).
        Renames based on context (Store/Load).
        """
        original_name = node.id
        self.seen_identifiers.add(original_name)

        # Only rename if it's a variable being defined (Store context) or a parameter (Param context)
        # For Load contexts, we rely on the name already being in renaming_map from its definition
        if isinstance(node.ctx, (ast.Store, ast.Param)):
            node.id = self._rename_node_name(original_name)
        elif isinstance(node.ctx, ast.Load):
            # If the original name was renamed, use the new name
            if original_name in self.renaming_map:
                node.id = self.renaming_map[original_name]
        
        self.generic_visit(node) # Continue traversing children
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Visits function definitions."""
        original_name = node.name
        self.seen_identifiers.add(original_name)
        
        # Rename the function itself
        node.name = self._rename_node_name(original_name)

        # Arguments are ast.arg nodes, their names are in arg.arg
        for arg in node.args.args:
            arg.arg = self._rename_node_name(arg.arg)
        if node.args.vararg: # *args
            node.args.vararg.arg = self._rename_node_name(node.args.vararg.arg)
        if node.args.kwarg: # **kwargs
            node.args.kwarg.arg = self._rename_node_name(node.args.kwarg.arg)

        self.generic_visit(node) # Visit the function body to rename local variables
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        # Same logic as visit_FunctionDef for async functions
        return self.visit_AsyncFunctionDef(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        """Visits class definitions."""
        original_name = node.name
        self.seen_identifiers.add(original_name)

        # Rename the class itself
        node.name = self._rename_node_name(original_name)

        # The generic_visit will traverse into methods and class attributes,
        # which will be handled by visit_FunctionDef and visit_Name respectively.
        self.generic_visit(node)
        return node

    def rename_identifiers(self, code: str) -> str:
        if random.random() >= self.obfuscation_chance:
            return code

        try:
            tree = ast.parse(code)
            self.visit(tree) # Start the AST traversal and modification
            ast.fix_missing_locations(tree) # Update line/col info after modification
            return ast.unparse(tree) # Convert the modified AST back to code
        except SyntaxError as e:
            print(f"Error parsing code for identifier renaming: {e}")
            return code # Return original code if parsing fails
        except AttributeError:
            print("Warning: ast.unparse not available (requires Python 3.9+). Skipping identifier renaming.")
            return code # Fallback for older Python versions

# Example Usage:
if __name__ == "__main__":
    import textwrap
    sample_source_code = textwrap.dedent("""
    import os
    import sys
    import hashlib # Used for hashing

    class VaultManager:
        __SECRET_KEY = "my_super_secret" # This should not be renamed easily
        _instance = None # Class variable

        def __new__(cls, vault_name):
            if cls._instance is None:
                cls._instance = super(VaultManager, cls).__new__(cls)
                cls._instance.name = vault_name
                cls._instance.data = {}
            return cls._instance

        def store_data(self, key_name, value_data):
            # This is a comment about storing data
            hashed_key = hashlib.sha256(key_name.encode()).hexdigest()
            encrypted_value = base64.b64encode(value_data.encode()).decode()
            self.data[hashed_key] = encrypted_value
            print(f"Data stored for {key_name}")

        def retrieve_data(self, key_name):
            hashed_key = hashlib.sha256(key_name.encode()).hexdigest()
            encrypted_value = self.data.get(hashed_key)
            if encrypted_value:
                decrypted_value = base64.b64decode(encrypted_value.encode()).decode()
                return decrypted_value
            return None

        def _cleanup_old_entries(self): # Private method
            print("Cleaning up old entries...")
            # Dummy cleanup logic
            temp_var = 123
            for i in range(temp_var):
                _ = i # Unused variable
            
    def main_execution_flow():
        vault = VaultManager("MySecureVault")
        user_name = "Alice"
        user_password = "password123"
        vault.store_data(user_name, user_password)
        retrieved = vault.retrieve_data(user_name)
        if retrieved:
            print(f"Retrieved data: {retrieved}")
        else:
            print("Data not found.")
        vault._cleanup_old_entries() # Call private method

    if __name__ == "__main__":
        main_execution_flow()
        another_var = "test_string_to_obfuscate"
        print(another_var)
        print("This is another string literal.")
        some_value = 42 # A simple value
        """
    )

    output_file_name = "mutated_code_example.py"
    
    mutator = MutatorEngine(source_path="dummy_path.py") # Source path doesn't matter for this example
    # We replace the source_path read with our sample_source_code for testing
    
    print("--- Original Code ---")
    print(sample_source_code)

    # Perform transformations manually for testing
    mutated_code = sample_source_code
    
    # Apply transformations in order for demonstration
    print("\n--- Applying transformations ---")
    
    # Inject header
    lines = mutated_code.splitlines()
    injected_header = [
        "# POLYMORPHIC HEADER (Manual Test)",
        f"__MUTANT_ID__ = '{hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]}'",
        f"__COMPRESSION_MODE__ = '{random.choice(['zlib', 'none', 'gzip', 'brotli'])}'",
        f"__RANDOM_SEED__ = {random.randint(0, 99999999)}",
        f"__TIMESTAMP__ = '{datetime.utcnow().isoformat()}'"
    ]
    main_block_index = -1
    for i, line in enumerate(lines):
        if '__main__' in line and line.strip().startswith('if'):
            main_block_index = i
            break
    if main_block_index != -1:
        code_parts = lines[:main_block_index] + injected_header + [""] + lines[main_block_index:]
        mutated_code = "\n".join(code_parts)
    else:
        mutated_code = "\n".join(injected_header + [""] + lines)
    print("Header injected.")

    # Add comments/newlines
    mutated_code = mutator._add_random_comments_and_newlines(mutated_code)
    print("Comments and newlines added.")

    # Inject dummy code
    mutated_code = mutator._inject_dummy_code(mutated_code)
    print("Dummy code injected.")

    # String obfuscation
    mutated_code = mutator._string_obfuscation(mutated_code)
    print("Strings obfuscated.")

    # Identifier renaming (uses AST)
    mutated_code = mutator._rename_identifiers(mutated_code)
    print("Identifiers renamed (via AST).")

    # Class member reordering (NEW - uses AST)
    mutated_code = mutator._reorder_class_members(mutated_code)
    print("Class members reordered (via AST).")

    print("\n--- Obfuscated Code ---")
    print(mutated_code)

    # Optional: Write to file and try to execute (for simple cases)
    with open(output_file_name, 'w', encoding='utf-8') as f:
        f.write(mutated_code)
    print(f"\nObfuscated code written to {output_file_name}")

    # To run the obfuscated code (be cautious with untrusted code)
    # import subprocess
    # try:
    #     print("\n--- Executing Obfuscated Code ---")
    #     result = subprocess.run(['python', output_file_name], capture_output=True, text=True, check=True)
    #     print(result.stdout)
    #     if result.stderr:
    #         print("Errors during execution:", result.stderr)
    # except subprocess.CalledProcessError as e:
    #     print(f"Execution failed: {e.returncode}")
    #     print(e.stdout)
    #     print(e.stderr)