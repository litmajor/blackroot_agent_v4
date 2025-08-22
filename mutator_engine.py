import ast
import random
import string
import keyword
import hashlib
import time
import base64
import zlib
from datetime import datetime
import re
import os # Added for file operations in example
from typing import Optional

# --- AdvancedIdentifierRenamer (AST-based) ---
class AdvancedIdentifierRenamer(ast.NodeTransformer):
    """
    Renames local variables, function names, and class names using AST manipulation.
    This version is more robust than regex-based renaming as it understands code structure.
    """
    def __init__(self, obfuscation_chance: float = 0.7):
        self.obfuscation_chance = obfuscation_chance
        self.renaming_map = {}
        self.current_scope_renaming_map = {} # To handle scope-specific renames
        self.scope_stack = []

        self.excluded_names = self._load_excluded_names()

    def _load_excluded_names(self) -> set:
        """Loads a comprehensive list of names to exclude from renaming."""
        excluded = set(keyword.kwlist)
        excluded.update(dir(__builtins__))

        # Common standard library imports and modules
        excluded.update([
            'self', 'super', 'cls', # Special Python identifiers
            'os', 'sys', 'json', 'hashlib', 'base64', 'time', 'datetime', 'timedelta',
            'random', 'string', 're', 'zlib', 'inspect', 'threading', 'socket',
            'queue', 'redis', 'typing', 'List', 'Dict', 'Union', 'Optional', # Common library names
            # Crypto library specifics (if applicable, from previous context)
            'AES', 'get_random_bytes', 'pad', 'unpad', 'hmac', 'Cipher', 'MODE_CBC',
            # Add variables from the polymorphic header
            '__MUTANT_ID__', '__COMPRESSION_MODE__', '__PADDING_SCHEME__', '__RANDOM_SEED__', '__TIMESTAMP__',
            # Standard dunder attributes/methods that should not be renamed
            '__init__', '__new__', '__str__', '__repr__', '__call__', '__len__', '__getitem__',
            '__setitem__', '__delitem__', '__contains__', '__enter__', '__exit__',
            '__main__', '__file__', '__name__', '__loader__', '__spec__', '__doc__',
            '__package__', '__cached__', '__path__', '__slots__', '__weakref__',
            '__hash__', '__eq__', '__ne__', '__lt__', '__le__', '__gt__', '__ge__',
            '__add__', '__sub__', '__mul__', '__truediv__', '__floordiv__', '__mod__',
            '__pow__', '__and__', '__or__', '__xor__', '__lshift__', '__rshift__',
            '__iadd__', '__isub__', # etc. for in-place ops
            '__getattr__', '__setattr__', '__delattr__', '__dir__', '__format__',
            '__getattribute__', '__setformat__', '__sizeof__', '__subclasshook__',
            '__missing__', # For dicts
            'copy', 'deepcopy' # Common method names
        ])
        return excluded

    def _generate_random_string(self, length: int) -> str:
        """Generates a random string suitable for an identifier."""
        first_char = random.choice(string.ascii_lowercase + '_')
        rest_chars = ''.join(random.choice(string.ascii_lowercase + string.digits + '_') for _ in range(length - 1))
        new_name = first_char + rest_chars
        # Ensure new name isn't a keyword, excluded, or already used in current renaming map
        while new_name in self.excluded_names or new_name in self.current_scope_renaming_map.values():
            first_char = random.choice(string.ascii_lowercase + '_')
            rest_chars = ''.join(random.choice(string.ascii_lowercase + string.digits + '_') for _ in range(length - 1))
            new_name = first_char + rest_chars
        return new_name

    def _get_new_name(self, original_name: str) -> str:
        """Helper to get or create a new name for an identifier within the current scope."""
        if original_name in self.current_scope_renaming_map:
            return self.current_scope_renaming_map[original_name]

        # Decide if we want to rename this specific identifier based on chance and exclusions
        # Names that are all caps are often constants and are typically not renamed.
        if original_name not in self.excluded_names and not original_name.isupper() and random.random() < self.obfuscation_chance:
            new_name = self._generate_random_string(random.randint(8, 15))
            self.current_scope_renaming_map[original_name] = new_name
            # print(f"Renamed '{original_name}' to '{new_name}' in scope {self.scope_stack[-1] if self.scope_stack else 'global'}") # Debugging
            return new_name
        return original_name # Don't rename if excluded, all caps, or chance not met

    def visit(self, node):
        """Override visit to manage scope for renaming."""
        # Push current scope onto stack
        self.scope_stack.append(self.current_scope_renaming_map)
        self.current_scope_renaming_map = {} # New scope for current node's children

        # Perform the actual visit
        new_node = super().visit(node)

        # Pop scope from stack
        self.current_scope_renaming_map = self.scope_stack.pop()
        return new_node

    def visit_Name(self, node: ast.Name) -> ast.Name:
        """
        Visits Name nodes (variables, function calls, etc.).
        Renames based on context (Store/Load).
        """
        original_name = node.id

        # Only rename if it's a variable being defined (Store context) or a parameter (Param context)
        # For Load contexts, we rely on the name already being in current_scope_renaming_map or a parent scope
        if isinstance(node.ctx, (ast.Store, ast.Param)):
            node.id = self._get_new_name(original_name)
        elif isinstance(node.ctx, ast.Load):
            # Check current scope first
            if original_name in self.current_scope_renaming_map:
                node.id = self.current_scope_renaming_map[original_name]
            else:
                # Check parent scopes
                for scope_map in reversed(self.scope_stack):
                    if original_name in scope_map:
                        node.id = scope_map[original_name]
                        break

        # self.generic_visit(node) # Handled by the overridden visit method
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Visits function definitions."""
        original_name = node.name

        # Rename the function itself (its definition)
        node.name = self._get_new_name(original_name)

        # Rename arguments within the function's new scope
        for arg in node.args.args:
            arg.arg = self._get_new_name(arg.arg)
        if node.args.vararg: # *args
            node.args.vararg.arg = self._get_new_name(node.args.vararg.arg)
        if node.args.kwarg: # **kwargs
            node.args.kwarg.arg = self._get_new_name(node.args.kwarg.arg)

        # The `visit` method (overridden) will call generic_visit for the body
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        """Visits async function definitions (same logic as regular functions)."""
        original_name = node.name

        # Rename the async function itself
        node.name = self._get_new_name(original_name)

        # Rename arguments within the async function's new scope
        for arg in node.args.args:
            arg.arg = self._get_new_name(arg.arg)
        if node.args.vararg:  # *args
            node.args.vararg.arg = self._get_new_name(node.args.vararg.arg)
        if node.args.kwarg:  # **kwargs
            node.args.kwarg.arg = self._get_new_name(node.args.kwarg.arg)

        # The overridden visit method will call generic_visit for the body
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        """Visits class definitions."""
        original_name = node.name

        # Rename the class itself
        node.name = self._get_new_name(original_name)

        # The `visit` method (overridden) will handle traversing into methods and class attributes,
        # which will be handled by visit_FunctionDef and visit_Name respectively.
        return node

    def rename_identifiers(self, code: str) -> str:
        if random.random() >= self.obfuscation_chance:
            return code

        try:
            tree = ast.parse(code)
            self.renaming_map = {} # Reset for a new renaming operation
            self.current_scope_renaming_map = {}
            self.scope_stack = []

            new_tree = self.visit(tree) # Start the AST traversal and modification
            ast.fix_missing_locations(new_tree) # Update line/col info after modification
            return ast.unparse(new_tree) # Convert the modified AST back to code
        except SyntaxError as e:
            print(f"Error parsing code for identifier renaming: {e}")
            return code # Return original code if parsing fails
        except AttributeError:
            print("Warning: ast.unparse not available (requires Python 3.9+). Skipping identifier renaming.")
            return code # Fallback for older Python versions

# --- MutatorEngine ---
class MutatorEngine:
    def evolve_composition(self, output_path: Optional[str] = None):
        """
        Applies advanced polymorphic transformations to the source code and writes to output_path if provided.
        Returns the mutated code as a string.
        """
        with open(self.source_path, 'r', encoding='utf-8') as f:
            code = f.read()
        mutated = self._polymorphic_transform(code)
        # Optionally, apply further randomization or mutation here
        mutated += f"\n# Evolved at {datetime.utcnow().isoformat()}\n"
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(mutated)
            print(f"[🧬] Evolved composition saved to {output_path}")
        return mutated
    """A class to create mutated, polymorphic copies of a source file."""
    def __init__(self, source_path: str, obfuscation_chance: float = 0.3):
        self.source_path = source_path
        self.obfuscation_chance = obfuscation_chance # Probability for applying certain obfuscations

        # Initialize the AST-based renamer and add its excluded names
        self.ast_renamer = AdvancedIdentifierRenamer(self.obfuscation_chance)


    def mutate_and_replicate(self, output_path: str):
        """Reads the source, applies transformations, and writes a new variant."""
        with open(self.source_path, 'r', encoding='utf-8') as f:
            code = f.read()
        mutated = self._polymorphic_transform(code)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(mutated)
        print(f"[🧬] Mutated variant saved to {output_path}")

    def _generate_random_string(self, length: int = 8) -> str:
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
        # Avoid injecting inside multiline strings, comments, or right after imports/defs
        injection_points = []
        in_multiline_string = False
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            if '"""' in stripped_line or "'''" in stripped_line:
                in_multiline_string = not in_multiline_string

            if not in_multiline_string and \
               not stripped_line.startswith(('import ', 'from ', 'def ', 'class ', '#')) and \
               stripped_line != "":
                injection_points.append(i)

        if not injection_points: # If no suitable points, append to end
            injection_points = [len(lines)]

        num_injections = random.randint(1, 3) # Inject 1 to 3 dummy elements
        for _ in range(num_injections):
            if random.random() < self.obfuscation_chance:
                insert_idx = random.choice(injection_points)

                # Determine indentation from the line at insert_idx or the previous one
                indent = ""
                if insert_idx < len(lines):
                    match = re.match(r"^\s*", lines[insert_idx])
                    indent = match.group(0) if match else ""
                elif insert_idx > 0:
                    match = re.match(r"^\s*", lines[insert_idx - 1])
                    indent = match.group(0) if match else ""


                dummy_choice = random.choice(dummy_functions)
                dummy_name = self._generate_random_string(random.randint(5, 12))

                if "{value}" in dummy_choice:
                    dummy_code = dummy_choice.format(name=dummy_name, value=random.choice(dummy_values))
                else:
                    dummy_code = dummy_choice.format(name=dummy_name)

                # Add indentation and ensure newlines
                dummy_code_indented = "\n".join([indent + line for line in dummy_code.splitlines()]) + "\n\n"

                lines.insert(insert_idx, dummy_code_indented)
                # Adjust subsequent injection points to account for inserted lines
                injection_points = [p + dummy_code_indented.count('\n') for p in injection_points if p >= insert_idx]


        return "\n".join(lines)

    def _rename_identifiers(self, code: str) -> str:
        """Renames local variables and simple function names using AST."""
        return self.ast_renamer.rename_identifiers(code)

    def _string_obfuscation(self, code: str) -> str:
        """Obfuscates simple strings by encoding them (e.g., base64) and decoding at runtime."""
        if random.random() >= self.obfuscation_chance:
            return code

        # Updated regex to be slightly more conservative to avoid common breakages.
        # It still targets standalone strings but tries to avoid attributes or function calls.
        # This is still a heuristic. For truly safe string obfuscation, an AST approach is best.
        string_pattern = r"(?<![.'\"_a-zA-Z0-9])(['\"])(.*?)(?<!\\)\1(?![.'\"_a-zA-Z0-9])"

        def replace_string(match):
            original_string = match.group(2)
            quote = match.group(1)

            # Avoid obfuscating very short strings, numeric strings, or empty strings
            if len(original_string) < 4 or original_string.isdigit() or not original_string.strip():
                return match.group(0) # Return original if too short/numeric/empty

            # Avoid obfuscating common variable names or simple values that might coincidentally match
            # This list can be expanded based on common code patterns.
            if original_string in ["True", "False", "None", "0", "1", "pass", "None", "self", "cls"]:
                return match.group(0)

            obf_method = random.choice(["base64", "zlib_b64"])

            if obf_method == "base64":
                encoded_bytes = base64.b64encode(original_string.encode()).decode()
                # Ensure base64 and decode are imported/available.
                return f"base64.b64decode('{encoded_bytes}'.encode()).decode()"
            elif obf_method == "zlib_b64":
                compressed_bytes = zlib.compress(original_string.encode())
                encoded_bytes = base64.b64encode(compressed_bytes).decode()
                # Ensure zlib, base64 and decode are imported/available.
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
                    # Separate nodes into fixed-position (like docstrings, __init__) and reorderable
                    fixed_position_nodes = []
                    reorderable_body = []
                    
                    # Handle docstring: If the very first node is an Expr with a string constant, treat as docstring.
                    if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, (ast.Str, ast.Constant)):
                        fixed_position_nodes.append(node.body[0])
                        remaining_body = node.body[1:]
                    else:
                        remaining_body = node.body

                    for item in remaining_body:
                        # __init__ should generally stay in place for proper object initialization
                        if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                            fixed_position_nodes.append(item)
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
        except Exception as e:
            print(f"An unexpected error occurred during class member reordering: {e}")
            return code # Catch any other unexpected errors

    def _polymorphic_transform(self, code: str) -> str:
        """Applies a series of polymorphic transformations."""

        # 1. Inject header with random values (original functionality)
        lines = code.splitlines()
        injected_header = [
            "# POLYMORPHIC HEADER",
            f"__MUTANT_ID__ = '{hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]}'",
            f"__COMPRESSION_MODE__ = '{random.choice(['zlib', 'none', 'gzip', 'brotli'])}' # Dummy flag",
            f"__PADDING_SCHEME__ = '{random.choice(['PKCS7', 'NoPadding', 'ZeroPadding'])}' # Dummy flag",
            f"__RANDOM_SEED__ = {random.randint(0, 99999999)}",
            f"__TIMESTAMP__ = '{datetime.utcnow().isoformat()}'"
        ]

        # Find a suitable injection point for the header (e.g., before the first non-import/comment line)
        insert_idx = 0
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            if stripped_line and not stripped_line.startswith(('#', 'import ', 'from ')):
                insert_idx = i
                break
        else: # If only comments/imports, add at the very top
            insert_idx = 0

        code_parts = lines[:insert_idx] + injected_header + [""] + lines[insert_idx:]
        code = "\n".join(code_parts)


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


# --- Example Usage ---
if __name__ == "__main__":
    import textwrap

    # Create a dummy source file for the MutatorEngine to read
    original_source_file = "original_code.py"
    sample_source_code = textwrap.dedent("""
    import os
    import sys
    import hashlib # Used for hashing
    import base64
    import zlib # Added for string obfuscation

    GLOBAL_CONSTANT = "This is a global constant" # Should not be renamed
    _internal_global_var = 123 # Should be renamed

    class VaultManager:
        \"\"\"Manages secure storage of data.\"\"\"
        __SECRET_KEY = "my_super_secret" # This should not be renamed easily due to convention/exclusion
        _instance = None # Class variable

        def __new__(cls, vault_name_param):
            if cls._instance is None:
                cls._instance = super(VaultManager, cls).__new__(cls)
                cls._instance.name = vault_name_param # Parameter is renamed
                cls._instance.data = {}
            return cls._instance

        def store_data(self, key_name_param, value_data_param):
            # This is a comment about storing data
            local_hash = hashlib.sha256(key_name_param.encode()).hexdigest()
            encrypted_value = base64.b64encode(value_data_param.encode()).decode()
            self.data[local_hash] = encrypted_value
            print(f"Data stored for {key_name_param}") # String literal
            ANOTHER_CONSTANT = 999 # Should not be renamed (local constant)

        def retrieve_data(self, key_name_param):
            local_hash = hashlib.sha256(key_name_param.encode()).hexdigest()
            encrypted_value = self.data.get(local_hash)
            if encrypted_value:
                decrypted_value = base64.b64decode(encrypted_value.encode()).decode()
                return decrypted_value
            return None

        def _cleanup_old_entries(self): # Private method, should be renamed
            print("Cleaning up old entries...")
            # Dummy cleanup logic
            temp_count = 123
            for i in range(temp_count):
                _ = i # Unused variable, potentially renamed

        def _another_utility_method(self, arg1, arg2):
            return arg1 + arg2 + "calculated" # String literal
            
    def main_execution_flow():
        vault_obj = VaultManager("MySecureVault") # Class name and variable should be renamed
        user_name = "Alice" # String literal and variable
        user_password = "password123" # String literal and variable
        vault_obj.store_data(user_name, user_password) # Method call
        retrieved_data = vault_obj.retrieve_data(user_name)
        if retrieved_data:
            print(f"Retrieved data: {retrieved_data}")
        else:
            print("Data not found.")
        vault_obj._cleanup_old_entries() # Call private method

    async def async_dummy_func(data_input): # Async function, should be renamed
        await_result = data_input + "processed"
        return await_result

    if __name__ == "__main__":
        main_execution_flow()
        another_var_for_test = "this_is_a_test_string_for_obfuscation"
        print(another_var_for_test)
        print("This is another string literal example.")
        some_int_value = 42
        some_bool_value = True
        # A short string that should NOT be obfuscated
        print("OK")
        print("ID:123")
        # Call the async dummy func
        import asyncio
        asyncio.run(async_dummy_func("async_test"))
        """)

    with open(original_source_file, 'w', encoding='utf-8') as f:
        f.write(sample_source_code)

    output_file_name = "mutated_code_final.py"

    print("--- Original Code ---")
    print(sample_source_code)

    mutator = MutatorEngine(source_path=original_source_file, obfuscation_chance=0.8) # Higher chance for demo

    print("\n--- Applying transformations and writing mutated file ---")
    mutator.mutate_and_replicate(output_file_name)

    print(f"\nObfuscated code written to {output_file_name}")

    # Optional: Read and print the mutated code to verify
    print("\n--- Content of Mutated Code ---")
    with open(output_file_name, 'r', encoding='utf-8') as f:
        print(f.read())

    # Optional: Try to execute the obfuscated code (be cautious with untrusted code)
    # Ensure you have asyncio if you enable async_dummy_func
    # print("\n--- Executing Obfuscated Code (if Python 3.9+) ---")
    # import subprocess
    # try:
    #     result = subprocess.run(['python', output_file_name], capture_output=True, text=True, check=True)
    #     print(result.stdout)
    #     if result.stderr:
    #         print("Errors during execution:", result.stderr)
    # except subprocess.CalledProcessError as e:
    #     print(f"Execution failed with return code {e.returncode}:")
    #     print(e.stdout)
    #     print(e.stderr)
    # except FileNotFoundError:
    #     print("Python executable not found. Make sure Python is in your PATH.")
    # except Exception as e:
    #     print(f"An unexpected error occurred during execution: {e}")

    # Clean up dummy source file
    if os.path.exists(original_source_file):
        os.remove(original_source_file)
