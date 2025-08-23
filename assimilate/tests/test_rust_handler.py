import unittest
import subprocess
import os

class TestRustHandler(unittest.TestCase):
    def setUp(self):
        self.rust_code = r'''
        fn main() {
            println!("Hello from Rust!");
        }
        '''
        self.rust_file = 'test_rust_program.rs'
        with open(self.rust_file, 'w') as f:
            f.write(self.rust_code)

    def tearDown(self):
        if os.path.exists(self.rust_file):
            os.remove(self.rust_file)

    def test_rust_compilation(self):
        result = subprocess.run(['rustc', self.rust_file], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, f"Compilation failed: {result.stderr}")

    def test_rust_execution(self):
        self.test_rust_compilation()
        result = subprocess.run(['./test_rust_program'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, f"Execution failed: {result.stderr}")
        self.assertIn("Hello from Rust!", result.stdout)

if __name__ == '__main__':
    unittest.main()