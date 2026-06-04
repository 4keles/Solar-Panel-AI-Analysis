import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path.cwd()))

from tests.integration.test_active_learning import test_active_learning_flow
import tempfile

with tempfile.TemporaryDirectory() as d:
    test_active_learning_flow(Path(d))
    print("Test passed!")
