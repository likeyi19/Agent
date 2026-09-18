import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
# Reuse the unchanged, accepted M14.6 scientific fixture.
from fragment_features.conftest import factory
