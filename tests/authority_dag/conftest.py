import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

from bam_fragments.conftest import bam_factory
from external_fragments.conftest import source_factory
