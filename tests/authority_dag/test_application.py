"""Observe native calls while downstream tests read accepted matrix evidence."""
from collections import Counter
from contextlib import contextmanager
import sys

@contextmanager
def owner_calls():
    from agent.tools.analysis.native_matrix import MatrixPackage
    from agent.tools.data.cell_by_ccre_verifier import verify_cell_by_ccre
    codes = {MatrixPackage.native.__code__: 'native', verify_cell_by_ccre.__wrapped__.__code__: 'matrix'}
    counts = Counter(); previous = sys.getprofile()
    def profile(frame,event,arg):
        if event == 'call' and frame.f_code in codes: counts[codes[frame.f_code]] += 1
    sys.setprofile(profile)
    try: yield counts
    finally: sys.setprofile(previous)
