"""Existing trusted R installations shared by research adapters and direct calls."""
import os
from pathlib import Path

R_RUNTIMES = {
    'Signac': ('AGENT_SIGNAC_RSCRIPT', 'signac-1'),
    'chromVAR': ('AGENT_CHROMVAR_RSCRIPT', 'chromvar'),
    'motifmatchr': ('AGENT_CHROMVAR_RSCRIPT', 'chromvar'),
    'cicero': ('AGENT_CICERO_RSCRIPT', 'cicero-1'),
    'ArchR': ('AGENT_ARCHR_RSCRIPT', 'archr-1'),
    'monaLisa': ('AGENT_MONALISA_RSCRIPT', 'archr-1'),
}


def rscript(package):
    from .sequence_prediction import library
    variable, runtime = R_RUNTIMES[package]
    return Path(os.environ[variable]) if os.environ.get(variable) else library().root.parent / 'runtimes' / runtime / 'env/bin/Rscript'


def macs3():
    from .sequence_prediction import library
    return Path(os.environ['AGENT_MACS3_EXECUTABLE']) if os.environ.get('AGENT_MACS3_EXECUTABLE') else library().root.parent / 'runtimes/macs3/env/bin/macs3'
