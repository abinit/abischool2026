import numpy as np

from abipy import abilab
from abipy import flowtk
import abipy.data as abidata

def make_gs_input():

    inp = abilab.AbinitInput(structure=abidata.cif_file('al.cif'),
                             pseudos=abidata.pseudos('13al.pspnc'))

    # These variables are the same in each input.
    inp.set_vars(ecut=8, toldfe=1e-6, occopt=7, tsmear=0.01)
    shiftk = [float(s) for s in "0.5 0.5 0.5 0.5 0.0 0.0 0.0 0.5 0.0 0.0 0.0 0.5".split()]
    inp.set_kmesh(ngkpt=[2,2,2], shiftk=shiftk)

    return inp

def build_flow(options):
    """
    Crystalline aluminium:
    Convergence of total energy with respect to ecut.

    Args:
        options: Command line options.

    Return:
        Abinit Flow object.
    """
    flow = flowtk.Flow(workdir='flow_al_ecut_conv')

    ecut_list = [6, 8, 10, 12, 16]

    for ecut in ecut_list:
        inp = make_gs_input()
        inp.set_vars(ecut=ecut)
        flow.register_scf_task(inp)
    return flow


def setup_manager(flow, mpi_procs, omp_threads):
    manager = flow.manager.new_with_fixed_mpi_omp(mpi_procs=mpi_procs, omp_threads=omp_threads)
    for work in flow:
        work.set_manager(manager)
    return flow


def main():
    flow = build_flow(options=None)
    setup_manager(flow, mpi_procs=4, omp_threads=1)
    flow.build_and_pickle_dump()

if __name__ == '__main__':
    import sys
    sys.exit(main())
