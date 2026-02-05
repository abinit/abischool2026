import numpy as np

from abipy import abilab
from abipy import flowtk
import abipy.data as abidata

def make_gs_input():

    # Build MultiDataset object (container of `ndtset` inputs).
    # Structure is initialized from CIF file.
    inp = abilab.AbinitInput(structure=abidata.cif_file('al.cif'),
                             pseudos=abidata.pseudos('13al.pspnc'))

    # These variables are the same in each input.
    inp.set_vars(ecut=8, toldfe=1e-6, occopt=7)

    return inp

def build_flow(options):
    """
    Crystalline aluminium:
    Convergence of total energy with respect to the number of k points and tsmear.

    Args:
        options: Command line options.

    Return:
        Abinit Flow object.
    """

    tsmear_list = [0.001, 0.005, 0.01, 0.025, 0.05]

    # Definition of the different grids
    ngkpt_list = [(2, 2, 2), (4, 4, 4), (6, 6, 6), (8, 8, 8)]

    # These shifts will be the same for all grids
    shiftk = [float(s) for s in "0.5 0.5 0.5 0.5 0.0 0.0 0.0 0.5 0.0 0.0 0.0 0.5".split()]

    flow = flowtk.Flow(workdir='flow_al_ngkpt_tsmear')
    for tsmear in tsmear_list:
        work = flowtk.Work()
        for ngkpt in ngkpt_list:
            inp = make_gs_input()
            inp.set_vars(tsmear=tsmear)
            inp.set_kmesh(ngkpt=ngkpt, shiftk=shiftk)
            work.register_task(inp)
        flow.register_work(work)
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
