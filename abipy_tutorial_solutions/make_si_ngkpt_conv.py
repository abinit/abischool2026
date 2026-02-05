import numpy as np

from abipy import abilab
from abipy import flowtk
import abipy.data as abidata

def build_ngkpt_flow(options):
    """
    Crystalline silicon: computation of the total energy
    Convergence with respect to the number of k points. Similar to tbase3_3.in

    Args:
        options: Command line options.

    Return:
        Abinit Flow object.
    """
    # Definition of the different grids
    ngkpt_list = [(2, 2, 2), (4, 4, 4), (6, 6, 6), (8, 8, 8)]

    # These shifts will be the same for all grids
    shiftk = [float(s) for s in "0.5 0.5 0.5 0.5 0.0 0.0 0.0 0.5 0.0 0.0 0.0 0.5".split()]

    # Build MultiDataset object (container of `ndtset` inputs).
    # Structure is initialized from CIF file.
    multi = abilab.MultiDataset(structure=abidata.cif_file("si.cif"),
                                pseudos=abidata.pseudos("14si.pspnc"), ndtset=len(ngkpt_list))

    # These variables are the same in each input.
    multi.set_vars(ecut=8, toldfe=1e-6, diemac=12.0, iomode=3)

    # Each input has its own value of `ngkpt`. shiftk is constant.
    for i, ngkpt in enumerate(ngkpt_list):
        multi[i].set_kmesh(ngkpt=ngkpt, shiftk=shiftk)

    workdir = options.workdir if (options and options.workdir) else "flow_base3_ngkpt"

    # Split the inputs by calling multi.datasets() and pass the list of inputs to Flow.from_inputs.
    return flowtk.Flow.from_inputs(workdir, inputs=multi.split_datasets())


def setup_manager(flow, mpi_procs, omp_threads):
    manager = flow.manager.new_with_fixed_mpi_omp(mpi_procs=mpi_procs, omp_threads=omp_threads)
    for work in flow:
        work.set_manager(manager)
    return flow


def main():
    flow = build_ngkpt_flow(options=None)
    setup_manager(flow, mpi_procs=4, omp_threads=1)
    flow.build_and_pickle_dump()
    #flow.make_scheduler().start()

if __name__ == '__main__':
    import sys
    sys.exit(main())
