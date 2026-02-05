import numpy as np

from abipy import abilab
from abipy import flowtk
import abipy.data as abidata

def build_relax_flow(options):
    """
    Crystalline silicon: computation of the optimal lattice parameter.
    Convergence with respect to the number of k points. Similar to tbase3_4.in
    """
    # Structural relaxation for different k-point samplings.
    ngkpt_list = [(2, 2, 2)]

    shiftk = [float(s) for s in "0.5 0.5 0.5 0.5 0.0 0.0 0.0 0.5 0.0 0.0 0.0 0.5".split()]

    multi = abilab.MultiDataset(structure=abidata.cif_file("al.cif"),
                                pseudos=abidata.pseudos("13al.pspnc"), ndtset=len(ngkpt_list))

    # Global variables
    multi.set_vars(
        ecut=8,
        tolvrs=1e-9,
        optcell=1,
        ionmov=3,
        occopt=3,
        ntime=10,
        dilatmx=1.05,
        ecutsm=0.5,
        diemac=12,
        iomode=3,
    )

    for i, ngkpt in enumerate(ngkpt_list):
        multi[i].set_kmesh(ngkpt=ngkpt, shiftk=shiftk)

    workdir = options.workdir if (options and options.workdir) else "flow_base3_relax"

    return flowtk.Flow.from_inputs(workdir, inputs=multi.split_datasets(), task_class=flowtk.RelaxTask)


def setup_manager(flow, mpi_procs, omp_threads):
    manager = flow.manager.new_with_fixed_mpi_omp(mpi_procs=mpi_procs, omp_threads=omp_threads)
    for work in flow:
        work.set_manager(manager)
    return flow


def main():
    flow = build_relax_flow(options=None)
    setup_manager(flow, mpi_procs=4, omp_threads=1)
    flow.build_and_pickle_dump()
    #flow.make_scheduler().start()

if __name__ == '__main__':
    import sys
    sys.exit(main())
