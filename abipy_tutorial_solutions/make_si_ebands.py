import numpy as np

from abipy import abilab
from abipy import flowtk
import abipy.data as abidata

def build_ebands_flow(options):
    """
    Band structure calculation.
    First, a SCF density computation, then a non-SCF band structure calculation.
    Similar to tbase3_5.in
    """
    multi = abilab.MultiDataset(structure=abidata.cif_file("si.cif"),
                                pseudos=abidata.pseudos("14si.pspnc"), ndtset=2)
    # Global variables
    shiftk = [float(s) for s in "0.5 0.5 0.5 0.5 0.0 0.0 0.0 0.5 0.0 0.0 0.0 0.5".split()]
    multi.set_vars(ecut=8, diemac=12, iomode=3)

    # Dataset 1
    multi[0].set_vars(tolvrs=1e-9)
    multi[0].set_kmesh(ngkpt=[4, 4, 4], shiftk=shiftk)

    # Dataset 2
    multi[1].set_vars(tolwfr=1e-15)
    multi[1].set_kpath(ndivsm=5)

    scf_input, nscf_input = multi.split_datasets()

    workdir = options.workdir if (options and options.workdir) else "flow_base3_ebands"

    return flowtk.bandstructure_flow(workdir, scf_input=scf_input, nscf_input=nscf_input)


def setup_manager(flow, mpi_procs, omp_threads):
    manager = flow.manager.new_with_fixed_mpi_omp(mpi_procs=mpi_procs, omp_threads=omp_threads)
    for work in flow:
        work.set_manager(manager)
    return flow


def main():
    flow = build_ebands_flow(options=None)
    setup_manager(flow, mpi_procs=4, omp_threads=1)
    flow.build_and_pickle_dump()    # Build the workflow to run it later with abirun.py
    #flow.make_scheduler().start()  # This is to run the workflow instantly.


if __name__ == '__main__':
    import sys
    sys.exit(main())
