from abipy import abilab
import matplotlib.pyplot as plt

data_files = [
    ['flow_al_ngkpt_tsmear/w0/t0/outdata/out_GSR.nc',
     'flow_al_ngkpt_tsmear/w0/t1/outdata/out_GSR.nc',
     'flow_al_ngkpt_tsmear/w0/t2/outdata/out_GSR.nc',
     'flow_al_ngkpt_tsmear/w0/t3/outdata/out_GSR.nc',
    ],
    ['flow_al_ngkpt_tsmear/w1/t0/outdata/out_GSR.nc',
     'flow_al_ngkpt_tsmear/w1/t1/outdata/out_GSR.nc',
     'flow_al_ngkpt_tsmear/w1/t2/outdata/out_GSR.nc',
     'flow_al_ngkpt_tsmear/w1/t3/outdata/out_GSR.nc',
    ],
    ['flow_al_ngkpt_tsmear/w2/t0/outdata/out_GSR.nc',
     'flow_al_ngkpt_tsmear/w2/t1/outdata/out_GSR.nc',
     'flow_al_ngkpt_tsmear/w2/t2/outdata/out_GSR.nc',
     'flow_al_ngkpt_tsmear/w2/t3/outdata/out_GSR.nc',
    ],
    ['flow_al_ngkpt_tsmear/w3/t0/outdata/out_GSR.nc',
     'flow_al_ngkpt_tsmear/w3/t1/outdata/out_GSR.nc',
     'flow_al_ngkpt_tsmear/w3/t2/outdata/out_GSR.nc',
     'flow_al_ngkpt_tsmear/w3/t3/outdata/out_GSR.nc',
    ],
]

for gsr_fname_l in data_files:

    tsmear = 0
    nkpt = []
    energy = []

    for gsr_fname in gsr_fname_l:

        gsr = abilab.abiopen(gsr_fname)

        tsmear = gsr.tsmear
        nkpt.append(gsr.nkpt)
        energy.append(gsr.energy)

    plt.plot(nkpt, energy, '-o', label=f'tsmear={tsmear}')

plt.xlabel('nkpt')
plt.ylabel('Energy (eV)')

plt.legend()
plt.tight_layout()
plt.show()
