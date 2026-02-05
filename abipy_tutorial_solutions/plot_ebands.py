
from abipy import abilab
import matplotlib.pyplot as plt

gsr = abilab.abiopen('flow_base3_ebands/w0/t1/outdata/out_GSR.nc')

ebands = gsr.ebands

ebands.plot(show=False)

plt.title('Silicon band structure')

plt.show()
