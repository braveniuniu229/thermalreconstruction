import numpy as np
from utils.plotfigure import plot3ddistribution
from utils.drawfigure import Exact_plot
exp = "data/Heat_Types500_source4_number200fixed.npz"
set  = np.load(exp)
data =set['S']
data2 =set['T'].reshape(500,200,64,64)

plot3ddistribution(exp=exp,type_num=500,data_num_per_type=10,data=data)
# for i in range(5):
#    for j in range(1):
#        Exact_plot(i,j,data2[i,j])