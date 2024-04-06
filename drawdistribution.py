import numpy as np
from utils.plotfigure import plot3ddistribution
from utils.drawfigure import Exact_plot
exp = "data/Heat_Types1_source4_number100000fixed.npz"
set  = np.load(exp)
data =set['S']
data2 =set['T'].reshape(1,100000,64,64)

# plot3ddistribution(exp=exp,type_num=2,data_num_per_type=1,data=data)
for i in range(2):
   for j in range(1):
       Exact_plot(i,j,data2[i,j])