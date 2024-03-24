import numpy as np
from utils.plotfigure import plot3ddistribution
from utils.drawfigure import Exact_plot
exp = "data/Heat_Types100_source4_number10fixed.npz"
set  = np.load(exp)
data =set['S']
data2 =set['T'].reshape(100,10,64,64)

plot3ddistribution(exp=exp,type_num=10,data_num_per_type=5,data=data)
for i in range(5):
   for j in range(5):
       Exact_plot(i,j,data2[i,j])