import numpy as np
from utils.plotfigure import plot3ddistribution

exp = ""
set  = np.load(exp)
data = set['s']
plot3ddistribution(exp=exp,type_num=1,data_num_per_type=1,data=data)
