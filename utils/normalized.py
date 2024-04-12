
import numpy as np





def normalized():
    path = '../data/Heat_Types2000_source4_number50fixed.npz'
    data = np.load(path)  # 使用self.path来访问实例变量
    origindata = data['T']
    source = data['S']
    mean = np.mean(origindata.reshape(-1))
    std_var = np.std(origindata.reshape(-1))  # 直接计算标准差
    normalized_data = (origindata - mean) / std_var
    mid_path = path.rsplit('.npz', 1)[0]  # 移除.npz
    new_path = mid_path + "_normalized.npz"
    np.savez(new_path, T=normalized_data,m=mean,v=std_var,S=source)
if __name__ == '__main__':
    normalized()
