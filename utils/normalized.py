
import numpy as np

class Normalized():
    def __init__(self, path):
        self.path = path
        self.normalized()  # 在初始化时调用normalized方法

    def normalized(self):
        data = np.load(self.path)  # 使用self.path来访问实例变量
        origindata = data['T']
        mean = np.mean(origindata.reshape(-1))
        std_var = np.std(origindata.reshape(-1))  # 直接计算标准差
        normalized_data = (origindata - mean) / std_var
        mid_path = self.path.rsplit('.npz', 1)[0]  # 移除.npz
        new_path = mid_path + "_normalized.npz"
        np.savez(new_path, T=normalized_data)