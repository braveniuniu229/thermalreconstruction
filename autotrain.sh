#!/bin/bash

# 切换到脚本所在目录


# 运行第一个训练脚本
python heatdatasquare_zwz.py --type_num 500 --data_num_per_type 200

# 检查第一个脚本是否成功执行
if [ $? -eq 0 ]; then
    echo "train_script1.py 完成，正在启动 train_script2.py..."
    # 运行第二个训练脚本
    python python heatdatasquare_zwz.py --type_num 10000 --data_num_per_type 10
else
    echo "train_script1.py 执行失败。"
fi
