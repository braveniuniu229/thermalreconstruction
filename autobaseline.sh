#!/bin/bash

# 进入脚本所在目录（如果需要）


# 顺序执行脚本，只有在前一个脚本成功执行后才会执行下一个

python train_shallowdecoder_type_50.py && \
python train_shallowdecoder_type_100.py && \
python train_shallowdecoder_type_200.py && \
python train_shallowdecoder_type_1000.py && \
python train_shallowdecoder_type_10000.py

# 如果所有脚本都成功执行，打印消息
if [ $? -eq 0 ]; then
    echo "All scripts have been executed successfully."
else
    echo "An error occurred. Stopping execution."
fi
