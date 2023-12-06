#!/bin/bash

# 设置视频文件所在的目录
video_directory="/13994058190/WYH/UNINEXT/datasets/a2d_sentences/Release/clips320H"

# 遍历视频目录中的每个视频文件
for video_file in "$video_directory"/*.mp4; do
    if [ -f "$video_file" ]; then
        # 提取帧到对应的视频目录
        output_directory="${video_file%.mp4}"  # 移除文件扩展名
        mkdir -p "$output_directory"           # 创建输出目录（如果不存在）
        ffmpeg -i "$video_file" "$output_directory"/%05d.png
    fi
done
