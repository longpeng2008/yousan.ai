(1) 数据集介绍
这是一个嘴唇数据集，包括无状态的嘴唇和微笑状态的嘴唇，为本项目的caffe和tensorflow例子使用。
所有图片的尺寸为60*60
0 无状态
1 微笑

(2) 脚本介绍
script中包含若干脚本
-reformat.py 批量修改图片格式
-checksize.py 批量查看图片大小
-genelist.py 从文件夹中生成分类格式的list
-gene_train_val.py 将图片按照一定比例分为train和val
-shuffle.py 用于随机打乱图片list
-rename_files_function,run_rename.sh 批量重命名图片，用于统一格式
