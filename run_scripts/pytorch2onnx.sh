export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=${PYTHONPATH}:`pwd`/cn_clip

DATAPATH=`pwd`
# ${DATAPATH}的指定，请参考Readme"代码组织"部分创建好目录，尽量使用相对路径：https://github.com/OFA-Sys/Chinese-CLIP#代码组织
checkpoint_path=${DATAPATH}/pretrained_weights/clip_cn_vit-b-16_finetune.pt # 指定要转换的ckpt完整路径
mkdir -p ${DATAPATH}/deploy/ # 创建ONNX模型的输出文件夹

python cn_clip/deploy/pytorch_to_onnx.py \
       --model-arch ViT-B-16 \
       --pytorch-ckpt-path ${checkpoint_path} \
       --save-onnx-path ${DATAPATH}/deploy/vit-b-16 \
       --convert-text --convert-vision
