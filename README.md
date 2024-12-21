# Chinese-Clip-ImageText-Retrieval
## 介绍

本项目为本人在学习多模态视觉语言模型的入门项目，非常适合初学者入门进行操作。通过该项目，初学者可以自己搭建一个图文检索应用。主要内容包括：

- 数据集预处理
- 基于chinese-clip的模型微调
- 图文检索模型推理

## 目录结构

```
Chinese-Clip-ImageText-Retrieval
    ├── cn_clip
    ├── datasets
    ├── data
    ├── deploy
    ├── pretrained_weights
    ├── process.py
    ├── run_scripts
    ├── app.py
    ├── text2image.py
    ├── model_loader.py
    ├── requirement.txt
    └── README.md
```

## 环境构建

```
pip -r requirement.txt
```

## 数据集预处理

### 1.标注转换

本项目基于[Chinese-clip](https://github.com/OFA-Sys/Chinese-CLIP)构建，公开的数据集标注与Chinese-clip库中的模型并不适配，为了便于后续模型微调、推理，同时保证数据处理和读取的效率。我们需要以下处理：

- 我们需要将所有图片以base64格式保存在`${split}_imgs.tsv`文件中，而不是把所有图片单独存放在文件夹中，从而提高读取效率。tsv文件每行包括包含图片id（int型）与图片base64，以tab隔开，格式如下：

  ```
  1	/9j/4AAQSkZJ...YQj7314oA//2Q==
  ```

- 将原始数据集标注的图文匹配关系更改为`image_ids`与`text_id`关系，将文件保存在`${split}_texts.jsonl`。例如：

  ```
  # 原始数据集格式：
  [
   {
      "image": "flickr8k-images\\3393035454_2d2370ffd4.jpg",
      "caption": [
        "有一个男孩从上往下跳。",
        "一个小男孩在跳跃。",
        "一个男孩在跳跃。"
      ]
    },
  ]
  
  # 处理后图文匹配关系
  {"text_id": 8428, "text": "有一个男孩从上往下跳。", "image_ids": [1223]}
  {"text_id": 8429, "text": "一个小男孩在跳跃。", "image_ids": [1223]}
  {"text_id": 84210, "text": "一个男孩在跳跃。", "image_ids": [1223]}
  ```

我们以Flickr8K-CN数据集(数据来源<https://github.com/bubbliiiing/clip-pytorch>)为例，给出一个预处理脚本。运行以下命令可以获取预处理后的数据集文件。

```bash
python process.py
```

### 2.格式转换

为方便训练时的随机读取，我们还需要将tsv和jsonl文件一起序列化，转换为内存索引的LMDB数据库文件。使用Chinese-clip中`build_lmdb_dataset.py`进行处理。以Flickr8K-CN数据集为例：

```
python cn_clip/preprocess/build_lmdb_dataset.py \
    --data_dir ./datasets/Flickr8K_CN
    --splits train,valid
```

经过以上两步处理，我们可以得到[Chinese-clip]()对应的数据格式的数据集。下面可以进行模型微调。

```
datasets/
    └── Flickr8K_CN/
    	├── train_imgs.tsv
    	├── train_texts.jsonl
    	├── valid_imgs.tsv
    	├── valid_texts.jsonl
        └── lmdb/
            ├── train
            │   ├── imgs
            │   └── pairs
            └── valid
```

## 基于chinese-clip的模型微调

首先，我们需要下载预训练权重，放在pretrained\_weights文件夹下。我们使用[CN-CLIP]()[~ViT-B/16~](https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/clip_cn_vit-b-16.pt)作为例子，如需使用其他模型权重，可在[Chinese-clip](https://github.com/OFA-Sys/Chinese-CLIP)下载。下面运行run_scripts脚本进行微调。其中`${DATAPATH}`使用的是项目目录

```markdown
bash run_scripts/flickr8k_finetune_vit-b-16_rbt-base.sh ${DATAPATH}
```

注意: 如果您的机器是单卡，scipts里面的内容需要设置为单卡而不是分布式，可参考以下配置。

```markdown
GPUS_PER_NODE=1
export CUDA_VISIBLE_DEVICES=0
WORKER_CNT=1
export MASTER_ADDR=localhost
```

微调了之后，在`${DATAPATH}`下出现experiments的文件，结构如下所示。

```
experiments/
	└── flickr8k_finetune_vit-b-16_roberta-base_bs128_8gpu	
		└── checkpoints
			└── epoch1.pt
```

## 图文检索模型推理

### 1.图像、文本特征提取

在进行推理前，为避免反复调用模型提取特征，提前采用模型提取图像特征和文本特征。注：推理过程中仅需要使用提取的图像特征

```
bash run_scripts/flickr8k_extract_vit-b-16_rbt-base.sh ${DATAPATH}
```

运行结束后，可以得到图像特征`${split}_imgs.img_feat.jsonl` 和 文本特征`${split}_text.txt_feat.jsonl`,对应的格式如下：

```
{"image_id": 1000002, "feature": [0.0198, ..., -0.017, 0.0248]}  #img_feat

{"text_id": 248816, "feature": [0.1314, ..., 0.0018, -0.0002]}   #txt_feat
```

### 2.ONNX模型加速推理

为了加速特征推理，我们采用了[Chinese-clip](https://github.com/OFA-Sys/Chinese-CLIP)中提供的部署的ONNX模型方法，将pytorch模型转换为ONNX模型

```
bash run_scripts/pytorch2onnx.sh
```

运行结束后，在`${DATAPATH}`下出现deploy目录，目录下是得到的ONNX模型文件，结构如下所示。

```
deploy/
	├── vit-b-16.img.fp16.onnx
	├── vit-b-16.img.fp16.onnx.extra_file
	├── vit-b-16.img.fp32.onnx
	├── vit-b-16.txt.fp16.onnx
	├── vit-b-16.txt.fp16.onnx.extra_file
	├── vit-b-16.txt.fp32.onnx
```

### 3.推理演示

推理前，将需要用到数据放到`data`目录下，包括需要检索的图片文件夹`images`，id转换图片名`id2image.json`，使用微调模型提取的图像特征`${split}_imgs.img_feat.jsonl`

运行`app.py`即可通过图形化界面，进行图文检索

```
python app.py
```

## 参考

本项目参考了

[Text2Image-Retrieval](https://github.com/sugarandgugu/Text2Image-Retrieval/tree/main)

[Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP)
