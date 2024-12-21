from PIL import Image
from io import BytesIO
import base64
import os
import json




# 指定数据路径
data_path = 'datasets/Flickr8K_CN'
images = 'flickr8k-images'
train_ano = 'cn_train.json'
val_ano = 'cn_val.json'

# 输出文件
imgID2name = 'id2image.json'

train_ano_output = 'train_texts.jsonl'
train_image_output = 'train_imgs.tsv'

val_ano_output = 'valid_texts.jsonl'
val_image_output = 'valid_imgs.tsv'


image2id = {}
id2image = {}
files = os.listdir(os.path.join(data_path,images))
for id, file in zip(range(len(files)),files):
    file_name = os.path.join(images,file)
    image2id[file_name] = id
    id2image[id] = file

# 获得id2image文件，用于模型推理
with open(os.path.join(data_path,imgID2name), "w", encoding="utf-8") as json_file:
    json.dump(id2image, json_file, ensure_ascii=False, indent=4)


# 预处理训练集
with open(os.path.join(data_path,train_ano), 'r', encoding='utf-8') as ano_file:
    with open(os.path.join(data_path, train_ano_output), 'w', encoding='utf-8') as ano_file_o:
        with open(os.path.join(data_path, train_image_output), 'w', encoding='utf-8') as image_file_o:
            anos = json.load(ano_file)
            text_id = 0
            for ano in anos:
                img = Image.open(os.path.join(data_path,ano['image'])) # 访问图片路径
                img_buffer = BytesIO()
                img.save(img_buffer, format=img.format)
                byte_data = img_buffer.getvalue()
                base64_str = base64.b64encode(byte_data) # bytes
                base64_str = base64_str.decode("utf-8") # str
                # 输入图像数据
                image_file_o.write(f"{image2id[ano['image']]}\t{base64_str}" + '\n')

                for item in ano['caption']:
                    jsonstr = {
                        'text_id':text_id,
                        'text':item,
                        'image_ids':[image2id[ano['image']]]
                    }
                    jsonstr = json.dumps(jsonstr,ensure_ascii=False)
                    # 输入文本数据
                    ano_file_o.write(jsonstr + '\n')

                    text_id = text_id + 1

# 预处理验证集
with open(os.path.join(data_path,val_ano), 'r', encoding='utf-8') as ano_file:
    with open(os.path.join(data_path, val_ano_output), 'w', encoding='utf-8') as ano_file_o:
        with open(os.path.join(data_path, val_image_output), 'w', encoding='utf-8') as image_file_o:
            anos = json.load(ano_file)
            text_id = 0
            for ano in anos:
                img = Image.open(os.path.join(data_path,ano['image'])) # 访问图片路径
                img_buffer = BytesIO()
                img.save(img_buffer, format=img.format)
                byte_data = img_buffer.getvalue()
                base64_str = base64.b64encode(byte_data) # bytes
                base64_str = base64_str.decode("utf-8") # str
                # 输入图像数据
                image_file_o.write(f"{image2id[ano['image']]}\t{base64_str}" + '\n')

                for item in ano['caption']:
                    jsonstr = {
                        'text_id':text_id,
                        'text':item,
                        'image_ids':[image2id[ano['image']]]
                    }
                    jsonstr = json.dumps(jsonstr,ensure_ascii=False)
                    # 输入文本数据
                    ano_file_o.write(jsonstr + '\n')

                    text_id = text_id + 1
