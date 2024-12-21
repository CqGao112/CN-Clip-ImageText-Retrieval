from tqdm import tqdm
import json
import os
import numpy as np
import torch
import onnxruntime
from cn_clip.clip import tokenize

config = {
    'img_path':'data/flickr8k-images',
    'id2img':'data/id2image.json',
    'image_feats':'data/train_imgs.img_feat.jsonl',
    'eval_batch_size':32,
    'device':'cuda',

    'onnx_image_model':'deploy/vit-b-16.img.fp32.onnx',
    'onnx_text_model':'deploy/vit-b-16.txt.fp32.onnx',

    'max_txt_length':52
}



def load_txt_img_model_onnx():
    if config['device'] == "cpu":
        provider = "CPUExecutionProvider"
    else:
        provider = "CUDAExecutionProvider"

    img_sess_options = onnxruntime.SessionOptions()
    img_run_options = onnxruntime.RunOptions()
    img_run_options.log_severity_level = 2
    img_session = onnxruntime.InferenceSession(config['onnx_image_model'],
                                               sess_options=img_sess_options,
                                               providers=[provider])

    txt_sess_options = onnxruntime.SessionOptions()
    txt_run_options = onnxruntime.RunOptions()
    txt_run_options.log_severity_level = 2
    txt_session = onnxruntime.InferenceSession(config['onnx_text_model'],
                                               sess_options=txt_sess_options,
                                               providers=[provider])

    return img_session,txt_session


def make_topk_prediction(config, text, top_k, txt_model):
    print("Begin to load image features...")
    image_ids = []
    image_feats = []
    with open(config['image_feats'], "r") as fin:
        for line in tqdm(fin):
            obj = json.loads(line.strip())
            image_ids.append(obj['image_id'])
            image_feats.append(obj['feature'])
    image_feats_array = np.array(image_feats, dtype=np.float32)
    print("Finished loading image features.")

    print("Begin to compute top-{} predictions for texts...".format(top_k))
    with torch.no_grad():
        text_feat_tensor = txt_model.run(["unnorm_text_features"], {"text": text.numpy()})[0]
        text_feat_tensor = torch.tensor(text_feat_tensor).to(config['device'])
        text_feat_tensor /= text_feat_tensor.norm(dim=-1, keepdim=True)

    score_tuples = []
    # text_feat_tensor = torch.tensor([text_feat, dtype=torch.float).cuda()  # [1, feature_dim]
    idx = 0
    while idx < len(image_ids):
        img_feats_tensor = torch.from_numpy(image_feats_array[idx: min(idx + config['eval_batch_size'],
                                                                       len(image_ids))]).cuda()  # [batch_size, feature_dim]
        batch_scores = text_feat_tensor @ img_feats_tensor.t()  # [1, batch_size]
        for image_id, score in zip(image_ids[idx: min(idx + config['eval_batch_size'], len(image_ids))],
                                   batch_scores.squeeze(0).tolist()):
            score_tuples.append((image_id, score))
        idx += config['eval_batch_size']
    top_k_predictions = sorted(score_tuples, key=lambda x: x[1], reverse=True)[:top_k]

    fout = [entry[0] for entry in top_k_predictions]

    return fout

def preprocess_text(text):
    # from cn_clip.eval.data
    # adapt the text to Chinese BERT vocab
    text = text.lower().replace("“", "\"").replace("”", "\"")
    return text





img_session, txt_session = load_txt_img_model_onnx()

with open(config['id2img'], "r", encoding="utf-8") as json_file:
    imgId2name = json.load(json_file)

def clip_text2image(text, num, model, thumbnail):
    img_path = config['img_path']
    text = tokenize([preprocess_text(str(text))], context_length=config['max_txt_length'])
    img_ids = make_topk_prediction(config, text, num, txt_session)
    img = [os.path.join(img_path,imgId2name[str(id)]) for id in img_ids]

    return img


