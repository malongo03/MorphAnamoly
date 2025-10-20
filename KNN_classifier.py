import json
import math

import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
from tqdm import *

from morphFM.train.ssl_meta_arch import SSLMetaArch
from morphFM.train.train import get_args_parser
from morphFM.utils.config import setup

all_dataset = ['allen_cell_type_processed', 'allen_region_processed', 'BBP_cell_type_processed', 'BIL_cell_type_processed', 'M1_EXC_cell_type_processed', 'M1_EXC_region_processed']
#all_dataset = ['allen_cell_type_processed']
root_dir = 'benchmark_datasets'
checkpoint_path = '60423_student_checkpoint.pth'

def KNN(all_labels, all_embedding):
    
    acc_mean = 0.0

    all_labels = np.array(all_labels)
    all_data = np.squeeze(all_embedding, axis=1)

    num_all = len(all_labels)
    num_train = int( 0.6 * len(all_labels) )

    sum_val = 0.0
    num_repeat = 5

    for k in range(num_repeat):
        
        indices = np.arange(len(all_data))
        np.random.shuffle(indices)

        train_indices = indices[:int(0.6 * len(all_data))] 
        val_indices = indices[int(0.4 * len(all_data)):] 
        
        train_data = all_data[train_indices]
        train_labels = all_labels[train_indices]

        val_data = all_data[val_indices]
        val_labels = all_labels[val_indices]

        max_val = 0.0

        for now_n in range(1,20):

            k_nn = KNeighborsClassifier(n_neighbors=now_n)

            k_nn.fit(train_data, train_labels)

            predicted_labels = k_nn.predict(train_data)
            acc_num = np.sum((train_labels == predicted_labels) + 0)
            total_num = len(train_labels)

            training_acc = round(acc_num * 100.0 / total_num, 2)

            predicted_labels = k_nn.predict(val_data)
            acc_num = np.sum((val_labels == predicted_labels) + 0)
            total_num = len(val_labels)

            val_acc = round(acc_num * 100.0 / total_num, 2)

            if val_acc > max_val:
                max_val = val_acc

        sum_val += max_val
    
    sum_val = round(sum_val * 1.0 / num_repeat, 2)
    acc_mean += sum_val
    print(sum_val)


def build_model(cfg):

    backbonemodel = SSLMetaArch(cfg).to(torch.device("cuda"))
    backbonemodel.prepare_for_distributed_training()

    student_state_dict = torch.load(checkpoint_path)

    backbonemodel.student.load_state_dict(student_state_dict["student"])

    return backbonemodel

def get_embedding(now_dir, model, cfg):

    all_embedding = []
    all_labels = []

    with open(now_dir + 'processed/processed_data.json', 'r') as f:
        all_data = json.load(f)
    
    all_keys = all_data.keys()

    for key in tqdm(all_keys):
        
        now_data = all_data[key]
        adj = torch.tensor(np.array(now_data['adj'])).half().cuda()
        lapl = torch.tensor(np.array(now_data['lapl'])).float().cuda()
        feat = torch.from_numpy(np.array(now_data['feat'])).half().cuda()


        embedding = model.student.backbone(feat, adj, lapl)["x_norm_clstoken"].detach()

        for embed in embedding[0]:
            assert not math.isnan(embed.item())

        all_embedding.append(embedding.cpu().numpy())  
        all_labels.append(now_data['label'])

        del feat, adj, lapl, embedding 
        torch.cuda.empty_cache()  

    return all_embedding, all_labels



def run(now_dataset, cfg):

    embedding, labels = get_embedding(root_dir + now_dataset + '/', model, cfg)
    KNN(labels, embedding)


args = get_args_parser(add_help=True).parse_args()
args.config_file = 'configs/ours_final.yaml'
args.output_dir = 'neuron_org_embedding'
cfg = setup(args)
model = build_model(cfg)

for dataset in all_dataset:
    print('now test :' + dataset)
    run(dataset, cfg)







