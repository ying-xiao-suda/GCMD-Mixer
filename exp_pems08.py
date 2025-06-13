import argparse
import torch
import datetime
import json
import yaml
import os
from Model.gcmd_mxier import GCMDMixer
from dataset_pems08 import get_dataloader
from utils import train, evaluate
import torch
from get_adj import get_adj_pems08


parser = argparse.ArgumentParser(description="GCMDMixer")

parser.add_argument("--config", type=str, default="pems08.yaml")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--modelfolder", type=str, default="")
args = parser.parse_args()
print(args)

path = "Config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)


print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

name = "./save/pems08/pems08_" + current_time + "/"
print('model folder:', name)
os.makedirs(name, exist_ok=True)
with open(name + "config.json", "w") as f:
    json.dump(config, f, indent=4)


train_loader, valid_loader, test_loader,max,min = get_dataloader(config['md'],batch_size=config['train']['batchsize'])

adj=get_adj_pems08()
adj=torch.tensor(adj,dtype=torch.float32).to(args.device)
model = GCMDMixer(
                sequence_len=config['model']['sequence_len'],
                l_sequence_hid=config['model']['l_sequence_hid'],
                g_sequence_hid=config['model']['g_sequence_hid'],
                input_dim=config['model']['input_dim'],
                emb_dim=config['model']['emb_dim'],
                split=config['model']['split'],
                adj=adj,
                modes=config['model']['modes'],
                dropout=config['model']['dropout']
                ).to(args.device)

min=torch.tensor(min).to(args.device).float()
max=torch.tensor(max).to(args.device).float()
if args.modelfolder == "":
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        name=name,
    )
else:
    model.load_state_dict(torch.load("./save/pems08/" + args.modelfolder + "/model.pth"))

evaluate(model,config['train'], test_loader, max=max, min=min, name=name)
