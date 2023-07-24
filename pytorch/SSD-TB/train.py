"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os
from datetime import datetime
import glob
from argparse import ArgumentParser

import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter

from src.model import SSD, ResNet, Textboxes
from src.utils import generate_dboxes, Encoder, coco_classes
from src.transform import SSDTransformer
from src.loss import Loss
from src.process import train, evaluate
from src.dataset import collate_fn, CustomDataset

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def get_args():
    parser = ArgumentParser(description="Implementation of SSD/TB")
    parser.add_argument("--data-path", type=str, default="coco",
                        help="the root folder of dataset")
    parser.add_argument("--dataset-name", type=str, default="V2",
                        help="the name of dataset")
    parser.add_argument("--save-folder", type=str, default="run/V2",
                         help="path to folder containing model checkpoint file")

    parser.add_argument("--experiment-name", type=str, default="")
    parser.add_argument("--pretrained", type=str, default="")

    parser.add_argument("--model", type=str, default="TB_noOffset", choices=["TB_noOffset", "TB", "SSD" "SSD_custom"],
                        help="model name")
    parser.add_argument("--trunc", type=str, default="yes", choices=["yes", "no"],
                        help="Truncation of the model after conv8_2")
    parser.add_argument("--backbone", type=str, default="resnet50", help="resnet[18,34,50,101,152]",
                        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"] )
    parser.add_argument("--figsize", type=int, default=300, help="input size, either 300x300 or 512x512",
                        choices=[300,512] )
 
    parser.add_argument("--epochs", type=int, default=1, help="number of total epochs to run")
    parser.add_argument("--batch-size", type=int, default=32, help="number of samples for each iteration")
    parser.add_argument("--multistep", nargs="*", type=int, default=[43, 54],
                        help="epochs at which to decay learning rate")


    parser.add_argument("--lr", type=float, default=2.6e-3, help="initial learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum argument for SGD optimizer")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="momentum argument for SGD optimizer")
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=4)

    args = parser.parse_args()
    return args


def main(opt):
    if torch.cuda.is_available():
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        num_gpus = torch.distributed.get_world_size()

        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
        num_gpus = 1

    experiment_dir = os.path.join(opt.save_folder, 'experiment_{}'.format(opt.experiment_name))
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    

    train_params = {"batch_size": opt.batch_size * num_gpus,
                    "shuffle": True,
                    "drop_last": False,
                    "num_workers": opt.num_workers,
                    "collate_fn": collate_fn}

    test_params = {"batch_size": opt.batch_size * num_gpus,
                   "shuffle": False,
                   "drop_last": False,
                   "num_workers": opt.num_workers,
                   "collate_fn": collate_fn}
    
    dboxes = generate_dboxes(opt.model, opt.trunc, opt.figsize)

    train_set = CustomDataset(opt.data_path, opt.dataset_name, "train", SSDTransformer(dboxes, (opt.figsize,opt.figsize), val=False))
    test_set = CustomDataset(opt.data_path, opt.dataset_name, "val", SSDTransformer(dboxes, (opt.figsize,opt.figsize), val=True))
        
    train_loader = DataLoader(train_set, **train_params)
    test_loader = DataLoader(test_set, **test_params)

    num_classes = len(test_loader.dataset.coco.getCatIds()) + 1 # background included
            
    if "SSD" in opt.model:
        model = SSD(opt.model, str2bool(opt.trunc), backbone=ResNet(opt.backbone), figsize=opt.figsize, num_classes=num_classes)
    else:
        model = Textboxes(opt.model, str2bool(opt.trunc), backbone=ResNet(opt.backbone), figsize=opt.figsize, num_classes=num_classes)


    encoder = Encoder(dboxes)

    opt.lr = opt.lr * num_gpus * (opt.batch_size / 32)
    criterion = Loss(dboxes)

    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum,
                                weight_decay=opt.weight_decay,
                                nesterov=True)
    scheduler = MultiStepLR(optimizer=optimizer, milestones=opt.multistep, gamma=0.1)

    if torch.cuda.is_available():
        model.cuda()
        criterion.cuda()



    checkpoint_path = os.path.join(experiment_dir, "checkpoint.pth")

    if os.path.isfile(opt.pretrained):
        checkpoint = torch.load(opt.pretrained, map_location=torch.device('cpu'))
        first_epoch = checkpoint["epoch"] + 1


        #model_state_dict = {k.replace('module.', ''): v for k, v in checkpoint["model_state_dict"].items()}
        model_state_dict = checkpoint["model_state_dict"]
        model.load_state_dict(model_state_dict)
        scheduler.load_state_dict(checkpoint["scheduler"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        first_epoch = 0

    for epoch in range(first_epoch, opt.epochs):
        train(model, train_loader, epoch, experiment_dir, criterion, optimizer, scheduler)
        evaluate(model, test_loader, epoch, experiment_dir, criterion, encoder, opt.nms_threshold)

        checkpoint = {"epoch": epoch,
                      "model_state_dict": model.state_dict(),
                      "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict()}
        torch.save(checkpoint, checkpoint_path)


if __name__ == "__main__":
    opt = get_args()
    start_time = datetime.now()
    main(opt)
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
