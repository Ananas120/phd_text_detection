"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os
import traceback
import numpy as np
from tqdm.autonotebook import tqdm
import torch
#from pycocotools.cocoeval import COCOeval

def saveLoss(log_file, losses, epoch):
    with open(log_file, 'a') as f:
        f.write('Epoch {0} - {1:.5f} \n'.format(epoch + 1, np.mean(losses)))

def saveMAP(log_file, mAP, epoch):
    with open(log_file, 'a') as f:
        f.write('Epoch {0} - {1:.5f} \n'.format(epoch + 1, mAP))

def train(model, train_loader, epoch, experiment_dir, criterion, optimizer, scheduler):
    model.train()
    num_iter_per_epoch = len(train_loader)
    progress_bar = tqdm(train_loader)
    
    losses = np.zeros(num_iter_per_epoch)

    for i, (img, _, _, gloc, glabel) in enumerate(progress_bar):
        if torch.cuda.is_available():
            img = img.cuda()
            gloc = gloc.cuda()
            glabel = glabel.cuda()

        ploc, plabel = model(img)
        ploc, plabel = ploc.float(), plabel.float()
        gloc = gloc.transpose(1, 2).contiguous()
        loss = criterion(ploc, plabel, gloc, glabel)

        losses[i] = loss.item()
        progress_bar.set_description("Epoch: {}. Loss: {:.5f}".format(epoch + 1, loss.item()))

        #writer.add_scalar("Train/Loss", loss.item(), epoch * num_iter_per_epoch + i)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    scheduler.step()

    saveLoss(os.path.join(experiment_dir, "train_loss.txt"), losses, epoch)


def evaluate(model, test_loader, epoch, experiment_dir, criterion, encoder, nms_threshold):
    model.eval()
    detections = []
    category_ids = test_loader.dataset.coco.getCatIds()

    losses = np.zeros(len(test_loader))

    for nbatch, (img, img_id, img_size, gloc, glabel) in enumerate(test_loader):
        print("Parsing batch: {}/{}".format(nbatch, len(test_loader)), end="\r")
        if torch.cuda.is_available():
            img = img.cuda()
        with torch.no_grad():
            # Get predictions
            ploc, plabel = model(img)
            ploc, plabel = ploc.float(), plabel.float()
            gloc = gloc.transpose(1, 2).contiguous()
            loss = criterion(ploc, plabel, gloc, glabel)

            losses[nbatch] = loss.item()

            for idx in range(ploc.shape[0]):
                ploc_i = ploc[idx, :, :].unsqueeze(0)
                plabel_i = plabel[idx, :, :].unsqueeze(0)
                try:
                    result = encoder.decode_batch(ploc_i, plabel_i, nms_threshold, 200)[0]
                except Exception:
                    print(traceback.format_exc())
                    print("No object detected in idx: {}".format(idx))
                    continue

                height, width = img_size[idx]
                loc, label, prob = [r.cpu().numpy() for r in result]
                for loc_, label_, prob_ in zip(loc, label, prob):
                    detections.append([img_id[idx], loc_[0] * width, loc_[1] * height, (loc_[2] - loc_[0]) * width,
                                       (loc_[3] - loc_[1]) * height, prob_,
                                       category_ids[label_ - 1]])
                    
    detections = np.array(detections, dtype=np.float32)

    # coco_eval = COCOeval(test_loader.dataset.coco, test_loader.dataset.coco.loadRes(detections), iouType="bbox")
    # coco_eval.evaluate()
    # coco_eval.accumulate()
    # coco_eval.summarize()

    #writer.add_scalar("Test/mAP", coco_eval.stats[0], epoch)
    saveLoss(os.path.join(experiment_dir, "test_loss.txt"), losses, epoch)
    #saveMAP(os.path.join(experiment_dir, "test_mAP.txt"), coco_eval.stats[0], epoch)
