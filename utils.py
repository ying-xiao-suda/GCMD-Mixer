import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
from torch import nn

def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=50,
    name="",
):
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    if name != "":
        output_path = name + "/model.pth"

    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )
    l=nn.MSELoss()
    best_valid_loss = 1e10
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):

                x,md,y=train_batch
                x=x.to(config['device'])
                md=md.to(config['device'])
                y=y.to(config['device'])
                y_hat=model(x,md)
                loss = l(y_hat, y).mean()

                optimizer.zero_grad()
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
                if batch_no >= config["itr_per_epoch"]:
                    break
            lr_scheduler.step()

        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0 and (epoch_no + 1) > config["epochs"] * 0.5:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        x,md,y=valid_batch
                        x=x.to(config['device'])
                        md=md.to(config['device'])
                        y=y.to(config['device'])
                        y_hat=model(x,md)  
                        loss = l(y_hat, y).mean()
                        avg_loss_valid += loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )
            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print(
                    "\n best loss is updated to ",
                    avg_loss_valid / batch_no,
                    "at",
                    epoch_no,
                )
    if name != "":
        torch.save(model.state_dict(), output_path)


def evaluate(model,config, test_loader, max, min, name=""):

    with torch.no_grad():
        model.eval()
        pred=[]
        truth=[]
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, valid_batch in enumerate(it, start=1):
                x,md,y=valid_batch
                x=x.to(config['device'])
                md=md.to(config['device'])
                y=y.to(config['device'])
                y_hat=model(x,md)
                pred.append(y_hat)
                truth.append(y)

        pred = torch.cat(pred, dim=0)
        truth= torch.cat(truth, dim=0)

        pred=pred*(max-min+1e-9)+min
        truth=truth*(max-min+1e-9)+min


        pred=pred.cpu().numpy()
        truth=truth.cpu().numpy()

        nonzero_mask = truth.reshape(-1) != 0
        nonzero_preds = pred.reshape(-1)[nonzero_mask]
        nonzero_val = truth.reshape(-1)[nonzero_mask]

        smape = np.mean(2 * np.abs(nonzero_preds - nonzero_val) / (np.abs(nonzero_preds) + np.abs(nonzero_val) + 1e-8))

        # smape = np.mean(2 * np.abs(pred.reshape(-1)- truth.reshape(-1)) / (np.abs(pred.reshape(-1)) + np.abs(truth.reshape(-1)) + 1e-16))
        rmse = np.sqrt(np.mean(np.square(pred.reshape(-1)-truth.reshape(-1))))
        mae = np.mean(np.abs(pred.reshape(-1)-truth.reshape(-1)))

        print('mae:',mae)
        print('rmse:',rmse)
        print('smape:',smape)
        np.save(name + "/pred",pred)
        np.save(name + "/truth",truth)
        
