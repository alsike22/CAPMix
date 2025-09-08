import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from merlion.evaluate.anomaly import ScoreType
from models import *
from models.reasonable_metric import tsad_reasonable
from models.reasonable_metric import reasonable_accumulator
from torch.autograd import Variable
# from torch.utils.tensorboard import SummaryWriter
from .early_stopping import EarlyStopping

sys.path.append("../../")
def Trainer(model, model_optimizer, train_dl, val_dl, test_dl, device, config, idx):
    # Start training
    print("Training started ....")
    save_path = "./best_network/" + config.dataset
    os.makedirs(save_path, exist_ok=True)
    early_stopping = EarlyStopping(save_path, idx, patience=300)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')
    all_epoch_train_loss, all_epoch_test_loss = [], []

    for epoch in range(1, config.num_epoch + 1):
        # Train and validate
        train_target, train_score, train_loss = model_train(model, model_optimizer, train_dl, config, device, epoch)
        val_target, val_score_origin, val_loss = model_evaluate(model, val_dl, config, device, epoch)
        test_target, test_score_origin, test_loss = model_evaluate(model, test_dl, config, device, epoch)
        scheduler.step(train_loss)
        if epoch % 1 == 0:
            print(f'\nEpoch : {epoch}\n'
                     f'Train Loss     : {train_loss:.4f}\t | \n'
                     f'Valid Loss     : {val_loss:.4f}\t  | \n'
                     f'Test Loss     : {test_loss:.4f}\t  | \n'
                    )
        all_epoch_train_loss.append(train_loss.item())
        all_epoch_test_loss.append(test_loss.item())
        if config.dataset == 'UCR':
            val_affiliation, val_score, _, _, _ = ad_predict(val_target, val_score_origin, config.threshold_determine,
                                                       config.detect_nu)
            test_affiliation, test_score, _, _, predict = ad_predict(test_target, test_score_origin, config.threshold_determine,
                                                       config.detect_nu)
            score_reasonable = tsad_reasonable(test_target, predict, config.time_step)
            indicator = test_score.f1(ScoreType.RevisedPointAdjusted)
            early_stopping(score_reasonable, test_affiliation, test_score, indicator, val_score_origin,
                           test_score_origin, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        elif config.dataset == 'SWaT' or config.dataset == 'WADI':
            early_stopping(0, 0, 0, -val_loss.item(), val_score_origin, test_score_origin, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    print("\n################## Training is Done! #########################")
    # according to scores to create predicting labels
    if config.dataset == 'UCR':
        score_reasonable = early_stopping.best_score_reasonable
        # The UCR validation set has no anomaly, so it does not print.
        test_score_origin = early_stopping.best_predict2
        test_affiliation, test_rpa_score, test_pa_score, test_pw_score, predict = ad_predict(test_target,
                                                                                             test_score_origin,
                                                                                             config.threshold_determine,
                                                                                             config.detect_nu)

    elif config.dataset == 'SWaT' or config.dataset == 'WADI':
        val_score_origin = early_stopping.best_predict1
        test_score_origin = early_stopping.best_predict2
        print('best loss: {:.4f}'.format(early_stopping.best_indicator))
        val_affiliation, val_score, _, _, _ = ad_predict(val_target, val_score_origin, config.threshold_determine,
                                                         config.detect_nu)
        test_affiliation, test_rpa_score, test_pa_score, test_pw_score, predict = ad_predict(test_target,
                                                                                             test_score_origin,
                                                                                             config.threshold_determine,
                                                                                             config.detect_nu)
        score_reasonable = reasonable_accumulator(1, 0)
        val_f1 = val_score.f1(ScoreType.RevisedPointAdjusted)
        val_precision = val_score.precision(ScoreType.RevisedPointAdjusted)
        val_recall = val_score.recall(ScoreType.RevisedPointAdjusted)
        print("Valid affiliation-metrics")
        print(
            f'Test precision: {val_affiliation["precision"]:2.4f}  | \tTest recall: {val_affiliation["recall"]:2.4f}\n')
        print("Valid RAP F1")
        print(
            f'Valid F1: {val_f1:2.4f}  | \tValid precision: {val_precision:2.4f}  | \tValid recall: {val_recall:2.4f}\n')


    else:
        val_affiliation, val_score, _, _, _ = ad_predict(val_target, val_score_origin, config.threshold_determine,
                                                         config.detect_nu)
        test_affiliation, test_rpa_score, test_pa_score, test_pw_score, predict = ad_predict(test_target,
                                                                                             test_score_origin,
                                                                                             config.threshold_determine,
                                                                                             config.detect_nu)
        score_reasonable = reasonable_accumulator(1, 0)
        val_f1 = val_score.f1(ScoreType.RevisedPointAdjusted)
        val_precision = val_score.precision(ScoreType.RevisedPointAdjusted)
        val_recall = val_score.recall(ScoreType.RevisedPointAdjusted)
        print("Valid affiliation-metrics")
        print(
            f'Test precision: {val_affiliation["precision"]:2.4f}  | \tTest recall: {val_affiliation["recall"]:2.4f}\n')
        print("Valid RAP F1")
        print(
            f'Valid F1: {val_f1:2.4f}  | \tValid precision: {val_precision:2.4f}  | \tValid recall: {val_recall:2.4f}\n')


    print("Test affiliation-metrics")
    print(
        f'Test precision: {test_affiliation["precision"]:2.4f}  | \tTest recall: {test_affiliation["recall"]:2.4f}\n')
    test_rpa_f1 = test_rpa_score.f1(ScoreType.RevisedPointAdjusted)
    test_rpa_precision = test_rpa_score.precision(ScoreType.RevisedPointAdjusted)
    test_rpa_recall = test_rpa_score.recall(ScoreType.RevisedPointAdjusted)
    print("Test RAP F1")
    print(f'Test F1: {test_rpa_f1:2.4f}  | \tTest precision: {test_rpa_precision:2.4f}  | \tTest recall: {test_rpa_recall:2.4f}\n')

    test_pa_f1 = test_pa_score.f1(ScoreType.PointAdjusted)
    test_pa_precision = test_pa_score.precision(ScoreType.PointAdjusted)
    test_pa_recall = test_pa_score.recall(ScoreType.PointAdjusted)
    print("Test PA F1")
    print(
        f'Test F1: {test_pa_f1:2.4f}  | \tTest precision: {test_pa_precision:2.4f}  | \tTest recall: {test_pa_recall:2.4f}\n')

    test_pw_f1 = test_pw_score.f1(ScoreType.PointAdjusted)
    test_pw_precision = test_pw_score.precision(ScoreType.PointAdjusted)
    test_pw_recall = test_pw_score.recall(ScoreType.PointAdjusted)
    print("Test PW F1")
    print(
        f'Test F1: {test_pw_f1:2.4f}  | \tTest precision: {test_pw_precision:2.4f}  | \tTest recall: {test_pw_recall:2.4f}\n')

    # writer = SummaryWriter()
    # for i in range(config.num_epoch):
    #     writer.add_scalars('loss', {'train': all_epoch_train_loss[i],
    #                                 'test': all_epoch_test_loss[i]}, i)
    # # writer.add_embedding(part_embedding_feature, metadata=part_embedding_target, tag='test embedding')
    # writer.close()

    return test_score_origin, test_affiliation, test_rpa_score, test_pa_score, test_pw_score, score_reasonable, predict

def mixup_data(x, y, alpha=1.0, device="cuda"):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if device == "cuda":
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam



def model_train(model, model_optimizer, train_loader, config, device, epoch):

    total_loss, total_f1, total_precision, total_recall = [], [], [], []
    all_target, all_predict = [], []

#在这mixup
    model.train()
    # torch.autograd.set_detect_anomaly(True)
    for batch_idx, (data, target) in enumerate(train_loader):
        # send to device
        data, target = data.float().to(device), target.float().to(device)
        # optimizer
        model_optimizer.zero_grad()
        #mixup的样本进去学
        # inputs, targets_a, targets_b, lam = mixup_data(data, target,
        #                                                config.alpha, device)
        # inputs, targets_a, targets_b = map(Variable, (data,
        #                                               targets_a, targets_b))

        if config.alpha > 0 and config.layer_mix != 0 and config.ab_mix:
            logits, y_a, y_b, lam = model(data, target, True, device)
            lam = torch.tensor(lam).to(device)
            target = lam * y_a + (1-lam) * y_b
            # print('logits1:',logits)
        else:
            logits = model(data, target, False)
            # print('logits2:',logits)
        # loss, score = cal_mix_loss_score(logits, targets_a, targets_b, config, lam)
        loss, score = train(logits, target, device)
        # Update hypersphere radius R on mini-batch distances
        total_loss.append(loss.item())

        loss.backward()
        model_optimizer.step()
        target = target.reshape(-1)

        predict = score.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        all_target.extend(target)
        all_predict.extend(predict)

    total_loss = torch.tensor(total_loss).mean()

    return all_target, all_predict, total_loss


def model_evaluate(model, test_dl, config, device, epoch):
    model.eval()
    total_loss, total_f1, total_precision, total_recall = [], [], [], []
    all_target, all_predict = [], []
    with torch.no_grad():
        for data, target in test_dl:
            data, target = data.float().to(device), target.float().to(device)
            logits = model(data, target, False)
            loss, score = train(logits, target, device)
            total_loss.append(loss.item())
            predict = score.detach().cpu().numpy()
            target = target.reshape(-1)
            all_target.extend(target.detach().cpu().numpy())
            all_predict.extend(predict)

    total_loss = torch.tensor(total_loss).mean()  # average loss
    all_target = np.array(all_target)

    return all_target, all_predict, total_loss


def train(logits, target, device):
    # normalize feature vectors
    loss_func = torch.nn.BCELoss()
    m1 = torch.nn.Sigmoid()
    pro = m1(logits)
    tar = torch.stack([1 - target, target], dim=1).to(device)
    loss = loss_func(pro, tar)
    # print('loss:', loss)
    score = pro[:, 1]
    return loss, score


# def mixup_criterion(criterion, pred, y_a, y_b, lam):
#     return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def cal_mix_loss_score(logits, targets_a, targets_b, config, lam):
    criterion = torch.nn.CrossEntropyLoss()
    loss = lam * criterion(logits, targets_a) + (1 - lam) * criterion(logits, targets_b)
    m = torch.nn.Softmax(dim=1)
    p = m(logits)
    score = p[:, 1]
    return loss, score





