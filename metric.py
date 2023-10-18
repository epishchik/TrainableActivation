from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import torch


def metrics(model, dataloader, device):
    model.eval()

    accuracy = 0.0
    precision = 0.0
    recall = 0.0
    f1 = 0.0

    length = len(dataloader)

    with torch.no_grad():
        for batch in dataloader:
            imgs, labels = batch
            imgs = imgs.to(device)

            labels = labels.to(device).cpu()
            preds = model(imgs).argmax(dim=1).cpu()

            tmp_accuracy = accuracy_score(labels, preds)
            tmp_precision = precision_score(labels, preds, average='weighted')
            tmp_recall = recall_score(labels, preds, average='weighted')
            tmp_f1 = f1_score(labels, preds, average='weighted')

            accuracy += tmp_accuracy
            precision += tmp_precision
            recall += tmp_recall
            f1 += tmp_f1

    accuracy /= length
    precision /= length
    recall /= length
    f1 /= length

    acc_tup = ('accuracy', accuracy)
    prec_tup = ('precision', precision)
    rec_tup = ('recall', recall)
    f1_tup = ('f1', f1)

    return acc_tup, prec_tup, rec_tup, f1_tup
