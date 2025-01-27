import copy
import random
import sys
import threading

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import Curtains_model

mask = 0
num_masked_burst = 10
batch_size = 128
classPreds = []


def train(args, X, Y, X2, opDir, p=0):
    global classPreds
    interEpochAvgTrainLoss = []
    interEpochAvgTrainPredScore = []
    interEpochAvgTrainRelAcc = []
    interEpochAvgTestLoss = []
    interEpochAvgTestPredScore = []
    interEpochAvgTestRelAcc = []
    print(X.shape)
    print("creating model")

    model = Curtains_model.CurtainsLSTMModel().to(args["device"])
    optimizer = torch.optim.AdamW(model.parameters(recurse=True), lr=1e-3)
    lr_decay = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)
    loss = torch.nn.CrossEntropyLoss()
    bestLoss = sys.float_info.max
    prev_score = sys.float_info.min

    num_batch = X.shape[0] // batch_size

    print("running model")
    testBatchSet = set(np.random.randint(0, num_batch, int(0.3 * num_batch)))
    trainBatchSet = np.delete(
        np.arange(0, X.shape[0], dtype=int), np.array(list(testBatchSet), dtype=int)
    )
    if p > 0:
        num_noise_samples = int(p * len(trainBatchSet))
        indices = random.sample(range(0, len(trainBatchSet)), num_noise_samples)
        # noisy_labels = np.random.random_integers(0, 10, size=(num_noise_samples,))
        # noisy_labels = torch.nn.functional.one_hot(torch.from_numpy(noisy_labels), num_classes=11).to(args["device"],
        #                                                                                              torch.float32)
        # Y[indices] = torch.from_numpy(noisy_labels).to(Y.device)
        trainBatchSet = trainBatchSet[indices]

    for epoch in range(20):
        global classPreds
        classPreds = np.zeros(args["outputSize"])
        model.train()
        print("epoch: " + str(epoch))
        groundTruth = []
        outputs = []
        pred_score = 0
        relAccuracy = 0
        trainLoss = []
        testLoss = []
        for batch in range(num_batch):
            optimizer.zero_grad()

            if batch not in trainBatchSet:
                continue
            X_tmp, Y_tmp, X2_tmp = getDataForModel(X, Y, X2, batch)
            # print(f"size of x_tmp {len(X_tmp)}")
            # print(f"size of x2_tmp {len(X2_tmp)}")
            # print(f"size of y_tmp {len(Y_tmp)}")
            output = model(X_tmp, X2_tmp)
            outputs.extend(
                np.array(
                    [np.argmax(i) for i in output.cpu().detach().numpy()],
                    dtype=np.dtype("float"),
                )
            )
            groundTruth.extend(Y_tmp.cpu().detach().numpy())
            # print(f"size of output {len(output)}")
            # print(batch)
            l = loss(output, Y_tmp)
            # print(l)

            pred_score, relAccuracy = calculateClassificationPreds(
                output, Y_tmp, pred_score, relAccuracy, batch
            )
            trainLoss.append(l.item())
            l.backward()
            optimizer.step()
        # print("Learning rate: "+ str(lr_decay.get_lr()))
        lr_decay.step()
        interEpochAvgTrainPredScore.append(pred_score)
        interEpochAvgTrainRelAcc.append(relAccuracy)
        interEpochAvgTrainLoss.append(np.mean(trainLoss))
        pd.DataFrame(trainLoss).to_csv(opDir + "/trainloss" + str(epoch) + ".csv")
        model.eval()
        print(pred_score)
        print(np.mean(trainLoss))
        pred_score = 0
        relAccuracy = 0
        classPreds = np.zeros(args["outputSize"])
        Y_gt = []
        yPreds = []
        for batch in testBatchSet:
            X_tmp, Y_tmp, X2_tmp = getDataForModel(X, Y, X2, batch)

            output = model(X_tmp, X2_tmp)
            pred_score, relAccuracy = calculateClassificationPreds(
                output, Y_tmp, pred_score, relAccuracy, batch
            )
            l = loss(output, Y_tmp)
            testLoss.append(l.item())
            Y_gt.append(Y_tmp.detach().cpu())
            yPreds.append(output.detach().cpu())
        if prev_score < pred_score:
            # pd.DataFrame([torch.concat(Y_gt, dim=0).squeeze(1).detach().cpu().numpy(),torch.concat(yPreds, dim=0).squeeze(1).detach().cpu().numpy()]).transpose().to_pickle("/data/compareOP.pkl")
            prev_score = pred_score
        interEpochAvgTestPredScore.append(pred_score)
        interEpochAvgTestRelAcc.append(relAccuracy)
        interEpochAvgTestLoss.append(np.mean(testLoss))

        pd.DataFrame(testLoss).to_csv(opDir + "/testloss" + str(epoch) + ".csv")
        if np.mean(testLoss) < bestLoss:
            torch.save(model.state_dict(), opDir + "/model.pt")
        with open(opDir + "/classification_rep.txt", "a") as f:
            f.write(
                classification_report(
                    torch.concatenate([Y_gt[i] for i in range(len(Y_gt))]),
                    torch.concatenate(
                        [yPreds[i].argmax(dim=1) for i in range(len(yPreds))]
                    ),
                    digits=5,
                )
            )
            f.write("\n")
        print(
            classification_report(
                torch.concatenate([Y_gt[i] for i in range(len(Y_gt))]),
                torch.concatenate(
                    [yPreds[i].argmax(dim=1) for i in range(len(yPreds))]
                ),
                digits=5,
            )
        )
        print(
            "f1: "
            + str(f1_score(np.array(groundTruth), np.array(outputs), average="macro"))
        )
        print(
            "precision: "
            + str(
                precision_score(
                    np.array(groundTruth), np.array(outputs), average="macro"
                )
            )
        )
        print(
            "recall: "
            + str(
                recall_score(np.array(groundTruth), np.array(outputs), average="macro")
            )
        )
    pd.DataFrame(interEpochAvgTestPredScore).to_csv(
        opDir + "/interEpochAvgTestPredScore.csv"
    )
    pd.DataFrame(interEpochAvgTestRelAcc).to_csv(opDir + "/interEpochAvgTestRelAcc.csv")
    pd.DataFrame(interEpochAvgTrainPredScore).to_csv(
        opDir + "/interEpochAvgTrainPredScore.csv"
    )
    pd.DataFrame(interEpochAvgTrainRelAcc).to_csv(
        opDir + "/interEpochAvgTrainRelAcc.csv"
    )
    pd.DataFrame(interEpochAvgTestLoss).to_csv(opDir + "/interEpochAvgTestLoss.csv")
    pd.DataFrame(interEpochAvgTrainLoss).to_csv(opDir + "/interEpochAvgTrainLoss.csv")


def getDataForModel(X, Y, X2, batch):
    X_tmp = X[batch * batch_size : min((batch + 1) * batch_size, X.shape[0]), :]
    Y_tmp = Y[batch * batch_size : min((batch + 1) * batch_size, Y.shape[0])]
    X2_tmp = X2[batch * batch_size : min((batch + 1) * batch_size, X2.shape[0]), :]
    return X_tmp, Y_tmp, X2_tmp


def calculateClassificationPreds(output, Y_tmp, pred_score, relAccuracy, batch):
    global classPreds
    ops = np.array([np.argmax(i) for i in output.cpu().detach().numpy()])
    batch_size = len(output.cpu().detach())
    ys = np.array([np.argmax(i) for i in Y_tmp.cpu().detach().numpy()])
    diff = np.abs(ops != ys)
    for i in range(len(ops)):
        if ops[i] == ys[i]:
            classPreds[ys[i]] += 1
    pred_score = pred_score + (1 - (np.sum(diff) / batch_size) - pred_score) / (
        batch + 1
    )
    # TODO: what should be the value of relAccuracy? it was commneted
    # relAccuracy += (np.mean(np.abs(ys-ops/(ys+0.0001)))-relAccuracy)/(batch+1)
    # print(pred_score)
    # print(f"pred score {pred_score}is and relAccuracy is {relAccuracy}")
    return pred_score, relAccuracy


count0 = []
semaphore = threading.Semaphore(1)


def dummyFlowData(args, path, path2):
    unchangedDf = pd.read_pickle(path)
    df = unchangedDf[["PacketSize", "Direction", "InterArrivalTime"]]
    # df2 = pd.Series([i[0] for i in np.array_split(unchangedDf["Label"].to_numpy(), len(unchangedDf["Label"]) / 1024)])
    df2File = pd.Series(
        [
            i[0]
            for i in np.array_split(
                unchangedDf["FileID"].to_numpy(), len(unchangedDf["FileID"]) / 1024
            )
        ]
    )
    # df2File = pd.Series([i[0] for i in np.array_split(unchangedDf["FileID"].to_numpy(), len(unchangedDf["FileID"]) / 256)])
    nparr = np.array_split(df.to_numpy(), len(df) / 1024)
    # nparr = [i[:100, :] for i in nparr]
    del unchangedDf
    unchangedDf = pd.read_pickle(path2)

    # this line
    unchangedDf.replace([np.inf, -np.inf], np.nan, inplace=True)
    unchangedDf = unchangedDf.dropna().reset_index(drop=True)
    cicLabels = unchangedDf["Label"]
    unchangedDf = unchangedDf.drop(
        columns=[
            "Flow ID",
            "Src IP",
            "Src Port",
            "Dst IP",
            "Dst Port",
            "Protocol",
            "Timestamp",
            "Label",
        ]
    )
    # unchangedDf = unchangedDf.drop(columns=["Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol", "Timestamp"])
    # labels = unchangedDf["Label"]
    files = unchangedDf["FileID"]
    unchangedDf = unchangedDf.drop(columns=["FileID"])
    unchangedDf = unchangedDf.astype(float)
    unchangedDf = (
        (unchangedDf - unchangedDf.mean()) / (unchangedDf.std() + 0.6)
    ).dropna(axis=1)
    unchangedDf["Label"] = cicLabels
    fileToStats = {}
    # unchandeddf: for cicflow
    # labels: for cicflow
    for i in range(len(files)):
        fileToStats[str(files[i]).split(".")[0]] = unchangedDf.iloc[i]

    orderedFlowStats = []
    orderedFlowTS = []
    orderedFlowLabels = []
    # going over all the rows of df2
    # print(f"Number of rows is {df2.shape[0]}")
    # df2 for time series
    filter7LabelCount = 0
    for i in range(df2File.shape[0]):
        theFile = str(df2File.iloc[i]).split(".")[0]
        # thelabel = int(str(df2File.iloc[i]).split(".")[0][0])
        if theFile in fileToStats.keys():
            # if ("7label" in thelabel):
            #     if filter7LabelCount >= 5000:
            #         continue
            #     filter7LabelCount +=1
            cicfl = fileToStats[str(theFile)].reset_index(drop=True)
            thelabel = int(cicfl[76])  # int(df2.iloc[i])
            cicfl = cicfl[:76]
            orderedFlowStats.append(cicfl)
            orderedFlowTS.append(nparr[i])
            orderedFlowLabels.append(int(thelabel))

    X_stats = (
        torch.from_numpy(pd.concat(orderedFlowStats, axis=1).transpose().to_numpy())
        .to(args["device"])
        .to(torch.float32)
    )
    X_LSTM = (
        torch.from_numpy(np.array(orderedFlowTS)).to(args["device"]).to(torch.float32)
    )
    Y = pd.Series(
        orderedFlowLabels,
    )

    return X_LSTM, Y, X_stats


args = {}
args["outputSize"] = (
    11  # the output dimension of decoder (the dimension of each burst).
)
args["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Xlist, Y , X2= dummyFlowData(args, "/data/Curtains-CICIDS-2017-timeseries-data.pkl", "/data/Curtains-CICIDS-2017-CICflow-data.pkl")
# Xlist, Y , X2= dummyFlowData(args, "/data/UCSBTimeseries_spark.pkl", "/data/UCSB_CICFLOW.pkl")
Xlist, Y, X2 = dummyFlowData(
    args,
    "/data/patator-multi-cloud-Curtains-timeseries_merged_spark.csv",
    "/data/patator-multi-cloud-CICflow-data.csv",
)

# TODO: Should we change this part for UCSB data?
# I think since we alread have the labels, the value of y should be set to labels in
# dummyFlow data
# TODO: Write it again or create a dictionary of file names and assign values based on that?
# TODO: or keep the mapping of label column and FileID
#  in a dictionary and then assign it to the series

# print(type(Y))
# print(Y[0])
# Y = Y.replace(regex=r'0label*', value=0)
# Y = Y.replace(regex=r'1label*', value=1)
# Y = Y.replace(regex=r'2label*', value=2)
# Y = Y.replace(regex=r'3label*', value=3)
# Y = Y.replace(regex=r'4label*', value=4)
# Y = Y.replace(regex=r'5label*', value=5)
# Y = Y.replace(regex=r'6label*', value=6)
# Y = Y.replace(regex=r'7label*', value=7)


# Y = Y.replace(regex=r'.*ftp.*', value=5)
# Y = Y.replace(regex=r'.*vimeo.*', value=2)
# Y = Y.replace(regex=r'.*spotify.*', value=2)
print(Y[0])
#
Y = torch.from_numpy(Y.to_numpy()).to(args["device"], torch.int64)

# Y= F.one_hot(Y, num_classes = args["outputSize"]).float().to(args["device"])
# train(args, Xlist, Y, X2, "/data/tbd", 0)
train(
    args,
    copy.deepcopy(Xlist),
    copy.deepcopy(Y),
    copy.deepcopy(X2),
    "/data/curtains-APT/0",
    0,
)
# train(args, copy.deepcopy(Xlist), copy.deepcopy(Y), copy.deepcopy(X2), "/data/missing_labels-curtains/0.2", 0.8)
# train(args, copy.deepcopy(Xlist), copy.deepcopy(Y), copy.deepcopy(X2), "/data/missing_labels-curtains/0.4", 0.6)
# train(args, copy.deepcopy(Xlist), copy.deepcopy(Y), copy.deepcopy(X2), "/data/missing_labels-curtains/0.6", 0.4)
# train(args, copy.deepcopy(Xlist), copy.deepcopy(Y), copy.deepcopy(X2), "/data/missing_labels-curtains/0.8", 0.2)
# train(args, Xlist, Y, X2, "/data/missing_labels-curtains/0.8", 0.2)
