import copy
import pandas as pd
import os
import threading
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
import numpy as np
from multiprocessing import Pool

# from autogluon.tabular import TabularDataset, TabularPredictor
from scipy import stats
from src.pre_process.finetune import labeling

inputDir = "/data/UCSBFinetuning_with_payload/"
dirLs = os.listdir(inputDir)
args = []
dataList = []
futures = []
semaphore = threading.Semaphore(1)
colDict = {}
num_packets = 5


def diff_p_value(diff):
    ref = np.zeros_like(diff)
    t, p = stats.ttest_rel(diff, ref, alternative="greater")
    return p


def padOrTrunk(nptDf):
    cols = nptDf.columns.values.tolist()
    # retDf = pd.DataFrame(colDict)
    dictOfCols = {}
    for i in range(num_packets):
        for col in cols:
            if i < nptDf.shape[0]:
                dictOfCols[str(col) + "_" + str(i)] = pd.Series([nptDf[col][i]])
                # retDf[str(col) + "_" + str(i)] = [nptDf[col][i]]
            else:
                dictOfCols[str(col) + "_" + str(i)] = pd.Series([-1] * nptDf.shape[0])
                # retDf[str(col) + "_" + str(i)] = [-1]
    retDf = pd.DataFrame(dictOfCols)
    return retDf


def getNptData(nptFile, label, opfile):
    nptDf = pd.read_csv(nptFile)
    nptDf = padOrTrunk(nptDf)
    nptDf["label"] = label
    nptDf.to_pickle(opfile)


def _paralell_process(func, input_args, cores=0):
    if cores == 0:
        cores = os.cpu_count()
    with Pool(cores) as p:
        return p.starmap(func, input_args)


def waitAndWritePkl(path):
    global dataList
    global futures
    doneCount = 0
    print("in writing")
    while doneCount < len(futures):
        time.sleep(10)
        doneCount = 0
        for i in futures:
            if i.done():
                doneCount += 1
        print(doneCount)
    df = pd.concat(dataList)
    print(df.shape)
    print("Writing to : " + path)
    pd.to_pickle(df, path)
    dataList = []


dataset_path = "/data/patator_multicloud_npt_data.csv"
# dataset_path = "/data/patator_multicloud_npt_data.csv"
# dataset_path = "/data/cicids-npts-nprint.pkl"
# dataset_path = "/data/UCSB-nprint.csv"
labelingObj = labeling.Labeling.get_sig2021service_labeling()
if not os.path.exists(dataset_path):
    balancingList = {}
    for i in range(11):
        balancingList[i] = 0
    count = 0
    tmpDf = pd.read_csv(inputDir + dirLs[0] + "/" + dirLs[0] + ".npt")

    if len(colDict.keys()) == 0:
        cols = tmpDf.columns.values.tolist()
        retDf = pd.DataFrame()
        for i in range(num_packets):
            for col in cols:
                colDict[str(col) + "_" + str(i)] = []
        colDict["label"] = []

    for f in dirLs:
        label = 0
        for fil in os.listdir(inputDir + f):
            if ".txt" in fil:
                label = labelingObj.label_func(fil.split(".txt")[0])
                if label != None and label != -1:
                    if balancingList[label] >= 10000:
                        break
                    balancingList[label] += 1
                    args.append(
                        (
                            inputDir + f + "/" + f + ".npt",
                            label,
                            "/data/UCSB-collected-nprintOP/" + f + ".pkl",
                        )
                    )
                    # futures.append(executor.submit(getNptData, nptFile = inputDir + f +"/"+f+".npt", label = label))
                    count += 1
                    print(count)
                break
    _paralell_process(getNptData, args)

    exit(-1)
    count = 0
    # for label in range(0,2,1):
    #     print("started for label: "+ str(label))
    #     for f in os.listdir(inputDir+str(label)):
    #         args.append((inputDir + str(label) +"/"+f, label, "/data/patator-multi-cloud-npts-5pkts/"+f+".pkl"))
    #         count += 1
    #         # if count == 200000:
    #         #     break
    #         print(count)
    #     print("finished label : "+str(label))
    #     count = 0
    # _paralell_process(getNptData, args)
    exit(-1)
df = pd.read_csv(dataset_path)
# df = pd.concat([df[df.label==i] for i in range(8)])
df = df.dropna()
X = df.drop(
    ["label", "src_ip_0", "src_ip_1", "src_ip_2", "src_ip_3", "src_ip_4"], axis=1
).reset_index(drop=True)
dropls = []
for i in range(num_packets):
    for j in range(32):
        dropls.append("ipv4_dst_" + str(j) + "_" + str(i))
        dropls.append("ipv4_src_" + str(j) + "_" + str(i))
    for j in X.columns.values.tolist():
        if (
            "flow" in j
            or "Timestamp" in j
            or "prt" in j
            or "port" in j
            or "chksum" in j
            or "proto" in j
        ):  # or "ttl" in j or "seq" in j or "ack" in j:
            dropls.append(j)
        if "payload" in j:
            if int(j.split("_")[2]) > 96:
                dropls.append(j)
p = 0.4
X2 = X.drop(dropls, axis=1).reset_index(drop=True)
macrof1 = []
macroprec = []
macrorecall = []
for i in range(3):
    X = copy.deepcopy(X2)
    # addls = []
    # for i in range (num_packets):
    #     for j in X.columns.values.tolist():
    #         if "payload" in j:#
    #             addls.append(j)
    # X=X[addls]
    Y = copy.deepcopy(df["label"].reset_index(drop=True))
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    Y_train = Y_train.astype(int)
    Y_test = Y_test.astype(int)
    # num_noise_samples = int((1-(i+1)*0.2)*X_train.shape[0])
    # num_noise_samples = int(p * X_train.shape[0])
    # indices = random.sample(range(0, X_train.shape[0]), num_noise_samples)
    # noisy_labels = np.random.random_integers(0, 10, size=(num_noise_samples,))
    # Y_train.iloc[indices] = noisy_labels
    # X_train = X_train.iloc[indices]
    # Y_train = Y_train.iloc[indices]
    clf = RandomForestClassifier(random_state=0)
    clf.fit(X_train, Y_train)
    y_pred = clf.predict(X_test)
    print(np.mean((y_pred == Y_test)))
    print(classification_report(Y_test, y_pred, digits=5))
    macrof1.append(f1_score(Y_test, y_pred, average="macro"))
    macroprec.append(precision_score(Y_test, y_pred, average="macro"))
    macrorecall.append(recall_score(Y_test, y_pred, average="macro"))

    # trustee = ClassificationTrustee(expert=clf)
    # trustee.fit(X_train, Y_train, num_iter=50, num_stability_iter=10, samples_size=0.3, verbose=True)
    # dt, pruned_dt, agreement, reward = trustee.explain()
    # dt_y_pred = dt.predict(X_test)
    #
    # print("Model explanation global fidelity report:")
    # print(classification_report(y_pred, dt_y_pred))
    # print("Model explanation score report:")
    # print(classification_report(Y_test, dt_y_pred))
    # graph = Source(tree.export_graphviz(dt, out_file=None, filled=True, rounded=True, special_characters=True, feature_names=X_train.columns))
    # png_bytes = graph.pipe(format='png')
    # with open('/data/NprinttreeRF.png','wb') as f:
    #     f.write(png_bytes)
print(diff_p_value(0.8385 - np.array(macroprec)))
print(diff_p_value(0.8342 - np.array(macrorecall)))
print(diff_p_value(0.8345 - np.array(macrof1)))
print(np.std(macroprec))
print(np.std(macrorecall))
print(np.std(macrof1))
print(np.mean(macroprec))
print(np.mean(macrorecall))
print(np.mean(macrof1))
print(macroprec)
print(macrorecall)
print(macrof1)
