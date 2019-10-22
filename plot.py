import matplotlib.pyplot as plt
import numpy as np

def read_data(path):
    with open(path, 'r') as file:
        f = file.readlines()[0]
    return eval(f)
y = [2, 4, 8, 16]
dataNames = ["data2", "data4", "data8", "data16"]
datasVals = list(map(read_data, dataNames))
data = {}
for i in range(len(dataNames)):
    data[dataNames[i]] = datasVals[i]

acc = {}
accStd = {}
clock = {}
clockStd = {}

for nCores, value in data.items():
    for model, valModel in value.items():
        if model not in acc:
            acc[model] = []
            accStd[model] = []
            clock[model] = []
            clockStd[model] = []


        accArr = np.array(list(map(lambda x: x['acc'], valModel)))
        clockArr = np.array(list(map(lambda x: x['clock'], valModel)))

        acc[model].append(np.mean(accArr))
        accStd[model].append(np.std(accArr))
        clock[model].append(np.mean(clockArr))
        clockStd[model].append(np.std(clockArr))

for model in acc.keys():
    plt.errorbar(y, acc[model], label=f"{model}", yerr=accStd[model])
    plt.ylim(0.8, 1)
    plt.xlabel("Total amount of cores")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("all-acc")

plt.figure()

for model in acc.keys():
    plt.errorbar(y, clock[model], label=f"{model}", yerr=clockStd[model])
    plt.ylim(30, 100)
    plt.xlabel("Total amount of cores")
    plt.ylabel("Training time")
    plt.legend()
    plt.savefig("all-train")
