from skopt import gp_minimize
import subprocess
import re
import json


class Result:
    def __init__(self,epoch, iter, clock, correct, count, acc):
        self.epoch = int(epoch)
        self.iter = int(iter)
        self.clock = float(clock)
        self.correct = int(correct)
        self.count = int(count)
        self.acc = float(acc)
        self.cost = 0

    def __repr__(self):
        return json.dumps(self.__dict__)

class Settings:
    def __init__(self, batchSize, epoch, learningRate, learningRateDecay):
        self.batchSize = str(batchSize)
        self.epoch = str(epoch)
        self.learningRate = str(learningRate)
        self.learningRateDecay = str(learningRateDecay)

    def __repr__(self):
        return json.dumps(self.__dict__)


def run_func(settings):
    torun = """spark-submit --master spark://10.0.2.15:7077 --driver-cores 1 --driver-memory 1G --total-executor-cores 12 --executor-cores 4 --executor-memory 4G --py-files /home/test/bd/spark/lib/bigdl-0.8.0-python-api.zip,/home/test/bd/codes/lenet5.py --properties-file /home/test/bd/spark/conf/spark-bigdl.conf --jars /home/test/bd/spark/lib/bigdl-SPARK_2.3-0.8.0-jar-with-dependencies.jar --conf spark.driver.extraClassPath=/home/test/bd/spark/lib/bigdl-SPARK_2.3-0.8.0-jar-with-dependencies.jar --conf spark.executer.extraClassPath=bigdl-SPARK_2.3-0.8.0-jar-with-dependencies.jar /home/test/bd/codes/lenet5.py --action train --dataPath /tmp/mnist --batchSize {0} --endTriggerNum {1} --learningRate {2} --learningrateDecay {3}"""\
    .format(settings.batchSize, settings.epoch, settings.learningRate, settings.learningRateDecay)

    str = subprocess.check_output(torun, shell=True)

    # Epoch: 1
    # Iter: 2
    # Clock: 3
    # Correct: 4
    # Count: 5
    # Acc: 6
    pattern = re.compile(r"\[Epoch ([0-9]+) .*\]\[Iteration ([0-9]+)\]\[Wall Clock ([0-9]+\.[0-9]+)s] Top1Accuracy is Accuracy\(correct: ([0-9]+), count: ([0-9]+), accuracy: (0.[0-9]+)\)")

    ress = []
    for match in pattern.finditer(str):
        epoch = match.group(1)
        iter = match.group(2)
        clock = match.group(3)
        correct = match.group(4)
        count = match.group(5)
        acc = match.group(6)
        res = Result(epoch, iter, clock, correct, count, acc)
        ress.append(res)

    return ress


settings_random = Settings(256, 4, 0.09163533775602171, 0.0004024987748515579)
settings_bo = Settings(512, 7, 0.0997846527896473, 0.00010164576228415226)
settings_default = Settings(128, 5, 0.01, 0.0002)

s = [settings_default, settings_bo, settings_random]
names = ["default", "bo", "random"]

calls = 10
cores = 2

resses = {}

for i in range(len(s)):
    sett = s[i]
    name = names[i]
    resses[name] = []
    for j in range(calls):
        print("Running {} for the {} time".format(name, str(j)))
        res = run_func(sett)[-1]
        resses[name].append(res)
        print(res)

print(resses)

with open('benchmark.json', 'w') as outfile:
    outfile.write(str(resses))
