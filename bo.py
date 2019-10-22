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
    torun = """spark-submit --master spark://10.0.2.15:7077 --driver-cores 1 --driver-memory 1G --total-executor-cores 2 --executor-cores 1 --executor-memory 1G --py-files /home/test/bd/spark/lib/bigdl-0.8.0-python-api.zip,/home/test/bd/codes/lenet5.py --properties-file /home/test/bd/spark/conf/spark-bigdl.conf --jars /home/test/bd/spark/lib/bigdl-SPARK_2.3-0.8.0-jar-with-dependencies.jar --conf spark.driver.extraClassPath=/home/test/bd/spark/lib/bigdl-SPARK_2.3-0.8.0-jar-with-dependencies.jar --conf spark.executer.extraClassPath=bigdl-SPARK_2.3-0.8.0-jar-with-dependencies.jar /home/test/bd/codes/lenet5.py --action train --dataPath /tmp/mnist --batchSize {0} --endTriggerNum {1} --learningRate {2} --learningrateDecay {3}"""\
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

def f(settings):
    alpha = 0.85
    settings = Settings(2**settings[0], settings[1], settings[2], settings[3])
    print(json.dumps(settings.__dict__))
    res = run_func(settings)
    res = res[-1]

    if alpha == res.acc:
        cost = 99999999
    else:
        cost = 20 / (res.acc - alpha)

    if cost < 0:
        cost = 99999999

    cost = cost + res.clock

    res.cost = cost
    print(json.dumps(res.__dict__))

    with open('out.json', 'a') as f:
        f.write(json.dumps(settings.__dict__))
        f.write("\n")
        f.write(json.dumps(res.__dict__))
        f.write("\n")

    return cost


calls = 50
random_calls = 50

res = gp_minimize(f, [(6, 10), (3, 15), (0.00001, 0.1), (0.00001, 0.001)], n_calls=calls, n_random_starts=random_calls, acq_func="EI")
