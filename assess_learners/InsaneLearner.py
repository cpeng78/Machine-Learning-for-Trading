import numpy as np
import BagLearner as bl
import LinRegLearner as lrl
class InsaneLearner(object):
    def __init__(self, verbose=False):
        self.verbose = verbose
    def author(self):
        return "cpeng78"  # replace tb34 with your Georgia Tech username
    def add_evidence(self, data_x, data_y):
        self.learner_list = [bl.BagLearner(learner=lrl.LinRegLearner, kwargs={}, bags=20, boost=False, verbose=False) for i in range(20)]
        for learner in self.learner_list:
            learner.add_evidence(data_x, data_y)
    def query(self, data_x):
        return np.mean([learner.query(data_x) for learner in self.learner_list], axis=0)