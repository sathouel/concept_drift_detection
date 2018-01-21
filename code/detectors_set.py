import numpy as np

def scorer(pred, label):
    return np.sum((pred - label) ** 2)

class DDSet:
    def __init__(self, detectors=[], lossFun=scorer):
        self.detectors = detectors
        self.lossFun = lossFun

    def addDetector(self, detector):
        if type(detector) is list:
            self.detectors += detector
            return

        self.detectors.append(detector)

    def reset(self):
        for detector in self.detectors:
            detector.reset()

    def predict(self, sample, error=None):
        if len(self.detectors) is 0:
            raise "DDSet is empty !"

        if error is None:
            error = self.lossFun(sample[0], sample[1])

        detection = False
        vector = []
        for detector in self.detectors:
            prediction = detector.predict(sample, error)
            vector.append(prediction)
            if prediction == 1:
                detection = True

        return detection, vector
    




