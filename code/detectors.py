import numpy as np


class RDDM:
    def __init__(self, training_steps=50):
        self.min_avg, self.min_std = None, None
        self.warning_window = []
        self.warning_time = 0
        self.training_steps = training_steps
        self.curr_len, self.curr_sum = 0, 0

    def predict(self, sample, error):
        self.curr_len += 1
        self.curr_sum += error

        avg = float(self.curr_sum) / self.curr_len
        std = np.sqrt((1.0 - avg) * avg / self.curr_len)

        if self.min_avg is None or self.min_avg > avg:
            self.min_avg = avg
            self.min_std = std
            return -1, []

        if avg + std >= self.min_avg + 2 * self.min_std:
            # Warning
            # DEBUG
            if self.training_steps > self.curr_len:
                print("false warning ! its training sted !")
            self.warning_time = self.curr_len
            self.warning_window.append(sample)
            print("Warning")
            return -1, []

        if avg + std >= self.min_avg + 3 * self.min_std:
            # Detection
            print("Detection !")
            return self.warning_time, self.warning_window

        self.warning_window = []
        return -1, []


class WinRDDM:
    def __init__(self, win_len=20):
        self.time = 0
        self.min_avg, self.min_std = None, None
        self.warning_window, self.errors_window = [], []
        self.warning_time = 0
        self.window_len = win_len

    def predict(self, sample, error):
        self.time += 1
        if len(self.errors_window) < self.window_len :
            self.errors_window.append(error)
            return -1, []

        self.errors_window.append(error)
        self.pop(0)

        avg = float(sum(self.errors_window)) / self.window_len
        std = np.sqrt((1.0 - avg) * avg / self.window_len)

        if self.min_avg is None or self.min_avg > avg:
            self.min_avg = avg
            self.min_std = std
            return -1, []

        if avg + std >= self.min_avg + 2 * self.min_std:
            # Warning
            self.warning_time = self.time
            self.warning_window.append(sample)
            print("Warning")
            return -1, []

        if avg + std >= self.min_avg + 3 * self.min_std:
            # Detection
            print("Detection !")
            return self.warning_time, self.warning_window

        self.warning_window = []
        return -1, []

