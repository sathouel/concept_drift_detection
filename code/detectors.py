import numpy as np


# Mapping : -1 - filling window, 0 - RAS, 1 - Warning, 2 - Detection
# Sample : (X, y)
class WinRDDM:
    def __init__(self, win_len=20, wrn_bd=2, dtc_bd=4):
        self.time = 0
        self.min_avg, self.min_std = None, None
        self.warning_window, self.errors_window = [], []
        self.warning_time, self.warning_bd, self.detection_bd = -1, wrn_bd, dtc_bd
        self.window_len = win_len

    def get_warning_params(self):
        return self.warning_time, self.warning_window

    def reset(self):
        self.min_avg, self.min_std = None, None
        self.warning_window, self.errors_window = [], []
        self.warning_time = -1

    def predict(self, sample, error):
        self.time += 1
        if len(self.errors_window) < self.window_len :
            self.errors_window.append(error)
            return -1

        self.errors_window.append(error)
        self.errors_window.pop(0)

        avg = float(sum(self.errors_window)) / self.window_len
        std = np.sqrt(sum((np.array(self.errors_window) - avg) ** 2) / self.window_len)

        print(" avg : ", avg, " std : ", std)
        print("min avg : ", self.min_avg, " min std : ", self.min_std)

        if self.min_avg is None or self.min_avg > avg:
            self.min_avg = avg
            self.min_std = std
            return 0

        if avg + std >= self.min_avg + self.detection_bd * self.min_std:
            # Detection
            print("Detection !")
            return 2

        if avg + std >= self.min_avg + self.warning_bd * self.min_std:
            # Warning
            self.warning_time = self.time
            self.warning_window.append(sample)
            print("Warning ", len(self.warning_window))
            return 1

        self.warning_window = []
        self.warning_time = -1
        return 0


class REDDM:
    def __init__(self, win_len=20, wrn_bd=0.95, dtc_bd=0.9, percentile=90):
        self.time = 0
        self.min_avg, self.min_std = None, None
        self.warning_window, self.errors_window = [], []
        self.warning_time, self.warning_bd, self.detection_bd = -1, wrn_bd, dtc_bd
        self.window_len = win_len
        self.percentile = percentile
        self.difs = []
        self.max_avg, self.max_err_time_avg, self.max_err_time_std = 0, 0, 0

    # check that the percentile is greater than the average before classifying it as error

    def predict(self, sample, error):
        self.time += 1
        if len(self.errors_window) < self.window_len:
            self.errors_window.append(error)
            return False

        qtile = np.percentile(np.array(self.errors_window), self.percentile)
        avg = np.mean(np.array(self.errors_window))

        if error > qtile and error > 1.2 * avg:
            if self.max_avg == 0:
                self.max_avg = self.time
                return False
            self.difs.append(self.time - self.max_avg)
            self.max_avg = 0

            curr_avg = np.mean(self.difs)
            curr_std = np.std(self.difs)

            self.max_err_time_avg = self.max_err_time_avg if self.max_err_time_avg > curr_avg else curr_avg
            self.max_err_time_std = self.max_err_time_std if self.max_err_time_avg > curr_avg else curr_std

            if (curr_avg + 2 * curr_std) / (self.max_err_time_avg + 2 * self.max_err_time_std) < self.warning_bd:
                print("Warning !")
                self.warning_window.append(sample)
                self.warning_time.append(self.time)
                return False

            if (curr_avg + 2 * curr_std) / (self.max_err_time_avg + 2 * self.max_err_time_std) < self.detection_bd:
                print("Detection !")
                return True

        self.errors_window.append(error)
        self.errors_window.pop(0)
        return False




