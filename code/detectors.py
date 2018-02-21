import numpy as np

# Utils

def scorer(pred, label):
    return np.sum((pred - label) ** 2)

# Mapping : -1 - filling window, 0 - RAS, 1 - Warning, 2 - Detection
# Sample : (X, y)
class WinRDDM:
    def __init__(self, win_len=20, wrn_bd=2, dtc_bd=4, verbose=False):
        self.verbose = verbose
        self.time = 0
        self.min_avg, self.min_std = None, None
        self.warning_window, self.errors_window, self.data = [], [], []
        self.warning_time, self.warning_bd, self.detection_bd = -1, wrn_bd, dtc_bd
        self.last_warning_time_recorded, self.last_warning_window_recorded = -1, None
        self.window_len = win_len
        self.warning_time_updated = False

    def get_warning_params(self):
        return self.last_warning_window_recorded, self.last_warning_time_recorded


    # refill error window from res_time according to new model
    def reset_window(self, new_model, res_time=None, scorer=scorer):
        if res_time is None:
            res_time = self.warning_time
            if self.warning_time < 0:
                res_time = self.last_warning_time_recorded

        n_el_to_add = self.time - res_time
        self.data.reverse()
        self.data = self.data[:n_el_to_add]
        self.errors_window = []
        for sample, label in self.data:
            pred = new_model.predict([sample])
            error = scorer(pred, label)
            self.errors_window.insert(0, error)
            if len(self.errors_window) == self.window_len:
                break
        self.data.reverse()
        self.reset(True)


    # function called after cd detected
    def reset(self, half_err_win=True):
        self.min_avg, self.min_std = None, None
        self.warning_window = []
        self.warning_time_updated = False
        # possible improvement
        if half_err_win:
            half = int(len(self.errors_window)/2)
            self.errors_window = self.errors_window[half:]


    def predict(self, sample, pred, label, scorer=scorer):        
        self.time += 1
        error = scorer(pred, label)
        if len(self.errors_window) < self.window_len :
            self.warning_time = -1
            self.errors_window.append(error)
            self.data.append((sample, label))
            return 0

        self.errors_window.append(error)
        self.errors_window.pop(0)

        self.data.append((sample, label))
        self.data.pop(0)

        avg = float(sum(self.errors_window)) / self.window_len
        std = np.sqrt(sum((np.array(self.errors_window) - avg) ** 2) / self.window_len)

        if self.verbose:
            print(" avg : ", avg, " std : ", std)
            print("min avg : ", self.min_avg, " min std : ", self.min_std)

        if self.min_avg is None or self.min_avg > avg:
            self.min_avg = avg
            self.min_std = std
            self.warning_time_updated = False
            return 0

        if avg + std >= self.min_avg + self.detection_bd * self.min_std:
            # Detection
            if self.verbose:
                print("Detection !")
            self.last_warning_time_recorded = self.warning_time
            self.last_warning_window_recorded = self.warning_window.copy()

            # re init params
            self.reset()
            return 2

        if avg + std >= self.min_avg + self.warning_bd * self.min_std:
            # Warning
            if self.warning_time_updated is False:
                self.warning_time = self.time
                self.warning_time_updated = True
                self.warning_window = []

            self.warning_window.append((sample, label))
            if self.verbose:
                print("Warning ", len(self.warning_window))
            return 1

        self.warning_time_updated = False
        self.warning_time = -1
        return 0


class DetectorsSet:

    def __init__(self, win_lens, wrn_bds, dtc_bds, base_detector=WinRDDM, reset_thr=None, verbose=False):
        if type(win_lens) != list or type(wrn_bds) != list or type(dtc_bds) != list:
            print('ERROR TYPE list objects expected')
            return

        self.detectors_set = []
        for win_len in win_lens:
            for wrn_bd in wrn_bds:
                for dtc_bd in dtc_bds:
                    new_detector = base_detector(win_len=win_len ,wrn_bd=wrn_bd, dtc_bd=dtc_bd)
                    self.detectors_set.append(new_detector)

        self.detection_counter = 0
        self.reset_thr = reset_thr

    def reset_detectors(self, new_model):
        for i in range(len(self.detectors_set)):
            self.detectors_set[i].reset_window(new_model)

    def predict(self, sample, pred, label, scorer=scorer):
        sol = []
        for d in self.detectors_set:
            prediction = d.predict(sample, pred, label, scorer)
            sol.append(prediction)
            if prediction == 2:
                self.detection_counter += 1

        if self.reset_thr != None and self.detection_counter >= int(self.reset_thr*len(self.detectors_set)):
            self.detection_counter = 0
            self.reset_detectors()
            
        return sol

class MainDetector:

    def __init__(self, detectors_set, clf):
        self.detectors_set = detectors_set
        self.clf = clf

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, sample, pred, label, scorer=scorer):
        x = self.detectors_set.predict(sample, pred, label, scorer=scorer)
        cd_pred = self.clf.predict([x])
        return cd_pred

    def reset_detectors(self, new_model):
        self.detectors_set.reset_detectors(new_model)


# class REDDM:
#     def __init__(self, win_len=20, wrn_bd=0.95, dtc_bd=0.9, percentile=90):
#         self.time = 0
#         self.min_avg, self.min_std = None, None
#         self.warning_window, self.errors_window = [], []
#         self.warning_time, self.warning_bd, self.detection_bd = -1, wrn_bd, dtc_bd
#         self.window_len = win_len
#         self.percentile = percentile
#         self.difs = []
#         self.max_avg, self.max_err_time_avg, self.max_err_time_std = 0, 0, 0

#     # check that the percentile is greater than the average before classifying it as error

#     def predict(self, sample, error):
#         self.time += 1
#         if len(self.errors_window) < self.window_len:
#             self.errors_window.append(error)
#             return False

#         qtile = np.percentile(np.array(self.errors_window), self.percentile)
#         avg = np.mean(np.array(self.errors_window))

#         if error > qtile and error > 1.2 * avg:
#             if self.max_avg == 0:
#                 self.max_avg = self.time
#                 return False
#             self.difs.append(self.time - self.max_avg)
#             self.max_avg = 0

#             curr_avg = np.mean(self.difs)
#             curr_std = np.std(self.difs)

#             self.max_err_time_avg = self.max_err_time_avg if self.max_err_time_avg > curr_avg else curr_avg
#             self.max_err_time_std = self.max_err_time_std if self.max_err_time_avg > curr_avg else curr_std

#             if (curr_avg + 2 * curr_std) / (self.max_err_time_avg + 2 * self.max_err_time_std) < self.warning_bd:
#                 print("Warning !")
#                 self.warning_window.append(sample)
#                 self.warning_time.append(self.time)
#                 return False

#             if (curr_avg + 2 * curr_std) / (self.max_err_time_avg + 2 * self.max_err_time_std) < self.detection_bd:
#                 print("Detection !")
#                 return True

#         self.errors_window.append(error)
#         self.errors_window.pop(0)
#         return False
