import numpy as np

# Utils

def scorer(pred, label):
    return np.sum((pred - label) ** 2)

# ----- Hyper params ---- #
SOFT_RESET = 0
HALF_RESET = 1
WRN_RESET = 2
HARD_RESET = 3

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
    def reset_window(self, new_model=None, reset_mode=HALF_RESET, scorer=scorer):
            
        if reset_mode == HALF_RESET:
            half = int(len(self.errors_window)/2)
            self.errors_window = self.errors_window[half:]
            self.data = self.data[half:]
        elif reset_mode == WRN_RESET:
            n_el_to_add = self.time - self.last_warning_time_recorded
            idx = len(self.errors_window) - n_el_to_add
            self.errors_window = self.errors_window[idx:]
            self.data = self.data[idx:]
        elif reset_mode == HARD_RESET:
            self.errors_window = []
            self.data = []
            return

        if new_model is not None:
            self.errors_window = [] 
            for sample, label in self.data:
                pred = new_model.predict([sample])
                error = scorer(pred, label)
                self.errors_window.append(error)


    # function called after cd detected
    def reset(self, new_model=None, reset_mode=HALF_RESET, scorer=scorer):
        self.min_avg, self.min_std = None, None
        self.warning_window = []
        self.warning_time_updated = False
        self.warning_time = -1
        self.reset_window(new_model=new_model, reset_mode=reset_mode, scorer=scorer)

        
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
    
    def get_time(self):
        return self.time


class DetectorsSet:

    def __init__(self, win_lens, wrn_bds, dtc_bds, reset_mode=HALF_RESET, base_detector=WinRDDM, reset_thr=None, verbose=False):
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
        self.reset_mode = reset_mode

    def reset_detectors(self, new_model=None):
        for i in range(len(self.detectors_set)):
            self.detectors_set[i].reset_window(new_model=new_model, reset_mode=self.reset_mode)

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

    def get_max_warning_window(self):
        max_wrn_win, max_len = None, 0
        for d in self.detectors_set:
            curr_win, _ = d.get_warning_params()
            max_wrn_win = max_wrn_win if max_len >= len(curr_win) else curr_win
            max_len = max_len if max_len >= len(curr_win) else len(curr_win)

        return max_wrn_win

# NOTES : need to add avg warning time and sample window for storing samples and return samples of the new concept after cdd (for retraining the model)
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

    def get_new_training_set(self):
        return self.detectors_set.get_max_warning_window()

    def reset_detectors(self, new_model=None):
        self.detectors_set.reset_detectors(new_model=new_model)


