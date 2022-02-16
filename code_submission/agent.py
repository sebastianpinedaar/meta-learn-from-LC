from functools import lru_cache
import random
from sqlite3 import enable_shared_cache
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sys import path
import os
from fsbo import FSBO, MLP
import torch
from torch.utils.data import DataLoader
from scipy.spatial import distance_matrix
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

import torch
from torch.utils.tensorboard import SummaryWriter


class Agent():
    def __init__(self, number_of_algorithms):
        """
        Initialize the agent

        Parameters
        ----------
        number_of_algorithms : int
            The number of algorithms

        """
        self.nA = number_of_algorithms
        self.numerical_dataset_metafeatures = ["time_budget", "feat_num", "target_num", "label_num", "train_num", "valid_num","test_num", "has_categorical", "has_missing", "is_sparse"]
        self.categorical_dataset_metafeatures = ["task", "target_type", "feat_type", "metric"]
        self.metatrained = 0
        self.observed_configs = []
        self.observed_response = []
        self.observed_cost = []
        #self.device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"
        self.acquistion_type = "ei"
        #self.acquistion_type = "cost_balanced_ei"
        self.conf = {"kernel":"52", 
                    "nu":2.5, 
                    "ard":None, 
                    "device": self.device, 
                    "context_size": 10, 
                    "lr":0.05, 
                    "loss_tol": 0.001,
                    "use_perf_hist": False}

        self.meta_learning_epochs = 5
        self.finetuning_lr = 0.01
        self.finetuning_epochs = 10
        self.finetuning_patience = 100
        self.n_hidden = 10
        self.n_layers = 2
        self.n_batches = 1000
        self.n_init = 1
        self.bias = 0
        # cost predictor setting
        self.cost_model_type = "RF"  # from {"MLP", "RF"}
        self.time_normalization = True
    
    def reset(self, dataset_meta_features, algorithms_meta_features):
        """
        Reset the agents' memory for a new dataset

        Parameters
        ----------
        dataset_meta_features : dict of {str : str}
            The meta-features of the dataset at hand, including:
                usage = 'AutoML challenge 2014'
                name = name of the dataset
                task = 'binary.classification', 'multiclass.classification', 'multilabel.classification', 'regression'
                target_type = 'Binary', 'Categorical', 'Numerical'
                feat_type = 'Binary', 'Categorical', 'Numerical', 'Mixed'
                metric = 'bac_metric', 'auc_metric', 'f1_metric', 'pac_metric', 'a_metric', 'r2_metric'
                time_budget = total time budget for running algorithms on the dataset
                feat_num = number of features
                target_num = number of columns of target file (one, except for multi-label problems)
                label_num = number of labels (number of unique values of the targets)
                train_num = number of training examples
                valid_num = number of validation examples
                test_num = number of test examples
                has_categorical = whether there are categorical variable (yes=1, no=0)
                has_missing = whether there are missing values (yes=1, no=0)
                is_sparse = whether this is a sparse dataset (yes=1, no=0)

        algorithms_meta_features : dict of dict of {str : str}
            The meta_features of each algorithm:
                meta_feature_0 = 1 or 0
                meta_feature_1 = 0.1, 0.2, 0.3,…, 1.0

        Examples
        ----------
        >>> dataset_meta_features
        {'usage': 'AutoML challenge 2014', 'name': 'Erik', 'task': 'regression',
        'target_type': 'Binary', 'feat_type': 'Binary', 'metric': 'f1_metric',
        'time_budget': '600', 'feat_num': '9', 'target_num': '6', 'label_num': '10',
        'train_num': '17', 'valid_num': '87', 'test_num': '72', 'has_categorical': '1',
        'has_missing': '0', 'is_sparse': '1'}

        >>> algorithms_meta_features
        {'0': {'meta_feature_0': '0', 'meta_feature_1': '0.1'},
         '1': {'meta_feature_0': '1', 'meta_feature_1': '0.2'},
         '2': {'meta_feature_0': '0', 'meta_feature_1': '0.3'},
         '3': {'meta_feature_0': '1', 'meta_feature_1': '0.4'},
         ...
         '18': {'meta_feature_0': '1', 'meta_feature_1': '0.9'},
         '19': {'meta_feature_0': '0', 'meta_feature_1': '1.0'},
         }
        """

        self.dataset_meta_features = dataset_meta_features
        self.algorithms_meta_features = algorithms_meta_features
        self.validation_last_scores = [0.0 for i in range(self.nA)]
        self.algorithms_cost_count = [0.0 for i in range(self.nA)]
        self.algorithms_perf_hist = [[] for i in range(self.nA)]
        self.count = torch.zeros(self.nA).to(self.device)
        self.convergence = torch.ones(self.nA).to(self.device)
        self.trials  = torch.zeros(self.nA).to(self.device)
        self.algorithms_list = list(self.algorithms_meta_features.keys())
        self.algorithms_list.sort()
        self.iter = 0
        self.time_budget = float(dataset_meta_features["time_budget"])
        self.max_trials = 3
        self.observation_history = {"X": [], "y": [], "cost": []}
        self.delta_t_buffer = None

        if self.metatrained:
            _, _, self.dataset_features, _ = self.prep_metafeatures(
                {"0": dataset_meta_features}, self.enc, self.scaler, self.col_info
            )

            d = distance_matrix(self.dataset_features, self.metatrain_dataset_features)
            self.most_similar_metatrain_dataset = self.index_dataset[np.argmin(d)]
            self.similar_datasets_sorted = np.argsort(d)
            self.initial_alg_test_id = self.best_algorithm_per_dataset[self.most_similar_metatrain_dataset]["id_test"]
            self.initial_alg_val_id = self.best_algorithm_per_dataset[self.most_similar_metatrain_dataset]["id_val"]

        # pre-computed features for input vectors, already containing the HPs and metafeatures
        self.X_pre = []
        for alg_name in self.algorithms_list:
            alg = algorithms_meta_features[alg_name]
            hp = [float(x) for x in list(alg.values())]
            self.X_pre.append(hp + self.dataset_features.tolist()[0])
        self.X_pre = torch.FloatTensor(self.X_pre).to(self.device)

    def _one_hot_for_categoricals(self, data_cat):
        enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
        enc.fit(data_cat)
        return enc

    def _scaling_for_numeric(self, data_num):
        scaler = MinMaxScaler()
        scaler.fit(data_num)
        return scaler

    def prep_metafeatures(self, dataset_meta_features, enc=None, scaler=None, col_info=None):
        """ Function to apply some basic sanity checks and cleaning for metafeatures only

        For the meta-training phase, enc, scaler and col_info passed should be None.
        For the meta-testing phase, enc, scaler and col_info passed should be NOT be None.

        Parameters
        ----------
        dataset_meta_features : dict

        Returns
        -------
        OneHotEncoder object, object, np.array, dict
        """

        meta_f = pd.DataFrame.from_dict(dataset_meta_features).transpose().sort_index()
        if col_info is None:
            # finding columns with 0 variance
            drop_cols = meta_f.columns[meta_f.nunique(axis=0) == 1].tolist()
            # explicit hard-coded check to remove names from meta features
            # expecting `name` to be a noisy, misleading feature that can blow up dimensionality
            if "name" in meta_f.columns:
                drop_cols.append("name")
        else:
            drop_cols = col_info["drop_cols"]
        # dropping select columns
        meta_f = meta_f.drop(labels=drop_cols, axis=1)
        # enforcing type checks on columns by converting to numeric or strings
        if col_info is None:
            num_cols = []
            cat_cols = []
            for cols in meta_f.columns:
                try:
                    meta_f[cols] = meta_f[cols].astype(float)
                    num_cols.append(cols)
                except ValueError:
                    meta_f[cols] = meta_f[cols].astype(str)
                    cat_cols.append(cols)
        else:
            num_cols = col_info["num_cols"]
            cat_cols = col_info["cat_cols"]
        # subsetting meta features for numeric and categorical types
        meta_f_num = meta_f[num_cols]
        meta_f_cat = meta_f[cat_cols]
        # obtaining transformed metafeatures
        if enc is None:
            enc = self._one_hot_for_categoricals(meta_f_cat)
        data_cat = enc.transform(meta_f_cat)
        if scaler is None:
            scaler = self._scaling_for_numeric(meta_f_num)
        data_num = scaler.transform(meta_f_num)
        transformed_data = np.hstack((data_cat, data_num))

        col_info = {"cat_cols": cat_cols, "num_cols": num_cols, "drop_cols": drop_cols}

        return enc, scaler, transformed_data, col_info

    def normalize_time(self, ts, time_budget, t_0=60):
        if self.time_normalization:
            # normalize as y = log(1 + t/t0) / log(1 + T/t0)
            ts_new = np.log(1 + ts/t_0) / np.log(1 + time_budget/t_0)
        else:
            ts_new = ts / time_budget
        return ts_new

    def denormalize_time(self, ts, time_budget, t_0=60):
        if self.time_normalization:
            # extracting t from y = log(1 + t/t0) / log(1 + T/t0)
            # as t = t0 * (exp(y * log(1 + T/t0)) - 1)
            ts_new = t_0 * (np.exp(ts * np.log(1 + time_budget/t_0)) - 1)
        else:
            ts_new = ts * time_budget
        return ts_new

    def process_data(self, dataset_meta_features, algorithms_meta_features, validation_learning_curves, test_learning_curves):

        """
        Returns a training matrix and validation matrix for initializing the meta-learning model
        """
        datasets = list(dataset_meta_features.keys())
        algorithms = list(algorithms_meta_features.keys())
        n_datasets = len(datasets)
        split = int(n_datasets * 0.9)

        # preprocessing metafeatures
        self.enc, self.scaler, self.metatrain_dataset_features, self.col_info = self.prep_metafeatures(
            dataset_meta_features)
        self.index_dataset = list(dataset_meta_features.keys())

        train_data = {}
        test_data = {}

        self.best_algorithm_per_dataset = {}

        X_time = []
        y_time = []

        X_best_time = []
        y_best_time = []

        x_first_time = []

        for i, dataset in enumerate(datasets):
            X = []
            y_val = []
            y_test = []
            perf_hist = []
            time_budget = float(dataset_meta_features[dataset]["time_budget"])
            self.best_algorithm_per_dataset[dataset] = {
                "id_test": 0, "perf_test": 0, "id_val": 0, "perf_val": 0
            }
            idx = self.index_dataset.index(dataset)
            temp_dataset_feat = self.metatrain_dataset_features[idx].tolist()
            temp_x_first_time = []

            for algorithm, meta_features in algorithms_meta_features.items():

                y_val += validation_learning_curves[dataset][algorithm].scores.tolist()
                y_test += test_learning_curves[dataset][algorithm].scores.tolist()
                timestamps = validation_learning_curves[dataset][algorithm].timestamps
                values = [float(x) for x in list(meta_features.values())]
                ts_prev = [0]
                val_prev = [0]
                temp_y_val = validation_learning_curves[dataset][algorithm].scores.tolist()
                temp_y_test = test_learning_curves[dataset][algorithm].scores.tolist()
                X_best_time.append(values + temp_dataset_feat)
                y_best_time.append(self.normalize_time(timestamps[np.array(temp_y_val).argmax()], time_budget))
                temp_x_first_time.append(
                    (values, self.normalize_time(timestamps[0], time_budget), temp_y_val[0])
                )

                for j, ts in enumerate(timestamps):

                    for k in range(j, len(timestamps)):
                        ts2 = timestamps[k]

                        # X.append(values + temp_dataset_feat + [j, ts_prev/time_budget, (ts2-ts_prev)/time_budget, val_prev])
                        if self.conf["use_perf_hist"]:
                            X.append(
                                # values + temp_dataset_feat + [j,  t_delta]
                                values + temp_dataset_feat + [j, self.normalize_time(ts2 - ts_prev[-1], time_budget)]
                            )
                        else:
                            # X.append(values + temp_dataset_feat + [j, ts_prev[-1], ts2/time_budget-ts_prev[-1], val_prev[-1]])
                            X.append(
                                values + temp_dataset_feat + [
                                    # j, ts_prev[-1],  ts2 / time_budget - ts_prev[-1]
                                    j,
                                    self.normalize_time(ts_prev[-1], time_budget),
                                    self.normalize_time(ts2 - ts_prev[-1], time_budget)  #  t_delta
                                ]
                            )

                        y_val.append(temp_y_val[k])
                        y_test.append(temp_y_test[k])
                        perf_hist.append([ts_prev.copy(), val_prev.copy()])

                    # X.append(values + temp_dataset_feat + [j, ts_prev/time_budget])
                    X_time.append(values + temp_dataset_feat + [j, self.normalize_time(ts_prev[-1], time_budget)])
                    y_time.append(self.normalize_time(ts - ts_prev[-1], time_budget))  # * time_budget)

                    ts_prev.append(ts)  # / time_budget)
                    val_prev.append(y_val[j])

                    # val_prev = y_val[j]

                if self.best_algorithm_per_dataset[dataset]["perf_test"] < y_test[-1]:
                    self.best_algorithm_per_dataset[dataset]["id_test"] = algorithm
                    self.best_algorithm_per_dataset[dataset]["perf_test"] = y_test[-1]

                # if self.best_algorithm_per_dataset[dataset]["perf_val"] < y_val[-1]:
                #    self.best_algorithm_per_dataset[dataset]["id_val"] = algorithm
                #    self.best_algorithm_per_dataset[dataset]["perf_val"] = y_val[-1]

                if self.best_algorithm_per_dataset[dataset]["perf_val"] < temp_y_val[0] / (
                        timestamps[0] + 1):
                    self.best_algorithm_per_dataset[dataset]["id_val"] = algorithm
                    self.best_algorithm_per_dataset[dataset]["perf_val"] = temp_y_val[0] / (
                                timestamps[0] + 1)

            future_tensor = []
            if self.conf["use_perf_hist"]:
                for conf in perf_hist:
                    future_tensor.append([])
                    for j, (x_j, y_j) in enumerate(zip(conf[0], conf[1])):
                        future_tensor[-1].append([x_j, y_j])

                    # assuming max_length = 9
                    for z in range(j, 10):
                        future_tensor[-1].append([0, 0])

            if i < split:
                train_data[dataset] = {
                    "X": X, "y_val": y_val, "y_test": y_test,
                    "perf_hist": future_tensor, "x_first": temp_x_first_time,
                    "meta_feat": temp_dataset_feat
                }
            else:
                test_data[dataset] = {
                    "X": X, "y_val": y_val, "y_test": y_test,
                    "perf_hist": future_tensor, "x_first": temp_x_first_time,
                    "meta_feat": temp_dataset_feat
                }

        return train_data, test_data, X_time, y_time, X_best_time, y_best_time, len(X[0])


    def meta_train(self, dataset_meta_features, algorithms_meta_features, validation_learning_curves, test_learning_curves):
        """
        Start meta-training the agent with the validation and test learning curves

        Parameters
        ----------
        datasets_meta_features : dict of dict of {str: str}
            Meta-features of meta-training datasets

        algorithms_meta_features : dict of dict of {str: str}
            The meta_features of all algorithms

        validation_learning_curves : dict of dict of {int : Learning_Curve}
            VALIDATION learning curves of meta-training datasets

        test_learning_curves : dict of dict of {int : Learning_Curve}
            TEST learning curves of meta-training datasets

        Examples:
        To access the meta-features of a specific dataset:
        >>> datasets_meta_features['Erik']
        {'name':'Erik', 'time_budget':'1200', ...}

        To access the validation learning curve of Algorithm 0 on the dataset 'Erik' :

        >>> validation_learning_curves['Erik']['0']
        <learning_curve.Learning_Curve object at 0x9kwq10eb49a0>

        >>> validation_learning_curves['Erik']['0'].timestamps
        [196, 319, 334, 374, 409]

        >>> validation_learning_curves['Erik']['0'].scores
        [0.6465293662860659, 0.6465293748988077, 0.6465293748988145, 0.6465293748988159, 0.6465293748988159]
        """

        print("meta_train")
        self.validation_learning_curves = validation_learning_curves
        self.test_learning_curves = test_learning_curves

        train_data, test_data, X_cost, y_cost, X_best, y_best, input_size = self.process_data(
            dataset_meta_features, algorithms_meta_features, validation_learning_curves, test_learning_curves
        )
        feature_extractor = MLP(
            input_size, n_hidden=self.n_hidden, n_layers=self.n_layers, use_cnn = self.conf["use_perf_hist"]
        ).to(self.device)
        self.fsbo_model = FSBO(
            train_data, test_data, feature_extractor=feature_extractor, conf=self.conf
        ).to(self.device)
        self.fsbo_model.train(epochs=self.meta_learning_epochs, n_batches=self.n_batches)
        # self.fsbo_model.train(epochs=1, n_batches=self.n_batches)
        
        if self.cost_model_type == "MLP":
            model_type = "base"
            if model_type == "base":
                arch = {
                    "n_input": len(X_cost[0]),
                    "n_hidden": 64,  # self.n_hidden,
                    "n_layers": 2,  # self.n_layers,
                    "n_output": 1,
                    "dropout_rate": 0.05,
                    "use_cnn": False,
                    "output_relu": True
                }
                train_hps = {"lr": 0.001, "epochs": 400, "batch_size": 64, "alpha": -0.1}
            elif model_type == "hadi":
                # Hadi's model
                arch = {
                    "n_input": len(X_cost[0]),
                    "n_hidden": 441,
                    "n_layers": 8,
                    "n_output": 1,
                    "dropout_rate": 0.0010000000000000002,
                    "use_cnn": False
                }
                train_hps = {"lr": 0.001, "epochs": 400, "batch_size": 64, "alpha": -0.018479149770895}
            else:
                raise ValueError("Model type not recognized!")
            self.cost_model = MLP(**arch).to(self.device)

            y_cost = torch.FloatTensor(y_cost).to(self.device)
            self.y_max_cost = torch.max(y_cost)
            X_cost = torch.FloatTensor(X_cost).to(self.device)
            _ = self.train_cost_model(X_cost, y_cost, **train_hps)
            torch.save(self.cost_model, "cost_model.pt")
        else:
            self.cost_model = RandomForestRegressor(max_depth=50)
            # y_cost = np.log(np.array(y_cost)+1)
            self.cost_model.fit(np.array(X_cost), y_cost)

        self.best_cost_model = RandomForestRegressor(n_estimators=10)
        self.best_cost_model.fit(np.array(X_best), np.array(y_best))

        self.metatrained = 1

    def custom_loss_fn(self, y_pred, y_true, alpha=-0.1):
        # Hadi: out = -torch.minimum(alpha*(target-input),target-input)
        # penalizing undershooting
        l1 = y_pred - y_true
        # penalizing overshooting
        l2 = alpha * (l1)
        loss = -torch.min(torch.hstack((l1, l2)), axis=1).values.mean()
        return loss

    def train_cost_model(self,  X, y, lr=0.001, epochs=100, batch_size=512, alpha=-0.1):
        writer = None
        writer =  SummaryWriter("./runs/custom_time/")
        optimizer = torch.optim.Adam(self.cost_model.parameters(), lr=lr)

        loss_fn = self.custom_loss_fn  # torch.nn.MSELoss()
        losses = [np.inf]

        data = torch.hstack((X, y.reshape(y.shape[0], 1)))

        tr_size = 0.7 if writer is not None else 1.0
        idx = int(tr_size * len(data))
        D_tr = data[:idx]
        D_val = data[-(len(data) - idx):]

        dloader = DataLoader(D_tr, batch_size=batch_size, shuffle=False)
        per_epoch_loss = []
        for epoch in range(epochs):
            losses = []
            tracker = []
            val_tracker = []
            for (idx, batch) in enumerate(dloader):
                batch_x = batch[:, :-1]
                batch_y = batch[:, -1]
                optimizer.zero_grad()
                pred = self.cost_model(batch_x)
                loss = loss_fn(pred, batch_y.reshape(-1, 1), alpha)
                loss.backward()
                optimizer.step()
                losses.append(loss.detach().cpu().numpy().item())
                # analyzing
                tracker.extend((pred.reshape(-1) - batch_y.reshape(-1)).detach().cpu().numpy().tolist())
            # to ignore potential inf/nan as losses in mean computation
            mean_loss = np.ma.masked_invalid(losses).mean()
            per_epoch_loss.append(mean_loss)
            print("Epoch {:>4}/{:>4}: loss={:.5f}".format(epoch + 1, epochs, mean_loss), end='\r')
            # prediction on the validation set for an epoch
            if writer is not None:
                with torch.no_grad():
                    val_pred = self.cost_model(D_val[:, :-1])
                    val_tracker = (
                            val_pred.reshape(-1) - D_val[:, -1].reshape(-1)
                    ).detach().cpu().numpy().tolist()
                self._log_writer(writer, epoch+1, mean_loss, tracker, val_tracker)

            if per_epoch_loss[-1] < 0.001:
                break
        if writer is not None:
            writer.close()

        return per_epoch_loss

    def _log_writer(self, writer, i, loss, tracker, val_tracker):
        writer.add_scalar("loss", loss, i)
        # for training set
        def write_for_split(tracker, data="train"):
            tracker = np.array(tracker)
            overshoots = tracker[tracker >= 0]
            undershoots = tracker[tracker < 0]
            err = len(undershoots) / len(tracker)  # % of undershoots
            mean_under = np.mean(undershoots)
            mean_over = np.mean(overshoots)
            writer.add_scalar("{}/% of undershoots".format(data), err, i)
            writer.add_scalar("{}/mean undershoot".format(data), mean_under, i)
            writer.add_scalar("{}/mean overshoot".format(data), mean_over, i)
        write_for_split(tracker, "train")
        write_for_split(val_tracker, "valid")

    def _map_algo_index(self, index):
        return self.algorithms_list[index]

    def suggest(self, observation):
        """
        Return a new suggestion based on the observation

        Parameters
        ----------
        observation : tuple of (int, float, float)
            The last observation returned by the environment containing:
                (1) A: the explored algorithm,
                (2) C_A: time has been spent for A
                (3) R_validation_C_A: the validation score of A given C_A

        Returns
        ----------
        action : tuple of (int, int, float)
            The suggested action consisting of 3 things:
                (1) A_star: algorithm for revealing the next point on its test learning curve
                            (which will be used to compute the agent's learning curve)
                (2) A:  next algorithm for exploring and revealing the next point
                       on its validation learning curve
                (3) delta_t: time budget will be allocated for exploring the chosen algorithm in (2)

        Examples
        ----------
        >>> action = agent.suggest((9, 151.73, 0.5))
        >>> action
        (9, 9, 80)
        """
        self.iter += 1

        # first observation/iteration
        if observation is None:
            # initializing episode trackers
            self.time_used = 0
            self.observed_configs = []
            self.observed_response = []
            self.observed_cost = []
            self.predicted_cost = []
            self.last_queried_cost = 0.0
            self.remaining_budget_counter = self.time_budget
            self.per_algo_cost_tracker = {algo: [] for algo in self.algorithms_list}
            self.is_success_last_cost = False
            self.algorithms_perf_hist = torch.zeros(self.nA, 2, 11)
            self.observed_lc = []
            self.count = torch.zeros(self.nA).to(self.device)
            a = None
            b = self.algorithms_list.index(self.initial_alg_val_id)
            # loading hyperparameter/algorithm
            algorithm_meta_feat = [
                float(x) for x in list(
                    self.algorithms_meta_features[self.initial_alg_val_id].values()
                )
            ]
            # creating input vector for surrogates
            X = algorithm_meta_feat + self.dataset_features.tolist()[0] + [0, 0]
            # add observation to history
            self.observation_history["X"].append(X)
            self.observation_history["y"].append([0.0])
            self.observation_history["cost"].append([0.0])
            X = torch.FloatTensor(X).to(self.device)
            # estimating cost to use
            if self.cost_model_type == "MLP":
                c = self.denormalize_time(self.cost_model(X).detach().numpy(), self.time_budget)[0]
            else:
                c = self.denormalize_time(self.cost_model.predict(np.array(X).reshape(1, -1)).item(), self.time_budget)
            delta_t = min(c, max(self.time_budget * 0.99, 1))
            self.last_predicted_cost = delta_t
            self.remaining_budget_counter -= delta_t
            a_star = a
            a = b
            suggestion = (a_star, a, delta_t)

            return suggestion

        # all observations from step 2 in the episode
        algorithm_index, ts, y_val = observation
        ts_norm = self.normalize_time(ts, self.time_budget)

        #############
        # Section 1 #
        #############
        # Book-keeping of reported observation and updating variables

        # self.convergence keeps track of the algorithms that converged (=0) to omit them in EI
        if y_val == self.validation_last_scores[
            algorithm_index] and y_val != 0:  # and ts==self.algorithms_cost_count[algorithm_index]:
            self.convergence[algorithm_index] = 0
            prev_score = y_val
        else:
            prev_score = self.validation_last_scores[algorithm_index]
            self.validation_last_scores[algorithm_index] = y_val

        if ts != self.algorithms_cost_count[algorithm_index]:
            self.is_success_last_cost = True
            self.per_algo_cost_tracker[self._map_algo_index(algorithm_index)].append(ts)
            self.algorithms_cost_count[algorithm_index] = ts
            self.count[algorithm_index] += 1
            self.trials[algorithm_index] = 0
            self.delta_t_buffer = 0
        else:
            self.is_success_last_cost = False
            self.trials[algorithm_index] += 1
            if self.trials[algorithm_index] == self.max_trials:
                self.convergence[algorithm_index] = 0
            # initialized in reset()
            if self.delta_t_buffer is None or self.last_algo_index != algorithm_index:
                self.delta_t_buffer = 0
            else:
                # tracks the sum of sequence of delta_t that failed
                self.delta_t_buffer += self.last_predicted_cost

        if ts != self.algorithms_cost_count[algorithm_index] or len(self.observed_configs) == 0:
            algorithm = self.algorithms_list[algorithm_index]
            j = self.count[algorithm_index]
            prev_ts = self.algorithms_cost_count[algorithm_index]
            algorithm_meta_feat = [
                float(x) for x in list(self.algorithms_meta_features[algorithm].values())
            ]
            self.observed_cost.append(ts - self.algorithms_cost_count[algorithm_index])
            self.predicted_cost.append(self.last_predicted_cost)

            if self.conf["use_perf_hist"]:
                self.observed_configs.append(
                    algorithm_meta_feat +
                    self.dataset_features.tolist()[0] +
                    [
                        j,
                        self.normalize_time(ts - prev_ts, self.time_budget)
                    ]
                )
            else:
                self.observed_configs.append(
                    algorithm_meta_feat +
                    self.dataset_features.tolist()[0] + [
                        j,
                        self.normalize_time(prev_ts, self.time_budget),
                        self.normalize_time(ts - prev_ts, self.time_budget)
                    ]
                )
            self.algorithms_perf_hist[algorithm_index, :, int(j.item())] = torch.FloatTensor(
                [self.normalize_time(prev_ts, self.time_budget), prev_score]
            )
            self.observed_lc.append(self.algorithms_perf_hist[algorithm_index].tolist())
            self.algorithms_cost_count[algorithm_index] = ts
            self.observed_response.append(y_val)

        #############
        # Section 2 #
        #############
        # Finetuning and preparing next query

        current_time = np.array(self.algorithms_cost_count)
        best_time_pred = self.denormalize_time(self.best_cost_model.predict(np.array(self.X_pre)), self.time_budget)
        time_to_best = best_time_pred - current_time
        # TODO: handle negative
        time_to_best[time_to_best <= 0] = self.remaining_budget_counter
        time_to_best = np.min(
            (time_to_best, np.ones(len(time_to_best)) * self.remaining_budget_counter), axis=0
        )
        # normalize times
        time_to_best = self.normalize_time(time_to_best, self.time_budget)
        current_time = self.normalize_time(current_time, self.time_budget)
        # convert to tensors
        time_to_best = torch.Tensor(time_to_best).reshape(-1, 1)
        current_time = torch.Tensor(current_time).reshape(-1, 1)

        if self.conf["use_perf_hist"]:
            w_spt = torch.FloatTensor(self.observed_lc).to(self.device)
            x_qry = torch.concat((
                self.X_pre,
                self.count.reshape(-1, 1),
                time_to_best
            ), axis=1)
            w_qry = self.algorithms_perf_hist
        else:
            w_spt = None
            w_qry = None
            x_qry = torch.concat((
                self.X_pre,
                self.count.reshape(-1, 1),
                current_time,
                time_to_best
            ), axis=1)
        # finetune surrogate
        x_spt = torch.FloatTensor(self.observed_configs).to(self.device)
        y_spt = torch.FloatTensor(self.observed_response).to(self.device)
        _ = self.fsbo_model.finetuning(
            x_spt, y_spt, w_spt,
            epochs=self.finetuning_epochs,
            finetuning_lr=self.finetuning_lr,
            patience=self.finetuning_patience
        )

        # accounting for failures by changing current time
        x_qry[algorithm_index, -2] = x_qry[algorithm_index, -2] + self.delta_t_buffer

        if self.cost_model_type == "MLP":
            # dropping the last column with time_to_best
            y_pred_cost = self.cost_model(x_qry[:, :-1])
        else:
            y_pred_cost = torch.FloatTensor(self.cost_model.predict(np.array(x_qry[:, :-1])))

        # appending predicted delta_t to current_time for EI predictions
        x_qry[:, -2] += y_pred_cost.reshape(-1)

        #############
        # Section 3 #
        #############
        # Compute acqusition function

        best_y = max(self.observed_response)
        mean, std = self.fsbo_model.predict(x_spt, y_spt, x_qry, w_spt, w_qry)
        ei = torch.FloatTensor(self.EI(mean, std, best_y)).to(self.device)

        # masks to avoid picking converged algorithms
        ei = torch.multiply(ei, self.convergence)

        if self.acquistion_type == "cost_balanced_ei":
            ei = torch.divide(ei, y_pred_cost)

        next_algorithm = torch.argmax(ei).item()
        if y_pred_cost[next_algorithm].item() < 0:
            print("Predicted c is -ve:", y_pred_cost[next_algorithm].item())
        if y_pred_cost[next_algorithm].item() == 0:
            print("Predicted c is 0:", y_pred_cost[next_algorithm].item())
            # TODO: better heuristic or fix
            y_pred_cost[next_algorithm] = 10
        c = self.denormalize_time(y_pred_cost[next_algorithm].item(), self.time_budget)
        b = next_algorithm
        a = np.argmax(self.validation_last_scores)
        a_star = a
        a = b
        c = min(c, self.remaining_budget_counter)
        self.remaining_budget_counter -= c
        self.last_predicted_cost = c
        self.predicted_cost.append(c)
        suggestion = (a_star, a, c)
        self.last_algo_index = a

        return suggestion

    def EI(self, mean, sigma, best_f, epsilon=0):
        with np.errstate(divide='warn'):
            imp = mean - best_f - epsilon
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return ei
