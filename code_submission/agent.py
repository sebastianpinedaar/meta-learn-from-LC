from functools import lru_cache
import json
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
from scipy.spatial import distance_matrix
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
torch.manual_seed(0)

class Agent():
    def __init__(self, number_of_algorithms, config={}):
        """
        Initialize the agent

        Parameters
        ----------
        number_of_algorithms : int
            The number of algorithms

        """
        torch.manual_seed(0)
        self.nA = number_of_algorithms
        self.numerical_dataset_metafeatures = ["time_budget", "feat_num", "target_num", "label_num", "train_num", "valid_num","test_num", "has_categorical", "has_missing", "is_sparse"]
        self.categorical_dataset_metafeatures = ["task", "target_type", "feat_type", "metric"]
        #self.categorical_dataset_metafeatures = []

        self.metatrained = 0
        self.observed_configs = []
        self.observed_response = []
        self.observed_cost = []
        #self.device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"
        path    = os.path.dirname(os.path.realpath(__file__))



        if ~bool(config):
            dir_list = os.listdir(path)
            file =  [f for f in dir_list if f[-4:]=="json"][0]
            with open(path+"/"+file) as f:
                config = json.load(f)


        self.conf = {"kernel":"52", 
                    "nu":2.5, 
                    "ard":None, 
                    "device": self.device, 
                    "context_size": 10, 
                    "lr":0.05, 
                    "loss_tol": 0.001,
                    "use_perf_hist": config.get("use_perf_hist",False)}

        self.meta_learning_epochs = 5
        self.finetuning_lr = 0.01
        self.finetuning_epochs = 10
        self.finetuning_patience = 100
        self.n_hidden = 10
        self.n_layers = 3
        self.n_batches = 1000
        self.n_init = 1
        self.bias = 0
        self.seq_len = 20
        self.use_ranker = config.get("use_ranker", False)
        self.max_trials = config.get("max_trials", 3)
        self.use_best_alg = config.get("use_best_alg", False)
        self.norm_type = config.get("norm_type", "log")
        self.perf_measure = config.get("perf_measure","ratio") #possibilities: ratio, alc, accuracy
        self.surrogate_target = config.get("surrogate_target", "accuracy")
        self.time_input = config.get("time_input", "time_to_best") # "time_to_best" #possibilities: "predicted_time", "remaining_budget
        self.percentile = config.get("percentile", 50)
        self.initial_factor = config.get("initial_factor", 1)
        self.cost_model_type = config.get("cost_model_type","Percentile") # Constant, Percentile, MLP, RF
        self.acquistion_type = config.get("acquistion_type", "ei_cost_balanced")
        self.include_subset_metafeat = config.get("include_subset_metafeat", False)
        self.include_dataset_metafeat = config.get("include_dataset_metafeat",True)
        self.use_categorical_hp = config.get("use_categorical_hp", False)

        if self.include_subset_metafeat:
            self.col_info = {"drop_cols":["name", "usage" ], "num_cols":["feat_num","train_num", "valid_num","test_num"], "cat_cols":[]}
        else:
            self.col_info = None

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
        


        if self.metatrained:

            _, _, self.dataset_features, _ = self.prep_metafeatures(
                {"0": dataset_meta_features}, self.enc, self.scaler, self.col_info
            )

            d = distance_matrix(self.dataset_features, self.metatrain_dataset_features)
            self.most_similar_metatrain_dataset = self.index_dataset[np.argmin(d)]
            self.similar_datasets_sorted = np.argsort(d)
            self.initial_alg_test_id= self.best_algorithm_per_dataset[self.most_similar_metatrain_dataset]["id_test"]
            self.initial_alg_val_id= self.best_algorithm_per_dataset[self.most_similar_metatrain_dataset]["id_val"]

        self.X_pre = []

        for alg_name in self.algorithms_list:
            alg = algorithms_meta_features[alg_name]
            hp = self.prep_algo_feat(list(alg.values()))

            if self.include_dataset_metafeat:

                self.X_pre.append(hp+self.dataset_features.tolist()[0])
            else:
                self.X_pre.append(hp)
        
        self.X_pre = torch.FloatTensor(self.X_pre).to(self.device)

    def _one_hot_for_categoricals(self, data_cat):
        enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
        enc.fit(data_cat)
        return enc

    def _scaling_for_numeric(self, data_num):
        scaler = MinMaxScaler()
        scaler.fit(data_num)
        return scaler

    def prep_algo_feat(self, values, fit=False):

        if self.use_categorical_hp:

            values = self.hp_enc.transform([values])

            return values.tolist()[0]

        else:

            return  [float(x) for x in values]

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

    def substract_original_scale_numpy(self,  x1, x2, b):


        out = np.log(np.exp(x1*b) - np.exp(x2*b)+1)/b
        return out
     
    def substract_original_scale_torch(self,  x1, x2, b):
        out = torch.log(torch.exp(x1*b) - torch.exp(x2*b))/b
        return out

    def denorm(self, t, time_budget, t0=1):

        if self.norm_type=="log":
            out = t0*(np.exp(t*np.log(time_budget/t0+1))-1)
        elif self.norm_type=="linear":
            out = t*time_budget
        else:
            out = t
        return out

    def norm(self, t, time_budget, t0=1):

        t = np.clip(t, a_min = 0, a_max=None)

        if self.norm_type =="log":
            out = np.log(t/t0+1)/np.log(time_budget/t0+1)
        elif self.norm_type =="linear":
            out = t/time_budget
        else:
            out = t
        
        return out


    def process_data(self, dataset_meta_features, algorithms_meta_features, validation_learning_curves, test_learning_curves):

        """
        Returns a training matrix and validation matrix for initializing the meta-learning model
        """
        datasets = list(dataset_meta_features.keys())
        algorithms = list(algorithms_meta_features.keys())

        if self.use_categorical_hp:
            hps=pd.DataFrame(algorithms_meta_features).T
            self.hp_enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
            self.hp_enc.fit(np.array(hps))
    
        n_datasets = len(datasets)
        split = int(n_datasets*0.9)
        self.ranges_budget = np.linspace(0,1,10)

        # preprocessing metafeatures
        self.enc, self.scaler, self.metatrain_dataset_features, self.col_info = self.prep_metafeatures(dataset_meta_features, col_info= self.col_info)
        self.index_dataset = list(dataset_meta_features.keys())

        train_data = {}
        test_data = {}

        self.best_algorithm_per_dataset = {}

        X_time = []
        y_time = []

        X_best_time = []
        y_best_time = []

        y_first_budget = []

        matrix_first_recommendation = np.zeros((len(datasets), len(algorithms)))
        self.budget_history = {}
        max_len = 0

        for i, dataset in enumerate(datasets):
            X =[]
            y_val = []
            y_test = []
            perf_hist = []
            
            time_budget = float(dataset_meta_features[dataset]["time_budget"])
            
            self.best_algorithm_per_dataset[dataset] = {"id_test":0, "perf_test":0, "id_val":0, "perf_val":0}
            idx = self.index_dataset.index(dataset)

            if self.include_dataset_metafeat:
                temp_dataset_feat = self.metatrain_dataset_features[idx].tolist()
            else:
                temp_dataset_feat = []
            temp_x_first_time = []

           

            for algorithm, meta_features in algorithms_meta_features.items():

                if algorithm not in self.budget_history.keys():
                    self.budget_history[algorithm] = [[0] for i in range(10)]
                

                timestamps = validation_learning_curves[dataset][algorithm].timestamps

                max_len = max(max_len, len(timestamps))
                
                values = self.prep_algo_feat(list(meta_features.values()))
                ts_prev = [0]
                val_prev = [0]
                temp_y_val = validation_learning_curves[dataset][algorithm].scores.tolist()
                temp_y_test = test_learning_curves[dataset][algorithm].scores.tolist()

                X_best_time.append(values + temp_dataset_feat)
                y_best_time.append(timestamps[np.array(temp_y_val).argmax()])
                temp_x_first_time.append(( values, float(timestamps[0]), temp_y_val[0]))
                y_first_budget.append((float(timestamps[0]), temp_y_val[0]))

                algorithm_idx = self.metatrain_algorithms_list.index(algorithm)
                dataset_idx = self.index_dataset.index(dataset)

                for j, ts in enumerate(timestamps):

                    
                    for k in range(j, len(timestamps)):
                        ts2 = timestamps[k]
                        step = self.norm(ts2-ts_prev[-1], time_budget)
                        if self.conf["use_perf_hist"]:
                            
                            X.append(values +  temp_dataset_feat + [j, step])

                        else:
                            X.append(values + temp_dataset_feat + [j, self.norm(ts_prev[-1], time_budget), step, val_prev[-1]])

                        if self.surrogate_target=="ratio":
                            y_val.append(temp_y_val[k]/(step+1e-5))
                            y_test.append(temp_y_test[k]/(step+1e-5))
                        
                        else:
                            y_val.append(temp_y_val[k])
                            y_test.append(temp_y_test[k])
                        perf_hist_ts = [self.norm(t, time_budget) for t in ts_prev ]
                        perf_hist.append([perf_hist_ts, val_prev.copy()])


                    X_time.append(values + temp_dataset_feat + [j, self.norm(ts_prev[-1], time_budget)] )
                    y_time.append(self.norm(ts - ts_prev[-1], time_budget))

                    ts_prev.append(ts)
                    val_prev.append(y_val[j])

                    current_budget = self.norm(ts_prev[-2], time_budget)
                    bucket = self.find_bucket(current_budget)
                    self.budget_history[algorithm][bucket].append(self.norm(ts_prev[-1]-ts_prev[-2], time_budget))

                    #val_prev = y_val[j]

                if self.best_algorithm_per_dataset[dataset]["perf_test"] < y_test[-1]:
                    self.best_algorithm_per_dataset[dataset]["id_test"] = algorithm
                    self.best_algorithm_per_dataset[dataset]["perf_test"] = y_test[-1]

                if self.perf_measure=="ratio":
                    val_perf = temp_y_val[0]/(1e-5+self.norm(timestamps[0], time_budget))
                elif self.perf_measure=="alc":
                    val_perf = temp_y_val[0]*(1-self.norm(timestamps[0], time_budget))
                else:
                    val_perf = temp_y_val[0]

                matrix_first_recommendation[dataset_idx, algorithm_idx] = val_perf

                if self.best_algorithm_per_dataset[dataset]["perf_val"] < val_perf:
                    self.best_algorithm_per_dataset[dataset]["id_val"] = algorithm
                    self.best_algorithm_per_dataset[dataset]["perf_val"] = val_perf


            future_tensor = []

            if self.conf["use_perf_hist"]:
                for conf in perf_hist:
                    future_tensor.append([])
                    for j, (x_j, y_j) in enumerate(zip(conf[0], conf[1])):
                        future_tensor[-1].append([x_j, y_j])

                    #assuming max_length = 9
                    for z in range(j, self.seq_len-1):
                        future_tensor[-1].append([0,0])


            if i < split:
                train_data[dataset] = {"X": X, "y_val": y_val, "y_test":y_test, "perf_hist":future_tensor, "x_first":temp_x_first_time, "meta_feat":temp_dataset_feat}
            else:
                test_data[dataset] = {"X": X, "y_val": y_val, "y_test":y_test, "perf_hist":future_tensor, "x_first": temp_x_first_time, "meta_feat":temp_dataset_feat}
       
        print("Max len:", max_len)

        return train_data, test_data, X_time, y_time, X_best_time, y_best_time, y_first_budget, matrix_first_recommendation, len(X[0])


    def find_bucket(self, x):

        current_budget_bucket = np.where(self.ranges_budget>=x)[0]
        if len(current_budget_bucket)==0:
            return -1
        else:
            return current_budget_bucket[0]

    def find_budget(self, algorithm, current_time, pct=0.75):

        bucket = self.find_bucket(current_time)
        return np.percentile(self.budget_history[algorithm][bucket], pct )


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

        self.metatrain_algorithms_list = list(algorithms_meta_features.keys())
        self.metatrain_algorithms_list.sort()

        train_data, test_data, X_cost, y_cost, X_best, y_best, y_first, matrix_first_recommendation, input_size = self.process_data(dataset_meta_features, algorithms_meta_features, validation_learning_curves, test_learning_curves)
        feature_extractor = MLP(input_size, n_hidden=self.n_hidden, n_layers=self.n_layers,
                                     use_cnn = self.conf["use_perf_hist"], seq_len=self.seq_len).to(self.device)        
        self.fsbo_model = FSBO(train_data, test_data, feature_extractor=feature_extractor , conf=self.conf).to(self.device)
        self.fsbo_model.train(epochs=self.meta_learning_epochs, n_batches=self.n_batches)
        
        if self.cost_model_type == "MLP":
            self.cost_model = MLP(n_input = len(X_cost[0]),
                                    n_hidden = self.n_hidden,
                                    n_layers = self.n_layers,
                                    n_output = 1,
                                    use_cnn=False).to(self.device)
            y_cost = torch.FloatTensor(y_cost).to(self.device)
            #y_cost = torch.log(y_cost+1)
            #self.y_max_cost = torch.max(y_cost)
            X_cost = torch.FloatTensor(X_cost).to(self.device)
            losses = self.train_cost_model(X_cost, y_cost, lr = 0.001, epochs = 1000)

            torch.save(self.cost_model, "cost_model.pt")
        else:
            #self.cost_model = RandomForestRegressor(n_estimators=10)
            self.cost_model = RandomForestRegressor(max_depth=50)
            #y_cost = np.log(np.array(y_cost)+1)
            self.cost_model.fit(np.array(X_cost), y_cost)

        self.best_cost_model = RandomForestRegressor(n_estimators=10)
        self.best_cost_model.fit(np.array(X_best), np.array(y_best))

        if self.use_ranker:
            self.ranker = MLP(len(X_best[0]), n_hidden=20, n_layers=1, n_output = 1, use_cnn =False).to(self.device)  
            y = torch.FloatTensor(y_first)
            y = torch.FloatTensor(torch.multiply(0.9-y[:,0].reshape(-1), y[:,1].reshape(-1)))
            losses = self.train_ranker_model( torch.FloatTensor(X_best).to(self.device),y.to(self.device), self.ranker, epochs=50, batch_size=100)

            print("Loss ranker:", losses[-3:])

        ranks = pd.DataFrame(matrix_first_recommendation.T).rank().mean(axis=1).tolist()
        self.best_algorithm_idx = np.argmax(ranks)
        self.best_algorithm_name =  self.metatrain_algorithms_list[self.best_algorithm_idx]
        self.best_algorithm_metafeat = algorithms_meta_features[self.best_algorithm_name]
        self.best_algorithm_metafeat_code = "".join([str(x) for x in self.best_algorithm_metafeat.values()])

        self.metatrained = 1

    def train_cost_model (self,  X, y_cost, lr = 0.001, epochs = 100):

        optimizer = torch.optim.Adam(self.cost_model.parameters(), lr= lr)
        loss_fn = torch.nn.MSELoss()
        losses = [np.inf]
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            pred = self.cost_model(X)
            loss = loss_fn(pred, y_cost.reshape(-1,1))
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().cpu().numpy().item())

            if(losses[-1]<0.001):
                break

        return losses

    def train_ranker_model(self, X, y, ranker, epochs=20, lr=0.005, batch_size=100):


        optimizer = torch.optim.Adam(ranker.parameters(), lr= lr)
        loss_fn = torch.nn.MarginRankingLoss(margin=0.0)

        losses = [np.inf]
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            with torch.no_grad():
                batch_id = torch.randint(0, X.shape[0], (batch_size,))
                X_batch = X[batch_id]
                y_batch = y[batch_id]
                
                idx = torch.arange(batch_size)
                id_cross = torch.cartesian_prod(idx, idx)
                target = y_batch[id_cross[:,0]] >= y_batch[id_cross[:,1]]
                target = target.long()
                target[target==0] = -1
                target[id_cross[:,0]==id_cross[:,1]]=0
            
            pred = ranker(X_batch)
            loss = loss_fn(pred[id_cross[:,0]], pred[id_cross[:,1]], target)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().cpu().numpy().item())

        return losses


    


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
        self.iter +=1
        

        if (observation is None):
           
            self.time_used = 0
            self.observed_configs = []
            self.observed_response = []
            self.observed_cost = []
            self.predicted_cost = []
            self.remaining_budget_counter = self.time_budget
            #self.time_budget = np.log(self.time_budget+1)
            self.algorithms_perf_hist = torch.zeros(self.nA,2,self.seq_len)
            self.observed_lc = []
            self.count = torch.zeros(self.nA).to(self.device)
            a = None
            
            

            if self.use_ranker:
                b = self.ranker(self.X_pre).argmax().item()
                alg_id = self.algorithms_list[b]

            elif self.use_best_alg:

                b = self.best_algorithm_idx
                alg_id = self.best_algorithm_name

            else: 
                b =  self.algorithms_list.index(self.initial_alg_val_id)
                alg_id = self.initial_alg_val_id
                
            if self.include_dataset_metafeat:
                dataset_meta_feat = self.dataset_features.tolist()[0]
            else:
                dataset_meta_feat = []
            
            algorithm_meta_feat = self.prep_algo_feat(list(self.algorithms_meta_features[alg_id].values()))
            x = torch.FloatTensor(algorithm_meta_feat+dataset_meta_feat +[0,0]).to(self.device)

            if self.cost_model_type == "MLP":
                c = self.cost_model(x).item()
    
            elif self.cost_model_type == "Percentile":

               c= self.find_budget(alg_id, 0, self.percentile).item()               
            else:
                c = self.cost_model.predict(np.array(x).reshape(1,-1)).item()
            c = self.denorm(c, self.time_budget)
            self.last_predicted_cost=c
            c = np.exp(self.initial_factor)*c
            c = min(c, self.time_budget*0.99)
            self.remaining_budget_counter-=c

            return a,b, c


        else:


            algorithm_index, ts, y_val = observation
            #ts = (np.log(ts+1)/self.time_budget).item()

            if y_val == self.validation_last_scores[algorithm_index]  and y_val !=0 :#and ts==self.algorithms_cost_count[algorithm_index]: 
                self.convergence[algorithm_index]=0
                prev_score = y_val
            else:
                prev_score = self.validation_last_scores[algorithm_index] 
                self.validation_last_scores[algorithm_index] = y_val

            if ts!=self.algorithms_cost_count[algorithm_index]:
                self.count[algorithm_index]+=1
                self.trials[algorithm_index]=0           
            else:
                self.trials[algorithm_index]+=1  

                if self.trials[algorithm_index] == self.max_trials:
                    self.convergence[algorithm_index]=0


            if  ts!=self.algorithms_cost_count[algorithm_index] or len(self.observed_configs)==0:

                algorithm = self.algorithms_list[algorithm_index]
                j = self.count[algorithm_index]
                prev_ts = self.algorithms_cost_count[algorithm_index]
                algorithm_meta_feat = self.prep_algo_feat(list(self.algorithms_meta_features[algorithm].values()))
                if self.include_dataset_metafeat:
                    dataset_meta_feat = self.dataset_features.tolist()[0]
                else:
                    dataset_meta_feat = []
                self.observed_cost.append(ts-self.algorithms_cost_count[algorithm_index])
                self.predicted_cost.append(self.last_predicted_cost)

                if self.conf["use_perf_hist"]:
                    self.observed_configs.append(algorithm_meta_feat+dataset_meta_feat+[j, self.norm(ts-prev_ts,self.time_budget)])
                else:
                    self.observed_configs.append(algorithm_meta_feat+dataset_meta_feat+[j, self.norm(prev_ts, self.time_budget), self.norm(ts-prev_ts,self.time_budget), prev_score])

                self.algorithms_perf_hist[algorithm_index,:,int(j.item())] = torch.FloatTensor([self.norm(prev_ts, self.time_budget), prev_score])
                self.observed_lc.append(self.algorithms_perf_hist[algorithm_index].tolist())
                self.algorithms_cost_count[algorithm_index] = ts
                self.observed_response.append(y_val)
                
            current_time = np.array(self.algorithms_cost_count).reshape(-1,1)
            time_to_full_budget = self.norm(self.remaining_budget_counter, self.time_budget)
            current_time_tensor = torch.FloatTensor(self.norm(current_time, self.time_budget)).reshape(-1,1)

            x_qry_cost = torch.concat((self.X_pre, 
                self.count.reshape(-1,1), 
                current_time_tensor), 
                axis=1)

            #predicts cost
            if self.cost_model_type == "MLP":
                self.cost_model = torch.load("cost_model.pt")

                x = torch.FloatTensor(self.observed_configs)
                y = torch.FloatTensor(self.observed_cost)
                if (y!=0).sum() !=0:
                    self.train_cost_model(x[y!=0],y[y!=0], epochs=50, lr=0.1)

                y_pred_cost = self.cost_model(x_qry_cost)

            elif self.cost_model_type == "Percentile":

               y_pred_cost = torch.FloatTensor( 
                   [self.find_budget(self.algorithms_list[i], self.algorithms_cost_count[i], self.percentile) for i in range(self.nA)] ).reshape(-1)   

            else:
                y_pred_cost = self.cost_model.predict(np.array(x_qry_cost)) 
                y_pred_cost = torch.FloatTensor(y_pred_cost)


            if self.time_input=="time_to_best":
                best_time_pred = self.best_cost_model.predict(np.array(self.X_pre)).reshape(-1,1)
                best_time_pred  = torch.FloatTensor(self.norm(best_time_pred-current_time, self.time_budget))
                a=torch.cat((time_to_full_budget*torch.ones((self.nA,1)), best_time_pred), axis=1)
                time_step = torch.min(a, axis=1)[0].reshape(-1,1)
            elif self.time_input == "remaining_time":
                time_step = torch.FloatTensor([time_to_full_budget])*torch.ones((self.nA,1))
            else:
                time_step = torch.FloatTensor(y_pred_cost).reshape(-1,1)

            last_scores = torch.FloatTensor(self.validation_last_scores).reshape(-1,1)
            current_time = torch.FloatTensor(self.norm(current_time, self.time_budget)).reshape(-1,1)

            #finetune surrogate
            x_spt = torch.FloatTensor(self.observed_configs).to(self.device)
            y_spt = torch.FloatTensor(self.observed_response).to(self.device)

            if self.conf["use_perf_hist"]:
                w_spt = torch.FloatTensor(self.observed_lc).to(self.device)
                x_qry = torch.concat((self.X_pre, 
                    self.count.reshape(-1,1), 
                    time_step),
                    axis=1)

                w_qry = self.algorithms_perf_hist

            else:
                w_spt = None
                w_qry = None
                x_qry = torch.concat((self.X_pre, 
                    self.count.reshape(-1,1), 
                    current_time,
                    time_step,
                    last_scores), 
                    axis=1)

            loss_history = self.fsbo_model.finetuning(x_spt,y_spt, w_spt, epochs = self.finetuning_epochs, finetuning_lr = self.finetuning_lr, patience=self.finetuning_patience)
            

            
            #compute acqusition function
            best_y =max(self.observed_response)
            mean, std = self.fsbo_model.predict(x_spt,y_spt, x_qry, w_spt, w_qry) 
            ei = torch.FloatTensor(self.EI(mean,std, best_y)).to(self.device)

            ei = torch.multiply(ei, self.convergence)

            if self.acquistion_type == "cost_balanced_ei":
                ei = torch.divide(ei, y_pred_cost)

            next_algorithm = torch.argmax(ei).item()
            c = self.denorm(y_pred_cost[next_algorithm], self.time_budget).item()

            self.last_predicted_cost = c
            c = (np.exp(self.trials[next_algorithm])*c).item()
            b = next_algorithm
            
            c = np.min((self.remaining_budget_counter, c)).item()

            a = np.argmax(self.validation_last_scores)
            self.remaining_budget_counter-=c
            
            return a,b, c

    def EI(self, mean, sigma, best_f, epsilon = 0):


        with np.errstate(divide='warn'):
            imp = mean -best_f - epsilon
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return ei
