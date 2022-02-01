from functools import lru_cache
import random
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
        self.cost_model_type = "RF"
        self.acquistion_type = "ei"
        self.conf = {"kernel":"52", 
                    "nu":2.5, 
                    "ard":None, 
                    "device": self.device, 
                    "context_size": 10, 
                    "lr":0.05, 
                    "loss_tol": 0.001}

        self.meta_learning_epochs = 5
        self.finetuning_lr = 0.01
        self.finetuning_epochs = 10
        self.finetuning_patience = 100
        self.n_hidden = 10
        self.n_layers = 2
        self.n_batches = 1000
        self.n_init = 1
        self.bias = 0
    
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
        self.count = torch.zeros(self.nA).to(self.device)
        self.convergence = torch.ones(self.nA).to(self.device)
        self.trials  = torch.zeros(self.nA).to(self.device)
        self.algorithms_list = list(self.algorithms_meta_features.keys())
        self.algorithms_list.sort()
        self.iter = 0
        self.time_budget = float(dataset_meta_features["time_budget"])


        if self.metatrained:

            df = pd.DataFrame([self.dataset_meta_features])
            df_numerical = np.array(df[self.numerical_dataset_metafeatures].astype(float))
            df_categorical = df[self.categorical_dataset_metafeatures]
            df_categorical = self.enc.transform(np.array(df_categorical))
            dataset_features = np.concatenate([df_numerical, df_categorical.toarray()], axis=1)
            self.dataset_features  = self.scaler.transform(dataset_features)

            d = distance_matrix(self.dataset_features, self.metatrain_dataset_features)
            self.most_similar_metatrain_dataset = self.index_dataset[np.argmin(d)]
            self.similar_datasets_sorted = np.argsort(d)
            self.initial_alg_test_id= self.best_algorithm_per_dataset[self.most_similar_metatrain_dataset]["id_test"]
            self.initial_alg_val_id= self.best_algorithm_per_dataset[self.most_similar_metatrain_dataset]["id_val"]

        self.X_pre = []

        for alg_name in self.algorithms_list:
            alg = algorithms_meta_features[alg_name]
            hp = [float(x) for x in list(alg.values())]
            self.X_pre.append(hp+self.dataset_features.tolist()[0])
        
        self.X_pre = torch.FloatTensor(self.X_pre).to(self.device)

    def process_data(self, dataset_meta_features, algorithms_meta_features, validation_learning_curves, test_learning_curves):

        """
        Returns a training matrix and validation matrix for initializing the meta-learning model
        """
        datasets = list(dataset_meta_features.keys())
        algorithms = list(algorithms_meta_features.keys())
        n_datasets = len(datasets)
        split = int(n_datasets*0.9)

        train_data = {}
        test_data =  {}

        self.enc = OneHotEncoder(handle_unknown="ignore")
        self.scaler = MinMaxScaler()

        self.index_dataset = list(dataset_meta_features.keys())
        df_dataset_features = pd.DataFrame(dataset_meta_features.values(), index=dataset_meta_features.keys())
        df_numerical = np.array(df_dataset_features[self.numerical_dataset_metafeatures].astype(float))
        df_categorical = df_dataset_features[self.categorical_dataset_metafeatures]
        df_categorical = self.enc.fit_transform(np.array(df_categorical))

        dataset_features = np.concatenate([df_numerical, df_categorical.toarray()], axis=1)
        self.metatrain_dataset_features  = self.scaler.fit_transform(dataset_features)
        self.best_algorithm_per_dataset = {}

        X_time = []
        y_time = []

        for i, dataset in enumerate(datasets):
            X =[]
            y_val = []
            y_test = []
            time_budget = float(dataset_meta_features[dataset]["time_budget"])
            self.best_algorithm_per_dataset[dataset] = {"id_test":0, "perf_test":0, "id_val":0, "perf_val":0}
            idx = self.index_dataset.index(dataset)
            temp_dataset_feat = self.metatrain_dataset_features[idx].tolist()
            
            for algorithm, meta_features in algorithms_meta_features.items():
                y_val+=validation_learning_curves[dataset][algorithm].scores.tolist()
                y_test+=test_learning_curves[dataset][algorithm].scores.tolist()
                timestamps = validation_learning_curves[dataset][algorithm].timestamps
                values = [float(x) for x in list(meta_features.values())]
                ts_prev = 0
                for j, ts in enumerate(timestamps):
                    X.append(values+temp_dataset_feat+[j,ts_prev/time_budget])
                    X_time.append(values+temp_dataset_feat+[j,ts_prev/time_budget])
                    y_time.append(ts-ts_prev)
                    ts_prev = ts

                if self.best_algorithm_per_dataset[dataset]["perf_test"]<y_test[-1]:
                    self.best_algorithm_per_dataset[dataset]["id_test"]=algorithm
                    self.best_algorithm_per_dataset[dataset]["perf_test"]=y_test[-1]

                if self.best_algorithm_per_dataset[dataset]["perf_val"]<y_val[-1]:
                    self.best_algorithm_per_dataset[dataset]["id_val"]=algorithm
                    self.best_algorithm_per_dataset[dataset]["perf_val"]=y_val[-1]

            if i<split:
                train_data[dataset] = {"X": X, "y_val": y_val, "y_test":y_test}
            else:
                test_data[dataset] = {"X": X, "y_val": y_val, "y_test":y_test}
        print(y_time)
        return train_data, test_data, X_time, y_time, len(X[0])


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

        train_data, test_data, X_cost, y_cost, input_size = self.process_data(dataset_meta_features, algorithms_meta_features, validation_learning_curves, test_learning_curves)
        feature_extractor = MLP(input_size, n_hidden=self.n_hidden, n_layers=self.n_layers).to(self.device)        
        self.fsbo_model = FSBO(train_data, test_data, feature_extractor=feature_extractor , conf=self.conf).to(self.device)
        self.fsbo_model.train(epochs=self.meta_learning_epochs, n_batches=self.n_batches)
        
        if self.cost_model_type == "MLP":
            self.cost_model = MLP(n_input = len(X_cost[0]),
                                    n_hidden = self.n_hidden,
                                    n_layers = self.n_layers,
                                    n_output = 1).to(self.device)
            losses = self.train_cost_model(X_cost, y_cost, lr = 0.0001, epochs = 1000)
            torch.save(self.cost_model, "cost_model.pt")
        else:
            self.cost_model = RandomForestRegressor(max_depth=50)
            y_target = np.log(np.array(y_cost)+1)
            self.cost_model.fit(np.array(X_cost), y_target)

        self.metatrained = 1

    def train_cost_model (self,  X, y_cost, lr = 0.001, epochs = 100):

        optimizer = torch.optim.Adam(self.cost_model.parameters(), lr= lr)
        X = torch.FloatTensor(X).to(self.device)
        y_cost = torch.FloatTensor(y_cost).to(self.device)
        y_cost = torch.log(y_cost+1)
        self.y_max_cost = torch.max(y_cost)
        y_cost /= self.y_max_cost

        loss_fn = torch.nn.MSELoss()
        losses = [np.inf]
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            pred = self.cost_model(X)
            loss = loss_fn(pred, y_cost.reshape(-1,1))
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
            a = None
            b =  self.algorithms_list.index(self.initial_alg_val_id)

            algorithm_meta_feat = [float(x) for x in list(self.algorithms_meta_features[self.initial_alg_val_id].values())]
            x = torch.FloatTensor(algorithm_meta_feat+self.dataset_features.tolist()[0] +[0,0]).to(self.device)

            if self.cost_model_type == "MLP":
                c = (self.cost_model(x)*self.y_max_cost).item()
            
            else:
                c = self.cost_model.predict(np.array(x).reshape(1,-1)).item()
            
            self.predicted_cost.append(c)
            c = np.exp(c).item()
                
            return a,b,c+self.bias


        else:


            algorithm_index, ts, y_val = observation
            
            #self.convergence keeps track of the algorithms that converged (=0) to omit them in EI
            if y_val == self.validation_last_scores[algorithm_index]  and y_val !=0 and ts==self.algorithms_cost_count[algorithm_index]: 
                self.convergence[algorithm_index]=0
            else:
                self.validation_last_scores[algorithm_index] = y_val

            #self.trials -> how many times the algorithm has returned 0
            #if y_val !=0:
            #    self.trials[algorithm_index]=0           
            #else:
            #    self.trials[algorithm_index]+=1
                
            #self.count ->  budge count,  just increase when the algorithm did finished
            if ts!=self.algorithms_cost_count[algorithm_index]:
                self.count[algorithm_index]+=1

            if y_val !=0:
                self.trials[algorithm_index]=0           
            else:
                self.trials[algorithm_index]+=1            

            
            #adds the new observation to the history

            if  ts!=self.algorithms_cost_count[algorithm_index] or len(self.observed_configs)==0:

                algorithm = self.algorithms_list[algorithm_index]
                j = self.count[algorithm_index]
                algorithm_meta_feat = [float(x) for x in list(self.algorithms_meta_features[algorithm].values())]
                self.observed_cost.append(ts-self.algorithms_cost_count[algorithm_index])
                self.algorithms_cost_count[algorithm_index] = ts
                self.observed_configs.append(algorithm_meta_feat+self.dataset_features.tolist()[0]+[j, self.algorithms_cost_count[algorithm_index]/self.time_budget])
                self.observed_response.append(y_val)
               
                

            #finetune surrogate
            x_spt = torch.FloatTensor(self.observed_configs).to(self.device)
            y_spt = torch.FloatTensor(self.observed_response).to(self.device)
            loss_history = self.fsbo_model.finetuning(x_spt,y_spt, epochs = self.finetuning_epochs, finetuning_lr = self.finetuning_lr, patience=self.finetuning_patience)
            
            x_qry = torch.concat((self.X_pre, self.count.reshape(-1,1), torch.FloatTensor(self.algorithms_cost_count).reshape(-1,1)/self.time_budget), axis=1)

            #predicts cost
            if self.cost_model_type == "MLP":
                self.cost_model = torch.load("cost_model.pt")
                y_pred_cost = self.cost_model(x_qry)
                ##here: finetune##
                
            else:
                y_pred_cost = self.cost_model.predict(np.array(x_qry))
                y_pred_cost = torch.FloatTensor(y_pred_cost)


            #second prediction
            x = np.array(self.observed_configs)
            y = np.array(self.observed_cost)
            non_zero_id = np.where(y!=0)[0]
            n_observed = len(non_zero_id)
            if n_observed>0:
                neigh = KNeighborsRegressor(n_neighbors=min(5,n_observed))
                neigh.fit(x[non_zero_id], y[non_zero_id])
                y_pred = neigh.predict(np.array(x_qry))            
            
            
            #compute acqusition function
            best_y =max(self.observed_response)
            mean, std = self.fsbo_model.predict(x_spt,y_spt, x_qry) 
            ei = torch.FloatTensor(self.EI(mean,std, best_y)).to(self.device)

            #masks to avoid picking converged algorithms
            ei = torch.multiply(ei, self.convergence)

            if self.acquistion_type == "cost_balanced_ei":
                ei = torch.divide(ei, y_pred_cost)

            #formats the output

            if self.iter <= self.n_init:
                d = self.similar_datasets_sorted[0][self.iter]
                next_algrithm_name = self.best_algorithm_per_dataset[self.index_dataset[d]]["id_val"]
                next_algorithm = self.algorithms_list.index(next_algrithm_name)

                algorithm_meta_feat = [float(x) for x in list(self.algorithms_meta_features[next_algrithm_name].values())]
                x = torch.FloatTensor(algorithm_meta_feat+self.dataset_features.tolist()[0] +[0,0]).to(self.device)
               
                b = next_algorithm

                if self.cost_model_type == "MLP":
                    c = (self.cost_model(x)*self.y_max_cost).item()
                else:
                    c = self.cost_model.predict(np.array(x).reshape(1,-1)).item()
                c = np.exp(c).item()
                self.predicted_cost.append(c)
                    
            else:
                next_algorithm = torch.argmax(ei).item()
                c = y_pred_cost[next_algorithm].item()
                self.predicted_cost.append(np.exp(c).item())
                c = np.exp(self.trials[next_algorithm]+c).item()
                b = next_algorithm

            a = np.argmax(self.validation_last_scores)

            return a,b, c+self.bias

    def EI(self, mean, sigma, best_f, epsilon = 0):


        with np.errstate(divide='warn'):
            imp = mean -best_f - epsilon
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return ei
