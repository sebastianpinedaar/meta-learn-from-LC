import numpy as np 
import pandas as pd
import json
import subprocess
import copy
from scipy.stats import norm as norm
import itertools
from pyGPGO.surrogates.RandomForest import RandomForest
from sklearn.preprocessing import OneHotEncoder

search_space = {"use_ranker": [True, False],
                "max_trials" : [1,2,3],
                "norm_type" : ["log", "linear"],
                "perf_measure" : ["ratio", "alc", "accuracy"],
                "time_input" : ["time_to_best", "predicted_time", "remaining_budget"],
                "percentile" : [50, 60, 70],
                "initial_factor" : [0,1 ],
                "cost_model_type" : ["Percentile", "RF"],
                "acquisition_type" : ["ei", "cost_balanced_ei"],
                "include_subset_metafeat" : [True, False],
                "include_dataset_metafeat" : [True, False],
                "use_categorical_hp": [True, False],
                "use_perf_hist": [True, False], 
                "use_best_alg": [True, False]}

default_conf =  {"use_ranker": False,
                "max_trials" : 3,
                "norm_type" : "log",
                "perf_measure" : "ratio",
                "time_input" : "predicted_time",
                "percentile" : 70,
                "initial_factor" : 1,
                "cost_model_type" : "RF",
                "acquisition_type" : "cost_balanced_ei",
                "include_subset_metafeat" : False,
                "include_dataset_metafeat" : True,
                "use_categorical_hp": False,
                "use_perf_hist": False,
                "use_best_alg": False}

def generate_random_conf(search_space):

    conf = {}

    for hp in search_space.keys():
        conf[hp] = np.random.choice(search_space[hp], 1).item()

    return conf

def generate_ablation(search_space, default_conf, hp):

    list_conf = []

    
    if hp == "Percentile":
        default_conf["cost_model_type"] = "Percentile"

    for value in search_space[hp]:
        default_conf[hp] = value
        list_conf.append(default_conf.copy())

    return list_conf



def EI(mean, sigma, best_f, epsilon = 0):    
    with np.errstate(divide='warn'):
        imp = mean -best_f - epsilon
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei



def run_config (conf, config_name):

    with open("config/"+config_name, "w") as f:
        json.dump(conf, f)

    cmd1 = 'python ingestion_program/ingestion.py --config_file '+config_name
    cmd2 = 'python scoring_program/score.py --config_file '+config_name

    p = subprocess.Popen(cmd1, stdout=subprocess.PIPE, shell=True)
    out, err = p.communicate() 
    p= subprocess.Popen(cmd1, shell=True)
    output = p.communicate()
    print (output[0])
    p= subprocess.Popen(cmd2, shell=True)

    output = p.communicate()
    print (output[0])

    with open("results/"+config_name, "r") as f:
        results = json.load(f)

    return results["total"]

def create_surrogate():

    return RandomForest()
    



tensor_ids = []
for a, b, c, d, e, f, g, h, i, j, k, l, m, n in itertools.product(search_space["use_ranker"], 
                                        search_space["max_trials"], 
                                        search_space["norm_type"], 
                                        search_space["perf_measure"],
                                        search_space["time_input"],
                                        search_space["percentile"],
                                        search_space["initial_factor"],
                                        search_space["cost_model_type"],
                                        search_space["acquisition_type"],
                                        search_space["include_subset_metafeat" ],
                                        search_space["include_dataset_metafeat"],
                                        search_space["use_categorical_hp"],
                                        search_space[ "use_perf_hist"],
                                        search_space["use_best_alg"]):
    tensor_ids.append([a,b,c,d,e, f, g, h, i, j, k , l , m, n])

n = 50

Lambda  =  np.array(tensor_ids)

response = np.arange(Lambda.shape[0])  
first_conf_ix = np.random.randint(Lambda .shape[0])
hp_names = list(default_conf.keys())

surrogate = create_surrogate()
pending_x = list(range(Lambda.shape[0]))
observed_x = [first_conf_ix]
pending_x.remove(first_conf_ix)
y = []
config_list = []

enc = OneHotEncoder(handle_unknown='ignore')
Lambda_encoded = enc.fit_transform(Lambda).toarray()

for _ in range(n):


    q_encoded = Lambda_encoded [observed_x]
    q = Lambda[observed_x]
    config = dict(zip(hp_names, q[-1]))
    response = run_config(config, "temp_conf.json")

    y.append(response)
    config["y"] =  response
    config_list.append(config)
    best_f = max(y)



    surrogate.fit(q_encoded,y)
    ei_list = []

    for i in range(100, len(pending_x)+100,100):
        
        mean, std = surrogate.predict(Lambda_encoded[i-100:min(i,len(pending_x))])
        ei = EI(mean, std, best_f = best_f)
        ei_list+=ei.tolist()


    candidate = pending_x[np.argmax(ei_list)]
    observed_x.append(candidate)
    pending_x.remove(candidate)

    pd.DataFrame(config_list).to_csv("results/random_search2.csv")
