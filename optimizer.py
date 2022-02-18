import numpy as np 
import pandas as pd
import json
import subprocess
import copy

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
   


#conf = generate_random_conf(search_space)
#config_file = "C1.json"
#result = run_config(conf, config_file)
#print(result)
#repetitions = 3

#results = []
#for hp in ["percentile", "acquisition_type","include_subset_metafeat", "include_dataset_metafeat","use_categorical_hp", "use_perf_hist"]:
#for hp in ["use_best_alg", "time_input"]:
#    conf_list = generate_ablation(search_space, default_conf.copy(), hp)
#    for i, conf in enumerate(conf_list):
#        for j in range(repetitions):
#            config_name = "_".join(["AB", hp, str(i), "rep", str(j)])+".json"
#            result =  run_config(conf, config_name)
#            results.append((hp, j, conf[hp], result))

#pd.DataFrame(results).to_csv("results/ablation.csv")

results = []
for i in range(7,100):

    config = generate_random_conf(search_space)
    config_name = "RS"+str(i)+".json"
    result =  run_config(config, config_name)
    results.append((config_name, result))
    pd.DataFrame(results).to_csv("results/random_search2.csv")


