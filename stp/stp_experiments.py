# ===========
#    SETUP
# ===========
try:
    import serene
except ImportError as e:
    import sys
    sys.path.insert(0, '.')
    import serene

import os
import tarfile
from importlib import reload
import csv

if "SERENE_PYTHON" not in os.environ:
    ###############################
    # ** VALUE HAS BEEN MODIFIED **
    os.environ["SERENE_PYTHON"] = "/workspace/serene-python-client"

project_path = os.environ["SERENE_PYTHON"]

from serene import SSD
from serene.elements.semantics.base import KARMA_DEFAULT_NS, DEFAULT_NS, ALL_CN, OBJ_PROP, UNKNOWN_CN, UNKNOWN_DN

try:
    import integration as integ
    import alignment as al
except ImportError as e:
    import sys
    sys.path.insert(0, '.')
    import integration as integ
    import alignment as al

import logging
from logging.handlers import RotatingFileHandler


######################################
# ** THIS SECTION HAS BEEN MODIFIED **
from pathlib import Path
import pickle
def serialize_obj(obj, file):
    with open(file, "wb") as f:
        pickle.dump(obj, f)


def deserialize_obj(file):
    with open(file, "rb") as f:
        return pickle.load(f)
######################################


# setting up the logging
log_file = os.path.join(project_path, "stp", "resources", 'benchmark_stp.log')
log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(module)s: %(message)s',
                                  '%Y-%m-%d %H:%M:%S')
my_handler = RotatingFileHandler(log_file, mode='a', maxBytes=5 * 1024 * 1024,
                                 backupCount=5, encoding=None, delay=0)
my_handler.setFormatter(log_formatter)
my_handler.setLevel(logging.DEBUG)
# logging.basicConfig(filename=log_file,
#                     level=logging.DEBUG, filemode='w+',
#                     format='%(asctime)s %(levelname)s %(module)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

log = logging.getLogger()
log.setLevel(logging.DEBUG)
log.addHandler(my_handler)
# =======================
#
#  Start with a connection to the server...
#
# =======================
###############################
# ** VALUE HAS BEEN MODIFIED **
sn = serene.Serene(
    host='serene',
    port=8080,
)
print(sn)

if "SERENE_BENCH" not in os.environ:
    os.environ["SERENE_BENCH"] = os.path.join(project_path, "tests", "resources")

benchmark_path = os.path.join(os.environ["SERENE_BENCH"], "museum_benchmark")
print("benchmark_path: ", benchmark_path)

print("========Optional cleaning=============")
# Removes all server elements
for octo in sn.octopii.items:
    sn.octopii.remove(octo)

for ssd in sn.ssds.items:
    sn.ssds.remove(ssd)

for ds in sn.datasets.items:
    sn.datasets.remove(ds)

for on in sn.ontologies.items:
    sn.ontologies.remove(on)

# =======================
#
#  Add ontologies from the museum benchmark
#
# =======================
owl_dir = os.path.join(benchmark_path, "owl")

ontologies = []
for path in os.listdir(owl_dir):
    f = os.path.join(owl_dir, path)
    ontologies.append(sn.ontologies.upload(f))
# add Unknown class
# unknown_owl = os.path.join("unknown", "unknown_class.ttl")
unknown_owl = os.path.join(project_path, "stp", "unknown", "unknown_class.ttl")
print("Unknown ontology ", unknown_owl)
ontologies.append(sn.ontologies.upload(unknown_owl))

print("*********** Ontologies from museum benchmark")
for onto in ontologies:
    print(onto)
print()

# =======================
#
#  Add datasets from the museum benchmark and corresponding semantic source descriptions
#
# =======================
dataset_dir = os.path.join(benchmark_path, "dataset")
ssd_dir = os.path.join(benchmark_path, "ssd")

datasets = [] # list of museum datasets
ssds = [] # list of semantic source descriptions from museum benchmark

# we need to unzip first the data
if os.path.exists(os.path.join(dataset_dir, "data.tar.gz")):
    with tarfile.open(os.path.join(dataset_dir, "data.tar.gz")) as f:
        f.extractall(path=dataset_dir)

# input("Press enter to upload datasets and ssds...")

for ds in sorted(os.listdir(dataset_dir)):
    if ds.endswith(".gz"):
        continue
    ds_f = os.path.join(dataset_dir, ds)
    ds_name = os.path.splitext(ds)[0]
    # associated semantic source description for this dataset
    ssd_f = os.path.join(ssd_dir, ds_name + ".ssd")

    print("Adding dataset: " + str(ds_f))
    dataset = sn.datasets.upload(ds_f, description="museum_benchmark")
    datasets.append(dataset)

    print("Adding ssd: " + str(ssd_f))
    new_json = dataset.bind_ssd(ssd_f, ontologies, KARMA_DEFAULT_NS)
    # _logger.info("+++++++++++ obtained ssd json")
    # _logger.info(new_json)
    if len(new_json["mappings"]) < 1:
        print("     --> skipping this file...")
        continue
    else:
        empty_ssd = SSD(dataset, ontologies)
        ssd = empty_ssd.update(new_json, sn.datasets, sn.ontologies)
        ssds.append(sn.ssds.upload(ssd))
    # we remove the csv dataset
    os.remove(ds_f)


# =======================
#
#  Helper methods
#
# =======================

reload(integ)
reload(al)
reload(serene)

# =======================
#
#  Karma style: Changing # known semantic models
#
# =======================
######################################
# ** THIS SECTION HAS BEEN MODIFIED **
# chuffed_lookup_path = os.path.join(project_path, "stp", "minizinc")
chuffed_lookup_path = "/home/serene/stp/minizinc"
######################################


result_csv = os.path.join(project_path, "stp", "resources", "benchmark_museum_loo_nounknown.csv")
with open(result_csv, "w+") as f:
    csvf = csv.writer(f)
    csvf.writerow(["experiment", "octopus", "dataset", "name", "ssd",
                   "method", "solution", "status",
                   "jaccard", "precision", "recall",
                   "karma_jaccard", "karma_precision", "karma_recall",
                   "train_flag", "time", "chuffed_path", "simplify", "chuffed_time",
                   "soft_assumptions", "pattern_significance", "sm_accuracy", "sol_accuracy",
                   "match_score", "cost", "objective", "cor_match_score", "cor_cost"])  # header

    sample_range = list(range(len(ssds)))
    len_sample = len(sample_range)
    benchmark_results = [["experiment", "octopus", "dataset", "name", "ssd",
                          "method", "solution", "status",
                          "jaccard", "precision", "recall",
                          "karma_jaccard", "karma_precision", "karma_recall",
                          "train_flag", "time", "chuffed_path", "simplify", "chuffed_time",
                          "soft_assumptions", "pattern_significance", "sm_accuracy", "sol_accuracy",
                          "match_score", "cost", "objective", "cor_match_score", "cor_cost"]]

    print("Starting benchmark")

    ###############################
    # ** VALUE HAS BEEN MODIFIED **
    for cur_id in sample_range[:3]:
        print("Currently selected ssd: ", cur_id)
        test_sample = [cur_id]

        # here we run only leave-one-out setting
        num = len(sample_range) -1
        train_sample = sample_range[:max(cur_id+num+1 - len_sample,0)] + sample_range[cur_id+1: cur_id+1+num]
        # ** MIRA MODIFIES HERE **
        train_sample = train_sample[:14]
        print("     train sample size: ", num)

        try:
            ######################################
            # ** THIS SECTION HAS BEEN MODIFIED **
            octo_file = "/workspace/serene-python-client/stp/resources/octopus/" + ssds[cur_id].name + ".pkl"
            octo_pattern_file = "/workspace/serene-python-client/stp/resources/octopus/" + ssds[cur_id].name + ".pattern.pkl"
            octo_matcher_predict_file = "/workspace/serene-python-client/stp/resources/octopus/" + ssds[cur_id].name + ".matcher_predict.pkl"

            print("OCTO_FILE:", octo_file)

            if Path(octo_file).exists():
                octo = deserialize_obj(octo_file)
            else:
                octo = al.create_octopus(sn, ssds, train_sample, ontologies)
                octo._matcher._pp = None
                serialize_obj(octo, octo_file)

            ######################################

        except Exception as e:
            logging.error("Octopus creation failed: {}".format(e))
            print("Octopus creation failed: {}".format(e))
            logging.exception(e)
            continue

        octo_csv = os.path.join(project_path, "stp", "resources", "storage",
                                "patterns.{}.csv".format(octo.id))
        try:
            ######################################
            # ** THIS SECTION HAS BEEN MODIFIED **
            if Path(octo_pattern_file).exists():
                octo_patterns = deserialize_obj(octo_pattern_file)
            if Path(octo_pattern_file + ".error").exists():
                octo_patterns = None
            else:
                octo_patterns = octo.get_patterns(octo_csv)
                serialize_obj(octo_patterns, octo_pattern_file)

            ######################################
        except Exception as e:
            print("failed to get patterns: {}".format(e))
            logging.warning("failed to get patterns: {}".format(e))
            logging.exception(e)
            octo_patterns = None

            with open(octo_pattern_file + ".error", "w") as f:
                f.write("error: {}".format(e))

            # ** MIRA MODIFIES HERE **
            # import IPython
            # IPython.embed()

        print("Uploaded octopus:", octo)

        chuffed_paths = [
            (os.path.join(chuffed_lookup_path, "chuffed-rel2onto_20170718"),
             True, None, False, 1.0),
            (os.path.join(chuffed_lookup_path, "chuffed-rel2onto_20170722"),
             True, None, True, 1.0),
            (os.path.join(chuffed_lookup_path, "chuffed-rel2onto_20170728"),
             False, octo_patterns, False, 1.0),
            (os.path.join(chuffed_lookup_path, "chuffed-rel2onto_20170728"),
             False, octo_patterns, True, 1.0),
            (os.path.join(chuffed_lookup_path, "chuffed-rel2onto_20170728"),
             False, octo_patterns, True, 5.0),
            (os.path.join(chuffed_lookup_path, "chuffed-rel2onto_20170728"),
             False, octo_patterns, True, 10.0),
            (os.path.join(chuffed_lookup_path, "chuffed-rel2onto_20170728"),
             False, octo_patterns, True, 20.0),
        ]

        print("Getting labels in the training set")
        train_labels = []
        for i in train_sample:
            train_labels += [d.full_label for d in ssds[i].data_nodes]
        train_labels = set(train_labels)

        ssd = ssds[cur_id]
        dataset = [ds for ds in datasets if ds.filename == ssd.name + ".csv"][0]
        # we do not specify columns
        column_names = None
        print("Chosen dataset: ", dataset)
        print("Chosen SSD id: ", ssd.id)

        # we should do prediction outside of solvers so that solvers use the cached results!
        print("Doing schema matcher part")
        ######################################
        # ** THIS SECTION HAS BEEN MODIFIED **
        if Path(octo_matcher_predict_file).exists():
            df = deserialize_obj(octo_pattern_file)
        else:
            df = octo.matcher_predict(dataset)
            serialize_obj(df, octo_matcher_predict_file)
        ######################################

        cor_ssd = al.process_unknown(ssd, train_labels)
        try:
            sm_accuracy = al.compare_semantic_label(al.semantic_label_df(cor_ssd, "user_label"),
                                             df[["column_id", "label"]])["accuracy"]
        except Exception as e:
            logging.error("Match score fail: {}".format(e))
            print("Match score fail: {}".format(e))
            sm_accuracy = None

        # cp solver approach
        for (chuffed, simpl, pat, soft, pts) in chuffed_paths:
            print("---> trying chuffed {}".format(chuffed))
            res = al.do_chuffed(sn, octo, dataset, column_names, cor_ssd, ssd,
                             train_flag=False,
                             chuffed_path=chuffed, simplify=simpl,
                             patterns=pat, soft_assumptions=soft,
                             pattern_sign=pts, accu=sm_accuracy,
                             experiment="eknown_{}".format(len(train_sample)))
            benchmark_results += res
            csvf.writerows(res)

        # karma approach
        print("---> trying karma")
        res = al.do_karma(sn, octo, dataset, cor_ssd, ssd, train_flag=False,
                       accu=sm_accuracy, experiment="eknown_{}".format(len(train_sample)))
        benchmark_results += res
        csvf.writerows(res)
        f.flush()

    print("Benchmark finished!")
    print(benchmark_results)