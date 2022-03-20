import csv
import os
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn import metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###########################################################################
# Metric Computation
###########################################################################

# fpr_recall
def fpr_recall(conf, label, tpr):
    ind_conf = conf[label != -1]
    ood_conf = conf[label == -1]
    num_ind = len(ind_conf)
    num_ood = len(ood_conf)

    recall_num = int(np.floor(tpr * num_ind))
    thresh = np.sort(ind_conf)[-recall_num]
    num_fp = np.sum(ood_conf > thresh)
    fpr = num_fp / num_ood
    return fpr, thresh


def auc(conf, label):
    ind_indicator = np.zeros_like(label)
    ind_indicator[label != -1] = 1

    fpr, tpr, thresholds = metrics.roc_curve(ind_indicator, conf)
    precision_in, recall_in, thresholds_in = metrics.precision_recall_curve(
        ind_indicator, conf
    )
    precision_out, recall_out, thresholds_out = metrics.precision_recall_curve(
        1 - ind_indicator, 1 - conf
    )

    auroc = metrics.auc(fpr, tpr)
    aupr_in = metrics.auc(recall_in, precision_in)
    aupr_out = metrics.auc(recall_out, precision_out)

    return auroc, aupr_in, aupr_out


def compute_all_metrics(conf, label, file_path=None, verbose=True):
    # normalize conf
    conf = (conf - np.min(conf)) / (np.max(conf) - np.min(conf))
    
    recall = 0.95
    fpr, thresh = fpr_recall(conf, label, recall)
    auroc, aupr_in, aupr_out = auc(conf, label)

    if verbose:
        print(
            "FPR@{}: {:.2f}, AUROC: {:.2f}, AUPR_IN: {:.2f}, AUPR_OUT: {:.2f}".format(
                recall, 100 * fpr, 100 * auroc, 100 * aupr_in, 100 * aupr_out
            )
        )
    
    results = [fpr, auroc, aupr_in, aupr_out]

    if file_path:
        save_csv(file_path, results)

    return results


def save_csv(file_path, results):
    save_path = os.path.join(file_path, "..")
    filename = file_path.split("/")[-1]
    [fpr, auroc, aupr_in, aupr_out] = results

    save_exp_name = os.path.join(
        save_path, "summary_{}.csv".format(file_path.split("/")[-2])
    )
    fieldnames = [
        "Experiment_PATH",
        "FPR@95",
        "AUROC",
        "AUPR_IN",
        "AUPR_OUT",
    ]
    write_content = {
        "Experiment_PATH": filename,
        "FPR@95": round(100 * fpr, 2),
        "AUROC": round(100 * auroc, 2),
        "AUPR_IN": round(100 * aupr_in, 2),
        "AUPR_OUT": round(100 * aupr_out, 2),
    }

    if not os.path.exists(save_exp_name):
        with open(save_exp_name, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(write_content)
    else:
        with open(save_exp_name, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(write_content)


###########################################################################
# Evaluator
###########################################################################

class Evaluator:
    def __init__(
        self,
        net: nn.Module,
    ):
        self.net = net

    def inference(self, data_loader: DataLoader, post_processor, num_break=None):
        conf_list, ddood_list = [], []

        cnt = 0
        for data, label in data_loader:
            data = data.to(device)
            label = label.to(device)

            conf = post_processor(self.net, data)

            for idx in range(len(data)):
                conf_list.append(conf[idx].cpu().tolist())
                ddood_list.append(label[idx].cpu().tolist())

            cnt += data.shape[0]

            if num_break is not None:
                if cnt > num_break:
                    break

            
        conf_list = np.array(conf_list)
        ddood_list = np.array(ddood_list, dtype=int)
        

        return conf_list, ddood_list

    
    def eval_ood(
        self,
        id_data_loader: DataLoader,
        ood_data_loaders: List[DataLoader],
        post_processor,
        method: str = "each",
    ):
        self.net.eval()

        if method == "each":
            results_matrix = []

            id_name = id_data_loader.dataset.name

            print(f"Performing inference on {id_name} dataset...")
            id_conf, id_ddood = self.inference(
                id_data_loader, post_processor, num_break=None
            )

            for i, ood_dl in enumerate(ood_data_loaders):
                # ood_name = ood_dl.dataset.name

                # print(f"Performing inference on {ood_name} dataset...")
                ood_conf, ood_ddood = self.inference(
                    ood_dl, post_processor, len(id_data_loader.dataset)
                )

                # set ood to -1 label
                ood_ddood[:] = -1

               
                conf = np.concatenate([id_conf, ood_conf])
                label = np.concatenate([id_ddood, ood_ddood])

                # print(f"Computing metrics on {id_name} + {ood_name} dataset...")
                results = compute_all_metrics(conf, label)
                # self._log_results(results, csv_path, dataset_name=ood_name)

                results_matrix.append(results)

            results_matrix = np.array(results_matrix)

            print(f"Computing mean metrics...")
            results_mean = np.mean(results_matrix, axis=0)
            # self._log_results(results_mean, csv_path, dataset_name="mean")
            print_results = results_mean

        elif method == "full":
            data_loaders = [id_data_loader] + ood_data_loaders

            conf_list, ddood_list = list(), list()

            for i, test_loader in enumerate(data_loaders):
                # name = test_loader.dataset.name
                # print(f"Performing inference on {name} dataset...")
                if i == 0:
                    conf, ddood = self.inference(test_loader, post_processor, None)
                else:
                    conf, ddood = self.inference(test_loader, post_processor, len(id_data_loader.dataset))

                if i != 0:
                    ddood[:] = -1

                
                conf_list.extend(conf)
                ddood_list.extend(ddood)
               

           
            conf_list = np.array(conf_list)
            label_list = np.array(ddood_list).astype(int)

            # print(f"Computing metrics on combined dataset...")
            results = compute_all_metrics(conf_list, label_list)

            # if csv_path:
            #     self._log_results(results, csv_path, dataset_name="full")

            print_results = results
        
        [fpr, auroc, aupr_in, aupr_out] = print_results
        print_dict = {
            "FPR@95": fpr,
            "AUROC": auroc,
            "AUPR_IN": aupr_in,
            "AUPR_OUT": aupr_out,
        }
        return print_dict
            

    def _log_results(self, results, csv_path, dataset_name=None):
        [fpr, auroc, aupr_in, aupr_out] = results

        write_content = {
            "dataset": dataset_name,
            "FPR@95": "{:.2f}".format(100 * fpr),
            "AUROC": "{:.2f}".format(100 * auroc),
            "AUPR_IN": "{:.2f}".format(100 * aupr_in),
            "AUPR_OUT": "{:.2f}".format(100 * aupr_out),
        }
        fieldnames = list(write_content.keys())
        # print(write_content, flush=True)

        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(write_content)
        else:
            with open(csv_path, "a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(write_content)
   