import math

import numpy as np
import pandas as pd
from sklearn.metrics import auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import matthews_corrcoef

from .confusion_matrix import ConfusionMatrix
from .accuracy import Accuracy


class AUC:
    def __init__(self, class_mappings):
        self.class_mappings = class_mappings
        self.class_order = [int(i) for i in list(class_mappings.keys())]

        self.confusion_matrix = ConfusionMatrix(class_mappings=class_mappings)
        self.accuracy = Accuracy(class_mappings=class_mappings)

    def calculate_sense_spec(self, true, pred_categorical):
        """
        Calculate statistics for each class
        :return: statistics dict
        """
        # TO DO: Completely replace with scikit functions
        class_order = self.class_order.copy()
        class_order.append("overall")
        accuracy_dict = {}

        for class_ in class_order:
            accuracy_dict[class_] = self.accuracy.calculate_accuracy(
                true=true, pred_categorical=pred_categorical, target_class=class_
            )
        class_order.remove("overall")

        try:
            class_order = [int(i) for i in class_order]
        except BaseException:
            pass

        confusion_matrix = self.confusion_matrix.calculate_confusion_matrix(
            true=true, pred_categorical=pred_categorical, labels=class_order
        )
        FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
        FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
        TP = np.diag(confusion_matrix)
        TN = confusion_matrix.sum() - (FP + FN + TP)
        matthews = self.calculate_matthews_corr_coef(
            true, pred_categorical, target_class=None
        )

        FP = FP.astype(float)
        FN = FN.astype(float)
        TP = TP.astype(float)
        TN = TN.astype(float)

        with np.errstate(divide="ignore", invalid="ignore"):

            # Sensitivity, hit rate, recall, or true positive rate
            TPR = TP / (TP + FN)
            TPR = TPR.tolist()
            # Macro Sensitivity
            macro_sensitivity = np.mean([0 if math.isnan(x) else x for x in TPR])
            # Micro Sensitivity
            micro_sensitivity = sum(TP) / sum(TP + FN)
            # Macro Recall
            macro_rec = np.mean([0 if math.isnan(x) else x for x in TPR])
            # Micro Recall
            micro_rec = sum(TP) / sum(TP + FN)
            # Specificity or true negative rate
            TNR = TN / (TN + FP)
            TNR = TNR.tolist()
            # Precision or positive predictive value
            PPV = TP / (TP + FP)
            PPV = PPV.tolist()
            # Macro Precision
            macro_prec = np.mean([0 if math.isnan(x) else x for x in PPV])
            # Micro Precision
            micro_prec = sum(TP) / sum(TP + FP)
            # Negative predictive value
            NPV = TN / (TN + FN)
            NPV = NPV.tolist()
            # Fall out or false positive rate
            FPR = FP / (FP + TN)
            FPR = FPR.tolist()
            # False negative rate
            FNR = FN / (TP + FN)
            FNR = FNR.tolist()
            # F1
            F1 = 2 * np.array(PPV) * np.array(TPR) / (np.array(PPV) + np.array(TPR))
            F1 = F1.tolist()
            # Macro F1
            macro_f1 = np.mean([0 if math.isnan(x) else x for x in F1])
            # Micro F1
            micro_f1 = 2 * (micro_rec * micro_prec) / (micro_rec + micro_prec)
            # Macro Specificity
            macro_specificity = np.mean([0 if math.isnan(x) else x for x in TNR])
            # Micro Specificity
            micro_specificity = sum(TN) / sum(TN + FP)

        FP = FP.astype(float).tolist()
        FN = FN.astype(float).tolist()
        TP = TP.astype(float).tolist()
        TN = TN.astype(float).tolist()

        FP_dict = dict(zip(class_order, FP))
        FN_dict = dict(zip(class_order, FN))
        TP_dict = dict(zip(class_order, TP))
        TN_dict = dict(zip(class_order, TN))
        TPR_dict = dict(zip(class_order, ["nan" if math.isnan(x) else x for x in TPR]))
        TNR_dict = dict(zip(class_order, ["nan" if math.isnan(x) else x for x in TNR]))
        PPV_dict = dict(zip(class_order, ["nan" if math.isnan(x) else x for x in PPV]))
        NPV_dict = dict(zip(class_order, ["nan" if math.isnan(x) else x for x in NPV]))
        FPR_dict = dict(zip(class_order, ["nan" if math.isnan(x) else x for x in FPR]))
        FNR_dict = dict(zip(class_order, ["nan" if math.isnan(x) else x for x in FNR]))
        F1_dict = dict(zip(class_order, ["nan" if math.isnan(x) else x for x in F1]))
        matthews_dict = self.calculate_matthews_corr_coef(
            true, pred_categorical, target_class="all"
        )

        sense_spec_dict = {
            "FP": FP_dict,
            "FN": FN_dict,
            "TP": TP_dict,
            "TN": TN_dict,
            "TPR": TPR_dict,
            "TNR": TNR_dict,
            "PPV": PPV_dict,
            "NPV": NPV_dict,
            "FPR": FPR_dict,
            "FNR": FNR_dict,
            "F1": F1_dict,
            "Matthew's Classwise C C": matthews_dict,
            "Accuracy": accuracy_dict,
            "Macro F1": macro_f1,
            "Micro F1": micro_f1,
            "Macro Precision": macro_prec,
            "Micro Precision": micro_prec,
            "Macro Recall": macro_rec,
            "Micro Recall": micro_rec,
            "Macro Sensitivity": macro_sensitivity,
            "Micro Sensitivity": micro_sensitivity,
            "Macro Specificity": macro_specificity,
            "Micro Specificity": micro_specificity,
            "Matthew's Correlation Coefficient": matthews,
        }

        return sense_spec_dict

    def _create_precision_recall_points(self, prec, rec, threshold):
        points_list = []
        auc_value = auc(rec, prec)

        for i in range(len(prec) - len(threshold)):
            threshold = np.append(threshold, threshold[-1])

        for sample_idx in range(len(prec)):
            points_list.append(
                dict(
                    zip(
                        ["y", "x", "threshold"],
                        [prec[sample_idx], rec[sample_idx], threshold[sample_idx]],
                    )
                )
            )

        return points_list, auc_value

    def calculate_matthews_corr_coef(self, true, pred_categorical, target_class=None):
        if target_class == "overall" or target_class is None:
            return matthews_corrcoef(true, pred_categorical)

        elif target_class == "all":
            class_matthews = {}
            for target_class in self.class_order:
                true_idx = true == target_class
                class_true = true[true_idx]
                class_pred_categorical = pred_categorical[true_idx]
                class_matthews[target_class] = matthews_corrcoef(
                    class_true, class_pred_categorical
                )
            return class_matthews

        elif target_class is not None:
            true_idx = true == target_class
            class_true = true[true_idx]
            class_pred_categorical = pred_categorical[true_idx]
            return self._calculate_accuracy(class_true, class_pred_categorical)
        else:
            return self._calculate_accuracy(true, pred_categorical)

    def calculate_multiclass_precision_recall_statistics(self, true, pred_logits):
        """
        Calculates multiclass precision-recall curve statistics for precision-recall plots. May be renamed if it's suitable for binary cases.
        Reference : https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#sphx-glr-auto-examples-model-selection-plot-precision-recall-py
        :return:
        """
        y = true
        # Binarize the output
        y = label_binarize(y, classes=self.class_order)

        # Scikit Binarize doesn't make binary problem to one-hot
        if len(self.class_order) == 2:
            y_ = np.zeros((len(true), len(self.class_order)))
            y_[np.arange(y.shape[0]), y.flatten()] = 1
            y = y_

        n_classes = len(self.class_order)

        # Compute ROC curve and ROC area for each class
        y_score = pred_logits.T.values
        class_precs_macro = dict()
        class_recs_macro = dict()
        points = dict()
        aucs = dict()

        average_metrics = ["Micro", "Macro"]
        inputs = [*range(n_classes)] + average_metrics

        for class_idx in inputs:
            if class_idx not in average_metrics:
                prec, rec, threshold = precision_recall_curve(
                    y[:, class_idx], y_score[:, class_idx]
                )
                class_precs_macro[class_idx] = prec
                class_recs_macro[class_idx] = rec
            elif class_idx == "Micro":
                fpr, tpr, threshold = precision_recall_curve(y.ravel(), y_score.ravel())
            elif class_idx == "Macro":
                prec = np.unique(
                    np.concatenate([class_precs_macro[i] for i in range(n_classes)])
                )
                rec = np.zeros_like(prec)
                for i in range(n_classes):
                    rec += (
                        np.interp(prec, class_precs_macro[i], class_recs_macro[i])
                        / n_classes
                    )

            points_list, auc_value = self._create_precision_recall_points(
                prec=prec, rec=rec, threshold=threshold
            )

            metric_name = (
                self.class_mappings[str(class_idx)]
                if class_idx not in average_metrics
                else class_idx
            )

            points[metric_name] = points_list
            aucs[metric_name] = "nan" if np.isnan(auc_value) else auc_value

        return points, aucs

    def _create_roc_points(self, fpr, tpr, threshold):
        points_list = []
        for i in range(len(fpr) - len(threshold)):
            threshold = np.append(threshold, threshold[-1])

        for sample_idx in range(len(fpr)):
            points_list.append(
                dict(
                    zip(
                        ["fpr", "tpr", "threshold"],
                        [fpr[sample_idx], tpr[sample_idx], threshold[sample_idx]],
                    )
                )
            )
        return points_list

    def calculate_multiclass_roc_statistics(
        self, true, pred_logits, return_points=False, use_class_name=True
    ):
        """
        Calculates multiclass ROC curve statistics for ROC plots. May be renamed if it's suitable for binary cases.
        Reference : https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
        :return:
        """
        y = true
        # Binarize the output
        y = label_binarize(y, classes=self.class_order)

        # Scikit Binarize doesn't make binary problem to one-hot
        if len(self.class_order) == 2:
            y_ = np.zeros((len(true), len(self.class_order)))
            y_[np.arange(y.shape[0]), y.flatten()] = 1
            y = y_

        n_classes = len(self.class_order)

        # Compute ROC curve and ROC area for each class
        y_score = pred_logits.T.values
        class_fprs_macro = dict()
        class_tprs_macro = dict()
        points = dict()
        aucs = dict()

        average_metrics = ["Micro", "Macro"]
        inputs = [*range(n_classes)] + average_metrics

        for class_idx in inputs:
            if class_idx not in average_metrics:
                fpr, tpr, threshold = roc_curve(y[:, class_idx], y_score[:, class_idx])
                class_fprs_macro[class_idx] = fpr
                class_tprs_macro[class_idx] = tpr
            elif class_idx == "Micro":
                fpr, tpr, threshold = roc_curve(y.ravel(), y_score.ravel())
            elif class_idx == "Macro":
                non_nan_macro_classes = [
                    i
                    for i in class_tprs_macro
                    if not np.isnan(class_tprs_macro[i].sum())
                ]
                fpr = np.unique(
                    np.concatenate([class_fprs_macro[i] for i in non_nan_macro_classes])
                )
                tpr = np.zeros_like(fpr)
                for i in non_nan_macro_classes:
                    tpr += (
                        np.interp(fpr, class_fprs_macro[i], class_tprs_macro[i])
                        / n_classes
                    )

            if use_class_name:
                metric_name = (
                    self.class_mappings[str(class_idx)]
                    if class_idx not in average_metrics
                    else class_idx
                )
            else:
                metric_name = (
                    str(class_idx) if class_idx not in average_metrics else class_idx
                )
            # roc_curve(y[:, 0], y_score[:, 1])[0:2]
            try:
                auc_value = auc(fpr, tpr)
            except:
                aucs[metric_name] = "nan" if np.isnan(auc_value) else auc_value
            aucs[metric_name] = "nan" if np.isnan(auc_value) else auc_value

            if return_points:
                points_list = self._create_roc_points(
                    fpr=fpr, tpr=tpr, threshold=threshold
                )

                points[metric_name] = points_list

        if return_points:
            return points, aucs
        return aucs
