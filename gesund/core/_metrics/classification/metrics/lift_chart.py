import numpy as np
import pandas as pd
from sklearn.metrics import auc
import sklearn


class LiftChart:
    def __init__(self, class_mappings):
        self.class_mappings = class_mappings
        self.class_order = [int(i) for i in list(class_mappings.keys())]

    def _decile_table(
        self,
        true,
        pred_logits,
        predicted_class=0,
        change_deciles=20,
        labels=True,
        round_decimal=3,
    ):
        """Generates the Decile Table from labels and probabilities

        The Decile Table is creared by first sorting the customers by their predicted
        probabilities, in decreasing order from highest (closest to one) to
        lowest (closest to zero). Splitting the customers into equally sized segments,
        we create groups containing the same numbers of customers, for example, 10 decile
        groups each containing 10% of the customer base.
        """
        df = pd.DataFrame([true, pred_logits.loc[predicted_class].T]).T
        df = df.rename(columns={predicted_class: "pred_logits"})

        # df['decile']=pd.qcut(df['pred_logits'], 10, labels=list(np.arange(10,0,-1)))
        # ValueError: Bin edges must be unique

        change_deciles = max(change_deciles, len(true))

        df.sort_values("pred_logits", ascending=False, inplace=True)
        df["decile"] = np.linspace(1, change_deciles + 1, len(df), False, dtype=int)

        # lift_df abbreviation for decile_table
        lift_df = (
            df.groupby("decile")
            .apply(
                lambda x: pd.Series(
                    [
                        np.min(x["pred_logits"]),
                        np.max(x["pred_logits"]),
                        np.mean(x["pred_logits"]),
                        np.size(x["pred_logits"]),
                        np.sum(x["true"]),
                        np.size(x["true"][x["true"] == 0]),
                    ],
                    index=(
                        [
                            "prob_min",
                            "prob_max",
                            "prob_avg",
                            "cnt_cust",
                            "cnt_resp",
                            "cnt_non_resp",
                        ]
                    ),
                )
            )
            .reset_index()
        )

        lift_df["prob_min"] = lift_df["prob_min"].round(round_decimal)
        lift_df["prob_max"] = lift_df["prob_max"].round(round_decimal)
        lift_df["prob_avg"] = round(lift_df["prob_avg"], round_decimal)

        tmp = df[["true"]].sort_values("true", ascending=False)
        tmp["decile"] = np.linspace(1, change_deciles + 1, len(tmp), False, dtype=int)

        lift_df["cnt_resp_rndm"] = np.sum(df["true"]) / change_deciles
        lift_df["cnt_resp_wiz"] = tmp.groupby("decile", as_index=False)["true"].sum()[
            "true"
        ]

        lift_df["resp_rate"] = round(
            lift_df["cnt_resp"] * 100 / lift_df["cnt_cust"], round_decimal
        )
        lift_df["cum_cust"] = np.cumsum(lift_df["cnt_cust"])
        lift_df["cum_resp"] = np.cumsum(lift_df["cnt_resp"])
        lift_df["cum_resp_wiz"] = np.cumsum(lift_df["cnt_resp_wiz"])
        lift_df["cum_non_resp"] = np.cumsum(lift_df["cnt_non_resp"])
        lift_df["cum_cust_pct"] = round(
            lift_df["cum_cust"] * 100 / np.sum(lift_df["cnt_cust"]), round_decimal
        )
        lift_df["cum_resp_pct"] = round(
            lift_df["cum_resp"] * 100 / np.sum(lift_df["cnt_resp"]), round_decimal
        )
        lift_df["cum_resp_pct_wiz"] = round(
            lift_df["cum_resp_wiz"] * 100 / np.sum(lift_df["cnt_resp_wiz"]),
            round_decimal,
        )
        lift_df["cum_non_resp_pct"] = round(
            lift_df["cum_non_resp"] * 100 / np.sum(lift_df["cnt_non_resp"]),
            round_decimal,
        )
        lift_df["KS"] = round(
            lift_df["cum_resp_pct"] - lift_df["cum_non_resp_pct"], round_decimal
        )
        lift_df["lift"] = round(
            lift_df["cum_resp_pct"] / lift_df["cum_cust_pct"], round_decimal
        )
        return lift_df

    def calculate_lift_curve_points(
        self,
        true,
        pred_logits,
        predicted_class=None,
        change_deciles=20,
        labels=True,
        round_decimal=3,
    ):

        class_lift_dict = dict()
        if predicted_class in [None, "all", "overall"]:
            for class_ in list(self.class_mappings.keys()):
                lift_df = self._decile_table(
                    true,
                    pred_logits,
                    predicted_class=int(class_),
                    change_deciles=20,
                    labels=True,
                    round_decimal=3,
                )
                xy_points = [
                    {"x": i[0], "y": i[1]}
                    for i in lift_df[["prob_avg", "lift"]].values.tolist()
                ]
                class_lift_dict[self.class_mappings[str(class_)]] = xy_points
        else:
            lift_df = self._decile_table(
                true,
                pred_logits,
                predicted_class=predicted_class,
                change_deciles=20,
                labels=True,
                round_decimal=3,
            )
            xy_points = [
                {"x": i[0], "y": i[1]}
                for i in lift_df[["prob_avg", "lift"]].values.tolist()
            ]
            class_lift_dict[self.class_mappings[str(predicted_class)]] = xy_points

        return class_lift_dict
