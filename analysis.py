import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from business_eda_helpers import (
    column_pie_chart,
    funnel_plot,
    confidence_ellipse,
    FunnelType,
    PlotStyle,
    bar_plot,
    box_plot,
)

from typing import Union, List
from enum import Enum

from os.path import join

# this is just used in sanity checks
SAMPLE_NUM = 3
FIG_SAVE_PATH = join(os.getcwd(), "Figures")

LOGGING = False

# setting plot style
sns.set_style("darkgrid")
plt.rc("axes", labelsize=14)  # fontsize of the x and y labels
plt.rc("font", size=12)  # controls default text sizes
COLOR_PALETTE = np.array(sns.color_palette(palette="Accent"))
COLOR_PALETTE[:5, :] = COLOR_PALETTE[[4, 6, 2, 1, 0], :]

user_stages = [
    "signup_dt",
    "first_application_start_ts",
    "first_application_complete_ts",
    "first_awaiting_payment_ts",
    "first_paystub_dt",
    "first_repayment_ts",
]
user_namemap = {
    "signup_dt": "User Created",
    "first_application_start_ts": "Started 1st App.",
    "first_application_complete_ts": "Completed 1st App.",
    "first_awaiting_payment_ts": "Awaiting Payment",
    "first_paystub_dt": "Submitted Paystub",
    "first_repayment_ts": "Repayment Started",
}

loan_stages = [
    "application_start_ts",
    "application_complete_ts",
    "awaiting_payment_ts",
    "repayment_ts",
]

loan_namemap = {
    "application_start_ts": "App. Started",
    "application_complete_ts": "App. Completed",
    "awaiting_payment_ts": "Awaiting Payment",
    "repayment_ts": "Repayment Started",
}

loans_df = pd.read_csv(join(os.getcwd(), "loan_dataset.csv"), index_col=0)

new_col = np.full((len(loans_df),), len(loan_stages))
settled = np.zeros(len(loans_df), dtype=bool)
for stage in reversed(loan_stages):
    settled |= loans_df[stage].notnull()
    # for the unsettled ones, we subtract one
    new_col -= (~settled).astype(np.int8)
loans_df["max_stage"] = new_col

user_df = pd.read_csv(join(os.getcwd(), "user_dataset.csv"), index_col=0)
joint_df = pd.merge(user_df, loans_df, left_index=True, right_on="user_id", how="inner")
joint_df["credit_ratio"] = joint_df["amount"] / joint_df["spending_limit_est"]

print(len(joint_df))
print(len(loans_df))


def basic_print(text, fun):
    print(f"\n{text}\n")
    print(fun(loans_df))
    print()
    print(fun(user_df))


######################### For my understanding #################################
SAMPLE_NUM = 3
if LOGGING:
    print("FOR FOLLOWING, ORDER IS LOANS_DF THEN USERS_DF\n\n")
    basic_print("Dataframe Fields:", lambda df: df.columns)
    basic_print("Descriptions:", lambda df: df.describe(include="all"))
    basic_print("Unique column values:", lambda df: df.nunique())
    print("\nCombined dataset result:", joint_df[:SAMPLE_NUM])
    print(loans_df["approval_type"].unique())
    print(loans_df["cancellation_type"].unique())


################################ EDA (Q1) ######################################

# proportion of loans vs value. Distribution shapes tell something about how likely people are to complete loan vs value
completed_loan_df = loans_df[loans_df["repayment_ts"].notnull()]
cancelled_loan_df = loans_df[loans_df["cancellation_type"].notnull()]

max_round_val = 250
bins = np.linspace(
    0, math.ceil(max(loans_df["amount"]) / max_round_val) * max_round_val, 20
)


def get_prob_mass(data, bins):
    heights, _ = np.histogram(data, bins)
    return heights / sum(heights)


# TODO: refactor to use bar plot helper
completed_props, cancelled_props = [
    get_prob_mass(df["amount"], bins) for df in (completed_loan_df, cancelled_loan_df)
]

bins = bins[:-1]
bin_width = bins[1] - bins[0]
bar_scale = 0.4
bar_offset = (bar_scale / 2) * bin_width
bar_width = bar_scale * bin_width

palette = iter(COLOR_PALETTE)
plt.bar(
    bins - bar_offset,
    completed_props,
    width=bar_width,
    align="center",
    color=next(palette),
    label="Completed loans",
)
plt.bar(
    bins + bar_offset,
    cancelled_props,
    width=bar_width,
    align="center",
    color=next(palette),
    label="Cancelled loans",
)
plt.title("Loan Value Distributions")
plt.xlabel("Loan Amount")
plt.ylabel("Proportion")
plt.legend(loc="upper right")
plt.savefig(join(FIG_SAVE_PATH, "Loan Value Distributions.png"), bbox_inches="tight")
plt.close()

# fig, ax = plt.subplots(1, 1)
# box_plot(
#     ax,
#     [completed_loan_df["amount"], cancelled_loan_df["amount"]],
#     labels=["Completed loans", "Cancelled loans"],
#     palette=iter(COLOR_PALETTE),
#     orientation="hor",
# )

# print([completed_loan_df["amount"], cancelled_loan_df["amount"]])
# print(["Completed loans", "Cancelled loans"])

# ax.set_title("Loan Value for cancelled and non-cancelled loans")
# ax.set_xlabel("Loan Value")
# ax.set_xlim(left=0)
# fig.savefig(join(FIG_SAVE_PATH, "Loan Value Boxplot.png"), bbox_inches="tight")

# PROFILE THE USERS DATASET

# User Dataset EDA
# TODO: proportion of people who get through user phases (e.g., starting out, added company, etc.)

column_pie_chart(
    user_df,
    "valid_phone_ind",
    "User Phone Status",
    COLOR_PALETTE[[1, 0]],
    join(FIG_SAVE_PATH, "EDA", "Phone Status Pie Chart"),
    namemap={0: "Invalid Number", 1: "Valid Number"},
    aggregation_kwargs={"max_size": 10},
)

column_pie_chart(
    user_df,
    "was_referred_ind",
    "User Referral Status",
    COLOR_PALETTE,
    join(FIG_SAVE_PATH, "EDA", "Referral Status Pie Chart"),
    namemap={0: "Not Referred", 1: "Referred"},
    aggregation_kwargs={"max_size": 10},
)

column_pie_chart(
    user_df,
    "company_name",
    "User Company",
    COLOR_PALETTE,
    join(FIG_SAVE_PATH, "EDA", "User Company Pie Chart"),
    pop_other=False,
    aggregation_kwargs={"other_cutoff": 0.007},
    has_legend=True,
)

# Joint EDA

# loan stage vs credit ratio
fig, ax = plt.subplots(1, 1)
this_data = [
    joint_df[(joint_df["max_stage"] == i) & joint_df["credit_ratio"].notnull()][
        "credit_ratio"
    ]
    for i in range(1, len(loan_stages) + 1)
]

box_plot(
    ax,
    this_data,
    labels=[loan_namemap[stage] for stage in loan_stages],
    palette=iter(COLOR_PALETTE),
    orientation="hor",
)
ax.set_title("Credit to Limit Ratio vs Terminal Stage")
ax.set_xlabel("Loan Value as proportion of Credit Limit")
ax.set_xlim(left=0, right=3)
ax.figure.savefig(
    join(FIG_SAVE_PATH, "Loan Credit Ratio vs max step.png"), bbox_inches="tight"
)


# TODO: plot loans "max stage" vs user credit limit
fig, ax = plt.subplots(1, 1)
ax.scatter(joint_df["amount"], joint_df["max_stage"], s=0.5)
confidence_ellipse(joint_df["amount"], joint_df["max_stage"], ax, n_std=2)
ax.set_title("Amount vs. Stage Attained")
ax.set_xlabel("Loan Amount")
ax.set_ylabel("Loan Stages Passed")
ax.figure.savefig(
    join(FIG_SAVE_PATH, "EDA", "Amount vs Stage scatter"), bbox_inches="tight"
)

################################ FUNNEL (Q2) ###################################

# TODO (big): consider using metrics that are expected value of loans ($$$ matter more than probabilities)

# the most basic metrics for seeing where the funnel works and where it doesn't
# is to look at how far customers get through the funnel

# another way of looking at it is looking at yield from stage to stage

# definitely look at what customer demographic is bringing in the most revenue, and focus on them as well as focusing on the pain point for "second tier" customers

# first, we'll only select customers for whom we have loan data (see issues in 3)

# more sophisticated approach might incorporate the dates of the events into the funnel (e.g., perhaps we're interested in how the probability of proceeding is affected by the time between steps)


funnel_plot(
    (
        user_df[user_df["was_referred_ind"] == 0],
        user_df[user_df["was_referred_ind"] == 1],
    ),
    user_stages,
    user_namemap,
    "User",
    join(
        FIG_SAVE_PATH,
        "funnels",
        f"referral user permeability scatter.png",
    ),
    labels=["Nonreferred Users", "Referred Users"],
    funnel_type=FunnelType.permeability,
    plot_style=PlotStyle.scatter,
)

funnel_plot(
    (
        user_df[user_df["was_referred_ind"] == 0],
        user_df[user_df["was_referred_ind"] == 1],
    ),
    user_stages,
    user_namemap,
    "User",
    join(
        FIG_SAVE_PATH,
        "funnels",
        f"referral user proportion bar.png",
    ),
    labels=["Nonreferred Users", "Referred Users"],
    funnel_type=FunnelType.proportions,
    plot_style=PlotStyle.bar,
)

# print("Summary statistics for Loan value of nonreferred users")
print(
    joint_df[joint_df["was_referred_ind"] == 0 & joint_df["amount"].notnull()][
        "amount"
    ].describe()
)

# print("Summary statistics for Loan value of referred users")
print(
    joint_df[joint_df["was_referred_ind"] == 1 & joint_df["amount"].notnull()][
        ["amount"]
    ].describe()
)

fig, ax = plt.subplots()
box_plot(
    ax,
    [
        joint_df[joint_df["was_referred_ind"] == 0 & joint_df["amount"].notnull()][
            "amount"
        ],
        joint_df[joint_df["was_referred_ind"] == 1 & joint_df["amount"].notnull()][
            "amount"
        ],
    ],
    iter(COLOR_PALETTE),
    labels=["Nonreferred User", "Referred User"],
    orientation='hor'
)
ax.set_xlabel("Loan Value")
ax.set_title("Referral Loan Values")
ax.figure.savefig(join(FIG_SAVE_PATH, "referral value boxplot.png"), bbox_inches='tight')

funnel_plot(
    (
        loans_df[loans_df["user_pinwheel_eligible_at_ap"] == 0],
        loans_df[loans_df["user_pinwheel_eligible_at_ap"] == 1],
    ),
    loan_stages,
    loan_namemap,
    "Loan",
    join(
        FIG_SAVE_PATH,
        "funnels",
        f"pinwheel loan permeability scatter.png",
    ),
    labels=["Pinwheel Unavailable", "Pinwheel Available"],
    funnel_type=FunnelType.permeability,
    plot_style=PlotStyle.scatter,
)

funnel_plot(
    (
        loans_df[loans_df["user_pinwheel_eligible_at_ap"] == 0],
        loans_df[loans_df["user_pinwheel_eligible_at_ap"] == 1],
    ),
    loan_stages,
    loan_namemap,
    "Loan",
    join(
        FIG_SAVE_PATH,
        "funnels",
        f"pinwheel loan proportion bar.png",
    ),
    labels=["Pinwheel Unavailable", "Pinwheel Available"],
    funnel_type=FunnelType.proportions,
    plot_style=PlotStyle.bar,
)

funnel_plot(
    user_df,
    user_stages,
    user_namemap,
    "User",
    join(
        FIG_SAVE_PATH,
        "funnels",
        f"user totals bar.png",
    ),
    labels=["User"],
    funnel_type=FunnelType.totals,
    plot_style=PlotStyle.bar,
)

funnel_plot(
    loans_df,
    loan_stages,
    loan_namemap,
    "Loan",
    join(
        FIG_SAVE_PATH,
        "funnels",
        f"loan totals bar.png",
    ),
    labels=["Loans"],
    funnel_type=FunnelType.totals,
    plot_style=PlotStyle.bar,
)

funnel_plot(
    user_df,
    user_stages,
    user_namemap,
    "User",
    join(FIG_SAVE_PATH, "funnels", "user permeability scatter.png"),
    labels=["User"],
    funnel_type=FunnelType.permeability,
    plot_style=PlotStyle.scatter,
)

funnel_plot(
    loans_df,
    loan_stages,
    loan_namemap,
    "Loan",
    join(FIG_SAVE_PATH, "funnels", "loans permeability scatter.png"),
    labels=["Loans"],
    funnel_type=FunnelType.permeability,
    plot_style=PlotStyle.scatter,
)


# TODO: investigate covariance between time taken and probability of passing through to the next step. Focus on low permeability steps.
# TODO: See how pinwheel impacts time distributions; consider if this can be an experiment on the previous todo. Check out how many purchases pinwheel users make compared to regular users (also splitting across whether they have made a purchase or not. I suspect it will have the greatest impact for the first purchase).

# impact of pinwheel and time differences
# TODO: Why does pinwheel affect multiple steps?
# TODO: could still check if pinwheel makes it more likely that multiple loans are in dataset per user

# NOTE: it might be better to do processing initially and just convert the actual loans_df strings to datetimes

time_data = pd.DataFrame(index=loans_df.index)
for stage in loan_stages:
    time_data[stage] = pd.to_datetime(loans_df[stage], format=r"%Y-%m-%d %H:%M:%S.%f")

for i in range(1, len(loan_stages)):
    prev_stage, curr_stage = loan_stages[i - 1], loan_stages[i]
    time_data[f"Time Delta to {curr_stage}"] = (
        time_data[curr_stage] - time_data[prev_stage]
    ).dt.total_seconds()

# TODO: apply techniques to explore the response of certain metrics: ANCOVA
# Idea: use relationship of time between steps and probability of progression
# to motivate suggestion for further study (e.g., causality of time taken/friction and likelihood to proceed)

# TODO: also consider the impact of loan value on each step
# TODO: check the impact of loan value on cancellation reason

################################# ISSUES (Q3) #################################


def quality_check(comparison, check_message, extra_information, blocking=False):
    """Helper for simple test cases"""
    if blocking:
        assert comparison, "FAILED CHECK: " + check_message
    else:
        try:
            assert comparison, "FAILED CHECK: " + check_message
        except AssertionError as e:
            print(e)
            print(extra_information)


# check that every user id in the loans dataset is also in the user dataset
reified_user_ids = set(loans_df["user_id"])
user_ids = set(user_df.index)
quality_check(
    reified_user_ids <= user_ids,
    "Every user should be identified in the user_df.",
    f"Out of {len(reified_user_ids)} user ids in the loans table, {len(reified_user_ids - user_ids)} were not represented in the user table",
)

# Check if any loans were canceled after repayment was initiated, or repayment was initiated on a canceled loan
entered_repayment = loans_df[loans_df["repayment_ts"].notnull()]
quality_check(
    np.all(entered_repayment["cancellation_type"].isnull()),
    "There should be no situation where repayment was started on a cancelled loan. Since items are shipped on first payment, an item was stolen in this case.",
    entered_repayment[entered_repayment["cancellation_type"].notnull()][:SAMPLE_NUM],
)


# Check that no loans "skip steps"
curr_df = loans_df
curr_index = set(curr_df.index)
indices = [set(loans_df[loans_df[stage].notnull()].index) for stage in loan_stages]
checks = [indices[i - 1] >= indices[i] for i in range(1, len(loan_stages))]
quality_check(
    all(checks),
    "A loan should only pass onto the next stage if it has passed the previous stages",
    f"Failures occured moving into stages {[loan_stages[i + 1] for i, check in enumerate(checks) if check]}",
)

################################### Q4 #########################################

time_data["Time Delta for Q4"] = (
    time_data["repayment_ts"] - time_data["awaiting_payment_ts"]
).dt.total_seconds()

# if there is a repayment_ts, then there must be the previous
q4_data = time_data[time_data["Time Delta for Q4"].notnull()]["Time Delta for Q4"]

fig, ax = plt.subplots()
counts, bins = np.histogram(q4_data, bins=20)

seconds_in_day = 24 * 60 * 60
# TODO: refactor bar_plot to not expect a list of ys when it is not plotting multiple
bar_plot(
    ax,
    bins[1:] / seconds_in_day,
    [counts],
    plot_multiple=False,
    palette=iter(COLOR_PALETTE),
)
ax.set_xlabel("Time from Approval to Repayment (days)")
ax.set_ylabel("Loan Counts")
ax.figure.savefig(join(FIG_SAVE_PATH, "Repayment Timescale.png"), bbox_inches="tight")

# how many loans enter repayment within 15 days of approval? for now I'll assume that I've already done the check above
# TODO: should the difference be awaiting payment??
print(
    f"Out of {len(q4_data)} loans that entered repayment, {len(q4_data[q4_data < 15 * seconds_in_day])} were entered repayment within 15 days"
)
################################### Q5 #########################################

# Perhaps discuss notes I took while signing up, and just ideas about the rest of the pipeline that we're not seeing
# TODO: would be nice to do a very coarse study of metrics for repeat buyers, even though dataset is restricted to ppls first loan app
# Propose further studies with the full dataset

# xTODO: Create a database of only those loans whose user_id exists in the other database
# xTODO: loan funnel too. If user funnel looks good but users don't make many loans, then the concern is on the loan funnel.
# If not many users make it through first funnel, but make lots of purchases if they do, then that's a good sign

# nahh, not using cross-dataset information in any meaningful way
