from pathlib import Path
from typing import List
import tqdm

import pandas as pd
import statsmodels.formula.api as smf


from src.utils.iv_regression import IVRegression
from src.utils.linear_regression import LinearRegression
from src.utils.named_regression import NamedRegression

CURRENT_FILE = Path(__file__).resolve()
OHIE_DATA_PATH = CURRENT_FILE.parent / 'data_files' / 'OHIE_data.csv'
# A list of the controls used in the linear regression in formula format:
# CONTROLS_FORMULA = 'ddddraw_survey_12m1:numhh_list1 + ddddraw_survey_12m2:numhh_list1 + ddddraw_survey_12m3:numhh_list1 + ddddraw_survey_12m4:numhh_list1 + ddddraw_survey_12m5:numhh_list1 + ddddraw_survey_12m6:numhh_list1 + ddddraw_survey_12m7:numhh_list1 + ddddraw_survey_12m1:numhh_list2 + ddddraw_survey_12m2:numhh_list2 + ddddraw_survey_12m3:numhh_list2 + ddddraw_survey_12m4:numhh_list2 + ddddraw_survey_12m5:numhh_list2 + ddddraw_survey_12m6:numhh_list2 + ddddraw_survey_12m7:numhh_list2 + ddddraw_survey_12m1:numhh_list3 + ddddraw_survey_12m2:numhh_list3 + ddddraw_survey_12m3:numhh_list3'.replace(":", "__")

def load_ohie_regressions(iv: bool = False) -> List[NamedRegression]:
    """
    Loads a list of the linear regressions in the OHIE (Oregon Health Insurance Enrollments) study.
    This study ran both OLS and IV (instrumental variables) regressions.
    :param iv: load the OLS or IV regressions (OLS by default)
    :return: A list of pairs of regression name and
    """
    data = pd.read_csv(OHIE_DATA_PATH, low_memory=False)
    data = data[data['sample_12m_resp'] == 1]
    data.columns = [col.replace(":", "__") for col in data.columns]
    instrument = 'treatment'  # Assuming this is the instrumental variable
    endogenous = "ohp_all_ever_survey"
    cluster = 'household_id'
    weight = "weight_12m"
    labels = [
        "health_genflip_bin_12m",
        "health_notpoor_12m",
        "health_chgflip_bin_12m",
        "notbaddays_tot_12m",
        "notbaddays_phys_12m",
        "notbaddays_ment_12m",
        "nodep_screen_12m"
    ]
    controls = [col for col in data.columns if col.startswith("ddd")]
    data = data[[instrument, endogenous, cluster, weight] + controls + labels].dropna(
        subset=[instrument, cluster, weight] + controls
    )
    data[cluster] = data[cluster] - data[cluster].min()



    res = []
    for label in tqdm.tqdm(labels):
        outcome_regression = LinearRegression(
            data=data.dropna(subset=[label]),
            formula=label + " ~ " + "+".join([instrument] + controls) + "-1",
            weight=weight,
            hc1_cluster=cluster
        )
        if iv:
            endogenous_regression = LinearRegression(
                data=data.dropna(subset=[endogenous]),
                formula=endogenous + " ~ " + "+".join([instrument] + controls) + "-1",
                weight=weight,
                hc1_cluster=cluster
            )
            res.append(NamedRegression(
                name=label,
                regression=IVRegression(outcome_regression=outcome_regression, endogenous_regression=endogenous_regression)
            ))
        else:
            res.append(NamedRegression(
                name=label,
                regression=outcome_regression
            ))
    return res
#
# class RegressionInputs(NamedTuple):
#     data: pd.DataFrame
#     label: str
#     features: List[str]
#     feature_of_interest: str
#     weights: Optional[np.ndarray] = None
#     categorical_features: Optional[List[str]] = None
#
#
#
# def extract_XY(inputs: RegressionInputs) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Extracts and prepares the matrix X and vector Y for weighted OLS regression,
#     ensuring the feature of interest is the first column in X.
#
#     Args:
#     - inputs: An instance of RegressionInputs containing regression data and metadata.
#
#     Returns:
#     A tuple (X, Y), where:
#     - X is the feature matrix with the feature of interest as the first column.
#     - Y is the target variable vector.
#     """
#     df = inputs.data.copy()
#
#     # Ensure the feature of interest is the first column
#     columns_order = [inputs.feature_of_interest] + [f for f in inputs.features if f != inputs.feature_of_interest]
#     X = df[columns_order].to_numpy()
#
#     # Apply weights if they exist
#     if inputs.weights is not None:
#         weights = np.sqrt(inputs.weights)  # Square root of weights for linear regression
#         X = X * weights[:, np.newaxis]  # Apply weights to each column of X
#
#     Y = df[inputs.label].to_numpy()
#
#     # Apply weights to Y if they exist
#     if inputs.weights is not None:
#         Y = Y * weights
#
#     return X, Y
#
#
# def load_ohie_data(verbose: bool = False):
#     """
#     Loads the data from OHIE study and prepares it for regression analysis.
#     This function replaces colons in column names with double underscores to avoid issues with statsmodels formula API.
#
#     Args:
#     - verbose: If True, print summary of each regression model.
#
#     Returns:
#     A list of RegressionInputs instances for different experiments in the OHIE study.
#     """
#     results = []
#     raw_data = pd.read_csv(CURRENT_FILE.parent / 'data_files' / 'OHIE_data.csv')
#
#     experiment_names = [
#         "health_genflip_bin_12m",
#         "health_notpoor_12m",
#         "health_chgflip_bin_12m",
#         "notbaddays_tot_12m",
#         "notbaddays_phys_12m",
#         "notbaddays_ment_12m",
#         "nodep_screen_12m"
#     ]
#
#     # Replace colons in column names with double underscores
#     raw_data.columns = [col.replace(":", "__") for col in raw_data.columns]
#
#     for i, experiment in enumerate(experiment_names, start=1):
#         if verbose:
#             print(f"Running for experiment {experiment}")
#
#         df = raw_data[raw_data['sample_12m_resp'] == 1].copy()
#         df['row'] = range(1, len(df) + 1)
#
#         # Select only relevant columns
#         cols = ['treatment', 'household_id'] + [col for col in df.columns if col.startswith('ddd')] + [experiment]
#         regression_df = df[cols].dropna()
#
#         # Handling weights
#         regression_df['weights'] = df.loc[regression_df.index, 'weight_12m']
#
#         # Re-index household_id for clustering
#         regression_df['household_id'] = regression_df['household_id'] - regression_df['household_id'].min()
#
#         # Allow for potential inclusion of intercept (prbly not really relevant)
#         regression_df['intercept'] = 1
#         features = ['treatment'] + [col for col in regression_df.columns if col.startswith('ddd')]
#
#         if verbose:
#             formula = f"{experiment} ~ {' + '.join(features)} - 1"
#             print(f"{formula=}")
#             print(f"{regression_df.shape=}")
#             try:
#                 # model = smf.wls(formula=formula, data=regression_df, weights=regression_df['weights']).fit()
#                 model = smf.wls(
#                     formula=formula, data=regression_df, weights=regression_df['weights']
#                 ).fit(
#                     cov_type='cluster', cov_kwds={'groups': regression_df['household_id']}
#                 )
#                 print(model.summary())
#             except Exception as e:
#                 print(f"Error running model for {experiment}: {e}")
#
#         # results.append(RegressionInputs(
#         #     data=regression_df, label=experiment, features=features,
#         #     feature_of_interest="treatment", weights=regression_df["weights"]
#         # ))
#
#     return results
#
#
# def load_ohie_data_iv(verbose: bool = False) -> List[Tuple[RegressionInputs, RegressionInputs]]:
#     """
#     Loads the data from the OHIE study and prepares it for regression analysis.
#     This function prepares input for both the first-stage and reduced-form regressions for instrumental variable analysis.
#
#     Args:
#     - verbose: If True, print summary of each regression model.
#
#     Returns:
#     A list of pairs of RegressionInputs instances. Each pair consists of inputs for the first-stage regression and the reduced-form regression, respectively.
#     """
#     results = []
#     raw_data = pd.read_csv(CURRENT_FILE.parent / 'data_files' / 'OHIE_data.csv', low_memory=False)
#     raw_data.columns = [col.replace(":", "__") for col in raw_data.columns]  # Replace colons in column names
#
#     table9_outcomes = [
#         "health_genflip_bin_12m",
#         "health_notpoor_12m",
#         "health_chgflip_bin_12m",
#         "notbaddays_tot_12m",
#         "notbaddays_phys_12m",
#         "notbaddays_ment_12m",
#         "nodep_screen_12m"
#     ]
#
#     df = raw_data[raw_data['sample_12m_resp'] == 1].copy()
#
#     instrument = 'treatment'  # Assuming this is the instrumental variable
#     endogenous = "ohp_all_ever_survey"
#     cluster = 'household_id'
#     controls = [col for col in df.columns if col.startswith('ddd')]
#     weight_column = 'weight_12m'
#
#     for outcome in table9_outcomes:
#         df['intercept'] = 1
#         df['row'] = range(1, len(df) + 1)
#
#         # Common columns for both regressions
#         cols = [endogenous, cluster, outcome, instrument, weight_column] + controls
#         regression_df = df[cols].dropna()
#         regression_df['household_id'] = regression_df['household_id'] - regression_df['household_id'].min()  # Re-index household_id
#
#         # First-stage Regression Inputs: Treatment regressed on Instrument and Controls
#         first_stage_features = [instrument] + controls
#         first_stage_inputs = RegressionInputs(
#             data=regression_df,
#             label=endogenous,
#             features=first_stage_features,
#             feature_of_interest=instrument,
#             weights=regression_df[weight_column]
#         )
#
#         # Reduced-form Regression Inputs: Outcome regressed on Instrument and Controls
#         reduced_form_features = [instrument] + controls
#         reduced_form_inputs = RegressionInputs(
#             data=regression_df,
#             label=outcome,
#             features=reduced_form_features,
#             feature_of_interest=instrument,
#             weights=regression_df[weight_column]
#         )
#
#         results.append((first_stage_inputs, reduced_form_inputs))
#
#         if verbose:
#             # First-stage regression
#             first_stage_formula = f"{endogenous} ~ {' + '.join(first_stage_features)} - 1"
#             first_stage_model = smf.wls(
#                 first_stage_formula, data=first_stage_inputs.data, weights=first_stage_inputs.weights
#             ).fit(
#                 cov_type='cluster', cov_kwds={'groups': regression_df['household_id']}
#             )
#             # print(f"First-stage regression for {outcome}:\n{first_stage_model.summary()}")
#             # first_stage_model:statsmodels.base.model.Model
#             # regression_df["endogenous_fit"] = first_stage_model.predict(first_stage_inputs.data)
#
#             # Reduced-form regression
#             # reduced_form_features.remove(instrument)
#             # reduced_form_features = ["endogenous_fit"] + reduced_form_features
#             reduced_form_formula = f"{outcome} ~ {' + '.join(reduced_form_features)} - 1"
#             reduced_form_model = smf.wls(
#                 reduced_form_formula, data=regression_df, weights=reduced_form_inputs.weights
#             ).fit(
#                 cov_type='cluster', cov_kwds={'groups': regression_df['household_id']}
#             )
#             # print(f"Reduced-form regression for {outcome}:\n{reduced_form_model.summary()}")
#
#             # Extracting coefficients and standard errors
#             fit1_coefficient, fit1_error = first_stage_model.params[instrument], first_stage_model.bse[instrument]
#             fit2_coefficient, fit2_error = reduced_form_model.params[instrument], reduced_form_model.bse[
#                 instrument]
#
#             # Calculating IV regression parameter and its error
#             iv_parameter = fit2_coefficient / fit1_coefficient
#             iv_error = abs(iv_parameter) * ((fit2_error / fit2_coefficient) ** 2 + (fit1_error / fit1_coefficient) ** 2) ** 0.5
#
#
#             # Formatting output
#             value_format = "{:.3f}".format(iv_parameter)
#             error_format = "{:.3f}".format(iv_error)
#             output = f"fit parameter = {value_format} +- ({error_format})"
#
#             print(output)
#
#     return results