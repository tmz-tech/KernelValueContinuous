#!/usr/bin/env python
# coding: utf-8

# In[14]:

import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import wilcoxon
from scipy.stats.mstats import gmean
from scipy.special import expit
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import GridSearchCV, LeaveOneOut
import io
import ast
import os
import json

from . import np, pd
from .nonparametric import *
from .basis import *


# In[10]:


class scaling:
    """
    A class used to scale data to the range [-1, 1] and inverse scale it back to the original range.
    
    Methods
    -------
    scale_to_minus_one_to_one(column)
        Scales a single column to the range [-1, 1].

    scale_to_minus_one_to_one_df(df)
        Scales all columns in a DataFrame to the range [-1, 1].

    inverse_scale(column, min_val, max_val)
        Inverse scales a single column back to its original range.

    inverse_scale_df(scaled_df)
        Inverse scales all columns in a scaled DataFrame back to their original ranges.
    """
    def __init__(self):
        """
        Initialize the Scaling class with attributes to store scaled data and min-max values.
        """
        # No initialization of variables is strictly needed here, but we define the __init__ method for potential future extensions.
        return None
    
    def scale_to_minus_one_to_one(self, column):
        """
        Scales a single column to the range [-1, 1] using min-max normalization.
        
        Parameters
        ----------
        column : pandas Series
            The column to scale.
        
        Returns
        -------
        scaled_column : pandas Series
            The scaled column with values in the range [-1, 1].
        min_val : float
            The minimum value of the original column.
        max_val : float
            The maximum value of the original column.
        """
        # Find the minimum and maximum values in the column
        min_val = column.min()
        max_val = column.max()
        # Scale the column to the range [-1, 1]
        scaled_column = 2 * (column - min_val) / (max_val - min_val) - 1
        # Return the scaled column along with its original min and max values
        return scaled_column, min_val, max_val
    
    def scale_to_minus_one_to_one_df(self, df):
        """
        Scales all columns in a DataFrame to the range [-1, 1] using min-max normalization.
        
        Parameters
        ----------
        df : pandas DataFrame
            The DataFrame to scale.
        
        Returns
        -------
        self.scaled_df : pandas DataFrame
            A new DataFrame with all columns scaled to the range [-1, 1].
        """
        # Initialize an empty DataFrame to store the scaled data
        self.scaled_df = pd.DataFrame()
        # Initialize a dictionary to store the min and max values for each column
        self.min_max_values = {}
        
        # Iterate through each column in the DataFrame
        for column in df.columns:
            # Scale the column and retrieve its min and max values     
            scaled_column, min_val, max_val = self.scale_to_minus_one_to_one(df[column])
            # Add the scaled column to the new DataFrame
            self.scaled_df[column] = scaled_column
            # Store the min and max values for this column
            self.min_max_values[column] = (min_val, max_val)
            
        # Return the DataFrame with all columns scaled to [-1, 1]   
        return self.scaled_df
    
    def inverse_scale(self, column, min_val, max_val):
        """
        Inverse scales a single column from the range [-1, 1] back to its original range.
        
        Parameters
        ----------
        column : pandas Series
            The scaled column to inverse scale.
        min_val : float
            The minimum value of the original column before scaling.
        max_val : float
            The maximum value of the original column before scaling.
        
        Returns
        -------
        original_column : pandas Series
            The column rescaled back to its original range.
        """
        # Reverse the scaling process to return the column to its original range
        original_column = (column + 1) / 2 * (max_val - min_val) + min_val
        # Return the rescaled column
        return original_column
    
    def inverse_scale_df(self, scaled_df):
        """
        Inverse scales all columns in a DataFrame from the range [-1, 1] back to their original ranges.
        
        Parameters
        ----------
        scaled_df : pandas DataFrame
            The DataFrame with scaled columns to inverse scale.
        
        Returns
        -------
        self.original_df : pandas DataFrame
            A new DataFrame with all columns rescaled back to their original ranges.
        """
        # Initialize an empty DataFrame to store the original data
        self.original_df = pd.DataFrame()
        
        # Iterate through each column in the scaled DataFrame
        for column in scaled_df.columns:
            # Retrieve the original min and max values for this column
            min_val, max_val = self.min_max_values[column]
            # Inverse scale the column and add it to the new DataFrame
            self.original_df[column] = self.inverse_scale(scaled_df[column], min_val, max_val)
            
        # Return the DataFrame with all columns rescaled to their original ranges
        return self.original_df


# In[11]:


class scalingDataFrame:
    """
    A class used to scale a DataFrame's entire range to [-1, 1] and inverse scale it back to the original range.
    
    Methods
    -------
    scale_to_minus_one_to_one_df(df)
        Scales all values in a DataFrame to the range [-1, 1].
    
    inverse_scale_df(scaled_df)
        Inverse scales a scaled DataFrame back to its original range.
    """
    
    def __init__(self):
        """
        Initialize the Scaling class with attributes to store the original min and max values for the entire DataFrame.
        """
        # These will store the overall min and max values for the entire DataFrame
        self.min_val = None
        self.max_val = None
    
    def scale_to_minus_one_to_one_df(self, df):
        """
        Scales the entire DataFrame to the range [-1, 1] based on the global min and max of all values in the DataFrame.
        
        Parameters
        ----------
        df : pandas DataFrame
            The DataFrame to scale.
        
        Returns
        -------
        scaled_df : pandas DataFrame
            A new DataFrame with all values scaled to the range [-1, 1].
        """
        # Find the overall min and max values for the entire DataFrame
        self.min_val = df.min().min()
        self.max_val = df.max().max()
        
        # Scale the entire DataFrame to the range [-1, 1]
        scaled_df = 2 * (df - self.min_val) / (self.max_val - self.min_val) - 1
        
        # Return the scaled DataFrame
        return scaled_df
    
    def inverse_scale_df(self, scaled_df):
        """
        Inverse scales a DataFrame from the range [-1, 1] back to its original range.
        
        Parameters
        ----------
        scaled_df : pandas DataFrame
            The DataFrame with scaled values to inverse scale.
        
        Returns
        -------
        original_df : pandas DataFrame
            A new DataFrame with all values rescaled back to their original range.
        """
        # Reverse the scaling process to return the DataFrame to its original range
        original_df = (scaled_df + 1) / 2 * (self.max_val - self.min_val) + self.min_val
        
        # Return the rescaled DataFrame
        return original_df


# In[12]:



class CalculateValues:
    """
    Class to calculate theta values and value functions using policy estimation, reward estimation, and basis function expansion.
    
    Parameters
    ----------
    state_df : DataFrame
        DataFrame containing the state values.
    action_df : DataFrame
        DataFrame containing the action values.
    reward_df : DataFrame
        DataFrame containing the reward values.
    next_x_df : DataFrame
        DataFrame containing the next state values.
    """

    def __init__(self, state_df, action_df, reward_df, next_x_df):
        # Initialize dataframes for state, action, reward, and next state
        self.state_df = state_df
        self.action_df = action_df
        self.reward_df = reward_df
        self.next_x_df = next_x_df
        
        # Scale the state and next state dataframes to the range [-1, 1]
        
        state_next_state_df = pd.concat([self.state_df, self.next_x_df], axis=1)
        self.sc = scalingDataFrame()#scaling()
        self.scaled_state_next_state_df = self.sc.scale_to_minus_one_to_one_df(state_next_state_df)
        
        self.scaled_state_df = self.scaled_state_next_state_df.iloc[:,:-1]
        self.scaled_next_x_df = pd.DataFrame(self.scaled_state_next_state_df.iloc[:,-1])
        
        #self.sc_state = scaling()
        #self.scaled_state_df = self.sc_state.scale_to_minus_one_to_one_df(self.state_df)
        
        #self.sc_next_x = scaling()
        #self.scaled_next_x_df = self.sc_next_x.scale_to_minus_one_to_one_df(self.next_x_df)
        
        self.sc_reward = scaling()
        self.scaled_reward_df = self.sc_reward.scale_to_minus_one_to_one_df(self.reward_df)
        
        # Calculate the maximum lag using a rule of thumb based on the sample size
        self.max_lag = int(4*((self.state_df.shape[0]/100)** (1/3)))
        
        
    def fit_state(self, order: int = 3, cv_x=True, search_interval_x=np.linspace(0.05, 0.06, 100), cv_k_x: int = 20):
        """
        Fit the model by searching for the optimal bandwidth for the state data and computing policy, reward, and basis functions.

        Parameters
        ----------
        order : int, optional
            The order of the Chebyshev basis function expansion, by default 3.
        cv_x : bool, optional
            If True, perform cross-validation to find the optimal bandwidth for kernel density estimation (KDE), by default True.
        search_interval_x : array-like, optional
            Interval to search for the optimal bandwidth for KDE, by default np.linspace(0.05, 0.06, 100).
        cv_k_x : int, optional
            Number of folds for cross-validation when searching for the optimal bandwidth, by default 20.

        Returns
        -------
        int
            Always returns 0 after fitting.
        """

        # Instantiate the Chebyshev basis object with the specified order
        self.c_basis = ChebyshevBasis(order)  
        
        # Search for the optimal bandwidth for kernel density estimation
        if cv_x:
            # Perform k-fold cross-validation to find the optimal bandwidth for KDE on the state data
            self.h_x = self.search_optimal_bandwidth_x(self.scaled_state_df, search_interval_x, cv_k_x)
        else:
            # Use Silverman's Rule of Thumb to estimate the bandwidth for KDE
            n, d = self.scaled_state_df.shape
            std_devs = self.scaled_state_df.std().values
            if d==1:
                self.h_x = 1.06 * std_devs * n**(-1/5) # Rule of thumb for univariate data
            else:
                bandwidths = (4 / ((d + 2) * n)) ** (1 / (d + 4)) * std_devs
                self.h_x = gmean(bandwidths) # Geometric mean of the estimated bandwidthss
            print(f"Bandwidth for state KDE: {self.h_x}")
            
        # Set the regularization parameter based on sample size to control overfitting
        if self.scaled_state_df.shape[0]>1e+3:
            p = self.scaled_state_df.shape[1]
            T = self.scaled_state_df.shape[0]
            self.alpha = self.alpha = 100*T**(-(2*p+7)/(2*p+2*4)) * self.h_x**(-1) #self.scaled_state_df.shape[0] * 5e-5 #1e-4
        else:
            self.alpha =0
        
        # Compute the basis functions
        self.basis_df, self.basis_dict, self.hat_psi_next = self.compute_basis(self.h_x, self.alpha, self.scaled_state_df, self.scaled_next_x_df, self.c_basis)
        
        
        # Estimate the reward
        self.r_pi_est = self.estimation_reward(self.h_x, self.alpha, self.scaled_state_df, self.reward_df)
        
    
    
    def fit_action(self, cv_a=True, search_interval_a=np.linspace(0.05, 0.06, 100), cv_k_a: int = 20):
        """
        Fit the model by searching for the optimal bandwidth for the action data and estimating the reward and basis functions.

        Parameters
        ----------
        cv_a : bool, optional
            If True, perform cross-validation to find the optimal bandwidth for kernel density estimation (KDE), by default True.
        search_interval_a : array-like, optional
            Interval to search for the optimal bandwidth for KDE, by default np.linspace(0.05, 0.06, 100).
        cv_k_a : int, optional
            Number of folds for cross-validation when searching for the optimal bandwidth, by default 20.

        Returns
        -------
        int
            Always returns 0 after fitting.
        """

        # Search for the optimal bandwidth for kernel density estimation on the action data
        if cv_a:
            # Perform k-fold cross-validation to find the optimal bandwidth for KDE on the action data
            self.h_a = self.search_optimal_bandwidth_a(self.scaled_state_df, self.action_df, self.h_x, search_interval_a, cv_k_a)
        else:
            # Use Silverman's Rule of Thumb to estimate the bandwidth for KDE on the action data
            n, d = self.action_df.shape
            std_devs = self.action_df.std().values
            if d==1:
                self.h_a = 1.06 * std_devs * n**(-1/5) # Rule of thumb for univariate data
            else:
                bandwidths = (4 / ((d + 2) * n)) ** (1 / (d + 4)) * std_devs 
                self.h_a = gmean(bandwidths) # Geometric mean of the estimated bandwidths

            print(f"Bandwidth for action KDE: {self.h_a}")
            
            
        # Estimate reward and next basis function data for the actions taken
        self.est_r_sa_actual, self.est_next_psi_SA_actual = self.estimation_r_SA_next_psi_SA(self.h_x, self.h_a, self.alpha, self.scaled_state_df, self.action_df, self.scaled_next_x_df, self.reward_df, self.c_basis)
    
    
    def calculate_VQ_w(self, w: int = 10, gamma: float = 0.7):
        """
        Calculate value function V and Q-values for a given time lag `w` and discount factor `gamma`.

        Parameters
        ----------
        w : int, optional
            The time lag used in the reward estimation. Default is 10.
        gamma : float, optional
            The discount factor for future rewards. Default is 0.7.

        Returns
        -------
        original_results_w : dict
            Dictionary containing results from theta estimation.
        df : DataFrame
            DataFrame of the formatted results of theta estimation.
        latex_table : str
            The LaTeX formatted table of the results.
        est_V : ndarray, shape (n_samples, 1)
            Estimated value function V.
        est_Q_actual : ndarray, shape (n_samples, 1)
            Estimated Q-values for the actual actions taken.
        """
        
        # Step 1: Calculate theta_w and retrieve results, estimated policies, and reward data
        self.original_results_w, df, latex_table, self.est_r_pi_sa_w_actual = self.calculate_theta_w(self.scaled_state_df, self.action_df, self.scaled_next_x_df, self.reward_df, w, gamma)
        
        # Extract the coefficient estimates (theta) from the results
        theta_hat = self.original_results_w['coef_estimate']
        
        # Step 2: Compute the estimated value function V using the basis functions and theta estimates
        est_V_hat = self.basis_df.values @ theta_hat.reshape(-1,1)
        est_V_tilde = self.r_pi_est[:, np.newaxis]+self.hat_psi_next @ theta_hat.reshape(-1,1) * gamma - self.r_pi_w_est[:, np.newaxis]*(gamma**w)
        
        # Step 3: Calculate Q-values for the actual actions taken
        # Using the reward data and next-state basis function estimates for the actual actions
        est_Q_actual = self.est_r_sa_actual[:, np.newaxis]+self.est_next_psi_SA_actual @ theta_hat.reshape(-1,1) * gamma - self.est_r_pi_sa_w_actual[:, np.newaxis]*(gamma**w)
           
        return self.original_results_w, df, latex_table, est_V_hat, est_V_tilde, est_Q_actual
    
    
    def calculate_VQ_inf(self, gamma: float = 0.7):
        """
        Calculate value function V and Q-values for the infinite horizon setting with a given discount factor `gamma`.

        Parameters
        ----------
        gamma : float, optional
            The discount factor for future rewards. Default is 0.7.

        Returns
        -------
        original_results_inf : dict
            Dictionary containing results from theta estimation.
        df : DataFrame
            DataFrame of the formatted results of theta estimation.
        latex_table : str
            The LaTeX formatted table of the results.
        est_V : ndarray, shape (n_samples, 1)
            Estimated value function V.
        est_Q_actual : ndarray, shape (n_samples, 1)
            Estimated Q-values for the actual actions taken.
        """
        
        # Step 1: Calculate theta_inf and retrieve results
        # This method calculates the parameter estimates (`theta_hat`) for the infinite horizon case
        self.original_results_inf, df, latex_table = self.calculate_theta_inf(gamma)
        
        # Extract the coefficient estimates (theta) from the results
        theta_hat = self.original_results_inf['coef_estimate']
        
        # Step 2: Compute the estimated value function V using the basis functions and theta estimates
        est_V_hat = self.basis_df.values @ theta_hat.reshape(-1,1)
        est_V_tilde = self.r_pi_est[:, np.newaxis]+self.hat_psi_next @ theta_hat.reshape(-1,1) * gamma
        
        # Step 3: Calculate Q-values for the actual actions taken
        # Using the reward data and next-state basis function estimates for the actual actions
        est_Q_actual = self.est_r_sa_actual[:, np.newaxis]+self.est_next_psi_SA_actual @ theta_hat.reshape(-1,1) * gamma
        
        return self.original_results_inf, df, latex_table, est_V_hat, est_V_tilde, est_Q_actual
        
    
    def calculate_theta_w(self, scaled_state_df, action_df, scaled_next_x_df, reward_df, w: int = 10, gamma: float = 0.7):
        """
        Calculate theta_w based on time lag `w` and discount factor `gamma`.

        Parameters
        ----------
        scaled_state_df : DataFrame
            DataFrame containing the scaled state variables at the current time step.
        action_df : DataFrame
            DataFrame containing the actions taken in the current time step.
        scaled_next_x_df : DataFrame
            DataFrame containing the scaled state variables at the next time step (used for future predictions).
        reward_df : DataFrame
            DataFrame containing the reward values associated with the actions taken in the current state.
        w : int, optional
            The time lag used in the reward estimation. Default value is 10.
        gamma : float, optional
            The discount factor for future rewards. A value between 0 and 1 that indicates how much future rewards are discounted 
            relative to immediate rewards. Default value is 0.7.

        Returns
        -------
        original_results : DataFrame
            DataFrame containing the results from the ordinary least squares model fit, including coefficients and statistics.
        df : DataFrame
            DataFrame of basis function estimates, adjusted for the specified time lag.
        latex_table : str
            String representation of the results formatted for LaTeX output, useful for academic reports.
        est_r_pi_sa_w_actual : DataFrame
            DataFrame of the actual reward estimates for each action, adjusted according to the time lag.
        """

        # Estimate reward with time lag `w`
        self.r_pi_w_est = self.estimation_reward_w(self.h_x, self.alpha, scaled_state_df, reward_df, w)
        
        # Compute the right-hand side based on estimated rewards
        R_pi = self.r_pi_w_est * (gamma ** w) - self.r_pi_est 

        # Calculate the left-hand matrix for basis function estimation
        self.zeta_w = gamma * self.hat_psi_next - self.basis_df.values
        
        # Fit an ordinary least squares model with HAC standard errors
        model = sm.OLS(R_pi[:, np.newaxis], self.zeta_w).fit(cov_type='HAC', cov_kwds={'maxlags': self.max_lag})
        
        # Organize results for output
        original_results, df, latex_table = self.organizing_result(model)
        
        # Initialize time-windowed reward estimator
        r_pi_sa_w_hat = est_r_pi_sa_w(self.h_x, self.h_a, self.alpha)  
        r_pi_sa_w_hat.fit(scaled_state_df.values, action_df.values, reward_df.values, w)
        
        # Estimate time-windowed reward  for the actions taken
        est_r_pi_sa_w_actual = r_pi_sa_w_hat(scaled_state_df.values, action_df.values)

        
        
        return original_results, df, latex_table, est_r_pi_sa_w_actual
    
    def calculate_theta_inf(self, gamma: float = 0.99):
        """
        Calculate theta in the infinite-horizon case using the discount factor `gamma`.

        Parameters
        ----------
        gamma : float, optional
            The discount factor for future rewards, determining their present value.

        Returns
        -------
        original_results : DataFrame
            DataFrame containing the results from the ordinary least squares model fit, including coefficients and statistics.
        basis_function_df : DataFrame
            DataFrame of basis function estimates used in the model.
        latex_table : str
            String representation of the results formatted for LaTeX output, useful for academic reports.
        """

        # Compute the right-hand side based on estimated rewards
        R_pi = - self.r_pi_est 

        # Calculate the left-hand matrix for basis function estimation
        self.zeta_inf = gamma * self.hat_psi_next - self.basis_df.values
        
        # Fit an ordinary least squares model with HAC standard errors
        model = sm.OLS(R_pi[:, np.newaxis], self.zeta_inf).fit(cov_type='HAC', cov_kwds={'maxlags': self.max_lag})
        
        # Organize results for output
        original_results, df, latex_table = self.organizing_result(model) 
        
        
        
        return original_results, df, latex_table
        
    
    def search_optimal_bandwidth_x(self, scaled_state_df, search_interval_x=np.linspace(0.04, 0.05, 100), cv_k_x=20):
        """
        Search for the optimal bandwidth for kernel density estimation on state data using cross-validation.

        Parameters
        ----------
        scaled_state_df : DataFrame
            Scaled state data. This DataFrame contains the state variables after scaling or normalization.
        search_interval_x : array-like, optional
            Range of bandwidth values to search for kernel density estimation (KDE). Default is np.linspace(0.04, 0.05, 100).
        cv_k_x : int, optional
            Number of folds for cross-validation. Default is 20.

        Returns
        -------
        float
            The optimal bandwidth value for the state data.
        """

        # Perform a grid search for the optimal bandwidth using cross-validation
        grid_search_custom_x = GridSearchCV(estimator=KDE(),  # Use a KDE estimator
                                              param_grid={'bandwidth': search_interval_x}, # Define search range for bandwidth
                                          cv=cv_k_x) # Perform k-fold cross-validation with cv_k_x folds
        grid_search_custom_x.fit(scaled_state_df.values) # Fit the KDE model to the scaled state data
        
        # Extract the best bandwidth value from the grid search results
        h_x = grid_search_custom_x.best_params_["bandwidth"]

        print(f"Optimal bandwidth for state KDE: {grid_search_custom_x.best_params_}")  # Output the optimal bandwidth
        
        return h_x # Return the optimal bandwidth
    
    
    def search_optimal_bandwidth_a(self, scaled_state_df, action_df, h_x, search_interval_a=np.linspace(0.04, 0.05, 100), cv_k_a=20):
        """
        Search for the optimal bandwidth for kernel density estimation on action data using cross-validation.

        Parameters
        ----------
        scaled_state_df : DataFrame
            Scaled state data, used to condition the action bandwidth search.
        action_df : DataFrame
            Action data. This DataFrame contains the actions corresponding to each state.
        h_x : float
            Optimal bandwidth for the state data obtained from the previous search.
        search_interval_a : array-like, optional
            Range of bandwidth values to search for kernel density estimation (KDE) for the action data. Default is np.linspace(0.04, 0.05, 100).
        cv_k_a : int, optional
            Number of folds for cross-validation. Default is 20.

        Returns
        -------
        float
            The optimal bandwidth value for the action data.
        """

        # Perform a grid search for the optimal bandwidth for the action data using cross-validation
        grid_search_custom_a = GridSearchCV(est_policy(bandwidth_x=h_x), # Pass the previously computed bandwidth for state KDE
                               param_grid={'bandwidth_a': search_interval_a}, # Define search range for action bandwidth
                           cv=cv_k_a) # Perform k-fold cross-validation with cv_k_a folds
        
        # Fit the model to the scaled state and action data
        grid_search_custom_a.fit(scaled_state_df.values, action_df.values)
        # Extract the best bandwidth value from the grid search results
        h_a = grid_search_custom_a.best_params_['bandwidth_a']

        print(f"Optimal bandwidth for action KDE: {grid_search_custom_a.best_params_}")  # Output the optimal bandwidth
        
        return h_a # Return the optimal bandwidth for the action data
    
    
    def estimation_r_SA_next_psi_SA(self, h_x, h_a, alpha, scaled_state_df, action_df, scaled_next_x_df, reward_df, basis):
        """
        Estimate the expected reward and expected next-state basis functions for each state and action using kernel density estimation.

        Parameters
        ----------
        h_x : float
            The bandwidth for kernel density estimation, influencing the smoothness of the kernel function for state variables.
        h_a : float
            The bandwidth for kernel density estimation, influencing the smoothness of the kernel function for action variables.
        alpha : float
            Regularization parameter to control overfitting, particularly with larger datasets.
        scaled_state_df : DataFrame
            Scaled state data representing the current states in the estimation process.
        action_df : DataFrame
            Action data corresponding to each state, indicating the actions taken in each state.
        scaled_next_x_df : DataFrame
            Scaled next-state data, used for estimating the expected value of the next state.
        reward_df : DataFrame
            Reward data corresponding to each action taken in each state.
        basis : object
            Basis functions for the next-state estimation.

        Returns
        -------
        est_r_sa_actual : array-like, shape (n_samples,)
            Actual expected rewards corresponding to the actions taken.
        est_next_psi_SA_actual : array-like
            Actual next-state basis estimates corresponding to the actions taken.
        """
            
        # Instantiate the reward estimator and fit it to the data
        r_sa_hat = est_r_sa(h_x, h_a, alpha)
        r_sa_hat.fit(scaled_state_df.values, action_df.values, reward_df.values)

        # Estimate expected rewards for the actions taken
        est_r_sa_actual = r_sa_hat(scaled_state_df.values, action_df.values) 

        
        # Instantiate the next-state basis expectation estimator and fit it to the data
        next_psi_SA = BasisNextSAExpect(h_x, h_a, alpha)
        next_psi_SA.fit(scaled_state_df.values, action_df.values, scaled_next_x_df.values)

        # Estimate the next-state basis values for the actions taken
        est_next_psi_SA_actual = next_psi_SA(scaled_state_df.values, action_df.values, basis)

        
        return est_r_sa_actual, est_next_psi_SA_actual

    def estimation_reward(self, h_x, alpha, scaled_state_df, reward_df):
        """
        Estimate the expected reward for each state.

        Parameters
        ----------
        h_x : float
            The bandwidth parameter for kernel density estimation.
        alpha : float
            Regularization parameter to prevent overfitting in the reward estimation process.
        scaled_state_df : DataFrame
            Scaled state data.
        reward_df : DataFrame
            DataFrame containing the reward values.

        Returns
        -------
        ndarray
            Estimated reward values for each state.
        """
        # Initialize reward estimator
        r_pi_hat = est_r_pi(h_x, alpha)  # Instantiate the reward estimator
        r_pi_hat.fit(scaled_state_df.values, reward_df.values)

        r_pi_est = r_pi_hat(scaled_state_df.values)  # Estimate rewards

        return r_pi_est

    def estimation_reward_w(self, h_x, alpha, scaled_state_df, reward_df, w):
        """
        Estimate the reward over a time window `w`.

        Parameters
        ----------
        h_x : float
            The bandwidth parameter for kernel density estimation.
        alpha : float
            Regularization parameter to prevent overfitting in the reward estimation process.
        scaled_state_df : DataFrame
            Scaled state data.
        reward_df : DataFrame
            DataFrame containing the reward values.
        w : int
            The time window over which to estimate the reward.

        Returns
        -------
        ndarray
            Estimated reward values over the time window.
        """
        
        # Initialize time-windowed reward estimator
        r_pi_w_hat = est_r_pi_w(h_x, alpha)  
        r_pi_w_hat.fit(scaled_state_df.values, reward_df.values, w)

        r_pi_w_est = r_pi_w_hat(scaled_state_df.values)  # Estimate rewards over time window `w`

        return r_pi_w_est
    
    

    def compute_basis(self, h_x, alpha, scaled_state_df, scaled_next_x_df, basis):
        """
        Compute the Chebyshev basis functions and their expectations for the next state.

        Parameters
        ----------
        h_x : float
            The bandwidth parameter for kernel density estimation.
        alpha : float
            Regularization parameter to prevent overfitting in kernel estimation.
        scaled_state_df : DataFrame
            Scaled state data.
        scaled_next_x_df : DataFrame
            Scaled next state data.
        order : int, optional
            The order of the Chebyshev basis function expansion.

        Returns
        -------
        DataFrame
            DataFrame containing the basis functions.
        dict
            Dictionary of basis function definitions.
        ndarray
            Estimated expectations of the next state's basis functions.
        """
        
        basis_df, basis_dict = basis(scaled_state_df.values)  # Compute the basis functions

        next_psi = BasisNextExpect(h_x, alpha)  # Instantiate the next-state basis expectation estimator
        next_psi.fit(scaled_state_df.values, scaled_next_x_df.values)

        # Estimate expectations for the next state
        hat_psi_next = next_psi(scaled_state_df.values, basis)

        return basis_df, basis_dict, hat_psi_next
    
    
    def significance_stars(self, p_value):
        """
        Assign significance stars based on p-value.
        
        Parameters
        ----------
        p_value : float
            The p-value for statistical significance.
            
        Returns
        -------
        str
            Stars representing the level of significance ('***' for p < 0.01, '**' for p < 0.05, '*' for p < 0.1).
        """
        if p_value < 0.01:
            return "***"  # Very significant
        elif p_value < 0.05:
            return "**"   # Significant
        elif p_value < 0.1:
            return "*"    # Marginally significant
        else:
            return ""      # Not significant
        
    
    def scientific_to_latex(self, value):
        """
        Convert a numerical value to a LaTeX string in scientific notation.

        Parameters
        ----------
        value : float
            The value to convert.

        Returns
        -------
        str
            The LaTeX formatted string.
        """
        if value == 0:
            return "$0$"
        else:
            exp = int(np.floor(np.log10(np.abs(value))))

            # Return the base directly if the exponent is 0
            if exp == 0:
                return f"${value:.3f}$"
            
            base = value / 10**exp

            return f"${base:.3f} \\times 10^{{{exp}}}$"
        
    # Function to convert a string to a tuple of floats
    def convert_string_to_tuple(self, s):
        return ast.literal_eval(s)

    # Function to convert a tuple to LaTeX formatted strings
    def format_tuple_to_latex(self, values):
        latex_values = [self.scientific_to_latex(float(value)) for value in values]  # Use self.scientific_to_latex to format each value
        return f"({', '.join(latex_values)})"  # Return formatted tuple as a string
    
    def combine_pval_stars(self, row, pval_col, stars_col):
        """
        Combine p-value and significance stars into a single string.
        
        Parameters
        ----------
        row : pandas.Series
            The row of the DataFrame containing the p-value and stars.
        pval_col : str
            The name of the p-value column.
        stars_col : str
            The name of the stars column.
        
        Returns
        -------
        str
            Combined string of p-value and significance stars.
        """
        return f"{row[pval_col]} {row[stars_col]}"
    
    def organizing_result(self, model):
        """
        Organize the results from a fitted regression model into a structured format.

        Parameters
        ----------
        model : statsmodels regression model
            A fitted regression model from which to extract results.

        Returns
        -------
        original_results : dict
            A dictionary containing the original numeric results, including coefficient estimates,
            standard errors, t-values, p-values, confidence intervals, covariance matrix,
            and significance stars.
        df : DataFrame
            A DataFrame of formatted results suitable for display, including coefficients, 
            HAC standard errors, t-values, p-values, confidence intervals, and significance stars.
        latex_table : str
            A LaTeX formatted string representation of the results DataFrame for inclusion in LaTeX documents.
        """

        # Extract various statistical measures from the model
        coef_estimate = model.params # Coefficient estimate
        t_value = model.tvalues  # Extract t-statistic
        p_value = model.pvalues  # Extract p-value
        hac_std_error = model.bse  # Extract HAC standard error
        conf_interval = model.conf_int() # Extract CI
        cov_theta_hat = model.cov_HC0 # Extract HAC covariance matrix

        stars = [self.significance_stars(p) for p in p_value]

        # Format the results into a tuple for display
        formatted_results = [
            (
                f"{coef_estimate[i]:.4g}",
                f"{hac_std_error[i]:.4g}",
                f"{t_value[i]:.4g}",
                f"{p_value[i]:.4g}",
                f"({conf_interval[i, 0]:.4g}, {conf_interval[i, 1]:.4g})",
                stars[i]
            )
            for i in range(len(coef_estimate))
        ]

        # Store original numeric results in a dictionary
        original_results = {
            "coef_estimate": coef_estimate,
            "hac_std_error": hac_std_error,
            "t_value": t_value,
            "p_value": p_value,
            "95%_confidence_interval": conf_interval,
            "hac_cov_matrix": cov_theta_hat,
            "stars": stars
        }

        # Create a list to store the formatted strings
        formatted_list = []

        # Generate the corresponding list of formatted strings for each basis
        for key, value in self.basis_dict.items():
            formatted_string = f"$\\hat{{\\theta}}_{{{value[0]}, {value[1]}}}$"
            formatted_list.append(formatted_string)

        # Create DataFrame of results
        column_names = ["Coef", "HAC SE", "t-val", "p-val", "95\% CI", 'stars']
        df = pd.DataFrame(formatted_results, columns=column_names, index=formatted_list)

        # Convert numeriself columns to LaTeX scientific notation
        for col in ["Coef", "HAC SE", "t-val", "p-val"]:
            df[col] = df[col].astype(float).apply(self.scientific_to_latex)

        # Convert strings to tuples
        df["95\% CI"] = df["95\% CI"].apply(self.convert_string_to_tuple)
        # Apply the formatting function to the "95% CI" column
        df["95\% CI"] = df["95\% CI"].apply(self.format_tuple_to_latex)

        # Combine p-val (2-sided) and stars into a single column
        df['p-val'] = df.apply(lambda row: self.combine_pval_stars(row, 'p-val', 'stars'), axis=1)

        # Drop the original stars column as it's no longer needed
        df = df.drop(columns=['stars'])

        # Create LaTeX table format
        latex_table = df.to_latex(index=True, escape=False).replace(r'\toprule', r'\hline').replace(r'\midrule', r'\hline').replace(r'\bottomrule', r'\hline')


        return original_results, df, latex_table



# In[ ]:


class MultipleHacTest:
    def __init__(self, est_Q_actual, est_V_hat, est_V_tilde):
        """
        Initialize the HacTest class for causal inference testing in time series.

        This class implements the testing framework to assess the causal effects
        of taking actual actions versus not taking them, using state-action value functions.

        Parameters
        ----------
        est_Q_actual : array-like
            Estimated values of the actual state-action quality function, reflecting the 
            outcomes of actions taken in the time series.

        est_V_hat : array-like
            Estimated values of the state value function using basis function expansion with the 
            estimated parameters, representing the expected outcomes of states under the current policy 
            and used for comparison in causal inference testing.

        est_V_tilde : array-like
            Estimated values of the state value function obtained using Nadaraya-Watson estimators 
            with the estimated parameters, representing the expected outcomes of states under the 
            current policy and used for comparison in causal inference testing.
        """
        # Store the actual estimated state-action value function values
        self.est_Q_actual = est_Q_actual
        
        # Store the estimated state value function values (hat)
        self.est_V_hat = est_V_hat
        
        # Store the estimated state value function values (tilde)
        self.est_V_tilde = est_V_tilde

    def __call__(self, displaying = True, displaying_latex = False):
        """
        Execute the HacTest by computing tau, performing statistical tests,
        and printing results, including t-statistics, p-values, and confidence intervals.
        This testing incorporates HAC standard errors for robust inference.

        Parameters
        ----------
        displaying : bool
            Whether to display the histogram of tau values.
        displaying_latex : bool
            Whether to print the LaTeX formatted results.
        """
        # Compute the advantage values (difference between actual and counterfactual quality functions)
        self.Advantage_hat = self.est_Q_actual- self.est_V_hat
        self.Advantage_tilde = self.est_Q_actual- self.est_V_tilde
        
        # Determine max lag for HAC based on sample size
        max_lag = int(4 * ((self.est_Q_actual.shape[0] / 100) ** (1 / 3)))

        # Compute statistics for tau, including HAC standard errors
        self.coefficient_hat, self.t_stat_hat, self.p_value_hat, self.hac_std_error_hat, self.p_value_hat_upper, self.p_value_hat_lower = self.compute_statistics(self.Advantage_hat, max_lag)
        self.coefficient_tilde, self.t_stat_tilde, self.p_value_tilde, self.hac_std_error_tilde, self.p_value_tilde_upper, self.p_value_tilde_lower = self.compute_statistics(self.Advantage_tilde, max_lag)
        
        self.p_values = [self.p_value_hat, self.p_value_tilde]
        self.p_values_upper = [self.p_value_hat_upper, self.p_value_tilde_upper]
        self.p_values_lower = [self.p_value_hat_lower, self.p_value_tilde_lower]

        # Adjusting p-values using the Bonferroni method
        self.p_values_adjusted = multipletests(self.p_values, method='holm')[1]
        self.p_values_upper_adjusted = multipletests(self.p_values_upper, method='holm')[1]
        self.p_values_lower_adjusted = multipletests(self.p_values_lower, method='holm')[1]
        
        # Assign significance stars based on p-values
        stars_hat_original_two_sided = self.significance_stars(self.p_value_hat)
        stars_hat_original_upper = self.significance_stars(self.p_value_hat_upper)
        stars_hat_original_lower = self.significance_stars(self.p_value_hat_lower)
        
        stars_tilde_original_two_sided = self.significance_stars(self.p_value_tilde)
        stars_tilde_original_upper = self.significance_stars(self.p_value_tilde_upper)
        stars_tilde_original_lower = self.significance_stars(self.p_value_tilde_lower)
        
        stars_hat_adjusted_two_sided = self.significance_stars(self.p_values_adjusted[0])
        stars_hat_adjusted_upper = self.significance_stars(self.p_values_upper_adjusted[0])
        stars_hat_adjusted_lower = self.significance_stars(self.p_values_lower_adjusted[0])
        
        stars_tilde_adjusted_two_sided = self.significance_stars(self.p_values_adjusted[1])
        stars_tilde_adjusted_upper = self.significance_stars(self.p_values_upper_adjusted[1])
        stars_tilde_adjusted_lower = self.significance_stars(self.p_values_lower_adjusted[1])
        
        alpha = 0.05
        
        # Compute 95% confidence interval using HAC standard errors
        critical_value_hat = stats.t.ppf(1 - alpha / 2, df=self.Advantage_hat.shape[0] - 1)  # Two-tailed critical value
        lower_bound_hat = self.coefficient_hat - critical_value_hat * self.hac_std_error_hat # Lower bound of CI
        upper_bound_hat = self.coefficient_hat + critical_value_hat * self.hac_std_error_hat # Upper bound of CI
        conf_interval_hat = (lower_bound_hat, upper_bound_hat) # Confidence interval
        
        critical_value_tilde = stats.t.ppf(1 - alpha / 2, df=self.Advantage_tilde.shape[0] - 1)  # Two-tailed critical value
        lower_bound_tilde = self.coefficient_tilde - critical_value_tilde * self.hac_std_error_tilde # Lower bound of CI
        upper_bound_tilde = self.coefficient_tilde + critical_value_tilde * self.hac_std_error_tilde # Upper bound of CI
        conf_interval_tilde = (lower_bound_tilde, upper_bound_tilde) # Confidence interval
        
        # Print coefficient, t-statistic, and p-value
        print("\n\nt-test using HAC SE for Advantage_hat;")
        print(f"Coefficient: {self.coefficient_hat}")
        print(f"t-statistic: {self.t_stat_hat}")
        print(f"95% Confidence Interval for the coefficient: {conf_interval_hat}")
        
        print(f"\np-value (two-tailed test): {self.p_value_hat}")
        print(f"adjusted p-value (two-tailed test): {self.p_values_adjusted[0]}")
        
        print(f"\np-value (upper-tailed test): {self.p_value_hat_upper}")
        print(f"adjusted p-value (upper-tailed test): {self.p_values_upper_adjusted[0]}")
        
        print(f"\np-value (lower-tailed test): {self.p_value_hat_lower}")
        print(f"adjusted p-value (lower-tailed test): {self.p_values_lower_adjusted[0]}")
        
        
        print("\n\nt-test using HAC SE for Advantage_tilde;")
        print(f"Coefficient: {self.coefficient_tilde}")
        print(f"t-statistic: {self.t_stat_tilde}")
        print(f"95% Confidence Interval for the coefficient: {conf_interval_tilde}")
        
        print(f"\np-value (two-tailed test): {self.p_value_tilde}")
        print(f"adjusted p-value (two-tailed test): {self.p_values_adjusted[1]}")
        
        print(f"\np-value (upper-tailed test): {self.p_value_tilde_upper}")
        print(f"adjusted p-value (upper-tailed test): {self.p_values_upper_adjusted[1]}")
        
        print(f"\np-value (lower-tailed test): {self.p_value_tilde_lower}")
        print(f"adjusted p-value (lower-tailed test): {self.p_values_lower_adjusted[1]}")
        
        
        # Two-tailed hypothesis test for mean different from 0
        print("\nTwo-tailed test (whether the mean is different from 0):")
        if (self.p_values_adjusted[0] < alpha) & (self.p_values_adjusted[1] < alpha):
            print(">> Reject the null hypothesis. The sample mean is considered to be significantly different from 0.\n")
        else:
            print(">> Fail to reject the null hypothesis.\n")

        # One-tailed tests setup
        print("Upper-tailed test (whether the mean is greater than 0):")
        if (self.p_values_upper_adjusted[0] < alpha/2) & (self.p_values_upper_adjusted[1] < alpha/2):
            print(">> Reject the null hypothesis. The sample mean is considered to be greater than 0.\n")
        else:
            print(">> Fail to reject the null hypothesis.\n")
            
        print("Lower-tailed test (whether the mean is less than 0):")
        if (self.p_values_lower_adjusted[0] < alpha/2) & (self.p_values_lower_adjusted[1] < alpha/2):
            print(">> Reject the null hypothesis. The sample mean is considered to be less than 0.\n")
        else:
            print(">> Fail to reject the null hypothesis.\n")
            
        # Plot histogram of the advantage values
        percentile_hat_lower_bound = np.percentile(self.Advantage_hat, 2.5)  # Lower 2.5 percentile
        percentile_hat_upper_bound = np.percentile(self.Advantage_hat, 97.5)  # Upper 97.5 percentile

        # Filter data within the specified percentiles for the histogram
        filtered_data_hat = self.Advantage_hat[(self.Advantage_hat >= percentile_hat_lower_bound) & (self.Advantage_hat <= percentile_hat_upper_bound)]
        
        # Plot histogram of the advantage values
        percentile_tilde_lower_bound = np.percentile(self.Advantage_tilde, 2.5)  # Lower 2.5 percentile
        percentile_tilde_upper_bound = np.percentile(self.Advantage_tilde, 97.5)  # Upper 97.5 percentile

        # Filter data within the specified percentiles for the histogram
        filtered_data_tilde = self.Advantage_tilde[(self.Advantage_tilde >= percentile_tilde_lower_bound) & (self.Advantage_tilde <= percentile_tilde_upper_bound)]
        
        
        if displaying:
            fig_hat = plt.figure(figsize=(8, 6))
            self.ax_hat = fig_hat.add_subplot(111) # Create a subplot for the histogram
            self.ax_hat.hist(filtered_data_hat, bins=100, edgecolor='k') # Plot histogram with 100 bins
            self.ax_hat.set_xlabel(r'$\hat{A}_{\pi}(S_{t}, A_{t};w, \hat{\theta})$', fontsize=30) # Set x-label
            self.ax_hat.set_ylabel('Frequency', fontsize=30) # Set y-label

            # Increase the size of tick labels
            self.ax_hat.tick_params(axis='both', labelsize=24) # Increase tick label size

            plt.show()  # Show the plot if desired

            # Save plot to a BytesIO object 
            buf_hat = io.BytesIO() # Create a buffer to save the plot
            plt.savefig(buf_hat, format='eps', transparent=False, bbox_inches='tight') # Save the figure to the buffer
            buf_hat.seek(0)  # Move to the beginning of the buffer

            # Close the plot to free memory
            #plt.close(fig)

            # Check the buffer size
            print(f"Buffer size: {buf_hat.getbuffer().nbytes} bytes") # Print the size of the buffer

            fig_tilde = plt.figure(figsize=(8, 6))
            self.ax_tilde = fig_tilde.add_subplot(111) # Create a subplot for the histogram
            self.ax_tilde.hist(filtered_data_tilde, bins=100, edgecolor='k') # Plot histogram with 100 bins
            self.ax_tilde.set_xlabel(r'$\tilde{A}_{\pi}(S_{t}, A_{t};w, \hat{\theta})$', fontsize=30) # Set x-label
            self.ax_tilde.set_ylabel('Frequency', fontsize=30) # Set y-label

            # Increase the size of tick labels
            self.ax_tilde.tick_params(axis='both', labelsize=24) # Increase tick label size

            plt.show()  # Show the plot if desired

            # Save plot to a BytesIO object 
            buf_tilde = io.BytesIO() # Create a buffer to save the plot
            plt.savefig(buf_tilde, format='eps', transparent=False, bbox_inches='tight') # Save the figure to the buffer
            buf_tilde.seek(0)  # Move to the beginning of the buffer

            # Close the plot to free memory
            #plt.close(fig)

            # Check the buffer size
            print(f"Buffer size: {buf_tilde.getbuffer().nbytes} bytes") # Print the size of the buffer
        
        else: buf_hat =None; buf_tilde =None

        
        # Format the results into a tuple for display
        formatted_results_hat = (
            f"{self.coefficient_hat:.4g}",
            f"{self.hac_std_error_hat:.4g}",
            f"{self.t_stat_hat:.4g}",
            f"({lower_bound_hat:.4g}, {upper_bound_hat:.4g})",
            f"{self.p_value_hat:.4g}",
            f"{self.p_values_adjusted[0]:.4g}",
            f"{self.p_value_hat_upper:.4g}",
            f"{self.p_values_upper_adjusted[0]:.4g}",
            f"{self.p_value_hat_lower:.4g}",
            f"{self.p_values_lower_adjusted[0]:.4g}",
            stars_hat_original_two_sided,
            stars_hat_adjusted_two_sided,
            stars_hat_original_upper,
            stars_hat_adjusted_upper,
            stars_hat_original_lower,
            stars_hat_adjusted_lower
        )
        
        formatted_results_tilde = (
            f"{self.coefficient_tilde:.4g}",
            f"{self.hac_std_error_tilde:.4g}",
            f"{self.t_stat_tilde:.4g}",
            f"({lower_bound_tilde:.4g}, {upper_bound_tilde:.4g})",
            f"{self.p_value_tilde:.4g}",
            f"{self.p_values_adjusted[1]:.4g}",
            f"{self.p_value_tilde_upper:.4g}",
            f"{self.p_values_upper_adjusted[1]:.4g}",
            f"{self.p_value_tilde_lower:.4g}",
            f"{self.p_values_lower_adjusted[1]:.4g}",
            stars_tilde_original_two_sided,
            stars_tilde_adjusted_two_sided,
            stars_tilde_original_upper,
            stars_tilde_adjusted_upper,
            stars_tilde_original_lower,
            stars_tilde_adjusted_lower
        )

        # Store original numeric results in a dictionary
        original_results_hat = {
            "coef_estimate": self.coefficient_hat,
            "hac_std_error": self.hac_std_error_hat,
            "t_value": self.t_stat_hat,
            "95%_confidence_interval": {
                "lower_bound": lower_bound_hat,
                "upper_bound": upper_bound_hat
            },
            "original_p_value_two_sided": self.p_value_hat,
            "adjusted_p_value_two_sided": self.p_values_adjusted[0],
            "original_p_value_upper": self.p_value_hat_upper,
            "adjusted_p_value_upper": self.p_values_upper_adjusted[0],
            "original_p_value_lower": self.p_value_hat_lower,
            "adjusted_p_value_lower": self.p_values_lower_adjusted[0],
            "original_stars_two_sided": stars_hat_original_two_sided,
            "adjusted_stars_two_sided": stars_hat_adjusted_two_sided,
            "original_stars_upper": stars_hat_original_upper,
            "adjusted_stars_upper": stars_hat_adjusted_upper,
            "original_stars_lower": stars_hat_original_lower,
            "adjusted_stars_lower": stars_hat_adjusted_lower
        }
        
        original_results_tilde = {
            "coef_estimate": self.coefficient_tilde,
            "hac_std_error": self.hac_std_error_tilde,
            "t_value": self.t_stat_tilde,
            "95%_confidence_interval": {
                "lower_bound": lower_bound_tilde,
                "upper_bound": upper_bound_tilde
            },
            "original_p_value_two_sided": self.p_value_tilde,
            "adjusted_p_value_two_sided": self.p_values_adjusted[1],
            "original_p_value_upper": self.p_value_tilde_upper,
            "adjusted_p_value_upper": self.p_values_upper_adjusted[1],
            "original_p_value_lower": self.p_value_tilde_lower,
            "adjusted_p_value_lower": self.p_values_lower_adjusted[1],
            "original_stars_two_sided": stars_tilde_original_two_sided,
            "adjusted_stars_two_sided": stars_tilde_adjusted_two_sided,
            "original_stars_upper": stars_tilde_original_upper,
            "adjusted_stars_upper": stars_tilde_adjusted_upper,
            "original_stars_lower": stars_tilde_original_lower,
            "adjusted_stars_lower": stars_tilde_adjusted_lower
        }
        
        # Create a LaTeX table from the formatted results
        df_hat = self.create_latex_table(formatted_results_hat)
        df_tilde = self.create_latex_table(formatted_results_tilde) 
        # Create LaTeX table format
        latex_table_hat = df_hat.to_latex(index=True, escape=False).replace(r'\toprule', r'\hline').replace(r'\midrule', r'\hline').replace(r'\bottomrule', r'\hline')
        latex_table_tilde = df_tilde.to_latex(index=True, escape=False).replace(r'\toprule', r'\hline').replace(r'\midrule', r'\hline').replace(r'\bottomrule', r'\hline')
        
        # Print the LaTeX formatted table if desired
        if displaying_latex:
            print("LaTeX formatted table for Advantage_hat:")
            print(latex_table_hat)
            
            print("LaTeX formatted table for Advantage_tilde:")
            print(latex_table_tilde)
        
        return original_results_hat, original_results_tilde, latex_table_hat, latex_table_tilde, buf_hat, buf_tilde
    
    def significance_stars(self, p_value):
        """
        Assign significance stars based on p-value.
        
        Parameters
        ----------
        p_value : float
            The p-value for statistical significance.
            
        Returns
        -------
        str
            Stars representing the level of significance ('***' for p < 0.01, '**' for p < 0.05, '*' for p < 0.1).
        """
        if p_value < 0.01:
            return "***"
        elif p_value < 0.05:
            return "**"
        elif p_value < 0.1:
            return "*"
        else:
            return ""
    
    def compute_statistics(self, advantage, max_lag):
        """
        Compute statistics related to advantage functions, including standard error, t-statistic, and p-values.

        This method fits an Ordinary Least Squares (OLS) regression model to the computed tau values 
        (advantage function values) and calculates HAC standard errors for robust inference.

        Parameters
        ----------
        advantage : array-like
            Computed tau values representing the advantage function values, which indicate the 
            relative performance of the actions taken.

        max_lag : int
            Maximum lag to consider for the HAC (Heteroskedasticity and Autocorrelation Consistent) 
            standard error calculation.

        Returns
        -------
        tuple
            Contains the following statistics:
            - Coefficient (mean of the advantage values)
            - t-statistic for the coefficient
            - p-value (two-tailed) for the significance of the coefficient
            - HAC standard error of the coefficient
            - Upper one-tailed p-value for the hypothesis test
            - Lower one-tailed p-value for the hypothesis test
        """
        # Prepare the design matrix for OLS regression (constant term)
        X = np.ones_like(advantage)

        # Fit an OLS model to the advantage with HAC standard errors
        model = sm.OLS(advantage, X)
        results = model.fit(cov_type='HAC', cov_kwds={'maxlags': max_lag})
        
        # Extract results
        coefficient = results.params[0]  # Extract coefficient (mean)
        t_stat = results.tvalues[0]      # Extract t-statistic
        p_value = results.pvalues[0]     # Extract p-value (two-tailed)
        hac_std_error = results.bse[0]   # Extract HAC standard error

        # Calculate p-values for upper and lower one-tailed tests
        p_value_upper = 1 - stats.norm.cdf(t_stat)  # Upper-tailed p-value
        p_value_lower = stats.norm.cdf(t_stat)      # Lower-tailed p-value
        
        return coefficient, t_stat, p_value, hac_std_error, p_value_upper, p_value_lower
    
    def scientific_to_latex(self, value):
        """
        Convert a numerical value to a LaTeX string in scientific notation.

        Parameters
        ----------
        value : float
            The value to convert.

        Returns
        -------
        str
            The LaTeX formatted string.
        """
        if value == 0:
            return "$0$"
        else:
            exp = int(np.floor(np.log10(np.abs(value))))

            # Return the base directly if the exponent is 0
            if exp == 0:
                return f"${value:.3f}$"
            
            base = value / 10**exp

            return f"${base:.3f} \\times 10^{{{exp}}}$"
        
   # Function to convert a string to a tuple of floats
    def convert_string_to_tuple(self, s):
        return ast.literal_eval(s)

    # Function to convert a tuple to LaTeX formatted strings
    def format_tuple_to_latex(self, values):
        latex_values = [self.scientific_to_latex(float(value)) for value in values]
        return f"({', '.join(latex_values)})"

    
    def combine_pval_stars(self, row, pval_col, stars_col):
        return f"{row[pval_col]} {row[stars_col]}"


    def create_latex_table(self, formatted_results):
        """
        Create a DataFrame from formatted results to generate a LaTeX table.

        Parameters
        ----------
        formatted_results : tuple
            Tuple containing formatted statistical results.

        Returns
        -------
        DataFrame
            DataFrame formatted for LaTeX output.
        """
        # Convert results to a NumPy array
        results_array = np.array(formatted_results)
        # Create DataFrame of results
        index_names = ["Coef.", "HAC SE", "t-val","95\% CI", 'p-val (2-sided)', 'adj. p-val (2-sided)', 'p-val (upper)', 'adj. p-val (upper)', 'p-val (lower)', 'adj. p-val (lower)', 'stars (2-sided)', 'adj. stars (2-sided)','stars (upper)', 'adj. stars (upper)', 'stars (lower)', 'adj. stars (lower)']
        df = pd.DataFrame(results_array, index=index_names)
        df = df.T
        
        # Convert numerical columns to LaTeX scientific notation
        for col in ["Coef.", "HAC SE", "t-val", 'p-val (2-sided)', 'adj. p-val (2-sided)', 'p-val (upper)', 'adj. p-val (upper)', 'p-val (lower)', 'adj. p-val (lower)']:
            df[col] = df[col].astype(float).apply(self.scientific_to_latex)
            
        # Convert strings to tuples for confidence intervals
        df["95\% CI"] = df["95\% CI"].apply(self.convert_string_to_tuple)
        # Apply the formatting function to the "95% CI" column
        df["95\% CI"] = df["95\% CI"].apply(self.format_tuple_to_latex)
        
        # Combine p-values and stars for various significance tests
        df['p-val (2-sided)'] = df.apply(lambda row: self.combine_pval_stars(row, 'p-val (2-sided)', 'stars (2-sided)'), axis=1)
        df['p-val (upper)'] = df.apply(lambda row: self.combine_pval_stars(row, 'p-val (upper)', 'stars (upper)'), axis=1)
        df['p-val (lower)'] = df.apply(lambda row: self.combine_pval_stars(row, 'p-val (lower)', 'stars (lower)'), axis=1)
        
        df['adj. p-val (2-sided)'] = df.apply(lambda row: self.combine_pval_stars(row, 'adj. p-val (2-sided)', 'adj. stars (2-sided)'), axis=1)
        df['adj. p-val (upper)'] = df.apply(lambda row: self.combine_pval_stars(row, 'adj. p-val (upper)', 'adj. stars (upper)'), axis=1)
        df['adj. p-val (lower)'] = df.apply(lambda row: self.combine_pval_stars(row, 'adj. p-val (lower)', 'adj. stars (lower)'), axis=1)

        # Drop the individual stars columns as they are no longer needed
        df = df.drop(columns=['stars (2-sided)', 'adj. stars (2-sided)','stars (upper)', 'adj. stars (upper)', 'stars (lower)', 'adj. stars (lower)'])

        return df
    
    
class MultipleWilcoxonSignedRankTest:
    def __init__(self, est_Q_actual, est_V_hat, est_V_tilde):
        """
        Initialize the WilcoxonSignedRankTest class for causal inference testing in time series.

        This class implements the Wilcoxon signed-rank test framework to assess the causal effects
        of taking actual actions versus not taking them, using state-action quality functions.

        Parameters
        ----------
        est_Q_actual : array-like
            Estimated values of the actual state-action quality function, reflecting the 
            outcomes of actions taken in the time series.

        est_V_hat : array-like
            Estimated values of the state value function, which follows the same policy as 
            est_Q_actual, representing the expected outcomes of states under the current policy.

        est_V_tilde : array-like
            Estimated values of the state value function under a different policy or
            estimation method, used for comparison in causal inference testing.
        """
        # Store the actual estimated state-action value function values
        self.est_Q_actual = est_Q_actual
        
        # Store the estimated state value function values (hat)
        self.est_V_hat = est_V_hat
        
        # Store the estimated state value function values (tilde)
        self.est_V_tilde = est_V_tilde

    def __call__(self, displaying = True, displaying_latex = False):
        """
        Execute the Wilcoxon signed-rank test for the advantage function.
        This function performs statistical tests and prints results, including t-statistics,
        p-values, and confidence intervals. It incorporates HAC standard errors for robust inference.

        Parameters:
        displaying (bool): If True, displays plots of advantage values.
        displaying_latex (bool): Currently not used, but can be implemented for LaTeX formatting.
        """
        # Compute the advantage values (differences between actual Q values and estimated V values)
        self.Advantage_hat = self.est_Q_actual- self.est_V_hat
        self.Advantage_tilde = self.est_Q_actual- self.est_V_tilde
        
        # Compute the test statistics for the Wilcoxon signed-rank test
        self.stat_hat, self.p_value_hat, self.p_value_hat_upper, self.p_value_hat_lower = self.compute_statistics(self.Advantage_hat)
        self.stat_tilde, self.p_value_tilde, self.p_value_tilde_upper, self.p_value_tilde_lower = self.compute_statistics(self.Advantage_tilde)
        
        # Store p-values for both estimates
        self.p_values = [self.p_value_hat, self.p_value_tilde]
        self.p_values_upper = [self.p_value_hat_upper, self.p_value_tilde_upper]
        self.p_values_lower = [self.p_value_hat_lower, self.p_value_tilde_lower]
        
        # Adjusting p-values using the Bonferroni method
        self.p_values_adjusted = multipletests(self.p_values, method='holm')[1]
        self.p_values_upper_adjusted = multipletests(self.p_values_upper, method='holm')[1]
        self.p_values_lower_adjusted = multipletests(self.p_values_lower, method='holm')[1]
        
        
        # Print statistics and p-values for the Wilcoxon signed-rank test for Advantage_hat
        
        ## Print statistics and p-values for the two-tailed test
        print("\n\nWilcoxon Signed-Rank Test for Advantage_hat;")
        print(f"Wilcoxon-statistic: {self.stat_hat}")
        print(f"\np-value (two-tailed test): {self.p_value_hat}")
        print(f"adjusted p-value (two-tailed test): {self.p_values_adjusted[0]}")
        
        ## Print statistics and p-values for upper-tailed test
        print(f"\np-value (upper-tailed test): {self.p_value_hat_upper}")
        print(f"adjusted p-value (upper-tailed test): {self.p_values_upper_adjusted[0]}")
        
        ## Print statistics and p-values for lower-tailed test
        print(f"\np-value (lower-tailed test): {self.p_value_hat_lower}")
        print(f"adjusted p-value (lower-tailed test): {self.p_values_lower_adjusted[0]}")
        
        
        # Print statistics and p-values for the Wilcoxon signed-rank test for Advantage_tilde
        
        ## Print statistics and p-values for the two-tailed test
        print("\n\nWilcoxon Signed-Rank Rest for Advantage_tilde;")
        print(f"Wilcoxon-statistic: {self.stat_tilde}")
        print(f"\np-value (two-tailed test): {self.p_value_tilde}")
        print(f"adjusted p-value (two-tailed test): {self.p_values_adjusted[1]}")
        
        ## Print statistics and p-values for upper-tailed test
        print(f"\np-value (upper-tailed test): {self.p_value_tilde_upper}")
        print(f"adjusted p-value (upper-tailed test): {self.p_values_upper_adjusted[1]}")
        
        ## Print statistics and p-values for lower-tailed test
        print(f"\np-value (lower-tailed test): {self.p_value_tilde_lower}")
        print(f"adjusted p-value (lower-tailed test): {self.p_values_lower_adjusted[1]}")
        
        
         
        # Assign significance stars based on p-values
        stars_hat_original_two_sided = self.significance_stars(self.p_value_hat)
        stars_hat_original_upper = self.significance_stars(self.p_value_hat_upper)
        stars_hat_original_lower = self.significance_stars(self.p_value_hat_lower)
        
        stars_tilde_original_two_sided = self.significance_stars(self.p_value_tilde)
        stars_tilde_original_upper = self.significance_stars(self.p_value_tilde_upper)
        stars_tilde_original_lower = self.significance_stars(self.p_value_tilde_lower)
        
        stars_hat_adjusted_two_sided = self.significance_stars(self.p_values_adjusted[0])
        stars_hat_adjusted_upper = self.significance_stars(self.p_values_upper_adjusted[0])
        stars_hat_adjusted_lower = self.significance_stars(self.p_values_lower_adjusted[0])
        
        stars_tilde_adjusted_two_sided = self.significance_stars(self.p_values_adjusted[1])
        stars_tilde_adjusted_upper = self.significance_stars(self.p_values_upper_adjusted[1])
        stars_tilde_adjusted_lower = self.significance_stars(self.p_values_lower_adjusted[1])
        
        # Significance level
        alpha = 0.05

        # Two-tailed Wilcoxon test for the hypothesis of median different from 0
        print("\n\nTwo-tailed test (whether the median is different from 0):")
        if (self.p_values_adjusted[0] < alpha) & (self.p_values_adjusted[1] < alpha):
            print(">> Reject the null hypothesis. The sample median is considered to be significantly different from 0.\n")
        else:
            print(">> Fail to reject the null hypothesis.\n")

        # One-tailed Wilcoxon tests setup
        print("Upper-tailed test (whether the median is greater than 0):")
        if (self.p_values_upper_adjusted[0] < alpha/2) & (self.p_values_upper_adjusted[1] < alpha/2):
            print(">> Reject the null hypothesis. The sample median is considered to be greater than 0.\n")
        else:
            print(">> Fail to reject the null hypothesis.\n")

        print("Lower-tailed test (whether the median is less than 0):")
        if (self.p_values_lower_adjusted[0] < alpha/2) & (self.p_values_lower_adjusted[1] < alpha/2):
            print(">> Reject the null hypothesis. The sample median is considered to be less than 0.\n")
        else:
            print(">> Fail to reject the null hypothesis.\n")
        
        # Plot histogram of the advantage values for Advantage_hat
        percentile_hat_lower_bound = np.percentile(self.Advantage_hat, 2.5)  # Lower 2.5 percentile
        percentile_hat_upper_bound = np.percentile(self.Advantage_hat, 97.5)  # Upper 97.5 percentile

        # Filter data within the specified percentiles for the histogram
        filtered_data_hat = self.Advantage_hat[(self.Advantage_hat >= percentile_hat_lower_bound) & (self.Advantage_hat <= percentile_hat_upper_bound)]
        
        # Plot histogram of the advantage values for Advantage_tilde
        percentile_tilde_lower_bound = np.percentile(self.Advantage_tilde, 2.5)  # Lower 2.5 percentile
        percentile_tilde_upper_bound = np.percentile(self.Advantage_tilde, 97.5)  # Upper 97.5 percentile

        # Filter data within the specified percentiles for the histogram
        filtered_data_tilde = self.Advantage_tilde[(self.Advantage_tilde >= percentile_tilde_lower_bound) & (self.Advantage_tilde <= percentile_tilde_upper_bound)]
        
        # Generate histograms if displaying is set to True
        if displaying:
            # Generate histogram for Advantage_hat
            fig_hat = plt.figure(figsize=(8, 6))
            self.ax_hat = fig_hat.add_subplot(111) # Create a subplot for the histogram
            self.ax_hat.hist(filtered_data_hat, bins=100, edgecolor='k') # Plot histogram with 100 bins
            self.ax_hat.set_xlabel(r'$\hat{A}_{\pi}(S_{t}, A_{t};w, \hat{\theta})$', fontsize=30) # Set x-label
            self.ax_hat.set_ylabel('Frequency', fontsize=30) # Set y-label

            # Increase the size of tick labels
            self.ax_hat.tick_params(axis='both', labelsize=24) # Increase tick label size

            plt.show()  # Show the plot if desired

            # Save plot to a BytesIO object 
            buf_hat = io.BytesIO() # Create a buffer to save the plot
            plt.savefig(buf_hat, format='eps', transparent=False, bbox_inches='tight') # Save the figure to the buffer
            buf_hat.seek(0)  # Move to the beginning of the buffer

            # Close the plot to free memory
            #plt.close(fig)

            # Check the buffer size
            print(f"Buffer size: {buf_hat.getbuffer().nbytes} bytes") # Print the size of the buffer
            
            # Generate histogram for Advantage_tilde
            fig_tilde = plt.figure(figsize=(8, 6))
            self.ax_tilde = fig_tilde.add_subplot(111) # Create a subplot for the histogram
            self.ax_tilde.hist(filtered_data_tilde, bins=100, edgecolor='k') # Plot histogram with 100 bins
            self.ax_tilde.set_xlabel(r'$\tilde{A}_{\pi}(S_{t}, A_{t};w, \hat{\theta})$', fontsize=30) # Set x-label
            self.ax_tilde.set_ylabel('Frequency', fontsize=30) # Set y-label

            # Increase the size of tick labels
            self.ax_tilde.tick_params(axis='both', labelsize=24) # Increase tick label size

            plt.show()  # Show the plot if desired

            # Save plot to a BytesIO object 
            buf_tilde = io.BytesIO() # Create a buffer to save the plot
            plt.savefig(buf_tilde, format='eps', transparent=False, bbox_inches='tight') # Save the figure to the buffer
            buf_tilde.seek(0)  # Move to the beginning of the buffer

            # Close the plot to free memory
            #plt.close(fig)

            # Check the buffer size
            print(f"Buffer size: {buf_tilde.getbuffer().nbytes} bytes") # Print the size of the buffer
        
        else: buf_hat =None; buf_tilde =None

        
        # Format the results into a tuple for display
        formatted_results_hat = (
            f"{self.stat_hat:.4g}",
            f"{self.p_value_hat:.4g}",
            f"{self.p_values_adjusted[0]:.4g}",
            f"{self.p_value_hat_upper:.4g}",
            f"{self.p_values_upper_adjusted[0]:.4g}",
            f"{self.p_value_hat_lower:.4g}",
            f"{self.p_values_lower_adjusted[0]:.4g}",
            stars_hat_original_two_sided,
            stars_hat_adjusted_two_sided,
            stars_hat_original_upper,
            stars_hat_adjusted_upper,
            stars_hat_original_lower,
            stars_hat_adjusted_lower
        )
        
        formatted_results_tilde = (
            f"{self.stat_tilde:.4g}",
            f"{self.p_value_tilde:.4g}",
            f"{self.p_values_adjusted[1]:.4g}",
            f"{self.p_value_tilde_upper:.4g}",
            f"{self.p_values_upper_adjusted[1]:.4g}",
            f"{self.p_value_tilde_lower:.4g}",
            f"{self.p_values_lower_adjusted[1]:.4g}",
            stars_tilde_original_two_sided,
            stars_tilde_adjusted_two_sided,
            stars_tilde_original_upper,
            stars_tilde_adjusted_upper,
            stars_tilde_original_lower,
            stars_tilde_adjusted_lower
        )
        
        
        # Store original numeric results in a dictionary
        original_results_hat = {
            "W": self.stat_hat,
            "original_p_value_two_sided": self.p_value_hat,
            "adjusted_p_value_two_sided": self.p_values_adjusted[0],
            "original_p_value_upper": self.p_value_hat_upper,
            "adjusted_p_value_upper": self.p_values_upper_adjusted[0],
            "original_p_value_lower": self.p_value_hat_lower,
            "adjusted_p_value_lower": self.p_values_lower_adjusted[0],
            "original_stars_two_sided": stars_hat_original_two_sided,
            "adjusted_stars_two_sided": stars_hat_adjusted_two_sided,
            "original_stars_upper": stars_hat_original_upper,
            "adjusted_stars_upper": stars_hat_adjusted_upper,
            "original_stars_lower": stars_hat_original_lower,
            "adjusted_stars_lower": stars_hat_adjusted_lower   
        }
        
        original_results_tilde = {
            "W": self.stat_tilde,
            "original_p_value_two_sided": self.p_value_tilde,
            "adjusted_p_value_two_sided": self.p_values_adjusted[1],
            "original_p_value_upper": self.p_value_tilde_upper,
            "adjusted_p_value_upper": self.p_values_upper_adjusted[1],
            "original_p_value_lower": self.p_value_tilde_lower,
            "adjusted_p_value_lower": self.p_values_lower_adjusted[1],
            "original_stars_two_sided": stars_tilde_original_two_sided,
            "adjusted_stars_two_sided": stars_tilde_adjusted_two_sided,
            "original_stars_upper": stars_tilde_original_upper,
            "adjusted_stars_upper": stars_tilde_adjusted_upper,
            "original_stars_lower": stars_tilde_original_lower,
            "adjusted_stars_lower": stars_tilde_adjusted_lower 
        }
        
        # Create a LaTeX table from the formatted results
        df_hat = self.create_latex_table(formatted_results_hat)
        df_tilde = self.create_latex_table(formatted_results_tilde) 
        # Create LaTeX table format
        latex_table_hat = df_hat.to_latex(index=True, escape=False).replace(r'\toprule', r'\hline').replace(r'\midrule', r'\hline').replace(r'\bottomrule', r'\hline')
        latex_table_tilde = df_tilde.to_latex(index=True, escape=False).replace(r'\toprule', r'\hline').replace(r'\midrule', r'\hline').replace(r'\bottomrule', r'\hline')
        
        # Print the LaTeX formatted table if desired
        if displaying_latex:
            print("LaTeX formatted table for Advantage_hat:")
            print(latex_table_hat)
            
            print("LaTeX formatted table for Advantage_tilde:")
            print(latex_table_tilde)
        
        return original_results_hat, original_results_tilde, latex_table_hat, latex_table_tilde, buf_hat, buf_tilde
    
    def significance_stars(self, p_value):
        """
        Assign significance stars based on p-value.
        
        Parameters
        ----------
        p_value : float
            The p-value for statistical significance.
            
        Returns
        -------
        str
            Stars representing the level of significance ('***' for p < 0.01, '**' for p < 0.05, '*' for p < 0.1).
        """
        if p_value < 0.01:
            return "***"
        elif p_value < 0.05:
            return "**"
        elif p_value < 0.1:
            return "*"
        else:
            return ""
    
    def compute_statistics(self, Advantage):
        """
        Compute statistics related to advantage function including standard error, t-statistic, and p-values.

        Parameters
        ----------
        Advantage : array-like
            Computed advantage values.

        Returns
        -------
        tuple
            Stat, p-value, upper p-value, lower p-value.
        """
        # Perform the Wilcoxon signed-rank test for two-sided, upper, and lower alternatives
        stat, p_value = wilcoxon(Advantage, alternative='two-sided')
        _, p_value_upper = wilcoxon(Advantage, alternative='greater')
        _, p_value_lower = wilcoxon(Advantage, alternative='less')
        
        return stat[0], p_value[0], p_value_upper[0], p_value_lower[0]
    
    def scientific_to_latex(self, value):
        """
        Convert a numerical value to a LaTeX string in scientific notation.

        Parameters
        ----------
        value : float
            The value to convert.

        Returns
        -------
        str
            The LaTeX formatted string.
        """
        if value == 0:
            return "$0$"
        else:
            exp = int(np.floor(np.log10(np.abs(value))))

            # Return the base directly if the exponent is 0
            if exp == 0:
                return f"${value:.3f}$"
            
            base = value / 10**exp

            return f"${base:.3f} \\times 10^{{{exp}}}$"
        
   # Function to convert a string to a tuple of floats
    def convert_string_to_tuple(self, s):
        return ast.literal_eval(s)

    # Function to convert a tuple to LaTeX formatted strings
    def format_tuple_to_latex(self, values):
        latex_values = [self.scientific_to_latex(float(value)) for value in values]
        return f"({', '.join(latex_values)})"

    
    def combine_pval_stars(self, row, pval_col, stars_col):
        return f"{row[pval_col]} {row[stars_col]}"


    def create_latex_table(self, formatted_results):
        """
        Create a DataFrame from formatted results to generate a LaTeX table.

        Parameters
        ----------
        formatted_results : tuple
            Tuple containing formatted statistical results.

        Returns
        -------
        DataFrame
            DataFrame formatted for LaTeX output.
        """
        # Convert results to a NumPy array
        results_array = np.array(formatted_results)
        # Create DataFrame of results
        index_names = ['$W$', 'p-val (2-sided)', 'adj. p-val (2-sided)', 'p-val (upper)', 'adj. p-val (upper)', 'p-val (lower)', 'adj. p-val (lower)', 'stars (2-sided)', 'adj. stars (2-sided)','stars (upper)', 'adj. stars (upper)', 'stars (lower)', 'adj. stars (lower)']
        df = pd.DataFrame(results_array, index=index_names)
        df = df.T
        
        # Convert numerical columns to LaTeX scientific notation
        for col in ['$W$', 'p-val (2-sided)', 'adj. p-val (2-sided)', 'p-val (upper)', 'adj. p-val (upper)', 'p-val (lower)', 'adj. p-val (lower)']:
            df[col] = df[col].astype(float).apply(self.scientific_to_latex)
            
        # Combine p-values and stars for various significance tests
        df['p-val (2-sided)'] = df.apply(lambda row: self.combine_pval_stars(row, 'p-val (2-sided)', 'stars (2-sided)'), axis=1)
        df['p-val (upper)'] = df.apply(lambda row: self.combine_pval_stars(row, 'p-val (upper)', 'stars (upper)'), axis=1)
        df['p-val (lower)'] = df.apply(lambda row: self.combine_pval_stars(row, 'p-val (lower)', 'stars (lower)'), axis=1)
        
        df['adj. p-val (2-sided)'] = df.apply(lambda row: self.combine_pval_stars(row, 'adj. p-val (2-sided)', 'adj. stars (2-sided)'), axis=1)
        df['adj. p-val (upper)'] = df.apply(lambda row: self.combine_pval_stars(row, 'adj. p-val (upper)', 'adj. stars (upper)'), axis=1)
        df['adj. p-val (lower)'] = df.apply(lambda row: self.combine_pval_stars(row, 'adj. p-val (lower)', 'adj. stars (lower)'), axis=1)

        # Drop the individual stars columns as they are no longer needed
        df = df.drop(columns=['stars (2-sided)', 'adj. stars (2-sided)','stars (upper)', 'adj. stars (upper)', 'stars (lower)', 'adj. stars (lower)'])

        return df


class HacTest:
    def __init__(self, est_Q_actual, est_V):
        """
        Initialize the HacTest class for causal inference testing in time series.

        This class implements the Tau testing framework to assess the causal effects
        of taking actual actions versus not taking them, using state-action value functions.

        Parameters
        ----------
        est_Q_actual : array-like
            Estimated values of the actual state-action quality function, reflecting the 
            outcomes of actions taken in the time series.
        est_V : array-like
            Estimated values of the state value function, which follows the same policy as 
            est_Q_actual, representing the expected outcomes of states under the current policy.
        """
        # Store the actual estimated quality function values
        self.est_Q_actual = est_Q_actual
        
        # Store the estimated state value function values  
        self.est_V = est_V

    def __call__(self, displaying = True, displaying_latex = False):
        """
        Execute the HacTest by computing tau, performing statistical tests,
        and printing results, including t-statistics, p-values, and confidence intervals.
        This testing incorporates HAC standard errors for robust inference.

        Parameters
        ----------
        displaying : bool
            Whether to display the histogram of tau values.
        displaying_latex : bool
            Whether to print the LaTeX formatted results.
        """
        # Compute the advantage values (difference between actual and counterfactual quality functions)
        self.Advantage = self.est_Q_actual- self.est_V
        
        # Determine max lag for HAC based on sample size
        max_lag = int(4 * ((self.Advantage.shape[0] / 100) ** (1 / 3)))

        # Compute statistics for tau, including HAC standard errors
        self.coefficient, self.t_stat, self.p_value, self.hac_std_error, self.p_value_upper, self.p_value_lower = self.compute_statistics(self.Advantage, max_lag)
        
        # Print coefficient, t-statistic, and p-value
        print(f"\nCoefficient: {self.coefficient}")
        print(f"t-statistic: {self.t_stat}")
        print(f"p-value (two-tailed test): {self.p_value}")
        print(f"p-value (upper-tailed test): {self.p_value_upper}")
        print(f"p-value (lower-tailed test): {self.p_value_lower}")
        
        # Assign significance stars based on p-values
        stars_two_sided = self.significance_stars(self.p_value)
        stars_upper = self.significance_stars(self.p_value_upper)
        stars_lower = self.significance_stars(self.p_value_lower)
        
        # Compute 95% confidence interval using HAC standard errors
        alpha = 0.05
        critical_value = stats.t.ppf(1 - alpha / 2, df=self.Advantage.shape[0] - 1)  # Two-tailed critical value
        lower_bound = self.coefficient - critical_value * self.hac_std_error # Lower bound of CI
        upper_bound = self.coefficient + critical_value * self.hac_std_error # Upper bound of CI
        conf_interval = (lower_bound, upper_bound) # Confidence interval
        
        print(f"\n95% Confidence Interval for the coefficient: {conf_interval}")
        
        # Two-tailed hypothesis test for mean different from 0
        print("\nTwo-tailed test (whether the mean is different from 0):")
        if self.p_value < alpha:
            print(">> Reject the null hypothesis. The sample mean is considered to be significantly different from 0.\n")
        else:
            print(">> Fail to reject the null hypothesis.\n")

        # One-tailed tests setup
        print("Upper-tailed test (whether the mean is greater than 0):")
        if self.t_stat > 0 and self.p_value_upper < alpha:
            print(">> Reject the null hypothesis. The sample mean is considered to be greater than 0.\n")
        else:
            print(">> Fail to reject the null hypothesis.\n")
            
        print("Lower-tailed test (whether the mean is less than 0):")
        if self.t_stat < 0 and self.p_value_lower < alpha:
            print(">> Reject the null hypothesis. The sample mean is considered to be less than 0.\n")
        else:
            print(">> Fail to reject the null hypothesis.\n")
        
        # Plot histogram of the advantage values
        percentile_lower_bound = np.percentile(self.Advantage, 2.5)  # Lower 2.5 percentile
        percentile_upper_bound = np.percentile(self.Advantage, 97.5)  # Upper 97.5 percentile

        # Filter data within the specified percentiles for the histogram
        filtered_data = self.Advantage[(self.Advantage >= percentile_lower_bound) & (self.Advantage <= percentile_upper_bound)]
        
        
        if displaying:
            fig = plt.figure(figsize=(8, 6))
            self.ax1 = fig.add_subplot(111) # Create a subplot for the histogram
            self.ax1.hist(filtered_data, bins=100, edgecolor='k') # Plot histogram with 100 bins
            self.ax1.set_xlabel(r'$A_{\pi}(S_{t}, A_{t};w, \hat{\theta})$', fontsize=30) # Set x-label
            self.ax1.set_ylabel('Frequency', fontsize=30) # Set y-label

            # Increase the size of tick labels
            self.ax1.tick_params(axis='both', labelsize=24) # Increase tick label size

            plt.show()  # Show the plot if desired
        
            # Save plot to a BytesIO object 
            buf = io.BytesIO() # Create a buffer to save the plot
            plt.savefig(buf, format='eps', transparent=False, bbox_inches='tight') # Save the figure to the buffer
            buf.seek(0)  # Move to the beginning of the buffer

            # Close the plot to free memory
            #plt.close(fig)

            # Check the buffer size
            print(f"Buffer size: {buf.getbuffer().nbytes} bytes") # Print the size of the buffer
        
        else: buf=None

        
        # Format the results into a tuple for display
        formatted_results = (
            f"{self.coefficient:.4g}",
            f"{self.hac_std_error:.4g}",
            f"{self.t_stat:.4g}",
            f"{self.p_value:.4g}",
            f"{self.p_value_upper:.4g}",
            f"{self.p_value_lower:.4g}",
            f"({lower_bound:.4g}, {upper_bound:.4g})",
            stars_two_sided,
            stars_upper,
            stars_lower
        )

        # Store original numeric results in a dictionary
        original_results = {
            "coef_estimate": self.coefficient,
            "hac_std_error": self.hac_std_error,
            "t_value": self.t_stat,
            "p_value_two_sided": self.p_value,
            "p_value_upper": self.p_value_upper,
            "p_value_lower": self.p_value_lower,
            "95%_confidence_interval": {
                "lower_bound": lower_bound,
                "upper_bound": upper_bound
            },
            "stars_two_sided": stars_two_sided,
            "stars_upper": stars_upper,
            "stars_lower": stars_lower
        }
        
        # Create a LaTeX table from the formatted results
        df = self.create_latex_table(formatted_results) 
        # Create LaTeX table format
        latex_table = df.to_latex(index=True, escape=False).replace(r'\toprule', r'\hline').replace(r'\midrule', r'\hline').replace(r'\bottomrule', r'\hline')
        
        # Print the LaTeX formatted table if desired
        if displaying_latex:
            print("LaTeX formatted table:")
            print(latex_table)
        
        return original_results, latex_table, buf
    
    def significance_stars(self, p_value):
        """
        Assign significance stars based on p-value.
        
        Parameters
        ----------
        p_value : float
            The p-value for statistical significance.
            
        Returns
        -------
        str
            Stars representing the level of significance ('***' for p < 0.01, '**' for p < 0.05, '*' for p < 0.1).
        """
        if p_value < 0.01:
            return "***"
        elif p_value < 0.05:
            return "**"
        elif p_value < 0.1:
            return "*"
        else:
            return ""
    
    def compute_statistics(self, advantage, max_lag):
        """
        Compute statistics related to tau including standard error, t-statistic, and p-values.

        Parameters
        ----------
        advantage : array-like
            Computed tau values (advantage function values).
        max_lag : int
            Maximum lag for the HAC calculation.

        Returns
        -------
        tuple
            Coefficient, t-statistic, p-value, HAC standard error, upper p-value, lower p-value.
        """
        # Prepare the design matrix for OLS regression (constant term)
        X = np.ones_like(advantage)

        # Fit an OLS model to the advantage with HAC standard errors
        model = sm.OLS(advantage, X)
        results = model.fit(cov_type='HAC', cov_kwds={'maxlags': max_lag})
        
        # Extract results
        coefficient = results.params[0]  # Extract coefficient (mean)
        t_stat = results.tvalues[0]      # Extract t-statistic
        p_value = results.pvalues[0]     # Extract p-value (two-tailed)
        hac_std_error = results.bse[0]   # Extract HAC standard error

        # Calculate p-values for upper and lower one-tailed tests
        p_value_upper = 1 - stats.norm.cdf(t_stat)  # Upper-tailed p-value
        p_value_lower = stats.norm.cdf(t_stat)      # Lower-tailed p-value
        
        return coefficient, t_stat, p_value, hac_std_error, p_value_upper, p_value_lower
    
    def scientific_to_latex(self, value):
        """
        Convert a numerical value to a LaTeX string in scientific notation.

        Parameters
        ----------
        value : float
            The value to convert.

        Returns
        -------
        str
            The LaTeX formatted string.
        """
        if value == 0:
            return "$0$"
        else:
            exp = int(np.floor(np.log10(np.abs(value))))

            # Return the base directly if the exponent is 0
            if exp == 0:
                return f"${value:.3f}$"
            
            base = value / 10**exp

            return f"${base:.3f} \\times 10^{{{exp}}}$"
        
   # Function to convert a string to a tuple of floats
    def convert_string_to_tuple(self, s):
        return ast.literal_eval(s)

    # Function to convert a tuple to LaTeX formatted strings
    def format_tuple_to_latex(self, values):
        latex_values = [self.scientific_to_latex(float(value)) for value in values]
        return f"({', '.join(latex_values)})"

    
    def combine_pval_stars(self, row, pval_col, stars_col):
        return f"{row[pval_col]} {row[stars_col]}"


    def create_latex_table(self, formatted_results):
        """
        Create a DataFrame from formatted results to generate a LaTeX table.

        Parameters
        ----------
        formatted_results : tuple
            Tuple containing formatted statistical results.

        Returns
        -------
        DataFrame
            DataFrame formatted for LaTeX output.
        """
        # Convert results to a NumPy array
        results_array = np.array(formatted_results)
        # Create DataFrame of results
        index_names = ["Coef", "HAC SE", "t-val", "p-val (2-sided)", "p-val (upper)", "p-val (lower)", "95\% CI", 'stars (2-sided)', 'stars (upper)', 'stars (lower)']
        df = pd.DataFrame(results_array, index=index_names)
        df = df.T
        
        # Convert numerical columns to LaTeX scientific notation
        for col in ["Coef", "HAC SE", "t-val", "p-val (2-sided)", "p-val (upper)", "p-val (lower)"]:
            df[col] = df[col].astype(float).apply(self.scientific_to_latex)
            
        # Convert strings to tuples for confidence intervals
        df["95\% CI"] = df["95\% CI"].apply(self.convert_string_to_tuple)
        # Apply the formatting function to the "95% CI" column
        df["95\% CI"] = df["95\% CI"].apply(self.format_tuple_to_latex)
        
        # Combine p-values and stars for various significance tests
        df['p-val (2-sided)'] = df.apply(lambda row: self.combine_pval_stars(row, 'p-val (2-sided)', 'stars (2-sided)'), axis=1)
        df['p-val (upper)'] = df.apply(lambda row: self.combine_pval_stars(row, 'p-val (upper)', 'stars (upper)'), axis=1)
        df['p-val (lower)'] = df.apply(lambda row: self.combine_pval_stars(row, 'p-val (lower)', 'stars (lower)'), axis=1)

        # Drop the individual stars columns as they are no longer needed
        df = df.drop(columns=['stars (upper)', 'stars (lower)', 'stars (2-sided)'])

        return df
        
        
        
class WilcoxonSignedRankTest:
    def __init__(self, est_Q_actual, est_V):
        """
        Initialize the WilcoxonSignedRankTest class for causal inference testing in time series.

        This class implements the Wilcoxon signed-rank test framework to assess the causal effects
        of taking actual actions versus not taking them, using state-action quality functions.

        Parameters
        ----------
        est_Q_actual : array-like
            Estimated values of the actual state-action quality function, reflecting the 
            outcomes of actions taken in the time series.
        est_V : array-like
            Estimated values of the state value function, which follows the same policy as 
            est_Q_actual, representing the expected outcomes of states under the current policy.
        """
        # Store the actual estimated quality function values
        self.est_Q_actual = est_Q_actual
        
        # Store the estimated state value function values
        self.est_V = est_V

    def __call__(self, displaying = True, displaying_latex = False):
        """
        Execute the Wilcoxon signed-rank test by computing tau, performing statistical tests,
        and printing results, including t-statistics, p-values, and confidence intervals.
        This testing incorporates HAC standard errors for robust inference.
        """
        # Compute the advantage values (difference between actual Q values and estimated V values)
        self.Advantage = self.est_Q_actual- self.est_V
        
        # Compute the test statistics for the Wilcoxon signed-rank test
        self.stat, self.p_value, self.stat_upper, self.p_value_upper, self.stat_lower, self.p_value_lower = self.compute_statistics(self.Advantage)
        
        
        # Print statistics and p-values for the two-tailed test
        print(f"\nWilcoxon-statistic (two-tailed test): {self.stat}")
        print(f"p-value (two-tailed test): {self.p_value}")
        
        # Print statistics and p-values for upper-tailed test
        print(f"\nWilcoxon-statistic (upper-tailed test): {self.stat_upper}")
        print(f"p-value (upper-tailed test): {self.p_value_upper}")
        
        # Print statistics and p-values for lower-tailed test
        print(f"\nWilcoxon-statistic (lower-tailed test): {self.stat_lower}")
        print(f"p-value (lower-tailed test): {self.p_value_lower}")
         
        # Assign significance stars based on p-values
        stars_two_sided = self.significance_stars(self.p_value)
        stars_upper = self.significance_stars(self.p_value_upper)
        stars_lower = self.significance_stars(self.p_value_lower)
        
        # Significance level
        alpha = 0.05

        # Two-tailed Wilcoxon test for the hypothesis of median different from 0
        print("\nTwo-tailed test (whether the median is different from 0):")
        if self.p_value < alpha:
            print(">> Reject the null hypothesis. The sample median is considered to be significantly different from 0.\n")
        else:
            print(">> Fail to reject the null hypothesis.\n")

        # One-tailed Wilcoxon tests setup
        print("Upper-tailed test (whether the median is greater than 0):")
        if self.p_value_upper < alpha:
            print(">> Reject the null hypothesis. The sample median is considered to be greater than 0.\n")
        else:
            print(">> Fail to reject the null hypothesis.\n")

        print("Lower-tailed test (whether the median is less than 0):")
        if self.p_value_lower < alpha:
            print(">> Reject the null hypothesis. The sample median is considered to be less than 0.\n")
        else:
            print(">> Fail to reject the null hypothesis.\n")
        
        # Plot histogram of the advantage values
        percentile_lower_bound = np.percentile(self.Advantage, 2.5)  # Lower 2.5 percentile
        percentile_upper_bound = np.percentile(self.Advantage, 97.5)  # Upper 97.5 percentile

        # Filter data within the specified percentiles for the histogram
        filtered_data = self.Advantage[(self.Advantage >= percentile_lower_bound) & (self.Advantage <= percentile_upper_bound)]
        
        
        if displaying:
            fig = plt.figure(figsize=(8, 6))
            self.ax1 = fig.add_subplot(111) # Create a subplot for the histogram
            self.ax1.hist(filtered_data, bins=100, edgecolor='k') # Plot histogram with 100 bins
            self.ax1.set_xlabel(r'$A_{\pi}(S_{t}, A_{t};w, \hat{\theta})$', fontsize=30) # Set x-label
            self.ax1.set_ylabel('Frequency', fontsize=30) # Set y-label

            # Increase the size of tick labels
            self.ax1.tick_params(axis='both', labelsize=24) # Increase tick label size

            plt.show()  # Show the plot if desired
        
            # Save plot to a BytesIO object 
            buf = io.BytesIO() # Create a buffer to save the plot
            plt.savefig(buf, format='eps', transparent=False, bbox_inches='tight') # Save the figure to the buffer
            buf.seek(0)  # Move to the beginning of the buffer

            # Close the plot to free memory
            #plt.close(fig)

            # Check the buffer size
            print(f"Buffer size: {buf.getbuffer().nbytes} bytes") # Print the size of the buffer
        
        else: buf =None

        
        # Format the results into a tuple for display
        formatted_results = (
            f"{self.stat:.4g}",
            f"{self.p_value:.4g}",
            f"{self.stat_upper:.4g}",
            f"{self.p_value_upper:.4g}",
            f"{self.stat_lower:.4g}",
            f"{self.p_value_lower:.4g}",
            stars_two_sided,
            stars_upper,
            stars_lower
        )
        
        
        # Store original numeric results in a dictionary
        original_results = {
            "W_two_sided": self.stat,
            "p_value_two_sided": self.p_value,
            "W_upper": self.stat_upper,
            "p_value_upper": self.p_value_upper,
            "W_lower": self.stat_lower,
            "p_value_lower": self.p_value_lower,
            "stars_two_sided": stars_two_sided,
            "stars_upper": stars_upper,
            "stars_lower": stars_lower
        }
        
        # Create a LaTeX table from the formatted results
        df = self.create_latex_table(formatted_results) 
        # Create LaTeX table format
        latex_table = df.to_latex(index=True, escape=False).replace(r'\toprule', r'\hline').replace(r'\midrule', r'\hline').replace(r'\bottomrule', r'\hline')
        
        # Print the LaTeX formatted table if desired
        if displaying_latex:
            print("LaTeX formatted table:")
            print(latex_table)
        
        return original_results, latex_table, buf
    
    def significance_stars(self, p_value):
        """
        Assign significance stars based on p-value.
        
        Parameters
        ----------
        p_value : float
            The p-value for statistical significance.
            
        Returns
        -------
        str
            Stars representing the level of significance ('***' for p < 0.01, '**' for p < 0.05, '*' for p < 0.1).
        """
        if p_value < 0.01:
            return "***"
        elif p_value < 0.05:
            return "**"
        elif p_value < 0.1:
            return "*"
        else:
            return ""
    
    def compute_statistics(self, Advantage):
        """
        Compute statistics related to tau including standard error, t-statistic, and p-values.

        Parameters
        ----------
        Advantage : array-like
            Computed advantage values.

        Returns
        -------
        tuple
            Stat, p-value, upper stat, upper p-value, lower stat, lower p-value.
        """
        # Perform the Wilcoxon signed-rank test for two-sided, upper, and lower alternatives
        stat, p_value = wilcoxon(Advantage, alternative='two-sided')
        stat_upper, p_value_upper = wilcoxon(Advantage, alternative='greater')
        stat_lower, p_value_lower = wilcoxon(Advantage, alternative='less')
        
        return stat[0], p_value[0], stat_upper[0], p_value_upper[0], stat_lower[0], p_value_lower[0]
    
    def scientific_to_latex(self, value):
        """
        Convert a numerical value to a LaTeX string in scientific notation.

        Parameters
        ----------
        value : float
            The value to convert.

        Returns
        -------
        str
            The LaTeX formatted string.
        """
        if value == 0:
            return "$0$"
        else:
            exp = int(np.floor(np.log10(np.abs(value))))

            # Return the base directly if the exponent is 0
            if exp == 0:
                return f"${value:.3f}$"
            
            base = value / 10**exp

            return f"${base:.3f} \\times 10^{{{exp}}}$"
        
   # Function to convert a string to a tuple of floats
    def convert_string_to_tuple(self, s):
        return ast.literal_eval(s)

    # Function to convert a tuple to LaTeX formatted strings
    def format_tuple_to_latex(self, values):
        latex_values = [self.scientific_to_latex(float(value)) for value in values]
        return f"({', '.join(latex_values)})"

    
    def combine_pval_stars(self, row, pval_col, stars_col):
        return f"{row[pval_col]} {row[stars_col]}"


    def create_latex_table(self, formatted_results):
        """
        Create a DataFrame from formatted results to generate a LaTeX table.

        Parameters
        ----------
        formatted_results : tuple
            Tuple containing formatted statistical results.

        Returns
        -------
        DataFrame
            DataFrame formatted for LaTeX output.
        """
        # Convert results to a NumPy array
        results_array = np.array(formatted_results)
        # Create DataFrame of results
        index_names = ['$W$ (2-sided)', 'p-val (2-sided)', '$W$ (upper)', 'p-val (upper)', '$W$ (lower)', 'p-val (lower)', 'stars (2-sided)', 'stars (upper)', 'stars (lower)']
        df = pd.DataFrame(results_array, index=index_names)
        df = df.T
        
        # Convert numerical columns to LaTeX scientific notation
        for col in ['$W$ (2-sided)', 'p-val (2-sided)', '$W$ (upper)', 'p-val (upper)', '$W$ (lower)', 'p-val (lower)']:
            df[col] = df[col].astype(float).apply(self.scientific_to_latex)
            
        # Combine p-values and stars for various significance tests
        df['p-val (2-sided)'] = df.apply(lambda row: self.combine_pval_stars(row, 'p-val (2-sided)', 'stars (2-sided)'), axis=1)
        df['p-val (upper)'] = df.apply(lambda row: self.combine_pval_stars(row, 'p-val (upper)', 'stars (upper)'), axis=1)
        df['p-val (lower)'] = df.apply(lambda row: self.combine_pval_stars(row, 'p-val (lower)', 'stars (lower)'), axis=1)

        # Drop the individual stars columns as they are no longer needed
        df = df.drop(columns=['stars (upper)', 'stars (lower)', 'stars (2-sided)'])

        return df
        

        
class ARMAWilcoxonSignedRankTest:
    def __init__(self, est_Q_actual, est_V):
        """
        Initialize the WilcoxonSignedRankTest class for causal inference testing in time series.

        This class implements the Wilcoxon signed-rank test framework to assess the causal effects
        of taking actual actions versus not taking them, using state-action quality functions.

        Parameters
        ----------
        est_Q_actual : array-like
            Estimated values of the actual state-action quality function, reflecting the 
            outcomes of actions taken in the time series.
        est_V : array-like
            Estimated values of the state value function, which follows the same policy as 
            est_Q_actual, representing the expected outcomes of states under the current policy.
        """
        # Store the actual estimated quality function values
        self.est_Q_actual = est_Q_actual
        
        # Store the estimated state value function values
        self.est_V = est_V

    def __call__(self, displaying = True, displaying_latex = False):
        """
        Execute the Wilcoxon signed-rank test by computing tau, performing statistical tests,
        and printing results, including t-statistics, p-values, and confidence intervals.
        This testing incorporates HAC standard errors for robust inference.
        """
        # Compute the advantage values (difference between actual Q values and estimated V values)
        self.Advantage = self.est_Q_actual- self.est_V
        
        # Compute the test statistics for the Wilcoxon signed-rank test
        self.stat, self.p_value, self.stat_upper, self.p_value_upper, self.stat_lower, self.p_value_lower = self.compute_statistics(self.Advantage)
        
        
        # Print statistics and p-values for the two-tailed test
        print(f"\nWilcoxon-statistic (two-tailed test): {self.stat}")
        print(f"p-value (two-tailed test): {self.p_value}")
        
        # Print statistics and p-values for upper-tailed test
        print(f"\nWilcoxon-statistic (upper-tailed test): {self.stat_upper}")
        print(f"p-value (upper-tailed test): {self.p_value_upper}")
        
        # Print statistics and p-values for lower-tailed test
        print(f"\nWilcoxon-statistic (lower-tailed test): {self.stat_lower}")
        print(f"p-value (lower-tailed test): {self.p_value_lower}")
         
        # Assign significance stars based on p-values
        stars_two_sided = self.significance_stars(self.p_value)
        stars_upper = self.significance_stars(self.p_value_upper)
        stars_lower = self.significance_stars(self.p_value_lower)
        
        # Significance level
        alpha = 0.05

        # Two-tailed Wilcoxon test for the hypothesis of median different from 0
        print("\nTwo-tailed test (whether the median is different from 0):")
        if self.p_value < alpha:
            print(">> Reject the null hypothesis. The sample median is considered to be significantly different from 0.\n")
        else:
            print(">> Fail to reject the null hypothesis.\n")

        # One-tailed Wilcoxon tests setup
        print("Upper-tailed test (whether the median is greater than 0):")
        if self.p_value_upper < alpha:
            print(">> Reject the null hypothesis. The sample median is considered to be greater than 0.\n")
        else:
            print(">> Fail to reject the null hypothesis.\n")

        print("Lower-tailed test (whether the median is less than 0):")
        if self.p_value_lower < alpha:
            print(">> Reject the null hypothesis. The sample median is considered to be less than 0.\n")
        else:
            print(">> Fail to reject the null hypothesis.\n")
        
        # Plot histogram of the advantage values
        percentile_lower_bound = np.percentile(self.Advantage, 2.5)  # Lower 2.5 percentile
        percentile_upper_bound = np.percentile(self.Advantage, 97.5)  # Upper 97.5 percentile

        # Filter data within the specified percentiles for the histogram
        filtered_data = self.Advantage[(self.Advantage >= percentile_lower_bound) & (self.Advantage <= percentile_upper_bound)]
        
        
        if displaying:
            fig = plt.figure(figsize=(8, 6))
            self.ax1 = fig.add_subplot(111) # Create a subplot for the histogram
            self.ax1.hist(filtered_data, bins=100, edgecolor='k') # Plot histogram with 100 bins
            self.ax1.set_xlabel(r'$A_{\pi}(S_{t}, A_{t};w, \hat{\theta})$', fontsize=30) # Set x-label
            self.ax1.set_ylabel('Frequency', fontsize=30) # Set y-label

            # Increase the size of tick labels
            self.ax1.tick_params(axis='both', labelsize=24) # Increase tick label size

            plt.show()  # Show the plot if desired
        
            # Save plot to a BytesIO object 
            buf = io.BytesIO() # Create a buffer to save the plot
            plt.savefig(buf, format='eps', transparent=False, bbox_inches='tight') # Save the figure to the buffer
            buf.seek(0)  # Move to the beginning of the buffer

            # Close the plot to free memory
            #plt.close(fig)

            # Check the buffer size
            print(f"Buffer size: {buf.getbuffer().nbytes} bytes") # Print the size of the buffer
        
        else: buf =None

        
        # Format the results into a tuple for display
        formatted_results = (
            f"{self.stat:.4g}",
            f"{self.p_value:.4g}",
            f"{self.stat_upper:.4g}",
            f"{self.p_value_upper:.4g}",
            f"{self.stat_lower:.4g}",
            f"{self.p_value_lower:.4g}",
            stars_two_sided,
            stars_upper,
            stars_lower
        )
        
        
        # Store original numeric results in a dictionary
        original_results = {
            "W_two_sided": self.stat,
            "p_value_two_sided": self.p_value,
            "W_upper": self.stat_upper,
            "p_value_upper": self.p_value_upper,
            "W_lower": self.stat_lower,
            "p_value_lower": self.p_value_lower,
            "stars_two_sided": stars_two_sided,
            "stars_upper": stars_upper,
            "stars_lower": stars_lower
        }
        
        # Create a LaTeX table from the formatted results
        df = self.create_latex_table(formatted_results) 
        # Create LaTeX table format
        latex_table = df.to_latex(index=True, escape=False).replace(r'\toprule', r'\hline').replace(r'\midrule', r'\hline').replace(r'\bottomrule', r'\hline')
        
        # Print the LaTeX formatted table if desired
        if displaying_latex:
            print("LaTeX formatted table:")
            print(latex_table)
        
        return original_results, latex_table, buf
    
    def significance_stars(self, p_value):
        """
        Assign significance stars based on p-value.
        
        Parameters
        ----------
        p_value : float
            The p-value for statistical significance.
            
        Returns
        -------
        str
            Stars representing the level of significance ('***' for p < 0.01, '**' for p < 0.05, '*' for p < 0.1).
        """
        if p_value < 0.01:
            return "***"
        elif p_value < 0.05:
            return "**"
        elif p_value < 0.1:
            return "*"
        else:
            return ""
    
    def compute_statistics(self, Advantage):
        """
        Compute statistics related to tau including standard error, t-statistic, and p-values.

        Parameters
        ----------
        Advantage : array-like
            Computed advantage values.

        Returns
        -------
        tuple
            Stat, p-value, upper stat, upper p-value, lower stat, lower p-value.
        """
        # Fit an AR(1) model to the Advantage data to capture the autoregressive structure
        model = ARIMA(Advantage, order=(1, 0, 0))  # Specify AR(1) model
        results = model.fit()

        # Obtain the residuals
        residuals = results.resid

        # Calculate the differences from zero
        differences = residuals - 0
        
        # Perform the Wilcoxon signed-rank test for two-sided, upper, and lower alternatives
        stat, p_value = wilcoxon(differences, alternative='two-sided')
        stat_upper, p_value_upper = wilcoxon(differences, alternative='greater')
        stat_lower, p_value_lower = wilcoxon(differences, alternative='less')
        
        return stat, p_value, stat_upper, p_value_upper, stat_lower, p_value_lower
    
    def scientific_to_latex(self, value):
        """
        Convert a numerical value to a LaTeX string in scientific notation.

        Parameters
        ----------
        value : float
            The value to convert.

        Returns
        -------
        str
            The LaTeX formatted string.
        """
        if value == 0:
            return "$0$"
        else:
            exp = int(np.floor(np.log10(np.abs(value))))

            # Return the base directly if the exponent is 0
            if exp == 0:
                return f"${value:.3f}$"
            
            base = value / 10**exp

            return f"${base:.3f} \\times 10^{{{exp}}}$"
        
   # Function to convert a string to a tuple of floats
    def convert_string_to_tuple(self, s):
        return ast.literal_eval(s)

    # Function to convert a tuple to LaTeX formatted strings
    def format_tuple_to_latex(self, values):
        latex_values = [self.scientific_to_latex(float(value)) for value in values]
        return f"({', '.join(latex_values)})"

    
    def combine_pval_stars(self, row, pval_col, stars_col):
        return f"{row[pval_col]} {row[stars_col]}"


    def create_latex_table(self, formatted_results):
        """
        Create a DataFrame from formatted results to generate a LaTeX table.

        Parameters
        ----------
        formatted_results : tuple
            Tuple containing formatted statistical results.

        Returns
        -------
        DataFrame
            DataFrame formatted for LaTeX output.
        """
        # Convert results to a NumPy array
        results_array = np.array(formatted_results)
        # Create DataFrame of results
        index_names = ['$W$ (2-sided)', 'p-val (2-sided)', '$W$ (upper)', 'p-val (upper)', '$W$ (lower)', 'p-val (lower)', 'stars (2-sided)', 'stars (upper)', 'stars (lower)']
        df = pd.DataFrame(results_array, index=index_names)
        df = df.T
        
        # Convert numerical columns to LaTeX scientific notation
        for col in ['$W$ (2-sided)', 'p-val (2-sided)', '$W$ (upper)', 'p-val (upper)', '$W$ (lower)', 'p-val (lower)']:
            df[col] = df[col].astype(float).apply(self.scientific_to_latex)
            
        # Combine p-values and stars for various significance tests
        df['p-val (2-sided)'] = df.apply(lambda row: self.combine_pval_stars(row, 'p-val (2-sided)', 'stars (2-sided)'), axis=1)
        df['p-val (upper)'] = df.apply(lambda row: self.combine_pval_stars(row, 'p-val (upper)', 'stars (upper)'), axis=1)
        df['p-val (lower)'] = df.apply(lambda row: self.combine_pval_stars(row, 'p-val (lower)', 'stars (lower)'), axis=1)

        # Drop the individual stars columns as they are no longer needed
        df = df.drop(columns=['stars (upper)', 'stars (lower)', 'stars (2-sided)'])

        return df
        
        
        

# For cal_values using "CalculateValues2"
def SensitivityAnalysis2(cal_values, w_values=[10, 100, 1000, np.inf], gamma_values=np.linspace(0, 1, 4)):
    # List to store results
    hac_p_values = []  # p-values for HAC test
    wilcoxon_p_values = []  # p-values for Wilcoxon test

    for w in w_values:
        for gamma in gamma_values:
            # Perform VQ calculation based on weight (w)
            if w == np.inf:
                original_results, df, latex_table, est_V, est_Q, est_Q_actual, est_pi_actual = cal_values.calculate_VQ_inf(gamma)
            else:
                original_results, df, latex_table, est_V, est_Q, est_Q_actual, est_pi_actual = cal_values.calculate_VQ_w(w, gamma)

            # HAC test
            hac_testing = HacTest(est_Q_actual, est_V)
            results_hac, _, _ = hac_testing(displaying=False, displaying_latex=False)
            hac_p_value = results_hac['p_value_two_sided']  # Extract two-sided p-value
            hac_p_values.append((w, gamma, hac_p_value))  # Append results

            # Wilcoxon Signed Rank Test
            wil_testing = WilcoxonSignedRankTest(est_Q_actual, est_V)
            results_wil, _, _ = wil_testing(displaying=False, displaying_latex=False)
            wilcoxon_p_value = results_wil['p_value_two_sided']  # Extract two-sided p-value
            wilcoxon_p_values.append((w, gamma, wilcoxon_p_value))  # Append results

    # Convert p-values to numpy arrays for easier manipulation
    hac_p_values = np.array(hac_p_values)
    wilcoxon_p_values = np.array(wilcoxon_p_values)

    ## Prepare for plotting
    plt.figure(figsize=(8, 6))

    # Plot HAC test p-values
    plt.subplot(1, 1, 1)
    for w in w_values:
        subset = hac_p_values[hac_p_values[:, 0] == w]  # Filter p-values for current w
        plt.plot(subset[:, 1], subset[:, 2], marker='o', label=f'w={w}')  # Plot against gamma
    plt.axhline(0.05, color='red', linestyle='--', label='Significance Level (0.05)')  # Add significance line
    plt.title('HAC Test p-values vs $\\gamma$', fontsize=20)
    plt.xlabel('$\\gamma$', fontsize=20)
    plt.ylabel('HAC p-value', fontsize=20)
    plt.legend(fontsize=15)
    plt.tick_params(axis='both', labelsize=15)  # Set tick label size
    plt.grid()

    # Save HAC plot to a BytesIO object
    hac_buf = io.BytesIO()
    plt.savefig(hac_buf, format='eps', transparent=False, bbox_inches='tight')
    hac_buf.seek(0)  # Move to the beginning of the buffer

    # Create a new figure for the Wilcoxon test plot
    plt.figure(figsize=(8, 6))

    # Plot Wilcoxon test p-values
    plt.subplot(1, 1, 1)  # Create a single subplot
    for w in w_values:
        subset = wilcoxon_p_values[wilcoxon_p_values[:, 0] == w]  # Filter p-values for current w
        plt.plot(subset[:, 1], subset[:, 2], marker='o', label=f'w={w}')  # Plot against gamma
    plt.axhline(0.05, color='red', linestyle='--', label='Significance Level (0.05)')  # Add significance line
    plt.title('Wilcoxon Test p-values vs $\\gamma$', fontsize=20)
    plt.xlabel('$\\gamma$', fontsize=20)
    plt.ylabel('Wilcoxon p-value', fontsize=20)
    plt.legend(fontsize=15)
    plt.tick_params(axis='both', labelsize=15)  # Set tick label size
    plt.grid()

    # Save Wilcoxon plot to a BytesIO object
    wil_buf = io.BytesIO()
    plt.savefig(wil_buf, format='eps', transparent=False, bbox_inches='tight')
    wil_buf.seek(0)  # Move to the beginning of the buffer

    # Display the plots
    plt.show()
    
    return hac_buf, wil_buf  # Return buffers for HAC and Wilcoxon plots
