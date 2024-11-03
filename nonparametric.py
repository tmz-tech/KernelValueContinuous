#!/usr/bin/env python
# coding: utf-8

#import pandas as pd
#import numpy as np
#from sklearn.base import BaseEstimator

from . import pd, np, BaseEstimator

class KDE(BaseEstimator):
    """
    Kernel Density Estimation (KDE) with Gaussian kernel.

    Parameters
    ----------
    bandwidth : float, optional, default=1.0
        The bandwidth of the kernel.

    Attributes
    ----------
    X_ : array-like, shape (n_samples, n_features)
        The data used for fitting the KDE model.
    """
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth

    def fit(self, X):
        """
        Fit the model using the given data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.X_ = X
        return self

    def score(self, X):
        """
        Calculate the log-likelihood of the data under the model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        score : float
            The sum of the log-likelihoods of the KDE density estimates.
        """
        # Use the sum of the log-likelihoods of the KDE density estimates as the score.
        dens = self.gaussian_kernel_matrix(X, self.bandwidth)
        return np.sum(np.log(dens + 1e-10)) # Add a small value to avoid log(0).

    def gaussian_kernel_matrix(self, data_matrix, h_x):
        """
        Calculate the density estimates for the data using Gaussian kernels.

        Parameters
        ----------
        data_matrix : array-like, shape (n_samples, n_features)
            The input data.
        h_x : float
            The bandwidth for the kernel.

        Returns
        -------
        dens : array-like, shape (n_samples,)
            The density estimates for each sample.
        """
        dens = np.apply_along_axis(self.gaussian_kernel, 1, data_matrix, h_x)
        return dens

    def gaussian_kernel(self, x, h_x):
        """
        Apply the Gaussian kernel to a data point.

        Parameters
        ----------
        x : array-like, shape (n_features,)
            The data point to estimate the density for.
        h_x : float
            The bandwidth for the kernel.

        Returns
        -------
        density : float
            The estimated density for the data point.
        """
        # Broadcasting to subtract x from each row of self.X_
        u = (x - self.X_) / h_x
        # Gaussian kernel, applied elementwise
        K = np.exp(-0.5 * np.sum(u**2, axis=1)) / ((2 * np.pi)**(self.X_.shape[1] / 2) * h_x**self.X_.shape[1])
        density = np.mean(K)
        return density



class KDE2(BaseEstimator):
    """
    Kernel Density Estimation (KDE) with Gaussian kernel.

    Parameters
    ----------
    bandwidth : float, optional, default=1.0
        The bandwidth of the kernel.

    Attributes
    ----------
    X_ : array-like, shape (n_samples, n_features)
        The data used for fitting the KDE model.
    """
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth

    def fit(self, X):
        """
        Fit the model using the given data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.X_ = X
        return self

    def score(self, X):
        """
        Calculate the log-likelihood of the data under the model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        score : float
            The sum of the log-likelihoods of the KDE density estimates.
        """
        # Use the sum of the log-likelihoods of the KDE density estimates as the score.
        dens = self.gaussian_kernel_matrix(X, self.bandwidth)
        return np.sum(np.log(dens + 1e-10))  # Add a small value to avoid log(0).

    def gaussian_kernel_matrix(self, data_matrix, h_x):
        """
        Calculate the density estimates for the data using Gaussian kernels.

        Parameters
        ----------
        data_matrix : array-like, shape (n_samples, n_features)
            The input data.
        h_x : float
            The bandwidth for the kernel.

        Returns
        -------
        dens : array-like, shape (n_samples,)
            The density estimates for each sample.
        """
        dens = np.apply_along_axis(self.gaussian_kernel, 1, data_matrix, h_x)
        return dens

    def gaussian_kernel(self, x, h_x):
        """
        Apply the Gaussian kernel to a data point.

        Parameters
        ----------
        x : array-like, shape (n_features,)
            The data point to estimate the density for.
        h_x : float
            The bandwidth for the kernel.

        Returns
        -------
        density : float
            The estimated density for the data point.
        """
        u = (x - self.X_) / h_x
        K = np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)
        density = np.mean(np.prod(K/h_x, axis=1))
        return density


class est_policy(BaseEstimator):
    
    """
    Estimator for policy using kernel density estimation (KDE) with Gaussian kernel.

    Parameters
    ----------
    bandwidth_x : float, optional, default=0.1
        The bandwidth for the state features in the kernel.
    bandwidth_a : float, optional, default=0.1
        The bandwidth for the action features in the kernel.
    
    Attributes
    ----------
    X_ : array-like, shape (n_samples, n_features)
        The feature matrix used for fitting the model.
    A_ : array-like, shape (n_samples,)
        The action vector corresponding to the feature matrix.
    """
    
    def __init__(self, bandwidth_x=0.1, bandwidth_a = 0.1):
        self.bandwidth_x = bandwidth_x # Bandwidth for state features
        self.bandwidth_a = bandwidth_a # Bandwidth for action features

    def fit(self, X, A):
        """
        Fit the model using the provided state and action data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input state data (features).
        A : array-like, shape (n_samples,)
            The input action data.

        Returns
        -------
        self : object
            Returns the instance itself for chaining.
        """
        self.X_ = X # Store the input states (features)
        self.A_ = A # Store the input actions
        return self

    def pi_est(self, x, a, h_x, h_a, epsilon=1e-10):
        """
        Compute the policy probability for a given state-action pair using KDE.

        Parameters
        ----------
        x : array-like, shape (n_features,)
            A single state data point.
        a : array-like, shape (1,)
            A single action data point.
        h_x : float
            Bandwidth for the state data.
        h_a : float
            Bandwidth for the action data.
        epsilon : float, optional, default=1e-10
            Small value added to the denominator to avoid division by zero.

        Returns
        -------
        density : float
            The estimated policy probability for the given state-action pair.
        """
        # Compute the pairwise differences for the state and action
        u_x = (x - self.X_) / h_x # Difference between input state and fitted states
        u_a = (a - self.A_) / h_a # Difference between input action and fitted actions
        
        # Gaussian kernel computation for state and action
        K_x = np.exp(-0.5 * np.sum(u_x**2, axis=1)) / ((2 * np.pi)**(self.X_.shape[1] / 2) * h_x**self.X_.shape[1])
        K_a = np.exp(-0.5 * np.sum(u_a**2, axis=1)) / ((2 * np.pi)**(self.A_.shape[1] / 2) * h_a**self.A_.shape[1])
        
        # KDE numerator and denominator
        density_num = np.sum(K_x*K_a) # Weighted sum of kernels (joint state-action density)
        density_denom = np.sum(K_x) # Marginal density of states
        
        # Return the estimated policy probability (avoid division by zero)
        density = density_num/(density_denom + epsilon)
        

        return density

    def pi_est_data(self, data_state, data_action):
        """
        Estimate policy probabilities for all data points (state-action pairs).

        Parameters
        ----------
        data_state : array-like, shape (n_samples, n_features)
            The input state data points.
        data_action : array-like, shape (n_samples,)
            The input action data points corresponding to each state.

        Returns
        -------
        pi_est_vec : array-like, shape (n_samples,)
            Vector of estimated policy probabilities for each state-action pair.
        """
        # Ensure that state and action data have the same number of samples
        assert data_state.shape[0] == data_action.shape[0], "data_state and data_action must have the same number of samples"
        
        # Apply the KDE estimator to each state-action pair and return the estimated probabilities
        pi_est_vec = np.array([self.pi_est(data_state[i], data_action[i], self.bandwidth_x, self.bandwidth_a) for i in range(len(data_state))])
        
        return pi_est_vec
    
    
    def score(self, X, A):
        """
        Compute the log-likelihood of the data under the estimated policy model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input state samples.
        A : array-like, shape (n_samples,)
            The input action samples corresponding to each state.

        Returns
        -------
        score : float
            The total log-likelihood of the KDE density estimates for the given data.
        """
        # Compute the log-likelihood (adding a small value to avoid log(0))
        dens = self.pi_est_data(X, A)
        log_likelihood = np.sum(np.log(dens + 1e-10)) 
        return log_likelihood
    
    
    
    


class est_r_pi(BaseEstimator):
    """
    Custom estimator using Nadaraya-Watson kernel regression for estimating a response function.

    Parameters
    ----------
    bandwidth : float, default=1.0
        The bandwidth parameter for the Gaussian kernel, which determines the smoothness of the estimate.

    alpha : float, optional, default=0.1
        The regularization parameter to improve numerical stability and prevent underflow in density calculations.

    Attributes
    ----------
    X_ : ndarray of shape (n_samples, n_features)
        The input data used for fitting the model.
        
    R_ : ndarray of shape (n_samples,)
        The response values associated with the input data X_.
    """
    def __init__(self, bandwidth=1.0, alpha=0.1):
        self.bandwidth = bandwidth # Initialize the bandwidth parameter for the kernel
        self.alpha = alpha # Initialize the regularization parameter for numerical stability

    def fit(self, X, R):
        """
        Fit the model using input data X and responses R.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data points, representing features.
        R : ndarray of shape (n_samples,)
            The response values (targets) corresponding to the input data X.

        Returns
        -------
        self : object
            Fitted estimator instance.
        """
        # Store the input data and response values for later use in prediction
        self.X_ = X # Store input data
        self.R_ = R # Store response values
        return self
    
    def __call__(self, data_state):
        """
        Apply Nadaraya-Watson kernel regression to each row in the data matrix.

        This method uses np.apply_along_axis to apply the kernel regression 
        estimation to each row of the input data (data_state).

        Parameters
        ----------
        data_state : ndarray of shape (n_samples, n_features)
            The state data points for which to estimate the response values.

        Returns
        -------
        nw_est_vec : ndarray of shape (n_samples,)
            Vector of estimated response values for each data point in data_state.
        """
        # Apply Nadaraya-Watson estimation for each row in data_state
        nw_est_vec = np.apply_along_axis(self.nw_est, 1, data_state, self.bandwidth)
        return nw_est_vec
    
    
    def nw_est(self, x, h_x, epsilon=1e-10):
        """
        Nadaraya-Watson kernel regression estimator for a given input state.

        This method computes the kernel-weighted average of the response values 
        based on the Gaussian kernel applied to the distances between the input 
        state x and the training data states self.X_.

        Parameters
        ----------
        x : array-like, shape (n_features,)
            The input state for which the response is estimated.
        h_x : float
            Bandwidth for the state data, controlling the smoothness of the kernel.
        epsilon : float, optional, default=1e-10
            Small constant to avoid division by zero in the denominator.

        Returns
        -------
        nw : float
            The estimated response value for the input state x.
        """
        # Compute the normalized pairwise distances between input x and the training data self.X_
        u_x = (x - self.X_) / h_x # Difference between input state and fitted states

        # Gaussian kernel function applied to the pairwise differences
        K_x = np.exp(-0.5 * np.sum(u_x**2, axis=1)) / ((2 * np.pi)**(self.X_.shape[1] / 2) * h_x**self.X_.shape[1])
        
        K_x += self.alpha # Add regularization to improve stability and avoid numerical issues

        # Compute the numerator: the weighted sum of response values
        nw_num = np.sum(self.R_ * K_x[:, np.newaxis]) # Element-wise multiplication between response values and kernel weights
        # Compute the denominator: the sum of kernel weights (i.e., marginal density)
        nw_denom = np.sum(K_x)
        
        # Calculate the Nadaraya-Watson estimate (handle potential division by zero)
        nw = nw_num/(nw_denom + epsilon)
        
        return nw
    



class est_r_pi_w(BaseEstimator):
    """
    Custom estimator using Nadaraya-Watson kernel regression with a window parameter.

    Parameters
    ----------
    bandwidth : float, default=1.0
        The bandwidth parameter for the Gaussian kernel, controlling the smoothness of the estimate.

    alpha : float, optional, default=0.1
        The regularization parameter to enhance numerical stability and prevent underflow in density calculations.

    Attributes
    ----------
    X_ : ndarray of shape (n_samples, n_features)
        The input data used for fitting the model.
        
    R_ : ndarray of shape (n_samples,)
        The response values associated with the input data X_.

    w : int
        The window parameter that determines the lag between the response and the input data. It should be set during model fitting.

    R_w : ndarray of shape (n_samples - w,)
        The truncated response values aligned with the window parameter, used for estimating the current state based on previous observations.

    X_w : ndarray of shape (n_samples - w, n_features)
        The truncated input data aligned with the window parameter, excluding the last w samples to maintain consistency with R_w.
    """
    def __init__(self, bandwidth=1.0, alpha = 0.1):
        self.bandwidth = bandwidth # Initialize the bandwidth parameter for the kernel
        self.alpha = alpha # Initialize the regularization parameter for numerical stability

    def fit(self, X, R, w):
        """
        Fit the model using input data X, responses R, and a window parameter w.

        The window parameter w introduces a lag between the input data (X) 
        and the response values (R). The input and response data are truncated
        based on the window value to match the lag structure.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data points (e.g., features or states).
        R : ndarray of shape (n_samples,)
            The response values (e.g., target values) corresponding to X.
        w : int
            The window parameter, which specifies the number of steps to lag the response.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Store the input data and response values
        self.X_ = X # Full input data
        self.R_ = R # Full response values
        self.w =w # Store the window parameter
        
        # Adjust the input and response data according to the window parameter
        # Truncate the response values by skipping the first w steps
        self.R_w = self.R_[w:] # Truncated response values
        # Truncate the input data by removing the last w steps
        self.X_w = self.X_[:-w] # Truncated input data
        return self
    
    def nw_est(self, x, h_x, epsilon=1e-10):
        """
        Estimate the function value at a given point using Nadaraya-Watson kernel regression.

        This method computes a weighted average of response values using Gaussian kernels.
        The kernel is applied to the difference between the input point and the fitted input data.
        Both the original and windowed data are used for kernel estimation.

        Parameters
        ----------
        x : ndarray of shape (n_features,)
            The input state for which the response value is estimated.
        h_x : float or ndarray of shape (n_features,)
            The bandwidth parameter(s) for the kernel, controlling the smoothness of the kernel.
        epsilon : float, optional, default=1e-10
            Small value to avoid division by zero in the denominator.

        Returns
        -------
        nw : float
            The estimated response value at the input point x.
        """
        
        # Compute the normalized pairwise distances between input x and the original training data X_
        u_x = (x - self.X_) / h_x # Scale the differences by bandwidth
        # Compute the normalized pairwise distances between input x and the truncated data X_w
        u_x_w = (x - self.X_w) / h_x # Scale the differences with truncated data

        # Apply the Gaussian kernel to the pairwise differences for original data
        K_x = np.exp(-0.5 * np.sum(u_x**2, axis=1)) / ((2 * np.pi)**(self.X_.shape[1] / 2) * h_x**self.X_.shape[1])
        
        # Apply the Gaussian kernel to the pairwise differences for truncated data
        K_x_w = np.exp(-0.5 * np.sum(u_x_w**2, axis=1)) / ((2 * np.pi)**(self.X_w.shape[1] / 2) * h_x**self.X_w.shape[1])
        
        # Add regularization to avoid numerical issues
        K_x += self.alpha
        K_x_w += self.alpha #* (len(K_x_w)/len(K_x))

        # Compute the numerator: the weighted sum of truncated response values
        nw_num = np.sum(self.R_w * K_x_w[:, np.newaxis]) # Element-wise multiplication of response values with kernel weights
        
        # Compute the denominator: the sum of kernel weights (i.e., marginal density based on the full data)
        nw_denom = np.sum(K_x) # Kernel weight sum for the marginal density
        
        # Return the estimated value (handle division by zero with epsilon)
        nw = (nw_num/(nw_denom + epsilon)) * (len(K_x)/len(K_x_w))
        
        return nw
        
    
    
    def __call__(self, data_state):
        """
        Apply Nadaraya-Watson kernel regression to each row in the input data matrix.

        This method estimates the response values for all input states in data_state
        by applying the Nadaraya-Watson estimator for each row.

        Parameters
        ----------
        data_state : ndarray of shape (n_samples, n_features)
            The data points (states) at which to estimate the response values.

        Returns
        -------
        nw_est_vec : ndarray of shape (n_samples,)
            Vector of estimated response values for each state in data_state.
        """
        # Apply the nw_est method to each row of the input data (data_state)
        nw_est_vec = np.apply_along_axis(self.nw_est, 1, data_state, self.bandwidth)
        
        return nw_est_vec
    
    
    
    

    
    
    
class est_r_sa(BaseEstimator):
    """
    Custom estimator using Nadaraya-Watson kernel regression with both state and action components.

    Parameters
    ----------
    bandwidth_x : float, default=0.1
        The bandwidth parameter for the Gaussian kernel applied to the state features.
    bandwidth_a : float, default=0.1
        The bandwidth parameter for the Gaussian kernel applied to the action labels.

    alpha : float, default=0.1
        The regularization parameter to prevent overfitting and improve numerical stability.
        It adds a small positive value to the kernel weights, ensuring that the computed densities remain finite
        and stable during evaluation, especially in regions where the weight of data points may be very low.

    Attributes
    ----------
    X_ : ndarray of shape (n_samples, n_features)
        The input data used for fitting, representing the features of each sample.
        
    A_ : ndarray of shape (n_samples,)
        The action labels corresponding to each sample in X_, indicating the actions taken for each input sample.
        
    R_ : ndarray of shape (n_samples,)
        The response values associated with X_, representing the outcomes we want to estimate or predict.
    """
    def __init__(self, bandwidth_x=0.1, bandwidth_a = 0.1, alpha = 0.1):
        # Initialize the estimator with the specified bandwidth and regularization parameter
        self.bandwidth_x = bandwidth_x # Bandwidth for state features
        self.bandwidth_a = bandwidth_a # Bandwidth for action features
        self.alpha = alpha # Set the regularization parameter to prevent overfitting and improve numerical stability.
        

    def fit(self, X, A, R):
        """
        Fit the model using input state data X, action labels A, and responses R.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input state data.
        A : ndarray of shape (n_samples,)
            The action labels corresponding to the states.
        R : ndarray of shape (n_samples,)
            The response values.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Store the input state data (X), action labels (A), and response values (R) as instance attributes
        self.X_ = X  # Input state data
        self.A_ = A  # Input action data
        self.R_ = R  # Response values
        return self
    

    def nw_est(self, x, a, h_x, h_a, epsilon = 1e-10):
        """
        Estimate the function value at a given state-action pair using Nadaraya-Watson kernel regression.

        Parameters
        ----------
        x : ndarray of shape (n_features,)
            The state for which to estimate the function value.
        a : scalar
            The action label to condition on.
        h_x : float or ndarray of shape (n_features,)
            The bandwidth parameter(s) for the state kernel.
        h_a : float
            The bandwidth parameter for the action kernel.
        epsilon : float, default=1e-10
            A small value to prevent division by zero.

        Returns
        -------
        nw : float
            The estimated value at the state-action pair (x, a).
        """
        # Compute the normalized pairwise differences for the state and action
        u_x = (x - self.X_) / h_x # Difference between input state and fitted states
        u_a = (a - self.A_) / h_a # Difference between input action and fitted actions
        
        # Apply the Gaussian kernel to the state and action components
        K_x = np.exp(-0.5 * np.sum(u_x**2, axis=1)) / ((2 * np.pi)**(self.X_.shape[1] / 2) * h_x**self.X_.shape[1])
        K_a = np.exp(-0.5 * np.sum(u_a**2, axis=1)) / ((2 * np.pi)**(self.A_.shape[1] / 2) * h_a**self.A_.shape[1])
        
        # Combine the state and action kernels
        K = K_x*K_a
        #K += self.alpha # Adding self.alpha ensures stability and prevents overfitting
        
        # Compute the weighted sum of response values (numerator) and the sum of kernel weights (denominator)
        nw_num = np.sum(self.R_ * K[:, np.newaxis]) # Weighted sum of responses
        nw_denom = np.sum(K) # Sum of kernel weights (marginal density)
        
        # Calculate the Nadaraya-Watson estimate (handling potential division by zero)
        nw = nw_num/(nw_denom + epsilon)
        
        return nw
    
    def __call__(self, data_state, data_action):
        """
        Apply Nadaraya-Watson kernel regression to estimate values for multiple state-action pairs.

        Parameters
        ----------
        data_state : ndarray of shape (n_samples, n_features)
            The state data points for which to estimate the function values.
        data_action : array-like, shape (n_samples,)
            The action labels corresponding to each state.

        Returns
        -------
        nw_est_vec : ndarray of shape (n_samples,)
            The estimated values for each state-action pair.
        """
        # Ensure that the state data and action labels have the same number of samples
        assert data_state.shape[0] == data_action.shape[0], "data_matrix and a_vec must have the same length"
        
        # Apply the Nadaraya-Watson estimator for each state-action pair
        nw_est_vec = np.array([self.nw_est(data_state[i], data_action[i], self.bandwidth_x, self.bandwidth_a) for i in range(len(data_state))])
        return nw_est_vec


    

    
    
class est_r_pi_sa_w(BaseEstimator):
    """
    Custom estimator using Nadaraya-Watson kernel regression with a window parameter.

    Parameters
    ----------
    bandwidth_x : float, default=1.0
        The bandwidth_x parameter for the Gaussian kernel used in the regression.
    bandwidth_a : float, default=1.0
        The bandwidth_a parameter for the action data in the Gaussian kernel.
    alpha : float, default=0.1
        The regularization parameter to prevent overfitting and improve numerical stability.
        It adds a small positive value to the kernel weights, ensuring that the computed densities remain finite
        and stable during evaluation, especially in regions where the weight of data points may be very low.

    Attributes
    ----------
    X_ : ndarray of shape (n_samples, n_features)
        The input data used for fitting (e.g., state variables or features).
    A_ : ndarray of shape (n_samples,)
        The action values associated with X_ (discrete or continuous actions).
    R_ : ndarray of shape (n_samples,)
        The response values (e.g., rewards) associated with X_ and A_.
    w : int
        The window parameter that determines the lag between the response and the input data.
    R_w : ndarray of shape (n_samples-w,)
        The truncated response values aligned with the window parameter.
    X_w : ndarray of shape (n_samples-w, n_features)
        The truncated input data aligned with the window parameter.
    A_w : ndarray of shape (n_samples-w,)
        The truncated action data aligned with the window parameter.
    """

    def __init__(self, bandwidth_x=0.1, bandwidth_a=0.1, alpha=0.1):
        # Initialize the estimator with the specified bandwidth and regularization parameter
        self.bandwidth_x = bandwidth_x # Bandwidth for state features
        self.bandwidth_a = bandwidth_a # Bandwidth for action features
        self.alpha = alpha # Set the regularization parameter to prevent overfitting and improve numerical stability.

    def fit(self, X, A, R, w):
        """
        Fit the model using input data X, action data A, responses R, and a window parameter w.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data (e.g., states or features).
        A : ndarray of shape (n_samples,)
            The action data associated with each input (e.g., discrete or continuous actions).
        R : ndarray of shape (n_samples,)
            The response values (e.g., rewards or outcomes).
        w : int
            The window parameter determining the lag between X and R.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Store input, action, and response data
        self.X_ = X  # Store the input data (states/features)
        self.A_ = A  # Store the action data corresponding to each state
        self.R_ = R  # Store the response values (rewards)
        self.w = w   # Store the window parameter (lag between X and R)

        # Adjust input, action, and response data based on the window parameter
        self.R_w = self.R_[w:]      # Truncate response values by skipping the first 'w' values
        self.X_w = self.X_[:-w]     # Truncate input data to align with the window-adjusted responses
        self.A_w = self.A_[:-w]     # Truncate action data to align with the window-adjusted inputs
        return self  # Return the fitted estimator

    def nw_est(self, x, a, h_x, h_a, epsilon = 1e-10):
        """
        Estimate the function value at a given point using Nadaraya-Watson kernel regression.

        Parameters
        ----------
        x : ndarray of shape (n_features,)
            The point at which to estimate the function value.
        a : scalar
            The action for which to estimate the value.
        h_x : float or ndarray of shape (n_features,)
            The bandwidth_x parameter(s) for the kernel.
        h_a : float
            The bandwidth_a parameter for the kernel.
        epsilon : float, default=1e-10
            A small value to avoid division by zero.

        Returns
        -------
        nw : float
            The estimated value at point x for action 'a'.
        """
        
        # Compute the normalized pairwise distances between input x and the original training data X_ and A_
        u_x = (x - self.X_) / h_x # Scale the differences by bandwidth for state
        u_a = (a - self.A_) / h_a # Scale the differences by bandwidth for action
        # Compute the normalized pairwise distances between input x and the truncated data X_w and A_w
        u_x_w = (x - self.X_w) / h_x # Scale the differences by bandwidth for truncated state
        u_a_w = (a - self.A_w) / h_a # Scale the differences by bandwidth for truncated action

        # Apply the Gaussian kernel to the pairwise differences for the full data
        K_x = np.exp(-0.5 * np.sum(u_x**2, axis=1)) / ((2 * np.pi)**(self.X_.shape[1] / 2) * h_x**self.X_.shape[1])
        K_a = np.exp(-0.5 * np.sum(u_a**2, axis=1)) / ((2 * np.pi)**(self.A_.shape[1] / 2) * h_a**self.A_.shape[1])
        
        # Combine the state and action kernels
        K = K_x*K_a
        #K += self.alpha # Adding self.alpha ensures stability and prevents overfitting
        
        
        # Apply the Gaussian kernel to the pairwise differences for the truncated data
        K_x_w = np.exp(-0.5 * np.sum(u_x_w**2, axis=1)) / ((2 * np.pi)**(self.X_w.shape[1] / 2) * h_x**self.X_w.shape[1])
        K_a_w = np.exp(-0.5 * np.sum(u_a_w**2, axis=1)) / ((2 * np.pi)**(self.A_w.shape[1] / 2) * h_a**self.A_w.shape[1])
        
        # Combine the truncated state and truncated action kernels
        K_w = K_x_w*K_a_w
        #K_w += self.alpha * (len(K_w)/len(K)) # Adding self.alpha ensures stability and prevents overfitting

        # Compute the numerator: the weighted sum of truncated response values
        nw_num = np.sum(self.R_w * (K_w)[:, np.newaxis]) # Element-wise multiplication of response values with kernel weights
        
        # Compute the denominator: the sum of kernel weights (i.e., marginal density based on the full data)
        nw_denom = np.sum(K) # Kernel weight sum for the marginal density
        
        # Return the estimated value (avoid division by zero with epsilon)
        nw = (nw_num/(nw_denom + epsilon)) * (len(K)/len(K_w))
        
        
        return nw

    def __call__(self, data_state, data_action):
        """
        Apply Nadaraya-Watson kernel regression across all rows in the data matrix with action vector.

        Parameters
        ----------
        data_state : ndarray of shape (n_samples, n_features)
            The state data points at which to estimate the function values.
        data_action : array-like, shape (n_samples,)
            The action vector for which the function is estimated.

        Returns
        -------
        nw_est_vec : ndarray of shape (n_samples,)
            The estimated values for each state-action pair.
        """
        # Ensure that data_state and data_action have the same length
        assert data_state.shape[0] == data_action.shape[0], "data_state and data_action must have the same length"
        
        # Apply the Nadaraya-Watson estimation method to each row of the state-action pairs
        nw_est_vec = np.array([self.nw_est(data_state[i], data_action[i], self.bandwidth_x, self.bandwidth_a) for i in range(len(data_state))])
        
        return nw_est_vec  # Return the estimated values for each data point and action
