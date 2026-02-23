from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

class FactorModel(nn.Module):
    def __init__(self, n_rows, n_cols, n_components, w_init=None, v_init=None, u_init=None, covariates=None, likelihood='gaussian'):
        super().__init__()

        if w_init is None:
            self.W = Parameter(torch.FloatTensor(np.random.normal(0,0.01, size=(n_rows, n_components))))
        else:
            self.W = Parameter(torch.FloatTensor(w_init))

        if v_init is None:
            self.V = Parameter(torch.FloatTensor(np.random.normal(0,0.01, size=(n_cols, n_components))))
        else:
            self.V = Parameter(torch.FloatTensor(v_init))

        if u_init is None:
            self.U = Parameter(torch.FloatTensor(np.random.normal(0,0.01, size=(n_cols, n_components))))
        else:
            self.U = Parameter(torch.FloatTensor(u_init))

        if covariates is not None:
            self.covariates = torch.FloatTensor(covariates)
            self.coefficients = Parameter(torch.FloatTensor(np.random.normal(0,0.01, size=(n_cols, covariates.shape[1]))))
        else:
            self.covariates = None

        self.likelihood = likelihood
        if self.likelihood == 'gaussian':
            self.link = lambda x: x
        elif self.likelihood == 'poisson':
            self.link = lambda x: nn.Softplus()(x)
        elif self.likelihood == 'bernoulli':
            self.link = lambda x: torch.special.expit(x)
        elif self.likelihood == 'nb':
            self.link = lambda x: nn.Softplus()(x)
                             
    def forward(self, row_idxs, col_idxs):
        mu = (self.W[row_idxs] * self.V[col_idxs]).sum(dim=-1)

        # Allow for sample-specific coefficients
        if self.covariates is not None:
            mu += (self.covariates[row_idxs] * self.coefficients[col_idxs]).sum(dim=-1)
        mu = self.link(mu)

        sigma = (self.W[row_idxs] * self.U[col_idxs]).sum(dim=-1)
        sigma = nn.functional.softplus(sigma) + 1e-6

        return mu, sigma

    def reconstruct_matrix(self):
        mu = self.W @ self.V.T
        if self.covariates is not None:
            mu += self.covariates @ self.coefficients.T
        mu = self.link(mu)

        sigma = self.W @ self.U.T
        sigma = nn.functional.softplus(sigma) + 1e-6

        return mu.detach().numpy(), sigma.detach().numpy()



def factorize(X, n_components, covariates=None, likelihood='gaussian',
                n_steps=5000, batch_size=2048, w_init=None, v_init=None, u_init=None,
                idxs=None):
    t_X = torch.FloatTensor(X)
    n_rows, n_cols = X.shape
    
    model = FactorModel(n_rows, n_cols, n_components, w_init=w_init, v_init=v_init, u_init=u_init, likelihood=likelihood, covariates=covariates)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if idxs is None:
        idxs = np.indices(X.shape).reshape(2,-1).T

    for step in tqdm(range(n_steps)):
        # Set the model to training mode
        model.train()

        # Reset the gradient
        model.zero_grad()

        # SMALL EDIT: Adjust batch size to the number of available indices
        current_batch_size = min(batch_size, idxs.shape[0])

        # Sample random columns and rows
        row_idxs, col_idxs = idxs[np.random.choice(idxs.shape[0], replace=False, size=current_batch_size)].T

        # Get the probabilities and observations
        mu, sigma = model(row_idxs, col_idxs)
        x_obs = t_X[row_idxs, col_idxs]

        # Calculate the negative log-likelihood
        if likelihood == 'gaussian':
            loss = -torch.distributions.Normal(mu, sigma).log_prob(x_obs).mean()
        elif likelihood == 'bernoulli':
            loss = -torch.distributions.Bernoulli(mu).log_prob(x_obs).mean()
        elif likelihood == 'poisson':
            loss = -torch.distributions.Poisson(mu).log_prob(x_obs).mean()
        elif likelihood == 'nb':
            probs = sigma / (sigma + mu + 1e-6)  
            loss = -torch.distributions.NegativeBinomial(total_count=sigma, probs=probs).log_prob(x_obs).mean()

        # Calculate gradients
        loss.backward()

        # Apply the update
        optimizer.step()
        if step % 1000 == 0:
            print(f"Step {step}, Loss: {loss.item()}, Mu Mean: {mu.mean().item()}, Sigma Mean: {sigma.mean().item()}")

        if torch.isnan(loss):
            raise Exception

    model.eval()
    W_hat = model.W.detach().numpy()
    V_hat = model.V.detach().numpy()
    U_hat = model.U.detach().numpy()

    X_hat_mu, X_hat_sigma = model.reconstruct_matrix()

    return X_hat_mu, X_hat_sigma, W_hat, V_hat, U_hat, model


def cross_validate_factor_model(X, n_components, covariates=None, likelihood='gaussian',
                                n_steps=5000, batch_size=2048, w_init=None, v_init=None, u_init=None,
                                K=5, runs=None, random_state=42):
    np.random.seed(random_state)
    kf = KFold(n_splits=K, shuffle=True, random_state=random_state)
    mse_scores = []
    n_rows, n_cols = X.shape
    all_indices = np.array([(i, j) for i in range(n_rows) for j in range(n_cols)])

    fold = 1
    if runs is None:
        runs = K
    for i, (train_indices, val_indices) in enumerate(kf.split(all_indices)):
        if i >= runs:
            break

        print(f"\nStarting Fold {fold}/{K}")
        
        train_entries = all_indices[train_indices]
        val_entries = all_indices[val_indices]
        
        # Create mask for only training entries
        train_mask = np.zeros(X.shape, dtype=bool)
        train_mask[train_entries[:,0], train_entries[:,1]] = True

        X_train = X.copy()
        X_train[~train_mask] = 0 
        
        idxs = train_entries

        X_hat_mu, X_hat_sigma, W_hat, V_hat, U_hat, model = factorize(
            X_train, 
            n_components=n_components, 
            covariates=covariates, 
            likelihood=likelihood,
            n_steps=n_steps,
            batch_size=batch_size,
            w_init=w_init,
            v_init=v_init,
            u_init=u_init,
            idxs=idxs
        )
        
        y_true = X[val_entries[:, 0], val_entries[:, 1]]
        y_pred = X_hat_mu[val_entries[:, 0], val_entries[:, 1]]
        mse = mean_squared_error(y_true, y_pred)
        mse_scores.append(mse)
        print(f"Fold {fold} MSE: {mse:.6f}")
        
        fold += 1

    avg_mse = np.mean(mse_scores)
    std_mse = np.std(mse_scores)
    print(f"\nCross-Validation Results:")
    print(f"Average MSE: {avg_mse:.6f} ± {std_mse:.6f}")

    return mse_scores

def cross_validate_factor_model_cols(X, n_components, holdout_fraction=0.2,
                                         covariates=None, likelihood='gaussian',
                                         n_steps=5000, batch_size=2048,
                                         w_init=None, v_init=None, u_init=None,
                                         runs=5, random_state=42):
    np.random.seed(random_state)
    mse_scores = []
    n_rows, n_cols = X.shape

    for run in range(runs):
        holdout_count = max(1, int(holdout_fraction * n_cols))
        # Randomly choose columns to hold out.
        holdout_cols = np.random.choice(np.arange(n_cols), size=holdout_count, replace=False)
        
        train_mask = np.ones_like(X, dtype=bool)
        train_mask[:, holdout_cols] = False
        
        val_mask = np.zeros_like(X, dtype=bool)
        val_mask[:, holdout_cols] = True
        
        X_train = X.copy()
        X_train[:, holdout_cols] = 0
        
        train_indices = np.column_stack(np.where(train_mask))
        
        X_hat_mu, X_hat_sigma, W_hat, V_hat, U_hat, model = factorize(
            X_train, 
            n_components=n_components, 
            covariates=covariates, 
            likelihood=likelihood,
            n_steps=n_steps,
            batch_size=batch_size,
            w_init=w_init,
            v_init=v_init,
            u_init=u_init,
            idxs=train_indices
        )

        val_indices = np.column_stack(np.where(val_mask))
        y_true = X[val_indices[:, 0], val_indices[:, 1]]
        y_pred = X_hat_mu[val_indices[:, 0], val_indices[:, 1]]
        mse = mean_squared_error(y_true, y_pred)
        mse_scores.append(mse)
        print(f"Run {run+1}/{runs} - MSE: {mse:.6f}")

    avg_mse = np.mean(mse_scores)
    std_mse = np.std(mse_scores)
    print(f"\nColumn-wise Cross-Validation Results:")
    print(f"Average MSE: {avg_mse:.6f} ± {std_mse:.6f}")
    return mse_scores



def select_optimal_components(X, ks, covariates=None, likelihood='gaussian',
                                n_steps=5000, batch_size=2048, K=5, runs=None, random_state=42):
    results = {}
    for k in ks:
        print(f"\nEvaluating factor model with k = {k}")
        mse_scores = cross_validate_factor_model(
            X, 
            n_components=k, 
            covariates=covariates, 
            likelihood=likelihood,
            n_steps=n_steps,
            batch_size=batch_size,
            K=K,
            runs=runs,
            random_state=random_state
        )
        # mse_scores = cross_validate_factor_model_cols(
        #     X,
        #     n_components=k,
        #     holdout_fraction=0.2,
        #     likelihood=likelihood,
        #     n_steps=n_steps,
        #     batch_size=batch_size,
        #     runs=5,
        #     random_state=random_state
        # )
        avg_mse = np.mean(mse_scores)
        std_mse = np.std(mse_scores)
        print(f"n_components = {k}, Average MSE: {avg_mse:.6f} ± {std_mse:.6f}")
        results[k] = {'avg_mse': avg_mse, 'std_mse': std_mse, 'mse_scores': mse_scores}
    
    
    best_k = min(results, key=lambda k: results[k]['avg_mse'])
    print(f"\nOptimal k: {best_k} with average MSE: {results[best_k]['avg_mse']:.6f}")
    return best_k, results



def check_difference(suffix=""):
    # Call this function to check how well the factor model fits the observed data
    suffix = f"_{suffix}" if suffix else ""
    
    X = np.load(f'inputs/X_continuous{suffix}.npy')
    # X_hat_mu = np.load(f'inputs/X_hat_mu{suffix}.npy')
    # X_hat_sigma = np.load(f'inputs/X_hat_sigma{suffix}.npy')
    W_hat = np.load(f'inputs/W_hat{suffix}.npy')
    V_hat = np.load(f'inputs/V_hat{suffix}.npy')
    U_hat = np.load(f'inputs/U_hat{suffix}.npy')

    means = np.dot(W_hat, V_hat.T)
    stds = np.dot(W_hat, U_hat.T)
    stds = np.log1p(np.exp(-np.abs(stds))) + np.maximum(stds, 0) + 1e-6

    # Load the bad ones
    W_hat_worst = np.load(f'inputs/W_hat{suffix}_worst.npy')
    V_hat_worst = np.load(f'inputs/V_hat{suffix}_worst.npy')
    U_hat_worst = np.load(f'inputs/U_hat{suffix}_worst.npy')

    means_worst = np.dot(W_hat_worst, V_hat_worst.T)
    stds_worst = np.dot(W_hat_worst, U_hat_worst.T)
    stds_worst = np.log1p(np.exp(-np.abs(stds_worst))) + np.maximum(stds_worst, 0) + 1e-6

    print(np.mean(X))
    plt.hist(X.flatten(), bins=30, density=True, label='X')
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.title('X Distribution')
    plt.legend()
    plt.show()
    # factorize(X, n_components=20, likelihood='poisson')
    mse_scores = cross_validate_factor_model(X, n_components=50, K=5, runs=5, likelihood='poisson')
    
    # Plot the MSE scores
    plt.figure(figsize=(8, 6))
    plt.boxplot(mse_scores, vert=True, patch_artist=True, 
                boxprops=dict(facecolor='skyblue'))
    plt.ylabel('Mean Squared Error')
    plt.title('Cross-Validation MSE Scores')
    plt.xticks([1], ['Factor Model'])
    plt.show()

    return
    
    
    # First check:
    # How similar is torch operations with numpy native?
    # reconstructing mean matrix and std matrix with np instead of torch.softplus

    # mean_diff = np.abs(means - X_hat_mu)
    # max_mean_diff = np.max(mean_diff)
    # mean_mean_diff = np.mean(mean_diff)

    # print(f"Max mean difference: {max_mean_diff}, Mean mean difference: {mean_mean_diff}")

    # std_diff = np.abs(stds - X_hat_sigma)
    # max_std_diff = np.max(std_diff)
    # mean_std_diff = np.mean(std_diff)

    # print(f"Max std difference: {max_std_diff}, Mean std difference: {mean_std_diff}")
    # print()

    # Second check:
    # How close is the observed data with the factor model distribution?
    # Calculate probability of each observed point given the (mean, std)
    # The distribution across the whole matrix should be Uniform

    cdf_values = stats.norm.cdf(X, loc=means, scale=stds)

    cdf_mean = np.mean(cdf_values)
    cdf_variance = np.var(cdf_values)

    print(f"Mean of CDF values: {cdf_mean:.4f}")
    print(f"Variance of CDF values: {cdf_variance:.4f}")

    percentiles = np.percentile(cdf_values, [25, 50, 75])

    print(f"25th Percentile: {percentiles[0]:.4f}")
    print(f"50th Percentile (Median): {percentiles[1]:.4f}")
    print(f"75th Percentile: {percentiles[2]:.4f}")

    reconstruction_error = np.linalg.norm(X - means) / np.linalg.norm(means)
    print(f"Reconstruction Error: {reconstruction_error}")

    # Plot:
    # X distribution plot

    X_pre = np.load(f'inputs/X_continuous_preprocess_generated.npy')
    
    plt.figure()
    # plt.hist(X.flatten(), bins=50, density=True, alpha=0.5, label='X (after normalization)')

    # plt.hist(X_pre.flatten(), bins=50, density=True, alpha=0.5, label='X (before normalization)')
    plt.hist(X.flatten(), bins=50, density=True, alpha=0.5, label='X')
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.title('X Distribution')
    plt.legend()
    plt.show()

    # Plot: 
    # X axis is the CDF values of the observation against our factor model
    # np.sort(cdf_values.flatten())
    # Y axis is the CDF values if it fit perfectly, aka uniform distribution
    # np.linspace(0, 1, len(cdf_values.flatten()))
    
    plt.figure()
    plt.plot(np.sort(cdf_values.flatten()), np.linspace(0, 1, len(cdf_values.flatten())), marker='o', linestyle='none')
    plt.plot([0, 1], [0, 1], color='r')  # Ideal line is one to one
    plt.xlabel("Empirical CDF (Observed Data)")
    plt.ylabel("Theoretical CDF (Uniform Distribution)")
    plt.title("Empirical CDF vs Theoretical CDF")
    plt.show()

    # How far off are the mean and variance?
    # Mean

    mean_diff = np.abs(X - means)
    max_mean_diff = np.max(mean_diff)
    print(f"Max difference of means and actuals: {max_mean_diff}")

    mean_diff_worst = np.abs(X - means_worst)
    max_mean_diff_worst = np.max(mean_diff_worst)
    print(f"Max difference of means and actuals (worst): {max_mean_diff_worst}")

    # Variance

    squared_residuals = (X - means) ** 2
    variance_diff = np.abs(squared_residuals - stds ** 2)
    max_variance_diff = np.max(variance_diff)
    print(f"Max difference between squared residuals and variances: {max_variance_diff}")

    squared_residuals_worst = (X - means_worst) ** 2
    variance_diff_worst = np.abs(squared_residuals_worst - stds_worst ** 2)
    max_variance_diff_worst = np.max(variance_diff_worst)
    print(f"Max difference between squared residuals and variances (worst): {max_variance_diff_worst}")

    # Plot
    # Same structure as the one above, QQ plot between means and actuals

    sorted_actual_values = np.sort(X.flatten())
    sorted_predicted_means = np.sort(means.flatten())
    sorted_predicted_means_worst = np.sort(means_worst.flatten())

    plt.figure()
    plt.plot(sorted_actual_values, sorted_predicted_means, marker='o', linestyle='none') # , label='Best Model')
    # plt.plot(sorted_actual_values, sorted_predicted_means_worst, marker='x', linestyle='none', label='Worst Model')
    min_value = min(sorted_actual_values.min(), sorted_predicted_means.min(), sorted_predicted_means_worst.min())
    max_value = max(sorted_actual_values.max(), sorted_predicted_means.max(), sorted_predicted_means_worst.max())
    plt.plot([min_value, max_value], [min_value, max_value], 'r')

    plt.xlim(min_value, max_value)
    plt.ylim(min_value, max_value)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Means")
    plt.title("Actual Values vs Predicted Means")
    plt.legend()
    plt.show()

    # Plot
    # Same structure as the one above, QQ plot between residuals and variance

    sorted_squared_residuals = np.sort(squared_residuals.flatten())
    sorted_variances = np.sort(stds.flatten() ** 2)
    sorted_squared_residuals_worst = np.sort(squared_residuals_worst.flatten())
    sorted_variances_worst = np.sort(stds_worst.flatten() ** 2)

    plt.figure()
    plt.plot(sorted_squared_residuals, sorted_variances, marker='o', linestyle='none') # , label='Best Model')
    # plt.plot(sorted_squared_residuals_worst, sorted_variances_worst, marker='x', linestyle='none', label='Worst Model')
    max_value = max(sorted_squared_residuals.max(), sorted_variances.max(), sorted_squared_residuals_worst.max(), sorted_variances_worst.max())
    plt.plot([0, max_value], [0, max_value], 'r')
    
    plt.xlabel("Squared Residuals")
    plt.ylabel("Predicted Variances")
    plt.title("Squared Residuals vs Predicted Variances")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # check_difference()
    # check_difference("mini")
    # check_difference("generated")
    # check_difference("normalized")
    check_difference("normalized_mini")
