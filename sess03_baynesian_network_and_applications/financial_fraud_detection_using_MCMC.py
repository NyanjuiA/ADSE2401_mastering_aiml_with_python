# Python script to demonstrate Financial Fraud detection using Baynesian Logistic Regression
# with Monte Carlo Markov Chain (MCMC)
# Features modelled:
#  - Transaction amount (continuous + binned: low,medium,high)
#  - Location (domestic/usual vs. international/unusual)
#  - Time of day ( day/normal vs. night/odd)
#  - User behaviour (normal/good vs. risky/suspicious)
# We'll see the strength/influence of each of the above features on fraud via MCMC,
# then compute posterior predictive probabilities for new transactions.

# NB: Ensure that pymc & arvis modules are installed ( pip install pymc arviz )

# Import the required modules
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

# Suppress warnings
import warnings

warnings.filterwarnings('ignore')


# -------------------------------------------------------------------------
# 1. Synthetic Data Generation
# -------------------------------------------------------------------------
# Data generation function
def generate_transaction_data(n_sample=1000, random_seed=42):
    """
    Generate synthetic transaction data with probabilistic fraud labels.

    Parameters
    ----------
    n_sample : int
        Number of transactions to generate.
    random_seed : int
        Seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        Dataset containing features and fraud labels.
    """
    np.random.seed(random_seed)
    np.random.seed(random_seed)

    # continuous features
    transaction_amount = np.random.normal(loc=200, scale=120, size=n_sample)
    # clip to reasonable range
    transaction_amount = np.clip(transaction_amount, 10, 2000)

    # Categorical features
    location = np.random.choice(['domestic', 'international'], size=n_sample, p=[.75, .25])
    time_of_day = np.random.choice(['day', 'night'], size=n_sample, p=[.65, .35])
    user_behaviour = np.random.choice(['normal', 'risky'], size=n_sample, p=[.82, .18])

    # Binned amount for intuitive interpretation
    amount_bin = pd.cut(transaction_amount,
                        bins=[0, 100, 500, np.inf],
                        labels=["low", "medium", "high"])

    # Fraud generation: influenced by all above feature (stronger effect from risky behaviour & international)
    fraud_prob = (
            .15 * (transaction_amount > 400) +
            .35 * (location == 'international') +
            .25 * (time_of_day == 'night') +
            .55 * (user_behaviour == 'risky') +
            np.random.normal(0, .05, n_sample)  # introduce some noise
    )

    # Calculate fraud
    fraud = np.random.binomial(1, np.clip(fraud_prob, .01, .95))

    # Create the pandas dataframe
    data = pd.DataFrame({
        'transaction_amount': transaction_amount,
        'amount_bin': amount_bin,
        'location': location,
        'time_of_day': time_of_day,
        'user_behaviour': user_behaviour,
        'fraud': fraud
    })

    # Display the generated data
    print(f"Generated {n_sample} transactions. Fraud rate: {fraud.mean():.3%}")
    return data


# -------------------------------------------------------------------------
# 2. Build the Baynesian Logistic Regression Model
# -------------------------------------------------------------------------
# Baynesian Logistic Regression model function
def build_fraud_model(data):
    """
    Build a Bayesian logistic regression model using PyMC.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataset.

    Returns
    -------
    pymc.Model
        Compiled probabilistic model.
    """
    # Encode the categorical data to numeric (0/1)
    location_int = (data['location'] == 'international').astype(int)
    time_int = (data['time_of_day'] == 'night').astype(int)
    behaviour_int = (data['user_behaviour'] == 'risky').astype(int)

    # Scale continuous feature for better sampling
    amount_scaled = data['transaction_amount'] / 100.0

    with pm.Model() as model:
        # Weakly informative priors (better than flat Normal(0,1))
        intercept = pm.Normal('intercept', mu=0, sigma=2)
        beta_amount = pm.Normal('beta_amount', mu=0, sigma=1)
        beta_location = pm.Normal('beta_location', mu=0, sigma=1.5)  # expect a stronger effect
        beta_time = pm.Normal('beta_time', mu=0, sigma=1)
        beta_behaviour = pm.Normal('beta_behaviour', mu=0, sigma=2)  # Strongest expected effect

        # Linear predictor
        logit_p = (
                intercept + beta_amount * amount_scaled +
                beta_location * location_int +
                beta_time * time_int +
                beta_behaviour * behaviour_int
        )

        # Likelihood
        p_fraud = pm.Deterministic('p_fraud', pm.math.sigmoid(logit_p))
        observed = pm.Bernoulli('observed_fraud', p=p_fraud, observed=data['fraud'])

    return model


# -------------------------------------------------------------------------
# 3. Run Monte Carlo Markov Chain (MCMC) Sampling
# -------------------------------------------------------------------------
def run_mcmc(model, draws=2000, tune=100, target_accept=0.95):
    """
    Run MCMC sampling using PyMC.

    Parameters
    ----------
    model : pymc.Model
        The probabilistic model.
    draws : int
        Number of posterior samples.
    tune : int
        Number of tuning steps.
    target_accept : float
        Target acceptance rate for NUTS.

    Returns
    -------
    arviz.InferenceData
        Posterior samples.
    """
    with (model):
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=4,
            target_accept=target_accept,
            return_inferencedata=True,
            progressbar=True
        )
    return trace


# -------------------------------------------------------------------------
# 4. Analyse and make predictions
# -------------------------------------------------------------------------
def analyse_results(trace, data):
    """
    Analyse posterior results and compute prediction for a new transaction.

    Parameters
    ----------
    trace : arviz.InferenceData
        Posterior samples.
    data : pandas.DataFrame
        Original dataset.
    """
    print("\n=== Posterior Summary (Coefficients) ===")
    summary = az.summary(trace,
                         var_names=['intercept', 'beta_amount','beta_location', 'beta_time', 'beta_behaviour'])
    print(summary)

    # Plot trace (convergence check
    az.plot_trace(trace,
                  var_names=['intercept', 'beta_amount',  'beta_location', 'beta_time', 'beta_behaviour'])
    plt.tight_layout()
    plt.show()

    # Fraud probability example for a new high-risk transaction
    new_transaction = {
        'amount': 650,  # high amount > 400
        'location': 'international',
        'time_of_day': 'night',
        'user_behaviour': 'risky'
    }

    # Encode the inputs
    amount_scaled = new_transaction['amount'] / 100.0
    loc_int = int(new_transaction['location'] == 'international')
    time_int = int(new_transaction['time_of_day'] == 'night')
    beh_int = int(new_transaction['user_behaviour'] == 'risky')

    # Extract the posterior samples as flattened arrays
    post = trace.posterior()

    # Stack the chains and draws to get 1D arrays of samples
    intercept_samples = post['intercept'].stack(sample=("chain","draw")).values
    beta_amount_samples = post['beta_amount'].stack(sample=("chain","draw")).values
    beta_location_samples = post['beta_location'].stack(sample=("chain","draw")).values
    beta_time_samples = post['beta_time'].stack(sample=("chain","draw")).values
    beta_behaviour_samples = post['beta_behaviour'].stack(sample=("chain","draw")).values

    # Compute the logit samples
    logit_samples = (
        intercept_samples +
        beta_amount_samples * amount_scaled +
        beta_location_samples * loc_int +
        beta_time_samples * time_int +
        beta_behaviour_samples * beh_int
    )

    # Convert to probatility
    p_samples = 1 / (1 + np.exp(-logit_samples)) # probability of overflow

    # mean probability
    mean_prob = float(p_samples.mean())

    # HDI
    hdi = az.hdi(p_samples,hdi_prob=.94)
    lower = hdi[0]
    upper = hdi[1]


    print("\n==== New Transaction Risk Assessment ====")
    print(f"Transaction: {new_transaction}")
    print(f"Esitimated P(Fraud) = {mean_prob:.4f}"
          f"94% HDI: [{hdi[0]:.4f}, {hdi[1]:.4f}]")

    if mean_prob > .75:
        print("⛔ HIGH RISK - Strong Indication of Fraud")
    elif mean_prob > .4:
        print("⚠ MEDIUM RISK - Recommend Manual Review")
    else:
        print("☑ LOW RISK - Likely Legitimate")


# -------------------------------------------------------------------------
# Run the application
# -------------------------------------------------------------------------
if __name__ == '__main__':
    print("Financial Fraud Detection - Baynesian Network Using MCMC")

    # a) Generate data
    data = generate_transaction_data(n_sample=1200)

    # b) Build the model
    model = build_fraud_model(data)

    # c) Run MCMC
    trace = run_mcmc(model, draws=1500, tune=800)

    # d) Analyse and predict on example transaction
    analyse_results(trace, data)



# Compute posterior predictive probability for the above transaction
# using posterior samples of coefficients
# post = trace.posterior
# amount_scaled = new_transaction['amount'] / 100.0
# loc_int = 1 if new_transaction['location'] == 'international' else 0
# time_int = 1 if new_transaction['time_of_day'] == 'night' else 0
# beh_int = 1 if new_transaction['user_behaviour'] == 'risky' else 0
#
# logit_samples = (
#         post['intercept'] +
#         post['beta_amount'] * amount_scaled +
#         post['beta_location'] * loc_int +
#         post['beta_time'] * time_int +
#         post['beta_behaviour'] * beh_int
# )
#
# p_fraud_samples = 1 / (1 + np.exp(-logit_samples))
# mean_prob = p_fraud_samples.mean().item()
# hdi = az.hdi(p_fraud_samples, hdi_prob=.94)
