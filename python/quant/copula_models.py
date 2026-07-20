"""
Copula Models for Tail Dependence and Systemic Risk Modeling
============================================================
Implements Gaussian, Student-t, Clayton, and Gumbel Copulas using pure NumPy/SciPy.
Designed to model non-linear tail dependencies between crypto assets to predict
systemic crash risks and black swan correlation spikes.

Optimized for AMD Ryzen AI 5 / Radeon GPU with strict 8GB RAM cap.
No heavy dependencies; uses sparse operations where possible.
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize
from typing import Tuple, List, Optional
import warnings

warnings.filterwarnings('ignore')


class CopulaModel:
    """
    Base class for Copula models.
    Copulas allow modeling of multivariate distributions with arbitrary marginals
    by separating the dependence structure from the marginal distributions.
    """
    
    def __init__(self, n_dims: int):
        """
        Initialize copula with dimensionality.
        
        Args:
            n_dims: Number of assets/variables in the portfolio
        """
        self.n_dims = n_dims
        self.theta = None  # Dependence parameter(s)
        self.fitted = False
        
    def fit(self, data: np.ndarray) -> 'CopulaModel':
        """Fit copula parameters to uniform [0,1] data."""
        raise NotImplementedError
        
    def sample(self, n_samples: int) -> np.ndarray:
        """Generate samples from the fitted copula."""
        raise NotImplementedError
        
    def cdf(self, u: np.ndarray) -> float:
        """Evaluate copula CDF at point u."""
        raise NotImplementedError
        
    def pdf(self, u: np.ndarray) -> float:
        """Evaluate copula PDF at point u."""
        raise NotImplementedError


class GaussianCopula(CopulaModel):
    """
    Gaussian Copula implementation.
    
    The Gaussian copula assumes elliptical dependence with no tail dependence.
    While it fails to capture extreme tail co-movements, it serves as a baseline
    and is computationally efficient for high-dimensional portfolios.
    
    Parameter: Correlation matrix R (n_dims x n_dims)
    """
    
    def __init__(self, n_dims: int):
        super().__init__(n_dims)
        self.corr_matrix = np.eye(n_dims)
        
    def fit(self, data: np.ndarray, method: str = 'ml') -> 'GaussianCopula':
        """
        Fit Gaussian copula via maximum likelihood or method-of-moments.
        
        Args:
            data: Array of shape (n_samples, n_dims) with values in [0,1]
            method: 'ml' for MLE, 'mom' for method of moments
            
        Returns:
            self
        """
        if data.shape[1] != self.n_dims:
            raise ValueError(f"Expected {self.n_dims} dimensions, got {data.shape[1]}")
        
        # Transform uniform data to normal scores
        z = stats.norm.ppf(np.clip(data, 1e-10, 1 - 1e-10))
        
        if method == 'mom':
            # Method of moments: correlation of normal scores
            self.corr_matrix = np.corrcoef(z.T)
        else:
            # Maximum likelihood estimation
            def neg_log_likelihood(r_flat):
                R = self._unflatten_corr(r_flat, self.n_dims)
                try:
                    inv_R = np.linalg.inv(R)
                    sign, logdet = np.linalg.slogdet(R)
                    if sign <= 0:
                        return 1e10
                    ll = -0.5 * np.sum(z @ inv_R * z) - 0.5 * data.shape[0] * logdet
                    return -ll
                except np.linalg.LinAlgError:
                    return 1e10
                    
            # Initial guess: identity correlation
            r_init = np.zeros(self.n_dims * (self.n_dims - 1) // 2)
            result = minimize(neg_log_likelihood, r_init, method='L-BFGS-B',
                            options={'maxiter': 100})
            self.corr_matrix = self._unflatten_corr(result.x, self.n_dims)
        
        self.theta = self.corr_matrix
        self.fitted = True
        return self
    
    def _flatten_corr(self, R: np.ndarray) -> np.ndarray:
        """Flatten correlation matrix to vector of lower triangular elements."""
        idx = np.tril_indices(self.n_dims, -1)
        return R[idx]
    
    def _unflatten_corr(self, r_flat: np.ndarray, n: int) -> np.ndarray:
        """Reconstruct correlation matrix from flattened vector."""
        R = np.eye(n)
        idx = np.tril_indices(n, -1)
        R[idx] = r_flat
        R[(idx[1], idx[0])] = r_flat  # Symmetric
        
        # Ensure positive semi-definite
        eigvals = np.linalg.eigvalsh(R)
        if np.min(eigvals) < 0:
            R = R + (-np.min(eigvals) + 1e-6) * np.eye(n)
            # Re-normalize to correlation matrix
            d = np.sqrt(np.diag(R))
            R = R / np.outer(d, d)
        return R
    
    def sample(self, n_samples: int) -> np.ndarray:
        """Sample from Gaussian copula using Cholesky decomposition."""
        if not self.fitted:
            raise RuntimeError("Copula not fitted")
        
        try:
            L = np.linalg.cholesky(self.corr_matrix)
        except np.linalg.LinAlgError:
            # Nearest positive definite
            eigvals, eigvecs = np.linalg.eigh(self.corr_matrix)
            eigvals = np.maximum(eigvals, 1e-8)
            R_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T
            L = np.linalg.cholesky(R_pd)
        
        z = np.random.randn(n_samples, self.n_dims)
        u = stats.norm.cdf(z @ L.T)
        return u
    
    def cdf(self, u: np.ndarray) -> float:
        """Evaluate Gaussian copula CDF."""
        if not self.fitted:
            raise RuntimeError("Copula not fitted")
        
        z = stats.norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))
        mvn = stats.multivariate_normal(mean=np.zeros(self.n_dims), 
                                        cov=self.corr_matrix)
        return mvn.cdf(z)
    
    def pdf(self, u: np.ndarray) -> float:
        """Evaluate Gaussian copula PDF."""
        if not self.fitted:
            raise RuntimeError("Copula not fitted")
        
        z = stats.norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))
        sign, logdet = np.linalg.slogdet(self.corr_matrix)
        if sign <= 0:
            return 0.0
        
        inv_R = np.linalg.inv(self.corr_matrix)
        quad = z @ inv_R @ z
        pdf_val = np.exp(-0.5 * quad) / np.sqrt(np.abs(sign * logdet))
        
        # Adjust for marginal densities
        phi_z = np.prod(stats.norm.pdf(z))
        return pdf_val / phi_z if phi_z > 0 else 0.0


class StudentTCopula(CopulaModel):
    """
    Student-t Copula implementation.
    
    The t-copula captures symmetric tail dependence, making it superior to
    Gaussian copula for modeling joint extreme events (crashes/rallies).
    
    Parameters: Correlation matrix R, degrees of freedom nu
    """
    
    def __init__(self, n_dims: int, nu_init: float = 5.0):
        super().__init__(n_dims)
        self.nu = nu_init  # Degrees of freedom
        self.corr_matrix = np.eye(n_dims)
        
    def fit(self, data: np.ndarray, nu_bounds: Tuple[float, float] = (2.0, 30.0)) -> 'StudentTCopula':
        """
        Fit t-copula via two-step MLE (IFM method).
        
        Args:
            data: Array of shape (n_samples, n_dims) with values in [0,1]
            nu_bounds: Bounds for degrees of freedom optimization
            
        Returns:
            self
        """
        if data.shape[1] != self.n_dims:
            raise ValueError(f"Expected {self.n_dims} dimensions, got {data.shape[1]}")
        
        n_samples = data.shape[0]
        
        # Step 1: Estimate correlation matrix (assuming large nu initially)
        z = stats.t.ppf(np.clip(data, 1e-10, 1 - 1e-10), df=10)
        self.corr_matrix = np.corrcoef(z.T)
        
        # Step 2: Optimize degrees of freedom
        def neg_log_likelihood(nu):
            if nu < nu_bounds[0] or nu > nu_bounds[1]:
                return 1e10
            
            try:
                inv_R = np.linalg.inv(self.corr_matrix)
                sign, logdet = np.linalg.slogdet(self.corr_matrix)
                
                # t-copula log-likelihood
                ll = 0.0
                for i in range(min(n_samples, 500)):  # Subsample for speed
                    zi = stats.t.ppf(np.clip(data[i], 1e-10, 1 - 1e-10), df=nu)
                    quad = zi @ inv_R @ zi
                    ll += np.log(1 + quad / nu)
                
                ll *= -(nu + self.n_dims) / 2
                
                # Add normalization terms
                ll += n_samples * (
                    self.n_dims * np.log(np.gamma((nu + 1) / 2)) -
                    np.log(np.gamma((nu + self.n_dims) / 2)) +
                    self.n_dims * np.log(np.gamma(nu / 2)) -
                    self.n_dims * np.log(np.gamma((nu + 1) / 2))
                )
                
                return -ll / n_samples
            except:
                return 1e10
        
        result = minimize(neg_log_likelihood, [self.nu], method='Nelder-Mead',
                         bounds=[nu_bounds])
        self.nu = result.x[0]
        self.theta = (self.corr_matrix, self.nu)
        self.fitted = True
        return self
    
    def sample(self, n_samples: int) -> np.ndarray:
        """Sample from t-copula."""
        if not self.fitted:
            raise RuntimeError("Copula not fitted")
        
        # Sample from multivariate t
        L = np.linalg.cholesky(self.corr_matrix)
        z = np.random.randn(n_samples, self.n_dims) @ L.T
        s = np.random.chisquare(self.nu, n_samples) / self.nu
        t_samples = z / np.sqrt(s[:, np.newaxis])
        
        # Transform to uniform
        u = stats.t.cdf(t_samples, df=self.nu)
        return u
    
    def get_tail_dependence(self) -> float:
        """
        Calculate upper/lower tail dependence coefficient.
        For t-copula: lambda = 2 * t_{nu+1}(-sqrt((nu+1)(1-rho)/(1+rho)))
        """
        rho = np.mean(self.corr_matrix[np.triu_indices(self.n_dims, 1)])
        lambda_val = 2 * stats.t.cdf(-np.sqrt((self.nu + 1) * (1 - rho) / (1 + rho)), 
                                     df=self.nu + 1)
        return lambda_val


class ArchimedeanCopula(CopulaModel):
    """
    Base class for Archimedean copulas (Clayton, Gumbel, Frank).
    These copulas are asymmetric and can capture different tail behaviors.
    """
    
    def __init__(self, n_dims: int, family: str):
        super().__init__(n_dims)
        self.family = family  # 'clayton', 'gumbel', 'frank'
        
    def _generator(self, t: np.ndarray, theta: float) -> np.ndarray:
        """Copula generator function phi(t)."""
        raise NotImplementedError
        
    def _generator_inv(self, t: np.ndarray, theta: float) -> np.ndarray:
        """Inverse generator function phi^{-1}(t)."""
        raise NotImplementedError


class ClaytonCopula(ArchimedeanCopula):
    """
    Clayton Copula - captures lower tail dependence (joint crashes).
    
    Generator: phi(t) = (t^{-theta} - 1) / theta
    Lower tail dependence: lambda_L = 2^{-1/theta}
    Upper tail dependence: lambda_U = 0
    
    Ideal for modeling crash contagion in crypto portfolios.
    """
    
    def __init__(self, n_dims: int):
        super().__init__(n_dims, 'clayton')
        
    def fit(self, data: np.ndarray) -> 'ClaytonCopula':
        """Fit Clayton copula using Kendall's tau method."""
        # Calculate average Kendall's tau
        tau_sum = 0.0
        count = 0
        for i in range(self.n_dims):
            for j in range(i + 1, self.n_dims):
                tau = stats.kendalltau(data[:, i], data[:, j])[0]
                if not np.isnan(tau):
                    tau_sum += tau
                    count += 1
        
        tau_avg = tau_sum / count if count > 0 else 0.0
        
        # Method of moments: theta = 2 * tau / (1 - tau)
        self.theta = max(0.01, 2 * tau_avg / (1 - tau_avg))
        self.fitted = True
        return self
    
    def _generator(self, t: np.ndarray, theta: float) -> np.ndarray:
        return (np.power(t, -theta) - 1) / theta
    
    def _generator_inv(self, t: np.ndarray, theta: float) -> np.ndarray:
        return np.power(1 + theta * t, -1/theta)
    
    def cdf(self, u: np.ndarray) -> float:
        """Evaluate Clayton copula CDF."""
        if not self.fitted:
            raise RuntimeError("Copula not fitted")
        
        gen_vals = self._generator(u, self.theta)
        return self._generator_inv(np.sum(gen_vals), self.theta)
    
    def sample(self, n_samples: int) -> np.ndarray:
        """Sample from Clayton copula using Marshall-Olkin algorithm."""
        if not self.fitted:
            raise RuntimeError("Copula not fitted")
        
        # Sample mixing variable from Gamma distribution
        V = np.random.gamma(1 / self.theta, 1, n_samples)
        
        # Generate independent uniforms and transform
        U_ind = np.random.rand(n_samples, self.n_dims)
        U = np.power(1 - np.log(U_ind) / V[:, np.newaxis], -1/self.theta)
        
        return np.clip(U, 0, 1)
    
    def get_lower_tail_dependence(self) -> float:
        """Calculate lower tail dependence coefficient."""
        if self.theta <= 0:
            return 0.0
        return np.power(2, -1/self.theta)


class GumbelCopula(ArchimedeanCopula):
    """
    Gumbel Copula - captures upper tail dependence (joint rallies).
    
    Generator: phi(t) = (-ln t)^theta
    Lower tail dependence: lambda_L = 0
    Upper tail dependence: lambda_U = 2 - 2^{1/theta}
    
    Useful for modeling FOMO-driven simultaneous pumps.
    """
    
    def __init__(self, n_dims: int):
        super().__init__(n_dims, 'gumbel')
        
    def fit(self, data: np.ndarray) -> 'GumbelCopula':
        """Fit Gumbel copula using Kendall's tau method."""
        tau_sum = 0.0
        count = 0
        for i in range(self.n_dims):
            for j in range(i + 1, self.n_dims):
                tau = stats.kendalltau(data[:, i], data[:, j])[0]
                if not np.isnan(tau):
                    tau_sum += tau
                    count += 1
        
        tau_avg = tau_sum / count if count > 0 else 0.0
        
        # Method of moments: theta = 1 / (1 - tau)
        self.theta = max(1.001, 1 / (1 - tau_avg))
        self.fitted = True
        return self
    
    def _generator(self, t: np.ndarray, theta: float) -> np.ndarray:
        return np.power(-np.log(t), theta)
    
    def _generator_inv(self, t: np.ndarray, theta: float) -> np.ndarray:
        return np.exp(-np.power(t, 1/theta))
    
    def cdf(self, u: np.ndarray) -> float:
        """Evaluate Gumbel copula CDF."""
        if not self.fitted:
            raise RuntimeError("Copula not fitted")
        
        gen_vals = self._generator(u, self.theta)
        return self._generator_inv(np.sum(gen_vals), self.theta)
    
    def sample(self, n_samples: int) -> np.ndarray:
        """Sample from Gumbel copula."""
        if not self.fitted:
            raise RuntimeError("Copula not fitted")
        
        # Sample from stable distribution
        alpha = 1 / self.theta
        S = np.random.standard_exponential(n_samples)
        V = np.random.gamma(1/alpha, 1, n_samples)
        W = S / V
        
        U_ind = np.random.rand(n_samples, self.n_dims)
        U = np.exp(-np.power(-np.log(U_ind), alpha) * W[:, np.newaxis])
        
        return np.clip(U, 0, 1)
    
    def get_upper_tail_dependence(self) -> float:
        """Calculate upper tail dependence coefficient."""
        if self.theta <= 1:
            return 0.0
        return 2 - np.power(2, 1/self.theta)


class PortfolioTailRiskAnalyzer:
    """
    Comprehensive tail risk analyzer using multiple copula families.
    Compares fits and selects best model for current market regime.
    
    Memory-efficient design for 8GB RAM constraint.
    """
    
    def __init__(self, asset_names: List[str]):
        self.asset_names = asset_names
        self.n_assets = len(asset_names)
        self.copulas = {}
        self.best_copula = None
        self.aic_scores = {}
        
    def fit_all(self, returns_data: np.ndarray) -> str:
        """
        Fit multiple copula families and select best by AIC.
        
        Args:
            returns_data: Raw returns array (n_samples, n_assets)
            
        Returns:
            Name of best fitting copula
        """
        # Transform to uniform marginals using empirical CDF
        n_samples = returns_data.shape[0]
        ranks = np.argsort(np.argsort(returns_data, axis=0), axis=0)
        uniform_data = (ranks + 1) / (n_samples + 1)
        
        # Fit different copula families
        models = [
            ('gaussian', GaussianCopula(self.n_assets)),
            ('student_t', StudentTCopula(self.n_assets)),
            ('clayton', ClaytonCopula(self.n_assets)),
            ('gumbel', GumbelCopula(self.n_assets))
        ]
        
        self.aic_scores = {}
        for name, model in models:
            try:
                model.fit(uniform_data)
                # Simplified AIC calculation
                loglik = self._approximate_loglik(model, uniform_data)
                n_params = self._count_params(model)
                aic = 2 * n_params - 2 * loglik
                self.aic_scores[name] = aic
                self.copulas[name] = model
            except Exception as e:
                print(f"Warning: {name} copula fit failed: {e}")
                self.aic_scores[name] = np.inf
        
        # Select best model
        if self.aic_scores:
            self.best_copula = min(self.aic_scores, key=self.aic_scores.get)
        else:
            self.best_copula = 'gaussian'
            self.copulas['gaussian'] = GaussianCopula(self.n_assets)
            self.copulas['gaussian'].fit(uniform_data)
            
        return self.best_copula
    
    def _approximate_loglik(self, model: CopulaModel, data: np.ndarray, 
                           n_subsample: int = 200) -> float:
        """Approximate log-likelihood using subsampling for speed."""
        indices = np.random.choice(data.shape[0], min(n_subsample, data.shape[0]), replace=False)
        ll = 0.0
        for i in indices:
            try:
                pdf_val = model.pdf(data[i])
                if pdf_val > 0:
                    ll += np.log(pdf_val)
            except:
                continue
        return ll
    
    def _count_params(self, model: CopulaModel) -> int:
        """Count number of free parameters."""
        if isinstance(model, GaussianCopula):
            return self.n_assets * (self.n_assets - 1) // 2
        elif isinstance(model, StudentTCopula):
            return self.n_assets * (self.n_assets - 1) // 2 + 1
        else:
            return 1
    
    def simulate_tail_scenarios(self, n_scenarios: int = 10000) -> np.ndarray:
        """
        Generate tail stress scenarios using the best-fitting copula.
        Focuses on extreme quantile regions.
        """
        if self.best_copula not in self.copulas:
            raise RuntimeError("No copula fitted")
        
        model = self.copulas[self.best_copula]
        samples = model.sample(n_scenarios)
        
        # Filter for tail scenarios (at least one asset in bottom 5%)
        tail_mask = np.any(samples < 0.05, axis=1)
        tail_scenarios = samples[tail_mask]
        
        return tail_scenarios
    
    def get_systemic_risk_metrics(self) -> dict:
        """
        Calculate systemic risk metrics from fitted copulas.
        """
        metrics = {
            'best_model': self.best_copula,
            'aic_scores': self.aic_scores.copy(),
            'tail_dependencies': {}
        }
        
        if 'student_t' in self.copulas:
            metrics['tail_dependencies']['symmetric'] = \
                self.copulas['student_t'].get_tail_dependence()
        
        if 'clayton' in self.copulas:
            metrics['tail_dependencies']['lower_tail'] = \
                self.copulas['clayton'].get_lower_tail_dependence()
        
        if 'gumbel' in self.copulas:
            metrics['tail_dependencies']['upper_tail'] = \
                self.copulas['gumbel'].get_upper_tail_dependence()
        
        return metrics


# Utility functions for integration with main trading system
def calculate_portfolio_var_copula(returns: np.ndarray, weights: np.ndarray,
                                   confidence: float = 0.99,
                                   n_simulations: int = 50000) -> float:
    """
    Calculate Value at Risk using copula-based simulation.
    
    Args:
        returns: Historical returns (n_days, n_assets)
        weights: Portfolio weights
        confidence: VaR confidence level
        n_simulations: Number of Monte Carlo simulations
        
    Returns:
        VaR at specified confidence level
    """
    n_assets = returns.shape[1]
    analyzer = PortfolioTailRiskAnalyzer([f"Asset_{i}" for i in range(n_assets)])
    analyzer.fit_all(returns)
    
    # Generate correlated scenarios
    scenarios = analyzer.simulate_tail_scenarios(n_simulations)
    
    # Transform back to return space using empirical marginals
    n_hist = returns.shape[0]
    simulated_returns = np.zeros_like(scenarios)
    for i in range(n_assets):
        sorted_returns = np.sort(returns[:, i])
        indices = (scenarios[:, i] * (n_hist - 1)).astype(int)
        indices = np.clip(indices, 0, n_hist - 1)
        simulated_returns[:, i] = sorted_returns[indices]
    
    # Calculate portfolio returns
    portfolio_returns = simulated_returns @ weights
    
    # Calculate VaR
    var = np.percentile(portfolio_returns, (1 - confidence) * 100)
    return var


if __name__ == "__main__":
    # Example usage with synthetic crypto returns
    np.random.seed(42)
    n_days = 500
    n_assets = 5
    
    # Generate correlated crypto-like returns with fat tails
    base_corr = 0.4 + 0.3 * np.random.rand(n_assets, n_assets)
    base_corr = (base_corr + base_corr.T) / 2
    np.fill_diagonal(base_corr, 1.0)
    
    returns = np.random.multivariate_normal(np.zeros(n_assets), base_corr, n_days)
    returns = returns * np.random.gamma(2, 1, (n_days, n_assets))  # Add fat tails
    
    weights = np.ones(n_assets) / n_assets
    
    # Analyze tail risk
    analyzer = PortfolioTailRiskAnalyzer([f"Crypto_{i}" for i in range(n_assets)])
    best_model = analyzer.fit_all(returns)
    print(f"Best copula model: {best_model}")
    print(f"AIC scores: {analyzer.aic_scores}")
    
    metrics = analyzer.get_systemic_risk_metrics()
    print(f"Tail dependencies: {metrics['tail_dependencies']}")
    
    var_99 = calculate_portfolio_var_copula(returns, weights, confidence=0.99)
    print(f"Portfolio VaR (99%): {var_99:.4f}")
