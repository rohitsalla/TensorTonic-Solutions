# <span style="font-size: 20px;">Lasso Regression</span>

<span style="font-size: 14px;">Lasso (Least Absolute Shrinkage and Selection Operator) regression adds an L1 penalty to the ordinary least squares objective. Its distinctive property is that it can set coefficients exactly to zero, performing automatic feature selection.</span>

---

## <span style="font-size: 16px;">Motivation</span>

- <span style="font-size: 14px;">When many features are available, some may be irrelevant or redundant</span>
- <span style="font-size: 14px;">Ridge regression shrinks all coefficients but never eliminates any</span>
- <span style="font-size: 14px;">Lasso produces sparse models by zeroing out unimportant features, making the model more interpretable</span>

---

## <span style="font-size: 16px;">Objective Function</span>

<span style="font-size: 14px;">The Lasso objective minimizes:</span>

$$
J(w) = \|y - Xw\|^2 + \alpha \|w_{1:}\|_1
$$

- $\|w_{1:}\|_1 = \sum_{j=1}^{d} |w_j|$ <span style="font-size: 14px;">is the L1 norm (sum of absolute values)</span>
- <span style="font-size: 14px;">The intercept</span> $w_0$ <span style="font-size: 14px;">is excluded from the penalty</span>
- $\alpha > 0$ <span style="font-size: 14px;">controls the regularization strength</span>

---

## <span style="font-size: 16px;">Why No Closed-Form Solution</span>

- <span style="font-size: 14px;">The absolute value</span> $|w_j|$ <span style="font-size: 14px;">is not differentiable at</span> $w_j = 0$
- <span style="font-size: 14px;">Unlike Ridge, there is no matrix equation that directly gives the optimal weights</span>
- <span style="font-size: 14px;">Instead, we use an iterative algorithm called coordinate descent</span>

---

## <span style="font-size: 16px;">Coordinate Descent</span>

<span style="font-size: 14px;">The idea is to optimize one coefficient at a time while holding all others fixed:</span>

- <span style="font-size: 14px;">Compute the partial residual:</span> $r_j = y - Xw + X_j w_j$
- <span style="font-size: 14px;">This represents what the model predicts without the contribution of feature</span> $j$
- <span style="font-size: 14px;">Compute</span> $\rho_j = X_j^T r_j$ <span style="font-size: 14px;">(correlation of feature</span> $j$ <span style="font-size: 14px;">with the partial residual)</span>
- <span style="font-size: 14px;">Compute</span> $z_j = X_j^T X_j$ <span style="font-size: 14px;">(squared norm of feature column)</span>

---

## <span style="font-size: 16px;">Soft Thresholding</span>

<span style="font-size: 14px;">The optimal update for each slope coefficient is:</span>

$$
w_j = \frac{\text{sign}(\rho_j) \cdot \max(|\rho_j| - \alpha, 0)}{z_j}
$$

- <span style="font-size: 14px;">If</span> $|\rho_j| \leq \alpha$<span style="font-size: 14px;">, the penalty outweighs the benefit and</span> $w_j = 0$
- <span style="font-size: 14px;">If</span> $|\rho_j| > \alpha$<span style="font-size: 14px;">, the coefficient is shrunk by</span> $\alpha$ <span style="font-size: 14px;">toward zero</span>
- <span style="font-size: 14px;">This is the mechanism that produces exact zeros and enables feature selection</span>

---

## <span style="font-size: 16px;">Feature Standardization</span>

- <span style="font-size: 14px;">Features must be standardized before applying Lasso so the penalty treats all features equally</span>
- <span style="font-size: 14px;">Each feature is transformed as:</span> $x_j' = \frac{x_j - \mu_j}{\sigma_j}$
- <span style="font-size: 14px;">Mean and standard deviation are computed from training data only</span>
- <span style="font-size: 14px;">The same transformation is applied to test data</span>

---

## <span style="font-size: 16px;">Effect of Alpha</span>

- <span style="font-size: 14px;">When</span> $\alpha \to 0$<span style="font-size: 14px;">, Lasso converges to the OLS solution</span>
- <span style="font-size: 14px;">As</span> $\alpha$ <span style="font-size: 14px;">increases, weaker features are zeroed out first</span>
- <span style="font-size: 14px;">When</span> $\alpha$ <span style="font-size: 14px;">is very large, all features are eliminated and predictions equal</span> $\bar{y}$

---

## <span style="font-size: 16px;">Lasso vs Ridge</span>

- <span style="font-size: 14px;">Ridge (L2): shrinks all coefficients but never zeros them out. Better when all features contribute.</span>
- <span style="font-size: 14px;">Lasso (L1): zeros out unimportant features. Better when only a few features truly matter.</span>
- <span style="font-size: 14px;">The geometric intuition: L1 constraint has corners at the axes where coefficients are exactly zero. L2 constraint is a smooth sphere with no corners.</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

- <span style="font-size: 14px;">**Q: Lasso vs Ridge?**</span>
  <span style="font-size: 14px;">A: Lasso (L1) drives coefficients to exactly zero, performing feature selection. Ridge (L2) shrinks coefficients but never to zero. Use Lasso when you believe only a few features matter; Ridge when all features contribute</span>

- <span style="font-size: 14px;">**Q: Why coordinate descent?**</span>
  <span style="font-size: 14px;">A: The L1 penalty makes the objective non-differentiable at $w_j = 0$. Standard gradient descent cannot handle this kink. Coordinate descent solves a one-dimensional subproblem for each coefficient, which has a closed-form solution via soft thresholding</span>

- <span style="font-size: 14px;">**Q: What is soft thresholding?**</span>
  <span style="font-size: 14px;">A: The operator $S(\rho, \alpha) = \text{sign}(\rho) \cdot \max(|\rho| - \alpha, 0)$ shrinks the correlation toward zero and clips it to exactly zero when its magnitude is below $\alpha$</span>

- <span style="font-size: 14px;">**Q: When does Lasso fail?**</span>
  <span style="font-size: 14px;">A: When features are highly correlated (grouped), Lasso tends to select one and zero out the rest arbitrarily. Elastic Net (L1+L2) handles this by encouraging correlated features to share similar coefficients</span>

- <span style="font-size: 14px;">**Q: How to choose alpha?**</span>
  <span style="font-size: 14px;">A: Use cross-validation over a log-scale grid. The regularization path (coefficients as a function of alpha) shows which features are selected at each penalty level</span>

- <span style="font-size: 14px;">**Q: Proximal gradient alternative?**</span>
  <span style="font-size: 14px;">A: Instead of coordinate descent, Lasso can be solved with proximal gradient descent (ISTA/FISTA), which applies soft thresholding to the entire gradient step at once</span>

---