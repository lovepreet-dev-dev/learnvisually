(function () {
  const categories = [
    {
      id: "foundations",
      title: "Foundations",
      page: "introduction.html",
      description:
        "Overview of machine learning, learning paradigms, generative modeling, parameter estimation, estimator quality, noisy data, density estimation, applications, and tools.",
    },
    {
      id: "classification",
      title: "Classification",
      page: "classification.html",
      description:
        "Instance-based, probabilistic, discriminative, tree-based, and margin-based classification methods.",
    },
    {
      id: "graphical-models",
      title: "Graphical Models",
      page: "graphical-models.html",
      description:
        "Bayesian networks, Markov random fields, inference, Markov chains, hidden states, decoding, and parameter learning.",
    },
    {
      id: "clustering",
      title: "Clustering",
      page: "clustering.html",
      description:
        "Partition-based, hierarchical, density-based, and graph-based clustering methods.",
    },
    {
      id: "neural",
      title: "Neural Networks",
      page: "neural.html",
      description:
        "Activation functions, gradient descent and backpropagation, then attention, the Transformer, large language models, and how small LLMs are made with distillation, pruning and quantization.",
    },
  ];

  function details(intuition, math, use, caution, analogy = null, keyFormulas = null) {
    return { intuition, math, use, caution, analogy, keyFormulas };
  }

  function page(definition) {
    return definition;
  }

  function gaussianPage(id, title, focus, summary, extra) {
    return page({
      id,
      title,
      category: "foundations",
      engine: "gaussian",
      options: { focus },
      subtitle: summary,
      summary,
      detail: extra,
    });
  }

  function linearPage(id, title, algorithm, summary, extra, options = {}) {
    return page({
      id,
      title,
      category: "classification",
      engine: "linear",
      options: Object.assign({ algorithm }, options),
      subtitle: summary,
      summary,
      detail: extra,
    });
  }

  function hmmPage(id, title, focus, summary, extra) {
    return page({
      id,
      title,
      category: "graphical-models",
      engine: "hmm",
      options: { focus },
      subtitle: summary,
      summary,
      detail: extra,
    });
  }

  function partitionPage(id, title, focus, summary, extra) {
    return page({
      id,
      title,
      category: "clustering",
      engine: "partition",
      options: { focus },
      subtitle: summary,
      summary,
      detail: extra,
    });
  }

  function hierarchyPage(id, title, focus, summary, extra) {
    return page({
      id,
      title,
      category: "clustering",
      engine: "hierarchy",
      options: { focus },
      subtitle: summary,
      summary,
      detail: extra,
    });
  }

  const allPages = [
    page({
      id: "ml-overview",
      title: "Overview of Machine Learning",
      category: "foundations",
      engine: "concept-overview",
      subtitle: "See the end-to-end learning workflow from data definition to validation and deployment.",
      summary:
        "See the end-to-end learning workflow from data definition to validation and deployment.",
      detail: details(
        "Machine learning is the discipline of choosing a parameterized model and adjusting it so that it performs well on data. The defining move is not memorizing a fixed procedure, but building a loop: define a task, choose an objective, optimize parameters, and evaluate generalization.",
        "The canonical problem is empirical risk minimization: finding model parameters θ that minimize the average loss L over a dataset D.",
        "Use this frame before diving into a specific algorithm. Ask: what are the inputs, what does the model output, what scalar quantity is being optimized, and how is success measured?",
        "A common failure mode is confusing training performance with real-world success. A model can perfectly fit the training set (zero loss) and still fail dramatically under distribution shift or out-of-sample data.",
        [
          "Think of ML as writing software where the 'code' is derived from examples rather than typed line-by-line by a programmer."
        ],
        [
          `θ* = argmin_θ (1/N) Σ_{i=1}^N L(y_i, f_θ(x_i))`
        ]
      ),
    }),
    page({
      id: "different-forms-of-learning",
      title: "Different Forms of Learning",
      category: "foundations",
      engine: "concept-forms",
      subtitle: "Compare supervised, unsupervised, semi-supervised, reinforcement, and generative paradigms.",
      summary: "Compare supervised, unsupervised, semi-supervised, reinforcement, and generative views.",
      detail: details(
        "Different ML paradigms are defined by the environment's feedback. Supervised has explicit targets; unsupervised has only inputs; reinforcement has delayed rewards; generative focuses on how data arise.",
        "Mathematical objectives divert sharply here. Supervised minimizes prediction loss; unsupervised optimizes structural criteria (distortion or likelihood); generative factors the joint distribution p(x,y).",
        "Match the paradigm to your data reality. Don't force supervised classification if you have zero labels. If uncertainty is exactly the focus, generative models are superior.",
        "Many practitioners treat these as isolated boxes. In reality, modern pipelines combine them—for example, unsupervised pretraining followed by supervised fine-tuning.",
        [
          "Supervised: Learning with a teacher giving exact answers.",
          "Unsupervised: Finding patterns in the dark without a guide.",
          "Reinforcement: Training a dog with treats for good behaviour over time."
        ],
        [
          `Supervised: min_θ E[L(y, f_θ(x))]`,
          `Unsupervised: max_θ E[log p_θ(x)]`,
          `RL: max_π E_{τ~π}[Σ γ^t R_t]`
        ]
      ),
    }),
    gaussianPage(
      "generative-learning",
      "Generative Learning",
      "generative",
      "Model how observations are produced, then update beliefs about the hidden parameters.",
      details(
        "Generative learning models the full story of how data comes to exist. Instead of just drawing a boundary between classes, you model the source distribution itself.",
        "By modeling the joint distribution p(x, y) = p(y)p(x|y), generative models can leverage Bayes' rule to compute p(y|x) for prediction, or sample from p(x) to synthesize new data.",
        "Use generative learning when uncertainty calibration matters, when you need to handle missing features gracefully, or when simulating realistic new data is a primary requirement.",
        "Because generative models must solve a harder problem (modeling the exact data density), they often underperform discriminative models on pure accuracy metrics if the generative assumptions are wrong.",
        [
          "Instead of just learning the difference between a dog and a cat (discriminative), learn the rules for drawing a dog and drawing a cat (generative)."
        ],
        [
          `p(x, y) = p(y)p(x|y)`,
          `p(y|x) = [p(x|y)p(y)] / p(x)`
        ]
      )
    ),
    gaussianPage(
      "gaussian-parameter-estimation",
      "Gaussian Parameter Estimation",
      "gaussian",
      "Estimate the mean and spread of a Gaussian source from observed data.",
      details(
        "Estimating parameters means compressing a large cloud of data points into a few summary numbers—like the centre of mass and the radius of spread—that describe the underlying source.",
        "For a Gaussian, the sample mean x̄ and unbiased sample variance s² are sufficient statistics. They contain all the information from the dataset needed to estimate the true μ and σ².",
        "Gaussians reflect the Central Limit Theorem. Use this estimation when working with sums of many small independent effects, such as measurement noise, sensor error, or biological traits.",
        "Gaussians have notoriously thin tails. A single extreme outlier will heavily pull the sample mean and explode the sample variance. Use robust estimators (like median) if outliers are common.",
        [
          "Treating the data like physical weights on a beam: the mean is the exact balance point (center of mass), and variance is the moment of inertia."
        ],
        [
          `μ̂ = x̄ = (1/n) Σ x_i`,
          `σ̂² = s² = 1/(n-1) Σ (x_i - x̄)²`
        ]
      )
    ),
    gaussianPage(
      "maximum-likelihood-estimation",
      "Maximum Likelihood Estimation",
      "mle",
      "Choose parameter values that make the observed data most probable under the model.",
      details(
        "Maximum likelihood estimation asks a direct question: for which parameter values would the observed dataset have been most likely? In many classical models, this produces simple analytic estimators.",
        "On the page the likelihood is written explicitly as a product over independent data points, then converted into a more convenient log-likelihood sum. You can inspect the substitution from your sample into the Gaussian MLE formulas.",
        "Use MLE when you want a clean, data-driven estimate without injecting prior beliefs. It is the default estimator behind a large amount of classical statistics and deep learning (cross-entropy is MLE).",
        "MLE can overreact when sample size is small or when parameters are weakly identified. It will happily assign zero probability to unseen events, which is catastrophic in language modeling.",
        [
          "Adjusting the dials on a machine until the sound it produces perfectly matches the recording you have."
        ],
        [
          `θ_{MLE} = argmax_θ ∏ p(x_i | θ)`,
          `L(θ) = argmax_θ Σ log p(x_i | θ)`
        ]
      )
    ),
    gaussianPage(
      "map-estimation",
      "MAP Estimation",
      "map",
      "Combine prior preference and data evidence, then choose the posterior mode.",
      details(
        "Maximum a posteriori estimation modifies pure likelihood with prior information. Instead of only asking what fits the data best, it asks what parameter value is most plausible after seeing both prior belief and evidence.",
        "The page shows the posterior as proportional to likelihood times prior. Taking the log shows that MAP is exactly equivalent to MLE plus a regularization penalty (like L2 or L1).",
        "Use MAP when you need regularization or prior structure to reliably fit a model on small datasets. Most regularized machine learning can be interpreted as MAP estimation.",
        "A MAP estimate is still just one point from the posterior. It is also not invariant under reparameterization: changing the units of measurement can literally move the MAP peak.",
        [
          "Tuning the dials on the machine, but with sturdy rubber bands pulling the dials toward their factory default settings."
        ],
        [
          `θ_{MAP} = argmax_θ p(X|θ)p(θ)`,
          `log p(θ|X) ∝ Σ log p(x_i|θ) + log p(θ)`
        ]
      )
    ),
    gaussianPage(
      "bayesian-estimation",
      "Bayesian Estimation",
      "bayesian",
      "Update a full posterior distribution instead of keeping only one best estimate.",
      details(
        "Bayesian estimation treats parameters as uncertain random variables. Before seeing data you have a prior distribution; after seeing data you have a posterior distribution; prediction is done by integrating over all plausible parameters.",
        "The page makes the posterior precision update explicit, showing how prior precision and data precision add. That is why conjugate Bayesian updates are so transparent in Gaussian models.",
        "Use full Bayesian estimation when uncertainty intervals, sequential updating, robust decision-making under risk, or predictive calibration are central to the task.",
        "The tradeoff is heavy computational cost. Simple conjugate cases are elegant, but general Bayesian models require approximation methods such as variational inference or MCMC.",
        [
          "Instead of betting all your money on the single most likely horse winning, you spread your bets across all horses proportional to their odds."
        ],
        [
          `p(θ|X) = p(X|θ)p(θ) / ∫p(X|θ')p(θ')dθ'`,
          `p(x_{new}|X) = ∫ p(x_{new}|θ)p(θ|X)dθ`
        ]
      )
    ),
    page({
      id: "bias-and-variance-of-estimators",
      title: "Bias and Variance of Estimators",
      category: "foundations",
      engine: "bias-variance",
      subtitle: "See how repeated sampling creates a distribution over estimators, not just one answer.",
      summary:
        "See how repeated sampling creates a distribution over estimators, not just one answer.",
      detail: details(
        "An estimator's output is random because the dataset it trains on is random. Bias measures systematic displacement from the truth, while variance measures its sensitivity to the exact sample that was drawn.",
        "The page computes empirical bias, variance, and mean squared error from repeated simulations. The decomposition MSE = bias² + variance is a fundamental theorem of statistics.",
        "Use this lens whenever you choose hyperparameters (like tree depth or regularization). A 'worse' (biased) estimator can easily outperform an unbiased one if it trades a little bias for a massive variance reduction.",
        "Zero bias is completely useless if variance is explosive. High-degree polynomial regression is unbiased but typically produces wild, unusable predictions on test data.",
        [
          "Bias is a bent rifle barrel; all shots clump together but miss the bullseye.",
          "Variance is a shaky marksman; shots are centered on the bullseye but scattered widely."
        ],
        [
          `E[(θ̂ - θ)^2] = (E[θ̂] - θ)^2 + E[(θ̂ - E[θ̂])^2]`,
          `MSE = Bias² + Variance`
        ]
      ),
    }),
    page({
      id: "missing-and-noisy-features",
      title: "Missing and Noisy Features",
      category: "foundations",
      engine: "cleanup",
      subtitle: "Preprocessing decisions change the data distribution before the model even begins.",
      summary: "Preprocessing decisions change the data distribution before the model even begins.",
      detail: details(
        "Missing and noisy features are not minor 'cleaning' issues; they directly alter the information geometry that reaches the learner. Imputation hallucinates data, and clipping destroys genuine outliers.",
        "Applying mean imputation artificially spikes the marginal distribution at the mean, shrinking the apparent variance. Z-score clamping behaves as a non-linear squashing function on the tails.",
        "Use this page to build skepticism before fitting classifiers on imperfect data. Characterizing the missingness mechanism (MCAR, MAR, MNAR) dictates what imputation is mathematically permissible.",
        "Naive preprocessing injects false certainty. Mean imputation artificially tightens confidence intervals, dropping rows distorts class balance, and aggressive clamping erases the real signal along with noise.",
        [
          "Imputing the mean is like filling in a damaged ancient fresco with beige paint. It stops your eyes from tripping, but the historical information is still gone."
        ],
        [
          `x_{imp} = x if x \\neq ∅, else (1/N)Σx`,
          `x_{clip} = max(min(x, μ + kσ), μ - kσ)`
        ]
      ),
    }),
    page({
      id: "nonparametric-density-estimation",
      title: "Nonparametric Density Estimation",
      category: "foundations",
      engine: "kde",
      subtitle: "Build a density estimate by summing local kernels instead of assuming one global Gaussian.",
      summary: "Build a density estimate by summing local kernels instead of assuming one global Gaussian.",
      detail: details(
        "Nonparametric density estimation (KDE) avoids committing to a fixed, rigid parametric shape. Instead, it lets the observed data dictate the contours, placing a bump of probability mass on every point.",
        "The KDE formula is literally the average of N kernel functions (like narrow Gaussians) centered on the N training points. The bandwidth hyperparameter controls the spread of each bump.",
        "Use KDE for exploratory data analysis, discovering multiple modes (peaks), anomaly detection, or when the true distribution is clearly not a simple Gaussian.",
        "Bandwidth choice is critical: too narrow memorizes the sample (spiky, high variance); too wide hides the real structure (smooth, high bias). It also scales horribly to high dimensions (Curse of Dimensionality).",
        [
          "Throwing a handful of sand on a table. KDE doesn't try to draw one giant circle over it; it just sweeps a small smoothing brush over every individual grain."
        ],
        [
          `p̂(x) = (1 / N h) Σ_{i=1}^N K((x - x_i) / h)`
        ]
      ),
    }),
    page({
      id: "applications-and-software-tools",
      title: "Applications and Software Tools",
      category: "foundations",
      engine: "concept-tools",
      subtitle: "Connect algorithms to domains and to the software stack used to build real systems.",
      summary:
        "Connect algorithms to domains and to the software stack used to build real systems.",
      detail: details(
        "Algorithms live inside pipelines. A useful mental model includes the domain, the data collection process, the modeling library, the evaluation setup, and the deployment constraints.",
        "The page links application settings to mathematical task types and common tools. For example, density models and Bayesian estimation connect naturally to PyMC or probabilistic programming, while large-scale discriminative training often lives in PyTorch or TensorFlow.",
        "Use this page to understand where each method fits in practice and which software ecosystem is typically used to implement it.",
        "Tool choice should follow the problem, not the other way around. A flashy framework does not fix weak data collection or a poorly matched objective."
      ),
    }),
    page({
      id: "nearest-neighbour",
      title: "Nearest Neighbour",
      category: "classification",
      engine: "knn",
      subtitle: "Predict from local evidence by comparing the query to labeled examples.",
      summary:
        "Predict from local evidence by comparing the query to labeled examples.",
      detail: details(
        "Nearest neighbour methods store the dataset and defer all computation until query time (lazy learning). The key idea is locality: points close together in feature space likely share the same label.",
        "The distance function (e.g., Euclidean or Manhattan) is the engine. The prediction is literally a majority vote or distance-weighted average of the k nearest known points.",
        "Use k-NN as a dead-simple baseline, when the dataset is modest, or when you need highly irregular, non-linear decision boundaries that perfectly trace the data.",
        "K-NN falls apart in high dimensions (the Curse of Dimensionality: every point becomes roughly equidistant). It is also intensely sensitive to feature scaling (always standardize inputs!) and irrelevant features.",
        [
          "Asking the 5 closest houses in a new neighborhood who they voted for, and assuming your new neighbor will vote the same way."
        ],
        [
          `d(x, q) = ||x - q||_2 = √(Σ(x_i - q_i)^2)`,
          `ŷ = argmax_c Σ_{i \in N_k(q)} I(y_i = c)`
        ]
      ),
    }),
    page({
      id: "decision-trees",
      title: "Decision Trees",
      category: "classification",
      engine: "tree",
      options: { mode: "classification", focus: "tree" },
      subtitle: "Split the feature space greedily using impurity reduction.",
      summary:
        "Split the feature space greedily using impurity reduction.",
      detail: details(
        "Decision trees recursively partition the input space using extremely simple yes/no questions. Each split is chosen greedily to maximize the immediate separation of classes.",
        "The algorithm evaluates every possible feature and threshold, calculating the Gini Impurity (or Entropy) of the resulting child nodes. It picks the split that yields the largest impurity reduction.",
        "Use decision trees when you absolutely need interpretability (e.g., medical diagnoses), or when you have mixed feature types (categorical + continuous) and want to skip scaling.",
        "A single deeply-grown tree will overfit wildly, memorizing isolated noisy points. Minor changes in the training data can completely change the tree structure, which is why Random Forests dominate in practice.",
        [
          "Playing a game of 20 Questions to guess an animal, picking the question that rules out the largest number of wrong answers at every step."
        ],
        [
          `Gini(t) = 1 - Σ_{c=1}^C (p_c)^2`,
          `Gain = Gini(parent) - [ (N_L/N)Gini(left) + (N_R/N)Gini(right) ]`
        ]
      ),
    }),
    linearPage(
      "linear-discriminant-analysis",
      "Linear Discriminant Analysis",
      "lda",
      "Derive a linear boundary from class means, shared covariance, and priors.",
      details(
        "LDA is a generative classifier disguised as a linear one. It assumes each class is a Gaussian blob and that all blobs have the exact same shape (shared covariance), just centered in different places.",
        "Because of the shared covariance assumption, the quadratic terms in the log-odds cancel out, leaving a perfectly flat, linear decision boundary defined by the vector Σ⁻¹(μ₁ - μ₀).",
        "Use LDA when you have a small dataset, classes are well separated, and the Gaussian assumption isn't wildly violated. It is a highly stable, fast baseline.",
        "If class shapes differ substantially (one is a tight ball, the other a long cigar), the shared covariance assumption fails and LDA misplaces the boundary. Use QDA instead.",
        [
          "Drawing a straight line exactly halfway between the centers of two identically shaped clouds of smoke."
        ],
        [
          `w = Σ^{-1}(μ_1 - μ_0)`,
          `δ_c(x) = x^T Σ^{-1} μ_c - 0.5 μ_c^T Σ^{-1} μ_c + log(P(y=c))`
        ]
      )
    ),
    linearPage(
      "logistic-regression",
      "Logistic Regression",
      "logistic",
      "Fit probabilities directly by minimizing log-loss.",
      details(
        "Logistic regression is the quintessential discriminative classifier. It doesn't care how the data was generated; it only aims to map inputs directly to the probability of class 1 using a squashing function.",
        "The model computes a linear score z = wᵀx + b, then passes it through the logistic sigmoid function σ(z) = 1/(1+e⁻ᶻ) to bound the output between 0 and 1. Training minimizes the cross-entropy loss via gradient descent.",
        "Use logistic regression natively as your first classification baseline. It provides incredibly well-calibrated probabilities, highly interpretable weights (odds ratios), and scales efficiently.",
        "It can only draw straight lines. If your data is a circle inside a ring (XOR problem), logistic regression will fail completely unless you manually engineer nonlinear features (like x²).",
        [
          "Applying a 'squash' filter to a raw tug-of-war score, so a +100 point lead reads as '99.9% sure they'll win' instead of an arbitrary unbounded number."
        ],
        [
          `P(y=1|x) = σ(w^T x) = 1 / (1 + exp(-w^T x))`,
          `L(w) = -Σ [y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)]`
        ]
      )
    ),
    linearPage(
      "perceptron",
      "Perceptron",
      "perceptron",
      "Correct mistakes with the simplest online linear update rule.",
      details(
        "The perceptron is the grandfather of neural networks. It is a strictly error-driven, online algorithm: it looks at one point, gases a prediction, and only adjusts its boundary if it makes a mistake.",
        "The update rule is beautiful in its simplicity: w ← w + y_i x_i. If the dot product sign contradicts the true label, you literally add or subtract the misclassified vector to pivot the boundary.",
        "Use the perceptron purely for educational intuition. It proves that a very simple, localized mistake-correction rule can globally separate data (if separable).",
        "It is fundamentally flawed for modern use: it will never stop twitching if data overlaps (not linearly separable), and it produces no probabilities or margin guarantees.",
        [
          "A blindfolded person guessing a boundary by walking until they bump into a wall, then turning slightly to avoid it."
        ],
        [
          `if y_i(w^T x_i) ≤ 0:`,
          `   w ← w + η y_i x_i`
        ]
      )
    ),
    linearPage(
      "large-margin-classification",
      "Large Margin Classification",
      "svm",
      "Push the boundary away from training points instead of only making them correct.",
      details(
        "Large-margin methods care about how confidently points are separated, not just whether they are on the correct side. A larger margin implies better generalization because the boundary has a wider 'safe zone'.",
        "The hinge loss function mathematically enforces this: if a point is on the correct side but within the margin, it still incurs a penalty. The update rule pushes the boundary until the margin distance is at least 1.",
        "Use large-margin thinking when you want robust separation that resists minor adversarial nudges or measurement noise in the test data.",
        "A strict margin can be hypersensitive to outliers. A single mislabeled training point can violently swing the boundary (the 'hard margin' problem), which is why we add slack variables (C parameter).",
        [
          "Driving down a road: you don't just want to be barely on the correct side of the center line, you want to drive exactly in the middle of your lane."
        ],
        [
          `L(w) = \\max(0, 1 - y_i(w^T x_i))`,
          `\\min_w ||w||^2 + C \\Sigma \\xi_i`
        ]
      )
    ),
    linearPage(
      "kernel-methods",
      "Kernel Methods",
      "svm",
      "Lift inputs into a richer feature space so a linear separator can solve a nonlinear problem.",
      details(
        "Kernel methods are the ultimate nonlinear 'cheat code'. Instead of trying to draw a wildly complex curved boundary in 2D, they mathematically lift the data into a high-dimensional space where a simple, flat plane can separate the classes.",
        "The 'Kernel Trick' avoids computing the high-dimensional coordinates explicitly. By just computing the similarity (dot product) between points, the optimization finds the boundary in the lifted space for free.",
        "Use kernels when your data is definitively not linearly separable (e.g., concentric circles, checkerboards) but you still want the mathematical guarantees of a convex optimizer (no local minima).",
        "Kernels scale poorly. The prediction cost grows with the number of training points (or support vectors). In the deep learning era, learned representations often replace fixed kernels.",
        [
          "If you have red and blue balls scattered on a flat sheet, a straight line can't separate them. But if you toss them into the air (add a Z axis), you can slide a flat sheet of cardboard right between them."
        ],
        [
          `K(x, z) = \\phi(x)^T \\phi(z)`,
          `K_{RBF}(x, z) = \\exp(-\\gamma ||x - z||^2)`
        ]
      ),
      { featureMap: "rbf" }
    ),
    linearPage(
      "support-vector-machines",
      "Support Vector Machines",
      "svm",
      "Optimize a large-margin separator and focus the solution on the boundary-critical points.",
      details(
        "Support Vector Machines combine the large-margin objective with the kernel trick. The boundary is completely defined by a sparse subset of training points called the 'support vectors'.",
        "The optimization uniquely relies only on the points that are actively contesting the boundary or violating the margin. Every other deeply correct point gets a weight (alpha) of precisely zero.",
        "Use SVMs (usually with an RBF kernel) as a heavyweight, highly optimized black-box baseline when you have a small-to-medium dataset (<100k rows) and nonlinear boundaries.",
        "SVMs are famously sensitive to hyperparameter tuning (C and Gamma). Guessing them randomly often fails; you essentially must do a grid search.",
        [
          "Building a fence to separate sheep from wolves. You only need to look at the animals standing right on the border to know where the fence must go; the animals deep inside the flock don't matter."
        ],
        [
          `f(x) = \\Sigma_{i=1}^N \\alpha_i y_i K(x_i, x) + b`,
          `\\alpha_i = 0 \\text{ for non-support vectors}`
        ]
      ),
      { featureMap: "rbf" }
    ),
    page({
      id: "classification-and-regression-trees",
      title: "Classification and Regression Trees",
      category: "classification",
      engine: "tree",
      options: { mode: "regression", focus: "cart" },
      subtitle: "Use the same tree idea for both class labels and continuous targets.",
      summary:
        "Use the same tree idea for both class labels and continuous targets.",
      detail: details(
        "CART unifies classification and regression trees. The recursive splitting logic is identical; only the mathematical scored used to pick the split changes. A regression tree simply predicts the average target value of all points in a leaf.",
        "Instead of Gini impurity, a regression tree minimizes the sum of squared errors (SSE) within each leaf. The split is chosen to minimize the variance of the target variable in the resulting left and right child nodes.",
        "Use CART for regression when the relationships are highly non-linear, discontinuous, or when you have many irrelevant features. Trees natively ignore useless variables.",
        "Because CART predicts a constant value per leaf, the prediction surface looks like a jagged staircase, not a smooth curve. It physically cannot extrapolate beyond the min/max values seen in training.",
        [
          "Sorting an unsorted pile of test scores into smaller and smaller boxes, where each box gets a sticky note with the average score written on it."
        ],
        [
          `ŷ_t = (1/N_t) \\Sigma_{i \\in t} y_i`,
          `Gain = SSE(parent) - [SSE(left) + SSE(right)]`
        ]
      ),
    }),
    page({
      id: "bayesian-networks",
      title: "Bayesian Networks",
      category: "graphical-models",
      engine: "bayes-net",
      options: { focus: "network" },
      subtitle: "Factor a joint distribution with a directed acyclic graph and local conditionals.",
      summary:
        "Factor a joint distribution with a directed acyclic graph and local conditionals.",
      detail: details(
        "Bayesian networks use directed arrows to tell a causal or conditional story. They break down a massive joint probability distribution into a simple chain of local cause-and-effect rules.",
        "The joint distribution is mathematically factored as the product of each variable given only its direct parents: P(A,B,C) = P(A)P(B|A)P(C|B).",
        "Use them when you understand the causal mechanisms (e.g., Disease -> Symptom -> Test Result) and want to do diagnostics (inferring Disease from Test Result).",
        "If you draw the arrows wrong (e.g., implying a symptom causes a disease), the network will confidently yield nonsensical inferences.",
        [
          "A rumor spreading through an office. Bob hears it from Alice, and Charlie hears it from Bob. Charlie doesn't need to talk to Alice; Bob is his only 'parent' for the information."
        ],
        [
          `P(X_1, \\dots, X_n) = \\prod_{i=1}^n P(X_i | Pa(X_i))`
        ]
      ),
    }),
    page({
      id: "conditional-independence",
      title: "Conditional Independence",
      category: "graphical-models",
      engine: "bayes-net",
      options: { focus: "independence" },
      subtitle: "Use graph structure to determine when information flow is blocked or active.",
      summary:
        "Use graph structure to determine when information flow is blocked or active.",
      detail: details(
        "Conditional independence is the language that lets graphical models scale. If two variables become independent once a third is known, the joint distribution can be factorized more efficiently.",
        "The page focuses on chains, forks, colliders, and separators. It makes the d-separation rules explicit so you can see why conditioning sometimes blocks a path and sometimes opens one.",
        "Use this page when you want to read a graphical model correctly instead of guessing dependence from the drawing.",
        "A frequent mistake is treating every path as the same. Colliders behave differently from chains and forks, and that difference matters enormously for inference."
      ),
    }),
    page({
      id: "markov-random-fields",
      title: "Markov Random Fields",
      category: "graphical-models",
      engine: "mrf",
      subtitle: "Model undirected dependencies through compatibility functions and energies.",
      summary:
        "Model undirected dependencies through compatibility functions and energies.",
      detail: details(
        "Markov random fields represent dependencies without assigning a directional parent-child story. They are often more natural when interactions are symmetric or when the main object is an energy function.",
        "The page shows the local conditional and the Gibbs-style update induced by pairwise compatibilities. It also makes the energy interpretation visible through compatibility scores.",
        "Use MRFs for spatial data, image labeling, smoothness priors, and other settings where neighboring variables influence each other symmetrically.",
        "The partition function Z can make exact probability calculations difficult. That is why approximate inference and sampling methods are so important in undirected models."
      ),
    }),
    page({
      id: "inference-in-graphical-models",
      title: "Inference in Graphical Models",
      category: "graphical-models",
      engine: "belief-propagation",
      options: { focus: "inference" },
      subtitle: "Compute marginals by pushing local information through the graph.",
      summary:
        "Compute marginals by pushing local information through the graph.",
      detail: details(
        "Inference asks what you can conclude about one variable after observing others. In graphical models, the answer comes from combining local factors without expanding the full joint distribution naively.",
        "The page uses message passing because it makes inference algebra local: each message summarizes how one part of the graph constrains another part.",
        "Use this page to build intuition for exact inference on trees and for why graph structure changes computational difficulty so dramatically.",
        "Exact inference is easy on some structures and hard on others. The same factorization idea can lead to very different computational costs depending on loops and treewidth."
      ),
    }),
    page({
      id: "belief-propagation",
      title: "Belief Propagation",
      category: "graphical-models",
      engine: "belief-propagation",
      options: { focus: "bp" },
      subtitle: "Send messages between factors and variables to compute marginals.",
      summary:
        "Send messages between factors and variables to compute marginals.",
      detail: details(
        "Belief propagation decomposes global inference into message updates on edges. Each message is a compact summary of how one side of the graph constrains the other.",
        "The page shows the binary message vectors numerically at every step so you can see how local evidence is transformed by pairwise compatibility and then normalized into beliefs.",
        "Use belief propagation on tree-structured models for exact marginals or on loopy graphs when a practical approximation is acceptable.",
        "On loopy graphs, belief propagation is no longer guaranteed to be exact or even to converge. The updates remain useful, but the interpretation becomes approximate."
      ),
    }),
    page({
      id: "markov-models",
      title: "Markov Models",
      category: "graphical-models",
      engine: "markov",
      subtitle: "Propagate a visible-state distribution forward with a transition matrix.",
      summary:
        "Propagate a visible-state distribution forward with a transition matrix.",
      detail: details(
        "A Markov model assumes the present state summarizes all relevant past information for the future. That simplifies sequence modeling to repeated matrix multiplication by transition probabilities.",
        "The page shows the distribution update p_t = p_{t-1}T step by step. Because the states are visible, there is no emission uncertainty yet; the transition model is the whole story.",
        "Use Markov models for simple sequence dynamics, queueing systems, population state evolution, and as the starting point before hidden-state models.",
        "The Markov assumption can be too restrictive if long-range history matters. In that case the state space must be enlarged or the model class changed."
      ),
    }),
    hmmPage(
      "hidden-markov-models",
      "Hidden Markov Models",
      "general",
      "Combine hidden state transitions with noisy observations.",
      details(
        "Hidden Markov Models treat reality like a shadow play. There is a true sequence of events happening privately (hidden states), and all we get to see are the noisy, probabilistic emissions (observations) they cast.",
        "The model is built on two matrices: a Transition matrix (how states jump from one to another) and an Emission matrix (what a given state typically produces).",
        "Use HMMs whenever you have time-series that jump between distinct regimes—like recognizing spoken phonemes from audio waves, or predicting market bull/bear phases from prices.",
        "The 'Markov' assumption means a state has no memory of how it got there. If a process depends heavily on long-term history, an HMM will fail terribly compared to an RNN.",
        [
          "Guessing the weather inside a windowless room by observing only whether the person walking in is carrying an umbrella."
        ],
        [
          `P(X, Y) = P(X_1) P(Y_1 | X_1) \\prod_{t=2}^T P(X_t | X_{t-1}) P(Y_t | X_t)`
        ]
      )
    ),
    hmmPage(
      "decoding-states-from-observations",
      "Decoding States from Observations",
      "decoding",
      "Use Viterbi recursion to recover the most likely hidden-state path.",
      details(
        "Decoding goes beyond just asking 'what is the most likely state right now?'. It asks for the single most coherent, logical sequence of states from beginning to end across all time.",
        "The Viterbi algorithm achieves this using dynamic programming. Instead of summing probabilities, it strictly maximizes them at every step and leaves 'breadcrumbs' (backpointers) to trace the winning path backwards.",
        "Use decoding (Viterbi) when the output must be a globally sensible sequence, such as turning audio into a sentence, or tagging words as Noun-Verb-Adjective.",
        "Viterbi only returns the absolute best path. It gives zero information about whether the second-best path was almost identical or completely different.",
        [
          "Finding the single fastest driving route from New York to LA. You don't care about the 'average' traffic time on any road; you just want to lock in the single ultimate winner."
        ],
        [
          `V_{t,k} = \\max_{x} \\left( P(y_t | k) \\cdot a_{x,k} \\cdot V_{t-1, x} \\right)`
        ]
      )
    ),
    hmmPage(
      "learning-hmm-parameters",
      "Learning HMM Parameters",
      "learning",
      "Use soft counts from forward-backward style inference to update transitions and emissions.",
      details(
        "Learning HMM parameters means inferring transition and emission probabilities from observed sequences when the true hidden states are unknown. The standard approach alternates inference and parameter re-estimation.",
        "The page shows one Baum-Welch style update using expected state occupancies and expected transition counts. That makes the EM logic concrete: infer hidden structure, then maximize expected complete-data likelihood.",
        "Use this page when you want to understand where HMM parameters come from rather than treating them as hand-tuned constants.",
        "Learning is sensitive to initialization and local optima. Different starting parameters can lead to different fitted models even on the same observations."
      )
    ),
    partitionPage(
      "partition-based-clustering",
      "Partition-Based Clustering",
      "general",
      "Alternate assignment and representative updates to divide points into groups.",
      details(
        "Partition-based clustering assumes the data can be summarized by a fixed number of representatives. Each iteration assigns points to representatives and then moves the representatives.",
        "The page lets you switch between centroid and medoid style updates, making the shared assignment-update structure explicit while keeping the mathematical difference visible.",
        "Use this page to understand the common skeleton behind K-means and K-medoids before specializing to each method.",
        "The main limitations are the need to choose k and the assumption that clusters are well captured by representative-based partitions."
      )
    ),
    partitionPage(
      "k-means",
      "K-Means",
      "kmeans",
      "Update centroids by averaging the points currently assigned to each cluster.",
      details(
        "K-means is the absolute classic in partition clustering. It seeks extremely compact groups by perpetually shifting the 'center of mass' of each cluster to the geometric center of its assigned points.",
        "The objective (inertia) is the sum of squared distances from points to their assigned centroids. Because both assignment and mean-updating strictly decrease or preserve this distance, it mathematically must converge.",
        "Use K-means everywhere as a baseline. It is unimaginably fast and scales beautifully to millions of points.",
        "It aggressively forces clusters into equivalent-sized spheres. If you feed it two intertwined spiral galaxies, or one massive sparse cluster next to a tiny dense one, it will slice them bizarrely in half.",
        [
          "Dropping 3 magnets onto a table of iron shavings. The shavings snap to the closest magnet, which pulls the magnet toward the center of its pile, repeating until nothing moves."
        ],
        [
          `\\mu_k = (1 / |C_k|) \\Sigma_{i \\in C_k} x_i`,
          `J = \\Sigma_k \\Sigma_{i \\in C_k} ||x_i - \\mu_k||^2`
        ]
      )
    ),
    partitionPage(
      "k-medoids",
      "K-Medoids",
      "kmedoids",
      "Restrict each representative to be an actual data point, improving robustness to outliers.",
      details(
        "K-medoids is the robust cousin of K-means. It shares the same assignment-update loop, but it restricts the 'centers' to be actual data points (medoids) rather than computed arithmetic means.",
        "Instead of minimizing squared Euclidean distance to a floating average, it minimizes the sum of absolute distances (or any robust metric) to the chosen medoid points.",
        "Use K-medoids when your dataset is riddled with extreme outliers that would yank a K-means centroid out into empty space, or when you are using a non-Euclidean distance metric.",
        "The cost is massive. Scanning every point in a cluster to find the one that minimizes total distance to all others is O(N^2) compared to K-means' O(N).",
        [
          "Choosing the capital of a state: K-means picks the exact geometric center (which might be in the middle of a desert), while K-medoids picks the most centrally located actual city."
        ],
        [
          `m_k = \\text{argmin}_{x \\in C_k} \\Sigma_{y \\in C_k} d(x, y)`,
          `J = \\Sigma_k \\Sigma_{i \\in C_k} d(x_i, m_k)`
        ]
      )
    ),
    hierarchyPage(
      "hierarchical-clustering",
      "Hierarchical Clustering",
      "general",
      "Build a nested cluster tree instead of committing immediately to one partition.",
      details(
        "Hierarchical clustering records how clusters form across multiple resolutions. Instead of selecting one k immediately, it builds a dendrogram that can be cut at different heights.",
        "The page supports both agglomerative and divisive views, along with linkage logic. That makes the hierarchy itself the object of study, not just the final partition.",
        "Use hierarchical clustering when nested structure matters or when you want to inspect clustering stability across scales.",
        "The hierarchy can be distorted by a poor linkage rule or noisy distances. Once a merge or split happens in a greedy hierarchy, it is typically not reconsidered."
      )
    ),
    hierarchyPage(
      "agglomerative-clustering",
      "Agglomerative Clustering",
      "agglomerative",
      "Start with singleton points and repeatedly merge the closest clusters.",
      details(
        "Agglomerative clustering is a bottom-up approach. It starts by assuming every single data point is its own microscopic cluster, then repeatedly glues the two closest clusters together until only one giant mass remains.",
        "The strategy for measuring distance between clusters (not just points) is called 'Linkage'. Single linkage measures the closest pair of points across clusters; Ward's minimizes the variance after merging.",
        "Use agglomerative clustering when you want a rich, nested hierarchy (a dendrogram) of your data, allowing you to choose the perfect number of clusters simply by drawing a horizontal cut line.",
        "It is greedy and irreversible. If two clusters are merged early on due to an anomaly, they can never be separated later, permanently distorting the tree.",
        [
          "Reverse-engineering a family tree. You start with individual siblings, merge them into parents, then grandparents, all the way up to a common ancestor."
        ],
        [
          `d_{min}(C_i, C_j) = \\min_{x \\in C_i, y \\in C_j} d(x, y)`
        ]
      )
    ),
    hierarchyPage(
      "divisive-clustering",
      "Divisive Clustering",
      "divisive",
      "Start with one cluster and repeatedly split the cluster with the largest internal spread.",
      details(
        "Divisive clustering is the top-down counterpart to agglomerative clustering. It begins with all points together and recursively splits clusters into smaller pieces.",
        "The page uses a simple two-way split heuristic so you can inspect the logic of choosing which cluster to cut and how the cut redistributes points.",
        "Use divisive clustering when you prefer reasoning from coarse structure to fine structure or when a top-down split heuristic matches the application.",
        "Divisive methods require a split strategy, and those heuristics can be less standardized than bottom-up linkage rules."
      )
    ),
    page({
      id: "distance-measures",
      title: "Distance Measures",
      category: "clustering",
      engine: "distance",
      subtitle: "Compare Euclidean, Manhattan, and cosine geometry on the same points.",
      summary:
        "Compare Euclidean, Manhattan, and cosine geometry on the same points.",
      detail: details(
        "Distance is not a neutral preprocessing detail; it defines what similarity means. Many algorithms change behavior dramatically when the metric changes.",
        "The page computes Euclidean, Manhattan, and cosine quantities from the same pair of vectors so you can inspect the arithmetic and the geometric interpretation together.",
        "Use this page before trusting nearest-neighbour or clustering outputs. Often the metric choice matters as much as the algorithm choice.",
        "A poor metric can destroy a good algorithm. If scaling, sparsity, or directional structure are ignored, closeness can become meaningless."
      ),
    }),
    page({
      id: "dbscan",
      title: "DBSCAN",
      category: "clustering",
      engine: "dbscan",
      subtitle: "Grow clusters from dense neighborhoods and leave sparse points as noise.",
      summary:
        "Grow clusters from dense neighborhoods and leave sparse points as noise.",
      detail: details(
        "DBSCAN Abandons geometric centers entirely in favor of density. It wanders the space, expanding clusters wherever enough points sit close together, and labels anything stranded in empty space as 'noise'.",
        "A point is a 'core' point if it has at least minPts neighbors within radius ε. Clusters are formed by connecting core points' neighborhoods via transitive reachability.",
        "Use DBSCAN when you know your data is riddled with chaotic outliers, or when cluster shapes look like contorted ribbons instead of spheres. It brilliantly discovers 'how many' clusters exist automatically.",
        "It hates varying density. If one cluster is a dense city center and another is a sparse suburb, no single 'radius' (ε) will work correctly for both.",
        [
          "A contagious disease spreading through a crowd. If people are standing close enough (density), they catch it and pass it on. Hermits standing far away remain uninfected (noise)."
        ],
        [
          `N_\\epsilon(p) = \\{q \\in D | d(p, q) \\leq \\epsilon\\}`,
          `p \\text{ is core if } |N_\\epsilon(p)| \\geq minPts`
        ]
      ),
    }),
    page({
      id: "spectral-clustering",
      title: "Spectral Clustering",
      category: "clustering",
      engine: "spectral",
      subtitle: "Turn a similarity graph into a Laplacian eigenproblem, then split by the Fiedler vector.",
      summary:
        "Turn a similarity graph into a Laplacian eigenproblem, then split by the Fiedler vector.",
      detail: details(
        "Spectral clustering realizes that grouping points by Euclidean distance often fails. Instead, it turns the data into a graph, cuts the weakest links, and clusters the points based on how the graph vibrates (its eigen-spectrum).",
        "It builds an adjacency matrix, computes the Graph Laplacian (L = D - W), and extracts the eigenvectors of the smallest eigenvalues. K-means is then run on these new eigenvector-coordinates.",
        "Use spectral clustering when your clusters are highly non-convex (like intertwined rings or spirals) and standard K-means completely fails.",
        "It is agonizingly slow for large datasets. Calculating the eigenvalues of an N×N matrix takes O(N³) time, making it utterly impractical beyond a few tens of thousands of points.",
        [
          "Trying to figure out which beads belong to which necklace in a tangled pile. Instead of looking at their coordinates, you shake the pile. Beads on the same string move together."
        ],
        [
          `L = D - W`,
          `L v = \\lambda D v`
        ]
      ),
    }),
    page({
      id: "geometric-transformations",
      title: "Geometric Transformations",
      category: "foundations",
      engine: "transform",
      subtitle: "Translate, scale, rotate and shear an image with one 3x3 matrix — and see why real code maps pixels backwards.",
      summary:
        "Translate, scale, rotate and shear an image with one 3x3 matrix — and see why real code maps pixels backwards.",
      detail: details(
        "Every affine transformation of an image — moving it, stretching it, spinning it, slanting it — is a single matrix multiplication applied to each pixel coordinate. Homogeneous coordinates are the trick that lets translation, which is an addition, be folded into the same matrix as rotation and scaling, which are multiplications.",
        "A point (x, y) is written as the column [x, y, 1]. The 3x3 matrix has a 2x2 linear block (rotation, scale, shear) in the top-left and a translation column on the right. Composing transformations is matrix multiplication, and undoing one is matrix inversion — which only exists when the determinant is non-zero.",
        "This is the foundation of image augmentation in computer vision pipelines, of camera calibration, and of every graphics transform stack. Understanding forward versus inverse mapping explains a whole class of rendering artefacts.",
        "Forward mapping — pushing each source pixel to its destination — leaves holes when the image is enlarged or rotated, because rounded destinations collide and skip. Production code always maps inversely: iterate over destination pixels and pull from the source. A determinant of zero collapses the image to a line and cannot be inverted at all.",
        [
          "Think of the matrix as instructions to a photocopier: rotate the page, stretch it, slide it. Forward mapping is spraying ink from the original onto a blank sheet and finding gaps. Inverse mapping is asking each blank spot 'which part of the original am I?' — every spot gets an answer."
        ],
        [
          `\\begin{bmatrix} x' \\\\ y' \\\\ 1 \\end{bmatrix} = \\begin{bmatrix} a & c & t_x \\\\ b & d & t_y \\\\ 0 & 0 & 1 \\end{bmatrix}\\begin{bmatrix} x \\\\ y \\\\ 1 \\end{bmatrix}`,
          `R(\\theta) = \\begin{bmatrix} \\cos\\theta & -\\sin\\theta \\\\ \\sin\\theta & \\cos\\theta \\end{bmatrix}`,
          `\\det M = ad - bc \\neq 0 \\iff M^{-1} \\text{ exists}`
        ]
      ),
    }),
    page({
      id: "activation-functions",
      title: "Activation Functions",
      category: "neural",
      engine: "activation-guide",
      layout: "guide",
      guide: {
        kicker: "Guide · Neural network basics, 1 of 3",
        heading: "Activation functions, from one neuron to deep networks",
        meta: ["9 chapters", "≈ 25 min read", "7 editable widgets", "Needs: nothing but a slope"],
      },
      subtitle: "Without a non-linearity, a deep network collapses into a single linear layer. Compare the choices and their gradients.",
      summary:
        "Without a non-linearity, a deep network collapses into a single linear layer. Compare the choices and their gradients.",
      detail: details(
        "Stacking linear layers achieves nothing: a matrix times a matrix is just another matrix. The activation function is the only reason depth buys you anything. Its shape decides what the network can represent, and its derivative decides whether gradients survive the trip back through many layers.",
        "Sigmoid squashes to (0, 1) with derivative s(1 - s), which peaks at just 0.25 — multiply a handful of those together during backpropagation and the gradient vanishes. Tanh is zero-centred with a peak derivative of 1, which helps. ReLU is max(0, z) with a derivative that is exactly 1 for positive inputs, so gradients pass through undiminished, at the cost of dead units for negative inputs.",
        "ReLU and its variants are the default for hidden layers in deep networks. Sigmoid survives at the output layer for binary probabilities, and softmax for multi-class. Tanh still appears inside recurrent architectures.",
        "Saturation is the enemy. Sigmoid and tanh both flatten at their extremes, and a flat function has a near-zero derivative — the layer stops learning. ReLU avoids this on the positive side but can die permanently if a unit's weights push every input negative; leaky ReLU exists to patch exactly that.",
        [
          "A neuron's activation is a dimmer switch, not a light switch. Sigmoid is a very gentle dimmer that barely responds near either end. ReLU is a switch that is off below zero and perfectly proportional above it."
        ],
        [
          `\\sigma(z) = \\frac{1}{1 + e^{-z}}, \\quad \\sigma'(z) = \\sigma(z)(1 - \\sigma(z))`,
          `\\tanh(z) = \\frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}, \\quad \\tanh'(z) = 1 - \\tanh^2(z)`,
          `\\mathrm{ReLU}(z) = \\max(0, z), \\quad \\mathrm{ReLU}'(z) = \\mathbb{1}[z > 0]`
        ]
      ),
    }),
    page({
      id: "gradient-descent",
      title: "Gradient Descent",
      category: "neural",
      engine: "gradient-descent-guide",
      layout: "guide",
      guide: {
        kicker: "Guide · Neural network basics, 2 of 3",
        heading: "Gradient descent: how networks learn",
        meta: ["9 chapters", "≈ 25 min read", "6 editable widgets", "Plain, momentum, Adam and mini-batch"],
      },
      subtitle: "Follow the slope downhill. Watch the learning rate decide between crawling, converging, and diverging.",
      summary:
        "Follow the slope downhill. Watch the learning rate decide between crawling, converging, and diverging.",
      detail: details(
        "Gradient descent is the entire optimisation story of deep learning in one line: compute which direction increases the loss, then step the other way. Everything else — momentum, Adam, schedules — is a refinement of how big that step should be and how much of the previous step to remember.",
        "The gradient points in the direction of steepest ascent, so the update subtracts it. The learning rate scales the step. Momentum accumulates an exponentially weighted average of past gradients, which damps oscillation across a narrow valley and accelerates progress along its floor.",
        "Every parameter in every neural network is trained this way. The same loop fits logistic regression, SVMs with hinge loss, and matrix factorisations.",
        "The learning rate is the single most consequential hyperparameter. Too small and training crawls; too large and each step overshoots the minimum, the loss climbs, and the run diverges to infinity. Non-convex surfaces add local minima, saddle points, and plateaus where the gradient is near zero in every direction.",
        [
          "Walking downhill in thick fog. You can only feel the slope under your feet, so you take a step in the steepest downhill direction and repeat. Big steps get you down fast but may launch you up the opposite slope; tiny steps are safe but you may never arrive."
        ],
        [
          `\\theta_{t+1} = \\theta_t - \\eta \\nabla_\\theta L(\\theta_t)`,
          `v_{t+1} = \\beta v_t + \\nabla_\\theta L(\\theta_t), \\quad \\theta_{t+1} = \\theta_t - \\eta v_{t+1}`
        ]
      ),
    }),
    page({
      id: "backpropagation",
      title: "Backpropagation",
      category: "neural",
      engine: "backprop-guide",
      layout: "guide",
      guide: {
        kicker: "Guide · Neural network basics, 3 of 3",
        heading: "Backpropagation, one number at a time",
        meta: ["10 chapters", "≈ 30 min read", "Every number of a training step", "Full network playground"],
      },
      subtitle: "The chain rule run backwards through a network — every gradient computed from your own inputs, weights and targets.",
      summary:
        "The chain rule run backwards through a network — every gradient computed from your own inputs, weights and targets.",
      detail: details(
        "Backpropagation answers one question: if I nudge this weight, how much does the loss change? Computing that naively for every weight would mean a separate forward pass each time. Backpropagation gets all of them in a single backward sweep by reusing shared sub-expressions — it is dynamic programming applied to the chain rule.",
        "The forward pass stores each layer's pre-activation z and activation a. The backward pass computes an error signal delta at the output, then repeatedly pulls it back through the transpose of the weight matrix and multiplies by the local activation derivative. The gradient for any weight is then simply the delta at its destination times the activation at its source.",
        "This is how every deep network is trained. Autograd frameworks automate it, but the vanishing-gradient behaviour you can watch on this page is exactly why architectures use ReLU, residual connections and normalisation layers.",
        "Because each layer multiplies the error signal by an activation derivative, deep networks compound those factors. With sigmoid activations, whose derivative never exceeds 0.25, the gradient shrinks by at least a factor of four per layer — the early layers barely learn. Step through this page and compare the gradient magnitudes in the last layer against the first.",
        [
          "A relay team running backwards. The anchor runner knows exactly how far off the finish line was; each earlier runner is told how much of that error was their share, scaled by how strongly they influenced the next runner."
        ],
        [
          `\\delta^{(L)} = (\\mathbf{a}^{(L)} - \\mathbf{y}) \\odot \\sigma'(\\mathbf{z}^{(L)})`,
          `\\delta^{(l)} = \\left( (W^{(l+1)})^{\\top} \\delta^{(l+1)} \\right) \\odot \\sigma'(\\mathbf{z}^{(l)})`,
          `\\frac{\\partial L}{\\partial W^{(l)}_{jk}} = \\delta^{(l)}_j \\, a^{(l-1)}_k`
        ]
      ),
    }),
    page({
      id: "sequence-primer",
      title: "Primer: Before Attention",
      category: "neural",
      engine: "sequence-primer",
      layout: "guide",
      guide: {
        kicker: "Guide · Part 1 of 5",
        heading: "Before attention: the building blocks",
        meta: ["9 chapters", "≈ 20 min read", "Live tokenizer from GPT-2 and GPT-4", "5 editable widgets"],
      },
      subtitle: "Tokens, embeddings, dot products, softmax and the RNNs that came before: everything the Attention guide assumes, with widgets you can edit.",
      summary:
        "Tokens, embeddings, dot products, softmax and the RNNs that came before: everything the Attention guide assumes, with widgets you can edit.",
      detail: details(
        "Language models read token IDs, turn them into learned vectors, compare vectors with dot products and turn scores into probabilities with softmax. Before Transformers, recurrent networks read text one token at a time, which made them forget over long distances and impossible to parallelise; attention was first added to them to fix a translation bottleneck.",
        "Byte-pair encoding builds a vocabulary by repeatedly merging the most frequent adjacent pair of symbols. An embedding is a one-hot vector times a matrix. The dot product a·b = |a||b|cos θ measures alignment; softmax(z/T) turns scores into probabilities with temperature T. An RNN computes h_t = tanh(W_hh h_{t−1} + W_xh x_t + b), and its gradients are products of many Jacobians, which vanish or explode.",
        "These ideas are prerequisites for understanding attention, Transformers and large language models.",
        "Tokenization quirks (leading spaces, split numbers, costly non-English text) affect model behaviour and cost. Softmax is shift-invariant: only differences between scores matter.",
        ["Reading with a sticky note: an RNN may keep only one small note of everything read so far, which is why long-range details get lost."],
        [`p_j = \frac{e^{z_j/T}}{\sum_k e^{z_k/T}}`, `h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t + b)`]
      ),
    }),
    page({
      id: "attention",
      title: "Attention",
      category: "neural",
      engine: "attention",
      subtitle: "How does the word “it” work out that it means “the cat”? A step-by-step guide that starts from plain words and builds up to the exact formula in the paper.",
      summary:
        "How “it” learns that it means “the cat”: queries, keys and values, built up step by step from the dot product you already know.",
      layout: "guide",
      guide: {
        kicker: "Guide · Part 2 of 5",
        heading: "Attention, from the ground up",
        meta: ["12 short chapters", "≈ 25 min read", "Needs: what a neuron is", "Follows “Attention Is All You Need”"],
      },
      detail: details(
        "A word's meaning depends on the words around it, but an embedding gives each word a single fixed vector. Attention fixes that by rebuilding every word's vector as a weighted average of all the words in the sentence. The weights are not fixed parameters. They are computed fresh for each sentence, from the words themselves: each word asks a question (its query), every word advertises an answer (its key), and the better the match, the more of that word's content (its value) gets mixed in.",
        "Three learned matrices project each word vector x into a query q = xW^Q, a key k = xW^K and a value v = xW^V. Stacking the sentence into a matrix X turns the whole computation into Equation 1 of the paper: the score matrix QKᵀ compares every query with every key at once, dividing by √dₖ keeps those scores at unit variance, a row-wise softmax turns them into weights that sum to 1, and multiplying by V mixes the values. Multi-head attention runs h of these with smaller dₖ = d_model/h, concatenates the results and projects them with W^O.",
        "Attention is the core operation of every transformer: machine translation (the paper's task), language models like GPT and BERT, vision transformers that treat image patches as words, and speech and protein models. The same query–key–value pattern also appears in retrieval systems and memory-augmented networks.",
        "Attention has no idea of word order. Shuffle the input and the outputs are only shuffled, which is why transformers add positional encodings. Its cost grows with the square of the sequence length, since every word scores every other word. And attention heatmaps are tempting to read as explanations, but they show where information flowed in one layer, not why the model made its final decision.",
        [
          "Searching a library. The query is what you type into the search box, keys are the labels on the spines, values are the books' contents. Unlike a real library, you don't take one book home. You take a little of every book, in proportion to how well its label matches your search."
        ],
        [
          `\\operatorname{Attention}(Q,K,V) = \\operatorname{softmax}\\!\\Big(\\frac{QK^{\\top}}{\\sqrt{d_k}}\\Big)V`,
          `Q = XW^Q,\\quad K = XW^K,\\quad V = XW^V`,
          `\\operatorname{MultiHead}(X) = \\operatorname{Concat}(\\text{head}_1, \\ldots, \\text{head}_h)\\,W^O`
        ]
      ),
    }),
    page({
      id: "transformer",
      title: "Transformer",
      category: "neural",
      engine: "transformer",
      subtitle: "The complete model from “Attention Is All You Need”, explained one box of its famous diagram at a time: from words in to a translation out.",
      summary:
        "The full architecture from “Attention Is All You Need”, one box at a time: word order, the encoder block, the masked decoder, and why it replaced RNNs.",
      layout: "guide",
      guide: {
        kicker: "Guide · Part 3 of 5",
        heading: "The Transformer, box by box",
        meta: ["11 chapters", "≈ 30 min read", "Needs: the Attention guide", "Every box of the paper's Figure 1"],
      },
      detail: details(
        "A transformer is attention plus the four things attention can't do by itself. Positional encodings give it a sense of word order. A small feed-forward network processes each word after it has gathered context. Residual connections and layer normalisation let dozens of layers train. And a decoder with a causal mask writes the output one word at a time while reading the input through cross-attention. There is no recurrence: every word is processed in parallel.",
        "The encoder is N identical layers, each computing h = LN(x + MultiHead(x)) and then LN(h + FFN(h)), where FFN(h) = max(0, hW₁ + b₁)W₂ + b₂ runs on each position separately. The decoder adds a masked self-attention (scores above the diagonal set to −∞) and a cross-attention whose queries come from the decoder and whose keys and values come from the encoder output. A final linear layer and softmax give next-token probabilities. Sinusoidal positional encodings are added to the input embeddings, and training uses Adam with a warm-up learning-rate schedule, dropout and label smoothing.",
        "This architecture underlies essentially all modern large language models (decoder-only, like GPT), text encoders (encoder-only, like BERT), sequence-to-sequence systems for translation and summarisation (the original encoder–decoder form), and vision, audio and multimodal models. Understanding the original paper's version makes every later variant a small change to one of its boxes.",
        "Self-attention's cost grows as n² with sequence length, so very long inputs are expensive. Without positional information the model is blind to order. Generation is still sequential, one token per step, even though training is parallel. The causal mask must be applied correctly, or the decoder learns to copy the answer during training and fails at inference. Transformers are also data-hungry: their weak built-in assumptions are a strength at scale and a weakness on small datasets.",
        [
          "A meeting instead of a game of telephone. An RNN passes a message along a line, one person at a time, and details fade with every hop. A transformer puts everyone in one room: each person can ask anyone else directly (attention), then thinks it over alone (feed-forward), and the meeting repeats a few rounds (layers)."
        ],
        [
          `h = \\operatorname{LayerNorm}\\big(x + \\operatorname{MultiHead}(x)\\big),\\quad y = \\operatorname{LayerNorm}\\big(h + \\operatorname{FFN}(h)\\big)`,
          `PE_{(pos,2i)} = \\sin\\!\\big(pos/10000^{2i/d_{\\text{model}}}\\big),\\quad PE_{(pos,2i+1)} = \\cos\\!\\big(pos/10000^{2i/d_{\\text{model}}}\\big)`,
          `\\text{lrate} = d_{\\text{model}}^{-0.5}\\cdot\\min\\big(\\text{step}^{-0.5},\\ \\text{step}\\cdot\\text{warmup}^{-1.5}\\big)`
        ]
      ),
    }),
    page({
      id: "llm",
      title: "From Transformer to LLM",
      category: "neural",
      engine: "llm",
      layout: "guide",
      guide: {
        kicker: "Guide · Part 4 of 5",
        heading: "From the Transformer to today's LLMs",
        meta: ["11 chapters", "≈ 30 min read", "Real GPT-2 tokenizer", "Published model sizes"],
      },
      subtitle: "How GPT-style models use the Transformer: decoder-only stacks, next-token prediction, sampling, RoPE, the KV cache, mixture of experts, scale and instruction tuning.",
      summary:
        "How GPT-style models use the Transformer: decoder-only stacks, next-token prediction, sampling, RoPE, the KV cache, mixture of experts, scale and instruction tuning.",
      detail: details(
        "Large language models are decoder-only Transformers trained to predict the next token on enormous amounts of text, then fine-tuned to follow instructions and match human preferences.",
        "Pretraining minimises next-token cross-entropy. Generation samples from softmax(z/T) after top-k or top-p filtering. Modern blocks use pre-norm RMSNorm, SwiGLU feed-forward layers, rotary position embeddings and grouped-query attention; parameters are roughly 12·L·d² + V·d.",
        "Chat assistants, code completion, summarisation, translation and most modern NLP systems.",
        "Generation is sequential and memory-bound: the KV cache grows with context length. Pretrained models predict text rather than follow instructions until they are fine-tuned.",
        ["Autocomplete, scaled up: an LLM is a very good next-word predictor that has been trained further to act as a helpful assistant."],
        [`\mathcal{L} = -\frac{1}{n}\sum_t \log p_\theta(x_t \mid x_{<t})`, `\text{params} \approx 12\,L\,d^2 + V d`]
      ),
    }),
    page({
      id: "small-llm",
      title: "Small LLMs & Distillation",
      category: "neural",
      engine: "small-llm",
      layout: "guide",
      guide: {
        kicker: "Guide · Part 5 of 5",
        heading: "Small LLMs: distillation, pruning and quantization",
        meta: ["11 chapters", "≈ 35 min read", "Two networks trained live in your browser", "9 editable widgets"],
      },
      subtitle: "How small language models get good: training past Chinchilla, spending parameters wisely, knowledge distillation from a teacher, pruning and quantization, with live widgets for each.",
      summary:
        "How small language models get good: scaling laws, knowledge distillation with soft targets, forward vs reverse KL, pruning and quantization, each with a live widget.",
      detail: details(
        "A small language model fits on a laptop or phone, answers faster and costs less per token. It gets good by being trained far longer than compute-optimal, by spending its parameters carefully (tied embeddings, grouped-query attention, deep-and-thin shapes), by learning from a large teacher model (knowledge distillation), by being cut down from a larger model (pruning) and by storing each weight in fewer bits (quantization).",
        "Distillation trains the student on the teacher's temperature-softened probabilities: L = α·CE(y, p_s) + (1−α)·T²·KL(p_t^(T) ‖ p_s^(T)). The Chinchilla law L(N, D) = E + A/N^α + B/D^β says how loss trades off model size and data. Pruning keeps large-magnitude weights or whole structures and re-fits the rest; quantization stores q = round(w/s) with a per-group scale s = max|w|/(2^(b−1)−1).",
        "On-device assistants, low-latency and low-cost serving, private on-premise deployment, draft models for speculative decoding, and cheap task-specific fine-tuning.",
        "A student cannot copy more than it has capacity for. Logit distillation needs the teacher's raw outputs and a shared tokenizer. Forward KL makes a student spread over everything the teacher might say; reverse KL makes it commit to one answer. Pruning and aggressive quantization lose accuracy unless the model is healed by further training.",
        ["An apprentice and a master: instead of only hearing whether each answer was right or wrong, the apprentice watches how confident the master is about every option, and learns much faster from that."],
        [
          `\\mathcal{L} = \\alpha\\,\\mathrm{CE}(y, p^s) + (1-\\alpha)\\,T^2\\,\\mathrm{KL}\\big(p^{t,(T)} \\,\\|\\, p^{s,(T)}\\big)`,
          `L(N, D) = E + \\frac{A}{N^{\\alpha}} + \\frac{B}{D^{\\beta}}`,
          `q = \\operatorname{round}(w/s),\\quad s = \\frac{\\max|w|}{2^{b-1}-1}`
        ]
      ),
    }),
  ];

  const algorithms = {};
  allPages.forEach((definition, index) => {
    algorithms[definition.id] = Object.assign({ order: index }, definition);
  });

  function get(id) {
    return algorithms[id] || null;
  }

  function byCategory(categoryId) {
    return Object.values(algorithms)
      .filter((entry) => entry.category === categoryId)
      .sort((a, b) => a.order - b.order);
  }

  window.MLAlgorithms = {
    categories,
    algorithms,
    get,
    byCategory,
  };
})();
