# Manifold Classification Capacity — study notes

*Cohen, Chung, Lee & Sompolinsky, "Separability and geometry of object manifolds in deep neural networks," Nat. Commun. 11:746 (2020), building on Chung, Lee & Sompolinsky (CLS), Phys. Rev. X 8, 031003 (2018).*

---

## 1. Setup

An object class presented under varying conditions gives rise to a collection of neural population responses of $N$ neurons, called a **neural manifold**, that lives in $N$-neuron activation space. So we have an activation space with many manifolds (one per class) and we want them well separated — strictly, **linearly separable**.

A brute-force way to gauge how separable the manifolds are:

- Give a random binary label ($\pm1$) to each manifold. One such assignment = **one random draw**.
- For each draw, ask whether a single hyperplane puts *all points* of every $+1$ manifold on one side and all of every $-1$ manifold on the other. ==The vector normal to that hyperplane is the **weight vector**==, built as a linear combination of the **anchor points** of the manifolds, which:
	- are a point within each manifold or its convex hull, and depend not only on the manifolds' shape but also on their location/orientation in state space and on the particular random labeling (for this draw you get some anchor points and thus some hyperplane);
	- depend on the *other* manifolds in the ensemble (this circularity yields a self-consistency equation).
- **Classification capacity.** Load $\alpha = P_{\text{manifolds}}/N_{\text{neurons}}$. The load where the separable fraction drops $1 \to 0$ is $\alpha_c$.
- Sweep $\alpha$ by changing how many manifolds you ask the system to separate while keeping $N$ fixed, and find where separability collapses. For a fixed manifold, as the location and labeling of the *other* manifolds vary, its anchor point changes — tracing out a **statistical distribution of anchor points** for a manifold embedded in an ensemble of others. The effective radius $R_M$ is the total variance of those anchor points (normalized by the average inter-center distance); the effective dimension $D_M$ is their spread along the manifold's axes.

**Brute-force draw, summarized:**
- A **draw** = one random $\pm1$ labeling of the $P$ manifolds.
- Outcome = separable or not.
- Sweep $\alpha = P/N$; collapse point = $\alpha_c$.
- "How many draws" = a Monte-Carlo sample size; not fundamental.


---

So now let's go to the OG approach to the linear separability to get a better intuition of the above part of taking the load to 0 adding the entropy.

## 2. Gardner's original picture (why "load → 0" + entropy)

Gardner's perceptron capacity uses $\alpha = \text{Points}/\text{Neurons}$; here we use $P_{\text{manifolds}}/N_{\text{neurons}}$. Define $V$ = the **volume of weight vectors $\vec{w}$** (the *version space*) — i.e. all separating hyperplanes — that correctly classify all $P$ manifolds with margin $\kappa$. **[CORRECTED: $V$ is the set/volume of weight vectors, not of anchor points. Anchor points pin down one solution; the version space is the whole solution set of valid $\vec w$.]** Each manifold imposes inequality constraints (all its points on the correct side), so $V$ is the intersection of those constraints — a convex region on the sphere $\|\vec{w}\|^2 = N$. It is larger when there are fewer manifolds (smaller $\alpha$), because many hyperplanes work.

At $\alpha_c$ the constraints have eaten this region down to a point → $V \to 0$. Beyond $\alpha_c$ there are more (effective) constraints than degrees of freedom; the system is over-determined and no solution survives.

So Gardner's approach: compute $V$, **average $\log V$ over the disorder (random labels + manifold placements), and find the $\alpha_c$ at which $V \to 0$.**

### 1 The "whys"

- **Why log?** The version-space volume scales exponentially in $N$: $V \sim e^{N f(\alpha)}$. We care about the *rate* $f(\alpha)$ (an entropy per dimension), not the astronomically large/small $V$. The capacity is where $f(\alpha) \to -\infty$ (i.e. $V \to 0$).
	- **Why "entropy per dimension"?** Both $V$ and $\log V$ grow with $N$ ($V$ exponentially, $\log V$ linearly), so divide by $N$:
	$$S(\alpha) = \frac{\log V}{N} \to f(\alpha) = \text{entropy density (per degree of freedom)}.$$
	$\alpha_c$ is where $f(\alpha)\to-\infty$; $S(\alpha)$ is a smooth order parameter whose collapse marks the **separability (capacity) phase transition**. *[CORRECTED: "phase transition," not "saddle-node transition" — the saddle point belongs to the replica saddle-point calculation; the transition itself is the solvable→unsolvable one.]*
	- **Why avg-of-log, not log-of-avg?** For an exponentially-fluctuating quantity, $\langle V\rangle$ is dominated by rare, anomalously huge draws; the *typical* value is $e^{\langle \log V\rangle}$. Hence the **replica** method: $\langle \log V\rangle = \lim_{n\to0}\frac{\langle V^n\rangle - 1}{n}$, computed with $n$ replicas sharing the same disorder.
	- **Entropy intuition:** entropy = log of the number of available configurations. Here the configurations are valid weight vectors; their volume collapses at $\alpha_c$.

---

## 3. Why mean-field theory (MFT)

Brute force is hopeless with many manifolds: it requires accounting for all $P$ manifolds simultaneously, because they interact (they jointly determine the shared hyperplane). To get **the** maximal-capacity hyperplane, approximate the field that **a typical manifold** feels. Two notes on "a typical manifold": all manifolds are statistically identical, and the calculation effectively sweeps over all of them; and a given manifold generates the very field it interacts with (the circularity).

So MFT is built on a **self-consistent equation involving a single manifold embedded in an ensemble of many others, the ensemble approximated by the field it generates, called $\vec{T}$.** The intractable "can one hyperplane separate $P$ manifolds under any random labeling?" reduces to "how does **one** manifold respond to a single abstract Gaussian field $\vec{T}$ that stands in for all the others?" — and the answer is read off from the statistics of the manifold's **anchor point** as $\vec{T}$ sweeps its Gaussian.

---

So before getting too deep into MFT let's go to the gaussian distribution, because it is used in the MFT when we sweep over all possible hyperplanes, by assigning each one a probability of it being the one that separates the manifolds (i think).

## 4. The Gaussian field $\vec{T}$ as an object

A Gaussian distribution is a **density** — assigns each $\vec{T}$ a non-negative $p(\vec{T})$, normalized to integrate to 1. Standard (mean-zero, identity-covariance) form in $d$ dimensions:

$$p(\vec{T}) = \frac{1}{(2\pi)^{d/2}} \exp\!\left(-\tfrac{1}{2}\|\vec{T}\|^2\right), \qquad \vec{T}\in\mathbb{R}^d.$$

A smooth hill centered at the origin, falling off radially. It is a *weighting* over every possible value of the field. Points further than 0 are exponentially rare.

$\vec{T}$ is the abstract field — the stand-in for the collective pressure of all the other manifolds on the shared hyperplane. It lives in the single manifold's local coordinate system: dimension $d = D+1$, where $D$ is the manifold's (effective) affine dimension and the $+1$ is the center/bias direction.

**Why Gaussian, why standard.** Gaussian: CLT on the many weak, random ($\pm1$) interactions from the other manifolds as $N\to\infty$. Standard: the calculation is in the manifold's own normalized coordinates — the field is the neutral, isotropic probe, and all structure comes from the manifold's geometry, not from a lopsided field.

**Why affine**: "Affine" = the manifold's linear span of variation directions *around its off-origin center* (hence the separate $+1$ for the center). "Effective" = a magnitude-weighted count of how many directions actually matter, and here the weighting is done by the anchor-point distribution under the Gaussian average:

$$D_M = \Big\langle \big(\vec{T}\cdot \hat{s}(\vec{T})/\|\hat{s}(\vec{T})\|\big)^2 \Big\rangle_{\vec{T}} .$$

*[Standardized to one convention throughout: $\hat{s}(\vec{T})$ is the projected anchor point (with magnitude — the paper's symbol), and the field is dotted with its **direction** $\hat{s}/\|\hat{s}\|$. So $D_M$ measures field–anchor **alignment**, while $R_M$ uses the anchor's **magnitude**. Same form used in §5 and §6.]*

A manifold axis contributes to $D_M$ only to the extent the anchor point lands along it across field draws; axes the manifold technically spans but that never host the support point contribute negligibly, so the integral over $\vec{T}$ discounts them automatically. This is stronger than the **participation ratio** (which weights axes by raw variance $\lambda_i$): $D_M$ weights them by variance *as it bears on the separating hyperplane*. A high-variance direction that never affects where the hyperplane must go inflates the participation ratio but not $D_M$.

So in $d = D+1$, the $D$ is this *effective* dimension $D_M$ — not the raw rank of the cloud — and the $+1$ is the center/offset (the affine translation). The "effective" is baked into the geometry the self-consistent equations are written in, via the Gaussian average over anchor points.

---

## 5. The expectation over $\vec{T}$

The whole point of the MFT is to say thanks to TCL that a typical manifold interacts weakly with all of the others, whether they're -1 or 1, and weak random (bc it's randomly -1 or +1) independent interactions tend to a gaussian field when N --> inf.
So the randomness over labels has been *absorbed into* the Gaussian field — that's the whole point of the reduction — so you no longer draw labels at all at this level. You draw fields. And how many? Conceptually: **infinitely many — because it's an integral, not a finite sample.** ==The mean-field "average over draws of T" is an expectation== over the full Gaussian distribution. In general, we know that ==an expectation is the integral of a function of the random variable, weighted by some distribution.==
The randomness over labels is **absorbed analytically** into the Gaussian field (CLT), so at the MFT level you no longer draw labels — you "draw fields," and "how many" is **all of them**: it's a continuous expectation (an integral), not a finite tally.

$$\langle \,\cdot\, \rangle_{\vec{T}} = \int (\,\cdot\,)\; \frac{e^{-\|\vec{T}\|^2/2}}{(2\pi)^{d/2}}\, d\vec{T}.$$

**One field, many values.** There is a single random vector $\vec{T}$; a "draw" is one *value* it takes, not a different field. (One die, many rolls; $\mathbb{R}^d$ is its range.) So there are no "other fields" — only other possible *values* of the one field, and the integral handles all of them at once. *[CORRECTED: removed the "(D_m or R_m)" aside — the field's values are vectors $\vec T$; $D_M, R_M$ are outputs of the expectation, not values of the field.]*

There's no finite count. You're integrating the anchor-point computation against the Gaussian measure over all possible field directions, weighted by how probable each is.

**There is one field; draws are samples of it**: The Gaussian field is a single random object — one random vector $\vec{T}$ with one fixed distribution $p(\vec{T}) = (2\pi)^{-d/2}e^{-\|\vec{T}\|^2/2}$. "Drawing" means sampling a value from that one distribution. Different draws are different *values* of the same random variable, not different fields.

An analogy: if I roll one six-sided die repeatedly, I don't have "six different dice." I have *one* die, and each roll is one outcome of it. The set $\{1,2,3,4,5,6\}$ is the range of possible outcomes; each roll lands on one of them. Likewise, $\vec{T}$ is one Gaussian random vector; $\mathbb{R}^d$ is its range of possible values; each draw lands on one $\vec{T}$ in that range. So when I say "for each field draw $\vec{T}$," I mean "for each possible value the field can take" — and the expectation sweeps over *all* those possible values.

So there's no "what about the other fields" — there's only "what about the other possible *values* (D_m or R_m) of the one field," and the answer is: the integral handles all of them at once. That's exactly what the integral *is*.

**The integral is "over all draws," done all at once**: You said you thought the integral was over field draws — and it is. The expectation

$$\langle g(\vec{T})\rangle = \int_{\mathbb{R}^d} g(\vec{T})\, p(\vec{T})\, d\vec{T}$$

*is* the operation "consider every possible draw $\vec{T}$, and combine them." The integral sign $\int \cdots d\vec{T}$ is precisely the instruction "sweep over all values of $\vec{T}$." So:

- "For each field draw $\vec{T}$, compute $g(\vec{T})$" describes the *integrand* — what you do at one value.
- "$\int \cdots p(\vec{T})\,d\vec{T}$" describes the *aggregation* — summing those over all draws, each weighted by its probability.


### 1 Three roles (the piece most easily muddled)

| Role | Object | Here |
|---|---|---|
| random variable | $\vec{T}$ | the Gaussian field draw |
| distribution (weighting) | $p(\vec{T})$ | the standard Gaussian density — **NOT** the thing averaged |
| function of the r.v. | $g(\vec{T})$ | a quantity built from the anchor point $\hat{s}(\vec{T})$ |

$$\big\langle g(\vec{T})\big\rangle \;=\; \int_{\mathbb{R}^{D+1}} g(\vec{T})\,p(\vec{T})\,d\vec{T}
\;\equiv\; \int D\vec{T}\; g(\vec{T}),
\qquad D\vec{T}\equiv\prod_{i=1}^{D+1}\frac{dT_i\,e^{-T_i^2/2}}{\sqrt{2\pi}}.$$

- $g(\vec{T})$ = what you compute at **one** draw (the integrand): the radius or effective dimensionality.
- $\int\!\cdots p(\vec{T})\,d\vec{T}$ = aggregation over **all** draws, each weighted by its probability.
- Result = a single number: the average over a typical manifold's experience of the field.

**$g$ is not fixed — it's whichever quantity you want the average of:**
- $g = \|\hat{s}(\vec{T})\|^2 \;\Rightarrow\;$ feeds $R_M$  *[CORRECTED LaTeX]*
- $g = \big(\vec{T}\cdot \hat{s}(\vec{T})/\|\hat{s}(\vec{T})\|\big)^2 \;\Rightarrow\;$ feeds $D_M$

> **Parallel to the VPL / Fisher-information case.** There: random variable $\mathbf{r}$ (response), distribution $p(\mathbf{r}\mid\theta)$ (Gaussian), function $=(\partial_\theta \log p)^2$ (squared *score*, not the log-likelihood). Same machinery: distribution = weighting; function = the thing whose average you want.

### 2 The anchor point

For each field draw $\vec{T}$, the anchor point is the support point the field picks out — schematically the (margin-constrained) projection of $\vec{T}$ onto the manifold's convex hull:

$$\hat{s}(\vec{T}) \;=\; \arg\min_{\vec{s}\,\in\, S}\;\tfrac{1}{2}\|\vec{T}-\vec{s}\|^2 \quad(\text{+ margin constraints}).$$

The Gaussian statistics of $\vec{T}$ push forward through this map into a **statistical measure on anchor points** $\hat{s}(\vec{T})$. The effective geometry is the **moments** of that measure (a moment = expected value of a power of the variable: mean = 1st, variance = 2nd central):

$$R_M^2 \;=\; \big\langle \|\hat{s}(\vec{T})\|^2\big\rangle_{\vec{T}},
\qquad
D_M \;=\; \Big\langle \big(\vec{T}\cdot \hat{s}(\vec{T})/\|\hat{s}(\vec{T})\|\big)^2\Big\rangle_{\vec{T}}.$$

**Weight-vector decomposition (paper's phrasing).** ==The weight vector== projected onto a manifold's subspace = (contribution from the manifold's **own** anchor point, scaled by its support coefficient $\lambda_\mu$) + (contribution from the field $\vec{T}$, i.e. everybody else). This split is why one manifold's anchor depends on the rest of the ensemble — hence the anchor is a distribution, not a fixed point.

---

## 6. Formulas from the paper (Methods), with the paper's nomenclature

*The blocks below map directly onto two Methods subsections:*
- *"Summary of manifold classification capacity and anchor points"* → **Anchor points and the separating hyperplane** + **Capacity (mean-field, CLS form)**, below.
- *"Geometric properties of manifolds"* → **Geometric properties (Eqs. 3–4)** + **Ball equivalence**, below.

### 1 Nomenclature

| Symbol | Meaning |
|---|---|
| $N$ | number of neurons (units) in the layer |
| $P$ | number of object manifolds |
| $\alpha = P/N$ | load |
| $\alpha_c$ | classification capacity (critical load) |
| $\kappa$ | classification margin |
| $\vec{x}^\mu$ | center (centroid) of manifold $\mu$ |
| $\tilde{x}^\mu$ | **anchor point of manifold $\mu$ in the full $N$-dim space** |
| $\hat{s}(\vec{T})$ | anchor point projected onto the manifold's own $(D{+}1)$-dim subspace (function of the field draw) |
| $\lambda_\mu$ | support coefficient of manifold $\mu$ (dual variable; weight of its anchor in $\vec w$) |
| $\vec{T}$ | Gaussian field, $\vec T \in \mathbb{R}^{D+1}$ |
| $R_M$ | manifold effective radius |
| $D_M$ | manifold effective dimension |
| $\rho_{CC}$ | inter-manifold center correlation |

> $\tilde{x}^\mu$ and $\hat{s}(\vec{T})$ are the **same anchor point in two coordinate systems**: $\tilde{x}^\mu$ ambient ($\mathbb{R}^N$), $\hat{s}$ projected into the manifold's $(D{+}1)$-dim subspace. $R_M, D_M$ are computed from $\hat{s}$ because they are intrinsic to shape, not ambient position.

### 2 Limiting cases (verbatim from the paper)

- manifolds = single points (full invariance): $\alpha_c = 2$ (Cover / Gardner).
- $M$ unstructured points per manifold: $\alpha_c = 2/M$ (capacity of $M{\cdot}P$ points).
- structured point-cloud manifolds, $M$ points each: $\dfrac{2}{M} \le \alpha_c \le 2$.
- unbounded $D$-dim linear subspaces, randomly oriented: $\alpha_c = \dfrac{1}{D + 1/2}$.
- empirical normalization used in the figures: $\alpha_c = 2/\langle M_\mu\rangle_\mu$ (random-points baseline).

### 3 Anchor points and the separating hyperplane

The weight vector normal to the separating plane is a linear combination of anchor points (one per manifold, in the manifold or its convex hull):

$$\vec{w} \;=\; \sum_{\mu=1}^{P} \lambda_\mu \, y_\mu \, \tilde{x}^\mu, \qquad \lambda_\mu \ge 0,$$

with $y_\mu = \pm 1$ the labels. Only manifolds with $\lambda_\mu > 0$ (the "supporting" ones) anchor the plane.

### 4 Capacity (mean-field, CLS form)

The capacity generalizes the Gardner point result. For **points** with margin $\kappa$:

$$\alpha^{-1}(\kappa) \;=\; \int_{-\kappa}^{\infty} Dt\;(t+\kappa)^2,\qquad Dt = \frac{e^{-t^2/2}}{\sqrt{2\pi}}\,dt \quad\Rightarrow\quad \alpha_c(\kappa{=}0)=2 .$$

For **manifolds**, the scalar slack $(t+\kappa)_+$ is replaced by a quantity determined by the anchor-point quadratic program for the Gaussian field $\vec{T}$; schematically,

$$\alpha_M^{-1}(\kappa) \;=\; \Big\langle\, \big[\text{margin slack from the anchor-point QP for } \vec{T}\,\big]^2 \,\Big\rangle_{\vec{T}} .$$

*(The exact closed-form integrand — the constrained QP defining $\hat{s}(\vec{T})$ and the resulting slack — is given in CLS 2018, Phys. Rev. X 8, 031003. The structure to remember: a Gaussian average over the field of a squared margin-effort, reducing to the Gardner point formula when the manifold shrinks to a point.)*

### 5 Geometric properties (Methods, Eqs. 3–4 of the paper)

$$\boxed{\,R_M^2 \;=\; \big\langle \|\hat{s}(\vec{T})\|^2\big\rangle_{\vec{T}}\,}\qquad\qquad
\boxed{\,D_M \;=\; \Big\langle \big(\vec{T}\cdot \hat{s}(\vec{T})/\|\hat{s}(\vec{T})\|\big)^2\Big\rangle_{\vec{T}}\,}$$

- $R_M$ = total variance (rms spread) of the anchor points, normalized by the average inter-center distance — the manifold's effective **extent**.
- $D_M$ = mean-squared projection of the field onto the anchor **direction** — the number of axes the anchors genuinely explore (effective **dimension**). For an isotropic $D$-dim manifold the anchor aligns with the field ($\hat{s}\parallel\vec{T}$), so $\vec{T}\cdot\hat{s}/\|\hat{s}\| \approx \|\vec{T}\|$ and $D_M \approx \langle\|\vec{T}\|^2\rangle = D$.

### 6 Ball equivalence

For $D \gg 1$, $R_M$ and $D_M$ determine the capacity: a manifold has approximately the capacity of a **ball** of radius $R_M$ and dimension $D_M$. (Verified causally in the paper by replacing each manifold with such a ball and recovering the measured capacity.)

### 7 Center correlation (Supplementary Eq. 1)

$$\rho_{CC} \;=\; \Big\langle \frac{|\vec{x}^\mu \cdot \vec{x}^\nu|}{\|\vec{x}^\mu\|\,\|\vec{x}^\nu\|} \Big\rangle_{\mu \neq \nu},$$

the mean absolute cosine between manifold centers. Clustering of centers (high $\rho_{CC}$) is detrimental for random-label separability → **lowers** capacity.

### 8 One-line capacity dependence

$$\alpha_c = \alpha_c(R_M,\, D_M,\, \rho_{CC}),\qquad \text{decreasing in each of } R_M,\, D_M,\, \rho_{CC}.$$

### 9 Empirical findings (AlexNet / VGG-16 / ResNet-50 on ImageNet)

- Capacity rises along the hierarchy, mostly in the last layers; deeper nets reach higher final capacity.
- Driven mainly by **decreasing $D_M$** (≈80 → ≈20 in AlexNet), with a **modest decrease in $R_M$** (≈1.4 → ≈0.8) and **decreasing $\rho_{CC}$**.
- Training-dependent: untrained (random-weight) nets and shuffled-label manifolds show little/no improvement.
- Building blocks trade off: ReLU lowers $R_M, \rho_{CC}$ but raises $D_M$; pooling lowers $R_M, D_M$ but can raise $\rho_{CC}$; only the trained composite blocks improve all three.

---

## 7. Corrections log (vs. the original notes)

1. **$V$ = volume of weight vectors (version space), not anchor points;** scalar $V$, not $\vec V$.
2. **The Gaussian is not a probability over candidate hyperplanes;** it is the distribution of the abstract field $\vec T$ standing in for the other manifolds.
3. **Field values are vectors $\vec T$**, not the moments $D_M/R_M$ (removed that aside).
4. **"saddle-node transition" → separability (capacity) phase transition.**

### 1 Verified against the paper (Nat. Commun. 11:746)

- Verbal definitions of $R_M$ (total variance of anchor points / avg inter-center distance) and $D_M$ (spread of anchor points along manifold axes) — **verbatim** from the "Geometrical framework" section; these are **Methods Eqs. (3)–(4)** as cited in the Results ("Network layers reduce dimension, radius…").
- $\rho_{CC} = \langle |\vec{x}^\mu\!\cdot\vec{x}^\nu| / (\|\vec{x}^\mu\|\,\|\vec{x}^\nu\|)\rangle_{\mu\neq\nu}$ — **exactly** as in the Fig. 7 caption (**Supplementary Eq. (1)**); $\vec{x}^\mu$ = center of object $\mu$.
- Limiting cases ($\alpha_c\le 2$; $2/M$; $1/(D+1/2)$), normalization $\alpha_c=2/\langle M_\mu\rangle_\mu$, ball equivalence, and the empirical numbers ($D_M$ ≈80→20, $R_M$ ≈1.4→0.8 in AlexNet) all confirmed.
- The exact manifold **capacity integrand** is not reproduced in this paper's main text; it is given in CLS 2018 (Phys. Rev. X 8, 031003). Kept schematic here on purpose.
