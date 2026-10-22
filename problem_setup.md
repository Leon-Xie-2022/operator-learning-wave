# Problem Setup

## 1. Scope of This Study

This project studies operator learning for a **one-dimensional variable-coefficient wave equation** in conservative form. The problem setting is fixed for the entire project and will serve as the single reference definition for all later components, including:

- numerical solver design,
- dataset generation,
- model implementation,
- experiment design,
- analysis scripts,
- and paper writing.

No later code, figure, table, or section of the manuscript should deviate from the definitions given in this file.

---

## 2. Governing PDE

We consider the following conservative-form wave equation:

$$
u_{tt} = \partial_x\big(c(x)^2 u_x\big), \qquad x \in [0,1],\; t \in [0,T].
$$

This is a one-dimensional second-order hyperbolic partial differential equation with spatially varying wave speed \(c(x)\).

### Interpretation

- \(u(x,t)\) denotes the scalar wave field.
- \(c(x)\) denotes the spatially varying wave speed.
- The conservative form \(\partial_x(c(x)^2 u_x)\) is used to preserve the physically meaningful structure of the model.
- The coefficient \(c(x)\) depends only on space and is fixed for each sample.

---

## 3. Boundary Conditions

We impose fixed-endpoint Dirichlet boundary conditions:

$$
u(0,t) = u(1,t) = 0, \qquad t \in [0,T].
$$

### Interpretation

These boundary conditions correspond to a string or wave medium whose two endpoints are held fixed. They are also numerically convenient and naturally compatible with sine-based initial conditions.

---

## 4. Initial Conditions

The initial state is defined by:

$$
u(x,0) = u_0(x), \qquad u_t(x,0) = 0.
$$

### Interpretation

- \(u_0(x)\) is the initial displacement profile.
- The initial velocity is set to zero for all samples.
- Therefore, each sample is fully determined by the pair \((u_0, c)\).

This choice reduces unnecessary complexity while keeping the operator-learning task physically meaningful and nontrivial.

---

## 5. Symbols and Definitions

The following notation is fixed throughout the project.

### Primary variables

- \(x \in [0,1]\): spatial coordinate
- \(t \in [0,T]\): time variable
- \(T > 0\): fixed terminal time
- \(u(x,t)\): wave solution
- \(u_0(x)\): initial displacement
- \(u_t(x,t)\): time derivative of \(u\)
- \(u_x(x,t)\): spatial derivative of \(u\)
- \(c(x)\): spatially varying wave speed

### Derived quantities

- \(u(\cdot, T)\): terminal-state solution function at time \(T\)
- \(\mathcal G\): solution operator to be learned

---

## 6. Operator-Learning Objective

The learning task is to approximate the terminal-time solution operator

$$
\mathcal G : (u_0, c) \mapsto u(\cdot, T).
$$

In words:

- **input**: the initial displacement function \(u_0(x)\) and the wave-speed function \(c(x)\),
- **output**: the full spatial solution profile \(u(x,T)\) at a fixed terminal time \(T\).

### Why the output is fixed at a single terminal time

The project does **not** learn the entire time trajectory \(u(x,t)\) for all \(t\). Instead, it learns only the map to the terminal state \(u(\cdot,T)\). This decision is fixed for the whole project.

#### Reasons for this choice

1. The task becomes a clean function-to-function operator-learning problem.
2. Data generation is simpler and more reliable.
3. The training target is lower-dimensional than full trajectory prediction.
4. The paper becomes more focused and easier to complete at workshop quality.
5. The comparison between FNO and DeepONet is cleaner in this setting.

---

## 7. Why This PDE Form Is Chosen

The conservative-form PDE

$$
u_{tt} = \partial_x\big(c(x)^2 u_x\big)
$$

is chosen deliberately and should not be replaced by a non-conservative surrogate such as \(u_{tt} = c(x)^2 u_{xx}\).

### Main reasons

#### 7.1 Physical consistency

The conservative form is more faithful to wave propagation in an inhomogeneous medium. It preserves the correct structural role of the spatially varying coefficient and is therefore more suitable for an AI for Physics paper.

#### 7.2 Natural weak-form and energy structure

This form admits a natural weak formulation and supports meaningful energy-based reasoning. That is important for:

- numerical discretization,
- solver validation,
- and later auxiliary physical analysis.

#### 7.3 Better alignment with numerical methods

A conservative discretization can be constructed directly from this PDE form. That gives a more principled reference solver and avoids mismatches between the physics in the paper and the numerics in the code.

#### 7.4 Better justification in paper writing

For workshop reviewers in AI for Physics, a physically consistent governing equation is significantly easier to justify than an expedient but non-conservative surrogate.

---

## 8. Why the Study Is Restricted to 1D

This project studies only the one-dimensional case.

### Reasons

1. The numerical solver can be implemented and verified reliably.
2. The data-generation pipeline remains lightweight and fully controlled.
3. Frequency-domain error analysis is much easier to interpret in 1D.
4. The project scope stays appropriate for a workshop paper.
5. The final paper can emphasize clarity and correctness rather than overextension.

This restriction is intentional. It is a design choice, not a weakness to be hidden.

---

## 9. Fixed Endpoint Setting

The project uses fixed endpoints for the entire study.

### Why this is fixed

- Dirichlet boundaries are physically interpretable.
- They work naturally with sine-basis initial conditions.
- They simplify solver design and verification.
- They avoid introducing an additional experimental axis such as boundary-condition variation.

No periodic or Neumann boundary conditions are considered in the main study.

---

## 10. Role of the Constant-Coefficient Case

The constant-coefficient case is retained only as an auxiliary reference setting.

### It is used for

1. **solver sanity checks**  
   to verify that the numerical solver reproduces expected wave behavior in the simplest case;

2. **physical intuition**  
   to provide a familiar reference for how waves propagate under simpler conditions;

3. **frequency-domain interpretation**  
   to support discussion of modal or spectral behavior at an intuitive level.

### It is not used for

- building a second full experimental track,
- dominating the main results section,
- replacing the variable-coefficient task as the real problem of the paper.

The main scientific task of the paper remains the **variable-coefficient conservative wave equation**.

---

## 11. Fixed Research Scope

The following restrictions are part of the official problem definition.

### Included

- 1D conservative variable-coefficient wave equation
- Dirichlet boundary conditions
- zero initial velocity
- terminal-state operator learning
- input pair \((u_0, c)\)
- output \(u(\cdot,T)\)

### Excluded from the main study

- full trajectory prediction
- 2D or higher-dimensional PDEs
- changing boundary conditions
- stochastic forcing terms
- time-dependent coefficients \(c(x,t)\)
- inverse problems
- coefficient identification tasks

These exclusions are intentional and must remain fixed unless the entire project plan is formally revised.

---

## 12. Practical Consequences for Later Stages

This problem definition determines the downstream implementation requirements.

### For the solver

- the solver must solve the conservative PDE, not a non-conservative approximation;
- Dirichlet boundaries must be enforced exactly in the numerical scheme;
- the solver input must be \((u_0, c)\);
- the solver output must be \(u(\cdot,T)\).

### For the dataset

- each sample must contain at least \(u_0\), \(c\), and \(u_T\);
- the coefficient field \(c(x)\) must remain positive;
- all data-generation scripts must follow this PDE setting exactly.

### For the models

- FNO and DeepONet must be trained to map \((u_0,c)\) to \(u(\cdot,T)\);
- no model should be trained on a different target without explicit redefinition of the project scope.

### For the paper

- the governing PDE, boundary conditions, and operator objective must be stated consistently in the manuscript;
- all experiments must reflect this same setup;
- all analysis claims must be compatible with this exact mathematical setting.

---

## 13. Final Frozen Definition

The project is formally defined as follows:

> Learn the terminal-time solution operator for the one-dimensional conservative variable-coefficient wave equation
> $$
> u_{tt} = \partial_x\big(c(x)^2 u_x\big), \qquad x\in[0,1],\ t\in[0,T],
> $$
> under fixed-endpoint Dirichlet boundary conditions
> $$
> u(0,t)=u(1,t)=0,
> $$
> with initial conditions
> $$
> u(x,0)=u_0(x),\qquad u_t(x,0)=0,
> $$
> by approximating the mapping
> $$
> \mathcal G:(u_0,c)\mapsto u(\cdot,T).
> $$

This definition is the single source of truth for the rest of the project.
