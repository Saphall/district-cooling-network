# Five-FSED Physics and Simulation Specification

Status notation:
- [✓] supported by source/code/literature
- [△] modelling choice / hypothesis requiring validation
- [?] unresolved

## FSED 1 — Poor balancing

### Physical question
Under the current cooling load, is each branch/ETS receiving the flow it actually requires?

### Healthy relationship
Q_actual(t) ≈ Q_required(t)

Candidate residual:
r_balance(t) = (Q_actual(t) - Q_required(t)) / Q_required(t)

### Fault mechanism
Hydraulic imbalance causes one or more branches/ETSs to receive inappropriate flow relative to demand.

### Important simulator issue
dcngen uses ideal FCVs at consumers. Simply increasing arbitrary pipe resistance may be compensated by the controller and therefore may not reproduce poor balancing.

### Candidate intervention
[△] Constrain/bias achievable branch/ETS flow relative to the healthy requested flow.

### Validation
Show that:
1. same operating demand is used,
2. required flow is known,
3. delivered flow deviates at the affected location,
4. network solution remains physically valid.

### Evidence status
[?] Numerical real-world severity range still required if available.
[△] Otherwise severity levels will be explicitly simulation-defined sensitivity levels.


## FSED 2 — Excessive control margin

### Physical question
Is the plant supplying substantially more hydraulic head / differential pressure than is necessary to satisfy terminal requirements?

### Healthy relationship
H_supplied(t) ≈ H_min_required(t)

where H_min_required is the minimum sufficient head under the current network operating condition.

### Fault mechanism
Plant/pump control maintains unnecessarily high head or differential pressure.

### Candidate intervention
[△] H_fault(t) = (1 + s) H_healthy(t)

where s is a controlled excess-head severity parameter.

### Expected physical consequence
Higher supplied head / differential pressure and increased hydraulic pumping energy for comparable demand.

### Required additional metric
P_hydraulic = rho * g * Q * H

Do NOT interpret existing thermal plant_power as pump electrical/hydraulic power without verification.

### Validation
Compare required versus supplied head and quantify excess hydraulic energy.

### Evidence status
[?] Real-world numerical excess-margin range still required if available.


## FSED 3 — Clogged water filter

### Physical question
For a given flow, does the filter exhibit greater pressure loss than its clean-state pressure-flow relationship?

### Healthy relationship
DeltaP_filter,healthy = f(Q)

### Fault mechanism
Accumulated material increases local hydraulic resistance.

### Candidate intervention
[△] Introduce an explicit localized filter loss/resistance and increase its resistance for faulted scenarios.

### Expected physical consequence
At comparable flow:
DeltaP_filter,fault > DeltaP_filter,healthy

Depending on controller authority, flow may subsequently decrease or additional head may be required.

### Important simulator issue
Current dcngen does not appear to contain an explicit filter component.

### Validation
Compare clean and clogged pressure-drop versus flow relationships.

### Evidence status
[?] Need defensible clean-filter representation and, where available, literature/manufacturer resistance or pressure-drop information.


## FSED 4 — Heat-exchanger fouling

### Physical question
Has effective heat-transfer conductance degraded relative to the healthy exchanger?

### Healthy relationship
UA = UA_healthy under the installed exchanger state.

### Fault mechanism
Fouling reduces effective heat-transfer conductance.

### Existing simulator mechanism
[✓] dcngen already implements:
UA_fault = m * UA_healthy
with 0 < m < 1.

This mechanism is inherited infrastructure, not claimed as a new implementation.

### Expected physical consequence
Reduced heat-transfer capability. Control may demand additional primary flow to maintain load; sufficiently severe degradation may lead to valve saturation and unmet load.

### Validation
Use a same-seed healthy counterfactual to isolate the effect of changing UA.

### Evidence status
[✓] Mechanism exists in inherited code.
[?] Physical/literature justification for selected severity levels still required; otherwise label them simulation-defined sensitivity levels.


## FSED 5 — Hydronic valve faults

### Healthy relationship
The realized valve/flow response follows the control demand as operating load changes.

### 5A — Stuck valve
Fault mechanism:
Valve ceases responding and remains at its realized state after fault onset.

Candidate intervention:
[△] Freeze valve/realized flow state at onset while required flow continues to evolve.

Expected signature:
command/request changes while actual response remains approximately fixed.

### 5B — Stiction
Fault mechanism:
Mechanical friction prevents smooth response until sufficient command change causes motion.

Candidate intervention:
[△] Stateful deadband / stick-slip model.

Expected signature:
delayed or step-like realized response relative to changing command.

### 5C — Leakage
Fault mechanism:
Valve fails to achieve complete shutoff.

Candidate intervention:
[△] Enforce nonzero flow when commanded closed.

Expected signature:
Q_actual > 0 while Q_command ≈ 0.

Important:
Existing bypass fault must NOT automatically be relabelled as valve leakage.

### Validation
Compare command/request against realized response through time.

### Evidence status
[?] Need literature/manufacturer support for realistic severity parameterization.


# Common Experimental Method

For each FSED:

1. Define healthy physics independently of the fault implementation.
2. Preserve network, load realization, seed and unaffected parameters.
3. Apply one explicit physical intervention.
4. Run healthy H(t) and faulted F(t).
5. Compute diagnostic residuals R(t) = F(t) - H(t) for simulator validation.
6. Verify network/thermal physics remain valid.
7. Verify the intended fault-specific mechanism appears.
8. Record configuration, seed, location, intervention and severity.
9. Distinguish sourced parameters from simulation-defined sensitivity choices.
10. Do not assume the eventual ML detector has access to H(t); the healthy twin is privileged simulator ground truth for validation.

# Contribution Boundary

Inherited dcngen:
- network generation
- hydraulic/thermal simulation infrastructure
- load generation
- existing fouling implementation
- existing leak/bypass infrastructure
- existing validation/recording infrastructure

This thesis extension:
- operational definition of the five SWiM FSEDs,
- healthy-state physical relationships for each,
- simulator interventions required to represent missing FSEDs,
- controlled severity specification,
- healthy-fault counterfactual validation,
- fault-specific observable/residual analysis,
- later comparison of detection methods using the validated synthetic scenarios.
