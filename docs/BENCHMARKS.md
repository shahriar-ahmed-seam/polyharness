# Multi-Harness Evaluation & Overfitting Benchmark Suite

This document defines the formal metrics, benchmarking methodology, and statistical validation used by PolyHarness to detect and quantify agent overfitting.

---

## 1. Mathematical Metric Formulations

### 1.1 Harness Overfitting Coefficient (HOC)
The **Harness Overfitting Coefficient** measures the relative performance collapse when an agent is evaluated on harnesses outside its native training environment.

$$\text{HOC} = \max\left(0, \frac{\text{Acc}_{\text{native}} - \overline{\text{Acc}}_{\text{unseen}}}{\max(\text{Acc}_{\text{native}}, 10^{-6})}\right)$$

Where:
- $\text{Acc}_{\text{native}}$ is the benchmark accuracy on the harness used during fine-tuning (e.g., ReAct).
- $\overline{\text{Acc}}_{\text{unseen}}$ is the unweighted mean accuracy across unseen evaluation harnesses:
  $$\overline{\text{Acc}}_{\text{unseen}} = \frac{1}{|H_{\text{unseen}}|} \sum_{h \in H_{\text{unseen}}} \text{Acc}_h$$

**Interpretation:**
- $\text{HOC} = 0.0$: Generalist agent; invariant across harness variations.
- $\text{HOC} > 0.35$: Significant overfitting; high risk of production collapse.
- $\text{HOC} \to 1.0$: Complete brittle collapse outside training framework.

### 1.2 Cross-Harness Transfer Score (CHTS)
The **Cross-Harness Transfer Score** quantifies how much task capability generalizes to foreign execution environments:

$$\text{CHTS} = \frac{\overline{\text{Acc}}_{\text{unseen}}}{\max(\text{Acc}_{\text{native}}, 10^{-6})}$$

- **Target**: $\text{CHTS} \ge 0.85$ for enterprise deployment.

### 1.3 Cascading Divergence Rate (CDR)
The **Cascading Divergence Rate** measures the probability that an off-distribution observation or tool error triggers an unrecoverable crash loop:

$$\text{CDR} = \frac{\sum_{h \in H} N_{\text{cascade\_failures}}(h)}{\sum_{h \in H} N_{\text{tasks}}(h)}$$

- **Target**: $\text{CDR} \le 0.10$ for safe autonomous operation.

---

## 2. Statistical Bootstrap Confidence Bounds

To ensure statistical rigor, PolyHarness computes empirical 95% bootstrap confidence intervals ($B = 1000$ iterations) for both HOC and CHTS:

$$\text{CI}_{95\%} = \left[ q_{0.025}(\{\text{HOC}^{(b)}\}_{b=1}^B), \; q_{0.975}(\{\text{HOC}^{(b)}\}_{b=1}^B) \right]$$

This prevents premature conclusions on small sample evaluation suites.

---

## 3. Production Readiness Criteria

An agent checkpoint is certified **Production-Ready** if and only if it satisfies all four threshold criteria:

| Metric | Threshold | Rationale |
|---|---|---|
| **HOC** | $\le 0.25$ | Bounds the maximum allowable performance degradation. |
| **CHTS** | $\ge 0.80$ | Ensures at least 80% cross-framework knowledge transfer. |
| **CDR** | $\le 0.15$ | Guarantees error-recovery self-healing capability. |
| **Tool Call Syntax Error Rate** | $\le 0.05$ | Verifies structural robustness to syntax perturbations. |

---

## 4. Running Multi-Harness Benchmarks via CLI

```bash
# Run multi-harness evaluation audit
polyharness eval \
  --profile polyharness_generalist \
  --native-harness openai \
  --output audit_results.json
```
