import { createFileRoute, Link } from "@tanstack/react-router";
import { useEffect, useState } from "react";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "FairLens — Bias Auditing Toolkit Dashboard" },
      {
        name: "description",
        content:
          "FairLens audits ML datasets and models for bias: preprocessing, training, fairness metrics, SHAP explainability, mitigation and downloadable models and reports.",
      },
      { property: "og:title", content: "FairLens — Bias Auditing Toolkit Dashboard" },
      {
        property: "og:description",
        content:
          "A Streamlit dashboard and Flask API for end-to-end ML bias auditing: detect, explain and mitigate unfairness, then export models and reports.",
      },
    ],
  }),
  component: Index,
});

const steps = [
  {
    num: "01",
    title: "Upload & Preprocess",
    body: "CSV or Excel upload, target and sensitive attribute selection, missing-value strategy, duplicate removal, encoding, scaling and train/test split with automatic task detection.",
  },
  {
    num: "02",
    title: "Train & Evaluate",
    body: "Four algorithms per task — logistic/linear, decision tree, random forest and XGBoost — with accuracy, precision, recall, F1, ROC-AUC or MAE/MSE/RMSE/R² and an interactive confusion matrix.",
  },
  {
    num: "03",
    title: "Bias & Explainability",
    body: "Demographic parity difference, disparate impact, equal opportunity difference, per-group accuracy and error, SHAP global and local explanations, plus a recommendation engine.",
  },
  {
    num: "04",
    title: "Mitigation & Downloads",
    body: "Reweighing, Exponentiated Gradient and Threshold Optimizer with before/after fairness and performance comparison, model exports and a full PDF audit report.",
  },
];

const metrics = [
  { label: "Demographic parity", value: "→ 0", note: "closer to zero is fairer" },
  { label: "Disparate impact", value: "≥ 0.80", note: "four-fifths rule" },
  { label: "Equal opportunity", value: "→ 0", note: "true-positive rate gap" },
  { label: "Group error spread", value: "→ 0", note: "regression MAE / RMSE" },
];

function Index() {
  const [backendStatus, setBackendStatus] = useState("Checking backend...");

  useEffect(() => {
    fetch("https://fairlens-amnt.onrender.com/api/health")
      .then((response) => {
        if (!response.ok) {
          throw new Error("Backend unavailable");
        }
        return response.json();
      })
      .then((data) => {
        setBackendStatus(data.message);
      })
      .catch(() => {
        setBackendStatus("Backend unavailable");
      });
  }, []);
  return (
    <main className="mx-auto max-w-6xl px-6 py-16">
      <section className="aurora panel px-8 py-12">
        <p className="text-xs font-semibold uppercase tracking-[0.22em] text-primary">
          Bias auditing toolkit
        </p>
        
        <h1 className="mt-3 text-4xl font-semibold text-foreground sm:text-5xl">
          FairLens
        </h1>
        <p className="mt-4 max-w-2xl text-muted-foreground">
          Analyze, understand, and improve the fairness of machine learning models. FairLens provides an end-to-end workflow for data preprocessing, model training, fairness assessment, explainability, bias mitigation, and comprehensive reporting.
        </p>
        
      </section>

      <section className="mt-14">
        <h2 className="text-xl font-semibold text-foreground">Pipeline</h2>
        <div className="mt-5 grid gap-4 sm:grid-cols-2">
          {steps.map((step) => (
            <Link
  key={step.num}
  to={step.num === "01" ? "/upload" : "/"}
  className="panel block p-6 transition-colors hover:border-primary/50"
>
  <span className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">
    Step {step.num}
  </span>

  <h3 className="mt-1 text-lg font-semibold text-foreground">
    {step.title}
  </h3>

  <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
    {step.body}
  </p>
</Link>
          ))}
        </div>
      </section>

      <section className="mt-14">
        <h2 className="text-xl font-semibold text-foreground">Fairness targets</h2>
        <div className="mt-5 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {metrics.map((metric) => (
            <article key={metric.label} className="panel p-5">
              <p className="text-xs uppercase tracking-[0.14em] text-muted-foreground">
                {metric.label}
              </p>
              <p className="mt-2 font-display text-2xl font-semibold text-primary">{metric.value}</p>
              <p className="mt-1 text-xs text-muted-foreground">{metric.note}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="panel mt-14 p-7">
        <h2 className="text-xl font-semibold text-foreground">Audit workflow</h2>
        <p className="mt-2 text-sm text-muted-foreground">
          Follow a clear four-step workflow from dataset preparation to final audit. Configure your data, train and evaluate a model, investigate fairness and explainability, then mitigate detected bias and export your results.
        </p>
        <div className="mt-5 flex flex-wrap gap-2">
          {["bg-background", "bg-card", "bg-primary", "bg-secondary", "bg-accent", "bg-destructive", "bg-success"].map(
            (token) => (
              <span
                key={token}
                className={`${token} h-10 w-24 rounded-lg border border-border`}
                title={token}
              />
            ),
          )}
        </div>
      </section>
    </main>
  );
}
