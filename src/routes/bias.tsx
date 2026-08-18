import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

export const Route = createFileRoute("/bias")({
  component: BiasPage,
});

const API_URL = "http://127.0.0.1:5000";

type BiasResult = {
  ok: boolean;
  success?: boolean;
  session_id: string;
  task_type: string;
  bias_detected: boolean;
  sensitive_features: string[];
  fairness_results: Record<string, FeatureFairnessResult>;
  recommendations: Record<string, Recommendation>;

  class_imbalance: ClassImbalanceResult;

  explainability: ExplainabilityResult;
};

type ExplainabilityResult = {
  feature_importance: FeatureImportance[];
};

type FeatureImportance = {
  Feature: string;
  "SHAP Importance": number;
};

type FeatureFairnessResult = {
  metrics: Record<string, any>;
  bias_detection: {
    bias_detected: boolean;
    reasons: string[];
  };
};

type Recommendation = {
  bias_detected: boolean;
  strategy: string | null;
  reason: string;
  group_imbalance: boolean;
  group_distribution: Record<string, number>;
};

type ClassImbalanceResult = {
  detected: boolean;
  distribution: Record<string, number>;
  imbalance_ratio?: number;
  groups: Record<
    string,
    Record<string, SensitiveGroupImbalance>
  >;
};

type SensitiveGroupImbalance = {
  distribution: Record<string, number>;
  imbalance_ratio: number | null;
  detected: boolean;
  samples: number;
};

function BiasPage() {
  const [result, setResult] = useState<BiasResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const sessionId = localStorage.getItem("fairlens_session_id");
  const taskType = localStorage.getItem("fairlens_task_type");

  useEffect(() => {
    const detectBias = async () => {
      if (!sessionId) {
        setError(
          "Dataset session not found. Please complete preprocessing and training first."
        );
        setLoading(false);
        return;
      }

      setLoading(true);
      setError("");

      try {
        const response = await fetch(
          `${API_URL}/api/bias-detection`,
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              session_id: sessionId,
            }),
          }
        );

        const data = await response.json();

        if (!response.ok || !data.ok) {
          throw new Error(
            data.error || "Bias detection failed."
          );
        }

        setResult(data);
        localStorage.setItem(
  "fairlens_sensitive_features",
  JSON.stringify(data.sensitive_features)
);

      } catch (err) {
        console.error(err);

        setError(
          err instanceof Error
            ? err.message
            : "Something went wrong during bias detection."
        );
      } finally {
        setLoading(false);
      }
    };

    detectBias();
  }, [sessionId]);

  return (
    <main className="mx-auto max-w-6xl px-6 py-16">

      {/* HEADER */}

      <section className="aurora panel px-8 py-10">
        <p className="text-xs font-semibold uppercase tracking-[0.22em] text-primary">
          Step 03
        </p>

        <h1 className="mt-3 text-4xl font-semibold text-foreground">
          Bias Detection & Explainability
        </h1>

        <p className="mt-4 max-w-3xl text-muted-foreground">
          Analyze your trained model for fairness across all
          selected sensitive features and understand which
          factors influence its predictions.
        </p>
      </section>

      {/* DATASET STATUS */}

      <section className="panel mt-10 p-6">
        <h2 className="text-xl font-semibold text-foreground">
          Analysis Status
        </h2>

        <div className="mt-5 grid gap-4 sm:grid-cols-2">

          <div className="rounded-xl border border-border bg-muted/40 p-5">
            <p className="text-xs uppercase tracking-[0.14em] text-muted-foreground">
              Task Type
            </p>

            <p className="mt-2 text-2xl font-semibold text-primary">
              {taskType || "Not detected"}
            </p>
          </div>

          <div className="rounded-xl border border-border bg-muted/40 p-5">
            <p className="text-xs uppercase tracking-[0.14em] text-muted-foreground">
              Dataset Session
            </p>

            <p className="mt-2 truncate text-sm font-medium text-foreground">
              {sessionId || "Not found"}
            </p>
          </div>

        </div>
      </section>

      {/* LOADING */}

      {loading && (
        <section className="panel mt-10 p-8 text-center">
          <p className="text-lg font-semibold text-foreground">
            Analyzing model fairness...
          </p>

          <p className="mt-2 text-sm text-muted-foreground">
            Evaluating all selected sensitive features.
          </p>
        </section>
      )}

      {/* ERROR */}

      {error && !loading && (
        <section className="mt-10 rounded-xl border border-destructive/30 bg-destructive/10 p-5 text-sm text-destructive">
          {error}
        </section>
      )}

      {/* RESULTS */}

      {result && !loading && (
        <section className="mt-10">

          <div className="panel p-6">
            <h2 className="text-xl font-semibold text-foreground">
              Fairness Analysis
            </h2>

            <p className="mt-1 text-sm text-muted-foreground">
              Results for each selected sensitive feature.
            </p>
          </div>

          {Object.entries(result.fairness_results).map(
            ([feature, featureResult]) => {

              const metrics = featureResult.metrics;
              const bias = featureResult.bias_detection;
              const recommendation =
                result.recommendations?.[feature];

              return (
                <section
                  key={feature}
                  className="panel mt-6 p-6"
                >

                  {/* FEATURE HEADER */}

                  <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">

                    <div>
                      <p className="text-xs font-semibold uppercase tracking-[0.14em] text-primary">
                        Sensitive Feature
                      </p>

                      <h3 className="mt-1 text-2xl font-semibold text-foreground">
                        {feature}
                      </h3>
                    </div>

                    <div
                      className={`rounded-full px-4 py-2 text-sm font-semibold ${
                        bias.bias_detected
                          ? "bg-destructive/10 text-destructive"
                          : "bg-primary/10 text-primary"
                      }`}
                    >
                      {bias.bias_detected
                        ? "Bias Detected"
                        : "No Significant Bias"}
                    </div>

                  </div>

                  {/* CLASSIFICATION METRICS */}

                  {result.task_type === "classification" && (
                    <div className="mt-6">

                      <h4 className="text-lg font-semibold text-foreground">
                        Fairness Metrics
                      </h4>

                      <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">

                        <MetricCard
                          label="Demographic Parity Difference"
                          value={metrics["Demographic Parity Difference"]}
                          description="Difference in positive prediction rates between groups."
                        />

                        <MetricCard
                          label="Disparate Impact"
                          value={metrics["Disparate Impact"]}
                          description="Ratio of positive prediction rates between groups."
                        />

                        <MetricCard
                          label="Equal Opportunity Difference"
                          value={metrics["Equal Opportunity Difference"]}
                          description="Difference in true positive rates between groups."
                        />

                        <MetricCard
                          label="Equalized Odds"
                          value={metrics["Equalized Odds"]}
                          description="Maximum difference in TPR or FPR between groups."
                        />

                      </div>


                    </div>
                  )}

                  {/* REGRESSION METRICS */}

                  {result.task_type === "regression" && (
                    <div className="mt-6">

                      <h4 className="text-lg font-semibold text-foreground">
                        Fairness Metrics
                      </h4>

                      <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">

                        <MetricCard
                          label="Mean Prediction Difference"
                          value={
                            metrics[
                              "Mean Prediction Difference"
                            ]
                          }
                          description="Difference in average predictions between groups."
                        />

                      </div>

                      {metrics["Group MAE"] && (
                        <div className="mt-6">

                          <h4 className="text-base font-semibold text-foreground">
                            Group MAE
                          </h4>

                          <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">

                            {Object.entries(
                              metrics["Group MAE"]
                            ).map(
                              ([group, value]) => (
                                <div
                                  key={group}
                                  className="rounded-xl border border-border bg-muted/40 p-4"
                                >
                                  <p className="text-xs uppercase tracking-wide text-muted-foreground">
                                    {group}
                                  </p>

                                  <p className="mt-2 text-xl font-semibold text-primary">
                                    {formatNumber(value)}
                                  </p>
                                </div>
                              )
                            )}

                          </div>
                        </div>
                      )}

                      {metrics["Group RMSE"] && (
                        <div className="mt-6">

                          <h4 className="text-base font-semibold text-foreground">
                            Group RMSE
                          </h4>

                          <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">

                            {Object.entries(
                              metrics["Group RMSE"]
                            ).map(
                              ([group, value]) => (
                                <div
                                  key={group}
                                  className="rounded-xl border border-border bg-muted/40 p-4"
                                >
                                  <p className="text-xs uppercase tracking-wide text-muted-foreground">
                                    {group}
                                  </p>

                                  <p className="mt-2 text-xl font-semibold text-primary">
                                    {formatNumber(value)}
                                  </p>
                                </div>
                              )
                            )}

                          </div>
                        </div>
                      )}

                    </div>
                  )}

                  {/* BIAS REASONS */}

                  {bias.bias_detected &&
                    bias.reasons.length > 0 && (
                      <div className="mt-6 rounded-xl border border-destructive/30 bg-destructive/10 p-5">

                        <h4 className="font-semibold text-destructive">
                          Why was bias detected?
                        </h4>

                        <ul className="mt-3 space-y-2 text-sm text-destructive">
                          {bias.reasons.map(
                            (reason, index) => (
                              <li key={index}>
                                • {reason}
                              </li>
                            )
                          )}
                        </ul>

                      </div>
                    )}

                  {/* RECOMMENDATION */}

                  {recommendation && (
                    <div className="mt-6 rounded-xl border border-border bg-muted/40 p-5">

                      <p className="text-xs uppercase tracking-[0.14em] text-muted-foreground">
                        Recommended Mitigation
                      </p>

                      <h4 className="mt-2 text-xl font-semibold text-primary">
                        {recommendation.strategy ||
                          "No mitigation required"}
                      </h4>

                      <p className="mt-2 text-sm text-muted-foreground">
                        {recommendation.reason}
                      </p>

                    </div>
                  )}


                </section>
              );
            }
          )}

        {/* ============================================================
    EXPLAINABILITY
   ============================================================ */}

<section className="panel mt-8 p-6">

  <div>
    <p className="text-xs font-semibold uppercase tracking-[0.14em] text-primary">
      Explainability
    </p>

    <h2 className="mt-2 text-2xl font-semibold text-foreground">
      Model Feature Importance
    </h2>

    <p className="mt-2 max-w-3xl text-sm text-muted-foreground">
      SHAP analysis shows which features have the greatest
      influence on the model's predictions.
    </p>
  </div>

  {result.explainability?.feature_importance?.length > 0 ? (

    <div className="mt-6 overflow-x-auto rounded-xl border border-border">

      <table className="w-full text-sm">

        <thead>
          <tr className="bg-muted/50 border-b border-border">

            <th className="px-5 py-4 text-left font-semibold text-foreground">
              Rank
            </th>

            <th className="px-5 py-4 text-left font-semibold text-foreground">
              Feature
            </th>

            <th className="px-5 py-4 text-right font-semibold text-foreground">
              SHAP Importance
            </th>

          </tr>
        </thead>

        <tbody>

          {result.explainability.feature_importance.map(
            (item, index) => (

              <tr
                key={`${item.Feature}-${index}`}
                className="border-b border-border last:border-b-0"
              >

                <td className="px-5 py-4 font-semibold text-primary">
                  {index + 1}
                </td>

                <td className="px-5 py-4 font-medium text-foreground">
                  {item.Feature}
                </td>

                <td className="px-5 py-4 text-right font-semibold text-primary">
                  {formatNumber(item["SHAP Importance"])}
                </td>

              </tr>

            )
          )}

        </tbody>

      </table>

      <div className="mt-8">

  <h3 className="text-lg font-semibold text-foreground">
    SHAP Importance Chart
  </h3>

  <p className="mt-1 text-sm text-muted-foreground">
    Higher values indicate a greater influence on model predictions.
  </p>

  <div className="mt-6 h-[400px] w-full">

    <ResponsiveContainer width="100%" height="100%">

      <BarChart
        data={result.explainability.feature_importance}
        layout="vertical"
        margin={{
          top: 10,
          right: 30,
          left: 40,
          bottom: 10,
        }}
      >

        <CartesianGrid strokeDasharray="3 3" />

        <XAxis
          type="number"
          domain={[0, "auto"]}
        />

        <YAxis
          type="category"
          dataKey="Feature"
          width={120}
        />

        <Tooltip />

        <Bar
          dataKey="SHAP Importance"
          radius={[0, 6, 6, 0]}
        />

      </BarChart>

    </ResponsiveContainer>

  </div>

</div>

    </div>

  ) : (

    <div className="mt-6 rounded-xl border border-border bg-muted/40 p-5">

      <p className="text-sm text-muted-foreground">
        Feature importance could not be generated for this model.
      </p>

    </div>

  )}

</section>

{/* ============================================================
   CLASS IMBALANCE ANALYSIS
   ============================================================ */}

{result.task_type === "classification" &&
  result.class_imbalance && (

  <section className="panel mt-8 p-6">

    <div>
      <p className="text-xs font-semibold uppercase tracking-[0.14em] text-primary">
        Imbalance Analysis
      </p>

      <h2 className="mt-2 text-2xl font-semibold text-foreground">
        Class Distribution
      </h2>

      <p className="mt-2 max-w-3xl text-sm text-muted-foreground">
        Analysis of target classes to identify potential
        class imbalance overall and across sensitive groups.
      </p>
    </div>

    {/* OVERALL STATUS */}

    <div
      className={`mt-6 rounded-xl border p-5 ${
        result.class_imbalance.detected
          ? "border-destructive/30 bg-destructive/10"
          : "border-primary/30 bg-primary/10"
      }`}
    >

      <h3
        className={`font-semibold ${
          result.class_imbalance.detected
            ? "text-destructive"
            : "text-primary"
        }`}
      >
        {result.class_imbalance.detected
          ? "Class Imbalance Detected"
          : "No Significant Class Imbalance"}
      </h3>

      <p className="mt-2 text-sm text-muted-foreground">
        {result.class_imbalance.detected
          ? "The target classes are not evenly distributed. This may affect model performance and fairness."
          : "The target classes appear reasonably balanced."}
      </p>

      {result.class_imbalance.imbalance_ratio && (
        <p className="mt-2 text-xs text-muted-foreground">
          Imbalance Ratio:{" "}
          <span className="font-semibold">
            {result.class_imbalance.imbalance_ratio.toFixed(2)}
          </span>
        </p>
      )}

    </div>

    {/* OVERALL TARGET DISTRIBUTION */}

    <div className="mt-8">

      <h3 className="text-lg font-semibold text-foreground">
        Overall Target Distribution
      </h3>

      <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">

        {Object.entries(
          result.class_imbalance.distribution
        ).map(([label, proportion]) => (

          <div
            key={label}
            className="rounded-xl border border-border bg-muted/40 p-5"
          >

            <p className="text-xs uppercase tracking-wide text-muted-foreground">
              Class {label}
            </p>

            <p className="mt-2 text-2xl font-semibold text-primary">
              {(Number(proportion) * 100).toFixed(2)}%
            </p>

          </div>

        ))}

      </div>

    </div>

    {/* PER SENSITIVE FEATURE */}

    {Object.entries(
      result.class_imbalance.groups
    ).map(([feature, groups]) => (

      <div
        key={feature}
        className="mt-8"
      >

        <h3 className="text-lg font-semibold text-foreground">
          Class Distribution by {feature}
        </h3>

        <div className="mt-4 space-y-4">

          {Object.entries(groups).map(
            ([group, groupResult]) => (

              <div
                key={group}
                className="rounded-xl border border-border bg-muted/40 p-5"
              >

                <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">

                  <div>
                    <p className="font-semibold text-foreground">
                      {group}
                    </p>

                    <p className="text-xs text-muted-foreground">
                      {groupResult.samples} samples
                    </p>
                  </div>

                  <div
                    className={`rounded-full px-3 py-1 text-xs font-semibold ${
                      groupResult.detected
                        ? "bg-destructive/10 text-destructive"
                        : "bg-primary/10 text-primary"
                    }`}
                  >
                    {groupResult.detected
                      ? "Imbalanced"
                      : "Balanced"}
                  </div>

                </div>

                <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">

                  {Object.entries(
                    groupResult.distribution
                  ).map(
                    ([label, proportion]) => (

                      <div
                        key={label}
                        className="rounded-lg border border-border p-4"
                      >

                        <p className="text-xs uppercase tracking-wide text-muted-foreground">
                          Class {label}
                        </p>

                        <p className="mt-2 text-xl font-semibold text-primary">
                          {(Number(proportion) * 100).toFixed(2)}%
                        </p>

                      </div>

                    )
                  )}

                </div>

                {groupResult.imbalance_ratio !== null && (
                  <p className="mt-4 text-xs text-muted-foreground">
                    Imbalance Ratio:{" "}
                    <span className="font-semibold text-foreground">
                      {groupResult.imbalance_ratio.toFixed(2)}
                    </span>
                  </p>
                )}

              </div>

            )
          )}

        </div>

      </div>

    ))}

  </section>
)}

{/* AVAILABLE MITIGATION TECHNIQUES */}

{result.bias_detected && (
  <div className="mt-6">

    <p className="text-xs uppercase tracking-[0.14em] text-muted-foreground">
      Available Mitigation Techniques
    </p>

    <div className="mt-4 grid gap-4 md:grid-cols-3">

      <div className="rounded-xl border border-border bg-muted/40 p-5">

        <h4 className="text-lg font-semibold text-primary">
          Reweighing
        </h4>

        <p className="mt-2 text-sm text-muted-foreground">
          Assigns different weights to training samples so
          underrepresented sensitive groups have greater influence.
        </p>

      </div>

      <div className="rounded-xl border border-border bg-muted/40 p-5">

        <h4 className="text-lg font-semibold text-primary">
          Exponentiated Gradient
        </h4>

        <p className="mt-2 text-sm text-muted-foreground">
          Trains a fairness-constrained model to reduce
          disparities between sensitive groups.
        </p>

      </div>

      <div className="rounded-xl border border-border bg-muted/40 p-5">

        <h4 className="text-lg font-semibold text-primary">
          Threshold Optimizer
        </h4>

        <p className="mt-2 text-sm text-muted-foreground">
          Adjusts classification thresholds to improve fairness
          while maintaining useful predictive performance.
        </p>

      </div>

    </div>

  </div>
)}

{/* ============================================================
   CONTINUE TO MITIGATION
   ============================================================ */}

<section className="mt-10 flex justify-end">
  <button
    type="button"
    onClick={() => {
      window.location.href = "/mitigation";
    }}
    className="rounded-xl bg-primary px-8 py-3 text-sm font-semibold text-primary-foreground transition hover:opacity-90"
  >
    Mitigation →
  </button>
</section>

        </section>
      )}

    </main>
  );
}


/* ============================================================
   METRIC CARD
   ============================================================ */

function MetricCard({
  label,
  value,
  description,
}: {
  label: string;
  value: any;
  description: string;
}) {
  return (
    <article className="rounded-xl border border-border bg-muted/40 p-5">

      <p className="text-xs uppercase tracking-[0.12em] text-muted-foreground">
        {label}
      </p>

      <p className="mt-2 text-2xl font-semibold text-primary">
        {formatNumber(value)}
      </p>

      <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
        {description}
      </p>

    </article>
  );
}


/* ============================================================
   FORMAT NUMBER
   ============================================================ */

function formatNumber(value: any) {
  const number = Number(value);

  if (!Number.isFinite(number)) {
    return "N/A";
  }

  return number.toFixed(4);
}