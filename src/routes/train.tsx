import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";

export const Route = createFileRoute("/train")({
  component: TrainPage,
});

const API_URL = "http://127.0.0.1:5000";

type TrainResult = {
  ok: boolean;
  success: boolean;
  session_id: string;
  model_name: string;
  task_type: string;
  metrics: Record<string, any>;
};

function TrainPage() {
  const [model, setModel] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<TrainResult | null>(null);
  const [error, setError] = useState("");

  const sessionId = localStorage.getItem("fairlens_session_id");
  const taskType = localStorage.getItem("fairlens_task_type");

  const classificationModels = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "XGBoost",
  ];

  const regressionModels = [
    "Linear Regression",
    "Decision Tree",
    "Random Forest",
    "XGBoost",
  ];

  const availableModels =
    taskType === "regression"
      ? regressionModels
      : classificationModels;

  const handleTrain = async () => {
    if (!sessionId) {
      setError(
        "Dataset session not found. Please go back to Upload & Preprocess."
      );
      return;
    }

    if (!taskType) {
      setError(
        "Task type not found. Please complete preprocessing first."
      );
      return;
    }

    if (!model) {
      setError("Please select a model.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API_URL}/api/train`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          session_id: sessionId,
          model: model,
        }),
      });

      const data = await response.json();

      if (!response.ok || !data.ok) {
        throw new Error(
          data.error || "Model training failed."
        );
      }

      setResult(data);

    } catch (err) {
      console.error(err);

      setError(
        err instanceof Error
          ? err.message
          : "Something went wrong during training."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="mx-auto max-w-6xl px-6 py-16">

      {/* HEADER */}

      <section className="aurora panel px-8 py-10">
        <p className="text-xs font-semibold uppercase tracking-[0.22em] text-primary">
          Step 02
        </p>

        <h1 className="mt-3 text-4xl font-semibold text-foreground">
          Train & Evaluate
        </h1>

        <p className="mt-4 max-w-3xl text-muted-foreground">
          Select a machine learning model and train it using
          the preprocessed dataset.
        </p>
      </section>

      {/* DATASET INFO */}

      <section className="panel mt-10 p-6">
        <h2 className="text-xl font-semibold text-foreground">
          Dataset Status
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

      {/* MODEL SELECTION */}

      <section className="panel mt-10 p-6">
        <h2 className="text-xl font-semibold text-foreground">
          Model Selection
        </h2>

        <p className="mt-1 text-sm text-muted-foreground">
          Choose a model for {taskType || "the detected task"}.
        </p>

        <div className="mt-6 grid gap-4 sm:grid-cols-2">

          {availableModels.map((modelName) => (
            <button
              key={modelName}
              type="button"
              onClick={() => {
                setModel(modelName);
                setError("");
              }}
              className={`rounded-xl border p-5 text-left transition ${
                model === modelName
                  ? "border-primary bg-primary/10"
                  : "border-border bg-muted/30 hover:border-primary/50"
              }`}
            >
              <p className="font-semibold text-foreground">
                {modelName}
              </p>

              <p className="mt-1 text-sm text-muted-foreground">
                {modelName} model
              </p>
            </button>
          ))}

        </div>

        {model && (
          <p className="mt-5 text-sm text-primary">
            Selected model: <strong>{model}</strong>
          </p>
        )}

        {error && (
          <div className="mt-5 rounded-xl border border-destructive/30 bg-destructive/10 p-4 text-sm text-destructive">
            {error}
          </div>
        )}

        <div className="mt-8 flex justify-end">
          <button
            type="button"
            onClick={handleTrain}
            disabled={loading || !model}
            className="rounded-xl bg-primary px-7 py-3 text-sm font-semibold text-primary-foreground disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loading ? "Training..." : "Train Model →"}
          </button>
        </div>
      </section>

      {/* RESULTS */}

      {result && (
        <section className="mt-10">

          <div className="panel p-6">
            <h2 className="text-xl font-semibold text-foreground">
              Evaluation Results
            </h2>

            <div className="mt-5 grid gap-4 sm:grid-cols-2">

              <ResultCard
                label="Model"
                value={result.model_name}
              />

              <ResultCard
                label="Task"
                value={result.task_type}
              />

              {Object.entries(result.metrics)
                .filter(
                  ([key]) =>
                    key !== "confusion_matrix" &&
                    key !== "classification_report" &&
                    key !== "Labels"
                )
                .map(([key, value]) => (
                  <ResultCard
                    key={key}
                    label={formatMetricName(key)}
                    value={
                      typeof value === "number"
                        ? value.toFixed(4)
                        : String(value ?? "N/A")
                    }
                  />
                ))}

            </div>
          </div>

          {/* CLASSIFICATION REPORT */}
        {/* CLASSIFICATION REPORT */}

{result.metrics["classification_report"] && (
  <div className="panel mt-6 p-6">
    <h3 className="text-lg font-semibold text-foreground">
      Classification Report
    </h3>

    <p className="mt-1 text-sm text-muted-foreground">
      Detailed performance for each class.
    </p>

    <div className="mt-5 overflow-x-auto">
      <table className="w-full text-left text-sm">
        <thead>
          <tr className="border-b border-border">
            <th className="px-4 py-3 font-semibold">
              Class
            </th>

            <th className="px-4 py-3 font-semibold">
              Precision
            </th>

            <th className="px-4 py-3 font-semibold">
              Recall
            </th>

            <th className="px-4 py-3 font-semibold">
              F1 Score
            </th>

            <th className="px-4 py-3 font-semibold">
              Support
            </th>
          </tr>
        </thead>

        <tbody>
          {Object.entries(
            result.metrics["classification_report"]
          )
            .filter(
              ([key]) =>
                ![
                  "accuracy",
                  "macro avg",
                  "weighted avg",
                ].includes(key)
            )
            .map(([label, metrics]: [string, any]) => (
              <tr
                key={label}
                className="border-b border-border"
              >
                <td className="px-4 py-3 font-medium text-foreground">
                  {label}
                </td>

                <td className="px-4 py-3">
                  {Number(metrics.precision).toFixed(3)}
                </td>

                <td className="px-4 py-3">
                  {Number(metrics.recall).toFixed(3)}
                </td>

                <td className="px-4 py-3">
                  {Number(metrics["f1-score"]).toFixed(3)}
                </td>

                <td className="px-4 py-3">
                  {Number(metrics.support).toLocaleString()}
                </td>
              </tr>
            ))}
        </tbody>
      </table>
    </div>

    <div className="mt-6 grid gap-4 sm:grid-cols-3">
      <div className="rounded-xl border border-border bg-muted/40 p-4">
        <p className="text-xs uppercase tracking-wide text-muted-foreground">
          Accuracy
        </p>

        <p className="mt-2 text-xl font-semibold text-primary">
          {Number(
            result.metrics["classification_report"]["accuracy"]
          ).toFixed(3)}
        </p>
      </div>

      <div className="rounded-xl border border-border bg-muted/40 p-4">
        <p className="text-xs uppercase tracking-wide text-muted-foreground">
          Macro F1
        </p>

        <p className="mt-2 text-xl font-semibold text-primary">
          {Number(
            result.metrics["classification_report"]["macro avg"][
              "f1-score"
            ]
          ).toFixed(3)}
        </p>
      </div>

      <div className="rounded-xl border border-border bg-muted/40 p-4">
        <p className="text-xs uppercase tracking-wide text-muted-foreground">
          Weighted F1
        </p>

        <p className="mt-2 text-xl font-semibold text-primary">
          {Number(
            result.metrics["classification_report"]["weighted avg"][
              "f1-score"
            ]
          ).toFixed(3)}
        </p>
      </div>
    </div>
  </div>
)}
{/* CONFUSION MATRIX */}

{result.metrics["confusion_matrix"] && (
  <div className="panel mt-6 p-6">
    <h3 className="text-lg font-semibold text-foreground">
      Confusion Matrix
    </h3>

    <p className="mt-1 text-sm text-muted-foreground">
      Rows represent actual classes and columns represent predicted classes.
    </p>

    <div className="mt-5 overflow-x-auto">
      <table className="border-collapse">
        <thead>
          <tr>
            <th className="border border-border px-5 py-3 text-sm font-semibold">
              Actual ↓ / Predicted →
            </th>

            {result.metrics["labels"]?.map(
              (label: string | number) => (
                <th
                  key={String(label)}
                  className="border border-border px-6 py-3 text-sm font-semibold"
                >
                  {String(label)}
                </th>
              )
            )}
          </tr>
        </thead>

        <tbody>
          {result.metrics["confusion_matrix"].map(
            (row: number[], rowIndex: number) => (
              <tr key={rowIndex}>
                <th className="border border-border px-6 py-5 text-sm font-semibold">
                  {String(
                    result.metrics["labels"]?.[rowIndex] ??
                      rowIndex
                  )}
                </th>

                {row.map(
                  (value: number, columnIndex: number) => (
                    <td
                      key={columnIndex}
                      className="border border-border px-8 py-5 text-center font-semibold"
                    >
                      {value.toLocaleString()}
                    </td>
                  )
                )}
              </tr>
            )
          )}
        </tbody>
      </table>
    </div>
  </div>
)}

{/* ================================================== */}
{/* CONTINUE TO BIAS DETECTION                        */}
{/* ================================================== */}

<div className="mt-8 flex justify-end">
  <button
    type="button"
    onClick={() => {
      window.location.href = "/bias";
    }}
    className="rounded-xl bg-primary px-7 py-3 text-sm font-semibold text-primary-foreground"
  >
    Detect Bias →
  </button>
</div>
        </section>
      )}

    </main>
  );
}

function ResultCard({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <article className="panel p-5">
      <p className="text-xs uppercase tracking-[0.14em] text-muted-foreground">
        {label}
      </p>

      <p className="mt-2 text-2xl font-semibold text-primary">
        {value}
      </p>
    </article>
  );
}

function formatMetricName(name: string) {
  return name
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) =>
      letter.toUpperCase()
    );
}