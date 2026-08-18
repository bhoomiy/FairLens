import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";

export const Route = createFileRoute("/mitigation")({
  component: MitigationPage,
});

const API_URL = "http://127.0.0.1:5000";
const sensitiveFeatures =
  JSON.parse(
    localStorage.getItem("fairlens_sensitive_features") || "[]"
  );


function MitigationPage() {
  const [selectedTechnique, setSelectedTechnique] =
    useState("");
  const [selectedFeature, setSelectedFeature] =useState("");
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");
  const [downloading, setDownloading] = useState("");
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState("");

  const sessionId = localStorage.getItem(
    "fairlens_session_id"
  );

  const runMitigation = async () => {
    if (!sessionId) {
      setError(
        "Dataset session not found. Please complete preprocessing first."
      );
      return;
    }

    if (!selectedTechnique) {
      setError("Please select a mitigation technique.");
      return;
    }

      if (!selectedFeature) {
    setError("Please select a sensitive feature.");
    return;
  }

    setLoading(true);
    setError("");
    setMessage("");

    try {
    const response = await fetch(
      `${API_URL}/api/mitigation`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          session_id: sessionId,
          technique: selectedTechnique,
          feature: selectedFeature,
        }),
      }
    );

      
      const data = await response.json();

      if (!response.ok || !data.ok) {
        throw new Error(
          data.error || "Mitigation failed."
        );
      }

      setResult(data);

        setMessage(
        `${data.technique} applied successfully to ${data.feature}.`
        );


    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Something went wrong."
      );
    } finally {
      setLoading(false);
    }
  };

  const downloadFile = async (
  type: "original-model" | "mitigated-model" | "report"
) => {
  if (!sessionId) {
    setError(
      "Dataset session not found. Please complete the workflow first."
    );
    return;
  }

  setDownloading(type);
  setError("");

  try {
    let endpoint = "";

    if (type === "original-model") {
      endpoint = `${API_URL}/api/download/original-model/${sessionId}`;
    }

    if (type === "mitigated-model") {
      endpoint = `${API_URL}/api/download/mitigated-model/${sessionId}`;
    }

    if (type === "report") {
      endpoint = `${API_URL}/api/report/${sessionId}`;
    }

    const response = await fetch(endpoint);

    if (!response.ok) {
      let errorMessage = "Download failed.";

      try {
        const data = await response.json();
        errorMessage = data.error || errorMessage;
      } catch {
        // Response was not JSON
      }

      throw new Error(errorMessage);
    }

    const blob = await response.blob();

    const url = window.URL.createObjectURL(blob);

    const link = document.createElement("a");
    link.href = url;

    if (type === "original-model") {
      link.download = "FairLens_original_model.pkl";
    } else if (type === "mitigated-model") {
      link.download = "FairLens_mitigated_model.pkl";
    } else {
      link.download = "FairLens_bias_audit_report.pdf";
    }

    document.body.appendChild(link);
    link.click();

    link.remove();
    window.URL.revokeObjectURL(url);

  } catch (err) {
    setError(
      err instanceof Error
        ? err.message
        : "Unable to download file."
    );
  } finally {
    setDownloading("");
  }
};

  return (
    <main className="mx-auto max-w-6xl px-6 py-16">

      <section className="aurora panel px-8 py-10">
        <p className="text-xs font-semibold uppercase tracking-[0.22em] text-primary">
          Step 04
        </p>

        <h1 className="mt-3 text-4xl font-semibold text-foreground">
          Bias Mitigation
        </h1>

        <p className="mt-4 max-w-3xl text-muted-foreground">
          Select a mitigation technique to reduce
          unfairness detected in your trained model.
        </p>
      </section>

      <section className="panel mt-10 p-6">

        <h2 className="text-xl font-semibold text-foreground">
          Select Mitigation Technique
        </h2>

        <div className="mt-6">

  <label className="text-sm font-semibold text-foreground">
    Select Sensitive Feature
  </label>

  <select
    value={selectedFeature}
    onChange={(e) =>
      setSelectedFeature(e.target.value)
    }
    className="mt-2 w-full rounded-xl border border-border bg-background px-4 py-3 text-sm text-foreground"
  >
    <option value="">
      Select a sensitive feature
    </option>

    {sensitiveFeatures.map(
      (feature: string) => (
        <option
          key={feature}
          value={feature}
        >
          {feature}
        </option>
      )
    )}
  </select>

</div>

        <div className="mt-6 grid gap-4 md:grid-cols-3">

          <TechniqueCard
            title="Reweighing"
            description="Adjusts training sample weights to reduce group imbalance."
            selected={selectedTechnique === "Reweighing"}
            onClick={() =>
              setSelectedTechnique("Reweighing")
            }
          />

          <TechniqueCard
            title="Exponentiated Gradient"
            description="Uses fairness constraints during model training."
            selected={
              selectedTechnique ===
              "Exponentiated Gradient"
            }
            onClick={() =>
              setSelectedTechnique(
                "Exponentiated Gradient"
              )
            }
          />

          <TechniqueCard
            title="Threshold Optimizer"
            description="Adjusts classification thresholds to improve fairness."
            selected={
              selectedTechnique ===
              "Threshold Optimizer"
            }
            onClick={() =>
              setSelectedTechnique(
                "Threshold Optimizer"
              )
            }
          />

        </div>

        <button
          type="button"
          onClick={runMitigation}
          disabled={loading}
          className="mt-8 rounded-xl bg-primary px-8 py-3 text-sm font-semibold text-primary-foreground transition hover:opacity-90 disabled:opacity-50"
        >
          {loading
            ? "Running Mitigation..."
            : "Run Mitigation →"}
        </button>

        {message && (
          <div className="mt-6 rounded-xl border border-primary/30 bg-primary/10 p-5 text-sm text-primary">
            {message}
          </div>
        )}

        {error && (
          <div className="mt-6 rounded-xl border border-destructive/30 bg-destructive/10 p-5 text-sm text-destructive">
            {error}
          </div>
        )}


      </section>

      {result && (
  <section className="mt-8">

    <div className="rounded-xl border border-primary/30 bg-primary/10 p-5">
      <p className="text-xs font-semibold uppercase tracking-[0.14em] text-primary">
        Mitigation Result
      </p>

      <h3 className="mt-2 text-2xl font-semibold text-foreground">
        {result.technique}
      </h3>

      <p className="mt-2 text-sm text-muted-foreground">
        Sensitive Feature:{" "}
        <span className="font-semibold text-foreground">
          {result.feature}
        </span>
      </p>
    </div>

    <div className="mt-6 grid gap-6 md:grid-cols-2">

      {/* BEFORE */}
      <div className="rounded-xl border border-border bg-muted/40 p-6">
        <p className="text-xs font-semibold uppercase tracking-[0.14em] text-muted-foreground">
          Before Mitigation
        </p>

        {Object.entries(
  result.before_fairness?.[result.feature]?.metrics || {}
).map(([metric, value]: [string, any]) => (
  <div
    key={metric}
    className="mt-4 rounded-lg border border-border bg-background p-4"
  >
    <p className="text-xs font-semibold text-muted-foreground">
      {metric}
    </p>

    {typeof value === "object" && value !== null ? (
      <div className="mt-3 space-y-2">
        {Object.entries(value).map(
          ([group, accuracy]: [string, any]) => (
            <div
              key={group}
              className="flex items-center justify-between rounded-lg bg-muted/40 px-3 py-2"
            >
              <span className="text-sm text-foreground">
                {group}
              </span>

              <span className="text-sm font-semibold text-primary">
                {typeof accuracy === "number"
                  ? accuracy.toFixed(4)
                  : String(accuracy)}
              </span>
            </div>
          )
        )}
      </div>
    ) : (
      <p className="mt-1 text-xl font-semibold text-primary">
        {formatMetric(value)}
      </p>
    )}
  </div>
))}
      </div>

      {/* AFTER */}
      <div className="rounded-xl border border-primary/30 bg-primary/10 p-6">
        <p className="text-xs font-semibold uppercase tracking-[0.14em] text-primary">
          After Mitigation
        </p>

        {Object.entries(
  result.after_fairness?.[result.feature]?.metrics || {}
).map(([metric, value]: [string, any]) => (
  <div
    key={metric}
    className="mt-4 rounded-lg border border-border bg-background p-4"
  >
    <p className="text-xs font-semibold text-muted-foreground">
      {metric}
    </p>

    {typeof value === "object" && value !== null ? (
      <div className="mt-3 space-y-2">
        {Object.entries(value).map(
          ([group, accuracy]: [string, any]) => (
            <div
              key={group}
              className="flex items-center justify-between rounded-lg bg-muted/40 px-3 py-2"
            >
              <span className="text-sm text-foreground">
                {group}
              </span>

              <span className="text-sm font-semibold text-primary">
                {typeof accuracy === "number"
                  ? accuracy.toFixed(4)
                  : String(accuracy)}
              </span>
            </div>
          )
        )}
      </div>
    ) : (
      <p className="mt-1 text-xl font-semibold text-primary">
        {formatMetric(value)}
      </p>
    )}
  </div>
))}
      </div>

    </div>

  </section>
)}

{result && (
  <section className="mt-10 panel p-6">

    <p className="text-xs font-semibold uppercase tracking-[0.14em] text-primary">
      Step 05
    </p>

    <h2 className="mt-2 text-2xl font-semibold text-foreground">
      Download Results
    </h2>

    <p className="mt-2 text-sm text-muted-foreground">
      Download the original model, the mitigated model, or the
      complete FairLens bias audit report.
    </p>

    <div className="mt-6 grid gap-4 md:grid-cols-3">

      {/* ORIGINAL MODEL */}

      <button
        type="button"
        onClick={() =>
          downloadFile("original-model")
        }
        disabled={downloading !== ""}
        className="rounded-xl border border-border bg-muted/40 p-5 text-left transition hover:border-primary/50 disabled:opacity-50"
      >
        <h3 className="text-lg font-semibold text-foreground">
          Original Model
        </h3>

        <p className="mt-2 text-sm text-muted-foreground">
          Download the model before fairness mitigation.
        </p>

        <div className="mt-4 text-sm font-semibold text-primary">
          {downloading === "original-model"
            ? "Downloading..."
            : "Download Model →"}
        </div>
      </button>


      {/* MITIGATED MODEL */}

      <button
        type="button"
        onClick={() =>
          downloadFile("mitigated-model")
        }
        disabled={downloading !== ""}
        className="rounded-xl border border-primary/30 bg-primary/10 p-5 text-left transition hover:border-primary disabled:opacity-50"
      >
        <h3 className="text-lg font-semibold text-foreground">
          Mitigated Model
        </h3>

        <p className="mt-2 text-sm text-muted-foreground">
          Download the model after applying{" "}
          {result.technique}.
        </p>

        <div className="mt-4 text-sm font-semibold text-primary">
          {downloading === "mitigated-model"
            ? "Downloading..."
            : "Download Model →"}
        </div>
      </button>


      {/* REPORT */}

      <button
        type="button"
        onClick={() =>
          downloadFile("report")
        }
        disabled={downloading !== ""}
        className="rounded-xl border border-border bg-muted/40 p-5 text-left transition hover:border-primary/50 disabled:opacity-50"
      >
        <h3 className="text-lg font-semibold text-foreground">
          Bias Audit Report
        </h3>

        <p className="mt-2 text-sm text-muted-foreground">
          Download the complete FairLens fairness analysis as
          a PDF report.
        </p>

        <div className="mt-4 text-sm font-semibold text-primary">
          {downloading === "report"
            ? "Generating Report..."
            : "Download Report →"}
        </div>
      </button>

    </div>

  </section>
)}

    </main>
  );
}

function TechniqueCard({
  title,
  description,
  selected,
  onClick,
}: {
  title: string;
  description: string;
  selected: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`rounded-xl border p-5 text-left transition ${
        selected
          ? "border-primary bg-primary/10"
          : "border-border bg-muted/40 hover:border-primary/50"
      }`}
    >
      <h3 className="text-lg font-semibold text-primary">
        {title}
      </h3>

      <p className="mt-2 text-sm text-muted-foreground">
        {description}
      </p>
    </button>
  );
}

function formatMetric(value: any) {
  // Normal metric
  if (typeof value === "number") {
    return value.toFixed(4);
  }

  // Group Accuracy
  if (
    value &&
    typeof value === "object" &&
    !Array.isArray(value)
  ) {
    return (
      <div className="mt-2 space-y-2">
        {Object.entries(value)
          .filter(([group]) => group !== "Missing")
          .map(([group, accuracy]) => (
            <div
              key={group}
              className="flex items-center justify-between rounded-lg border border-border bg-muted/40 px-4 py-3"
            >
              <span className="font-medium text-foreground">
                {group}
              </span>

              <span className="font-semibold text-primary">
                {typeof accuracy === "number"
                  ? accuracy.toFixed(4)
                  : String(accuracy)}
              </span>
            </div>
          ))}
      </div>
    );
  }

  return String(value ?? "N/A");
}


function renderMetricValue(metric: string, value: any) {
  // Group Accuracy is an object like:
  // { Female: 0.9259, Male: 0.8907 }

  if (
    metric.toLowerCase() === "group accuracy" &&
    value &&
    typeof value === "object"
  ) {
    return (
      <div className="mt-3 space-y-2">
        {Object.entries(value).map(
          ([group, accuracy]) => (
            <div
              key={group}
              className="flex items-center justify-between rounded-lg border border-border bg-muted/30 px-4 py-3"
            >
              <span className="text-sm font-medium text-foreground">
                {group}
              </span>

              <span className="text-base font-semibold text-primary">
                {typeof accuracy === "number"
                  ? accuracy.toFixed(4)
                  : String(accuracy)}
              </span>
            </div>
          )
        )}
      </div>
    );
  }

  return (
    <p className="mt-1 text-xl font-semibold text-primary">
      {formatMetric(value)}
    </p>
  );
}