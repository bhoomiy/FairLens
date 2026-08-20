import { useEffect, useState } from "react";
import { createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/upload")({
  component: UploadPage,
});

const API_URL = "https://fairlens-amnt.onrender.com/";

type UploadResult = {
  session_id: string;
  dataset_name: string;
  column_names: string[];
  rows: number;
  columns: number;
  duplicates: number;
  dtypes: Record<string, string>;
  missing: {
    column: string;
    missing: number;
  }[];
  summary: {
    Rows: number;
    Columns: number;
    "Missing Values": number;
    "Duplicate Rows": number;
    "Categorical Columns": string[];
    "Numerical Columns": string[];
  };
  preview: Record<string, unknown>[];
};

type SchemaResult = {
  quality_score: number;
  completeness: number;
  missing_cells: number;
  duplicates: number;
  memory_kb: number;
  schema: {
    column: string;
    dtype: string;
    missing: number;
    unique: number;
  }[];
  missing_by_column: Record<string, number>;
  numeric_stats: {
    column: string;
    count: number;
    mean: number;
    std: number;
    min: number;
    max: number;
  }[];
  categorical_stats: {
    column: string;
    unique: number;
    missing: number;
    top: string;
  }[];
  issues: {
    column: string;
    severity: string;
    message: string;
  }[];
};

type PreprocessResult = {
  rows: number;
  duplicates_removed: number;
  features: number;
  train_samples: number;
  test_samples: number;
  task_type: string;
  detection_reason: string;

  missing_before: {
    column: string;
    missing: number;
  }[];

  missing_after: {
    column: string;
    missing: number;
  }[];

  target_distribution?: Record<string, number>;

  target_statistics?: {
    min: number;
    max: number;
    mean: number;
    median: number;
    std: number;
  };
};

function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [profiling, setProfiling] = useState(false);
  const [preprocessing, setPreprocessing] = useState(false);

  const [result, setResult] = useState<UploadResult | null>(null);
  const [schema, setSchema] = useState<SchemaResult | null>(null);
  const [preprocessResult, setPreprocessResult] =
    useState<PreprocessResult | null>(null);

  const [target, setTarget] = useState("");
  const [sensitive, setSensitive] = useState<string[]>([]);

  const [testSize, setTestSize] = useState(0.2);
  const [missingStrategy, setMissingStrategy] = useState("Mean");
  const [encodingMethod, setEncodingMethod] =
    useState("Label Encoding");
  const [scalingMethod, setScalingMethod] =
    useState("StandardScaler");

  const [error, setError] = useState("");

  /*
   * ------------------------------------------------------------
   * Upload
   * ------------------------------------------------------------
   */

  const handleUpload = async () => {
    if (!file) {
      setError("Please select a CSV or Excel file first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);
    setSchema(null);
    setPreprocessResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_URL}/api/upload`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok || !data.ok) {
        throw new Error(data.error || "Upload failed.");
      }

      localStorage.setItem(
        "fairlens_session_id",
        data.session_id
      );

      setResult(data);

      /*
       * Automatically load schema after upload.
       */
      await loadSchema(data.session_id);

    } catch (err) {
      console.error(err);

      setError(
        err instanceof Error
          ? err.message
          : "Something went wrong while uploading."
      );
    } finally {
      setLoading(false);
    }
  };

  /*
   * ------------------------------------------------------------
   * Schema
   * ------------------------------------------------------------
   */

  const loadSchema = async (sessionId: string) => {
    setProfiling(true);

    try {
      const response = await fetch(
        `${API_URL}/api/schema/${sessionId}`
      );

      const data = await response.json();

      if (!response.ok || !data.ok) {
        throw new Error(
          data.error || "Unable to load dataset schema."
        );
      }

      setSchema(data);

    } catch (err) {
      console.error(err);

      setError(
        err instanceof Error
          ? err.message
          : "Unable to profile dataset."
      );
    } finally {
      setProfiling(false);
    }
  };

  /*
   * ------------------------------------------------------------
   * Sensitive feature selection
   * ------------------------------------------------------------
   */

  const toggleSensitive = (column: string) => {
    setSensitive((current) => {
      if (current.includes(column)) {
        return current.filter((item) => item !== column);
      }

      return [...current, column];
    });
  };

  /*
   * ------------------------------------------------------------
   * Run preprocessing
   * ------------------------------------------------------------
   */

const handlePreprocess = async () => {
  if (!result) return;

  if (!target) {
    setError("Please select a target column.");
    return;
  }

  if (sensitive.length === 0) {
    setError(
      "Select at least one sensitive feature — the fairness engine needs it."
    );
    return;
  }

  setPreprocessing(true);
  setError("");
  setPreprocessResult(null);

  try {
    const response = await fetch(`${API_URL}/api/preprocess`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        session_id: result.session_id,
        target,
        sensitive,
        missing_strategy: missingStrategy,
        encoding_method: encodingMethod,
        scaling_method: scalingMethod,
        test_size: testSize,
      }),
    });

    /*
     * Safely read the response.
     * This prevents React from crashing if Flask returns
     * an HTML error page instead of JSON.
     */
    const text = await response.text();

    let data: any;

    try {
      data = JSON.parse(text);
    } catch {
      console.error("Backend returned non-JSON response:", text);

      throw new Error(
        `Backend returned an invalid response (HTTP ${response.status}). Check the Flask terminal.`
      );
    }

    console.log("PREPROCESS RESPONSE:", data);

    if (!response.ok || !data.ok) {
      throw new Error(
        data.error || "Preprocessing failed."
      );
    }

    /*
     * Make sure the required fields exist before
     * giving the result to React.
     */
    if (!data.task_type) {
      throw new Error(
        "Preprocessing succeeded but the backend did not return task_type."
      );
    }

    setPreprocessResult(data);

    localStorage.setItem(
      "fairlens_task_type",
      String(data.task_type).toLowerCase()
    );

  } catch (err) {
    console.error("PREPROCESS ERROR:", err);

    setError(
      err instanceof Error
        ? err.message
        : "Something went wrong during preprocessing."
    );
  } finally {
    setPreprocessing(false);
  }
};

  /*
   * ------------------------------------------------------------
   * Render
   * ------------------------------------------------------------
   */

  return (
<main className="mx-auto w-full max-w-7xl px-6 py-16">
      {/* ====================================================== */}
      {/* HEADER                                                  */}
      {/* ====================================================== */}

      <section className="aurora panel px-8 py-10">
        <p className="text-xs font-semibold uppercase tracking-[0.22em] text-primary">
          Step 01
        </p>

        <h1 className="mt-3 text-4xl font-semibold text-foreground">
          Upload & Preprocess
        </h1>

        <p className="mt-4 max-w-3xl text-muted-foreground">
          Bring in your dataset, inspect its quality, declare the
          target and sensitive attributes, then clean, encode,
          scale and split the data for the FairLens pipeline.
        </p>
      </section>

      {/* ====================================================== */}
      {/* UPLOAD                                                  */}
      {/* ====================================================== */}

      <section className="panel mt-10 p-6">
        <SectionTitle
          title="Dataset"
          subtitle="Upload CSV or Excel"
        />

        <input
          type="file"
          accept=".csv,.xlsx,.xls"
          className="mt-5 block w-full rounded-xl border border-border bg-background p-3 text-sm text-foreground"
          onChange={(event) => {
            setFile(event.target.files?.[0] || null);
            setError("");
            setResult(null);
            setSchema(null);
            setPreprocessResult(null);
          }}
        />

        {file && (
          <p className="mt-3 text-sm text-muted-foreground">
            Selected: <span className="text-foreground">{file.name}</span>
          </p>
        )}

        <button
          type="button"
          onClick={handleUpload}
          disabled={loading}
          className="mt-5 rounded-xl bg-primary px-6 py-3 text-sm font-semibold text-primary-foreground disabled:opacity-50"
        >
          {loading ? "Uploading..." : "Upload Dataset"}
        </button>
      </section>

      {/* ====================================================== */}
      {/* ERROR                                                   */}
      {/* ====================================================== */}

      {error && (
        <div className="mt-6 rounded-xl border border-destructive/30 bg-destructive/10 p-4 text-sm text-destructive">
          {error}
        </div>
      )}

      {/* ====================================================== */}
      {/* UPLOAD RESULT                                           */}
      {/* ====================================================== */}

      {result && (
        <>
          {/* SUMMARY */}

          <section className="mt-10">
            <SectionTitle
              title="Dataset Summary"
              subtitle="Initial dataset profile"
            />

            <div className="mt-5 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              <SummaryCard
                label="Rows"
                value={result.rows}
              />

              <SummaryCard
                label="Columns"
                value={result.columns}
              />

              <SummaryCard
                label="Missing Values"
                value={result.summary["Missing Values"]}
              />

              <SummaryCard
                label="Duplicate Rows"
                value={result.duplicates}
              />
            </div>
          </section>

          {/* PREVIEW */}

          <section className="panel mt-10 p-6">
            <SectionTitle
              title="Dataset Preview"
              subtitle="First rows of the uploaded dataset"
            />

            <DataTable
              columns={result.column_names}
              rows={result.preview}
            />
          </section>

          {/* ================================================== */}
          {/* SCHEMA & QUALITY                                   */}
          {/* ================================================== */}

          <section className="mt-14">
            <SectionTitle
              title="Schema & Data Quality"
              subtitle="Inspect your dataset before preprocessing"
            />

            {profiling && (
              <div className="panel mt-5 p-6 text-sm text-muted-foreground">
                Profiling columns, types and data quality...
              </div>
            )}

            {schema && (
              <>
                {/* QUALITY METRICS */}

                <div className="mt-5 grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
                  <SummaryCard
                    label="Quality Score"
                    value={schema.quality_score}
                  />

                  <SummaryCard
                    label="Completeness %"
                    value={schema.completeness}
                  />

                  <SummaryCard
                    label="Missing Cells"
                    value={schema.missing_cells}
                  />

                  <SummaryCard
                    label="Duplicates"
                    value={schema.duplicates}
                  />

                  <SummaryCard
                    label="Memory KB"
                    value={schema.memory_kb}
                  />
                </div>

                {/* SCHEMA */}

                <div className="panel mt-6 p-6">
                  <h3 className="text-lg font-semibold text-foreground">
                    Schema
                  </h3>

                  <div className="mt-4 overflow-x-auto">
                    <table className="w-full text-left text-sm">
                      <thead>
                        <tr className="border-b border-border">
                          <th className="px-4 py-3">
                            Column
                          </th>
                          <th className="px-4 py-3">
                            Data Type
                          </th>
                          <th className="px-4 py-3">
                            Missing
                          </th>
                          <th className="px-4 py-3">
                            Unique
                          </th>
                        </tr>
                      </thead>

                      <tbody>
                        {schema.schema.map((item) => (
                          <tr
                            key={item.column}
                            className="border-b border-border"
                          >
                            <td className="px-4 py-3 font-medium text-foreground">
                              {item.column}
                            </td>

                            <td className="px-4 py-3 text-muted-foreground">
                              {item.dtype}
                            </td>

                            <td className="px-4 py-3 text-muted-foreground">
                              {item.missing}
                            </td>

                            <td className="px-4 py-3 text-muted-foreground">
                              {item.unique}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* MISSING VALUES */}

                <div className="panel mt-6 p-6">
                  <h3 className="text-lg font-semibold text-foreground">
                    Missing Values
                  </h3>

                  <div className="mt-4 overflow-x-auto">
                    <table className="w-full text-left text-sm">
                      <thead>
                        <tr className="border-b border-border">
                          <th className="px-4 py-3">
                            Column
                          </th>
                          <th className="px-4 py-3">
                            Missing
                          </th>
                        </tr>
                      </thead>

                      <tbody>
                        {Object.entries(
                          schema.missing_by_column
                        ).map(([column, count]) => (
                          <tr
                            key={column}
                            className="border-b border-border"
                          >
                            <td className="px-4 py-3 text-foreground">
                              {column}
                            </td>

                            <td className="px-4 py-3 text-muted-foreground">
                              {count}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* NUMERIC STATS */}

                <div className="panel mt-6 p-6">
                  <h3 className="text-lg font-semibold text-foreground">
                    Numeric Statistics
                  </h3>

                  {schema.numeric_stats.length > 0 ? (
                    <div className="mt-4 overflow-x-auto">
                      <table className="w-full text-left text-sm">
                        <thead>
                          <tr className="border-b border-border">
                            <th className="px-4 py-3">Column</th>
                            <th className="px-4 py-3">Mean</th>
                            <th className="px-4 py-3">Std</th>
                            <th className="px-4 py-3">Min</th>
                            <th className="px-4 py-3">Max</th>
                          </tr>
                        </thead>

                        <tbody>
                          {schema.numeric_stats.map((item) => (
                            <tr
                              key={item.column}
                              className="border-b border-border"
                            >
                              <td className="px-4 py-3 text-foreground">
                                {item.column}
                              </td>
                              <td className="px-4 py-3">
                                {item.mean.toFixed(3)}
                              </td>
                              <td className="px-4 py-3">
                                {item.std.toFixed(3)}
                              </td>
                              <td className="px-4 py-3">
                                {item.min.toFixed(3)}
                              </td>
                              <td className="px-4 py-3">
                                {item.max.toFixed(3)}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <p className="mt-4 text-sm text-muted-foreground">
                      No numeric columns found.
                    </p>
                  )}
                </div>

                {/* CATEGORICAL STATS */}

                <div className="panel mt-6 p-6">
                  <h3 className="text-lg font-semibold text-foreground">
                    Categorical Statistics
                  </h3>

                  {schema.categorical_stats.length > 0 ? (
                    <div className="mt-4 overflow-x-auto">
                      <table className="w-full text-left text-sm">
                        <thead>
                          <tr className="border-b border-border">
                            <th className="px-4 py-3">Column</th>
                            <th className="px-4 py-3">Unique</th>
                            <th className="px-4 py-3">Missing</th>
                            <th className="px-4 py-3">Top Value</th>
                          </tr>
                        </thead>

                        <tbody>
                          {schema.categorical_stats.map(
                            (item) => (
                              <tr
                                key={item.column}
                                className="border-b border-border"
                              >
                                <td className="px-4 py-3">
                                  {item.column}
                                </td>

                                <td className="px-4 py-3">
                                  {item.unique}
                                </td>

                                <td className="px-4 py-3">
                                  {item.missing}
                                </td>

                                <td className="px-4 py-3">
                                  {item.top}
                                </td>
                              </tr>
                            )
                          )}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <p className="mt-4 text-sm text-muted-foreground">
                      No categorical columns found.
                    </p>
                  )}
                </div>

                {/* QUALITY ISSUES */}

                <div className="panel mt-6 p-6">
                  <h3 className="text-lg font-semibold text-foreground">
                    Quality Issues
                  </h3>

                  {schema.issues.length > 0 ? (
                    <div className="mt-4 space-y-3">
                      {schema.issues.map((issue, index) => (
                        <div
                          key={index}
                          className="rounded-xl border border-border bg-muted/40 p-4"
                        >
                          <span className="font-medium text-foreground">
                            {issue.column}
                          </span>

                          <span className="ml-2 text-sm text-muted-foreground">
                            — {issue.message}
                          </span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="mt-4 text-sm text-muted-foreground">
                      No data-quality issues detected.
                    </p>
                  )}
                </div>
              </>
            )}
          </section>

          {/* ================================================== */}
          {/* AUDIT CONFIGURATION                                */}
          {/* ================================================== */}

          <section className="mt-14">
            <SectionTitle
              title="Audit Configuration"
              subtitle="Target, sensitive attributes and preprocessing strategy"
            />

            <div className="panel mt-5 p-6">
              <div className="grid gap-8 lg:grid-cols-2">

                {/* LEFT */}

                <div className="space-y-6">

                  <div>
                    <label className="block text-sm font-medium text-foreground">
                      Target Column
                    </label>

                    <select
                      value={target}
                      onChange={(e) => {
                        setTarget(e.target.value);

                        setSensitive((current) =>
                          current.filter(
                            (column) => column !== e.target.value
                          )
                        );
                      }}
                      className="mt-2 w-full rounded-xl border border-border bg-background p-3 text-sm text-foreground"
                    >
                      <option value="">
                        Select target column
                      </option>

                      {result.column_names.map((column) => (
                        <option
                          key={column}
                          value={column}
                        >
                          {column}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-foreground">
                      Sensitive Features
                    </label>

                    <p className="mt-1 text-xs text-muted-foreground">
                      Select attributes such as gender, race or age
                      group that the fairness engine should audit.
                    </p>

                    <div className="mt-3 space-y-2 rounded-xl border border-border p-4">
                      {result.column_names
                        .filter((column) => column !== target)
                        .map((column) => (
                          <label
                            key={column}
                            className="flex cursor-pointer items-center gap-3 text-sm"
                          >
                            <input
                              type="checkbox"
                              checked={sensitive.includes(column)}
                              onChange={() =>
                                toggleSensitive(column)
                              }
                              className="h-4 w-4"
                            />

                            <span className="text-foreground">
                              {column}
                            </span>
                          </label>
                        ))}
                    </div>

                    {sensitive.length > 0 && (
                      <p className="mt-2 text-xs text-primary">
                        Selected: {sensitive.join(", ")}
                      </p>
                    )}
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-foreground">
                      Test Size
                    </label>

                    <div className="mt-3 flex items-center gap-4">
                      <input
                        type="range"
                        min="0.1"
                        max="0.4"
                        step="0.05"
                        value={testSize}
                        onChange={(e) =>
                          setTestSize(
                            Number(e.target.value)
                          )
                        }
                        className="w-full"
                      />

                      <span className="min-w-16 text-right text-sm font-semibold text-primary">
                        {Math.round(testSize * 100)}%
                      </span>
                    </div>
                  </div>

                </div>

                {/* RIGHT */}

                <div className="space-y-6">

                  <SelectField
                    label="Missing Value Strategy"
                    value={missingStrategy}
                    onChange={setMissingStrategy}
                    options={[
                      "Mean",
                      "Median",
                      "Mode",
                      "Drop Rows",
                    ]}
                  />

                  <SelectField
                    label="Categorical Encoding"
                    value={encodingMethod}
                    onChange={setEncodingMethod}
                    options={[
                      "Label Encoding",
                      "One-Hot Encoding",
                    ]}
                  />

                  <SelectField
                    label="Feature Scaling"
                    value={scalingMethod}
                    onChange={setScalingMethod}
                    options={[
                      "None",
                      "StandardScaler",
                      "MinMaxScaler",
                    ]}
                  />

                </div>
              </div>

              {/* RUN */}

              <div className="mt-8 border-t border-border pt-6">
                <button
                  type="button"
                  onClick={handlePreprocess}
                  disabled={
                    preprocessing ||
                    !target ||
                    sensitive.length === 0
                  }
                  className="rounded-xl bg-primary px-7 py-3 text-sm font-semibold text-primary-foreground disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {preprocessing
                    ? "Preprocessing..."
                    : "Run Preprocessing"}
                </button>

                {!target && (
                  <p className="mt-3 text-xs text-muted-foreground">
                    Select a target column to continue.
                  </p>
                )}

                {target && sensitive.length === 0 && (
                  <p className="mt-3 text-xs text-muted-foreground">
                    Select at least one sensitive feature.
                  </p>
                )}
              </div>
            </div>
          </section>

          {/* ================================================== */}
          {/* PREPROCESSING RESULT                               */}
          {/* ================================================== */}

          {preprocessResult && (
            <section className="mt-14">
              <SectionTitle
                title="Preprocessing Summary"
                subtitle="Your dataset is ready for model training"
              />

              {/* TASK */}

              <div className="panel mt-5 p-6">
                <div className="flex flex-wrap items-center gap-3">
                  <span className="text-sm text-muted-foreground">
                    Detected task:
                  </span>

                  <span className="rounded-lg bg-primary/10 px-3 py-1.5 text-sm font-semibold text-primary">
                    {preprocessResult.task_type}
                  </span>

                  <span className="text-sm text-muted-foreground">
                    — {preprocessResult.detection_reason}
                  </span>
                </div>
              </div>

              {/* METRICS */}

              <div className="mt-5 grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
                <SummaryCard
                  label="Rows Kept"
                  value={preprocessResult.rows}
                />

                <SummaryCard
                  label="Duplicates Removed"
                  value={preprocessResult.duplicates_removed}
                />

                <SummaryCard
                  label="Features"
                  value={preprocessResult.features}
                />

                <SummaryCard
                  label="Train Samples"
                  value={preprocessResult.train_samples}
                />

                <SummaryCard
                  label="Test Samples"
                  value={preprocessResult.test_samples}
                />
              </div>

              {/* MISSING BEFORE / AFTER */}

              <div className="mt-6 grid gap-6 lg:grid-cols-2">

                <MissingTable
                  title="Missing Values Before"
                  data={preprocessResult.missing_before}
                />

                <MissingTable
                  title="Missing Values After"
                  data={preprocessResult.missing_after}
                />

              </div>

              {/* ================================================== */}
              {/* TARGET INFORMATION                                 */}
              {/* ================================================== */}

              {preprocessResult.task_type.toLowerCase() === "classification" ? (

                /* ---------------- CLASSIFICATION ---------------- */

                <div className="panel mt-6 p-6">

                  <h3 className="text-lg font-semibold text-foreground">
                    Target Distribution
                  </h3>

                  <p className="mt-1 text-sm text-muted-foreground">
                    Distribution of samples across each target class.
                  </p>

                  <div className="mt-5 space-y-3">

                    {Object.entries(
                      preprocessResult.target_distribution ?? {}
                    ).map(([label, count]) => {

                      const total = Object.values(
                      preprocessResult.target_distribution ?? {}
                    ).reduce(
                        (sum, value) => sum + value,
                        0
                      );

                      const percentage =
                        total > 0
                          ? (count / total) * 100
                          : 0;

                      return (
                        <div key={label}>

                          <div className="mb-1 flex justify-between text-sm">

                            <span className="text-foreground">
                              {label}
                            </span>

                            <span className="text-muted-foreground">
                              {count} ({percentage.toFixed(1)}%)
                            </span>

                          </div>

                          <div className="h-2 overflow-hidden rounded-full bg-muted">

                            <div
                              className="h-full rounded-full bg-primary"
                              style={{
                                width: `${percentage}%`,
                              }}
                            />

                          </div>

                        </div>
                      );
                    })}

                  </div>

                </div>

              ) : (

                /* ---------------- REGRESSION ---------------- */

                <div className="panel mt-6 p-6">

  <h3 className="text-lg font-semibold text-foreground">
    Target Statistics
  </h3>

  <p className="mt-1 text-sm text-muted-foreground">
    Statistical summary of the continuous regression target.
  </p>

  {preprocessResult.target_statistics ? (

    <div className="mt-5 grid gap-4 sm:grid-cols-2 lg:grid-cols-5">

      <SummaryCard
        label="Minimum"
        value={preprocessResult.target_statistics.min}
      />

      <SummaryCard
        label="Maximum"
        value={preprocessResult.target_statistics.max}
      />

      <SummaryCard
        label="Mean"
        value={preprocessResult.target_statistics.mean}
      />

      <SummaryCard
        label="Median"
        value={preprocessResult.target_statistics.median}
      />

      <SummaryCard
        label="Std. Deviation"
        value={preprocessResult.target_statistics.std}
      />

    </div>

  ) : (

    <p className="mt-4 text-sm text-muted-foreground">
      Target statistics were not returned by the backend.
    </p>

  )}

</div>

              )}

              {/* COMPLETE */}

              <div className="mt-8 flex justify-end">
                <button
                  type="button"
                  onClick={() => {
                    window.location.href = "/train";
                  }}
                  className="rounded-xl bg-primary px-7 py-3 text-sm font-semibold text-primary-foreground"
                >
                  Continue to Train & Evaluate →
                </button>
              </div>

            </section>
          )}
        </>
      )}
    </main>
  );
}

/* ============================================================ */
/* COMPONENTS                                                    */
/* ============================================================ */

function SectionTitle({
  title,
  subtitle,
}: {
  title: string;
  subtitle: string;
}) {
  return (
    <div>
      <h2 className="text-xl font-semibold text-foreground">
        {title}
      </h2>

      <p className="mt-1 text-sm text-muted-foreground">
        {subtitle}
      </p>
    </div>
  );
}

function SummaryCard({
  label,
  value,
}: {
  label: string;
  value: number;
}) {
  return (
    <article className="panel p-5">
      <p className="text-xs uppercase tracking-[0.14em] text-muted-foreground">
        {label}
      </p>

      <p className="mt-2 text-2xl font-semibold text-primary">
          {typeof value === "number"
          ? value.toLocaleString(undefined, {
              maximumFractionDigits: 2,
            })
          : value}
      </p>
    </article>
  );
}

function SelectField({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  options: string[];
}) {
  return (
    <div>
      <label className="block text-sm font-medium text-foreground">
        {label}
      </label>

      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="mt-2 w-full rounded-xl border border-border bg-background p-3 text-sm text-foreground"
      >
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </div>
  );
}

function DataTable({
  columns,
  rows,
}: {
  columns: string[];
  rows: Record<string, unknown>[];
}) {
  return (
    <div className="mt-5 overflow-x-auto">
      <table className="w-full text-left text-sm">
        <thead>
          <tr className="border-b border-border">
            {columns.map((column) => (
              <th
                key={column}
                className="px-4 py-3 font-semibold text-foreground"
              >
                {column}
              </th>
            ))}
          </tr>
        </thead>

        <tbody>
          {rows.map((row, index) => (
            <tr
              key={index}
              className="border-b border-border"
            >
              {columns.map((column) => (
                <td
                  key={column}
                  className="px-4 py-3 text-muted-foreground"
                >
                  {String(row[column] ?? "")}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function MissingTable({
  title,
  data,
}: {
  title: string;
  data: {
    column: string;
    missing: number;
  }[];
}) {
  return (
    <div className="panel p-6">
      <h3 className="text-lg font-semibold text-foreground">
        {title}
      </h3>

      <div className="mt-4 overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead>
            <tr className="border-b border-border">
              <th className="px-4 py-3">Column</th>
              <th className="px-4 py-3">Missing</th>
            </tr>
          </thead>

          <tbody>
            {data.map((item) => (
              <tr
                key={item.column}
                className="border-b border-border"
              >
                <td className="px-4 py-3">
                  {item.column}
                </td>

                <td className="px-4 py-3">
                  {item.missing}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}