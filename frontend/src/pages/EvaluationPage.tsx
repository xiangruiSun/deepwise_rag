import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  LabelList,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { api } from "../api";
import type { ExperimentDetail, ExperimentRun } from "../types";

const fmt = (value: unknown, digits = 4) =>
  typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "—";

const METRICS = ["ndcg@10", "recall@10", "precision@10"] as const;

function recipeOf(run: ExperimentRun): Record<string, unknown> {
  return (run.summary?.recipe as Record<string, unknown>) ?? {};
}

function describe(run: ExperimentRun) {
  const recipe = recipeOf(run);
  const branches = (recipe.branches as string[] | undefined) ?? [];
  const reranker = recipe.reranker as { enabled?: boolean; key?: string } | undefined;
  const multimodal = recipe.multimodal as { enabled?: boolean; modalities?: string[] } | undefined;
  return {
    embedding: branches.includes("dense") ? String(recipe.embedding ?? "—") : "—",
    method: branches.length ? branches.join("+") : "structured",
    reranker: reranker?.enabled ? String(reranker.key) : "—",
    chunking: String(recipe.chunking ?? "—"),
    modalities: multimodal?.enabled ? (multimodal.modalities ?? []).join("+") : "text",
  };
}

function ResourceUsage({ summary }: { summary: Record<string, unknown> }) {
  const resources = (summary.resources as Record<string, number | null> | undefined) ?? {};
  const generation = (summary.generation as Record<string, number | null> | undefined) ?? null;
  const latency = (summary.latency_ms as Record<string, Record<string, number | null>> | undefined) ?? {};
  return (
    <div className="stat-row" style={{ margin: "6px 0 12px" }}>
      <span>peak process memory <b>{fmt(resources.process_peak_memory_mb, 0)}</b> MB</span>
      <span>peak GPU memory <b>{fmt(resources.gpu_peak_memory_mb, 0)}</b> MB</span>
      <span>wall time <b>{fmt(summary.wall_seconds, 2)}</b> s</span>
      <span>rerank p95 <b>{fmt(latency.reranking?.p95, 1)}</b> ms</span>
      <span>query-embedding p95 <b>{fmt(latency.query_embedding?.p95, 1)}</b> ms</span>
      {generation && (
        <>
          <span>mean input tokens <b>{fmt(generation.input_tokens, 0)}</b></span>
          <span>mean output tokens <b>{fmt(generation.output_tokens, 0)}</b></span>
          <span>generation latency <b>{fmt(generation.generation_latency_ms, 1)}</b> ms</span>
        </>
      )}
      <span className="muted">external API cost: none (local inference only)</span>
    </div>
  );
}

export default function EvaluationPage() {
  const [runs, setRuns] = useState<ExperimentRun[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [suite, setSuite] = useState<string>("");
  const [metric, setMetric] = useState<(typeof METRICS)[number]>("ndcg@10");
  const [chosen, setChosen] = useState<string[]>([]);
  const [detail, setDetail] = useState<ExperimentDetail | null>(null);

  useEffect(() => {
    api
      .experiments()
      .then((all) => {
        setRuns(all);
        setSuite((current) => current || all[0]?.suite_id || "");
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : "Could not load experiments"));
  }, []);

  const suites = useMemo(() => Array.from(new Set((runs ?? []).map((r) => r.suite_id ?? "—"))), [runs]);
  const inSuite = useMemo(
    () =>
      (runs ?? [])
        .filter((r) => (r.suite_id ?? "—") === suite)
        .sort((a, b) => a.experiment_id.localeCompare(b.experiment_id)),
    [runs, suite],
  );
  const compared = inSuite.filter((r) => (chosen.length ? chosen.includes(r.run_id) : true) && r.status === "completed");
  const barData = compared.map((r) => ({ name: r.experiment_id, value: r.metrics[metric] ?? null }));
  const scatterData = compared
    .filter((r) => r.metrics["recall@10"] != null && r.metrics["p95_latency_ms"] != null)
    .map((r) => ({ name: r.experiment_id, x: r.metrics["p95_latency_ms"] as number, y: r.metrics["recall@10"] as number }));

  if (error)
    return (
      <p className="error" role="alert">
        {error}
      </p>
    );
  if (!runs) return <p className="empty">Loading experiments…</p>;
  if (runs.length === 0)
    return (
      <div className="panel empty" data-testid="no-experiments">
        No experiment has been recorded yet. Run <code>make run-baseline</code> to execute the baseline evaluation;
        this dashboard only shows results of completed runs.
      </div>
    );

  const synthetic = inSuite.some((r) => r.data_kind === "synthetic");
  return (
    <div className="stack">
      <section className="panel">
        <div className="row" style={{ justifyContent: "space-between" }}>
          <h1>Evaluation dashboard</h1>
          {synthetic && <span className="chip synthetic">SYNTHETIC DATA — not clinical evidence</span>}
        </div>
        <div className="row" style={{ marginTop: 10 }}>
          <label className="field" style={{ maxWidth: 320 }}>
            <span>Experiment suite</span>
            <select
              value={suite}
              onChange={(e) => {
                setSuite(e.target.value);
                setChosen([]);
                setDetail(null);
              }}
              data-testid="suite-select"
            >
              {suites.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </label>
          <label className="field" style={{ maxWidth: 220 }}>
            <span>Chart metric</span>
            <select value={metric} onChange={(e) => setMetric(e.target.value as (typeof METRICS)[number])}>
              {METRICS.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </label>
        </div>
      </section>

      <section className="panel">
        <h2>Experiments</h2>
        <p className="small muted">Tick rows to restrict the charts to selected experiments. Click an id for its configuration.</p>
        <div className="table-scroll">
          <table className="nowrap" data-testid="experiment-table">
            <thead>
              <tr>
                <th></th>
                <th>Experiment</th>
                <th>Embedding</th>
                <th>Retrieval</th>
                <th>Reranker</th>
                <th>Chunking</th>
                <th>Modalities</th>
                <th className="num">nDCG@10</th>
                <th className="num">Recall@10</th>
                <th className="num">Precision@10</th>
                <th className="num">p50 ms</th>
                <th className="num">p95 ms</th>
                <th className="num">Index bytes</th>
                <th className="num">Faithfulness</th>
                <th className="num">Citation coverage</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {inSuite.map((run) => {
                const d = describe(run);
                return (
                  <tr key={run.run_id}>
                    <td>
                      <input
                        type="checkbox"
                        style={{ width: "auto" }}
                        aria-label={`Compare ${run.experiment_id}`}
                        checked={chosen.includes(run.run_id)}
                        onChange={() =>
                          setChosen((c) => (c.includes(run.run_id) ? c.filter((x) => x !== run.run_id) : [...c, run.run_id]))
                        }
                      />
                    </td>
                    <td>
                      <button type="button" className="link" onClick={() => api.experiment(run.run_id).then(setDetail)}>
                        {run.experiment_id}
                      </button>
                    </td>
                    <td>{d.embedding}</td>
                    <td>{d.method}</td>
                    <td>{d.reranker}</td>
                    <td>{d.chunking}</td>
                    <td>{d.modalities}</td>
                    <td className="num">{fmt(run.metrics["ndcg@10"])}</td>
                    <td className="num">{fmt(run.metrics["recall@10"])}</td>
                    <td className="num">{fmt(run.metrics["precision@10"])}</td>
                    <td className="num">{fmt(run.metrics["p50_latency_ms"], 1)}</td>
                    <td className="num">{fmt(run.metrics["p95_latency_ms"], 1)}</td>
                    <td className="num">{run.metrics["index_size_bytes"]?.toLocaleString() ?? "—"}</td>
                    <td className="num">{fmt(run.metrics["faithfulness"], 3)}</td>
                    <td className="num">{fmt(run.metrics["citation_coverage"], 3)}</td>
                    <td>
                      {run.status} · {run.stage}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      <section className="panel">
        <h2>{metric} by experiment</h2>
        {barData.length === 0 ? (
          <p className="small muted">No completed experiment to chart.</p>
        ) : (
          <div style={{ height: 300 }} data-testid="metric-chart">
            <ResponsiveContainer>
              <BarChart data={barData} margin={{ top: 20, right: 16, bottom: 4, left: 0 }}>
                <CartesianGrid stroke="var(--line)" vertical={false} />
                <XAxis dataKey="name" stroke="var(--muted)" />
                <YAxis stroke="var(--muted)" domain={[0, "auto"]} />
                <Tooltip formatter={(value) => fmt(value)} />
                <Bar dataKey="value" name={metric} fill="var(--accent)" radius={[3, 3, 0, 0]} isAnimationActive={false}>
                  <LabelList dataKey="value" position="top" formatter={(v: number) => fmt(v, 3)} fill="var(--ink)" fontSize={11} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </section>

      <section className="panel">
        <h2>Recall@10 versus p95 latency</h2>
        {scatterData.length === 0 ? (
          <p className="small muted">No completed experiment to chart.</p>
        ) : (
          <div style={{ height: 300 }}>
            <ResponsiveContainer>
              <ScatterChart margin={{ top: 16, right: 24, bottom: 24, left: 0 }}>
                <CartesianGrid stroke="var(--line)" />
                <XAxis type="number" dataKey="x" name="p95 latency" unit=" ms" stroke="var(--muted)" label={{ value: "p95 end-to-end latency (ms)", position: "insideBottom", offset: -12, fill: "var(--muted)", fontSize: 12 }} />
                <YAxis type="number" dataKey="y" name="Recall@10" stroke="var(--muted)" domain={["auto", "auto"]} />
                <Tooltip formatter={(value) => fmt(value, 3)} />
                <Scatter data={scatterData} fill="var(--accent)" isAnimationActive={false}>
                  <LabelList dataKey="name" position="right" fill="var(--ink)" fontSize={11} />
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        )}
        <p className="small muted">Latencies are comparable only between runs measured on the same hardware.</p>
      </section>

      {detail && (
        <section className="panel" data-testid="experiment-detail">
          <h2>
            {detail.experiment_id} <span className="muted mono small">{detail.run_id}</span>
          </h2>
          <div className="stat-row" style={{ margin: "8px 0" }}>
            <span>data <b>{detail.data_kind}</b></span>
            <span>stage <b>{detail.stage}</b></span>
            <span>config hash <b className="mono">{detail.config_hash.slice(0, 12)}</b></span>
            <span>git <b className="mono">{detail.git_commit?.slice(0, 10) ?? "—"}</b></span>
          </div>
          <h3>Safety gates</h3>
          <div className="row" style={{ margin: "6px 0 12px" }}>
            {Object.entries(detail.metrics_by_group.gates ?? {}).map(([gate, value]) => (
              <span key={gate} className={`chip ${value === 1 ? "ok" : "missing"}`}>
                {value === 1 ? "✓" : "✗"} {gate.replace("gate:", "").replace(/_/g, " ")}
              </span>
            ))}
          </div>
          <h3>Cost and resource usage</h3>
          <ResourceUsage summary={detail.summary} />
          <details open>
            <summary>Experiment configuration</summary>
            <pre className="mono small" style={{ overflowX: "auto" }}>{JSON.stringify(detail.config.recipe ?? detail.config, null, 2)}</pre>
          </details>
          <details>
            <summary>Models, indexes and hardware</summary>
            <pre className="mono small" style={{ overflowX: "auto" }}>
              {JSON.stringify({ models: detail.model_revisions, indexes: detail.index_versions, hardware: detail.hardware }, null, 2)}
            </pre>
          </details>
        </section>
      )}
    </div>
  );
}
