import { useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";

import { api } from "../api";
import type { CompareResponse } from "../types";

export default function ComparePage({ selected, onToggle }: { selected: string[]; onToggle: (id: string) => void }) {
  const [params] = useSearchParams();
  const fromUrl = (params.get("ids") ?? "").split(",").filter(Boolean);
  const ids = fromUrl.length ? fromUrl : selected;
  const key = ids.join(",");
  const [data, setData] = useState<CompareResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setData(null);
    setError(null);
    const caseIds = key.split(",").filter(Boolean);
    if (caseIds.length < 2) return;
    api
      .compare(caseIds)
      .then(setData)
      .catch((e: unknown) => setError(e instanceof Error ? e.message : "Comparison failed"));
  }, [key]);

  if (ids.length < 2)
    return (
      <div className="panel empty">
        Select at least two cases with the “Compare” checkbox on the <Link to="/">search results</Link> to see
        them side by side.
      </div>
    );
  if (error)
    return (
      <p className="error" role="alert">
        {error}
      </p>
    );
  if (!data) return <p className="empty">Loading comparison…</p>;

  const concepts = Array.from(new Set(data.cases.flatMap((c) => c.laboratory.map((l) => l.concept_key)))).sort();
  return (
    <div className="stack">
      <section className="panel">
        <h1>Case comparison</h1>
        <div className="stat-row" style={{ marginTop: 8 }}>
          <span>
            <b>{data.cases.length}</b> cases
          </span>
          <span>
            laboratory lookback <b>{data.lookback_days}</b> days
          </span>
          <span>
            shared diagnosis codes <b>{data.shared_diagnosis_codes.join(", ") || "none"}</b>
          </span>
        </div>
        {data.warnings.map((w) => (
          <p className="notice" key={w}>
            {w}
          </p>
        ))}
      </section>

      <section className="panel">
        <h2>Laboratory measurements (canonical units)</h2>
        <div className="table-scroll">
          <table data-testid="lab-compare">
            <thead>
              <tr>
                <th>Measurement</th>
                {data.cases.map((c) => (
                  <th key={c.case_id} className="num mono">
                    {c.case_id.slice(-8)}
                  </th>
                ))}
                <th>Unit</th>
              </tr>
            </thead>
            <tbody>
              {concepts.map((concept) => {
                const any = data.cases.flatMap((c) => c.laboratory).find((l) => l.concept_key === concept);
                return (
                  <tr key={concept}>
                    <td>{any?.display_name ?? concept}</td>
                    {data.cases.map((c) => {
                      const lab = c.laboratory.find((l) => l.concept_key === concept);
                      const reason = c.laboratory_unavailable[concept];
                      return (
                        <td key={c.case_id} className="num" title={lab ? `recorded: ${lab.original_value} ${lab.original_unit ?? ""}` : reason}>
                          {lab ? Number(lab.value.toFixed(3)) : <span className="chip missing">{reason ? "not comparable" : "missing"}</span>}
                        </td>
                      );
                    })}
                    <td>{any?.unit}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        {concepts.length === 0 && <p className="small muted">No laboratory measurements are available for these cases.</p>}
      </section>

      <div className="compare-grid">
        {data.cases.map((column) => (
          <section className="panel" key={column.case_id} data-testid="compare-column" style={{ margin: 0 }}>
            <div className="row" style={{ justifyContent: "space-between" }}>
              <h2 className="mono">
                <Link to={`/cases/${column.case_id}`}>{column.case_id}</Link>
              </h2>
              <button type="button" className="link small" onClick={() => onToggle(column.case_id)}>
                {selected.includes(column.case_id) ? "Remove from selection" : "Add to selection"}
              </button>
            </div>
            <h3 style={{ marginTop: 12 }}>Demographics</h3>
            <p className="small">
              {Object.entries(column.demographics)
                .map(([k, v]) => `${k.replace("_", " ")}: ${v ?? "unavailable"}`)
                .join(" · ")}
            </p>
            <h3>Radiology impression</h3>
            {column.impressions.length === 0 && <p className="small muted">No impression available.</p>}
            {column.impressions.map((i) => (
              <blockquote className="excerpt" key={i.report_id}>
                {i.impression_text}
              </blockquote>
            ))}
            <h3>Documented diagnoses (before the procedure)</h3>
            <p className="small">
              {column.diagnoses.length ? column.diagnoses.map((d) => d.concept_code).join(", ") : "None documented."}
            </p>
            <h3>Imaging</h3>
            <p className="small">
              {column.imaging.length
                ? column.imaging
                    .map((s) => `${s.study_id} (${s.modality ?? "?"}, ${s.available ? s.feature_status : "source missing"})`)
                    .join("; ")
                : "No imaging study."}
            </p>
            <details>
              <summary>Follow-up after the procedure ({column.follow_up.length})</summary>
              <p className="small muted">Not known at procedure time; never used as query-time features.</p>
              <ul className="small">
                {column.follow_up.map((f) => (
                  <li key={f.event_id}>
                    +{f.hours_after_procedure} h · {f.event_type} <span className="mono">{f.concept_code}</span>
                    {f.numeric_value !== null ? ` · ${f.numeric_value} ${f.unit ?? ""}` : ""}
                  </li>
                ))}
              </ul>
            </details>
          </section>
        ))}
      </div>
    </div>
  );
}
