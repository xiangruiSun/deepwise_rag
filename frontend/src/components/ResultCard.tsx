import { Link } from "react-router-dom";

import type { CaseResult } from "../types";

const fmt = (value: number | null | undefined, digits = 3) =>
  value === null || value === undefined ? "—" : Number(value).toFixed(digits);

function Components({ result }: { result: CaseResult }) {
  if (!result.components.length) return null;
  const numeric = result.components.filter((c) => c.available && c.value !== null);
  const max = Math.max(...numeric.map((c) => Math.abs(c.value as number)), 1e-9);
  return (
    <div className="components" aria-label="Component relevance indicators">
      {result.components.map((component) => (
        <div
          key={component.name}
          className={`component${component.available ? "" : " unavailable"}`}
          title={component.description}
        >
          <div className="label">
            <span>{component.name}</span>
            <span>{component.available ? `rank ${component.rank ?? "—"}` : "unavailable"}</span>
          </div>
          {component.available ? (
            <>
              <div className="meter" aria-hidden="true">
                <div style={{ width: `${Math.min(100, (Math.abs(component.value ?? 0) / max) * 100)}%` }} />
              </div>
              <div className="small muted">
                {component.score_type} score {fmt(component.value, 4)}
                {result.effective_weights[component.name] !== undefined &&
                  ` · weight ${fmt(result.effective_weights[component.name], 2)}`}
              </div>
            </>
          ) : (
            <div className="small">Not available for this case; not used in its score.</div>
          )}
        </div>
      ))}
    </div>
  );
}

export default function ResultCard({
  result,
  selected,
  onToggle,
}: {
  result: CaseResult;
  selected: boolean;
  onToggle: (caseId: string) => void;
}) {
  const demographics = [
    result.age !== null ? `${result.age} y` : null,
    result.gender,
    result.race,
    result.ethnicity,
  ].filter(Boolean);
  return (
    <article className="result" data-testid="result-card">
      <header>
        <span className="rank">#{result.rank}</span>
        <h3>
          <Link to={`/cases/${result.case_id}`} className="mono">
            {result.case_id}
          </Link>
        </h3>
        <span className="muted small" data-testid="demographics">
          {demographics.length ? demographics.join(" · ") : "demographics unavailable"}
        </span>
        <span className="muted small">
          {result.study_ids.length
            ? `${result.study_modality.join("/") || "study"} ${result.study_ids.join(", ")}`
            : "no imaging study"}
        </span>
        <span style={{ marginLeft: "auto" }} className="row">
          {result.final_score !== null && (
            <span className="small muted" title="Ranking score — not a clinical probability">
              {result.score_type} {fmt(result.final_score, 4)}
            </span>
          )}
          <label className="row small">
            <input
              type="checkbox"
              style={{ width: "auto" }}
              checked={selected}
              onChange={() => onToggle(result.case_id)}
            />
            Compare
          </label>
        </span>
      </header>

      {result.evidence.map((evidence) => (
        <div key={evidence.document_id}>
          <blockquote className="excerpt">{evidence.excerpt}</blockquote>
          <div className="small muted">
            Source:{" "}
            <Link to={`/cases/${result.case_id}#source-${evidence.source_id}`}>
              {evidence.source_type.replace("_", " ")} {evidence.source_id}
            </Link>
            {Object.entries(evidence.branch_ranks).map(([branch, rank]) => (
              <span key={branch}>
                {" "}
                · {branch} rank {rank}
              </span>
            ))}
            {evidence.rerank_score !== null && <span> · reranker {fmt(evidence.rerank_score)}</span>}
          </div>
        </div>
      ))}

      <Components result={result} />

      {result.laboratory_comparison.length > 0 && (
        <div className="table-scroll">
          <table>
            <caption className="small muted" style={{ textAlign: "left", captionSide: "top" }}>
              Comparable laboratory measurements (canonical units)
            </caption>
            <thead>
              <tr>
                <th>Measurement</th>
                <th className="num">Query case</th>
                <th className="num">This case</th>
                <th>Unit</th>
                <th className="num">Scaled difference</th>
              </tr>
            </thead>
            <tbody>
              {result.laboratory_comparison.map((item) => (
                <tr key={item.concept_key}>
                  <td>{item.display_name}</td>
                  <td className="num">{fmt(item.query_value, 2)}</td>
                  <td className="num">{fmt(item.candidate_value, 2)}</td>
                  <td>{item.unit}</td>
                  <td className="num">{fmt(item.standardized_difference, 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {result.laboratory_note && <p className="small muted">{result.laboratory_note}</p>}

      <div className="row" style={{ marginTop: 8 }}>
        {Object.entries(result.modality_coverage).map(([modality, available]) => (
          <span key={modality} className={`chip ${available ? "ok" : "missing"}`}>
            {available ? "✓" : "missing:"} {modality}
          </span>
        ))}
        {result.shared_diagnosis_codes.map((code) => (
          <span key={code} className="chip" title="Diagnosis code documented before the procedure in both cases">
            shared {code}
          </span>
        ))}
      </div>

      <details style={{ marginTop: 10 }}>
        <summary>Why retrieved?</summary>
        <ul className="small">
          {result.why_retrieved.map((line) => (
            <li key={line}>{line}</li>
          ))}
        </ul>
      </details>
    </article>
  );
}
