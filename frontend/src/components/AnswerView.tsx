import { useNavigate } from "react-router-dom";

import type { Citation, Claim, RagAnswer } from "../types";

function Claims({ title, claims, onCite }: { title: string; claims: Claim[]; onCite: (id: string) => void }) {
  if (!claims.length) return null;
  return (
    <section>
      <h3 style={{ margin: "14px 0 6px" }}>{title}</h3>
      {claims.map((claim) => (
        <p className="claim" key={claim.text}>
          {claim.text}
          {claim.citations.map((source) => (
            <button
              key={source}
              type="button"
              className="cite"
              onClick={() => onCite(source)}
              aria-label={`Open source ${source}`}
            >
              {source}
            </button>
          ))}
        </p>
      ))}
    </section>
  );
}

export function citationTarget(citation: Citation): string {
  return `/cases/${citation.case_id}#source-${citation.record_id}`;
}

export default function AnswerView({ answer }: { answer: RagAnswer }) {
  const navigate = useNavigate();
  const bySource = new Map(answer.citations.map((c) => [c.source_id, c]));
  const open = (sourceId: string) => {
    const citation = bySource.get(sourceId);
    if (citation) navigate(citationTarget(citation));
  };
  const meta = answer.generation_metadata as Record<string, string | number | boolean | null>;
  return (
    <section className="panel" data-testid="rag-answer" aria-live="polite">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <h2>Grounded comparison</h2>
        <span className="small muted">
          generator: {String(meta.backend ?? "—")} / {String(meta.model ?? "—")}
        </span>
      </div>
      {answer.status === "insufficient_evidence" && (
        <p className="notice" role="status">
          Insufficient evidence — no statement was generated.
        </p>
      )}
      {answer.warnings.map((warning) => (
        <p className="notice" key={warning}>
          {warning}
        </p>
      ))}
      <Claims title="Summary" claims={answer.summary} onCite={open} />
      <Claims title="Similarities" claims={answer.similarities} onCite={open} />
      <Claims title="Differences" claims={answer.differences} onCite={open} />
      {answer.missing_evidence.length > 0 && (
        <section>
          <h3 style={{ margin: "14px 0 6px" }}>Missing evidence</h3>
          <ul className="small">
            {answer.missing_evidence.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </section>
      )}
      {answer.citations.length > 0 && (
        <details style={{ marginTop: 12 }} open>
          <summary>Source citations and evidence excerpts ({answer.citations.length})</summary>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Source</th>
                  <th>Record</th>
                  <th>Case</th>
                  <th>Evidence</th>
                </tr>
              </thead>
              <tbody>
                {answer.citations.map((citation) => (
                  <tr key={citation.source_id}>
                    <td>
                      <button type="button" className="cite" onClick={() => open(citation.source_id)}>
                        {citation.source_id}
                      </button>
                    </td>
                    <td>
                      {citation.record_type} <span className="mono">{citation.record_id}</span>
                    </td>
                    <td className="mono">
                      {citation.role === "query" ? "query · " : ""}
                      {citation.case_id}
                    </td>
                    <td>{citation.excerpt}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}
      {answer.limitations.length > 0 && (
        <details style={{ marginTop: 10 }}>
          <summary>Limitations</summary>
          <ul className="small">
            {answer.limitations.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </details>
      )}
    </section>
  );
}
