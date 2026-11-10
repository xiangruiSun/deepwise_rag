import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { api } from "../api";
import AnswerView from "../components/AnswerView";
import ResultCard from "../components/ResultCard";
import type { CaseSummary, LabFilter, Meta, RagAnswer, SearchRequest, SearchResponse } from "../types";

type Mode = "case" | "text" | "filter";

const numberOrNull = (value: string): number | null => (value.trim() === "" ? null : Number(value));

export default function SearchPage({
  meta,
  selected,
  onToggle,
}: {
  meta: Meta;
  selected: string[];
  onToggle: (caseId: string) => void;
}) {
  const [mode, setMode] = useState<Mode>("case");
  const [queryCases, setQueryCases] = useState<CaseSummary[]>([]);
  const [caseId, setCaseId] = useState("");
  const [text, setText] = useState("");
  const [ageMin, setAgeMin] = useState("");
  const [ageMax, setAgeMax] = useState("");
  const [gender, setGender] = useState("");
  const [diagnosis, setDiagnosis] = useState<string[]>([]);
  const [labs, setLabs] = useState<{ concept: string; min: string; max: string }[]>([]);
  const [recipe, setRecipe] = useState(meta.default_recipe);
  const [topK, setTopK] = useState(10);
  const [corpusOnly, setCorpusOnly] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [lastRequest, setLastRequest] = useState<SearchRequest | null>(null);
  const [question, setQuestion] = useState("What imaging and laboratory characteristics are shared by these cases?");
  const [answer, setAnswer] = useState<RagAnswer | null>(null);
  const [answering, setAnswering] = useState(false);

  useEffect(() => {
    // query cases come from the validation split (development queries)
    api
      .cases({ limit: 200, split: "valid" })
      .then((page) => {
        const withReport = page.items.filter((c) => c.has_report);
        setQueryCases(withReport);
        setCaseId((current) => current || withReport[0]?.case_id || "");
      })
      .catch(() => setQueryCases([]));
  }, []);

  const buildRequest = (): SearchRequest => {
    const labFilters: LabFilter[] = labs
      .filter((l) => l.concept && (l.min.trim() !== "" || l.max.trim() !== ""))
      .map((l) => ({ concept: l.concept, min_value: numberOrNull(l.min), max_value: numberOrNull(l.max) }));
    return {
      query_mode: mode,
      case_id: mode === "case" ? caseId : null,
      query_text: mode === "text" ? text : null,
      age_min: numberOrNull(ageMin),
      age_max: numberOrNull(ageMax),
      gender: gender || null,
      diagnosis_codes: diagnosis,
      lab_filters: labFilters,
      candidate_splits: corpusOnly ? ["train"] : null,
      top_k: topK,
      retrieval_recipe_id: recipe,
      exclude_same_patient: true,
    };
  };

  const runSearch = async (event: React.FormEvent) => {
    event.preventDefault();
    const request = buildRequest();
    setBusy(true);
    setError(null);
    setAnswer(null);
    try {
      setResponse(await api.search(request));
      setLastRequest(request);
    } catch (e) {
      setResponse(null);
      setError(e instanceof Error ? e.message : "Search failed");
    } finally {
      setBusy(false);
    }
  };

  const runAnswer = async () => {
    if (!lastRequest) return;
    setAnswering(true);
    setError(null);
    try {
      setAnswer(await api.answer(question, { ...lastRequest, top_k: Math.min(lastRequest.top_k, 5) }));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Answer generation failed");
    } finally {
      setAnswering(false);
    }
  };

  const activeRecipe = meta.recipes.find((r) => r.recipe_id === recipe);
  const canSearch = mode === "case" ? Boolean(caseId) : mode === "text" ? text.trim().length > 0 : true;

  return (
    <div className="layout">
      <form className="panel stack" onSubmit={runSearch} aria-label="Case search">
        <h1>Similar-case search</h1>
        <fieldset>
          <legend>Query type</legend>
          <div className="segmented" style={{ marginTop: 4 }}>
            {(["case", "text", "filter"] as Mode[]).map((m) => (
              <button key={m} type="button" aria-pressed={mode === m} onClick={() => setMode(m)}>
                {m === "case" ? "Existing case" : m === "text" ? "Free text" : "Filters only"}
              </button>
            ))}
          </div>
        </fieldset>

        {mode === "case" && (
          <label className="field">
            <span>Query case</span>
            <select value={caseId} onChange={(e) => setCaseId(e.target.value)} data-testid="query-case">
              {queryCases.length === 0 && <option value="">No authorized query cases</option>}
              {queryCases.map((c) => (
                <option key={c.case_id} value={c.case_id}>
                  {c.case_id}
                  {c.age !== null ? ` · ${c.age} y` : ""}
                  {c.gender ? ` · ${c.gender}` : ""}
                </option>
              ))}
            </select>
          </label>
        )}
        {mode === "text" && (
          <label className="field">
            <span>Clinical description</span>
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="e.g. acute segmental pulmonary embolism without right heart strain"
              data-testid="query-text"
            />
          </label>
        )}

        <fieldset className="stack">
          <legend>Eligibility filters (applied before retrieval)</legend>
          <div className="row">
            <label className="field">
              <span>Age from</span>
              <input type="number" min={0} max={130} value={ageMin} onChange={(e) => setAgeMin(e.target.value)} data-testid="age-min" />
            </label>
            <label className="field">
              <span>Age to</span>
              <input type="number" min={0} max={130} value={ageMax} onChange={(e) => setAgeMax(e.target.value)} data-testid="age-max" />
            </label>
          </div>
          {meta.genders.length > 0 && (
            <label className="field">
              <span>Gender</span>
              <select value={gender} onChange={(e) => setGender(e.target.value)}>
                <option value="">Any</option>
                {meta.genders.map((g) => (
                  <option key={g} value={g}>
                    {g}
                  </option>
                ))}
              </select>
            </label>
          )}
          {meta.diagnosis_codes.length > 0 && (
            <label className="field">
              <span>Diagnosis codes documented before the procedure (all required)</span>
              <select
                multiple
                size={4}
                value={diagnosis}
                onChange={(e) => setDiagnosis(Array.from(e.target.selectedOptions, (o) => o.value))}
              >
                {meta.diagnosis_codes.map((code) => (
                  <option key={code} value={code}>
                    {code}
                  </option>
                ))}
              </select>
            </label>
          )}
          {meta.laboratory_concepts.length > 0 && (
            <div className="stack">
              <span className="small muted" style={{ fontWeight: 600 }}>
                Laboratory ranges (latest value in the lookback window)
              </span>
              {labs.map((lab, index) => {
                const concept = meta.laboratory_concepts.find((c) => c.key === lab.concept);
                const update = (patch: Partial<typeof lab>) =>
                  setLabs(labs.map((l, i) => (i === index ? { ...l, ...patch } : l)));
                return (
                  <div className="row" key={index}>
                    <select value={lab.concept} onChange={(e) => update({ concept: e.target.value })} aria-label="Laboratory test" style={{ flex: 2 }}>
                      {meta.laboratory_concepts.map((c) => (
                        <option key={c.key} value={c.key}>
                          {c.display_name}
                        </option>
                      ))}
                    </select>
                    <input type="number" step="any" placeholder="min" aria-label="Minimum" value={lab.min} onChange={(e) => update({ min: e.target.value })} style={{ flex: 1 }} />
                    <input type="number" step="any" placeholder="max" aria-label="Maximum" value={lab.max} onChange={(e) => update({ max: e.target.value })} style={{ flex: 1 }} />
                    <span className="small muted">{concept?.unit}</span>
                    <button type="button" onClick={() => setLabs(labs.filter((_, i) => i !== index))} aria-label="Remove laboratory filter">
                      ×
                    </button>
                  </div>
                );
              })}
              <button
                type="button"
                onClick={() => setLabs([...labs, { concept: meta.laboratory_concepts[0].key, min: "", max: "" }])}
              >
                Add laboratory range
              </button>
            </div>
          )}
          <label className="check">
            <input type="checkbox" checked={corpusOnly} onChange={(e) => setCorpusOnly(e.target.checked)} />
            <span>Search the historical corpus (training split) only</span>
          </label>
        </fieldset>

        <label className="field">
          <span>Retrieval configuration</span>
          <select value={recipe} onChange={(e) => setRecipe(e.target.value)} data-testid="recipe">
            {meta.recipes.map((r) => (
              <option key={r.recipe_id} value={r.recipe_id}>
                {r.recipe_id}
              </option>
            ))}
          </select>
          {activeRecipe && <span className="small muted">{activeRecipe.description}</span>}
        </label>
        <label className="field">
          <span>Number of cases</span>
          <select value={topK} onChange={(e) => setTopK(Number(e.target.value))}>
            {[5, 10, 20, 50].map((k) => (
              <option key={k} value={k}>
                {k}
              </option>
            ))}
          </select>
        </label>
        <button type="submit" className="primary" disabled={busy || !canSearch} data-testid="search-button">
          {busy ? "Searching…" : "Search"}
        </button>
      </form>

      <div>
        {error && (
          <p className="error" role="alert">
            {error}
          </p>
        )}
        {!response && !error && (
          <div className="panel empty">
            Choose a query case or describe the clinical characteristics, then search. Results are distinct
            historical cases with the evidence that led to their retrieval.
          </div>
        )}
        {response && (
          <>
            <section className="panel" aria-label="Search summary">
              <div className="stat-row">
                <span>
                  <b>{response.results.length}</b> cases shown
                </span>
                <span>
                  <b data-testid="eligible-count">{response.total_eligible_cases ?? "—"}</b> eligible after filters
                </span>
                <span>
                  recipe <b>{response.recipe_id}</b> <span className="muted mono">{response.recipe_hash}</span>
                </span>
                <span>
                  <b>{response.latency_ms.total?.toFixed(1)}</b> ms
                </span>
              </div>
              {response.warnings.map((warning) => (
                <p className="notice" key={warning} style={{ marginBottom: 0 }}>
                  {warning}
                </p>
              ))}
              <details style={{ marginTop: 8 }}>
                <summary>Score types and stage latency</summary>
                <ul className="small">
                  {Object.entries(response.score_descriptions).map(([name, description]) => (
                    <li key={name}>
                      <b>{name}</b>: {description}
                    </li>
                  ))}
                </ul>
                <p className="small muted">
                  {Object.entries(response.latency_ms)
                    .map(([stage, ms]) => `${stage} ${ms.toFixed(1)} ms`)
                    .join(" · ")}
                </p>
              </details>
              {response.results.length > 0 && response.query_mode !== "filter" && (
                <div className="row" style={{ marginTop: 12 }}>
                  <input
                    aria-label="Question about the retrieved cases"
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    style={{ flex: 1, minWidth: 220 }}
                  />
                  <button type="button" onClick={runAnswer} disabled={answering || question.trim().length < 3} data-testid="answer-button">
                    {answering ? "Generating…" : "Generate grounded comparison"}
                  </button>
                  {selected.length >= 2 && <Link to={`/compare?ids=${selected.join(",")}`}>Compare {selected.length} selected</Link>}
                </div>
              )}
            </section>
            {answer && (
              <div style={{ marginTop: 16 }}>
                <AnswerView answer={answer} />
              </div>
            )}
            <section style={{ marginTop: 16 }} aria-label="Ranked cases">
              {response.results.length === 0 && (
                <div className="panel empty">No authorized case matches this query and these filters.</div>
              )}
              {response.results.map((result) => (
                <ResultCard
                  key={result.case_id}
                  result={result}
                  selected={selected.includes(result.case_id)}
                  onToggle={onToggle}
                  textLabel={meta.text_label}
                />
              ))}
            </section>
          </>
        )}
      </div>
    </div>
  );
}
