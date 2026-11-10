import { useCallback, useEffect, useState } from "react";
import { NavLink, Route, Routes } from "react-router-dom";

import { api, ApiError, getApiKey, setApiKey } from "./api";
import CasePage from "./pages/CasePage";
import ComparePage from "./pages/ComparePage";
import EvaluationPage from "./pages/EvaluationPage";
import SearchPage from "./pages/SearchPage";
import type { Meta } from "./types";

const NOTICE =
  "Research prototype on retrospective data. Not for diagnosis, treatment decisions or any clinical use. " +
  "Scores are ranking signals, not clinical probabilities.";

export default function App() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [key, setKey] = useState(getApiKey());
  const [selected, setSelected] = useState<string[]>([]);

  const load = useCallback(() => {
    setError(null);
    api
      .meta()
      .then(setMeta)
      .catch((e: unknown) => {
        setMeta(null);
        setError(
          e instanceof ApiError && e.status === 401
            ? "Enter a valid API key to use the research interface."
            : e instanceof Error
              ? e.message
              : "The backend is not reachable.",
        );
      });
  }, []);

  useEffect(load, [load]);

  const toggle = useCallback(
    (caseId: string) =>
      setSelected((current) => (current.includes(caseId) ? current.filter((c) => c !== caseId) : [...current, caseId])),
    [],
  );

  return (
    <>
      <div className="research-banner" role="note">
        <div>
          <strong>Research use only.</strong> {NOTICE}
        </div>
      </div>
      <header className="app-header">
        <div className="bar">
          <span className="brand">INSPECT-RAG</span>
          {meta?.data_kind === "synthetic" && (
            <span className="chip synthetic" title="Fabricated fixtures for pipeline verification">
              SYNTHETIC DATA
            </span>
          )}
          {meta && meta.data_kind !== "synthetic" && meta.data_kind !== "inspect" && (
            <span className="chip" title="Openly licensed dataset, not INSPECT" data-testid="dataset-chip">
              {meta.dataset}
            </span>
          )}
          <nav aria-label="Main">
            <NavLink to="/" end>
              Search
            </NavLink>
            <NavLink to={`/compare${selected.length ? `?ids=${selected.join(",")}` : ""}`}>
              Compare{selected.length ? ` (${selected.length})` : ""}
            </NavLink>
            <NavLink to="/evaluation">Evaluation</NavLink>
          </nav>
          <form
            className="row"
            onSubmit={(e) => {
              e.preventDefault();
              setApiKey(key);
              load();
            }}
          >
            <label className="small muted" htmlFor="api-key">
              API key
            </label>
            <input
              id="api-key"
              type="password"
              autoComplete="off"
              value={key}
              onChange={(e) => setKey(e.target.value)}
              style={{ width: 170 }}
            />
            <button type="submit">Use key</button>
          </form>
        </div>
      </header>
      <main>
        {error && (
          <p className="error" role="alert">
            {error}
          </p>
        )}
        {meta && (
          <Routes>
            <Route path="/" element={<SearchPage meta={meta} selected={selected} onToggle={toggle} />} />
            <Route
              path="/cases/:caseId"
              element={<CasePage selected={selected} onToggle={toggle} textLabel={meta.text_label} />}
            />
            <Route
              path="/compare"
              element={<ComparePage selected={selected} onToggle={toggle} textLabel={meta.text_label} />}
            />
            <Route path="/evaluation" element={<EvaluationPage />} />
            <Route path="*" element={<p className="empty">Page not found.</p>} />
          </Routes>
        )}
      </main>
    </>
  );
}
