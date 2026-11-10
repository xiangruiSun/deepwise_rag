import { useEffect, useState } from "react";
import { useLocation, useParams } from "react-router-dom";

import { api } from "../api";
import type { CaseOverview, ImageOut, ReportOut, TimelineEvent } from "../types";

const offset = (hours: number) =>
  Math.abs(hours) < 48 ? `${Math.abs(hours).toFixed(1)} h` : `${(Math.abs(hours) / 24).toFixed(0)} d`;

function EventTable({ events, emptyText }: { events: TimelineEvent[]; emptyText: string }) {
  if (!events.length) return <p className="small muted">{emptyText}</p>;
  return (
    <div className="table-scroll">
      <table>
        <thead>
          <tr>
            <th>Offset</th>
            <th>Type</th>
            <th>Code</th>
            <th className="num">Value</th>
            <th>Unit</th>
          </tr>
        </thead>
        <tbody>
          {events.map((event) => (
            <tr key={event.event_id} id={`source-${event.event_id}`} className="source-target">
              <td>{offset(event.hours_from_procedure)}</td>
              <td>{event.event_type}</td>
              <td className="mono">{event.concept_code}</td>
              <td className="num">{event.numeric_value ?? event.text_value ?? "—"}</td>
              <td>{event.unit ?? ""}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Preview({ caseId, image }: { caseId: string; image: ImageOut }) {
  const [url, setUrl] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    let objectUrl: string | null = null;
    if (image.preview_available) {
      api
        .imagePreview(caseId, image.study_id)
        .then((u) => {
          objectUrl = u;
          setUrl(u);
        })
        .catch(() => setFailed(true));
    }
    return () => {
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [caseId, image]);
  if (!image.preview_available || failed) return <p className="small muted">No preview available.</p>;
  return url ? (
    <img className="preview" src={url} alt={`Middle slice of imaging study ${image.study_id}`} />
  ) : (
    <p className="small muted">Loading preview…</p>
  );
}

export default function CasePage({ selected, onToggle }: { selected: string[]; onToggle: (id: string) => void }) {
  const { caseId = "" } = useParams();
  const location = useLocation();
  const [overview, setOverview] = useState<CaseOverview | null>(null);
  const [reports, setReports] = useState<ReportOut[]>([]);
  const [images, setImages] = useState<ImageOut[]>([]);
  const [events, setEvents] = useState<TimelineEvent[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setOverview(null);
    setError(null);
    Promise.all([api.caseOverview(caseId), api.reports(caseId), api.images(caseId), api.timeline(caseId)])
      .then(([o, r, i, t]) => {
        setOverview(o);
        setReports(r);
        setImages(i);
        setEvents(t.events);
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : "Could not load the case"));
  }, [caseId]);

  useEffect(() => {
    // navigate to the cited source record once the content is rendered
    if (!overview || !location.hash) return;
    const target = document.getElementById(decodeURIComponent(location.hash.slice(1)));
    if (target) {
      target.scrollIntoView({ block: "center" });
      target.classList.add("source-highlight");
    }
  }, [overview, location.hash, events, reports]);

  if (error)
    return (
      <p className="error" role="alert">
        {error}
      </p>
    );
  if (!overview) return <p className="empty">Loading case…</p>;

  const before = events.filter((e) => e.phase === "before_procedure");
  const followUp = events.filter((e) => e.phase === "follow_up");
  return (
    <div className="stack">
      <section className="panel">
        <div className="row" style={{ justifyContent: "space-between" }}>
          <h1 className="mono">{overview.case_id}</h1>
          <label className="row small">
            <input type="checkbox" style={{ width: "auto" }} checked={selected.includes(caseId)} onChange={() => onToggle(caseId)} />
            Select for comparison
          </label>
        </div>
        <div className="stat-row" style={{ marginTop: 8 }}>
          <span>Age <b>{overview.age ?? "unavailable"}</b></span>
          <span>Gender <b>{overview.gender ?? "unavailable"}</b></span>
          <span>Race <b>{overview.race ?? "unavailable"}</b></span>
          <span>Ethnicity <b>{overview.ethnicity ?? "unavailable"}</b></span>
          <span>Split <b>{overview.dataset_split ?? "—"}</b></span>
        </div>
        <div className="row" style={{ marginTop: 8 }}>
          {Object.entries(overview.modality_coverage).map(([modality, ok]) => (
            <span key={modality} className={`chip ${ok ? "ok" : "missing"}`}>
              {ok ? "✓" : "missing:"} {modality}
            </span>
          ))}
        </div>
      </section>

      <section className="panel">
        <h2>Radiology impression</h2>
        {reports.length === 0 && <p className="small muted">No impression is available for this case.</p>}
        {reports.map((report) => (
          <div key={report.report_id} id={`source-${report.report_id}`} className="source-target" data-testid="report">
            <p className="small muted">
              Report <span className="mono">{report.report_id}</span>
            </p>
            <blockquote className="excerpt">{report.impression_text}</blockquote>
          </div>
        ))}
      </section>

      <section className="panel">
        <h2>Laboratory measurements used for comparison</h2>
        <p className="small muted">Latest valid value before the procedure within the lookback window.</p>
        {overview.laboratory.length === 0 ? (
          <p className="small muted">No comparable laboratory measurement is available.</p>
        ) : (
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Measurement</th>
                  <th className="num">Recorded value</th>
                  <th>Recorded unit</th>
                  <th className="num">Canonical value</th>
                  <th>Canonical unit</th>
                </tr>
              </thead>
              <tbody>
                {overview.laboratory.map((lab) => (
                  <tr key={lab.concept_key} id={`source-${lab.event_id}`} className="source-target">
                    <td>{lab.display_name}</td>
                    <td className="num">{lab.original_value}</td>
                    <td>{lab.original_unit ?? ""}</td>
                    <td className="num">{Number(lab.value.toFixed(4))}</td>
                    <td>{lab.unit}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {Object.entries(overview.laboratory_unavailable).map(([concept, reason]) => (
          <p className="notice" key={concept}>
            {concept}: not comparable ({reason}).
          </p>
        ))}
      </section>

      <section className="panel">
        <h2>Imaging studies</h2>
        {images.length === 0 && <p className="small muted">No imaging study is linked to this case.</p>}
        {images.map((image) => (
          <div key={image.study_id} id={`source-${image.study_id}`} className="row source-target" style={{ alignItems: "flex-start", gap: 16 }}>
            <Preview caseId={caseId} image={image} />
            <div className="small">
              <div>
                Study <span className="mono">{image.study_id}</span> · {image.modality ?? "unknown modality"} ·{" "}
                {image.slice_count ?? "?"} slices · feature status {image.feature_status}
              </div>
              <p className="muted">{image.note}</p>
            </div>
          </div>
        ))}
      </section>

      <section className="panel">
        <h2>Clinical events before the procedure</h2>
        <EventTable events={before} emptyText="No events were recorded before the procedure." />
      </section>
      <section className="panel">
        <h2>Follow-up information (recorded after the procedure)</h2>
        <p className="notice">
          Retrospective follow-up. These records were not known at the time of the procedure and are never used
          as query-time features.
        </p>
        <EventTable events={followUp} emptyText="No follow-up events are available." />
      </section>
    </div>
  );
}
