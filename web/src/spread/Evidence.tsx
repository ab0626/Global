import { useEffect, useState, type ReactNode } from "react";
import { documentEvidence, type Evidence, type EvidenceEdge } from "./api";

type Props = {
  documentId: number;
  countryName: (code: string | null) => string;
  utc: (iso: string) => string;
  onClose: () => void;
};

type State =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "ready"; evidence: Evidence };

const LOADING: State = { kind: "loading" };

/** "Why is this article in this event?" — the document's assignment score, the
 * evidence checks derived from its strongest in-incident graph edges, and the
 * edges themselves (supporting = same incident, competing = the best edge into any
 * other incident). Scores are the clustering run's own pair channels. */
export function EvidencePanel(props: Props) {
  return <Loader key={props.documentId} {...props} />;
}

/** Remounted per document (keyed above) so state starts at LOADING without an
 * effect-driven reset. */
function Loader({ documentId, countryName, utc, onClose }: Props) {
  const [state, setState] = useState<State>(LOADING);

  useEffect(() => {
    let cancelled = false;
    documentEvidence(documentId)
      .then((evidence) => {
        if (!cancelled) setState({ kind: "ready", evidence });
      })
      .catch((e: Error) => {
        if (!cancelled) setState({ kind: "error", message: e.message });
      });
    return () => {
      cancelled = true;
    };
  }, [documentId]);

  return (
    <section className="evidence" aria-live="polite">
      <div className="evidence-head">
        <h3>Why is this article here?</h3>
        <button className="link" onClick={onClose} aria-label="close evidence">
          close
        </button>
      </div>
      {state.kind === "loading" && <p className="muted small">loading evidence…</p>}
      {state.kind === "error" && <p className="notice error">{state.message}</p>}
      {state.kind === "ready" && (
        <Body evidence={state.evidence} countryName={countryName} utc={utc} />
      )}
    </section>
  );
}

function Body({
  evidence,
  countryName,
  utc,
}: {
  evidence: Evidence;
  countryName: Props["countryName"];
  utc: Props["utc"];
}) {
  const { document: doc, checks, supporting, competing } = evidence;
  const best = competing[0];
  const strongest = supporting[0];
  const contested = best && strongest && best.gated > strongest.gated;
  return (
    <>
      <p className="evidence-doc">
        <a href={doc.url} target="_blank" rel="noreferrer">
          {doc.title ?? doc.url}
        </a>
        <span className="muted small">
          {doc.source_domain} · {countryName(doc.publisher_country)}
          {doc.language ? ` · ${doc.language}` : ""} · {utc(doc.observed_time)}
        </span>
      </p>

      <dl className="stats evidence-stats">
        <div>
          <dt>assignment score</dt>
          <dd>{evidence.assignment_score.toFixed(2)}</dd>
        </div>
        <div>
          <dt>supporting edges</dt>
          <dd>{supporting.length}</dd>
        </div>
        <div>
          <dt>best competitor</dt>
          <dd>{best ? best.gated.toFixed(2) : "—"}</dd>
        </div>
      </dl>

      <ul className="checks">
        <Check ok={checks.title_similarity != null && checks.title_similarity >= 0.5}>
          title semantic similarity{" "}
          {checks.title_similarity == null ? "—" : checks.title_similarity.toFixed(2)}
        </Check>
        <Check ok={checks.shared_gdelt_event}>shared GDELT event</Check>
        <Check ok={checks.hours_to_nearest_support != null}>
          same event window
          {checks.hours_to_nearest_support == null
            ? ""
            : ` (nearest support ${checks.hours_to_nearest_support.toFixed(1)} h away)`}
        </Check>
        <Check ok={checks.shared_entities}>matching person / organization</Check>
        <Check ok={checks.shared_url_tokens}>shared URL slug tokens</Check>
        <Check ok={checks.other_publisher_country} neutral>
          {checks.other_publisher_country
            ? "corroborated from another publisher country"
            : "only same-country corroboration"}
        </Check>
      </ul>

      <p className="small">
        <span className="muted">incident</span>{" "}
        {evidence.incident ? evidence.incident.title ?? evidence.incident.label : "not a macro-event (too small or unassigned)"}
        <br />
        <span className="muted">story family</span>{" "}
        {evidence.family ? evidence.family.title ?? evidence.family.label : "—"}
      </p>

      {supporting.length > 0 && (
        <>
          <h4>Strongest supporting edges (same incident)</h4>
          <ol className="edges">
            {supporting.map((e) => (
              <Edge key={e.neighbor.document_id} edge={e} countryName={countryName} />
            ))}
          </ol>
        </>
      )}
      {best && (
        <>
          <h4>
            Best competing edge (other incident)
            {contested && <span className="chip warn">contested</span>}
          </h4>
          <ol className="edges competing">
            <Edge edge={best} countryName={countryName} />
          </ol>
        </>
      )}
      {supporting.length === 0 && !best && (
        <p className="muted small">
          No graph edge survived the evidence gate for this article: it is a singleton.
        </p>
      )}
      <p className="muted small">{evidence.note}.</p>
    </>
  );
}

function Check({
  ok,
  neutral,
  children,
}: {
  ok: boolean;
  neutral?: boolean;
  children: ReactNode;
}) {
  return (
    <li className={ok ? "ok" : neutral ? "neutral" : "no"}>
      <span className="mark">{ok ? "✓" : neutral ? "·" : "✗"}</span>
      {children}
    </li>
  );
}

function Edge({
  edge,
  countryName,
}: {
  edge: EvidenceEdge;
  countryName: Props["countryName"];
}) {
  const n = edge.neighbor;
  return (
    <li>
      <span className="edge-title">{n.title ?? `document #${n.document_id}`}</span>
      <span className="muted small">
        {n.source_domain} · {countryName(n.publisher_country)}
        {n.language ? ` · ${n.language}` : ""} · {edge.delta_hours.toFixed(1)} h apart
      </span>
      <span className="edge-scores small">
        <b>edge {edge.gated.toFixed(2)}</b>
        {edge.title_score > 0 && <> · title {edge.title_score.toFixed(2)}</>}
        {edge.event_score > 0 && <> · event {edge.event_score.toFixed(2)}</>}
        {edge.entity_score > 0 && <> · entity {edge.entity_score.toFixed(2)}</>}
        {edge.url_score > 0 && <> · url {edge.url_score.toFixed(2)}</>}
        {" · "}
        {edge.evidence_channels} channel{edge.evidence_channels === 1 ? "" : "s"}
      </span>
    </li>
  );
}
