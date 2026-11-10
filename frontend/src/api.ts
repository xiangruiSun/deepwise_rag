import type {
  CaseOverview,
  CaseSummary,
  CompareResponse,
  ExperimentDetail,
  ExperimentRun,
  ImageOut,
  Meta,
  RagAnswer,
  ReportOut,
  SearchRequest,
  SearchResponse,
  TimelineEvent,
} from "./types";

const KEY_STORAGE = "inspect-rag-api-key";

export function getApiKey(): string {
  try {
    return window.localStorage.getItem(KEY_STORAGE) ?? import.meta.env.VITE_API_KEY ?? "";
  } catch {
    return import.meta.env.VITE_API_KEY ?? "";
  }
}

export function setApiKey(value: string): void {
  try {
    window.localStorage.setItem(KEY_STORAGE, value);
  } catch {
    /* storage unavailable: the key lives for this page only */
  }
}

export class ApiError extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api/v1${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", "X-API-Key": getApiKey(), ...(init?.headers ?? {}) },
  });
  if (!response.ok) {
    let code = "http_error";
    let message = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      code = body.error?.code ?? code;
      message = body.error?.message ?? message;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(response.status, code, message);
  }
  return (await response.json()) as T;
}

const post = <T>(path: string, body: unknown) => request<T>(path, { method: "POST", body: JSON.stringify(body) });

export const api = {
  meta: () => request<Meta>("/meta"),
  cases: (params: { limit?: number; offset?: number; split?: string }) => {
    const query = new URLSearchParams();
    if (params.limit) query.set("limit", String(params.limit));
    if (params.offset) query.set("offset", String(params.offset));
    if (params.split) query.set("split", params.split);
    return request<{ total: number; items: CaseSummary[] }>(`/cases?${query}`);
  },
  caseOverview: (id: string) => request<CaseOverview>(`/cases/${encodeURIComponent(id)}`),
  timeline: (id: string) =>
    request<{ total: number; events: TimelineEvent[] }>(`/cases/${encodeURIComponent(id)}/timeline?limit=200`),
  reports: (id: string) => request<ReportOut[]>(`/cases/${encodeURIComponent(id)}/reports`),
  images: (id: string) => request<ImageOut[]>(`/cases/${encodeURIComponent(id)}/images`),
  imagePreview: async (caseId: string, studyId: string): Promise<string> => {
    const response = await fetch(
      `/api/v1/cases/${encodeURIComponent(caseId)}/images/${encodeURIComponent(studyId)}/preview`,
      { headers: { "X-API-Key": getApiKey() } },
    );
    if (!response.ok) throw new ApiError(response.status, "preview_unavailable", "No preview available");
    return URL.createObjectURL(await response.blob());
  },
  search: (body: SearchRequest) => post<SearchResponse>("/search", body),
  compare: (caseIds: string[]) => post<CompareResponse>("/compare", { case_ids: caseIds }),
  answer: (question: string, search: SearchRequest) => post<RagAnswer>("/rag/answer", { question, search }),
  experiments: () => request<ExperimentRun[]>("/experiments?limit=200"),
  experiment: (runId: string) => request<ExperimentDetail>(`/experiments/${encodeURIComponent(runId)}`),
};
