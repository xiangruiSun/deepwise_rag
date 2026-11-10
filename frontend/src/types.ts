// Mirrors the backend Pydantic contracts (backend/app/schemas).

export interface LabFilter {
  concept: string;
  min_value?: number | null;
  max_value?: number | null;
}

export interface SearchRequest {
  query_mode: "case" | "text" | "filter";
  case_id?: string | null;
  query_text?: string | null;
  age_min?: number | null;
  age_max?: number | null;
  gender?: string | null;
  diagnosis_codes?: string[];
  lab_filters?: LabFilter[];
  study_modality?: string | null;
  candidate_splits?: string[] | null;
  top_k: number;
  retrieval_recipe_id?: string | null;
  exclude_same_patient?: boolean;
}

export interface ComponentScore {
  name: string;
  value: number | null;
  rank: number | null;
  available: boolean;
  score_type: string;
  description: string;
}

export interface EvidenceRef {
  document_id: string;
  source_type: string;
  source_id: string;
  chunk_index: number;
  excerpt: string;
  matched_text: string | null;
  branch_scores: Record<string, number>;
  branch_ranks: Record<string, number>;
  rerank_score: number | null;
}

export interface LabComparisonItem {
  concept_key: string;
  display_name: string;
  unit: string;
  query_value: number | null;
  candidate_value: number | null;
  standardized_difference: number | null;
}

export interface LabValue {
  concept_key: string;
  display_name: string;
  value: number;
  unit: string;
  original_value: number;
  original_unit: string | null;
  event_id: string;
  n_observations: number;
  aggregate: string;
}

export interface CaseResult {
  rank: number;
  case_id: string;
  age: number | null;
  gender: string | null;
  race: string | null;
  ethnicity: string | null;
  dataset_split: string | null;
  study_ids: string[];
  study_modality: string[];
  final_score: number | null;
  score_type: string;
  components: ComponentScore[];
  effective_weights: Record<string, number>;
  evidence: EvidenceRef[];
  laboratory_comparison: LabComparisonItem[];
  laboratory_values: LabValue[];
  laboratory_note: string | null;
  shared_diagnosis_codes: string[];
  modality_coverage: Record<string, boolean>;
  missing_modalities: string[];
  why_retrieved: string[];
}

export interface SearchResponse {
  request_id: string;
  query_id: string;
  query_mode: string;
  recipe_id: string;
  recipe_hash: string;
  recipe_summary: Record<string, unknown>;
  total_eligible_cases: number | null;
  results: CaseResult[];
  score_descriptions: Record<string, string>;
  modality_coverage: Record<string, number>;
  latency_ms: Record<string, number>;
  warnings: string[];
  notice: string;
}

export interface RecipeInfo {
  recipe_id: string;
  description: string;
  branches: string[];
  embedding: string | null;
  reranker: string | null;
  multimodal: { enabled: boolean; modalities?: string[] };
}

export interface Meta {
  dataset: string;
  data_kind: string;
  genders: string[];
  laboratory_concepts: { key: string; display_name: string; unit: string }[];
  diagnosis_codes: string[];
  study_modalities: string[];
  age_range: number[] | null;
  recipes: RecipeInfo[];
  default_recipe: string;
  generation_backend: string;
  image_modality_enabled: boolean;
  notice: string;
}

export interface CaseSummary {
  case_id: string;
  age: number | null;
  gender: string | null;
  dataset_split: string | null;
  has_report: boolean;
  has_image: boolean;
}

export interface CaseOverview {
  case_id: string;
  age: number | null;
  gender: string | null;
  race: string | null;
  ethnicity: string | null;
  dataset_split: string | null;
  report_ids: string[];
  study_ids: string[];
  n_events_before_procedure: number;
  n_events_after_procedure: number;
  laboratory: LabValue[];
  laboratory_unavailable: Record<string, string>;
  diagnosis_codes: string[];
  modality_coverage: Record<string, boolean>;
}

export interface TimelineEvent {
  event_id: string;
  event_type: string;
  concept_code: string;
  display_name: string | null;
  numeric_value: number | null;
  text_value: string | null;
  unit: string | null;
  hours_from_procedure: number;
  phase: "before_procedure" | "follow_up";
}

export interface ReportOut {
  report_id: string;
  case_id: string;
  impression_text: string;
}

export interface ImageOut {
  study_id: string;
  modality: string | null;
  slice_count: number | null;
  feature_status: string;
  preview_available: boolean;
  note: string;
}

export interface CompareColumn {
  case_id: string;
  demographics: Record<string, string | number | null>;
  laboratory: LabValue[];
  laboratory_unavailable: Record<string, string>;
  diagnoses: { event_id: string; concept_code: string; display_name: string | null; hours_from_procedure: number }[];
  impressions: { report_id: string; impression_text: string }[];
  imaging: { study_id: string; modality: string | null; slice_count: number | null; feature_status: string; available: boolean }[];
  follow_up: { event_id: string; event_type: string; concept_code: string; numeric_value: number | null; unit: string | null; hours_after_procedure: number }[];
}

export interface CompareResponse {
  cases: CompareColumn[];
  shared_diagnosis_codes: string[];
  comparable_laboratory_concepts: string[];
  lookback_days: number;
  warnings: string[];
}

export interface Claim {
  text: string;
  citations: string[];
}

export interface Citation {
  source_id: string;
  record_type: "report" | "laboratory" | "diagnosis" | "imaging";
  record_id: string;
  case_id: string;
  role: "query" | "retrieved";
  excerpt: string;
}

export interface RagAnswer {
  status: "answered" | "insufficient_evidence";
  summary: Claim[];
  similarities: Claim[];
  differences: Claim[];
  missing_evidence: string[];
  citations: Citation[];
  limitations: string[];
  generation_metadata: Record<string, unknown>;
  warnings: string[];
  retrieved_case_ids: string[];
}

export interface ExperimentRun {
  run_id: string;
  suite_id: string | null;
  experiment_id: string;
  status: string;
  stage: string;
  data_kind: string;
  config_hash: string;
  started_at: string;
  finished_at: string | null;
  metrics: Record<string, number | null>;
  summary: Record<string, unknown>;
}

export interface ExperimentDetail extends ExperimentRun {
  config: Record<string, unknown>;
  model_revisions: Record<string, unknown>;
  index_versions: Record<string, unknown>;
  hardware: Record<string, unknown>;
  git_commit: string | null;
  metrics_by_group: Record<string, Record<string, number | null>>;
}
