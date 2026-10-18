import { AssessmentFormData, PredictionResult, SimulationAdjustments, SimulationResult, ModelMetadata } from '../types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function fetchHealth(): Promise<{ status: string; model_loaded: boolean }> {
  const res = await fetch(`${API_BASE_URL}/health`);
  if (!res.ok) {
    throw new Error(`Health check failed: ${res.statusText}`);
  }
  return res.json();
}

export async function fetchMetadata(): Promise<ModelMetadata> {
  const res = await fetch(`${API_BASE_URL}/api/metadata`);
  if (!res.ok) {
    throw new Error(`Failed to fetch model metadata: ${res.statusText}`);
  }
  return res.json();
}

export async function submitAssessment(data: AssessmentFormData): Promise<PredictionResult> {
  const res = await fetch(`${API_BASE_URL}/api/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Prediction failed (${res.status})`);
  }

  return res.json();
}

export async function runSimulation(
  baseline: AssessmentFormData,
  adjustments: SimulationAdjustments
): Promise<SimulationResult> {
  const res = await fetch(`${API_BASE_URL}/api/simulate`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      baseline,
      adjustments,
    }),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Simulation failed (${res.status})`);
  }

  return res.json();
}
