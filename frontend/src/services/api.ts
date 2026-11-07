import type { GenerateResponse, CompareResponse, HealthResponse, BenchmarkReport } from '../types';

// API Mode configuration: 'local' (FastAPI) or 'gradio' (Hugging Face Space)
const API_MODE = import.meta.env.VITE_API_MODE || 'local';
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';
const GRADIO_SPACE_URL = import.meta.env.VITE_GRADIO_SPACE_URL || 'cynstl/atelierai-api';

let gradioClientPromise: Promise<any> | null = null;

async function getGradioClient() {
  if (!gradioClientPromise) {
    const { Client } = await import('@gradio/client');
    gradioClientPromise = Client.connect(GRADIO_SPACE_URL);
  }
  return gradioClientPromise;
}

export async function checkHealth(): Promise<HealthResponse> {
  if (API_MODE === 'gradio') {
    try {
      const app = await getGradioClient();
      const result = await app.predict('/health', []);
      return (result.data as any)[0] as HealthResponse;
    } catch (err: any) {
      console.warn('Gradio health check error (may be cold-starting):', err);
      return {
        status: 'cold_starting',
        app_name: 'AtelierAI',
        version: '1.0.0',
        device: 'cuda',
        cuda_available: true,
        gpu_name: 'ZeroGPU (NVIDIA A100/H100/T4 on demand)',
        checkpoints_available: {
          cgan_generator: true,
          vae: true,
          classifier: true,
          distilbert: true,
        },
      };
    }
  }

  const res = await fetch(`${API_BASE_URL}/api/health`);
  if (!res.ok) {
    throw new Error(`Health check failed with status: ${res.status}`);
  }
  return res.json();
}

export async function generateFashion(params: {
  prompt: string;
  model_type: 'cgan' | 'vae';
  strategy: 'zero-shot' | 'few-shot';
  seed?: number;
  evaluate_clip?: boolean;
}): Promise<GenerateResponse> {
  if (API_MODE === 'gradio') {
    const app = await getGradioClient();
    const result = await app.predict('/generate', [
      params.prompt,
      params.model_type,
      params.strategy,
      params.seed ?? 42,
    ]);
    const data = (result.data as any)[0];
    if (data && data.success === false && data.error) {
      throw new Error(data.error);
    }
    return data as GenerateResponse;
  }

  // Local FastAPI Mode
  const res = await fetch(`${API_BASE_URL}/api/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      prompt: params.prompt,
      model_type: params.model_type,
      strategy: params.strategy,
      seed: params.seed,
      evaluate_clip: params.evaluate_clip ?? true,
    }),
  });

  if (!res.ok) {
    const errorBody = await res.json().catch(() => ({}));
    throw new Error(errorBody.detail || `Generation request failed with status: ${res.status}`);
  }

  return res.json();
}

export async function compareFashion(prompt: string, seed: number = 42): Promise<CompareResponse> {
  if (API_MODE === 'gradio') {
    const app = await getGradioClient();
    const result = await app.predict('/compare', [prompt]);
    const data = (result.data as any)[0];
    if (data && data.success === false && data.error) {
      throw new Error(data.error);
    }
    return data as CompareResponse;
  }

  const res = await fetch(`${API_BASE_URL}/api/compare`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt, seed }),
  });

  if (!res.ok) {
    const errorBody = await res.json().catch(() => ({}));
    throw new Error(errorBody.detail || `Comparison request failed with status: ${res.status}`);
  }

  return res.json();
}

export async function fetchBenchmarkReport(): Promise<BenchmarkReport> {
  if (API_MODE === 'gradio') {
    const app = await getGradioClient();
    const result = await app.predict('/benchmark', []);
    return (result.data as any)[0] as BenchmarkReport;
  }

  const res = await fetch(`${API_BASE_URL}/api/benchmark`);
  if (!res.ok) {
    throw new Error(`Failed to load benchmark report: ${res.status}`);
  }
  return res.json();
}

export async function fetchModelInsights(): Promise<any> {
  if (API_MODE === 'gradio') {
    const app = await getGradioClient();
    const result = await app.predict('/insights', []);
    return (result.data as any)[0];
  }

  const res = await fetch(`${API_BASE_URL}/api/insights`);
  if (!res.ok) {
    throw new Error(`Failed to load model insights: ${res.status}`);
  }
  return res.json();
}
