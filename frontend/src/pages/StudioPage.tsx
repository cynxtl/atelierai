import React, { useState } from 'react';
import {
  Sparkles,
  ArrowRight,
  Download,
  CheckCircle2,
  AlertCircle,
  ShieldAlert,
  Info,
  Clock,
  Layers,
} from 'lucide-react';
import { generateFashion } from '../services/api';
import type { GenerateResponse } from '../types';

export const StudioPage: React.FC = () => {
  const [prompt, setPrompt] = useState<string>('Show me four ankle boots.');
  const [modelType, setModelType] = useState<'cgan' | 'vae'>('cgan');
  const [strategy, setStrategy] = useState<'zero-shot' | 'few-shot'>('zero-shot');
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<GenerateResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const examplePrompts = [
    'Show me four ankle boots.',
    'Generate a stylish summer dress.',
    'I need 2 casual denim trousers.',
    'Can you synthesize a warm winter pullover?',
    'Produce three white graphic t-shirts.',
    'A crisp formal button-down shirt.',
    'Two comfortable leather sandals for the beach.',
    'Generate a pair of athletic running sneakers.',
    'A designer leather handbag or tote bag.',
    'A heavy padded winter coat.',
    'Three luxury fedora hats.', // Out of scope test example
  ];

  const handleGenerate = async (targetPrompt?: string) => {
    const activePrompt = targetPrompt || prompt;
    if (!activePrompt.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const data = await generateFashion({
        prompt: activePrompt,
        model_type: modelType,
        strategy: strategy,
      });

      if (!data.success) {
        setError(data.error || 'Failed to interpret prompt.');
        setResult(data);
      } else {
        setResult(data);
      }
    } catch (err: any) {
      setError(err.message || 'Error communicating with backend server.');
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadImage = (b64Data: string, filename: string) => {
    const link = document.createElement('a');
    link.href = b64Data;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Scope disclaimer notification */}
      <div className="bg-neutral-900/70 border border-neutral-800 rounded-xl p-4 flex items-start space-x-3 text-xs text-neutral-400">
        <Info className="w-4 h-4 text-indigo-400 mt-0.5 shrink-0" />
        <p className="leading-relaxed">
          <strong className="text-neutral-200">Fashion-MNIST Domain:</strong> AtelierAI operates on genuine 28×28 grayscale
          silhouettes across 10 apparel classes (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot).
          It does not synthesize full photorealistic colors or arbitrary unconstrained clothing items.
        </p>
      </div>

      {/* Main Synthesis Workspace */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: Controls & Prompting */}
        <div className="lg:col-span-5 space-y-6">
          <div className="bg-neutral-900 border border-neutral-800 rounded-2xl p-6 shadow-sm space-y-6">
            <div>
              <label htmlFor="prompt-input" className="block text-sm font-medium text-neutral-200 mb-2">
                Natural Language Clothing Description
              </label>
              <div className="relative">
                <textarea
                  id="prompt-input"
                  rows={3}
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder="e.g., Show me 4 ankle boots..."
                  className="w-full bg-neutral-950 border border-neutral-800 rounded-xl p-3.5 text-sm text-neutral-100 placeholder-neutral-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 transition-all resize-none"
                />
              </div>
            </div>

            {/* Quick Prompt Chips */}
            <div>
              <span className="text-xs font-medium text-neutral-400 block mb-2">Example Requests</span>
              <div className="flex flex-wrap gap-1.5 max-h-36 overflow-y-auto pr-1">
                {examplePrompts.map((ex, i) => (
                  <button
                    key={i}
                    onClick={() => {
                      setPrompt(ex);
                      handleGenerate(ex);
                    }}
                    className="text-[11px] bg-neutral-950 hover:bg-neutral-800 text-neutral-300 hover:text-white px-2.5 py-1 rounded-md border border-neutral-800/80 transition-colors text-left"
                  >
                    {ex}
                  </button>
                ))}
              </div>
            </div>

            {/* Model & Strategy Controls */}
            <div className="grid grid-cols-2 gap-4 pt-2 border-t border-neutral-800/80">
              <div>
                <label className="block text-xs font-medium text-neutral-400 mb-1.5">Generative Engine</label>
                <div className="grid grid-cols-2 gap-1 bg-neutral-950 p-1 rounded-lg border border-neutral-800">
                  <button
                    onClick={() => setModelType('cgan')}
                    className={`text-xs py-1.5 rounded-md font-medium transition-all ${
                      modelType === 'cgan'
                        ? 'bg-indigo-600 text-white shadow'
                        : 'text-neutral-400 hover:text-neutral-200'
                    }`}
                  >
                    cGAN
                  </button>
                  <button
                    onClick={() => setModelType('vae')}
                    className={`text-xs py-1.5 rounded-md font-medium transition-all ${
                      modelType === 'vae'
                        ? 'bg-indigo-600 text-white shadow'
                        : 'text-neutral-400 hover:text-neutral-200'
                    }`}
                  >
                    C-VAE
                  </button>
                </div>
              </div>

              <div>
                <label className="block text-xs font-medium text-neutral-400 mb-1.5">Prompt Strategy</label>
                <div className="grid grid-cols-2 gap-1 bg-neutral-950 p-1 rounded-lg border border-neutral-800">
                  <button
                    onClick={() => setStrategy('zero-shot')}
                    className={`text-xs py-1.5 rounded-md font-medium transition-all ${
                      strategy === 'zero-shot'
                        ? 'bg-neutral-800 text-white shadow'
                        : 'text-neutral-400 hover:text-neutral-200'
                    }`}
                  >
                    Zero-Shot
                  </button>
                  <button
                    onClick={() => setStrategy('few-shot')}
                    className={`text-xs py-1.5 rounded-md font-medium transition-all ${
                      strategy === 'few-shot'
                        ? 'bg-neutral-800 text-white shadow'
                        : 'text-neutral-400 hover:text-neutral-200'
                    }`}
                  >
                    Few-Shot
                  </button>
                </div>
              </div>
            </div>

            {/* Generate Action Button */}
            <button
              onClick={() => handleGenerate()}
              disabled={loading || !prompt.trim()}
              className="w-full bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-500 hover:to-violet-500 text-white font-medium py-3 rounded-xl shadow-lg shadow-indigo-600/20 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2 transition-all"
            >
              {loading ? (
                <>
                  <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  <span>Synthesizing Images...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  <span>Synthesize Apparel</span>
                  <ArrowRight className="w-4 h-4 ml-1" />
                </>
              )}
            </button>
          </div>
        </div>

        {/* Right Column: Results Gallery & Metrics */}
        <div className="lg:col-span-7 space-y-6">
          {/* Error / Out of scope banner */}
          {error && (
            <div className="bg-red-950/40 border border-red-800/80 rounded-2xl p-5 text-red-200 space-y-2">
              <div className="flex items-center space-x-2 font-medium">
                <ShieldAlert className="w-5 h-5 text-red-400" />
                <span>Request Validation Notice</span>
              </div>
              <p className="text-xs text-red-300 leading-relaxed">{error}</p>
              {result?.unsupported_requested && (
                <div className="text-[11px] text-red-400 pt-1">
                  Requested item: <code className="bg-red-900/40 px-1 py-0.5 rounded">{result.unsupported_requested}</code> is
                  outside Fashion-MNIST categories.
                </div>
              )}
            </div>
          )}

          {/* Result Metadata Banner */}
          {result?.success && (
            <div className="bg-neutral-900 border border-neutral-800 rounded-2xl p-5 space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-neutral-800">
                <div className="flex items-center space-x-3">
                  <span className="text-xs font-mono uppercase px-2.5 py-1 rounded bg-indigo-950/80 text-indigo-300 border border-indigo-800/60 font-semibold">
                    Class #{result.class_idx}: {result.display_label}
                  </span>
                  <span className="text-xs text-neutral-400 font-mono">
                    Qty: {result.count} {result.count > 1 ? 'items' : 'item'}
                  </span>
                </div>
                <div className="flex items-center space-x-4 text-xs font-mono text-neutral-400">
                  <span className="flex items-center space-x-1">
                    <Clock className="w-3.5 h-3.5 text-neutral-500" />
                    <span>{result.latency_seconds}s</span>
                  </span>
                  <span className="flex items-center space-x-1">
                    <Layers className="w-3.5 h-3.5 text-neutral-500" />
                    <span className="uppercase">{result.model_type}</span>
                  </span>
                </div>
              </div>

              {/* Disclosed Quantity Adjustment Warning Banner */}
              {result.warning && (
                <div className="p-3 bg-amber-950/40 border border-amber-800/60 rounded-xl text-xs text-amber-200 flex items-center space-x-2">
                  <AlertCircle className="w-4 h-4 text-amber-400 shrink-0" />
                  <span>{result.warning}</span>
                </div>
              )}

              {/* Gallery Grid */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 pt-2">
                {result.images.map((img) => (
                  <div
                    key={img.index}
                    className="bg-neutral-950 rounded-xl border border-neutral-800 p-3 flex flex-col items-center group relative hover:border-neutral-700 transition-all"
                  >
                    {/* Render Image (Upscaled nearest-neighbor for crisp 28x28 pixels) */}
                    <img
                      src={img.image_b64}
                      alt={`Generated ${result.display_label}`}
                      className="w-28 h-28 object-contain rounded-lg bg-black border border-neutral-900 shadow-inner [image-rendering:pixelated]"
                    />

                    {/* Metadata & Confidence Badge */}
                    <div className="w-full mt-3 text-center space-y-1">
                      <div className="flex items-center justify-center space-x-1 text-[11px] font-medium text-neutral-300">
                        {img.category_match ? (
                          <CheckCircle2 className="w-3 h-3 text-emerald-400 shrink-0" />
                        ) : (
                          <AlertCircle className="w-3 h-3 text-amber-400 shrink-0" />
                        )}
                        <span className="truncate">{img.classified_as}</span>
                      </div>
                      <div className="text-[10px] font-mono text-neutral-500">
                        {(img.classifier_confidence * 100).toFixed(1)}% oracle match
                      </div>
                    </div>

                    {/* Download button */}
                    <button
                      onClick={() => handleDownloadImage(img.image_b64, `${result.parsed_label}_${img.index + 1}.png`)}
                      className="absolute top-2 right-2 p-1.5 rounded-md bg-neutral-900/90 text-neutral-400 hover:text-white hover:bg-neutral-800 opacity-0 group-hover:opacity-100 transition-all shadow"
                      title="Download PNG"
                    >
                      <Download className="w-3.5 h-3.5" />
                    </button>
                  </div>
                ))}
              </div>

              {/* CLIP Multimodal Alignment Card */}
              {result.clip_evaluation && (
                <div className="mt-4 pt-4 border-t border-neutral-800/80 bg-neutral-950/60 rounded-xl p-4 space-y-2">
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-semibold text-neutral-200">CLIP ViT-B/32 Multimodal Alignment</span>
                    <span className="font-mono text-indigo-400 font-bold bg-neutral-800/90 px-2 py-0.5 rounded border border-neutral-700">
                      Raw Cosine: {result.clip_evaluation.mean_target_similarity}
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-[11px] text-neutral-400">
                    <span>Uncalibrated Embedding Metric</span>
                    <span className="font-mono text-neutral-500 text-[10px]">Range: [-1.0, +1.0] (Empirical: ~0.15–0.35)</span>
                  </div>
                  <p className="text-[11px] text-neutral-500 leading-relaxed pt-1">
                    {result.clip_evaluation.limitations_note}
                  </p>
                </div>
              )}
            </div>
          )}

          {/* Empty State when no generation has run */}
          {!result && !error && (
            <div className="bg-neutral-900/50 border border-neutral-800 border-dashed rounded-2xl p-12 text-center flex flex-col items-center justify-center space-y-4">
              <div className="w-12 h-12 rounded-xl bg-neutral-800/80 flex items-center justify-center text-neutral-500">
                <Sparkles className="w-6 h-6" />
              </div>
              <div>
                <h3 className="text-base font-semibold text-neutral-300">Awaiting Synthesis Request</h3>
                <p className="text-xs text-neutral-500 max-w-sm mt-1">
                  Type a prompt or choose an example from the left to parse with Qwen2.5 and synthesize class-conditioned clothing with PyTorch.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
