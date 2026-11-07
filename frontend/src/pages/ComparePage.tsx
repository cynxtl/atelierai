import React, { useState } from 'react';
import { Sparkles, SplitSquareVertical, AlertCircle } from 'lucide-react';
import { compareFashion } from '../services/api';
import type { CompareResponse } from '../types';

export const ComparePage: React.FC = () => {
  const [prompt, setPrompt] = useState<string>('Show me four ankle boots.');
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<CompareResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleCompare = async () => {
    if (!prompt.trim()) return;
    setLoading(true);
    setError(null);

    try {
      const data = await compareFashion(prompt);
      if (!data.success) {
        setError(data.error || 'Failed to compare models.');
      } else {
        setResult(data);
      }
    } catch (err: any) {
      setError(err.message || 'Error communicating with backend.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Header */}
      <div>
        <h2 className="text-xl font-bold text-white tracking-tight flex items-center space-x-2">
          <SplitSquareVertical className="w-5 h-5 text-indigo-400" />
          <span>Conditional GAN vs Variational Autoencoder</span>
        </h2>
        <p className="text-sm text-neutral-400 mt-1">
          Direct side-by-side synthesis comparing adversarial generator dynamics against probabilistic latent autoencoding.
        </p>
      </div>

      {/* Comparison Input Bar */}
      <div className="bg-neutral-900 border border-neutral-800 rounded-2xl p-5 flex flex-col sm:flex-row gap-4 items-center">
        <div className="relative flex-1 w-full">
          <input
            type="text"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="e.g., Generate 4 dresses or sneakers..."
            className="w-full bg-neutral-950 border border-neutral-800 rounded-xl px-4 py-2.5 text-sm text-neutral-100 placeholder-neutral-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/50"
          />
        </div>
        <button
          onClick={handleCompare}
          disabled={loading || !prompt.trim()}
          className="w-full sm:w-auto px-6 py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl text-sm font-medium shadow-lg shadow-indigo-600/20 disabled:opacity-50 flex items-center justify-center space-x-2 transition-all shrink-0"
        >
          {loading ? (
            <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
          ) : (
            <>
              <Sparkles className="w-4 h-4" />
              <span>Compare Synthesis</span>
            </>
          )}
        </button>
      </div>

      {error && (
        <div className="bg-red-950/40 border border-red-800 rounded-xl p-4 text-xs text-red-300">
          {error}
        </div>
      )}

      {/* Side-by-Side Results Display */}
      {result?.success && (
        <div className="space-y-6">
          {result.warning && (
            <div className="p-3 bg-amber-950/40 border border-amber-800/60 rounded-xl text-xs text-amber-200 flex items-center space-x-2">
              <AlertCircle className="w-4 h-4 text-amber-400 shrink-0" />
              <span>{result.warning}</span>
            </div>
          )}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Left: Conditional GAN */}
          <div className="bg-neutral-900 border border-neutral-800 rounded-2xl p-6 space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-neutral-800">
              <div>
                <span className="text-xs uppercase font-mono px-2 py-0.5 rounded bg-indigo-950 text-indigo-300 border border-indigo-800 font-semibold">
                  Conditional GAN (cGAN)
                </span>
                <h3 className="text-sm font-bold text-white mt-1">Adversarial Generator</h3>
              </div>
              <span className="text-xs text-neutral-400 font-mono">
                {result.cgan?.latency_seconds}s
              </span>
            </div>

            <p className="text-xs text-neutral-400 leading-relaxed">
              Min-max game optimization between Generator and Discriminator yields high edge contrast and crisp pixel silhouettes.
            </p>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2">
              {result.cgan?.images.map((img) => (
                <div key={img.index} className="bg-neutral-950 rounded-xl border border-neutral-800 p-2 text-center">
                  <img
                    src={img.image_b64}
                    alt="cGAN generation"
                    className="w-24 h-24 object-contain rounded-lg mx-auto [image-rendering:pixelated]"
                  />
                  <div className="mt-2 text-[10px] font-mono text-neutral-400">
                    {(img.classifier_confidence * 100).toFixed(1)}% match
                  </div>
                </div>
              ))}
            </div>

            {result.cgan?.clip_evaluation && (
              <div className="text-xs bg-neutral-950 p-3 rounded-xl border border-neutral-800 text-neutral-400 space-y-1">
                <div className="flex justify-between font-semibold text-neutral-300">
                  <span>CLIP Cosine Similarity</span>
                  <span className="text-indigo-400 font-mono">
                    {result.cgan.clip_evaluation.mean_target_similarity}
                  </span>
                </div>
                <div className="text-[10px] text-neutral-500">Uncalibrated embedding metric (range [-1.0, 1.0])</div>
              </div>
            )}
          </div>

          {/* Right: Conditional VAE */}
          <div className="bg-neutral-900 border border-neutral-800 rounded-2xl p-6 space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-neutral-800">
              <div>
                <span className="text-xs uppercase font-mono px-2 py-0.5 rounded bg-violet-950 text-violet-300 border border-violet-800 font-semibold">
                  Conditional VAE (C-VAE)
                </span>
                <h3 className="text-sm font-bold text-white mt-1">Probabilistic Decoder</h3>
              </div>
              <span className="text-xs text-neutral-400 font-mono">
                {result.vae?.latency_seconds}s
              </span>
            </div>

            <p className="text-xs text-neutral-400 leading-relaxed">
              Continuous Gaussian prior sampling with reconstruction and KL divergence losses. Produces smoother but slightly blurrier averages.
            </p>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2">
              {result.vae?.images.map((img) => (
                <div key={img.index} className="bg-neutral-950 rounded-xl border border-neutral-800 p-2 text-center">
                  <img
                    src={img.image_b64}
                    alt="VAE generation"
                    className="w-24 h-24 object-contain rounded-lg mx-auto [image-rendering:pixelated]"
                  />
                  <div className="mt-2 text-[10px] font-mono text-neutral-400">
                    {(img.classifier_confidence * 100).toFixed(1)}% match
                  </div>
                </div>
              ))}
            </div>

            {result.vae?.clip_evaluation && (
              <div className="text-xs bg-neutral-950 p-3 rounded-xl border border-neutral-800 text-neutral-400 space-y-1">
                <div className="flex justify-between font-semibold text-neutral-300">
                  <span>CLIP Cosine Similarity</span>
                  <span className="text-violet-400 font-mono">
                    {result.vae.clip_evaluation.mean_target_similarity}
                  </span>
                </div>
                <div className="text-[10px] text-neutral-500">Uncalibrated embedding metric (range [-1.0, 1.0])</div>
              </div>
            )}
          </div>
        </div>
        </div>
      )}

      {/* Comparative Architecture Deep Dive Table */}
      <div className="bg-neutral-900 border border-neutral-800 rounded-2xl p-6 space-y-4">
        <h3 className="text-base font-bold text-white">Methodological Comparison Matrix</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead>
              <tr className="border-b border-neutral-800 text-neutral-400">
                <th className="py-3 px-4 font-semibold">Dimension</th>
                <th className="py-3 px-4 font-semibold text-indigo-400">Conditional GAN (cGAN)</th>
                <th className="py-3 px-4 font-semibold text-violet-400">Conditional VAE (C-VAE)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-neutral-800/60 text-neutral-300 font-normal">
              <tr>
                <td className="py-3 px-4 font-medium text-neutral-200">Objective Formulation</td>
                <td className="py-3 px-4">Min-max game: min_G max_D V(D, G)</td>
                <td className="py-3 px-4">Maximizing ELBO: E[log p(x|z)] - D_KL(q(z|x) || p(z))</td>
              </tr>
              <tr>
                <td className="py-3 px-4 font-medium text-neutral-200">Latent Space</td>
                <td className="py-3 px-4">Standard normal prior z ~ N(0, I) (100-dim)</td>
                <td className="py-3 px-4">Regularized continuous Gaussian z ~ q(z|x) (32-dim)</td>
              </tr>
              <tr>
                <td className="py-3 px-4 font-medium text-neutral-200">Visual Quality</td>
                <td className="py-3 px-4 text-emerald-400">Crisp, high-contrast silhouette boundaries</td>
                <td className="py-3 px-4 text-amber-300">Smooth, slightly blurred pixel reconstructions</td>
              </tr>
              <tr>
                <td className="py-3 px-4 font-medium text-neutral-200">Sampling Stability</td>
                <td className="py-3 px-4">Susceptible to mode collapse if D dominates</td>
                <td className="py-3 px-4 text-emerald-400">Stably covers the entire data support</td>
              </tr>
              <tr>
                <td className="py-3 px-4 font-medium text-neutral-200">Class Conditioning</td>
                <td className="py-3 px-4">Learned class embedding concatenated with latent noise</td>
                <td className="py-3 px-4">Spatial label map conditioning in both encoder and decoder</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
