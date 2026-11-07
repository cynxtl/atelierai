import React, { useState } from 'react';
import { BookOpen, Layers, Cpu, Network, ShieldCheck, Sparkles } from 'lucide-react';

export const InsightsPage: React.FC = () => {
  const [activeLayer, setActiveLayer] = useState<number>(1);

  const layers = [
    { id: 1, title: 'Layer I: Foundations', subtitle: 'Generative vs Discriminative & Latent Space', icon: Sparkles },
    { id: 2, title: 'Layer II: GANs & LLMs', subtitle: 'Min-Max Dynamics & Self-Attention', icon: Layers },
    { id: 3, title: 'Layer III: Multi-Model Architecture', subtitle: 'GPT, BERT, and Multimodal CLIP', icon: Network },
    { id: 4, title: 'Layer IV: Tools & Pipeline', subtitle: 'PyTorch, CUDA & Decoupled Inference', icon: Cpu },
    { id: 5, title: 'Layer V: Responsible AI', subtitle: 'Hallucination Mitigation & Licensing', icon: ShieldCheck },
  ];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Header */}
      <div>
        <h2 className="text-xl font-bold text-white tracking-tight flex items-center space-x-2">
          <BookOpen className="w-5 h-5 text-indigo-400" />
          <span>Model Insights & Theoretical Architecture</span>
        </h2>
        <p className="text-sm text-neutral-400 mt-1">
          In-depth architectural breakdown covering the five structural layers of the AtelierAI synthesis engine.
        </p>
      </div>

      {/* Layer Navigation Tabs */}
      <div className="grid grid-cols-1 sm:grid-cols-5 gap-2 bg-neutral-900 p-1.5 rounded-2xl border border-neutral-800">
        {layers.map((layer) => {
          const Icon = layer.icon;
          const isActive = activeLayer === layer.id;
          return (
            <button
              key={layer.id}
              onClick={() => setActiveLayer(layer.id)}
              className={`p-3 rounded-xl text-left transition-all ${
                isActive
                  ? 'bg-neutral-800 text-white shadow border border-neutral-700/80'
                  : 'text-neutral-400 hover:text-neutral-200 hover:bg-neutral-950/40'
              }`}
            >
              <div className="flex items-center space-x-2 mb-1">
                <Icon className={`w-4 h-4 ${isActive ? 'text-indigo-400' : 'text-neutral-500'}`} />
                <span className="font-semibold text-xs tracking-tight">{layer.title.split(':')[0]}</span>
              </div>
              <p className="text-[11px] truncate text-neutral-400">{layer.subtitle}</p>
            </button>
          );
        })}
      </div>

      {/* Layer Content View */}
      <div className="bg-neutral-900 border border-neutral-800 rounded-2xl p-6 lg:p-8 space-y-8">
        {/* Layer 1 */}
        {activeLayer === 1 && (
          <div className="space-y-6">
            <div className="border-b border-neutral-800 pb-4">
              <span className="text-xs uppercase font-mono text-indigo-400 font-bold">Layer I</span>
              <h3 className="text-lg font-bold text-white mt-1">Foundational Principles: Generative vs Discriminative Modeling</h3>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-neutral-950 p-5 rounded-xl border border-neutral-800 space-y-3">
                <h4 className="text-sm font-semibold text-indigo-300">Discriminative Modeling: \(P(Y|X)\)</h4>
                <p className="text-xs text-neutral-400 leading-relaxed">
                  Focuses on modeling the conditional probability of target class \(Y\) given input features \(X\).
                  In AtelierAI, this is exemplified by our <strong>Auxiliary CNN Classifier</strong> and <strong>Fine-Tuned DistilBERT</strong>,
                  which partition the input space with decision boundaries to identify whether an image is a sneaker or ankle boot.
                </p>
                <div className="text-[11px] font-mono bg-neutral-900 p-2.5 rounded-lg text-neutral-300 border border-neutral-800">
                  Input: Image (1×28×28) → Output: Class Logits (10)
                </div>
              </div>

              <div className="bg-neutral-950 p-5 rounded-xl border border-neutral-800 space-y-3">
                <h4 className="text-sm font-semibold text-violet-300">Generative Modeling: \(P(X|Y)\)</h4>
                <p className="text-xs text-neutral-400 leading-relaxed">
                  Models the underlying distribution of data features X conditioned on category label Y.
                  Our <strong>Conditional GAN</strong> maps random normal noise z ~ N(0, I) together with a class embedding into synthetic pixel space,
                  sampling realistic instances from the high-dimensional apparel manifold.
                </p>
                <div className="text-[11px] font-mono bg-neutral-900 p-2.5 rounded-lg text-neutral-300 border border-neutral-800">
                  Input: Noise \(z\) (100) + Class (1) → Output: Synthetic Image (1×28×28)
                </div>
              </div>
            </div>

            {/* Lifecycle Pipeline Diagram */}
            <div className="bg-neutral-950 p-6 rounded-xl border border-neutral-800 space-y-4">
              <h4 className="text-sm font-semibold text-white">End-to-End Synthesis Lifecycle</h4>
              <div className="grid grid-cols-1 sm:grid-cols-4 gap-3 text-center">
                <div className="p-3 bg-neutral-900 rounded-lg border border-neutral-800">
                  <span className="text-[10px] font-mono text-indigo-400 uppercase">1. Prompt Parsing</span>
                  <div className="text-xs font-semibold text-neutral-200 mt-1">Qwen2.5 / Schema</div>
                  <p className="text-[10px] text-neutral-500 mt-1">Extracts label & quantity</p>
                </div>
                <div className="p-3 bg-neutral-900 rounded-lg border border-neutral-800">
                  <span className="text-[10px] font-mono text-indigo-400 uppercase">2. Conditioning</span>
                  <div className="text-xs font-semibold text-neutral-200 mt-1">Embedding Vector</div>
                  <p className="text-[10px] text-neutral-500 mt-1">Concatenates with noise \(z\)</p>
                </div>
                <div className="p-3 bg-neutral-900 rounded-lg border border-neutral-800">
                  <span className="text-[10px] font-mono text-indigo-400 uppercase">3. Synthesis</span>
                  <div className="text-xs font-semibold text-neutral-200 mt-1">cGAN Generator</div>
                  <p className="text-[10px] text-neutral-500 mt-1">Deconvolution to 28×28</p>
                </div>
                <div className="p-3 bg-neutral-900 rounded-lg border border-neutral-800">
                  <span className="text-[10px] font-mono text-indigo-400 uppercase">4. Verification</span>
                  <div className="text-xs font-semibold text-neutral-200 mt-1">CLIP & CNN Oracle</div>
                  <p className="text-[10px] text-neutral-500 mt-1">Class fidelity & alignment</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Layer 2 */}
        {activeLayer === 2 && (
          <div className="space-y-6">
            <div className="border-b border-neutral-800 pb-4">
              <span className="text-xs uppercase font-mono text-indigo-400 font-bold">Layer II</span>
              <h3 className="text-lg font-bold text-white mt-1">GANs, VAEs, and Transformer Mechanics</h3>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-neutral-950 p-5 rounded-xl border border-neutral-800 space-y-3">
                <h4 className="text-sm font-semibold text-indigo-300">Conditional GAN Min-Max Optimization</h4>
                <p className="text-xs text-neutral-400 leading-relaxed">
                  The conditional GAN plays a two-player zero-sum game. The Discriminator D(x, y) aims to maximize log-likelihood of separating real pairs from generated pairs, while the Generator G(z, y) aims to fool D:
                </p>
                <div className="text-[11px] font-mono bg-neutral-900 p-2.5 rounded-lg text-neutral-300 border border-neutral-800 overflow-x-auto">
                  {'min_G max_D E[log D(x, y)] + E[log(1 - D(G(z, y), y))]'}
                </div>
              </div>

              <div className="bg-neutral-950 p-5 rounded-xl border border-neutral-800 space-y-3">
                <h4 className="text-sm font-semibold text-violet-300">Variational Autoencoder ELBO</h4>
                <p className="text-xs text-neutral-400 leading-relaxed">
                  The VAE approximates intractable posterior p(z|x) using a recognition network q(z|x), optimizing the Evidence Lower Bound:
                </p>
                <div className="text-[11px] font-mono bg-neutral-900 p-2.5 rounded-lg text-neutral-300 border border-neutral-800 overflow-x-auto">
                  {'L_ELBO = E_{q}[log p(x|z)] - \u03B2 * D_KL(q(z|x) || p(z))'}
                </div>
              </div>
            </div>

            <div className="bg-neutral-950 p-5 rounded-xl border border-neutral-800 space-y-2">
              <h4 className="text-sm font-semibold text-white">Transformer Attention Routing</h4>
              <p className="text-xs text-neutral-400 leading-relaxed">
                In Qwen2.5-0.5B-Instruct, scaled dot-product attention Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V allows the language model to dynamically connect user modifier words (e.g. "four", "denim", "warm") to their corresponding canonical Fashion-MNIST categories.
              </p>
            </div>
          </div>
        )}

        {/* Layer 3 */}
        {activeLayer === 3 && (
          <div className="space-y-6">
            <div className="border-b border-neutral-800 pb-4">
              <span className="text-xs uppercase font-mono text-indigo-400 font-bold">Layer III</span>
              <h3 className="text-lg font-bold text-white mt-1">Multi-Model Architecture: GPT, BERT, and Multimodal CLIP</h3>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-neutral-950 p-5 rounded-xl border border-neutral-800 space-y-3">
                <span className="text-[10px] uppercase font-mono text-indigo-400 font-bold">Causal LLM</span>
                <h4 className="text-sm font-semibold text-white">Qwen2.5-0.5B-Instruct</h4>
                <p className="text-xs text-neutral-400 leading-relaxed">
                  Autoregressive decoder-only Transformer. Intercepts conversational clothing requests, strips extraneous language, and returns structured JSON conforming to our Pydantic schema.
                </p>
              </div>

              <div className="bg-neutral-950 p-5 rounded-xl border border-neutral-800 space-y-3">
                <span className="text-[10px] uppercase font-mono text-indigo-400 font-bold">Discriminative NLP</span>
                <h4 className="text-sm font-semibold text-white">Fine-Tuned DistilBERT</h4>
                <p className="text-xs text-neutral-400 leading-relaxed">
                  Bidirectional encoder trained specifically for natural language classification into the 10 Fashion-MNIST classes, providing a lightweight discriminative baseline against prompt-based LLMs.
                </p>
              </div>

              <div className="bg-neutral-950 p-5 rounded-xl border border-neutral-800 space-y-3">
                <span className="text-[10px] uppercase font-mono text-indigo-400 font-bold">Multimodal Contrastive</span>
                <h4 className="text-sm font-semibold text-white">CLIP ViT-B/32</h4>
                <p className="text-xs text-neutral-400 leading-relaxed">
                  Jointly embeds 28×28 images (upscaled to 224×224 RGB) and text descriptions into a 512-dimensional hypersphere, measuring semantic cosine alignment without requiring retrained weights.
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Layer 4 */}
        {activeLayer === 4 && (
          <div className="space-y-6">
            <div className="border-b border-neutral-800 pb-4">
              <span className="text-xs uppercase font-mono text-indigo-400 font-bold">Layer IV</span>
              <h3 className="text-lg font-bold text-white mt-1">Production Architecture & Decoupled Execution</h3>
            </div>

            <p className="text-xs text-neutral-400 leading-relaxed">
              AtelierAI is architected with strict separation of concerns. The ML business logic in <code className="bg-neutral-950 px-1 py-0.5 rounded text-neutral-300">ml/inference/service.py</code> runs independently of web routing frameworks.
            </p>

            <div className="bg-neutral-950 p-5 rounded-xl border border-neutral-800 space-y-3 font-mono text-xs text-neutral-300">
              <div className="text-indigo-400 font-bold">// Decoupled Core Services</div>
              <div>• ml/models/cgan.py ➔ ConditionalGenerator & ConditionalDiscriminator</div>
              <div>• ml/models/vae.py ➔ ConditionalVAE & Reparameterization Trick</div>
              <div>• ml/models/llm_parser.py ➔ QwenClothingParser + Pydantic Schema</div>
              <div>• ml/models/classifier.py ➔ Auxiliary CNN Oracle</div>
              <div>• ml/models/clip_evaluator.py ➔ Cosine Multimodal Evaluator</div>
              <div className="pt-2 text-neutral-500">// Dual Deployment Adapters</div>
              <div className="text-emerald-400">├── backend/app/main.py (Local FastAPI Server)</div>
              <div className="text-violet-400">└── deployment/huggingface/app.py (Hosted Gradio ZeroGPU Space)</div>
            </div>
          </div>
        )}

        {/* Layer 5 */}
        {activeLayer === 5 && (
          <div className="space-y-6">
            <div className="border-b border-neutral-800 pb-4">
              <span className="text-xs uppercase font-mono text-indigo-400 font-bold">Layer V</span>
              <h3 className="text-lg font-bold text-white mt-1">Responsible AI, Guardrails, and Honest Boundaries</h3>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-neutral-950 p-5 rounded-xl border border-neutral-800 space-y-3">
                <h4 className="text-sm font-semibold text-white">Prompt Injection & Malicious Input Defense</h4>
                <p className="text-xs text-neutral-400 leading-relaxed">
                  Inputs undergo regex sanitization and length bounds checking before reaching model context. Requests containing system prompt override patterns (e.g. "ignore previous instructions") are safely blocked.
                </p>
              </div>

              <div className="bg-neutral-950 p-5 rounded-xl border border-neutral-800 space-y-3">
                <h4 className="text-sm font-semibold text-white">No False Promises / Honest Domain Scope</h4>
                <p className="text-xs text-neutral-400 leading-relaxed">
                  The interface explicitly informs the user that images are 28×28 grayscale silhouettes from Fashion-MNIST. If a user asks for unrepresented apparel (e.g. watches or swimwear), the system explicitly refuses rather than hallucinating an arbitrary class.
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
