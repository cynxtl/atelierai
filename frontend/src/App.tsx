import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { StudioPage } from './pages/StudioPage';
import { ComparePage } from './pages/ComparePage';
import { PerformancePage } from './pages/PerformancePage';
import { InsightsPage } from './pages/InsightsPage';
import { checkHealth } from './services/api';
import type { HealthResponse } from './types';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<string>('studio');
  const [health, setHealth] = useState<HealthResponse | null>(null);

  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const data = await checkHealth();
        setHealth(data);
      } catch (e) {
        console.warn('Backend not yet reachable on /api/health:', e);
      }
    };
    fetchHealth();
    const interval = setInterval(fetchHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100 flex flex-col font-sans selection:bg-indigo-500/30 selection:text-indigo-200">
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} health={health} />

      <main className="flex-1 pb-16">
        {activeTab === 'studio' && <StudioPage />}
        {activeTab === 'compare' && <ComparePage />}
        {activeTab === 'performance' && <PerformancePage />}
        {activeTab === 'insights' && <InsightsPage />}
      </main>

      <footer className="border-t border-neutral-900 bg-neutral-950 py-6 text-center text-xs text-neutral-600 font-mono">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>AtelierAI &bull; Text-to-Conditional-GAN Synthesis Studio</span>
          <span>Fashion-MNIST &bull; 28&times;28 Grayscale &bull; Zalando SE (MIT License)</span>
        </div>
      </footer>
    </div>
  );
};

export default App;
