import React from 'react';
import { Sparkles, Layers, BarChart3, BookOpen, Cpu } from 'lucide-react';
import type { HealthResponse } from '../types';

interface NavbarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  health: HealthResponse | null;
}

export const Navbar: React.FC<NavbarProps> = ({ activeTab, setActiveTab, health }) => {
  const navItems = [
    { id: 'studio', label: 'Studio', icon: Sparkles },
    { id: 'compare', label: 'Model Comparison', icon: Layers },
    { id: 'performance', label: 'Performance & Benchmarks', icon: BarChart3 },
    { id: 'insights', label: 'Model Insights', icon: BookOpen },
  ];

  return (
    <header className="border-b border-neutral-800 bg-neutral-950/80 backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Brand identity */}
        <div className="cursor-pointer" onClick={() => setActiveTab('studio')}>
          <div className="flex items-center space-x-2">
            <span className="font-semibold text-lg tracking-tight text-white">AtelierAI</span>
            <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-neutral-800 text-neutral-400 border border-neutral-700">
              Studio
            </span>
          </div>
          <p className="text-[11px] text-neutral-400 font-medium">Text-to-Conditional-Fashion Synthesis</p>
        </div>

        {/* Navigation Tabs */}
        <nav className="flex items-center space-x-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`flex items-center space-x-2 px-3.5 py-2 rounded-lg text-sm font-medium transition-all duration-150 ${
                  isActive
                    ? 'bg-neutral-800 text-white shadow-sm border border-neutral-700/60'
                    : 'text-neutral-400 hover:text-neutral-200 hover:bg-neutral-900'
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? 'text-indigo-400' : 'text-neutral-400'}`} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        {/* System Health / Device Indicator */}
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2 text-xs font-mono px-3 py-1.5 rounded-full bg-neutral-900 border border-neutral-800 text-neutral-300">
            <span className={`w-2 h-2 rounded-full ${health?.status === 'healthy' ? 'bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.5)]' : 'bg-amber-400'}`} />
            <Cpu className="w-3.5 h-3.5 text-neutral-400" />
            <span>{health?.gpu_name || health?.device?.toUpperCase() || 'CONNECTING...'}</span>
          </div>
        </div>
      </div>
    </header>
  );
};
