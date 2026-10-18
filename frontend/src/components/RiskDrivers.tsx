'use client';

import React, { useState } from 'react';
import { FeatureImpact } from '../types';
import { ChevronDown, ChevronUp } from 'lucide-react';

interface RiskDriversProps {
  keyDrivers: FeatureImpact[];
  protectiveFactors: FeatureImpact[];
}

export default function RiskDrivers({ keyDrivers, protectiveFactors }: RiskDriversProps) {
  const [showDetails, setShowDetails] = useState(false);

  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5 flex flex-col justify-between h-full space-y-4">
      <p className="text-[11px] text-slate-500">SHAP values use percentage points (pp) relative to the model background. They do not show causal effects. Only the largest contributions are shown.</p>
      <div className="space-y-4">
        {/* Top Header */}
        <div className="flex items-center justify-between gap-2 border-b border-slate-100 pb-2.5">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500">
            Key Contributing Factors
          </h3>
          <button
            type="button"
            onClick={() => setShowDetails(!showDetails)}
            className="text-[11px] font-medium text-slate-500 hover:text-slate-900 inline-flex items-center gap-1 cursor-pointer"
          >
            <span>{showDetails ? 'Hide details' : 'Show details'}</span>
            {showDetails ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
          </button>
        </div>

        {/* 1. Risk Factors */}
        <div>
          <span className="text-xs font-medium text-slate-700 block mb-2">
            Score-Increasing Associations
          </span>

          {keyDrivers && keyDrivers.length > 0 ? (
            <div className="space-y-1.5">
              {keyDrivers.map((driver, idx) => (
                <div
                  key={idx}
                  className="p-2.5 rounded-lg bg-slate-50 border border-slate-100"
                >
                  <div className="flex items-center justify-between gap-2">
                    <div className="flex items-baseline gap-1.5">
                      <span className="text-xs font-medium text-slate-800">{driver.feature}</span>
                      <span className="text-[11px] text-slate-400">({driver.user_value})</span>
                    </div>
                    <span className="text-xs font-semibold text-rose-700 bg-rose-50 border border-rose-200 px-1.5 py-0.5 rounded">
                      {driver.impact}
                    </span>
                  </div>

                  {showDetails && (
                    <p className="text-[11px] text-slate-600 mt-1.5 pt-1 border-t border-slate-200 leading-relaxed">
                      {driver.advice}
                    </p>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <p className="text-xs text-slate-400 italic">None identified.</p>
          )}
        </div>

        {/* 2. Score-Decreasing Associations */}
        <div>
          <span className="text-xs font-medium text-slate-700 block mb-2">
            Score-Decreasing Associations
          </span>

          {protectiveFactors && protectiveFactors.length > 0 ? (
            <div className="space-y-1.5">
              {protectiveFactors.map((factor, idx) => (
                <div
                  key={idx}
                  className="p-2.5 rounded-lg bg-emerald-50/40 border border-emerald-100"
                >
                  <div className="flex items-center justify-between gap-2">
                    <div className="flex items-baseline gap-1.5">
                      <span className="text-xs font-medium text-slate-800">{factor.feature}</span>
                      <span className="text-[11px] text-slate-400">({factor.user_value})</span>
                    </div>
                    <span className="text-xs font-semibold text-emerald-700 bg-emerald-100 px-1.5 py-0.5 rounded">
                      {factor.impact}
                    </span>
                  </div>

                  {showDetails && (
                    <p className="text-[11px] text-slate-600 mt-1.5 pt-1 border-t border-emerald-100 leading-relaxed">
                      {factor.advice}
                    </p>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <p className="text-xs text-slate-400 italic">None identified.</p>
          )}
        </div>
      </div>
    </div>
  );
}
