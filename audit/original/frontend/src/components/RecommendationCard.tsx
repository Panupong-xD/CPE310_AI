'use client';

import React, { useState } from 'react';
import { PredictionResult } from '../types';
import { ChevronDown, ChevronUp } from 'lucide-react';

interface RecommendationCardProps {
  result: PredictionResult;
}

export default function RecommendationCard({ result }: RecommendationCardProps) {
  const [showDisclaimer, setShowDisclaimer] = useState(false);
  const { waist_to_height_ratio, key_drivers } = result;

  const driverNames = key_drivers.map((d) => d.raw_feature_name);
  const hasWaistIssue = driverNames.includes('waist_cm') || driverNames.includes('waist_to_height_ratio') || waist_to_height_ratio > 0.5;
  const hasSugarIssue = driverNames.includes('daily_sugar_g') || driverNames.includes('sugar_to_fiber_ratio');

  return (
    <div className="space-y-3">
      {/* Action Checklist */}
      <section className="bg-white rounded-xl border border-slate-200 shadow-sm p-5 space-y-3">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500">
          Lifestyle Recommendations
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {/* Nutrition */}
          <div className="p-3 rounded-lg bg-slate-50 border border-slate-100 space-y-1.5">
            <span className="text-xs font-semibold text-slate-800 block">
              Nutrition &amp; Diet
            </span>
            <ul className="space-y-1 text-xs text-slate-600">
              <li className="flex items-start gap-1.5">
                <span className="text-slate-400">•</span>
                <span>
                  {hasSugarIssue
                    ? 'Limit sugar-sweetened beverages and refined carbohydrates.'
                    : 'Keep added sugar intake within recommended limits (<35g/day).'}
                </span>
              </li>
              <li className="flex items-start gap-1.5">
                <span className="text-slate-400">•</span>
                <span>Target 25–30g daily fiber via whole grains, legumes, and vegetables.</span>
              </li>
            </ul>
          </div>

          {/* Activity */}
          <div className="p-3 rounded-lg bg-slate-50 border border-slate-100 space-y-1.5">
            <span className="text-xs font-semibold text-slate-800 block">
              Physical Activity
            </span>
            <ul className="space-y-1 text-xs text-slate-600">
              <li className="flex items-start gap-1.5">
                <span className="text-slate-400">•</span>
                <span>
                  {hasWaistIssue
                    ? '150 min/week moderate aerobic exercise with resistance training.'
                    : 'Maintain regular daily movement (30+ minutes walking/sports).'}
                </span>
              </li>
              <li className="flex items-start gap-1.5">
                <span className="text-slate-400">•</span>
                <span>Short 10-minute post-meal walks help stabilize glucose response.</span>
              </li>
            </ul>
          </div>
        </div>
      </section>

      {/* Collapsible Clinical Disclaimer */}
      <div className="rounded-lg border border-slate-200 bg-white overflow-hidden">
        <button
          type="button"
          onClick={() => setShowDisclaimer(!showDisclaimer)}
          className="w-full px-3.5 py-2 flex items-center justify-between text-left text-xs text-slate-500 hover:text-slate-700 transition-colors cursor-pointer"
        >
          <span>Clinical Information &amp; Disclaimer</span>
          {showDisclaimer ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
        </button>

        {showDisclaimer && (
          <div className="px-3.5 pb-3 pt-1 text-[11px] text-slate-500 leading-relaxed border-t border-slate-100">
            This application provides statistical risk estimates derived from NHANES population health surveys. It is intended for screening and educational purposes only, not clinical diagnosis or treatment. Consult a licensed medical practitioner for formal laboratory blood glucose and HbA1c testing.
          </div>
        )}
      </div>
    </div>
  );
}
