'use client';

import React from 'react';
import { PredictionResult } from '../types';

interface RiskGaugeProps {
  result: PredictionResult;
}

export default function RiskGauge({ result }: RiskGaugeProps) {
  const { risk_score, risk_tier, percentile, bmi, waist_to_height_ratio } = result;

  const arcLength = 235.619;
  const clampedScore = Math.min(100, Math.max(0, risk_score));
  const strokeDashoffset = arcLength * (1 - clampedScore / 100);

  const getTierStyles = () => {
    switch (risk_tier) {
      case 'low':
        return {
          stroke: '#10b981',
          badge: 'bg-emerald-50 text-emerald-700 border-emerald-200',
          label: 'Low Risk',
        };
      case 'moderate':
        return {
          stroke: '#f59e0b',
          badge: 'bg-amber-50 text-amber-700 border-amber-200',
          label: 'Moderate Risk',
        };
      case 'high':
      default:
        return {
          stroke: '#e11d48',
          badge: 'bg-rose-50 text-rose-700 border-rose-200',
          label: 'Elevated Risk',
        };
    }
  };

  const style = getTierStyles();

  return (
    <article className="bg-white rounded-xl border border-slate-200 shadow-sm p-5 flex flex-col justify-between h-full">
      <div>
        {/* Top Header & Badge */}
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">Screening Result</span>
          <span className={`px-2 py-0.5 rounded text-xs font-semibold border ${style.badge}`}>
            {style.label}
          </span>
        </div>

        {/* Semi-Circle SVG Gauge */}
        <div className="flex flex-col items-center justify-center my-3 relative">
          <div className="w-60 h-36 relative flex items-end justify-center">
            <svg viewBox="0 0 200 120" className="w-60 h-36 overflow-visible">
              <path
                d="M 25 110 A 75 75 0 0 1 175 110"
                fill="none"
                stroke="#e2e8f0"
                strokeWidth="14"
                strokeLinecap="round"
              />
              <path
                d="M 25 110 A 75 75 0 0 1 175 110"
                fill="none"
                stroke={style.stroke}
                strokeWidth="14"
                strokeLinecap="round"
                strokeDasharray={arcLength}
                strokeDashoffset={strokeDashoffset}
                className="transition-all duration-700 ease-out"
              />
            </svg>

            {/* Score */}
            <div className="absolute bottom-2 flex flex-col items-center text-center">
              <div className="flex items-baseline justify-center">
                <span className="text-4xl font-extrabold text-slate-900 tracking-tight">{risk_score.toFixed(1)}</span>
                <span className="text-lg font-bold text-slate-400 ml-0.5">%</span>
              </div>
              <span className="text-[11px] text-slate-500 font-medium">Estimated Risk Score</span>
            </div>
          </div>

          <div className="w-56 flex justify-between text-[10px] font-medium text-slate-400 mt-1 px-1">
            <span>0% (Low)</span>
            <span>50%</span>
            <span>100% (High)</span>
          </div>
        </div>

        {/* Population Percentile Banner */}
        <div className="bg-slate-50 rounded-lg px-3 py-2 border border-slate-100 text-center">
          <p className="text-xs text-slate-600">
            Exceeds <strong className="font-semibold text-slate-900">{percentile}%</strong> of the adult population sample.
          </p>
        </div>
      </div>

      {/* Anthropometric Indicators */}
      <div className="grid grid-cols-2 gap-2.5 pt-3.5 mt-3.5 border-t border-slate-100 text-center">
        <div className="bg-slate-50 rounded-lg p-2">
          <span className="text-[10px] text-slate-400 block">BMI</span>
          <span className="text-sm font-bold text-slate-800">{bmi} <span className="text-[10px] font-normal text-slate-500">kg/m²</span></span>
          <span className={`block text-[10px] font-semibold mt-0.5 ${bmi >= 25 ? 'text-amber-600' : 'text-emerald-600'}`}>
            {bmi >= 30 ? 'Obese' : bmi >= 25 ? 'Overweight' : 'Normal'}
          </span>
        </div>

        <div className="bg-slate-50 rounded-lg p-2">
          <span className="text-[10px] text-slate-400 block">Waist / Height</span>
          <span className="text-sm font-bold text-slate-800">{waist_to_height_ratio}</span>
          <span className={`block text-[10px] font-semibold mt-0.5 ${waist_to_height_ratio > 0.5 ? 'text-amber-600' : 'text-emerald-600'}`}>
            {waist_to_height_ratio > 0.5 ? 'Elevated' : 'Optimal'}
          </span>
        </div>
      </div>
    </article>
  );
}
