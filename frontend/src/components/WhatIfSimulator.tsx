'use client';

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { AssessmentFormData, SimulationAdjustments, SimulationResult } from '../types';
import { runSimulation } from '../lib/api';

interface WhatIfSimulatorProps {
  baseline: AssessmentFormData;
  initialScore: number;
}

export default function WhatIfSimulator({ baseline, initialScore }: WhatIfSimulatorProps) {
  const [adjustments, setAdjustments] = useState<SimulationAdjustments>({
    delta_weight_kg: 0,
    delta_waist_cm: 0,
    delta_sugar_pct: 0,
    delta_fiber_g: 0,
    active_exercise: baseline.vigorous_activity,
    manage_hypertension: false,
  });

  const [simulation, setSimulation] = useState<SimulationResult | null>(null);
  const [isSimulating, setIsSimulating] = useState(false);
  const [simulationError, setSimulationError] = useState<string | null>(null);
  const latestRequest = useRef(0);

  const executeSimulation = useCallback(async (currentAdjustments: SimulationAdjustments) => {
    const requestId = ++latestRequest.current;
    setIsSimulating(true);
    setSimulationError(null);
    try {
      const result = await runSimulation(baseline, currentAdjustments);
      if (requestId === latestRequest.current) setSimulation(result);
    } catch (err: any) {
      if (requestId === latestRequest.current) setSimulationError(err.message || 'Simulation failed.');
    } finally {
      if (requestId === latestRequest.current) setIsSimulating(false);
    }
  }, [baseline]);

  useEffect(() => {
    const reset: SimulationAdjustments = {
      delta_weight_kg: 0, delta_waist_cm: 0, delta_sugar_pct: 0, delta_fiber_g: 0,
      active_exercise: baseline.vigorous_activity, manage_hypertension: false,
    };
    setAdjustments(reset);
    executeSimulation(reset);
    return () => { latestRequest.current += 1; };
  }, [baseline, executeSimulation]);

  const handleSliderChange = (field: keyof SimulationAdjustments, value: any) => {
    const nextAdjustments = { ...adjustments, [field]: value };
    setAdjustments(nextAdjustments);
    executeSimulation(nextAdjustments);
  };

  const handleReset = () => {
    const zeroAdjustments: SimulationAdjustments = {
      delta_weight_kg: 0,
      delta_waist_cm: 0,
      delta_sugar_pct: 0,
      delta_fiber_g: 0,
      active_exercise: baseline.vigorous_activity,
      manage_hypertension: false,
    };
    setAdjustments(zeroAdjustments);
    executeSimulation(zeroAdjustments);
  };

  return (
    <section className="bg-white rounded-xl border border-slate-200 shadow-sm p-5 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between gap-3 border-b border-slate-100 pb-2.5">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500">
          Model Input Scenario Simulator
        </h3>
        <button
          type="button"
          onClick={handleReset}
          className="text-xs text-slate-500 hover:text-slate-800 underline underline-offset-2 cursor-pointer"
        >
          Reset to Baseline
        </button>
      </div>

      {/* Simulated Outcome Banner (Light Theme) */}
      {simulation && (
        <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <span className="text-[10px] uppercase font-bold tracking-wider text-slate-400">
              Scenario Model Score
            </span>
            <div className="flex items-baseline gap-2.5 mt-0.5">
              <span className="text-3xl font-extrabold tracking-tight text-slate-900">
                {simulation.new_risk_score.toFixed(1)}%
              </span>
              <span className="text-xs text-slate-500">
                (Baseline: {simulation.original_risk_score.toFixed(1)}%)
              </span>
            </div>
            <div className="flex items-center gap-2 mt-1.5 text-xs text-slate-600">
              <span className="px-2 py-0.5 rounded bg-white border border-slate-200 text-slate-700 text-[11px] font-medium">
                {simulation.new_category}
              </span>
              <span>
                BMI: <strong className="text-slate-800">{simulation.simulated_bmi}</strong> | WHtR: <strong className="text-slate-800">{simulation.simulated_waist_to_height_ratio}</strong>
              </span>
            </div>
          </div>

          {simulation.delta_risk_score > 0 ? (
            <div className="bg-emerald-50 border border-emerald-200 rounded-lg px-3.5 py-2 text-left sm:text-right shrink-0">
              <span className="text-emerald-700 text-xs font-bold block">
                -{simulation.delta_risk_score.toFixed(1)} pp Lower Model Score
              </span>
              <span className="text-[11px] text-emerald-600">
                ({simulation.delta_percentage.toFixed(1)}% relative decrease)
              </span>
            </div>
          ) : simulation.delta_risk_score < 0 ? (
            <div className="bg-rose-50 border border-rose-200 rounded-lg px-3.5 py-2 text-left sm:text-right shrink-0">
              <span className="text-rose-700 text-xs font-bold block">
                +{Math.abs(simulation.delta_risk_score).toFixed(1)} pp Higher Model Score
              </span>
            </div>
          ) : (
            <div className="bg-white border border-slate-200 rounded-lg px-3.5 py-2 text-left sm:text-right shrink-0">
              <span className="text-xs text-slate-600 font-medium block">Baseline Values</span>
              <span className="text-[11px] text-slate-400">Adjust sliders to test changes</span>
            </div>
          )}
        </div>
      )}

      <p className="text-[11px] text-slate-500">{simulation?.summary_takeaway || 'Changes show model associations, not causal treatment outcomes.'}</p>
      {isSimulating && <p className="text-xs text-slate-500">Updating scenario...</p>}
      {simulationError && <p role="alert" className="text-xs text-rose-700">{simulationError}</p>}
      {/* Sliders */}
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3">
        {/* Waist */}
        <div className="p-3 rounded-lg bg-slate-50 border border-slate-100 space-y-1">
          <div className="flex justify-between text-xs">
            <span className="text-slate-600">Waist Trim</span>
            <span className="font-semibold text-slate-800">
              {adjustments.delta_waist_cm < 0 ? `${adjustments.delta_waist_cm} cm` : '0'}
            </span>
          </div>
          <input
            type="range"
            min="-15"
            max="0"
            step="1"
            value={adjustments.delta_waist_cm}
            onChange={(e) => handleSliderChange('delta_waist_cm', Number(e.target.value))}
            className="w-full h-1 bg-slate-300 rounded appearance-none cursor-pointer accent-slate-800"
          />
        </div>

        {/* Weight */}
        <div className="p-3 rounded-lg bg-slate-50 border border-slate-100 space-y-1">
          <div className="flex justify-between text-xs">
            <span className="text-slate-600">Weight Loss</span>
            <span className="font-semibold text-slate-800">
              {adjustments.delta_weight_kg < 0 ? `${adjustments.delta_weight_kg} kg` : '0'}
            </span>
          </div>
          <input
            type="range"
            min="-15"
            max="0"
            step="1"
            value={adjustments.delta_weight_kg}
            onChange={(e) => handleSliderChange('delta_weight_kg', Number(e.target.value))}
            className="w-full h-1 bg-slate-300 rounded appearance-none cursor-pointer accent-slate-800"
          />
        </div>

        {/* Sugar */}
        <div className="p-3 rounded-lg bg-slate-50 border border-slate-100 space-y-1">
          <div className="flex justify-between text-xs">
            <span className="text-slate-600">Sugar Cut</span>
            <span className="font-semibold text-slate-800">
              {adjustments.delta_sugar_pct < 0 ? `${adjustments.delta_sugar_pct}%` : '0'}
            </span>
          </div>
          <input
            type="range"
            min="-80"
            max="0"
            step="5"
            value={adjustments.delta_sugar_pct}
            onChange={(e) => handleSliderChange('delta_sugar_pct', Number(e.target.value))}
            className="w-full h-1 bg-slate-300 rounded appearance-none cursor-pointer accent-slate-800"
          />
        </div>

        {/* Fiber */}
        <div className="p-3 rounded-lg bg-slate-50 border border-slate-100 space-y-1">
          <div className="flex justify-between text-xs">
            <span className="text-slate-600">Fiber Boost</span>
            <span className="font-semibold text-slate-800">
              {adjustments.delta_fiber_g > 0 ? `+${adjustments.delta_fiber_g}g` : '0'}
            </span>
          </div>
          <input
            type="range"
            min="0"
            max="25"
            step="1"
            value={adjustments.delta_fiber_g}
            onChange={(e) => handleSliderChange('delta_fiber_g', Number(e.target.value))}
            className="w-full h-1 bg-slate-300 rounded appearance-none cursor-pointer accent-slate-800"
          />
        </div>
      </div>

      {/* Checkboxes */}
      <div className="flex flex-wrap items-center gap-4 pt-1 text-xs">
        <label className="inline-flex items-center gap-1.5 cursor-pointer text-slate-700">
          <input
            type="checkbox"
            checked={adjustments.active_exercise}
            onChange={(e) => handleSliderChange('active_exercise', e.target.checked)}
            className="w-3.5 h-3.5 rounded border-slate-300 text-slate-800 focus:ring-0"
          />
          <span>Any vigorous recreation (10+ continuous min)</span>
        </label>

        <label className="inline-flex items-center gap-1.5 cursor-pointer text-slate-700">
          <input
            type="checkbox"
            checked={adjustments.manage_hypertension}
            onChange={(e) => handleSliderChange('manage_hypertension', e.target.checked)}
            className="w-3.5 h-3.5 rounded border-slate-300 text-slate-800 focus:ring-0"
          />
          <span>Set measured BP to at most 120/80; retain history</span>
        </label>
      </div>
    </section>
  );
}
