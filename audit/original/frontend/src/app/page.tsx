'use client';

import React, { useState } from 'react';
import { AssessmentFormData, PredictionResult } from '../types';
import { submitAssessment } from '../lib/api';
import AssessmentForm from '../components/AssessmentForm';
import RiskGauge from '../components/RiskGauge';
import RiskDrivers from '../components/RiskDrivers';
import WhatIfSimulator from '../components/WhatIfSimulator';
import RecommendationCard from '../components/RecommendationCard';

export default function HomePage() {
  const [lastSubmittedData, setLastSubmittedData] = useState<AssessmentFormData | null>(null);
  const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleAssessmentSubmit = async (data: AssessmentFormData) => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await submitAssessment(data);
      setPredictionResult(res);
      setLastSubmittedData(data);
      setTimeout(() => {
        const resultsEl = document.getElementById('assessment-results');
        if (resultsEl) {
          resultsEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      }, 100);
    } catch (err: any) {
      setError(err.message || 'Failed to complete risk calculation. Please verify that the backend API is active.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setPredictionResult(null);
    setLastSubmittedData(null);
    setError(null);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <main className="min-h-screen bg-slate-50 text-slate-900 pb-16">
      {/* Top Banner & Header */}
      <header className="border-b border-slate-200 bg-white sticky top-0 z-30">
        <div className="max-w-4xl mx-auto px-4 py-3 flex items-center justify-between">
          <div>
            <h1 className="text-base font-semibold text-slate-900 tracking-tight">
              Diabetes Risk Assessment
            </h1>
            <p className="text-xs text-slate-500">NHANES Population Health Model</p>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <div className="max-w-4xl mx-auto px-4 pt-6 space-y-6">
        {/* Intro */}
        <section className="space-y-1">
          <h2 className="text-xl font-bold text-slate-900 tracking-tight">
            Non-Invasive Health Risk Screening
          </h2>
          <p className="text-xs text-slate-600 leading-relaxed max-w-2xl">
            Calculates diabetes risk using standard demographic, body measurement, and lifestyle inputs.
          </p>
        </section>

        {/* Error notification */}
        {error && (
          <div className="p-3 rounded-lg bg-rose-50 border border-rose-200 text-rose-800 text-xs">
            <span className="font-semibold">Unable to complete assessment: </span>
            <span>{error}</span>
          </div>
        )}

        {/* Assessment Form Card */}
        <AssessmentForm onSubmit={handleAssessmentSubmit} isLoading={isLoading} />

        {/* Assessment Results Section */}
        {predictionResult && lastSubmittedData && (
          <div id="assessment-results" className="pt-2 space-y-6">
            {/* Section Divider */}
            <div className="flex items-center justify-between border-t border-slate-200 pt-4">
              <h2 className="text-sm font-bold uppercase tracking-wider text-slate-700">Assessment Results</h2>
              <button
                type="button"
                onClick={handleReset}
                className="text-xs text-slate-600 hover:text-slate-900 font-medium underline underline-offset-2 cursor-pointer"
              >
                Reset Assessment
              </button>
            </div>

            {/* Dashboard: Gauge + Drivers */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              <div className="lg:col-span-5">
                <RiskGauge result={predictionResult} />
              </div>
              <div className="lg:col-span-7">
                <RiskDrivers
                  keyDrivers={predictionResult.key_drivers}
                  protectiveFactors={predictionResult.protective_factors}
                />
              </div>
            </div>

            {/* What-If Simulator */}
            <WhatIfSimulator
              baseline={lastSubmittedData}
              initialScore={predictionResult.risk_score}
            />

            {/* Action Plan & Medical Disclaimer */}
            <RecommendationCard result={predictionResult} />
          </div>
        )}
      </div>
    </main>
  );
}
