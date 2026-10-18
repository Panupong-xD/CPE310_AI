'use client';

import React, { useState } from 'react';
import { AssessmentFormData, BiologicalSex } from '../types';
import { ArrowRight, ArrowLeft, Check } from 'lucide-react';

interface AssessmentFormProps {
  onSubmit: (data: AssessmentFormData) => void;
  isLoading: boolean;
}

interface FormState {
  age: string;
  gender: BiologicalSex;
  weight_kg: string;
  height_cm: string;
  waist_cm: string;
  systolic_bp: string;
  diastolic_bp: string;
  daily_sugar_g: string;
  daily_fiber_g: string;
  daily_calories_kcal: string;
  hypertension_history: boolean;
  smoking_history: boolean;
  vigorous_activity: boolean;
  sleep_trouble: boolean;
}

const PRESETS: Record<string, { label: string; data: FormState }> = {
  healthy: {
    label: 'Profile A (Low Risk)',
    data: {
      age: '28',
      gender: 'female',
      weight_kg: '56',
      height_cm: '168',
      waist_cm: '72',
      systolic_bp: '112',
      diastolic_bp: '72',
      daily_sugar_g: '35',
      daily_fiber_g: '28',
      daily_calories_kcal: '1950',
      hypertension_history: false,
      smoking_history: false,
      vigorous_activity: true,
      sleep_trouble: false,
    },
  },
  moderate: {
    label: 'Profile B (Moderate)',
    data: {
      age: '45',
      gender: 'male',
      weight_kg: '82',
      height_cm: '174',
      waist_cm: '94',
      systolic_bp: '128',
      diastolic_bp: '82',
      daily_sugar_g: '85',
      daily_fiber_g: '14',
      daily_calories_kcal: '2200',
      hypertension_history: false,
      smoking_history: false,
      vigorous_activity: false,
      sleep_trouble: true,
    },
  },
  highRisk: {
    label: 'Profile C (Elevated)',
    data: {
      age: '62',
      gender: 'male',
      weight_kg: '98',
      height_cm: '171',
      waist_cm: '112',
      systolic_bp: '148',
      diastolic_bp: '92',
      daily_sugar_g: '135',
      daily_fiber_g: '9',
      daily_calories_kcal: '2600',
      hypertension_history: true,
      smoking_history: true,
      vigorous_activity: false,
      sleep_trouble: true,
    },
  },
};

export default function AssessmentForm({ onSubmit, isLoading }: AssessmentFormProps) {
  const [step, setStep] = useState<1 | 2>(1);
  const [formData, setFormData] = useState<FormState>({
    age: '42',
    gender: 'male',
    weight_kg: '78',
    height_cm: '172',
    waist_cm: '92',
    systolic_bp: '125',
    diastolic_bp: '80',
    daily_sugar_g: '75',
    daily_fiber_g: '15',
    daily_calories_kcal: '2100',
    hypertension_history: false,
    smoking_history: false,
    vigorous_activity: false,
    sleep_trouble: false,
  });

  const [formErrors, setFormErrors] = useState<Record<string, string>>({});

  const calculateBMI = () => {
    const w = parseFloat(formData.weight_kg);
    const h = parseFloat(formData.height_cm);
    if (!isNaN(w) && !isNaN(h) && h > 0) {
      const hM = h / 100;
      return (w / (hM * hM)).toFixed(1);
    }
    return '-';
  };

  const calculateWHtR = () => {
    const waist = parseFloat(formData.waist_cm);
    const h = parseFloat(formData.height_cm);
    if (!isNaN(waist) && !isNaN(h) && h > 0) {
      return (waist / h).toFixed(2);
    }
    return '-';
  };

  const handlePresetSelect = (presetKey: string) => {
    const preset = PRESETS[presetKey];
    if (preset) {
      setFormData(preset.data);
      setFormErrors({});
    }
  };

  const validateStep1 = (): boolean => {
    const errors: Record<string, string> = {};
    const age = parseFloat(formData.age);
    const weight = parseFloat(formData.weight_kg);
    const height = parseFloat(formData.height_cm);
    const waist = formData.waist_cm.trim() !== '' ? parseFloat(formData.waist_cm) : null;

    if (isNaN(age) || age < 18 || age > 110) errors.age = 'Age must be 18–110';
    if (isNaN(weight) || weight < 30 || weight > 220) errors.weight_kg = 'Weight must be 30–220 kg';
    if (isNaN(height) || height < 110 || height > 230) errors.height_cm = 'Height must be 110–230 cm';
    if (waist !== null && (isNaN(waist) || waist < 45 || waist > 170)) errors.waist_cm = 'Waist must be 45–170 cm';

    setFormErrors(errors);
    return Object.keys(errors).length === 0;
  };

  const handleGoToStep1 = (e?: React.MouseEvent) => {
    if (e) e.preventDefault();
    setStep(1);
  };

  const handleGoToStep2 = (e?: React.FormEvent | React.MouseEvent) => {
    if (e) e.preventDefault();
    if (validateStep1()) {
      setStep(2);
    }
  };

  const handleFinalSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const payload: AssessmentFormData = {
      age: parseFloat(formData.age) || 40,
      gender: formData.gender,
      weight_kg: parseFloat(formData.weight_kg) || 70,
      height_cm: parseFloat(formData.height_cm) || 170,
      waist_cm: formData.waist_cm.trim() !== '' ? parseFloat(formData.waist_cm) : undefined,
      systolic_bp: formData.systolic_bp.trim() !== '' ? parseFloat(formData.systolic_bp) : undefined,
      diastolic_bp: formData.diastolic_bp.trim() !== '' ? parseFloat(formData.diastolic_bp) : undefined,
      daily_sugar_g: formData.daily_sugar_g.trim() !== '' ? parseFloat(formData.daily_sugar_g) : undefined,
      daily_fiber_g: formData.daily_fiber_g.trim() !== '' ? parseFloat(formData.daily_fiber_g) : undefined,
      daily_calories_kcal: formData.daily_calories_kcal.trim() !== '' ? parseFloat(formData.daily_calories_kcal) : undefined,
      hypertension_history: formData.hypertension_history,
      smoking_history: formData.smoking_history,
      vigorous_activity: formData.vigorous_activity,
      sleep_trouble: formData.sleep_trouble,
    };
    onSubmit(payload);
  };

  const bmiStr = calculateBMI();
  const bmiVal = parseFloat(bmiStr);
  const whtrStr = calculateWHtR();
  const whtrVal = parseFloat(whtrStr);

  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
      {/* Top Bar with Step Tabs & Presets */}
      <div className="px-5 py-3.5 border-b border-slate-200 bg-slate-50 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        {/* Step Tabs */}
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={handleGoToStep1}
            className="flex items-center gap-1.5 cursor-pointer text-left"
          >
            <span className={`inline-flex items-center justify-center w-5 h-5 rounded text-xs font-bold ${
              step === 1 ? 'bg-slate-800 text-white' : 'bg-slate-200 text-slate-700'
            }`}>1</span>
            <span className={`text-xs font-medium ${step === 1 ? 'text-slate-900 font-semibold' : 'text-slate-500'}`}>
              Body Metrics
            </span>
          </button>

          <span className="text-slate-300 font-light">/</span>

          <button
            type="button"
            onClick={handleGoToStep2}
            className="flex items-center gap-1.5 cursor-pointer text-left"
          >
            <span className={`inline-flex items-center justify-center w-5 h-5 rounded text-xs font-bold ${
              step === 2 ? 'bg-slate-800 text-white' : 'bg-slate-200 text-slate-700'
            }`}>2</span>
            <span className={`text-xs font-medium ${step === 2 ? 'text-slate-900 font-semibold' : 'text-slate-500'}`}>
              Lifestyle &amp; Vitals
            </span>
          </button>
        </div>

        {/* Quick Presets */}
        <div className="flex items-center gap-1.5 flex-wrap">
          <span className="text-[11px] text-slate-400 font-medium mr-1">
            Presets:
          </span>
          {Object.entries(PRESETS).map(([k, p]) => (
            <button
              key={k}
              type="button"
              onClick={() => handlePresetSelect(k)}
              className="text-[11px] px-2 py-0.5 rounded bg-white border border-slate-200 text-slate-600 hover:border-slate-400 hover:text-slate-900 transition-colors cursor-pointer font-medium"
            >
              {p.label}
            </button>
          ))}
        </div>
      </div>

      {step === 1 ? (
        /* STEP 1: Body Metrics */
        <form onSubmit={handleGoToStep2} className="p-5 space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3.5">
            {/* Age */}
            <div>
              <label htmlFor="age" className="block text-xs font-medium text-slate-600 mb-1">
                Age
              </label>
              <div className="relative">
                <input
                  id="age"
                  type="number"
                  step="any"
                  value={formData.age}
                  onChange={(e) => setFormData({ ...formData, age: e.target.value })}
                  className="w-full rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-sm text-slate-900 focus:border-slate-600 outline-none pr-10"
                  placeholder="e.g. 42"
                  required
                />
                <span className="absolute inset-y-0 right-2.5 flex items-center text-xs text-slate-400 pointer-events-none">
                  yrs
                </span>
              </div>
              {formErrors.age && <p className="text-[11px] text-rose-600 mt-1">{formErrors.age}</p>}
            </div>

            {/* Biological Sex */}
            <div>
              <label className="block text-xs font-medium text-slate-600 mb-1">Biological Sex</label>
              <div className="grid grid-cols-2 gap-1 p-0.5 bg-slate-100 rounded-lg">
                <button
                  type="button"
                  onClick={() => setFormData({ ...formData, gender: 'male' })}
                  className={`py-1.5 text-xs font-semibold rounded transition-colors cursor-pointer ${
                    formData.gender === 'male'
                      ? 'bg-white text-slate-900 shadow-xs'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  Male
                </button>
                <button
                  type="button"
                  onClick={() => setFormData({ ...formData, gender: 'female' })}
                  className={`py-1.5 text-xs font-semibold rounded transition-colors cursor-pointer ${
                    formData.gender === 'female'
                      ? 'bg-white text-slate-900 shadow-xs'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  Female
                </button>
              </div>
            </div>

            {/* Standing Height */}
            <div>
              <label htmlFor="height_cm" className="block text-xs font-medium text-slate-600 mb-1">
                Height
              </label>
              <div className="relative">
                <input
                  id="height_cm"
                  type="number"
                  step="any"
                  value={formData.height_cm}
                  onChange={(e) => setFormData({ ...formData, height_cm: e.target.value })}
                  className="w-full rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-sm text-slate-900 focus:border-slate-600 outline-none pr-10"
                  placeholder="e.g. 172"
                  required
                />
                <span className="absolute inset-y-0 right-2.5 flex items-center text-xs text-slate-400 pointer-events-none">
                  cm
                </span>
              </div>
              {formErrors.height_cm && <p className="text-[11px] text-rose-600 mt-1">{formErrors.height_cm}</p>}
            </div>

            {/* Body Weight */}
            <div>
              <label htmlFor="weight_kg" className="block text-xs font-medium text-slate-600 mb-1">
                Weight
              </label>
              <div className="relative">
                <input
                  id="weight_kg"
                  type="number"
                  step="any"
                  value={formData.weight_kg}
                  onChange={(e) => setFormData({ ...formData, weight_kg: e.target.value })}
                  className="w-full rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-sm text-slate-900 focus:border-slate-600 outline-none pr-10"
                  placeholder="e.g. 78.5"
                  required
                />
                <span className="absolute inset-y-0 right-2.5 flex items-center text-xs text-slate-400 pointer-events-none">
                  kg
                </span>
              </div>
              {formErrors.weight_kg && <p className="text-[11px] text-rose-600 mt-1">{formErrors.weight_kg}</p>}
            </div>

            {/* Waist Circumference */}
            <div className="sm:col-span-2 md:col-span-2">
              <div className="flex items-center justify-between mb-1">
                <label htmlFor="waist_cm" className="block text-xs font-medium text-slate-600">
                  Waist Circumference
                </label>
                <span className="text-[10px] text-slate-400">At navel level (optional)</span>
              </div>
              <div className="relative">
                <input
                  id="waist_cm"
                  type="number"
                  step="any"
                  value={formData.waist_cm}
                  onChange={(e) => setFormData({ ...formData, waist_cm: e.target.value })}
                  className="w-full rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-sm text-slate-900 focus:border-slate-600 outline-none pr-10"
                  placeholder="e.g. 92.5 (estimated if omitted)"
                />
                <span className="absolute inset-y-0 right-2.5 flex items-center text-xs text-slate-400 pointer-events-none">
                  cm
                </span>
              </div>
              {formErrors.waist_cm && <p className="text-[11px] text-rose-600 mt-1">{formErrors.waist_cm}</p>}
            </div>
          </div>

          {/* Quick Metrics Bar */}
          <div className="px-3.5 py-2 rounded-lg bg-slate-50 border border-slate-200 flex items-center justify-between text-xs">
            <span className="text-slate-600">
              BMI:{' '}
              {bmiStr !== '-' ? (
                <>
                  <strong className="text-slate-900">{bmiStr} kg/m²</strong>{' '}
                  <span className="text-slate-500">
                    ({!isNaN(bmiVal) && bmiVal >= 30 ? 'Obese' : !isNaN(bmiVal) && bmiVal >= 25 ? 'Overweight' : 'Normal'})
                  </span>
                </>
              ) : (
                <span className="text-slate-400">-</span>
              )}
            </span>
            <span className="text-slate-600">
              Waist-to-Height:{' '}
              {whtrStr !== '-' ? (
                <>
                  <strong className="text-slate-900">{whtrStr}</strong>{' '}
                  <span className="text-slate-500">
                    ({!isNaN(whtrVal) && whtrVal > 0.5 ? 'Elevated' : 'Optimal'})
                  </span>
                </>
              ) : (
                <span className="text-slate-400">-</span>
              )}
            </span>
          </div>

          <div className="flex justify-end pt-1">
            <button
              type="submit"
              className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg bg-slate-900 text-white text-xs font-medium hover:bg-slate-800 transition-colors cursor-pointer"
            >
              Continue to Step 2 <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </form>
      ) : (
        /* STEP 2: Lifestyle & Vitals */
        <form onSubmit={handleFinalSubmit} className="p-5 space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3.5">
            {/* Systolic BP */}
            <div>
              <label htmlFor="systolic_bp" className="block text-xs font-medium text-slate-600 mb-1">
                Systolic BP
              </label>
              <div className="relative">
                <input
                  id="systolic_bp"
                  type="number"
                  step="any"
                  value={formData.systolic_bp}
                  onChange={(e) => setFormData({ ...formData, systolic_bp: e.target.value })}
                  className="w-full rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-sm text-slate-900 focus:border-slate-600 outline-none pr-14"
                  placeholder="120"
                />
                <span className="absolute inset-y-0 right-2.5 flex items-center text-[11px] text-slate-400 pointer-events-none">
                  mmHg
                </span>
              </div>
            </div>

            {/* Diastolic BP */}
            <div>
              <label htmlFor="diastolic_bp" className="block text-xs font-medium text-slate-600 mb-1">
                Diastolic BP
              </label>
              <div className="relative">
                <input
                  id="diastolic_bp"
                  type="number"
                  step="any"
                  value={formData.diastolic_bp}
                  onChange={(e) => setFormData({ ...formData, diastolic_bp: e.target.value })}
                  className="w-full rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-sm text-slate-900 focus:border-slate-600 outline-none pr-14"
                  placeholder="80"
                />
                <span className="absolute inset-y-0 right-2.5 flex items-center text-[11px] text-slate-400 pointer-events-none">
                  mmHg
                </span>
              </div>
            </div>

            {/* Daily Sugar */}
            <div>
              <label htmlFor="daily_sugar_g" className="block text-xs font-medium text-slate-600 mb-1">
                Daily Sugar
              </label>
              <div className="relative">
                <input
                  id="daily_sugar_g"
                  type="number"
                  step="any"
                  value={formData.daily_sugar_g}
                  onChange={(e) => setFormData({ ...formData, daily_sugar_g: e.target.value })}
                  className="w-full rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-sm text-slate-900 focus:border-slate-600 outline-none pr-12"
                  placeholder="75"
                />
                <span className="absolute inset-y-0 right-2.5 flex items-center text-[11px] text-slate-400 pointer-events-none">
                  g/day
                </span>
              </div>
            </div>

            {/* Daily Fiber */}
            <div>
              <label htmlFor="daily_fiber_g" className="block text-xs font-medium text-slate-600 mb-1">
                Daily Fiber
              </label>
              <div className="relative">
                <input
                  id="daily_fiber_g"
                  type="number"
                  step="any"
                  value={formData.daily_fiber_g}
                  onChange={(e) => setFormData({ ...formData, daily_fiber_g: e.target.value })}
                  className="w-full rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-sm text-slate-900 focus:border-slate-600 outline-none pr-12"
                  placeholder="15"
                />
                <span className="absolute inset-y-0 right-2.5 flex items-center text-[11px] text-slate-400 pointer-events-none">
                  g/day
                </span>
              </div>
            </div>
          </div>

          {/* Lifestyle Toggles Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-1">
            {/* Vigorous Activity */}
            <button
              type="button"
              onClick={() => setFormData({ ...formData, vigorous_activity: !formData.vigorous_activity })}
              className={`p-2.5 rounded-lg border text-left transition-colors cursor-pointer ${
                formData.vigorous_activity
                  ? 'border-emerald-600 bg-emerald-50 text-emerald-900'
                  : 'border-slate-200 bg-white text-slate-600 hover:bg-slate-50'
              }`}
            >
              <div className="flex items-center justify-between mb-0.5">
                <span className="text-xs font-semibold">Active Exercise</span>
                {formData.vigorous_activity && <Check className="w-3.5 h-3.5 text-emerald-600" />}
              </div>
              <span className="text-[10px] text-slate-500 block">75+ min/week</span>
            </button>

            {/* Hypertension History */}
            <button
              type="button"
              onClick={() => setFormData({ ...formData, hypertension_history: !formData.hypertension_history })}
              className={`p-2.5 rounded-lg border text-left transition-colors cursor-pointer ${
                formData.hypertension_history
                  ? 'border-amber-600 bg-amber-50 text-amber-900'
                  : 'border-slate-200 bg-white text-slate-600 hover:bg-slate-50'
              }`}
            >
              <div className="flex items-center justify-between mb-0.5">
                <span className="text-xs font-semibold">Hypertension</span>
                {formData.hypertension_history && <Check className="w-3.5 h-3.5 text-amber-600" />}
              </div>
              <span className="text-[10px] text-slate-500 block">Diagnosed</span>
            </button>

            {/* Smoking History */}
            <button
              type="button"
              onClick={() => setFormData({ ...formData, smoking_history: !formData.smoking_history })}
              className={`p-2.5 rounded-lg border text-left transition-colors cursor-pointer ${
                formData.smoking_history
                  ? 'border-slate-600 bg-slate-100 text-slate-900'
                  : 'border-slate-200 bg-white text-slate-600 hover:bg-slate-50'
              }`}
            >
              <div className="flex items-center justify-between mb-0.5">
                <span className="text-xs font-semibold">Smoking History</span>
                {formData.smoking_history && <Check className="w-3.5 h-3.5 text-slate-700" />}
              </div>
              <span className="text-[10px] text-slate-500 block">100+ cigarettes</span>
            </button>

            {/* Sleep Trouble */}
            <button
              type="button"
              onClick={() => setFormData({ ...formData, sleep_trouble: !formData.sleep_trouble })}
              className={`p-2.5 rounded-lg border text-left transition-colors cursor-pointer ${
                formData.sleep_trouble
                  ? 'border-indigo-600 bg-indigo-50 text-indigo-900'
                  : 'border-slate-200 bg-white text-slate-600 hover:bg-slate-50'
              }`}
            >
              <div className="flex items-center justify-between mb-0.5">
                <span className="text-xs font-semibold">Sleep Issues</span>
                {formData.sleep_trouble && <Check className="w-3.5 h-3.5 text-indigo-600" />}
              </div>
              <span className="text-[10px] text-slate-500 block">Frequent insomnia</span>
            </button>
          </div>

          {/* Action Buttons */}
          <div className="flex items-center justify-between pt-2 border-t border-slate-200">
            <button
              type="button"
              onClick={handleGoToStep1}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-slate-200 bg-white hover:bg-slate-50 text-slate-700 text-xs font-medium cursor-pointer"
            >
              <ArrowLeft className="w-3.5 h-3.5" /> Back
            </button>
            <button
              type="submit"
              disabled={isLoading}
              className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg bg-slate-900 text-white text-xs font-medium hover:bg-slate-800 disabled:opacity-50 transition-colors cursor-pointer"
            >
              {isLoading ? (
                <>
                  <div className="w-3 h-3 border-2 border-white/40 border-t-white rounded-full animate-spin" />
                  Calculating...
                </>
              ) : (
                <>
                  Calculate Risk <ArrowRight className="w-3.5 h-3.5" />
                </>
              )}
            </button>
          </div>
        </form>
      )}
    </div>
  );
}
