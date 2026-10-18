import React from 'react';
import { PredictionResult } from '../types';

export default function RecommendationCard({ result }: { result: PredictionResult }) {
  return (
    <section className="bg-white rounded-xl border border-slate-200 p-5 space-y-2 text-xs text-slate-600">
      <h3 className="font-semibold text-slate-800">How to interpret this score</h3>
      <p>This educational model estimates a cross-sectional HbA1c/diagnosis/therapy proxy in an unweighted US NHANES 2013–2014 sample. It does not predict when you will develop diabetes, diagnose prediabetes, or measure treatment benefit.</p>
      <p>{result.screening_positive ? 'The score exceeds the training-selected screening threshold.' : 'The score is below the training-selected screening threshold; this cannot rule out diabetes.'} A clinician can advise whether laboratory testing is appropriate.</p>
      <p>Diet and activity associations can reflect changes made after diagnosis. Consult a qualified healthcare professional for diagnosis and personal advice. The model has not been externally validated for people in Thailand.</p>
    </section>
  );
}
