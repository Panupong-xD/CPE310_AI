import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'NHANES Diabetes Risk Assessment & Lifestyle Simulator',
  description: 'Evidence-based non-invasive machine learning screening and interactive lifestyle intervention simulation powered by NHANES data and SHAP explainability.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="font-sans antialiased bg-slate-50 text-slate-900 min-h-screen">
        {children}
      </body>
    </html>
  );
}
