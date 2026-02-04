import { useEffect, useState } from 'react';
import { listen } from '@tauri-apps/api/event';
import { Activity, Bug } from 'lucide-react';

interface CaptureStatus {
  window_width: number;
  window_height: number;
  battle_state: string;
  fps: number;
}

export default function DebugPanel() {
  const [captureStatus, setCaptureStatus] = useState<CaptureStatus | null>(null);

  useEffect(() => {
    const unlistenPromise = listen<CaptureStatus>('capture-status', (event) => {
      setCaptureStatus(event.payload);
    });

    return () => {
      unlistenPromise.then(unlisten => unlisten());
    };
  }, []);

  return (
    <div className="p-6 bg-slate-50 dark:bg-slate-900 min-h-full rounded-lg text-slate-800 dark:text-slate-100 transition-colors duration-300">
      <div className="mb-6 flex items-center gap-3">
        <div className="p-2 rounded-full bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300">
          <Bug className="w-6 h-6" />
        </div>
        <div>
          <h2 className="text-xl font-bold">Debug Tools</h2>
          <p className="text-slate-500 dark:text-slate-400 text-sm">Internal developer metrics</p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Internal Metrics Card */}
        <div className="bg-white dark:bg-slate-800 p-4 rounded-lg shadow-sm border border-slate-200 dark:border-slate-700 transition-colors duration-300">
          <h3 className="text-sm font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-3 flex items-center gap-2">
            <Activity className="w-4 h-4" />
            Internal Performance
          </h3>
          <div className="space-y-4">
            <div className="flex flex-col gap-1">
              <span className="text-xs text-slate-500 dark:text-slate-400">Internal Processing FPS</span>
              <div className="flex items-baseline gap-2">
                <span className="text-3xl font-mono font-bold text-blue-600 dark:text-blue-400">
                  {captureStatus ? captureStatus.fps.toFixed(1) : "-"}
                </span>
                <span className="text-sm text-slate-400 dark:text-slate-500">fps</span>
              </div>
              <p className="text-xs text-slate-400 dark:text-slate-500 mt-1">
                Target: 30.0 fps (Throttled for stability)
              </p>
            </div>
          </div>
        </div>

        {/* Placeholder for future debug tools */}
        <div className="bg-white dark:bg-slate-800 p-4 rounded-lg shadow-sm border border-slate-200 dark:border-slate-700 transition-colors duration-300 opacity-50">
          <h3 className="text-sm font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-3">
            Future Tools
          </h3>
          <p className="text-sm text-slate-400 dark:text-slate-500 text-center py-4">
            OCR Debugging / Binary View
          </p>
        </div>

      </div>
    </div>
  );
}
