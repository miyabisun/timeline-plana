import { Activity, Bug, Monitor } from 'lucide-react';
import { ShittimPayload, ProcessCandidate } from '../../App';

interface DebugPanelProps {
  shittimData: ShittimPayload | null;
  targetInfo: ProcessCandidate | null;
}

export default function DebugPanel({ shittimData, targetInfo }: DebugPanelProps) {
  // Listener is now in App.tsx

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

      <div className="grid grid-cols-1 gap-4">
        {/* Window Info Card (Moved from Connection) */}

        {/* Window Info Card (Moved from Connection) */}
        <div className="bg-white dark:bg-slate-800 p-4 rounded-lg shadow-sm border border-slate-200 dark:border-slate-700 transition-colors duration-300">
          <h3 className="text-sm font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-3 flex items-center gap-2">
            <Monitor className="w-4 h-4" />
            Window Information
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="flex justify-between md:block border-b md:border-b-0 border-slate-100 dark:border-slate-700/50 pb-2 md:pb-0">
              <span className="text-slate-600 dark:text-slate-400 block text-xs mb-1">Target</span>
              <span className="font-mono font-medium">{targetInfo?.name || "-"}</span>
            </div>
            <div className="flex justify-between md:block border-b md:border-b-0 border-slate-100 dark:border-slate-700/50 pb-2 md:pb-0">
              <span className="text-slate-600 dark:text-slate-400 block text-xs mb-1">PID</span>
              <span className="font-mono text-slate-400 dark:text-slate-500">{targetInfo?.pid || "-"}</span>
            </div>
            <div className="flex justify-between md:block border-b md:border-b-0 border-slate-100 dark:border-slate-700/50 pb-2 md:pb-0">
              <span className="text-slate-600 dark:text-slate-400 block text-xs mb-1">Resolution</span>
              <span className="font-mono font-medium text-blue-600 dark:text-blue-400">
                16:9 (Native)
              </span>
            </div>
          </div>
        </div>

        {/* System Diagnostics (Moved from Connection) */}
        {shittimData && shittimData.stats && (
          <div className="bg-white dark:bg-slate-800 p-4 rounded-lg shadow-sm border border-slate-200 dark:border-slate-700 transition-colors duration-300">
            <h3 className="text-sm font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-3 flex items-center gap-2">
              <Activity className="w-4 h-4" />
              System Diagnostics
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="p-3 bg-slate-50 dark:bg-slate-900/50 rounded border border-slate-100 dark:border-slate-700/50">
                <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">Link Speed</div>
                <div className="font-mono text-lg font-semibold">{shittimData.fps.toFixed(1)} <span className="text-xs font-normal text-slate-400">FPS</span></div>
                <div className="text-[10px] text-slate-400 mt-1 leading-tight">Backend to Frontend Sync Rate</div>
              </div>
              <div className="p-3 bg-slate-50 dark:bg-slate-900/50 rounded border border-slate-100 dark:border-slate-700/50">
                <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">Capture Rate</div>
                <div className="font-mono text-lg font-semibold">{shittimData.stats.received.toFixed(1)} <span className="text-xs font-normal text-slate-400">FPS</span></div>
                <div className="text-[10px] text-slate-400 mt-1 leading-tight">Raw Windows Capture Speed</div>
              </div>
              <div className="p-3 bg-slate-50 dark:bg-slate-900/50 rounded border border-slate-100 dark:border-slate-700/50">
                <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">Process Rate</div>
                <div className="font-mono text-lg font-semibold">{shittimData.stats.accepted.toFixed(1)} <span className="text-xs font-normal text-slate-400">FPS</span></div>
                <div className="text-[10px] text-slate-400 mt-1 leading-tight">Internal processing speed</div>
              </div>
              <div className="p-3 bg-slate-50 dark:bg-slate-900/50 rounded border border-slate-100 dark:border-slate-700/50">
                <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">Frame Drops</div>
                <div className={`font-mono text-lg font-semibold ${shittimData.stats.queue_full > 0 ? 'text-yellow-500' : ''}`}>
                  {shittimData.stats.queue_full}
                </div>
                <div className="text-[10px] text-slate-400 mt-1 leading-tight">Frames dropped due to load</div>
              </div>
            </div>
          </div>
        )}

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
