import { Activity, Bug, Camera, Layers, Monitor } from 'lucide-react';
import { invoke } from '@tauri-apps/api/core';
import type { ShittimPayload, ProcessCandidate } from '../../types';

interface DebugPanelProps {
  shittimData: ShittimPayload | null;
  targetInfo: ProcessCandidate | null;
  overlayActive: boolean;
  onToggleOverlay: () => void;
}

export default function DebugPanel({ shittimData, targetInfo, overlayActive, onToggleOverlay }: DebugPanelProps) {
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
        {/* Window Info Card */}
        <div className="bg-white dark:bg-slate-800 p-4 rounded-lg shadow-sm border border-slate-200 dark:border-slate-700 transition-colors duration-300">
          <h3 className="text-sm font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-3 flex items-center gap-2">
            <Monitor className="w-4 h-4" />
            Window Information
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="flex justify-between md:block border-b md:border-b-0 border-slate-100 dark:border-slate-700/50 pb-2 md:pb-0">
              <span className="text-slate-600 dark:text-slate-400 block text-xs mb-1">Target</span>
              <span className="font-mono font-medium">{targetInfo?.name || "-"}</span>
            </div>
            <div className="flex justify-between md:block border-b md:border-b-0 border-slate-100 dark:border-slate-700/50 pb-2 md:pb-0">
              <span className="text-slate-600 dark:text-slate-400 block text-xs mb-1">PID</span>
              <span className="font-mono text-slate-400 dark:text-slate-500">{targetInfo?.pid || "-"}</span>
            </div>
            <div className="flex justify-between md:block border-b md:border-b-0 border-slate-100 dark:border-slate-700/50 pb-2 md:pb-0">
              <span className="text-slate-600 dark:text-slate-400 block text-xs mb-1">Client Size</span>
              <span className="font-mono font-medium text-blue-600 dark:text-blue-400">
                {shittimData ? `${shittimData.window.width}x${shittimData.window.height}` : "-"}
              </span>
            </div>
            <div className="flex justify-between md:block border-b md:border-b-0 border-slate-100 dark:border-slate-700/50 pb-2 md:pb-0">
              <span className="text-slate-600 dark:text-slate-400 block text-xs mb-1">Position</span>
              <span className="font-mono font-medium">
                {shittimData ? `(${shittimData.window.x}, ${shittimData.window.y})` : "-"}
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

        {/* Capture Tools */}
        <div className="bg-white dark:bg-slate-800 p-4 rounded-lg shadow-sm border border-slate-200 dark:border-slate-700 transition-colors duration-300">
          <h3 className="text-sm font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-3 flex items-center gap-2">
            <Camera className="w-4 h-4" />
            Capture Tools
          </h3>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium">Save Cost ROI</p>
              <p className="text-xs text-slate-400 dark:text-slate-500">
                Save skew-corrected cost gauge image to output/
              </p>
            </div>
            <button
              onClick={() => invoke('save_cost_roi_image')}
              disabled={!shittimData}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                shittimData
                  ? 'bg-blue-600 hover:bg-blue-700 active:bg-blue-800 text-white cursor-pointer'
                  : 'bg-slate-300 dark:bg-slate-600 text-slate-500 cursor-not-allowed'
              }`}
            >
              Capture
            </button>
          </div>
        </div>

        {/* Overlay Test */}
        <div className="bg-white dark:bg-slate-800 p-4 rounded-lg shadow-sm border border-slate-200 dark:border-slate-700 transition-colors duration-300">
          <h3 className="text-sm font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-3 flex items-center gap-2">
            <Layers className="w-4 h-4" />
            Overlay Test
          </h3>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium">Test Overlay</p>
              <p className="text-xs text-slate-400 dark:text-slate-500">
                Show a card on the right edge of the game window
              </p>
            </div>
            <button
              onClick={onToggleOverlay}
              disabled={!shittimData}
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                overlayActive
                  ? 'bg-blue-600'
                  : 'bg-slate-300 dark:bg-slate-600'
              } ${!shittimData ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
            >
              <span
                className={`inline-block h-4 w-4 rounded-full bg-white transition-transform ${
                  overlayActive ? 'translate-x-6' : 'translate-x-1'
                }`}
              />
            </button>
          </div>
        </div>

      </div>
    </div>
  );
}
