import { useEffect, useState, useRef, useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { CheckCircle2, Shield, Activity, Clock } from 'lucide-react';
import { ProcessCandidate, ShittimPayload } from '../../types';

interface ConnectionPanelProps {
  shittimData: ShittimPayload | null;
  targetInfo: ProcessCandidate | null;
  setTargetInfo: (info: ProcessCandidate | null) => void;
}

export default function ConnectionPanel({ shittimData, setTargetInfo }: ConnectionPanelProps) {
  const [status, setStatus] = useState<string>("Searching for BlueArchive...");
  const [isConnected, setIsConnected] = useState(false);
  const isConnectedRef = useRef(false);
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopPolling = useCallback(() => {
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
  }, []);

  const startCapture = useCallback(async (hwnd: number) => {
    try {
      setStatus("Target Found. Connecting...");
      await invoke('start_intercept_demo', { hwnd });
      setStatus("Connected");
      setIsConnected(true);
      isConnectedRef.current = true;
    } catch (e) {
      setStatus(`Connection Error: ${e}`);
    }
  }, []);

  const checkTarget = useCallback(async () => {
    if (isConnectedRef.current) return;

    try {
      const procs = await invoke<ProcessCandidate[]>('list_potential_targets');
      const target = procs.find(p => p.name === "BlueArchive.exe");

      if (target) {
        setTargetInfo(target);
        stopPolling();
        startCapture(target.hwnd);
      }
    } catch (e) {
      console.error("Failed to list targets:", e);
    }
  }, [setTargetInfo, stopPolling, startCapture]);

  useEffect(() => {
    checkTarget();
    if (!pollingRef.current) {
      pollingRef.current = setInterval(checkTarget, 3000);
    }

    return () => {
      stopPolling();
    };
  }, [checkTarget, stopPolling]);

  return (
    <div className="p-6 bg-slate-50 dark:bg-slate-900 min-h-full rounded-lg text-slate-800 dark:text-slate-100 transition-colors duration-300">
      <div className="mb-6 flex items-center gap-3">
        {isConnected ? (
          <CheckCircle2 className="w-8 h-8 text-green-500 dark:text-green-400" />
        ) : (
          <div className="w-8 h-8 rounded-full border-4 border-slate-300 dark:border-slate-600 border-t-blue-500 dark:border-t-blue-400 animate-spin" />
        )}
        <div>
          <h2 className="text-xl font-bold">{isConnected ? "Connected" : "Searching..."}</h2>
          <p className="text-slate-500 dark:text-slate-400 text-sm">{status}</p>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4">
        {/* Battle Status Card */}
        <div className="bg-white dark:bg-slate-800 p-4 rounded-lg shadow-sm border border-slate-200 dark:border-slate-700 transition-colors duration-300">
          <h3 className="text-sm font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-3 flex items-center gap-2">
            <Shield className="w-4 h-4" />
            Battle Status
          </h3>

          <div className="flex flex-col items-start justify-center h-32 gap-2 pl-2">
            {shittimData ? (
              <>
                <div className="flex items-center gap-3">
                  <div className={`p-2 rounded-full transition-colors duration-300 ${shittimData.battle_state === 'Active' ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400' :
                    shittimData.battle_state === 'Paused' ? 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-400' :
                      'bg-slate-100 dark:bg-slate-700 text-slate-500 dark:text-slate-400'
                    }`}>
                    <Activity className="w-6 h-6" />
                  </div>
                  <span className="text-3xl font-bold tracking-tight">
                    {shittimData.battle_state}
                  </span>
                </div>
                <div className="flex flex-col gap-1 mt-1">
                  <span className="text-xs text-slate-400 dark:text-slate-500 font-mono">
                    System Linked
                  </span>
                </div>
              </>
            ) : (
              <div className="text-slate-400 dark:text-slate-500 flex items-center gap-3">
                <Clock className="w-8 h-8 opacity-20" />
                <span>Waiting for Plana's Signal...</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
