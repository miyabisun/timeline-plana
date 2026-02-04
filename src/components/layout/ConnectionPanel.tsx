import { useEffect, useState, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { CheckCircle2, Monitor, Shield, Activity, Clock } from 'lucide-react';

interface ProcessCandidate {
  pid: number;
  name: string;
  window_title: string;
  hwnd: number;
}

interface CaptureStatus {
  window_width: number;
  window_height: number;
  battle_state: string;
  fps: number;
}

export default function ConnectionPanel() {
  const [status, setStatus] = useState<string>("Searching for BlueArchive...");
  const [isConnected, setIsConnected] = useState(false);
  const [targetInfo, setTargetInfo] = useState<ProcessCandidate | null>(null);
  const [captureStatus, setCaptureStatus] = useState<CaptureStatus | null>(null);
  const pollingRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    startPolling();

    const unlistenPromise = listen<CaptureStatus>('capture-status', (event) => {
      setCaptureStatus(event.payload);
    });

    return () => {
      stopPolling();
      unlistenPromise.then(unlisten => unlisten());
    };
  }, []);

  const startPolling = () => {
    checkTarget();
    if (!pollingRef.current) {
      pollingRef.current = setInterval(checkTarget, 3000);
    }
  };

  const stopPolling = () => {
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
  };

  const checkTarget = async () => {
    if (isConnected) return;

    try {
      const procs = await invoke<ProcessCandidate[]>('list_potential_targets');
      const target = procs.find(p => p.name === "BlueArchive.exe");

      if (target) {
        setTargetInfo(target);
        startCapture(target.hwnd);
        stopPolling();
      }
    } catch (e) {
      console.error("Failed to list targets:", e);
    }
  };

  const startCapture = async (hwnd: number) => {
    try {
      setStatus("Target Found. Connecting...");
      await invoke('start_intercept_demo', { hwnd: hwnd });
      setStatus("Connected");
      setIsConnected(true);
    } catch (e) {
      setStatus(`Connection Error: ${e}`);
      startPolling();
    }
  };

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

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Window Info Card */}
        <div className="bg-white dark:bg-slate-800 p-4 rounded-lg shadow-sm border border-slate-200 dark:border-slate-700 transition-colors duration-300">
          <h3 className="text-sm font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-3 flex items-center gap-2">
            <Monitor className="w-4 h-4" />
            Window Information
          </h3>
          <div className="space-y-2">
            <div className="flex justify-between border-b border-slate-100 dark:border-slate-700/50 pb-2">
              <span className="text-slate-600 dark:text-slate-400">Target</span>
              <span className="font-mono font-medium">{targetInfo?.name || "-"}</span>
            </div>
            <div className="flex justify-between border-b border-slate-100 dark:border-slate-700/50 pb-2">
              <span className="text-slate-600 dark:text-slate-400">PID</span>
              <span className="font-mono text-slate-400 dark:text-slate-500">{targetInfo?.pid || "-"}</span>
            </div>
            <div className="flex justify-between border-b border-slate-100 dark:border-slate-700/50 pb-2">
              <span className="text-slate-600 dark:text-slate-400">Resolution</span>
              <span className="font-mono font-medium text-blue-600 dark:text-blue-400">
                {captureStatus ? `${captureStatus.window_width}x${captureStatus.window_height}` : "-"}
              </span>
            </div>
          </div>
        </div>

        {/* Battle Status Card */}
        <div className="bg-white dark:bg-slate-800 p-4 rounded-lg shadow-sm border border-slate-200 dark:border-slate-700 transition-colors duration-300">
          <h3 className="text-sm font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-3 flex items-center gap-2">
            <Shield className="w-4 h-4" />
            Battle Status
          </h3>

          <div className="flex flex-col items-start justify-center h-32 gap-2 pl-2">
            {captureStatus ? (
              <>
                <div className="flex items-center gap-3">
                  <div className={`p-2 rounded-full transition-colors duration-300 ${captureStatus.battle_state === 'Active' ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400' :
                    captureStatus.battle_state === 'Paused' ? 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-400' :
                      'bg-slate-100 dark:bg-slate-700 text-slate-500 dark:text-slate-400'
                    }`}>
                    <Activity className="w-6 h-6" />
                  </div>
                  <span className="text-3xl font-bold tracking-tight">
                    {captureStatus.battle_state}
                  </span>
                </div>
                <div className="flex flex-col gap-1 mt-1">
                  <span className="text-xs text-slate-400 dark:text-slate-500 font-mono">
                    Last Update: {new Date().toLocaleTimeString()}
                  </span>
                </div>
              </>
            ) : (
              <div className="text-slate-400 dark:text-slate-500 flex items-center gap-3">
                <Clock className="w-8 h-8 opacity-20" />
                <span>Waiting for capture stream...</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
