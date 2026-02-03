import { useEffect, useState, useRef } from 'react';
import { invoke } from '@tauri-apps/api/core';

interface ProcessCandidate {
  pid: number;
  name: string;
  window_title: string;
  hwnd: number;
}

export default function CapturePreview() {
  const [status, setStatus] = useState<string>("Searching for BlueArchive...");
  const [isConnected, setIsConnected] = useState(false);
  const pollingRef = useRef<NodeJS.Timeout | null>(null);

  // Hardcoded local MJPEG server port
  const streamUrl = "http://localhost:12345";

  useEffect(() => {
    // Start polling for target on mount
    startPolling();

    return () => {
      stopPolling();
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
        console.log("Found target:", target);
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
    <div style={{ padding: '20px', border: '1px solid #333', marginTop: '20px' }}>
      <div style={{ marginBottom: '10px', display: 'flex', gap: '10px', alignItems: 'center' }}>
        <strong>Status: {status}</strong>
      </div>

      <div style={{
        width: '100%',
        height: '400px',
        backgroundColor: '#000',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        overflow: 'hidden',
        position: 'relative'
      }}>
        {isConnected ? (
          <img
            src={streamUrl}
            alt="MJPEG Stream"
            style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }}
            onError={() => {
              // Determine if we should retry or just log
              console.log("Stream disconnected");
              // setIsConnected(false); // Optional: reset if stream fails
            }}
          />
        ) : (
          <div style={{
            position: 'absolute',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            color: '#666',
            gap: '10px'
          }}>
            <div className="spinner" style={{
              width: '40px',
              height: '40px',
              border: '4px solid #333',
              borderTop: '4px solid #fff',
              borderRadius: '50%',
              animation: 'spin 1s linear infinite'
            }}></div>
            <span>Waiting for Blue Archive...</span>
            <style>{`
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
            `}</style>
          </div>
        )}
      </div>
    </div>
  );
}
