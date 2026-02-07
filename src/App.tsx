import { useState, useEffect, useCallback, useRef } from "react";
import ConnectionPanel from "./components/layout/ConnectionPanel";
import DebugPanel from "./components/layout/DebugPanel";
import TimelinePanel from "./components/layout/TimelinePanel";
import SettingsPanel from "./components/layout/SettingsPanel";
import { Activity, Bug, Settings, LayoutDashboard } from "lucide-react";
import { listen } from '@tauri-apps/api/event';
import { WebviewWindow } from '@tauri-apps/api/webviewWindow';
import { currentMonitor } from '@tauri-apps/api/window';
import type { ProcessCandidate, ShittimPayload, Theme, TabId } from './types';
import "./App.css";

function App() {
  const [activeTab, setActiveTab] = useState<TabId>("check");
  // Initialize from localStorage or default to 'auto'
  const [theme, setTheme] = useState<Theme>(() => {
    return (localStorage.getItem('app-theme') as Theme) || 'auto';
  });

  // Apply Theme Effect
  useEffect(() => {
    const root = document.documentElement;
    localStorage.setItem('app-theme', theme);

    const applyTheme = () => {
      if (theme === 'dark') {
        root.classList.add('dark');
      } else if (theme === 'light') {
        root.classList.remove('dark');
      } else {
        // Auto
        if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
          root.classList.add('dark');
        } else {
          root.classList.remove('dark');
        }
      }
    };

    applyTheme();

    // Listener for system changes when in auto mode
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const handleChange = () => {
      if (theme === 'auto') {
        applyTheme();
      }
    };
    mediaQuery.addEventListener('change', handleChange);

    return () => mediaQuery.removeEventListener('change', handleChange);
  }, [theme]);

  // Global State for Shittim Link (Bubbled up to share between panels)
  const [shittimData, setShittimData] = useState<ShittimPayload | null>(null);
  const [targetInfo, setTargetInfo] = useState<ProcessCandidate | null>(null);

  // Listen to Shittim Link globally (persists across tab switches)
  useEffect(() => {
    const unlistenPromise = listen<ShittimPayload>('link-sync', (event) => {
      setShittimData(event.payload);
    });

    return () => {
      unlistenPromise.then(unlisten => unlisten());
    };
  }, []);

  // Overlay state (lives in App so it survives tab switches)
  const [overlayActive, setOverlayActive] = useState(false);
  const overlayActiveRef = useRef(false);
  const shittimDataRef = useRef(shittimData);
  useEffect(() => { overlayActiveRef.current = overlayActive; }, [overlayActive]);
  useEffect(() => { shittimDataRef.current = shittimData; }, [shittimData]);

  const destroyOverlay = useCallback(async () => {
    try {
      const existing = await WebviewWindow.getByLabel('overlay');
      if (existing) await existing.destroy();
    } catch { /* ignore */ }
  }, []);

  const createOverlay = useCallback(async () => {
    const data = shittimDataRef.current;
    if (!data) return;
    const monitor = await currentMonitor();
    const scale = monitor?.scaleFactor ?? 1;
    const win = data.window;
    // Backend sends physical pixels (Win32 API); Tauri positions in logical pixels
    const logicalX = win.x / scale;
    const logicalY = win.y / scale;
    const logicalW = win.width / scale;
    const logicalH = win.height / scale;
    const overlayWidth = 140; // Windows enforces minimum window width ~125px
    const overlayHeight = 120;
    const x = Math.round(logicalX + logicalW - overlayWidth);
    const y = Math.round(logicalY + logicalH / 2 - overlayHeight / 2);

    const resolvedTheme = theme === 'auto'
      ? (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')
      : theme;

    const webview = new WebviewWindow('overlay', {
      url: `/overlay.html?theme=${resolvedTheme}`,
      width: overlayWidth,
      height: overlayHeight,
      x,
      y,
      decorations: false,
      transparent: true,
      alwaysOnTop: true,
      skipTaskbar: true,
      focus: false,
      resizable: false,
      shadow: false,
    });

    webview.once('tauri://created', () => {
      webview.setIgnoreCursorEvents(true);
    });
  }, [theme]);

  const toggleOverlay = useCallback(async () => {
    if (overlayActive) {
      await destroyOverlay();
      setOverlayActive(false);
    } else {
      if (!shittimData) return;
      await createOverlay();
      setOverlayActive(true);
    }
  }, [overlayActive, shittimData, createOverlay, destroyOverlay]);

  // Recreate overlay when theme changes
  useEffect(() => {
    if (!overlayActiveRef.current) return;
    (async () => {
      await destroyOverlay();
      await new Promise(r => setTimeout(r, 50));
      await createOverlay();
    })();
  }, [createOverlay]); // createOverlay changes when theme changes

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      WebviewWindow.getByLabel('overlay')
        .then(w => w?.destroy())
        .catch(() => {});
    };
  }, []);

  return (
    <main className="flex flex-col h-screen bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 font-sans transition-colors duration-300">
      {/* Compact Header / Tab Bar */}
      <header className="bg-white dark:bg-slate-900 border-b border-slate-200 dark:border-slate-800 flex items-end px-2 gap-1 shrink-0 h-10 select-none transition-colors duration-300">
        {/* Small Icon (No text) */}
        <div className="flex items-center justify-center w-10 h-full text-slate-400 dark:text-slate-500 opacity-50">
          <LayoutDashboard className="w-5 h-5" />
        </div>

        <nav className="flex gap-1 h-full items-end">
          <button
            onClick={() => setActiveTab("check")}
            className={`px-4 rounded-t-lg text-sm font-bold transition-all flex items-center gap-2 border-t border-x ${activeTab === "check"
              ? "bg-slate-200 dark:bg-slate-800 border-slate-300 dark:border-slate-700 text-slate-800 dark:text-slate-100 h-full translate-y-[1px] z-10"
              : "bg-transparent border-transparent text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800/50 h-[80%] mb-1"
              }`}
          >
            <Activity className="w-4 h-4" />
            Connection
          </button>
          <button
            onClick={() => setActiveTab("timeline")}
            className={`px-4 rounded-t-lg text-sm font-bold transition-all flex items-center gap-2 border-t border-x ${activeTab === "timeline"
              ? "bg-slate-200 dark:bg-slate-800 border-slate-300 dark:border-slate-700 text-slate-800 dark:text-slate-100 h-full translate-y-[1px] z-10"
              : "bg-transparent border-transparent text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800/50 h-[80%] mb-1"
              }`}
          >
            <LayoutDashboard className="w-4 h-4" />
            Timeline
          </button>
          <button
            onClick={() => setActiveTab("debug")}
            className={`px-4 rounded-t-lg text-sm font-bold transition-all flex items-center gap-2 border-t border-x ${activeTab === "debug"
              ? "bg-slate-200 dark:bg-slate-800 border-slate-300 dark:border-slate-700 text-slate-800 dark:text-slate-100 h-full translate-y-[1px] z-10"
              : "bg-transparent border-transparent text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800/50 h-[80%] mb-1"
              }`}
          >
            <Bug className="w-4 h-4" />
            Debug
          </button>
          <button
            onClick={() => setActiveTab("settings")}
            className={`px-4 rounded-t-lg text-sm font-bold transition-all flex items-center gap-2 border-t border-x ${activeTab === "settings"
              ? "bg-slate-200 dark:bg-slate-800 border-slate-300 dark:border-slate-700 text-slate-800 dark:text-slate-100 h-full translate-y-[1px] z-10"
              : "bg-transparent border-transparent text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800/50 h-[80%] mb-1"
              }`}
          >
            <Settings className="w-4 h-4" />
            Settings
          </button>
        </nav>
      </header>

      {/* Main Content Area */}
      <div className="flex-1 overflow-auto bg-slate-200/50 dark:bg-slate-950 transition-colors duration-300">
        {activeTab === "check" && (
          <div className="p-4 h-full">
            <ConnectionPanel
              shittimData={shittimData}
              targetInfo={targetInfo}
              setTargetInfo={setTargetInfo}
            />
          </div>
        )}

        {activeTab === "timeline" && <TimelinePanel />}

        {activeTab === "debug" && (
          <div className="p-4 h-full">
            <DebugPanel
              shittimData={shittimData}
              targetInfo={targetInfo}
              overlayActive={overlayActive}
              onToggleOverlay={toggleOverlay}
            />
          </div>
        )}

        {activeTab === "settings" && <SettingsPanel theme={theme} setTheme={setTheme} />}
      </div>
    </main>
  );
}

export default App;
