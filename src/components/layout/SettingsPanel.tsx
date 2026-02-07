import { Settings, Laptop, Sun, Moon } from "lucide-react";
import type { Theme } from '../../types';

interface SettingsPanelProps {
  theme: Theme;
  setTheme: (theme: Theme) => void;
}

export default function SettingsPanel({ theme, setTheme }: SettingsPanelProps) {
  return (
    <div className="p-8 max-w-2xl mx-auto">
      <div className="bg-white dark:bg-slate-900 rounded-lg shadow-sm border border-slate-200 dark:border-slate-800 p-6">
        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Settings className="w-5 h-5" />
          Appearance
        </h2>

        <div className="space-y-4">
          <div className="flex flex-col gap-2">
            <label className="text-sm font-medium text-slate-700 dark:text-slate-300">Theme</label>
            <div className="grid grid-cols-3 gap-2 p-1 bg-slate-100 dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">

              <button
                onClick={() => setTheme('auto')}
                className={`flex items-center justify-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-all ${theme === 'auto'
                  ? 'bg-white dark:bg-slate-700 text-blue-600 dark:text-blue-400 shadow-sm'
                  : 'text-slate-500 hover:text-slate-700 dark:hover:text-slate-300'
                  }`}
              >
                <Laptop className="w-4 h-4" />
                Auto
              </button>

              <button
                onClick={() => setTheme('light')}
                className={`flex items-center justify-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-all ${theme === 'light'
                  ? 'bg-white dark:bg-slate-700 text-blue-600 dark:text-blue-400 shadow-sm'
                  : 'text-slate-500 hover:text-slate-700 dark:hover:text-slate-300'
                  }`}
              >
                <Sun className="w-4 h-4" />
                Light
              </button>

              <button
                onClick={() => setTheme('dark')}
                className={`flex items-center justify-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-all ${theme === 'dark'
                  ? 'bg-white dark:bg-slate-700 text-blue-600 dark:text-blue-400 shadow-sm'
                  : 'text-slate-500 hover:text-slate-700 dark:hover:text-slate-300'
                  }`}
              >
                <Moon className="w-4 h-4" />
                Dark
              </button>

            </div>
            <p className="text-xs text-slate-400 dark:text-slate-500 mt-1">
              Choose "Auto" to sync with your system preferences.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
