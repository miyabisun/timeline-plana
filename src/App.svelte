<script lang="ts">
  import ConnectionPanel from "./components/layout/ConnectionPanel.svelte";
  import DebugPanel from "./components/layout/DebugPanel.svelte";
  import TimelinePanel from "./components/layout/TimelinePanel.svelte";
  import SettingsPanel from "./components/layout/SettingsPanel.svelte";
  import { Activity, Bug, Settings, LayoutDashboard } from "lucide-svelte";
  import { listen } from '@tauri-apps/api/event';
  import { createOverlay, destroyOverlay } from './lib/overlay';
  import type { ProcessCandidate, ShittimPayload, Theme, TabId } from './types';
  import { onDestroy } from "svelte";
  import "./App.css";

  let activeTab = $state<TabId>("check");
  let theme = $state<Theme>(
    (localStorage.getItem('app-theme') as Theme) || 'auto'
  );

  // Global State for Shittim Link
  let shittimData = $state<ShittimPayload | null>(null);
  let targetInfo = $state<ProcessCandidate | null>(null);

  // Overlay state
  let overlayActive = $state(false);

  // Apply Theme Effect
  $effect(() => {
    const root = document.documentElement;
    localStorage.setItem('app-theme', theme);

    const applyTheme = () => {
      if (theme === 'dark') {
        root.classList.add('dark');
      } else if (theme === 'light') {
        root.classList.remove('dark');
      } else {
        if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
          root.classList.add('dark');
        } else {
          root.classList.remove('dark');
        }
      }
    };

    applyTheme();

    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const handleChange = () => {
      if (theme === 'auto') {
        applyTheme();
      }
    };
    mediaQuery.addEventListener('change', handleChange);

    return () => mediaQuery.removeEventListener('change', handleChange);
  });

  // Listen to Shittim Link globally
  const unlistenPromise = listen<ShittimPayload>('link-sync', (event) => {
    shittimData = event.payload;
  });

  onDestroy(() => {
    unlistenPromise.then(unlisten => unlisten());
    destroyOverlay();
  });

  async function toggleOverlay() {
    if (overlayActive) {
      await destroyOverlay();
      overlayActive = false;
    } else {
      if (!shittimData) return;
      await createOverlay(shittimData.window, theme);
      overlayActive = true;
    }
  }

  // Recreate overlay when theme changes
  let prevTheme: Theme | undefined;
  $effect(() => {
    const currentTheme = theme;
    if (prevTheme !== undefined && currentTheme !== prevTheme && overlayActive) {
      (async () => {
        await destroyOverlay();
        await new Promise(r => setTimeout(r, 50));
        if (shittimData) await createOverlay(shittimData.window, currentTheme);
      })().catch(() => {});
    }
    prevTheme = currentTheme;
  });

  const tabButtonBase = "px-4 rounded-t-lg text-sm font-bold transition-all flex items-center gap-2 border-t border-x";
  const tabActive = "bg-slate-200 dark:bg-slate-800 border-slate-300 dark:border-slate-700 text-slate-800 dark:text-slate-100 h-full translate-y-[1px] z-10";
  const tabInactive = "bg-transparent border-transparent text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800/50 h-[80%] mb-1";
</script>

<main class="flex flex-col h-screen bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 font-sans transition-colors duration-300">
  <!-- Compact Header / Tab Bar -->
  <header class="bg-white dark:bg-slate-900 border-b border-slate-200 dark:border-slate-800 flex items-end px-2 gap-1 shrink-0 h-10 select-none transition-colors duration-300">
    <!-- Small Icon (No text) -->
    <div class="flex items-center justify-center w-10 h-full text-slate-400 dark:text-slate-500 opacity-50">
      <LayoutDashboard class="w-5 h-5" />
    </div>

    <nav class="flex gap-1 h-full items-end">
      <button
        onclick={() => activeTab = "check"}
        class="{tabButtonBase} {activeTab === 'check' ? tabActive : tabInactive}"
      >
        <Activity class="w-4 h-4" />
        Connection
      </button>
      <button
        onclick={() => activeTab = "timeline"}
        class="{tabButtonBase} {activeTab === 'timeline' ? tabActive : tabInactive}"
      >
        <LayoutDashboard class="w-4 h-4" />
        Timeline
      </button>
      <button
        onclick={() => activeTab = "debug"}
        class="{tabButtonBase} {activeTab === 'debug' ? tabActive : tabInactive}"
      >
        <Bug class="w-4 h-4" />
        Debug
      </button>
      <button
        onclick={() => activeTab = "settings"}
        class="{tabButtonBase} {activeTab === 'settings' ? tabActive : tabInactive}"
      >
        <Settings class="w-4 h-4" />
        Settings
      </button>
    </nav>
  </header>

  <!-- Main Content Area -->
  <div class="flex-1 overflow-auto bg-slate-200/50 dark:bg-slate-950 transition-colors duration-300">
    {#if activeTab === "check"}
      <div class="p-4 h-full">
        <ConnectionPanel
          {shittimData}
          bind:targetInfo
        />
      </div>
    {:else if activeTab === "timeline"}
      <TimelinePanel />
    {:else if activeTab === "debug"}
      <div class="p-4 h-full">
        <DebugPanel
          {shittimData}
          {targetInfo}
          {overlayActive}
          onToggleOverlay={toggleOverlay}
        />
      </div>
    {:else if activeTab === "settings"}
      <SettingsPanel bind:theme />
    {/if}
  </div>
</main>
