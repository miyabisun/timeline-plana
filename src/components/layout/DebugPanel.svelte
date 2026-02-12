<script lang="ts">
  import { Activity, Bug, Camera, Layers, Monitor } from 'lucide-svelte';
  import { invoke } from '@tauri-apps/api/core';
  import type { ShittimPayload, ProcessCandidate } from '../../types';

  let { shittimData, targetInfo, overlayActive, onToggleOverlay }: {
    shittimData: ShittimPayload | null;
    targetInfo: ProcessCandidate | null;
    overlayActive: boolean;
    onToggleOverlay: () => void;
  } = $props();
</script>

<div class="p-6 bg-slate-50 dark:bg-slate-900 min-h-full rounded-lg text-slate-800 dark:text-slate-100 transition-colors duration-300">
  <div class="mb-6 flex items-center gap-3">
    <div class="p-2 rounded-full bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300">
      <Bug class="w-6 h-6" />
    </div>
    <div>
      <h2 class="text-xl font-bold">Debug Tools</h2>
      <p class="text-slate-500 dark:text-slate-400 text-sm">Internal developer metrics</p>
    </div>
  </div>

  <div class="grid grid-cols-1 gap-4">
    <!-- Window Info Card -->
    <div class="bg-white dark:bg-slate-800 p-4 rounded-lg shadow-sm border border-slate-200 dark:border-slate-700 transition-colors duration-300">
      <h3 class="text-sm font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-3 flex items-center gap-2">
        <Monitor class="w-4 h-4" />
        Window Information
      </h3>
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div class="flex justify-between md:block border-b md:border-b-0 border-slate-100 dark:border-slate-700/50 pb-2 md:pb-0">
          <span class="text-slate-600 dark:text-slate-400 block text-xs mb-1">Target</span>
          <span class="font-mono font-medium">{targetInfo?.name || "-"}</span>
        </div>
        <div class="flex justify-between md:block border-b md:border-b-0 border-slate-100 dark:border-slate-700/50 pb-2 md:pb-0">
          <span class="text-slate-600 dark:text-slate-400 block text-xs mb-1">PID</span>
          <span class="font-mono text-slate-400 dark:text-slate-500">{targetInfo?.pid || "-"}</span>
        </div>
        <div class="flex justify-between md:block border-b md:border-b-0 border-slate-100 dark:border-slate-700/50 pb-2 md:pb-0">
          <span class="text-slate-600 dark:text-slate-400 block text-xs mb-1">Client Size</span>
          <span class="font-mono font-medium text-blue-600 dark:text-blue-400">
            {shittimData ? `${shittimData.window.width}x${shittimData.window.height}` : "-"}
          </span>
        </div>
        <div class="flex justify-between md:block border-b md:border-b-0 border-slate-100 dark:border-slate-700/50 pb-2 md:pb-0">
          <span class="text-slate-600 dark:text-slate-400 block text-xs mb-1">Position</span>
          <span class="font-mono font-medium">
            {shittimData ? `(${shittimData.window.x}, ${shittimData.window.y})` : "-"}
          </span>
        </div>
      </div>
    </div>

    <!-- System Diagnostics -->
    {#if shittimData?.stats}
      <div class="bg-white dark:bg-slate-800 p-4 rounded-lg shadow-sm border border-slate-200 dark:border-slate-700 transition-colors duration-300">
        <h3 class="text-sm font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-3 flex items-center gap-2">
          <Activity class="w-4 h-4" />
          System Diagnostics
        </h3>
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div class="p-3 bg-slate-50 dark:bg-slate-900/50 rounded border border-slate-100 dark:border-slate-700/50">
            <div class="text-xs text-slate-500 dark:text-slate-400 mb-1">Link Speed</div>
            <div class="font-mono text-lg font-semibold">{shittimData.fps.toFixed(1)} <span class="text-xs font-normal text-slate-400">FPS</span></div>
            <div class="text-[10px] text-slate-400 mt-1 leading-tight">Backend to Frontend Sync Rate</div>
          </div>
          <div class="p-3 bg-slate-50 dark:bg-slate-900/50 rounded border border-slate-100 dark:border-slate-700/50">
            <div class="text-xs text-slate-500 dark:text-slate-400 mb-1">Capture Rate</div>
            <div class="font-mono text-lg font-semibold">{shittimData.stats.received.toFixed(1)} <span class="text-xs font-normal text-slate-400">FPS</span></div>
            <div class="text-[10px] text-slate-400 mt-1 leading-tight">Raw Windows Capture Speed</div>
          </div>
          <div class="p-3 bg-slate-50 dark:bg-slate-900/50 rounded border border-slate-100 dark:border-slate-700/50">
            <div class="text-xs text-slate-500 dark:text-slate-400 mb-1">Process Rate</div>
            <div class="font-mono text-lg font-semibold">{shittimData.stats.accepted.toFixed(1)} <span class="text-xs font-normal text-slate-400">FPS</span></div>
            <div class="text-[10px] text-slate-400 mt-1 leading-tight">Internal processing speed</div>
          </div>
          <div class="p-3 bg-slate-50 dark:bg-slate-900/50 rounded border border-slate-100 dark:border-slate-700/50">
            <div class="text-xs text-slate-500 dark:text-slate-400 mb-1">Frame Drops</div>
            <div class="font-mono text-lg font-semibold {shittimData.stats.queue_full > 0 ? 'text-yellow-500' : ''}">
              {shittimData.stats.queue_full}
            </div>
            <div class="text-[10px] text-slate-400 mt-1 leading-tight">Frames dropped due to load</div>
          </div>
        </div>
      </div>
    {/if}

    <!-- Capture Tools -->
    <div class="bg-white dark:bg-slate-800 p-4 rounded-lg shadow-sm border border-slate-200 dark:border-slate-700 transition-colors duration-300">
      <h3 class="text-sm font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-3 flex items-center gap-2">
        <Camera class="w-4 h-4" />
        Capture Tools
      </h3>
      <div class="flex flex-col gap-3">
        <div class="flex items-center justify-between">
          <div>
            <p class="text-sm font-medium">Full Screenshot</p>
            <p class="text-xs text-slate-400 dark:text-slate-500">
              Save full game screen to output/
            </p>
          </div>
          <button
            onclick={() => invoke('save_full_screenshot')}
            disabled={!shittimData}
            class="{shittimData
              ? 'bg-blue-600 hover:bg-blue-700 active:bg-blue-800 text-white cursor-pointer'
              : 'bg-slate-300 dark:bg-slate-600 text-slate-500 cursor-not-allowed'} px-4 py-2 rounded-lg text-sm font-medium transition-colors"
          >
            Capture
          </button>
        </div>
        <div class="flex items-center justify-between">
          <div>
            <p class="text-sm font-medium">Timer ROI</p>
            <p class="text-xs text-slate-400 dark:text-slate-500">
              Save timer region image to output/
            </p>
          </div>
          <button
            onclick={() => invoke('trigger_screenshot')}
            disabled={!shittimData}
            class="{shittimData
              ? 'bg-blue-600 hover:bg-blue-700 active:bg-blue-800 text-white cursor-pointer'
              : 'bg-slate-300 dark:bg-slate-600 text-slate-500 cursor-not-allowed'} px-4 py-2 rounded-lg text-sm font-medium transition-colors"
          >
            Capture
          </button>
        </div>
        <div class="flex items-center justify-between">
          <div>
            <p class="text-sm font-medium">Cost ROI</p>
            <p class="text-xs text-slate-400 dark:text-slate-500">
              Save skew-corrected cost gauge image to output/
            </p>
          </div>
          <button
            onclick={() => invoke('save_cost_roi_image')}
            disabled={!shittimData}
            class="{shittimData
              ? 'bg-blue-600 hover:bg-blue-700 active:bg-blue-800 text-white cursor-pointer'
              : 'bg-slate-300 dark:bg-slate-600 text-slate-500 cursor-not-allowed'} px-4 py-2 rounded-lg text-sm font-medium transition-colors"
          >
            Capture
          </button>
        </div>
      </div>
    </div>

    <!-- Overlay Test -->
    <div class="bg-white dark:bg-slate-800 p-4 rounded-lg shadow-sm border border-slate-200 dark:border-slate-700 transition-colors duration-300">
      <h3 class="text-sm font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-3 flex items-center gap-2">
        <Layers class="w-4 h-4" />
        Overlay Test
      </h3>
      <div class="flex items-center justify-between">
        <div>
          <p class="text-sm font-medium">Test Overlay</p>
          <p class="text-xs text-slate-400 dark:text-slate-500">
            Show a card on the right edge of the game window
          </p>
        </div>
        <button
          onclick={onToggleOverlay}
          disabled={!shittimData}
          aria-label="Toggle overlay"
          class="relative inline-flex h-6 w-11 items-center rounded-full transition-colors {overlayActive ? 'bg-blue-600' : 'bg-slate-300 dark:bg-slate-600'} {!shittimData ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}"
        >
          <span
            class="inline-block h-4 w-4 rounded-full bg-white transition-transform {overlayActive ? 'translate-x-6' : 'translate-x-1'}"
          ></span>
        </button>
      </div>
    </div>
  </div>
</div>
