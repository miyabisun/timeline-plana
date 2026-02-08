<script lang="ts">
  import { invoke } from '@tauri-apps/api/core';
  import { CheckCircle2, Shield, Activity, Clock, Timer, Zap } from 'lucide-svelte';
  import type { ProcessCandidate, ShittimPayload } from '../../types';

  let { shittimData, targetInfo = $bindable() }: {
    shittimData: ShittimPayload | null;
    targetInfo: ProcessCandidate | null;
  } = $props();

  let status = $state("Searching for BlueArchive...");
  let isConnected = $state(false);
  let pollingId: ReturnType<typeof setInterval> | null = null;

  function stopPolling() {
    if (pollingId) {
      clearInterval(pollingId);
      pollingId = null;
    }
  }

  async function startCapture(hwnd: number) {
    try {
      status = "Target Found. Connecting...";
      await invoke('start_intercept_demo', { hwnd });
      status = "Connected";
      isConnected = true;
    } catch (e) {
      status = `Connection Error: ${e}`;
    }
  }

  async function checkTarget() {
    if (isConnected) return;

    try {
      const procs = await invoke<ProcessCandidate[]>('list_potential_targets');
      const target = procs.find(p => p.name === "BlueArchive.exe");

      if (target) {
        targetInfo = target;
        stopPolling();
        await startCapture(target.hwnd);
      }
    } catch (e) {
      console.error("Failed to list targets:", e);
    }
  }

  $effect(() => {
    checkTarget();
    pollingId = setInterval(checkTarget, 3000);

    return () => {
      stopPolling();
    };
  });
</script>

<div class="p-6 bg-slate-50 dark:bg-slate-900 min-h-full rounded-lg text-slate-800 dark:text-slate-100 transition-colors duration-300">
  <div class="mb-6 flex items-center gap-3">
    {#if isConnected}
      <CheckCircle2 class="w-8 h-8 text-green-500 dark:text-green-400" />
    {:else}
      <div class="w-8 h-8 rounded-full border-4 border-slate-300 dark:border-slate-600 border-t-blue-500 dark:border-t-blue-400 animate-spin"></div>
    {/if}
    <div>
      <h2 class="text-xl font-bold">{isConnected ? "Connected" : "Searching..."}</h2>
      <p class="text-slate-500 dark:text-slate-400 text-sm">{status}</p>
    </div>
  </div>

  <div class="grid grid-cols-1 gap-4">
    <!-- Battle Status Card -->
    <div class="bg-white dark:bg-slate-800 p-4 rounded-lg shadow-sm border border-slate-200 dark:border-slate-700 transition-colors duration-300">
      <h3 class="text-sm font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-3 flex items-center gap-2">
        <Shield class="w-4 h-4" />
        Battle Status
      </h3>

      <div class="flex flex-col items-start justify-center h-32 gap-2 pl-2">
        {#if shittimData}
          <div class="flex items-center gap-3">
            <div class="p-2 rounded-full transition-colors duration-300 {shittimData.battle_state === 'Active' ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400' : shittimData.battle_state === 'Paused' ? 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-400' : 'bg-slate-100 dark:bg-slate-700 text-slate-500 dark:text-slate-400'}">
              <Activity class="w-6 h-6" />
            </div>
            <span class="text-3xl font-bold tracking-tight">
              {shittimData.battle_state}
            </span>
          </div>
          <div class="flex flex-col gap-1 mt-1">
            <span class="text-xs text-slate-400 dark:text-slate-500 font-mono">
              System Linked
            </span>
          </div>
        {:else}
          <div class="text-slate-400 dark:text-slate-500 flex items-center gap-3">
            <Clock class="w-8 h-8 opacity-20" />
            <span>Waiting for Plana's Signal...</span>
          </div>
        {/if}
      </div>
    </div>

    <!-- Timer & Cost Cards -->
    {#if shittimData}
      <div class="grid grid-cols-2 gap-4">
        <!-- Timer Card -->
        <div class="bg-white dark:bg-slate-800 p-4 rounded-lg shadow-sm border border-slate-200 dark:border-slate-700 transition-colors duration-300">
          <h3 class="text-sm font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-3 flex items-center gap-2">
            <Timer class="w-4 h-4" />
            Timer
          </h3>
          {#if shittimData.timer}
            <div class="font-mono text-3xl font-bold tracking-tight text-center">
              {String(shittimData.timer.minutes).padStart(2, '0')}:{String(shittimData.timer.seconds).padStart(2, '0')}<span class="text-xl">.{String(shittimData.timer.milliseconds).padStart(3, '0')}</span>
            </div>
          {:else}
            <div class="text-slate-400 dark:text-slate-500 text-center text-sm">
              No active timer
            </div>
          {/if}
        </div>

        <!-- Cost Card -->
        <div class="bg-white dark:bg-slate-800 p-4 rounded-lg shadow-sm border border-slate-200 dark:border-slate-700 transition-colors duration-300">
          <h3 class="text-sm font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-3 flex items-center gap-2">
            <Zap class="w-4 h-4" />
            Cost
          </h3>
          {#if shittimData.cost}
            <div class="flex items-baseline justify-center gap-1">
              <span class="font-mono text-3xl font-bold tracking-tight">{shittimData.cost.current}</span>
              <span class="text-slate-400 dark:text-slate-500 text-lg">/ {shittimData.cost.max_cost}</span>
            </div>
            <!-- Cost bar -->
            <div class="mt-3 h-2 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
              <div
                class="h-full bg-blue-500 dark:bg-blue-400 rounded-full transition-all duration-300"
                style="width: {(shittimData.cost.current / shittimData.cost.max_cost) * 100}%"
              ></div>
            </div>
          {:else}
            <div class="text-slate-400 dark:text-slate-500 text-center text-sm">
              No cost data
            </div>
          {/if}
        </div>
      </div>
    {/if}
  </div>
</div>
