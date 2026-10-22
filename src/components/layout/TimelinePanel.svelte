<script lang="ts">
  import { invoke } from '@tauri-apps/api/core';
  import { CornerDownRight, FileUp, Layers, LayoutDashboard, TriangleAlert } from 'lucide-svelte';
  import { triggerLabel } from '../../lib/trigger';
  import type { Plan, ShittimPayload } from '../../types';

  let { plan = $bindable(), shittimData, overlayActive, onToggleOverlay }: {
    plan: Plan | null;
    shittimData: ShittimPayload | null;
    overlayActive: boolean;
    onToggleOverlay: () => void;
  } = $props();

  let errors = $state<string[]>([]);
  let fileName = $state('');

  const next = $derived(shittimData?.guidance?.next ?? null);

  async function open(event: Event) {
    const input = event.currentTarget as HTMLInputElement;
    const file = input.files?.[0];
    input.value = '';
    if (!file) return;
    try {
      plan = await invoke<Plan>('load_timeline', { yaml: await file.text() });
      fileName = file.name;
      errors = [];
    } catch (e) {
      errors = Array.isArray(e) ? e.map(String) : [String(e)];
    }
  }

  async function goOnFrom(row: number) {
    await invoke<boolean>('select_timeline_row', { row });
  }
</script>

<div class="p-4 min-h-full text-slate-800 dark:text-slate-100">
  <div class="flex flex-wrap items-center gap-2 mb-4">
    <label class="px-3 py-1.5 rounded-md text-sm font-bold bg-blue-600 text-white hover:bg-blue-500 cursor-pointer flex items-center gap-2">
      <FileUp class="w-4 h-4" />
      Open YAML
      <input type="file" accept=".yaml,.yml" class="hidden" onchange={open} />
    </label>
    <button
      onclick={onToggleOverlay}
      disabled={!shittimData}
      title={shittimData ? '' : 'Connect to the game first'}
      class="px-3 py-1.5 rounded-md text-sm font-bold border flex items-center gap-2 transition-colors disabled:opacity-40 disabled:cursor-not-allowed {overlayActive
        ? 'bg-slate-700 text-white border-slate-700 dark:bg-slate-200 dark:text-slate-900'
        : 'border-slate-300 dark:border-slate-600 hover:bg-slate-100 dark:hover:bg-slate-800'}"
    >
      <Layers class="w-4 h-4" />
      {overlayActive ? 'Hide overlay' : 'Show overlay'}
    </button>
    {#if plan}
      <span class="text-sm text-slate-500 dark:text-slate-400 truncate">
        {plan.boss}{plan.difficulty ? ` ${plan.difficulty}` : ''} · {fileName}
      </span>
    {/if}
  </div>

  {#if errors.length}
    <div class="mb-4 p-3 rounded-md border border-red-300 dark:border-red-800 bg-red-50 dark:bg-red-950/40 text-sm">
      <p class="font-bold flex items-center gap-2 text-red-700 dark:text-red-300">
        <TriangleAlert class="w-4 h-4" />
        This file cannot be used
      </p>
      <ul class="mt-1 list-disc pl-6">
        {#each errors as error}
          <li>{error}</li>
        {/each}
      </ul>
    </div>
  {/if}

  {#if plan}
    <table class="w-full text-sm border-collapse">
      <thead>
        <tr class="text-left text-xs uppercase tracking-wider text-slate-500 dark:text-slate-400">
          <th class="py-1 px-2 w-24">When</th>
          <th class="py-1 px-2">Skill</th>
          <th class="py-1 px-2">Target</th>
          <th class="py-1 px-2">Note</th>
        </tr>
      </thead>
      <tbody>
        {#each plan.rows as row, i}
          {#if row.kind === 'skill'}
            <tr
              onclick={() => goOnFrom(i)}
              title="Go on from this skill"
              class="cursor-pointer border-t border-slate-200 dark:border-slate-800 hover:bg-slate-100 dark:hover:bg-slate-800/60 {next === i
                ? 'bg-blue-50 dark:bg-blue-950/50 font-bold'
                : next !== null && i < next
                  ? 'text-slate-400 dark:text-slate-500'
                  : ''}"
            >
              <td class="py-1 px-2 font-mono">
                {#if next === i}<span aria-label="next">▶</span>{/if}
                {#if row.trigger}
                  {triggerLabel(row.trigger)}
                {:else}
                  <CornerDownRight class="w-3.5 h-3.5 inline text-slate-400" aria-label="right after" />
                {/if}
              </td>
              <td class="py-1 px-2">{row.skill}</td>
              <td class="py-1 px-2">{row.target ?? ''}</td>
              <td class="py-1 px-2 text-slate-500 dark:text-slate-400">{row.note ?? ''}</td>
            </tr>
          {:else}
            <tr class="border-t border-slate-200 dark:border-slate-800 text-slate-500 dark:text-slate-400 italic">
              <td class="py-1 px-2 font-mono">{row.time}</td>
              <td class="py-1 px-2" colspan="3">{row.note}</td>
            </tr>
          {/if}
        {/each}
      </tbody>
    </table>
  {:else if !errors.length}
    <div class="text-center text-slate-400 dark:text-slate-500 py-16">
      <LayoutDashboard class="w-16 h-16 mx-auto mb-4 opacity-20" />
      <p>Open a timeline YAML made in Timeline Arona.</p>
    </div>
  {/if}
</div>
