<script lang="ts">
  import { Settings, Laptop, Sun, Moon } from "lucide-svelte";
  import type { Theme } from '../../types';

  let { theme = $bindable() }: { theme: Theme } = $props();
</script>

<div class="p-8 max-w-2xl mx-auto">
  <div class="bg-white dark:bg-slate-900 rounded-lg shadow-sm border border-slate-200 dark:border-slate-800 p-6">
    <h2 class="text-lg font-semibold mb-4 flex items-center gap-2">
      <Settings class="w-5 h-5" />
      Appearance
    </h2>

    <div class="space-y-4">
      <div class="flex flex-col gap-2">
        <span class="text-sm font-medium text-slate-700 dark:text-slate-300">Theme</span>
        <div class="grid grid-cols-3 gap-2 p-1 bg-slate-100 dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700">

          <button
            onclick={() => theme = 'auto'}
            class="flex items-center justify-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-all {theme === 'auto'
              ? 'bg-white dark:bg-slate-700 text-blue-600 dark:text-blue-400 shadow-sm'
              : 'text-slate-500 hover:text-slate-700 dark:hover:text-slate-300'}"
          >
            <Laptop class="w-4 h-4" />
            Auto
          </button>

          <button
            onclick={() => theme = 'light'}
            class="flex items-center justify-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-all {theme === 'light'
              ? 'bg-white dark:bg-slate-700 text-blue-600 dark:text-blue-400 shadow-sm'
              : 'text-slate-500 hover:text-slate-700 dark:hover:text-slate-300'}"
          >
            <Sun class="w-4 h-4" />
            Light
          </button>

          <button
            onclick={() => theme = 'dark'}
            class="flex items-center justify-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-all {theme === 'dark'
              ? 'bg-white dark:bg-slate-700 text-blue-600 dark:text-blue-400 shadow-sm'
              : 'text-slate-500 hover:text-slate-700 dark:hover:text-slate-300'}"
          >
            <Moon class="w-4 h-4" />
            Dark
          </button>

        </div>
        <p class="text-xs text-slate-400 dark:text-slate-500 mt-1">
          Choose "Auto" to sync with your system preferences.
        </p>
      </div>
    </div>
  </div>
</div>
