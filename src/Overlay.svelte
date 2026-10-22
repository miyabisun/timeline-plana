<script lang="ts">
  import { listen } from '@tauri-apps/api/event';
  import { onDestroy } from 'svelte';
  import { triggerLabel } from './lib/trigger';
  import type { Guidance, ShittimPayload } from './types';

  const theme = new URLSearchParams(location.search).get('theme') === 'light' ? 'light' : 'dark';

  let guidance = $state<Guidance | null>(null);

  const unlisten = listen<ShittimPayload>('link-sync', (event) => {
    guidance = event.payload.guidance;
  });
  onDestroy(() => {
    unlisten.then((stop) => stop());
  });
</script>

{#if guidance && guidance.skills.length}
  <div class="card {theme}" class:due={guidance.due}>
    <div class="trigger">
      {#if guidance.due}<span class="now">NOW</span>{/if}
      {triggerLabel(guidance.trigger)}
    </div>
    <div class="skills">
      {#each guidance.skills as skill}
        <div class="skill-row" class:done={skill.done} class:next={skill.row === guidance.next}>
          {skill.row === guidance.next ? '▶ ' : ''}{skill.skill}
        </div>
        {#if skill.target}
          <div class="target">→ {skill.target}</div>
        {/if}
      {/each}
    </div>
    {#if guidance.mark?.soon}
      <div class="mark">{guidance.mark.time} {guidance.mark.note}</div>
    {/if}
  </div>
{/if}

<style>
  :global(*) {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }
  :global(html),
  :global(body) {
    width: 100%;
    height: 100%;
    background: transparent;
    overflow: hidden;
  }
  .card {
    width: fit-content;
    min-width: 95px;
    max-width: 175px;
    margin-left: auto;
    border-radius: 8px 0 0 8px;
    overflow: hidden;
    font-family: 'Yu Gothic UI', 'Segoe UI', sans-serif;
  }
  .card.dark {
    background: rgba(0, 0, 0, 0.75);
    color: white;
  }
  .card.light {
    background: rgba(230, 230, 230, 0.88);
    color: #1a1a1a;
  }
  .trigger {
    padding: 6px 12px;
    font-size: 15px;
    font-weight: bold;
    text-align: center;
    letter-spacing: 0.05em;
  }
  .dark .trigger {
    background: rgba(255, 255, 255, 0.1);
    border-bottom: 1px solid rgba(255, 255, 255, 0.15);
  }
  .light .trigger {
    background: rgba(0, 0, 0, 0.06);
    border-bottom: 1px solid rgba(0, 0, 0, 0.1);
  }
  .due .trigger {
    background: rgba(37, 99, 235, 0.85);
    color: white;
  }
  .now {
    font-size: 11px;
    margin-right: 4px;
  }
  .skills {
    padding: 6px 12px;
  }
  .skill-row {
    padding: 3px 0;
    font-size: 14px;
    font-weight: 600;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .skill-row.done {
    opacity: 0.4;
    text-decoration: line-through;
  }
  .target {
    padding: 0 0 3px 8px;
    font-size: 12px;
    font-weight: normal;
    opacity: 0.7;
  }
  .mark {
    padding: 4px 12px 6px;
    font-size: 11px;
    opacity: 0.8;
    border-top: 1px solid rgba(127, 127, 127, 0.3);
  }
</style>
