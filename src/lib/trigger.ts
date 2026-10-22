import type { Trigger } from '../types';

/** A trigger as the timeline writes it: the remaining time, or the EX cost. */
export function triggerLabel(trigger: Trigger | null): string {
  if (!trigger) return 'Now';
  return trigger.kind === 'at' ? trigger.text : `Cost ${trigger.cost}`;
}
