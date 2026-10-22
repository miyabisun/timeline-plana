export interface ProcessCandidate {
  pid: number;
  name: string;
  window_title: string;
  hwnd: number;
}

export interface WindowGeometry {
  /** Client area top-left X in screen pixels */
  x: number;
  /** Client area top-left Y in screen pixels */
  y: number;
  /** Client area width */
  width: number;
  /** Client area height */
  height: number;
}

export interface ShittimPayload {
  battle_state: string;
  fps: number;
  stats?: {
    received: number;
    accepted: number;
    queue_full: number;
  };
  timer: null | {
    minutes: number;
    seconds: number;
    milliseconds: number;
  };
  cost: null | {
    current: number;
    max_cost: number;
    confidence: number;
  };
  window: WindowGeometry;
  /** What to use next, once a timeline is loaded */
  guidance: Guidance | null;
}

export type Trigger =
  | { kind: 'at'; ms: number; text: string }
  | { kind: 'cost'; cost: number };

export type TimelineRow =
  | {
      kind: 'skill';
      trigger: Trigger | null;
      skill: string;
      target: string | null;
      note: string | null;
    }
  | { kind: 'mark'; ms: number; time: string; note: string };

export interface Plan {
  boss: string;
  difficulty: string | null;
  students: string[];
  rows: TimelineRow[];
}

export interface GuideSkill {
  row: number;
  skill: string;
  target: string | null;
  note: string | null;
  done: boolean;
}

export interface Guidance {
  /** Row of the next skill to use; null once every skill is used */
  next: number | null;
  /** Trigger of the next skill's group; null: use it now */
  trigger: Trigger | null;
  /** Whether the next skill is due */
  due: boolean;
  skills: GuideSkill[];
  mark: { row: number; ms: number; time: string; note: string; soon: boolean } | null;
}

export type Theme = 'auto' | 'light' | 'dark';

export type TabId = 'check' | 'timeline' | 'debug' | 'settings';
