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
  window: WindowGeometry;
}

export type Theme = 'auto' | 'light' | 'dark';

export type TabId = 'check' | 'timeline' | 'debug' | 'settings';
