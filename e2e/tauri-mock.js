// A stand-in for the Tauri backend in a plain browser: commands answer from
// `window.__mock.answers`, every call is recorded in `window.__mock.calls`, and
// `emitLink` sends a `link-sync` event to the page's listeners.

/** Installs the mock before the page's own scripts run. */
export async function mockTauri(page, answers = {}) {
  await page.addInitScript((answers) => {
    const callbacks = new Map();
    const listeners = [];
    let nextId = 1;
    window.__mock = { answers, calls: [] };
    window.__TAURI_EVENT_PLUGIN_INTERNALS__ = { unregisterListener() {} };
    window.__TAURI_INTERNALS__ = {
      metadata: {
        currentWindow: { label: "main" },
        currentWebview: { windowLabel: "main", label: "main" },
      },
      transformCallback(callback) {
        const id = nextId++;
        callbacks.set(id, callback);
        return id;
      },
      unregisterCallback(id) {
        callbacks.delete(id);
      },
      async invoke(cmd, args) {
        if (cmd === "plugin:event|listen") {
          listeners.push(args);
          return nextId++;
        }
        if (cmd === "plugin:event|unlisten") return null;
        window.__mock.calls.push({ cmd, args });
        const answer = window.__mock.answers[cmd];
        if (answer && answer.error) throw answer.error;
        return answer ? answer.ok : null;
      },
    };
    window.__emitLink = (payload) => {
      for (const listener of listeners.filter((l) => l.event === "link-sync")) {
        callbacks.get(listener.handler)?.({ event: "link-sync", id: 0, payload });
      }
    };
  }, answers);
}

/** Sends one frame of the backend's `link-sync` event. */
export async function emitLink(page, payload) {
  await page.evaluate((payload) => window.__emitLink(payload), payload);
}

/** The commands the page called, without the connection panel's polling. */
export async function calls(page) {
  const all = await page.evaluate(() => window.__mock.calls);
  return all.filter((c) => c.cmd !== "list_potential_targets");
}

/** A frame of the running battle, with the given guidance. */
export function frame(guidance) {
  return {
    battle_state: "Active",
    fps: 60,
    stats: null,
    timer: { minutes: 3, seconds: 28, milliseconds: 0 },
    cost: { current: 3.0, max_cost: 10, confidence: 1 },
    window: { x: 0, y: 0, width: 1600, height: 900 },
    guidance,
  };
}
