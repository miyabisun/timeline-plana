import { WebviewWindow } from '@tauri-apps/api/webviewWindow';
import { currentMonitor } from '@tauri-apps/api/window';
import type { WindowGeometry, Theme } from '../types';

export async function destroyOverlay(): Promise<void> {
  try {
    const existing = await WebviewWindow.getByLabel('overlay');
    if (existing) await existing.destroy();
  } catch { /* ignore */ }
}

export async function createOverlay(
  windowGeometry: WindowGeometry,
  theme: Theme,
): Promise<void> {
  const monitor = await currentMonitor();
  const scale = monitor?.scaleFactor ?? 1;
  const logicalX = windowGeometry.x / scale;
  const logicalY = windowGeometry.y / scale;
  const logicalW = windowGeometry.width / scale;
  const logicalH = windowGeometry.height / scale;
  const overlayWidth = 140;
  const overlayHeight = 120;
  const x = Math.round(logicalX + logicalW - overlayWidth);
  const y = Math.round(logicalY + logicalH / 2 - overlayHeight / 2);

  const resolvedTheme = theme === 'auto'
    ? (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')
    : theme;

  const webview = new WebviewWindow('overlay', {
    url: `/overlay.html?theme=${resolvedTheme}`,
    width: overlayWidth,
    height: overlayHeight,
    x,
    y,
    decorations: false,
    transparent: true,
    alwaysOnTop: true,
    skipTaskbar: true,
    focus: false,
    resizable: false,
    shadow: false,
  });

  webview.once('tauri://created', () => {
    webview.setIgnoreCursorEvents(true);
  });
}
