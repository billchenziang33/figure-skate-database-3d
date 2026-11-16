export function scheduleLibraryRefresh(refresh, {
  documentRef = globalThis.document,
  intervalMs = 30000,
  setIntervalFn = globalThis.setInterval,
  clearIntervalFn = globalThis.clearInterval,
} = {}) {
  const refreshIfVisible = () => {
    if (documentRef.visibilityState === 'visible') void refresh();
  };
  const onVisibilityChange = () => refreshIfVisible();
  const timer = setIntervalFn(refreshIfVisible, intervalMs);
  documentRef.addEventListener('visibilitychange', onVisibilityChange);
  return () => {
    clearIntervalFn(timer);
    documentRef.removeEventListener('visibilitychange', onVisibilityChange);
  };
}
