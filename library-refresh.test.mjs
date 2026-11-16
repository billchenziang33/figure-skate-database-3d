import assert from 'node:assert/strict';
import test from 'node:test';
import { scheduleLibraryRefresh } from './library-refresh.mjs';

test('refreshes when the page becomes visible and only polls while visible', () => {
  let visibilityListener;
  let timerCallback;
  let clearedTimer;
  const removedListeners = [];
  const calls = [];
  const documentRef = {
    visibilityState: 'hidden',
    addEventListener: (name, listener) => {
      assert.equal(name, 'visibilitychange');
      visibilityListener = listener;
    },
    removeEventListener: (name, listener) => removedListeners.push([name, listener]),
  };
  const stop = scheduleLibraryRefresh(() => calls.push('refresh'), {
    documentRef,
    intervalMs: 30000,
    setIntervalFn: (callback, intervalMs) => {
      assert.equal(intervalMs, 30000);
      timerCallback = callback;
      return 7;
    },
    clearIntervalFn: (timerId) => { clearedTimer = timerId; },
  });

  timerCallback();
  assert.deepEqual(calls, []);
  documentRef.visibilityState = 'visible';
  visibilityListener();
  assert.deepEqual(calls, ['refresh']);
  timerCallback();
  assert.deepEqual(calls, ['refresh', 'refresh']);

  stop();
  assert.equal(clearedTimer, 7);
  assert.deepEqual(removedListeners, [['visibilitychange', visibilityListener]]);
});
