import assert from 'node:assert/strict';
import test from 'node:test';

import { phaseAt, readEventSegments, selectEventSegment } from './trajectory-hud.mjs';

const comboEvents = {
	TAKEOFF_4T: 166,
	APEX_4T: 188,
	LANDING_4T: 209,
	TAKEOFF_2T: 238,
	APEX_2T: 253,
	LANDING_2T: 267,
	EXIT: 427,
};

test('reads each named jump segment in chronological order', () => {
	assert.deepEqual(readEventSegments(comboEvents), [
		{ name: '4T', takeoff: 166, apex: 188, landing: 209 },
		{ name: '2T', takeoff: 238, apex: 253, landing: 267 },
	]);
});

test('labels a combination transition instead of ending the phase timeline after jump one', () => {
	const segments = readEventSegments(comboEvents);
	assert.equal(phaseAt(218, { segments }), 'Transition · simulated');
	assert.equal(phaseAt(236, { segments }), 'Transition · simulated');
	assert.equal(phaseAt(237, { segments }), '2T · Take-off · simulated');
});

test('labels the second jump apex from its own event frames', () => {
	const segments = readEventSegments(comboEvents);
	assert.equal(phaseAt(252, { segments }), '2T · Apex · simulated');
	assert.equal(selectEventSegment(252, segments).name, '2T');
});

test('returns to exit glide after the final jump landing', () => {
	const segments = readEventSegments(comboEvents);
	assert.equal(phaseAt(266, { segments }), '2T · Landing · simulated');
	assert.equal(phaseAt(426, { segments }), 'Exit glide · simulated');
});

test('selects the previous jump metrics during transition and switches on the next take-off', () => {
	const segments = readEventSegments(comboEvents);
	assert.equal(selectEventSegment(0, segments).name, '4T');
	assert.equal(selectEventSegment(218, segments).name, '4T');
	assert.equal(selectEventSegment(236, segments).name, '4T');
	assert.equal(selectEventSegment(237, segments).name, '2T');
	assert.equal(selectEventSegment(238, segments).name, '2T');
	assert.equal(selectEventSegment(426, segments).name, '2T');
});

test('keeps the single-jump phase labels unchanged', () => {
	const events = { takeoff: 16, apex: 32, landing: 48 };
	assert.equal(phaseAt(15, events), 'Take-off · simulated');
	assert.equal(phaseAt(31, events), 'Apex · simulated');
	assert.equal(phaseAt(48, events), 'Exit glide · simulated');
});

import { cumulativeTurns, formatTurns, turnsAt } from './trajectory-hud.mjs';

test('counts air turns across the ±π wrap without losing a revolution', () => {
	const yaws = Array.from({ length: 41 }, (_, i) => ((i * Math.PI * 2 * 4) / 40 + Math.PI) % (2 * Math.PI) - Math.PI);
	const turns = cumulativeTurns(yaws);
	assert.ok(Math.abs(turns.at(-1) - 4) < 1e-9);
	const segment = { takeoff: 166, apex: 186, landing: 206 };
	assert.equal(turnsAt(160, segment, turns), 0);
	assert.ok(Math.abs(turnsAt(185, segment, turns) - 2) < 1e-9);
	assert.equal(turnsAt(300, segment, turns), turns.at(-1));
	assert.equal(formatTurns(2, 4), '2.0 / 4.0');
});
