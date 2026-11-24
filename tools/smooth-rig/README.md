# smooth-rig (r4)

Rebuilds every skater as one continuous skinned body (no rigid volumes) and re-authors its
motion from a pose library, while keeping all node names the website player looks up (root,
skates, cameras, ice, V102 event markers).

What is kept from the source GLB: the travel path and jump height (pelvis trajectory), the
body yaw (turn counts stay exactly as audited), the event frames, the skate meshes, cameras
and rink. What is replaced: every body pose.

1. `author_motion.py` (numpy): builds a key-pose timeline per element and solves the body.
   - Jumps: entry per type (toe loop / flip / Lutz pick, Salchow sweep, loop sit, Axel
     forward swing), tight air position scaled by the number of turns, one-foot landing with
     the free leg extended and the arms open. Euler lands on the left foot.
   - Spins: camel, sit, upright (arms pull in overhead), flying entries, change of foot. The
     total turns of the source are kept; the rate follows the position (tight = faster).
   - Sequences: StSq (three-turn, twizzle, bracket, toe steps), ChSq (spiral, spread eagle, lunge).
   - Fixed adult bone lengths, monotone-cubic interpolation between key poses, two-bone leg
     IK, grounded blades exactly on the ice, free blades never through it.
   - The arms rest on the body surface: points along each arm are pushed out of the torso
     and head volumes, so a folded arm does not pass through the chest.
2. `build_blender.py` (Blender 5.x, background): removes the rigid volumes, builds the skin
   mesh + armature parented to the skater root, keys the bones and skate parts, exports GLB.
   The arm weights are computed, not automatic: each vertex is weighted by its distance
   along and from the straight rest arm, and helper bones `shoulder.l/r` and `elbow.l/r`
   turn by half of the joint angle. With automatic weights the chest followed the upper arm
   and the shoulder caved in. Check a change with close-up renders of the shoulders.
3. `batch.py`: runs 1+2 for all 50 actions (4 in parallel) from the sources listed in
   `r4-build-report.json`; writes `<id>-r4-smooth-web.glb` and preview sheets.
4. `stick.py`: stick-figure contact sheet of an authored `.npz` (seconds, no Blender), for
   tuning poses. `process_motion.py` is the glTF reader shared by the scripts.

Blender needs GPU access, so run it outside a sandbox:
`SMOOTH_RIG_WORK=/tmp/smooth-rig-work python3 tools/smooth-rig/batch.py [id[:frame,frame,…]]`.
Copy `sheets/results.json` over `r4-build-report.json` and update the sha256 values in
`player-motions.json` when a build is accepted.
A rebuild gives the same motion, but not the same bytes: Blender's automatic skin weights
vary in the last float digits, so the sha256 changes on every build.
Everything remains illustrative / simulated; no Beijing 2022 athlete capture.
