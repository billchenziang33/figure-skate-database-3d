const $ = (selector) => document.querySelector(selector);
const cards = $('#cards');
const filter = $('#filter');
const search = $('#search');
const statusLine = $('#status-line');
const categoryNames = {jump:'单跳', combination:'连跳', spin:'旋转', sequence:'步法序列'};

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function makeCard(motion) {
  const card = el('article', 'card');
  const body = el(motion.file ? 'a' : 'div', 'card-link');
  if (motion.file) body.href = `./player.html?id=${encodeURIComponent(motion.id)}`;
  if (motion.poster) {
    const image = el('img', 'card-image');
    image.src = `./${motion.poster}`;
    image.alt = `${motion.name} · simulated side view`;
    image.loading = 'lazy';
    body.append(image);
  } else {
    body.append(el('div', 'card-image placeholder', motion.file ? '3D MODEL' : 'IN PRODUCTION'));
  }
  const content = el('div', 'card-content');
  const top = el('div', 'card-top');
  top.append(el('span', 'code', motion.id), el('span', 'occurrence', `${motion.occurrences ?? 0} 次节目出现`));
  content.append(top, el('h3', '', motion.name.replace(new RegExp(`^${motion.id}\\s*[·-]?\\s*`), '')));
  const bottom = el('div', 'card-bottom');
  const dots = el('span', 'round-dots');
  for (let i = 0; i < 3; i++) dots.append(el('span', `round-dot${i < motion.rounds.length ? ' done' : ''}`));
  bottom.append(dots, el('span', '', `${motion.rounds.length} / 3 轮已归档`));
  content.append(bottom);
  const badges = el('div', 'badges');
  if (motion.file) badges.append(el('span', 'badge ready', '模型可预览'));
  else badges.append(el('span', 'badge pending', '制作中 / 待开始'));
  if (motion.blender_geometry_reviewed && motion.blender_technique_reviewed) {
    badges.append(el('span', 'badge ready', 'Blender 检查通过'));
  }
  if (motion.public_side_view !== 'passed') badges.append(el('span', 'badge web', '网页侧视待验收'));
  content.append(badges);
  if (motion.file) content.append(el('div', 'play-tag', '打开第三轮 3D 预览 ↗'));
  body.append(content);
  card.append(body);
  return card;
}

function render(motions) {
  const visible = motions.filter((motion) => {
    const categoryMatch = filter.value === 'all' || motion.kind === filter.value;
    const query = search.value.trim().toLocaleLowerCase();
    const searchMatch = !query || `${motion.id} ${motion.name}`.toLocaleLowerCase().includes(query);
    return categoryMatch && searchMatch;
  });
  cards.replaceChildren();
  if (!visible.length) cards.append(el('div', 'empty', '没有符合条件的动作。'));
  else visible.forEach((motion) => cards.append(makeCard(motion)));
  statusLine.textContent = `显示 ${visible.length} / ${motions.length} 项 · ${motions.filter((motion) => motion.file).length} 项已有可播放模型`;
}

try {
  const [motionResponse, progressResponse] = await Promise.all([
    fetch('./motions.json'), fetch('./production-progress.json'),
  ]);
  if (!motionResponse.ok) throw new Error(`Action catalog returned ${motionResponse.status}`);
  const all = await motionResponse.json();
  const motions = all.filter((motion) => motion.kind !== 'baseline');
  const production = progressResponse.ok ? await progressResponse.json() : null;
  const generated = motions.filter((motion) => motion.file).length;
  const rounds = motions.reduce((sum, motion) => sum + motion.rounds.length, 0);
  const reviewed = motions.filter((motion) => motion.blender_geometry_reviewed && motion.blender_technique_reviewed).length;
  const accepted = motions.filter((motion) => motion.public_side_view === 'passed').length;

  $('#model-count').textContent = `${generated} / ${motions.length}`;
  Object.assign($('#model-progress'), {max:motions.length, value:generated});
  $('#round-count').textContent = `制作轮次 ${rounds} / ${motions.length * 3}`;
  $('#web-count').textContent = `公开验收 ${accepted} / ${motions.length}`;
  $('#metric-actions').textContent = `${generated} / ${motions.length}`;
  $('#metric-review').textContent = `${reviewed} / ${motions.length}`;
  $('#metric-web').textContent = `${accepted} / ${motions.length}`;
  const batch = production?.latest_batch;
  $('#latest-actions').textContent = batch?.assets?.join('、') || '—';
  $('#latest-detail').textContent = batch
    ? `${production.last_updated} · 新归档 ${batch.rounds_archived} 轮`
    : '动作资产同步后自动更新';
  $('#updated-at').textContent = production?.last_updated ? `Updated ${production.last_updated}` : '';
  filter.addEventListener('change', () => render(motions));
  search.addEventListener('input', () => render(motions));
  render(motions);
} catch (error) {
  statusLine.textContent = '动作目录暂时无法读取，请刷新页面后重试。';
  console.error(error);
}
