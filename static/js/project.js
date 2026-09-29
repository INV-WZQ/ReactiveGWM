// Control block j conditions source frames 4j+1 through 4j+4; frame 0 is the initial image.
function controlIndexAtTime(time, fps, introFrames, sourceFrames, controlCount) {
  const frame = Math.floor(time * fps + 1e-6) - introFrames;
  if (frame < 0) return -2;
  if (frame === 0) return -1;
  const framesPerControl = (sourceFrames - 1) / controlCount;
  return Math.min(controlCount - 1, Math.floor((frame - 1) / framesPerControl));
}

function actionLabel(action) {
  if (action === 'no_op') return 'No input';
  return action.split(' + ').map((part) => {
    const name = part.replaceAll('_', ' ');
    return name.charAt(0).toUpperCase() + name.slice(1);
  }).join(' + ');
}

document.querySelectorAll('.demo-media').forEach((media) => {
  const video = media.querySelector('video');
  const controls = media.querySelector('.live-controls');
  const phase = controls.querySelector('[data-control-phase]');
  const rows = [...controls.querySelectorAll('[data-actions]')].map((row) => ({
    row,
    value: row.querySelector('.live-action'),
    actions: JSON.parse(row.dataset.actions),
  }));
  const fps = Number(controls.dataset.fps);
  const introFrames = Number(controls.dataset.introFrames);
  const sourceFrames = Number(controls.dataset.sourceFrames);
  const controlCount = rows[0].actions.length;
  let lastIndex;
  let callbackId = null;
  const useVideoFrames = typeof video.requestVideoFrameCallback === 'function';

  const update = (time = video.currentTime) => {
    const index = controlIndexAtTime(time, fps, introFrames, sourceFrames, controlCount);
    if (index === lastIndex) return;
    lastIndex = index;
    const label = index === -2 ? 'Role guide' : index === -1 ? 'Initial frame' : '';
    phase.textContent = window.projectText ? window.projectText(label) : label;
    rows.forEach(({ row, value, actions }) => {
      const action = index < 0 ? null : actions[index];
      const label = action === null ? '—' : actionLabel(action);
      value.textContent = window.projectText ? window.projectText(label) : label;
      row.classList.toggle('is-active', action !== null && action !== 'no_op');
    });
  };
  const stop = () => {
    if (callbackId === null) return;
    if (useVideoFrames) video.cancelVideoFrameCallback(callbackId);
    else cancelAnimationFrame(callbackId);
    callbackId = null;
  };
  const schedule = () => {
    if (video.paused || video.ended || callbackId !== null) return;
    if (useVideoFrames) {
      callbackId = video.requestVideoFrameCallback((now, metadata) => {
        callbackId = null;
        update(metadata.mediaTime);
        schedule();
      });
    } else {
      callbackId = requestAnimationFrame(() => {
        callbackId = null;
        update();
        schedule();
      });
    }
  };

  ['loadedmetadata', 'loadeddata', 'timeupdate', 'seeking', 'seeked', 'emptied'].forEach((event) => {
    video.addEventListener(event, () => update());
  });
  video.addEventListener('play', () => { update(); schedule(); });
  ['pause', 'ended'].forEach((event) => {
    video.addEventListener(event, () => { stop(); update(); });
  });
  document.addEventListener('project-language-change', () => { lastIndex = undefined; update(); });
  update();
  controls.hidden = false;
});

document.querySelectorAll('.demo-carousel').forEach((carousel) => {
  const slides = [...carousel.querySelectorAll('.demo-card')];
  const counter = carousel.querySelector('[data-counter]');
  let current = 0;

  slides.forEach((slide, index) => {
    slide.hidden = index !== current;
    slide.setAttribute('aria-roledescription', 'slide');
    slide.setAttribute('aria-label', `${index + 1} of ${slides.length}`);
  });
  carousel.classList.add('is-ready');
  carousel.querySelector('.carousel-controls').hidden = false;

  carousel.querySelectorAll('[data-direction]').forEach((button) => {
    button.addEventListener('click', () => {
      const previous = slides[current];
      const previousVideo = previous.querySelector('video');
      previousVideo.pause();
      if (previousVideo.readyState > 0) previousVideo.currentTime = 0;
      previous.querySelector('details').open = false;
      previous.hidden = true;

      current = (current + Number(button.dataset.direction) + slides.length) % slides.length;
      const next = slides[current];
      next.hidden = false;
      counter.textContent = `${current + 1} / ${slides.length}`;

      const nextVideo = next.querySelector('video');
      if (nextVideo.readyState > 0) nextVideo.currentTime = 0;
      // Native controls remain available when the browser blocks playback.
      nextVideo.play().catch(() => {});
    });
  });
});

// Keep only one clip playing, including when moving between the two games.
const videos = [...document.querySelectorAll('.demo-carousel video')];
videos.forEach((video) => {
  video.addEventListener('play', () => {
    videos.forEach((other) => {
      if (other !== video) other.pause();
    });
  });
});
