// Car Placement Canvas - in-node interactive canvas.
//
// Shows the background plate and the car cutout ON the node, lets you drag the cutout,
// resize it from the corner handles (proportional) and rotate it, then writes x / y /
// scale / rotation back into the node's widgets so a headless /prompt run composites
// exactly the frame you see.
//
// Image sources, in priority order:
//   1. what the server last executed (temp cpc_<nodeid>_bg.png / _car.png) - so a plate
//      coming out of an upstream node shows up here automatically after one run.
//   2. the env_file / angle_file dropdowns: input/car_env/ plates + input/car_angles/ cutouts.
// Dropping an image file on the node uploads it to input/ and selects it in the dropdown.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const MIN_W = 260;
const MAX_W = 620;
const MIN_H = 200;
const MAX_H = 460;
const HANDLE = 16;
const ROT_ARM = 34;

function widget(node, name) {
  return (node.widgets || []).find((w) => w.name === name);
}

function setWidget(node, name, value) {
  const w = widget(node, name);
  if (!w || w.value === value) return;
  w.value = value;
  if (typeof w.callback === "function") w.callback(value, app.canvas, node);
}

const loadErrors = [];
const loadWarnings = [];

// createImageBitmap() is strict: it throws InvalidStateError on a PNG whose trailing
// bytes are missing (an interrupted upload), even though a plain <img> renders that
// exact file. Decode through <img> + drawImage as a second path so a slightly damaged
// plate still appears instead of silently falling back to a stashed frame.
function bitmapViaImg(url) {
  return new Promise((resolve, reject) => {
    const im = new Image();
    im.onload = () => {
      try {
        const c = document.createElement("canvas");
        c.width = im.naturalWidth || 1;
        c.height = im.naturalHeight || 1;
        c.getContext("2d").drawImage(im, 0, 0);
        resolve(c);
      } catch (e) {
        reject(e);
      }
    };
    im.onerror = () => reject(new Error("the browser could not decode the file"));
    im.src = url;
  });
}

async function loadBitmap(url) {
  // same origin as the ComfyUI page, so a plain fetch avoids depending on the
  // frontend's api helper shape (which differs across frontend versions).
  let status = 0;
  try {
    const r = await fetch(url, { cache: "no-store" });
    status = r.status;
    if (r.ok) {
      const b = await r.blob();
      try {
        const bm = await createImageBitmap(b);
        if (bm) return bm;
      } catch (e) {
        // the strict decoder refused the bytes - use the tolerant path and report it
        try {
          const c = await bitmapViaImg(url);
          loadWarnings.push(`${url.split("?")[0]}: shown via <img> fallback (${e.name || e})`);
          return c;
        } catch (e2) {
          loadErrors.push(`${e2.message || e2} ${url}`);
          return null;
        }
      }
    }
  } catch (e) {
    loadErrors.push(`${e} ${url}`);
    return null;
  }
  loadErrors.push(`${status} ${url}`);
  try {
    return await bitmapViaImg(url);
  } catch (e) {
    return null;
  }
}

async function uploadImage(file, subfolder) {
  const fd = new FormData();
  fd.append("image", file);
  fd.append("overwrite", "true");
  fd.append("type", "input");
  if (subfolder) fd.append("subfolder", subfolder);
  const r = await api.fetchApi("/upload/image", { method: "POST", body: fd });
  if (!r.ok) throw new Error("upload failed " + r.status);
  const j = await r.json();
  // the subfolder is already part of the widget's option namespace, so the stored value
  // is the bare file name (a "car_angles/x.png" value would double the folder)
  return j.name;
}

// The canvas can paint anything the browser decodes, but the server composite re-opens the
// file with Pillow. Formats Pillow reads natively are uploaded byte-for-byte; anything else
// (heic / avif / svg / exr ...) is re-encoded to PNG once, so a layer in any format still
// renders in the graph instead of failing at composite time.
const NATIVE_EXT = ["png", "jpg", "jpeg", "jfif", "webp", "bmp", "dib", "tif", "tiff",
  "gif", "tga", "ico", "ppm", "pgm", "pbm", "pcx", "dds", "jp2", "psd", "sgi", "xbm"];

function extOf(name) {
  const m = /\.([^.]+)$/.exec(String(name || ""));
  return m ? m[1].toLowerCase() : "";
}

function needsPNG(name) {
  const e = extOf(name);
  return e !== "" && !NATIVE_EXT.includes(e);
}

// <img> + drawImage (not createImageBitmap) so svg and other non-bitmap inputs decode too
async function toPNG(file) {
  const url = URL.createObjectURL(file);
  try {
    const c = await bitmapViaImg(url);
    const blob = await new Promise((res, rej) =>
      c.toBlob((b) => (b ? res(b) : rej(new Error("could not re-encode"))), "image/png"));
    return new File([blob], String(file.name).replace(/\.[^.]+$/, "") + ".png",
      { type: "image/png" });
  } finally {
    URL.revokeObjectURL(url);
  }
}

function selectFile(node, name, filename) {
  const w = widget(node, name);
  if (!w) return;
  const values = w.options?.values;
  if (Array.isArray(values) && !values.includes(filename)) values.push(filename);
  w.value = filename;
  if (typeof w.callback === "function") w.callback(filename, app.canvas, node);
}

function buildUI(node) {
  const state = {
    bg: null,
    car: null,
    out: null,
    showResult: false,
    drag: null,
    dispW: 320,
    dispH: 200,
    ready: 0,
    // the selection whose frames are actually on the canvas: the stashed temp frames
    // only win while they belong to it, and the pointer says whether a load is in flight
    loadedSel: null,
    pending: null,
    // images removed from each section (names only; the files stay in input/<folder>/)
    hidden: { env_file: [], angle_file: [] },
    // layer switches + the lock: kept in ui_state so a reload restores the frozen setup
    locked: false,
    layers: { car: true, env: true },
  };

  const container = document.createElement("div");
  container.className = "cpc-wrap";
  const bar = document.createElement("div");
  bar.className = "cpc-bar";
  const nameEl = document.createElement("span");
  nameEl.className = "cpc-name";
  const mkBtn = (label, title, fn) => {
    const b = document.createElement("button");
    b.className = "cpc-btn";
    b.textContent = label;
    b.title = title;
    b.onclick = (e) => {
      e.preventDefault();
      fn();
    };
    return b;
  };
  const resBtn = mkBtn("result", "toggle: show the last composited frame", () => {
    state.showResult = !state.showResult;
    resBtn.classList.toggle("on", state.showResult);
    draw();
  });
  const resetBtn = mkBtn("reset", "fit the cutout to 45% of the frame again", () => {
    setWidget(node, "x", 0.5);
    setWidget(node, "y", 0.62);
    setWidget(node, "scale", 0.45);
    setWidget(node, "rotation", 0);
    draw();
  });
  // lock the placement once it looks right: the canvas stops taking drags, so a later
  // click near the node can't nudge a finished composite.
  const lockBtn = mkBtn("lock", "freeze the cutout placement (no drag / resize / rotate / zoom)", () => {
    state.locked = !state.locked;
    syncBar();
    store();
    draw();
  });
  const delBtn = mkBtn("del layer", "delete the car cutout layer; the environment plate stays", () => {
    state.layers.car = false;
    state.car = null;
    setWidget(node, "angle_file", "(none)");
    store();
    draw();
  });
  const clearBtn = mkBtn("clear all", "delete every layer (car cutout and environment plate)", () => {
    state.layers.car = false;
    state.layers.env = false;
    state.car = null;
    state.bg = null;
    state.loadedSel = null;
    setWidget(node, "env_file", "(none)");
    setWidget(node, "angle_file", "(none)");
    store();
    draw();
  });
  bar.append(resBtn, resetBtn, lockBtn, delBtn, clearBtn, nameEl);

  const canvas = document.createElement("canvas");
  canvas.className = "cpc-canvas";
  container.append(bar, canvas);

  // --------------------------------------------- two tick-select image sections
  // One strip per source folder. Tiles are built from the combo widget's own option
  // list, so a new file uploaded into the folder shows up as soon as the combo grows.
  // Clicking a tile ticks it and drives the combo, which drives refresh() below.
  const strips = [];

  function viewURL(file, subfolder, thumb) {
    const q = `filename=${encodeURIComponent(file)}&type=input`;
    const sub = subfolder ? `&subfolder=${encodeURIComponent(subfolder)}` : "";
    return `/view?${q}${sub}${thumb ? "&preview=webp%3B50" : ""}`;
  }

  // Tiles are ~73px wide, so a cached 176px webp is plenty; the full-resolution /view
  // re-encode made the strips take ~20s to paint.
  function thumbURL(file, subfolder) {
    return `/car_placement_canvas/thumb?folder=${encodeURIComponent(subfolder || "")}&file=${encodeURIComponent(file)}`;
  }

  // Same cache, read straight out of the input tree: works even before the server has
  // been restarted with the thumbnail route registered.
  function thumbViewURL(file, subfolder) {
    const stem = String(file).replace(/\.[^.]+$/, "");
    return `/view?filename=${encodeURIComponent(stem + ".webp")}&type=input&subfolder=${encodeURIComponent("car_thumbs/" + (subfolder || ""))}`;
  }

  function buildStrip(label, name, subfolder) {
    const sect = document.createElement("div");
    sect.className = "cpc-sect";
    const hd = document.createElement("div");
    hd.className = "cpc-sect-hd";
    const ttl = document.createElement("span");
    // per-section upload: this is the only path into the folder, so it's on the header
    const up = mkBtn("+ upload", `upload an image into input/${subfolder}/`, () => pick(name, subfolder));
    // removed images are only hidden from the strip, so they can be brought back
    const undo = mkBtn("\u21ba", `bring back the images removed from ${label}`, () => {
      state.hidden[name] = [];
      store();
      rebuildStrips();
    });
    undo.style.display = "none";
    const toggle = document.createElement("button");
    toggle.className = "cpc-btn";
    toggle.textContent = "hide";
    const acts = document.createElement("span");
    acts.className = "cpc-sect-acts";
    const strip = document.createElement("div");
    strip.className = "cpc-strip";
    toggle.onclick = (e) => {
      e.preventDefault();
      const hid = strip.style.display === "none";
      strip.style.display = hid ? "" : "none";
      toggle.textContent = hid ? "hide" : "show";
      layout();
    };
    acts.append(up, undo, toggle);
    hd.append(ttl, acts);
    sect.append(hd, strip);

    const tiles = [];
    function sync() {
      const cur = widget(node, name)?.value;
      for (const t of tiles) t.el.classList.toggle("sel", t.value === cur);
    }
    function rebuild() {
      strip.textContent = "";
      tiles.length = 0;
      const hid = state.hidden[name] || [];
      const vals = (widget(node, name)?.options?.values || []).filter((v) => !hid.includes(v));
      for (const v of vals) {
        const card = document.createElement("div");
        card.className = "cpc-thumb";
        card.title = `${v}\n${subfolder ? "input/" + subfolder : "input"}`;
        if (v === "(none)") {
          card.classList.add("cpc-thumb-none");
          const t = document.createElement("span");
          t.textContent = "none";
          card.append(t);
        } else {
          const im = document.createElement("img");
          im.setAttribute("loading", "lazy");
          im.setAttribute("decoding", "async");
          im.alt = v;
          im.src = thumbURL(v, subfolder);
          im.onerror = () => {
            // route not registered yet (server not restarted) -> cached webp in the input tree
            if (!im.dataset.fb) {
              im.dataset.fb = "1";
              im.src = thumbViewURL(v, subfolder);
              return;
            }
            // thumb missing (image added after the cache was built) -> full-size preview
            if (im.dataset.fb === "1") {
              im.dataset.fb = "2";
              im.src = viewURL(v, subfolder, true);
              return;
            }
            im.style.display = "none";
            card.classList.add("cpc-thumb-err");
          };
          card.append(im);
          const lbl = document.createElement("div");
          lbl.className = "cpc-thumb-lbl";
          lbl.textContent = v.replace(/\.[a-z0-9]+$/i, "");
          card.append(lbl);
        }
        const tick = document.createElement("span");
        tick.className = "cpc-tick";
        tick.textContent = "\u2713";
        card.append(tick);
        if (v !== "(none)") {
          // per-image delete, in both sections: drop this one image from the picker
          const x = document.createElement("button");
          x.className = "cpc-x";
          x.textContent = "\u00d7";
          x.title = `remove ${v} from ${label}`;
          x.onclick = (e) => {
            e.preventDefault();
            e.stopPropagation();
            removeImage(name, label, v);
          };
          card.append(x);
        }
        card.onclick = (e) => {
          e.preventDefault();
          e.stopPropagation();
          choose(name, v);
          sync();
        };
        strip.append(card);
        tiles.push({ value: v, el: card });
      }
      const shown = vals.filter((v) => v !== "(none)").length;
      ttl.textContent = `${label} (${shown})` + (hid.length ? `  ${hid.length} removed` : "");
      undo.textContent = `\u21ba ${hid.length}`;
      undo.style.display = hid.length ? "" : "none";
      sync();
    }
    strips.push({ name, rebuild, sync, tiles });
    rebuild();
    return sect;
  }

  function rebuildStrips() {
    for (const s of strips) s.rebuild();
    // a new or removed tile changes how many rows each strip wraps to, and nothing
    // else runs after a strip rebuild, so the height has to be re-fitted here
    layout();
  }

  // Delete a single image from a section. The tile disappears and the name is kept in
  // ui_state, so it stays gone across reloads; the file itself is left on disk in
  // input/<folder>/ and "\u21ba n" in the section header brings the names back.
  function removeImage(name, label, file) {
    const hid = state.hidden[name] || (state.hidden[name] = []);
    if (!hid.includes(file)) hid.push(file);
    if (widget(node, name)?.value === file) {
      // the deleted image was the live selection: fall back to what is left
      const left = (widget(node, name)?.options?.values || []).filter((v) => v !== file && !hid.includes(v));
      choose(name, left.find((v) => v !== "(none)") || "(none)");
      // the strip has to drop the tile too: choose() only repaints the canvas
      rebuildStrips();
    } else {
      store();
      rebuildStrips();
    }
    syncTicks();
    const rest = (state.hidden[name] || []).length;
    nameEl.title = `${file} removed from ${label} (${rest} removed; \u21ba restores)`;
  }

  function syncTicks() {
    for (const s of strips) s.sync();
  }

  // one path for every "this file is now the selected layer": ticking a tile, uploading,
  // or dropping a file. Selecting a layer also un-deletes it, so del layer is undoable.
  function choose(name, filename) {
    // picking a plate/cutout means re-composing: leaving the result view on would keep
    // painting the last composited frame, so the click looks like it did nothing
    if (state.showResult) {
      state.showResult = false;
      resBtn.classList.toggle("on", false);
    }
    // an image that was removed from the section comes back when it is picked again
    const hid = state.hidden[name];
    if (hid && hid.length) state.hidden[name] = hid.filter((x) => x !== filename);
    selectFile(node, name, filename);
    if (name === "env_file") state.layers.env = true;
    if (name === "angle_file") state.layers.car = true;
    store();
    syncTicks();
    refresh();
  }

  const envStrip = buildStrip("background layer", "env_file", "car_env");
  const carStrip = buildStrip("object layer", "angle_file", "car_angles");
  container.append(envStrip, carStrip);

  const fileInput = document.createElement("input");
  fileInput.type = "file";
  // any format, not just the ones Windows maps to an image MIME: "image/*" alone hides
  // .heic / .avif / .tga / .psd / .jp2 files in the picker, so list them explicitly too
  fileInput.accept = "image/*," + [
    "png", "jpg", "jpeg", "jfif", "webp", "bmp", "dib", "gif", "tif", "tiff", "tga",
    "ico", "avif", "heic", "heif", "jp2", "j2k", "psd", "ppm", "pgm", "pbm", "pcx",
    "dds", "svg", "xbm", "exr", "hdr",
  ].map((e) => "." + e).join(",");
  fileInput.style.display = "none";
  container.append(fileInput);
  // one permanent handler: a section's "+ upload" button only sets the destination
  let pickTarget = null;
  fileInput.onchange = async () => {
    const f = fileInput.files?.[0];
    fileInput.value = "";
    if (!f || !pickTarget) return;
    let up = f;
    let note = "";
    if (needsPNG(f.name)) {
      try {
        up = await toPNG(f);
        note = `${f.name}: re-encoded to PNG so the server composite can read it`;
      } catch (e) {
        note = `${f.name}: uploaded as-is, this browser could not decode it (${e.message || e})`;
      }
    }
    const name = await uploadImage(up, pickTarget.subfolder);
    choose(pickTarget.widgetName, name);
    if (note) loadWarnings.push(note);
    // the new file must appear as a tile: the combo grew in choose(), so rebuild after
    rebuildStrips();
  };

  function pick(widgetName, subfolder) {
    pickTarget = { widgetName, subfolder };
    fileInput.click();
  }

  // ---------------------------------------------------------------- geometry
  function num(nm, fallback) {
    const w = widget(node, nm);
    const v = w ? Number(w.value) : NaN;
    return Number.isFinite(v) ? v : fallback;
  }

  function cutRect() {
    const scale = num("scale", 0.45);
    const car = state.car;
    const ar = car ? car.height / car.width : 0.45;
    const w = scale * state.dispW;
    const h = w * ar;
    return {
      cx: num("x", 0.5) * state.dispW,
      cy: num("y", 0.62) * state.dispH,
      w,
      h,
      rot: (num("rotation", 0) * Math.PI) / 180,
    };
  }

  function corners(r) {
    const pts = [
      [-r.w / 2, -r.h / 2],
      [r.w / 2, -r.h / 2],
      [r.w / 2, r.h / 2],
      [-r.w / 2, r.h / 2],
    ];
    const c = Math.cos(r.rot);
    const s = Math.sin(r.rot);
    return pts.map(([dx, dy]) => ({
      x: r.cx + dx * c - dy * s,
      y: r.cy + dx * s + dy * c,
    }));
  }

  function rotHandle(r) {
    const c = Math.cos(r.rot);
    const s = Math.sin(r.rot);
    const dy = -r.h / 2 - ROT_ARM;
    // a big cutout pushes the arm off the canvas, so keep the handle grabbable
    const m = 10;
    return {
      x: Math.min(state.dispW - m, Math.max(m, r.cx - dy * s)),
      y: Math.min(state.dispH - m, Math.max(m, r.cy + dy * c)),
    };
  }

  function toLocal(r, px, py) {
    const c = Math.cos(-r.rot);
    const s = Math.sin(-r.rot);
    const dx = px - r.cx;
    const dy = py - r.cy;
    return { x: dx * c - dy * s, y: dx * s + dy * c };
  }

  function hit(r, px, py) {
    const rh = rotHandle(r);
    if (Math.hypot(px - rh.x, py - rh.y) <= HANDLE) return { mode: "rotate" };
    const cs = corners(r);
    const names = ["nw", "ne", "se", "sw"];
    for (let i = 0; i < 4; i++) {
      if (Math.hypot(px - cs[i].x, py - cs[i].y) <= HANDLE) return { mode: "scale", corner: names[i] };
    }
    const l = toLocal(r, px, py);
    if (Math.abs(l.x) <= r.w / 2 && Math.abs(l.y) <= r.h / 2) return { mode: "move" };
    return null;
  }

  // ------------------------------------------------------------------ drawing
  function draw() {
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    canvas.width = state.dispW * dpr;
    canvas.height = state.dispH * dpr;
    canvas.style.width = state.dispW + "px";
    canvas.style.height = state.dispH + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, state.dispW, state.dispH);

    const img = state.showResult && state.out ? state.out : state.bg;
    if (img) {
      ctx.drawImage(img, 0, 0, state.dispW, state.dispH);
    } else {
      ctx.fillStyle = state.showResult ? "#101418" : "#16181d";
      ctx.fillRect(0, 0, state.dispW, state.dispH);
      ctx.fillStyle = "#5a6272";
      ctx.font = "11px sans-serif";
      ctx.fillText("no background: pick an environment above, or connect an IMAGE", 10, 20);
    }

    const notice = state.notice || state.noticeWarn;
    if (notice) {
      const bad = !!state.notice;
      const txt = notice.length > 78 ? notice.slice(0, 77) + "..." : notice;
      ctx.font = "11px sans-serif";
      const tw = Math.min(ctx.measureText(txt).width, state.dispW - 34);
      ctx.fillStyle = bad ? "rgba(154,28,28,0.92)" : "rgba(122,86,0,0.92)";
      ctx.fillRect(8, 8, tw + 18, 24);
      ctx.fillStyle = "#fff7ed";
      ctx.fillText(txt, 17, 24);
    }

    if (!state.car || state.showResult) {
      nameEl.textContent = state.showResult
        ? "showing last result"
        : state.bg
          ? `${widget(node, "env_file")?.value || "-"}  +  (car layer deleted)`
          : "";
      return;
    }

    const r = cutRect();
    ctx.save();
    ctx.translate(r.cx, r.cy);
    ctx.rotate(r.rot);
    if (num("flip_h", false)) ctx.scale(-1, 1);
    ctx.drawImage(state.car, -r.w / 2, -r.h / 2, r.w, r.h);
    ctx.restore();

    // frame + handles (locked: muted dashed frame, no grab points)
    const cs = corners(r);
    ctx.save();
    ctx.strokeStyle = state.locked ? "rgba(158,168,182,0.8)" : "rgba(70,170,255,0.9)";
    ctx.lineWidth = 1;
    if (state.locked) ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(cs[0].x, cs[0].y);
    for (let i = 1; i < 4; i++) ctx.lineTo(cs[i].x, cs[i].y);
    ctx.closePath();
    ctx.stroke();
    if (state.locked) {
      ctx.setLineDash([]);
      ctx.fillStyle = "rgba(0,0,0,0.6)";
      ctx.fillRect(cs[0].x, cs[0].y - 17, 50, 15);
      ctx.fillStyle = "#ffd25e";
      ctx.font = "10px sans-serif";
      ctx.fillText("locked", cs[0].x + 5, cs[0].y - 6);
    } else {
      const rh = rotHandle(r);
      ctx.beginPath();
      ctx.moveTo((cs[0].x + cs[1].x) / 2, (cs[0].y + cs[1].y) / 2);
      ctx.lineTo(rh.x, rh.y);
      ctx.stroke();
      ctx.fillStyle = "#2f80ed";
      for (const p of cs) ctx.fillRect(p.x - 5, p.y - 5, 10, 10);
      ctx.beginPath();
      ctx.arc(rh.x, rh.y, 6, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();

    // readouts
    ctx.fillStyle = "rgba(0,0,0,0.55)";
    ctx.fillRect(state.dispW - 116, 6, 110, 20);
    ctx.fillStyle = "#e8eef7";
    ctx.font = "11px monospace";
    ctx.fillText(
      `scale ${num("scale", 0.45).toFixed(2)}x  rot ${num("rotation", 0).toFixed(0)}deg`,
      state.dispW - 110,
      20
    );

    const bgName = widget(node, "env_file")?.value || "-";
    const carName = widget(node, "angle_file")?.value || "-";
    nameEl.textContent = `${bgName}  +  ${carName}${state.locked ? "  [locked]" : ""}`;
    nameEl.title = "car environment plate  +  car angle cutout (both from this node's dropdowns)";
  }

  // -------------------------------------------------------------- interaction
  function pointerPos(e) {
    const rect = canvas.getBoundingClientRect();
    return {
      x: ((e.clientX - rect.left) / rect.width) * state.dispW,
      y: ((e.clientY - rect.top) / rect.height) * state.dispH,
    };
  }

  function store() {
    const s = {
      v: 4,
      x: num("x", 0.5),
      y: num("y", 0.62),
      scale: num("scale", 0.45),
      rotation: num("rotation", 0),
      flip_h: num("flip_h", false),
      // lock + deleted layers live here so a reload (and the server) see the same state
      locked: state.locked,
      car_layer: state.layers.car,
      env_layer: state.layers.env,
      // images removed from the two sections, by widget name
      hidden: { env_file: state.hidden.env_file || [], angle_file: state.hidden.angle_file || [] },
    };
    setWidget(node, "ui_state", JSON.stringify(s));
  }

  function syncBar() {
    lockBtn.classList.toggle("on", state.locked);
    lockBtn.textContent = state.locked ? "unlock" : "lock";
    delBtn.classList.toggle("off", !state.layers.car);
    clearBtn.classList.toggle("off", !state.layers.car && !state.layers.env);
  }

  // restore the lock and any deleted layers from the saved node
  function applyUiState() {
    let s = null;
    try {
      s = JSON.parse(widget(node, "ui_state")?.value || "{}");
    } catch (err) {
      s = null;
    }
    if (!s || typeof s !== "object") return;
    state.locked = !!s.locked;
    state.layers.car = s.car_layer !== false;
    state.layers.env = s.env_layer !== false;
    const hid = s.hidden && typeof s.hidden === "object" ? s.hidden : {};
    for (const k of ["env_file", "angle_file"]) {
      state.hidden[k] = Array.isArray(hid[k]) ? hid[k].filter((x) => typeof x === "string").slice(0, 500) : [];
    }
    syncBar();
  }

  canvas.addEventListener("pointerdown", (e) => {
    state.lastPointer = { type: "down", x: e.clientX, y: e.clientY, hasCar: !!state.car, bg: !!state.bg, locked: state.locked };
    if (state.locked) return;
    if (!state.car) return;
    const p = pointerPos(e);
    const r = cutRect();
    const h = hit(r, p.x, p.y);
    if (!h) return;
    state.events = Object.assign(state.events || {}, { down: (state.events?.down || 0) + 1, mode: h.mode });
    // capture first so the drag survives leaving the element; synthetic pointer ids can
    // be rejected, and the drag still works without capture
    try {
      canvas.setPointerCapture(e.pointerId);
      state.events.captured = true;
    } catch (err) {
      state.events.captured = String(err).slice(0, 60);
    }
    state.drag = {
      ...h,
      p0: p,
      start: { x: num("x", 0.5), y: num("y", 0.62), scale: num("scale", 0.45), rotation: num("rotation", 0) },
      d0: Math.max(1, Math.hypot(p.x - r.cx, p.y - r.cy)),
      a0: Math.atan2(p.y - r.cy, p.x - r.cx),
    };
    e.preventDefault();
  });

  canvas.addEventListener("pointermove", (e) => {
    const p = pointerPos(e);
    if (state.locked) {
      state.drag = null;
      canvas.style.cursor = "default";
      return;
    }
    if (!state.drag) {
      const r = cutRect();
      const h = state.car ? hit(r, p.x, p.y) : null;
      canvas.style.cursor = !h ? "default" : h.mode === "move" ? "move" : h.mode === "rotate" ? "grab" : "nwse-resize";
      return;
    }
    const d = state.drag;
    state.events = Object.assign(state.events || {}, { move: (state.events?.move || 0) + 1 });
    const r = cutRect();
    if (d.mode === "move") {
      const nx = d.start.x + (p.x - d.p0.x) / state.dispW;
      const ny = d.start.y + (p.y - d.p0.y) / state.dispH;
      setWidget(node, "x", Math.min(2, Math.max(-1, nx)));
      setWidget(node, "y", Math.min(2, Math.max(-1, ny)));
    } else if (d.mode === "scale") {
      const dist = Math.hypot(p.x - r.cx, p.y - r.cy);
      const next = Math.min(5, Math.max(0.01, d.start.scale * (dist / d.d0)));
      setWidget(node, "scale", next);
    } else if (d.mode === "rotate") {
      const a = Math.atan2(p.y - r.cy, p.x - r.cx);
      let deg = d.start.rotation + ((a - d.a0) * 180) / Math.PI;
      if (e.shiftKey) deg = Math.round(deg / 15) * 15;
      else if (Math.abs(deg % 90) < 3) deg = Math.round(deg / 90) * 90;
      deg = ((deg + 180) % 360 + 360) % 360 - 180;
      setWidget(node, "rotation", deg);
      setWidget(node, "flip_h", deg > 90 || deg < -90 ? true : false);
      if (Math.abs(deg) > 90) setWidget(node, "rotation", deg > 0 ? deg - 180 : deg + 180);
    }
    store();
    draw();
    app.graph.setDirtyCanvas(true, false);
    e.preventDefault();
  });

  const endDrag = (e) => {
    if (!state.drag) return;
    state.drag = null;
    store();
    draw();
    app.graph.setDirtyCanvas(true, true);
  };
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);

  canvas.addEventListener(
    "wheel",
    (e) => {
      if (state.locked) return;
      if (!state.car) return;
      const p = pointerPos(e);
      const r = cutRect();
      if (!hit(r, p.x, p.y)) return;
      const next = Math.min(5, Math.max(0.01, num("scale", 0.45) * Math.exp(-e.deltaY * 0.0012)));
      setWidget(node, "scale", next);
      store();
      draw();
      app.graph.setDirtyCanvas(true, true);
      e.preventDefault();
    },
    { passive: false }
  );

  // drop an image file anywhere on the node -> upload + select
  container.addEventListener("dragover", (e) => e.preventDefault());
  container.addEventListener("drop", async (e) => {
    e.preventDefault();
    const f = e.dataTransfer?.files?.[0];
    if (!f) return;
    // left half of the node = environment plate, right half = car cutout
    const target = e.offsetX < container.clientWidth / 2 ? "env_file" : "angle_file";
    const sub = target === "env_file" ? "car_env" : "car_angles";
    const name = await uploadImage(f, sub);
    choose(target, name);
    rebuildStrips();
  });

  // ------------------------------------------------------------------- sizing
  function stripBlockHeight() {
    let h = 0;
    for (const s of [envStrip, carStrip]) if (s) h += (s.offsetHeight || 0) + 4;
    // before the widget is mounted offsetHeight is 0: fall back to header + one strip row
    return h || 96;
  }

  function layout() {
    const w = Math.max(MIN_W, Math.min(MAX_W, node.size[0] - 26));
    const ar = state.bg ? state.bg.height / state.bg.width : 0.62;
    state.dispW = w;
    state.dispH = Math.max(MIN_H, Math.min(MAX_H, Math.round(w * ar)));
    // the two galleries sit under the canvas: the node is fitted to them in both
    // directions, so a wrapped third row expands the node and removing that row takes
    // the space back instead of leaving it empty. Growing is always safe; shrinking
    // waits for container.isConnected, because stripBlockHeight() falls back to 96
    // until the strips are mounted and acting on that reading would clip them.
    const need = state.dispH + stripBlockHeight() + 54;
    const fit = node.size[1] < need || (container.isConnected && node.size[1] > need + 2);
    if (node.size && fit) node.setSize([node.size[0], need]);
    draw();
  }

  async function refresh() {
    const bgFile = widget(node, "env_file")?.value;
    const carFile = widget(node, "angle_file")?.value;
    const sel = `${bgFile}|${carFile}|${widget(node, "key_studio_bg")?.value}`;
    // one load per selection: a tile click fires this twice (selectFile runs the widget
    // callback, then choose() calls refresh again). The second pass used to see an
    // unchanged selKey, take the stashed-frame shortcut and repaint the canvas with the
    // previous run's plate - a click that visibly changed nothing.
    if (state.pending === sel) return;
    state.pending = sel;
    const token = ++state.ready;
    // node.id is -1 until the graph assigns one, so only ask the server for what the
    // last run produced when we actually have an id.
    const id = node.id > 0 ? node.id : null;
    const stamp = Date.now();
    // the stashed frame is authoritative only when it is the frame already on screen
    const same = state.loadedSel === sel;

    let bgErr = null;
    let bg = same && id ? await loadBitmap(`/view?filename=cpc_${id}_bg.png&type=temp&t=${stamp}`) : null;
    if (!bg && bgFile && bgFile !== "(none)") {
      bg = (await loadBitmap(`/view?filename=${encodeURIComponent(bgFile)}&type=input&subfolder=car_env`)) ||
           (await loadBitmap(`/view?filename=${encodeURIComponent(bgFile)}&type=input`));
      if (!bg) bgErr = `environment "${bgFile}": file missing or corrupt, nothing to show`;
    }
    // plate that arrived on a wired IMAGE input: no file to read back, only the stash
    if (!bg && id) bg = await loadBitmap(`/view?filename=cpc_${id}_bg.png&type=temp&t=${stamp}`);
    let car = same && id ? await loadBitmap(`/view?filename=cpc_${id}_car.png&type=temp&t=${stamp}`) : null;
    let carErr = null;
    if (!car && carFile && carFile !== "(none)") {
      car = (await loadBitmap(`/view?filename=${encodeURIComponent(carFile)}&type=input&subfolder=car_angles`)) ||
            (await loadBitmap(`/view?filename=${encodeURIComponent(carFile)}&type=input`));
      if (!car) carErr = `car image "${carFile}": file missing or corrupt, nothing to show`;
    }
    if (!car && id) car = await loadBitmap(`/view?filename=cpc_${id}_car.png&type=temp&t=${stamp}`);
    let out = id ? await loadBitmap(`/view?filename=cpc_${id}_out.png&type=temp&t=${stamp}`) : null;
    if (token !== state.ready) {
      if (state.pending === sel) state.pending = null;
      return;
    }
    state.pending = null;
    state.loadedSel = sel;
    // a deleted layer stays deleted: otherwise the temp file stashed by an execution
    // would put the cutout back on the canvas after every run
    state.bg = state.layers.env ? bg : null;
    state.car = state.layers.car ? car : null;
    state.out = out;
    state.loaded = { bg: !!state.bg, car: !!state.car, out: !!out };
    if (state.out) resBtn.classList.add("has");
    state.loadErrors = loadErrors.slice(-6);
    state.loadWarnings = loadWarnings.slice(-6);
    // A failed or degraded plate has to be visible. Before this, a tick that failed to
    // load simply redrew the stashed frame, so the selection looked like it did nothing.
    state.notice = bgErr || carErr || null;
    state.noticeWarn = state.notice ? null : (loadWarnings[loadWarnings.length - 1] || null);
    layout();
  }

  const domWidget = node.addDOMWidget("cpc_canvas", "cpc_canvas", container, {
    serialize: false,
    hideOnZoom: false,
    getMinHeight: () => state.dispH + stripBlockHeight() + 54,
  });
  domWidget.onResize = layout;

  // keep the canvas in step with the numeric widgets
  for (const nm of ["x", "y", "scale", "rotation", "flip_h", "key_studio_bg", "env_file", "angle_file"]) {
    const w = widget(node, nm);
    if (!w) continue;
    const prev = w.callback;
    w.callback = function () {
      const r = prev ? prev.apply(this, arguments) : undefined;
      if (nm === "env_file" || nm === "angle_file") {
        syncTicks();
        refresh();
      }
      else draw();
      return r;
    };
  }

  api.addEventListener("executed", (e) => {
    if (String(e?.detail?.node) !== String(node.id)) return;
    refresh();
  });
  api.addEventListener("execution_success", () => refresh());

  // the graph assigns the real id on configure, which is when the server files for this
  // node (written by the last run) become reachable
  const onConfigure = node.onConfigure;
  node.onConfigure = function () {
    const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
    applyUiState();
    refresh();
    return r;
  };

  const onResize = node.onResize;
  node.onResize = function () {
    const r = onResize ? onResize.apply(this, arguments) : undefined;
    layout();
    return r;
  };

  const onDraw = node.onDrawForeground;
  node.onDrawForeground = function () {
    const r = onDraw ? onDraw.apply(this, arguments) : undefined;
    return r;
  };

  // a workable default: MIN_W+26 (286) fits only ~3 tiles per gallery row
  const DEF_W = 520;
  node.setSize([Math.max(node.size[0], DEF_W), Math.max(node.size[1], state.dispH + stripBlockHeight() + 60)]);
  // handles() reports where the widget believes its handles are, in canvas-local px -
  // used by the CDP interaction test instead of duplicating the geometry there.
  node._cpc = {
    state, draw, refresh, layout, widget: domWidget,
    strips, syncTicks, rebuildStrips, applyUiState,
    ui: () => ({ locked: state.locked, layers: { ...state.layers }, ui_state: widget(node, "ui_state")?.value }),
    handles: () => {
      const r = cutRect();
      return { rect: r, corners: corners(r), rot: rotHandle(r) };
    },
  };
  applyUiState();
  refresh();
  return { refresh, layout, state };
}

app.registerExtension({
  name: "CarPlacementCanvas.UI",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "CarPlacementCanvas") return;

    const style = document.createElement("style");
    style.textContent = `
      .cpc-wrap { display:flex; flex-direction:column; gap:4px; padding:2px 4px 4px 4px;
                  box-sizing:border-box; font-family:sans-serif; overflow:hidden; }
      .cpc-bar { display:flex; flex-wrap:wrap; align-items:center; gap:5px; font-size:11px; color:#9aa4b4; }
      .cpc-btn { background:#2a2f39; border:1px solid #454b58; color:#dfe5ee; font-size:10px;
                 border-radius:3px; padding:2px 8px; cursor:pointer; }
      .cpc-btn:hover { background:#39414f; }
      .cpc-btn.on { background:#2f6f43; border-color:#4ec27a; color:#eafff2; }
      .cpc-btn.off { opacity:0.45; }
      .cpc-btn.has::after { content:"*"; color:#ffd25e; margin-left:3px; }
      .cpc-name { flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
                  color:#7fd6ff; text-align:right; }
      .cpc-canvas { display:block; border:1px solid #3a4049; border-radius:3px;
                    background:#0d0f13; touch-action:none; }
      .cpc-sect { display:flex; flex-direction:column; gap:3px; }
      .cpc-sect-hd { display:flex; align-items:center; justify-content:space-between;
                     font-size:10px; letter-spacing:0.06em; color:#8ef0c0; text-transform:uppercase; }
      .cpc-sect-acts { display:flex; gap:5px; }
      .cpc-strip { display:flex; flex-wrap:wrap; align-content:flex-start; gap:5px; overflow:visible;
                   padding:3px; background:#12141a; border:1px solid #2f3540; border-radius:3px; }
      .cpc-thumb { position:relative; flex:0 0 auto; width:76px; height:56px; cursor:pointer;
                   border:2px solid #3a4049; border-radius:3px; background:#0b0d11;
                   display:flex; align-items:center; justify-content:center; overflow:hidden; }
      .cpc-thumb:hover { border-color:#7fb2ff; }
      .cpc-thumb.sel { border-color:#4ec27a; box-shadow:0 0 0 2px rgba(78,194,122,0.35); }
      .cpc-thumb img { width:100%; height:100%; object-fit:cover; display:block; }
      .cpc-thumb-none span { font-size:10px; color:#6b7488; }
      .cpc-thumb-err { background:repeating-linear-gradient(45deg,#1b1d22,#1b1d22 6px,#22252c 6px,#22252c 12px); }
      .cpc-tick { position:absolute; top:2px; right:2px; width:15px; height:15px; border-radius:50%;
                  background:#4ec27a; color:#07240f; font-size:11px; font-weight:700; line-height:15px;
                  text-align:center; display:none; }
      .cpc-thumb.sel .cpc-tick { display:block; }
      .cpc-x { position:absolute; top:1px; left:1px; width:14px; height:14px; padding:0;
               border:1px solid rgba(255,120,120,0.5); border-radius:3px; background:rgba(26,10,10,0.8);
               color:#ff9d9d; font-size:10px; line-height:11px; text-align:center; cursor:pointer;
               opacity:0.55; }
      .cpc-x:hover { opacity:1; background:#5c1c1c; border-color:#ff9d9d; color:#fff0f0; }
      .cpc-thumb-lbl { position:absolute; left:0; right:0; bottom:0; font-size:8px; color:#cfd7e3;
                       background:rgba(0,0,0,0.55); text-align:center; overflow:hidden;
                       white-space:nowrap; text-overflow:ellipsis; padding:1px 2px; }
    `;
    document.head.appendChild(style);

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated ? onCreated.apply(this, arguments) : undefined;
      try {
        // ui_state is plumbing only - keep the string out of the node body.
        const st = this.widgets?.find((w) => w.name === "ui_state");
        if (st) {
          st.computeSize = () => [0, -4];
          st.draw = () => {};
          st.hidden = true;
        }
        buildUI(this);
      } catch (err) {
        console.error("[CarPlacementCanvas] UI failed:", err);
      }
      return r;
    };
  },
});
