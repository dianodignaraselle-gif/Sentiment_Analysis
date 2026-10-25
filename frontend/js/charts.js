const SVG_NS = "http://www.w3.org/2000/svg";

function svgEl(tag, attrs = {}) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}

// Charts draw via hardcoded hex (SVG attrs don't follow CSS variables), so we
// read the current theme's colors from :root at render time to stay in sync
// with the light/dark toggle.
function cssVar(name, fallback) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name);
  return v && v.trim() ? v.trim() : fallback;
}

/**
 * Renders a donut chart into `container` from segments
 * [{value, color, label?, labelColor?}] (labelColor defaults to white).
 * size/stroke control diameter and ring thickness. When a segment has a
 * `label`, it's drawn on the ring at the segment's midpoint (or just outside
 * the ring when the segment is too thin to hold the text).
 */
function renderDonut(container, segments, { size = 170, stroke = 22 } = {}) {
  container.innerHTML = "";
  const hasLabels = segments.some((seg) => seg.label);
  // Leave room around the ring for labels of thin segments drawn outside it.
  const margin = hasLabels ? 30 : 0;
  const radius = (size - stroke) / 2 - margin;
  const circumference = 2 * Math.PI * radius;
  const realTotal = segments.reduce((s, seg) => s + seg.value, 0);
  const total = realTotal || 1;

  const svg = svgEl("svg", { width: size, height: size, viewBox: `0 0 ${size} ${size}`, overflow: "visible" });
  const group = svgEl("g", { transform: `rotate(-90 ${size / 2} ${size / 2})` });

  if (realTotal === 0) {
    group.appendChild(
      svgEl("circle", {
        cx: size / 2,
        cy: size / 2,
        r: radius,
        fill: "none",
        stroke: cssVar("--border", "#1f3a56"),
        "stroke-width": stroke,
      })
    );
    svg.appendChild(group);
    container.appendChild(svg);
    return;
  }

  const labels = svgEl("g", { "font-weight": 700, "text-anchor": "middle", "dominant-baseline": "central" });
  let offset = 0;
  segments.forEach((seg) => {
    const fraction = seg.value / total;
    const dash = fraction * circumference;
    if (seg.label && fraction > 0) {
      // Angle measured clockwise from 12 o'clock, matching the rotated ring.
      const angle = ((offset + dash / 2) / circumference) * 2 * Math.PI;
      const fontSize = Math.max(12, Math.round(stroke * 0.22));
      // Rough bold-text width; only draw on the ring if the arc can hold it.
      const inside = dash >= seg.label.length * fontSize * 0.62;
      const r = inside ? radius : radius + stroke / 2 + margin / 2 + 4;
      const text = svgEl("text", {
        x: size / 2 + r * Math.sin(angle),
        y: size / 2 - r * Math.cos(angle),
        "font-size": inside ? fontSize : 14,
        fill: inside ? seg.labelColor || "#fff" : cssVar("--text", "#111827"),
      });
      text.textContent = seg.label;
      labels.appendChild(text);
    }
    const circle = svgEl("circle", {
      cx: size / 2,
      cy: size / 2,
      r: radius,
      fill: "none",
      stroke: seg.color,
      "stroke-width": stroke,
      "stroke-dasharray": `${dash} ${circumference - dash}`,
      "stroke-dashoffset": -offset,
      // Round caps overlap neighbours, which would misrepresent labelled shares.
      "stroke-linecap": dash > 0 && !hasLabels ? "round" : "butt",
    });
    group.appendChild(circle);
    offset += dash;
  });

  svg.appendChild(group);
  svg.appendChild(labels);
  container.appendChild(svg);
}

/**
 * Renders a multi-series line chart (values 0-100) into `container`.
 * With `showValues`, each point gets a "%" label in its series' `labelColor`
 * (or `color`); labels sharing an x position are nudged apart vertically.
 */
function renderLineChart(container, { labels, series }, { width, height = 220, showValues = false } = {}) {
  container.innerHTML = "";
  // Draw at the container's real width so the chart fills its card.
  width = width || container.clientWidth || 460;
  const padding = showValues
    ? { top: 26, right: 24, bottom: 30, left: 38 }
    : { top: 16, right: 12, bottom: 30, left: 38 };
  const chartW = width - padding.left - padding.right;
  const chartH = height - padding.top - padding.bottom;

  const svg = svgEl("svg", { width: "100%", height, viewBox: `0 0 ${width} ${height}` });

  // gridlines + y labels
  [0, 25, 50, 75, 100].forEach((v) => {
    const y = padding.top + chartH - (v / 100) * chartH;
    svg.appendChild(
      svgEl("line", {
        x1: padding.left,
        x2: width - padding.right,
        y1: y,
        y2: y,
        stroke: cssVar("--border", "#1f3a56"),
        "stroke-width": 1,
      })
    );
    const text = svgEl("text", {
      x: padding.left - 8,
      y: y + 4,
      "text-anchor": "end",
      "font-size": 13,
      fill: cssVar("--text-muted", "#93a8bf"),
    });
    text.textContent = v;
    svg.appendChild(text);
  });

  const n = labels.length;
  const xStep = n > 1 ? chartW / (n - 1) : 0;

  series.forEach((s) => {
    const points = s.values.map((v, i) => {
      const x = padding.left + i * xStep;
      const y = padding.top + chartH - (v / 100) * chartH;
      return [x, y];
    });
    const pointsAttr = points.map((p) => p.join(",")).join(" ");
    svg.appendChild(
      svgEl("polyline", {
        points: pointsAttr,
        fill: "none",
        stroke: s.color,
        "stroke-width": 2.5,
        "stroke-linejoin": "round",
        "stroke-linecap": "round",
      })
    );
    points.forEach(([x, y]) => {
      svg.appendChild(svgEl("circle", { cx: x, cy: y, r: showValues ? 3.5 : 2.5, fill: s.color }));
    });
  });

  if (showValues) {
    const FONT = 12;
    const GAP = FONT + 2;
    const minY = FONT;
    const maxY = padding.top + chartH + 6;
    for (let i = 0; i < n; i++) {
      const x = padding.left + i * xStep;
      // Edge columns hang their labels inward so they clear the y-axis
      // numbers and the card edge.
      const anchor = n > 1 && i === 0 ? "start" : n > 1 && i === n - 1 ? "end" : "middle";
      const dx = anchor === "start" ? 6 : anchor === "end" ? -6 : 0;
      const items = series
        .map((s) => {
          const v = s.values[i];
          return { s, v, py: padding.top + chartH - (v / 100) * chartH };
        })
        .sort((a, b) => a.py - b.py);
      const dots = items.map((it) => it.py);
      const placed = [];
      // Try just above the point, then just below; skip spots that overlap
      // a placed label or sit on another series' dot at this x.
      items.forEach((it) => {
        const clear = (y) =>
          y >= minY &&
          y <= maxY &&
          placed.every((p) => Math.abs(p - y) >= GAP) &&
          dots.every((d) => d === it.py || Math.abs(d - y) >= FONT / 2 + 4);
        const candidates = [it.py - 11, it.py + 12, it.py - 11 - GAP, it.py + 12 + GAP];
        it.y = candidates.find(clear);
        if (it.y === undefined) {
          it.y = Math.min(maxY, Math.max(minY, placed.length ? Math.max(...placed) + GAP : it.py - 11));
        }
        placed.push(it.y);
      });
      items.forEach(({ s, v, y }) => {
        const text = svgEl("text", {
          x: x + dx,
          y,
          "text-anchor": anchor,
          "dominant-baseline": "central",
          "font-size": FONT,
          "font-weight": 700,
          fill: s.labelColor || s.color,
          // Halo in the card colour keeps labels legible where lines cross.
          stroke: cssVar("--card-bg", "#ffffff"),
          "stroke-width": 3,
          "paint-order": "stroke",
        });
        text.textContent = `${Math.round(v * 10) / 10}%`;
        svg.appendChild(text);
      });
    }
  }

  const labelStep = Math.max(1, Math.ceil(n / 6));
  labels.forEach((lab, i) => {
    if (i % labelStep !== 0 && i !== n - 1) return;
    const x = padding.left + i * xStep;
    const text = svgEl("text", {
      x,
      y: height - 8,
      "text-anchor": "middle",
      "font-size": 13,
      fill: cssVar("--text-muted", "#93a8bf"),
    });
    text.textContent = lab;
    svg.appendChild(text);
  });

  container.appendChild(svg);
}

/**
 * Renders a grouped bar chart into `container`.
 */
function renderBarChart(container, { labels, series }, { width, height = 220, showShares = false } = {}) {
  container.innerHTML = "";
  // Draw at the container's real width so the chart fills its card.
  width = width || container.clientWidth || 460;
  const padding = { top: showShares ? 24 : 16, right: 12, bottom: 46, left: 38 };
  const chartW = width - padding.left - padding.right;
  const chartH = height - padding.top - padding.bottom;

  const svg = svgEl("svg", { width: "100%", height, viewBox: `0 0 ${width} ${height}` });

  // Y axis shows real counts: round the largest value up to 4 "nice"
  // integer steps (e.g. 361 -> 0/100/200/300/400).
  const maxRaw = Math.max(1, ...series.flatMap((s) => s.values));
  const rawStep = maxRaw / 4;
  const mag = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const step = Math.max(1, Math.ceil([1, 2, 2.5, 5, 10].map((m) => m * mag).find((c) => c >= rawStep)));
  const maxVal = step * 4;

  [0, 1, 2, 3, 4].forEach((k) => {
    const v = k * step;
    const y = padding.top + chartH - (v / maxVal) * chartH;
    svg.appendChild(
      svgEl("line", { x1: padding.left, x2: width - padding.right, y1: y, y2: y, stroke: cssVar("--border", "#1f3a56"), "stroke-width": 1 })
    );
    const text = svgEl("text", { x: padding.left - 8, y: y + 4, "text-anchor": "end", "font-size": 13, fill: cssVar("--text-muted", "#93a8bf") });
    text.textContent = v;
    svg.appendChild(text);
  });

  const n = labels.length;
  const groupWidth = chartW / n;
  const barGap = 3;
  // Capped so a few products on a wide card don't get oversized bars; the
  // group's bars are centered in their slot instead.
  const barWidth = Math.min(64, Math.max(4, (groupWidth - barGap * (series.length + 1)) / series.length));
  const groupInset = (groupWidth - (barWidth * series.length + barGap * (series.length + 1))) / 2;
  // Shrink share labels on crowded charts so they stay within their bar.
  const shareFont = Math.min(11, Math.max(7, barWidth / 3.4));

  labels.forEach((lab, i) => {
    const groupX = padding.left + i * groupWidth;
    const groupTotal = series.reduce((sum, s) => sum + (s.values[i] || 0), 0);
    series.forEach((s, si) => {
      const val = s.values[i] || 0;
      const barH = (val / maxVal) * chartH;
      const x = groupX + groupInset + barGap + si * (barWidth + barGap);
      const y = padding.top + chartH - barH;
      svg.appendChild(
        svgEl("rect", {
          x,
          y,
          width: barWidth,
          height: Math.max(0, barH),
          rx: 2,
          fill: s.color,
        })
      );
      // Share of this product's comments; skipped at 0% (no bar) and at
      // 100% (the bar is the whole product, so the label adds nothing).
      if (showShares && groupTotal > 0 && val > 0 && val < groupTotal) {
        const share = svgEl("text", {
          x: x + barWidth / 2,
          y: y - 5,
          "text-anchor": "middle",
          "font-size": shareFont,
          "font-weight": 700,
          fill: s.labelColor || s.color,
        });
        share.textContent = `${Math.round((val / groupTotal) * 1000) / 10}%`;
        svg.appendChild(share);
      }
    });
    const text = svgEl("text", {
      x: groupX + groupWidth / 2,
      y: height - 26,
      "text-anchor": "middle",
      "font-size": 12.5,
      fill: cssVar("--text-muted", "#93a8bf"),
    });
    const words = lab.split(" ");
    text.textContent = words.length > 1 ? words.slice(0, 2).join(" ") : lab;
    svg.appendChild(text);
  });

  container.appendChild(svg);
}

/**
 * Renders a simple word cloud from [{text, count}] into `container`.
 */
function renderWordCloud(container, words) {
  container.innerHTML = "";
  if (!words.length) {
    container.innerHTML = '<div class="empty-state">Not enough data yet.</div>';
    return;
  }
  const colors = ["#2dd4bf", "#5b8def", "#22c55e", cssVar("--text", "#e8eef5"), "#eab308", "#3b82f6"];
  const max = Math.max(...words.map((w) => w.count));
  const min = Math.min(...words.map((w) => w.count));
  words.forEach((w, i) => {
    const span = document.createElement("span");
    const scale = max === min ? 0.5 : (w.count - min) / (max - min);
    const fontSize = 15 + scale * 26;
    span.textContent = w.text;
    span.style.fontSize = `${fontSize}px`;
    span.style.color = colors[i % colors.length];
    container.appendChild(span);
  });
}
