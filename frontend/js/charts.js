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
 * Renders a donut chart into `container` from segments [{value, color}].
 * size/stroke control diameter and ring thickness.
 */
function renderDonut(container, segments, { size = 170, stroke = 22 } = {}) {
  container.innerHTML = "";
  const radius = (size - stroke) / 2;
  const circumference = 2 * Math.PI * radius;
  const realTotal = segments.reduce((s, seg) => s + seg.value, 0);
  const total = realTotal || 1;

  const svg = svgEl("svg", { width: size, height: size, viewBox: `0 0 ${size} ${size}` });
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

  let offset = 0;
  segments.forEach((seg) => {
    const fraction = seg.value / total;
    const dash = fraction * circumference;
    const circle = svgEl("circle", {
      cx: size / 2,
      cy: size / 2,
      r: radius,
      fill: "none",
      stroke: seg.color,
      "stroke-width": stroke,
      "stroke-dasharray": `${dash} ${circumference - dash}`,
      "stroke-dashoffset": -offset,
      "stroke-linecap": dash > 0 ? "round" : "butt",
    });
    group.appendChild(circle);
    offset += dash;
  });

  svg.appendChild(group);
  container.appendChild(svg);
}

/**
 * Renders a multi-series line chart (values 0-100) into `container`.
 */
function renderLineChart(container, { labels, series }, { width = 460, height = 220 } = {}) {
  container.innerHTML = "";
  const padding = { top: 16, right: 12, bottom: 28, left: 32 };
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
      "font-size": 10,
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
      svg.appendChild(svgEl("circle", { cx: x, cy: y, r: 2.5, fill: s.color }));
    });
  });

  const labelStep = Math.max(1, Math.ceil(n / 6));
  labels.forEach((lab, i) => {
    if (i % labelStep !== 0 && i !== n - 1) return;
    const x = padding.left + i * xStep;
    const text = svgEl("text", {
      x,
      y: height - 8,
      "text-anchor": "middle",
      "font-size": 10,
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
function renderBarChart(container, { labels, series }, { width = 460, height = 220 } = {}) {
  container.innerHTML = "";
  const padding = { top: 16, right: 12, bottom: 44, left: 32 };
  const chartW = width - padding.left - padding.right;
  const chartH = height - padding.top - padding.bottom;

  const svg = svgEl("svg", { width: "100%", height, viewBox: `0 0 ${width} ${height}` });

  [0, 25, 50, 75, 100].forEach((v) => {
    const y = padding.top + chartH - (v / 100) * chartH;
    svg.appendChild(
      svgEl("line", { x1: padding.left, x2: width - padding.right, y1: y, y2: y, stroke: cssVar("--border", "#1f3a56"), "stroke-width": 1 })
    );
    const text = svgEl("text", { x: padding.left - 8, y: y + 4, "text-anchor": "end", "font-size": 10, fill: cssVar("--text-muted", "#93a8bf") });
    text.textContent = v;
    svg.appendChild(text);
  });

  const n = labels.length;
  const groupWidth = chartW / n;
  const barGap = 3;
  const barWidth = Math.max(4, (groupWidth - barGap * (series.length + 1)) / series.length);
  const maxVal = Math.max(1, ...series.flatMap((s) => s.values));

  labels.forEach((lab, i) => {
    const groupX = padding.left + i * groupWidth;
    series.forEach((s, si) => {
      const val = s.values[i] || 0;
      const barH = (val / maxVal) * chartH;
      const x = groupX + barGap + si * (barWidth + barGap);
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
    });
    const text = svgEl("text", {
      x: groupX + groupWidth / 2,
      y: height - 26,
      "text-anchor": "middle",
      "font-size": 9,
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
    const fontSize = 13 + scale * 26;
    span.textContent = w.text;
    span.style.fontSize = `${fontSize}px`;
    span.style.color = colors[i % colors.length];
    container.appendChild(span);
  });
}
