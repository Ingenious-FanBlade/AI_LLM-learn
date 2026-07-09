import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "file:///C:/Users/114514/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const ROOT = "C:/Users/114514/Desktop/人工智能大模型";
const WORK = path.join(ROOT, "outputs/revised-llm-watermarking-ppt/tmp");
const PREVIEW = path.join(WORK, "preview");
const LAYOUT = path.join(WORK, "layout");
const OUT = path.join(ROOT, "outputs/llm-watermarking-comparison-revised.pptx");

const C = {
  bg: "#F7F4EE",
  paper: "#FFFFFF",
  paper2: "#FBFAF7",
  ink: "#17212B",
  muted: "#5A6470",
  faint: "#E8E1D7",
  hair: "#D8D2C8",
  exp: "#2457A6",
  expLight: "#DCE8FA",
  kgw: "#26705A",
  kgwLight: "#DCEBE4",
  combo: "#B7892F",
  comboLight: "#F1E5C8",
  warn: "#C7634B",
  warnLight: "#F5DED7",
  dark: "#17212B",
  dark2: "#24313D",
};

const data = {
  detection: [
    { label: "EXP", value: 1.0, color: C.exp },
    { label: "KGW", value: 0.8, color: C.kgw },
    { label: "Combined", value: 1.0, color: C.combo },
  ],
  ppl: [
    { label: "EXP", value: 11.17, delta: "-8.46", color: C.exp },
    { label: "KGW", value: 25.05, delta: "+5.42", color: C.kgw },
    { label: "Combined", value: 17.26, delta: "-2.37", color: C.combo },
    { label: "Natural", value: 19.63, delta: "baseline", color: C.muted },
  ],
  baselineAttacks: [
    { condition: "No attack", exp: 1.0, kgw: 0.8 },
    { condition: "Word-S", exp: 0.05, kgw: 0.05 },
    { condition: "Word-D", exp: 0.0, kgw: 0.1 },
  ],
  hybridRobustness: [
    { condition: "No attack", kgw: 0.94, exp: 0.98, combo: 1.0 },
    { condition: "Word-S 20%", kgw: 0.20, exp: 0.86, combo: 0.90 },
    { condition: "Word-D 30%", kgw: 0.12, exp: 0.36, combo: 0.68 },
  ],
  kgwDelta: [
    { label: "0.1", value: 0.0 },
    { label: "0.3", value: 0.0 },
    { label: "0.5", value: 0.2 },
    { label: "1.0", value: 0.6 },
    { label: "2.0", value: 1.0 },
  ],
  expTau: [
    { label: "1e-6", value: 0.90 },
    { label: "1e-4", value: 0.95 },
    { label: "1e-2", value: 0.95 },
  ],
};

function line(fill = "none", width = 0) {
  return { style: "solid", fill, width };
}

function shape(slide, geometry, left, top, width, height, fill = "none", lineFill = "none", lineWidth = 0, name) {
  return slide.shapes.add({
    geometry,
    name,
    position: { left, top, width, height },
    fill,
    line: line(lineFill, lineWidth),
  });
}

function rect(slide, left, top, width, height, fill = C.paper, lineFill = "none", lineWidth = 0, name) {
  return shape(slide, "rect", left, top, width, height, fill, lineFill, lineWidth, name);
}

function text(slide, value, left, top, width, height, opts = {}) {
  const t = shape(slide, "textbox", left, top, width, height, opts.fill ?? "none", opts.lineFill ?? "none", opts.lineWidth ?? 0, opts.name);
  t.text = value;
  t.text.style = {
    fontSize: opts.size ?? 20,
    color: opts.color ?? C.ink,
    bold: Boolean(opts.bold),
    typeface: opts.face ?? (opts.title ? "Aptos Display" : "Aptos"),
    alignment: opts.align ?? "left",
    verticalAlignment: opts.valign ?? "top",
    insets: opts.insets ?? { left: 0, right: 0, top: 0, bottom: 0 },
    wrap: "square",
  };
  return t;
}

function rule(slide, left, top, width, height = 1, color = C.hair) {
  return rect(slide, left, top, width, height, color, color, 0);
}

function bg(slide, fill = C.bg) {
  slide.background.fill = fill;
}

function header(slide, kicker, title, n, dark = false) {
  const base = dark ? C.bg : C.ink;
  const muted = dark ? "#BFC7CE" : C.muted;
  const marker = dark ? C.combo : C.exp;
  rect(slide, 64, 50, 5, 16, marker, marker, 0, `kicker-${n}-marker`);
  text(slide, kicker.toUpperCase(), 78, 47, 310, 22, {
    size: 11,
    color: muted,
    bold: true,
    valign: "middle",
    name: `kicker-${n}-label`,
  });
  text(slide, title, 64, 82, 900, 96, {
    title: true,
    size: 36,
    color: base,
    bold: true,
  });
  rule(slide, 64, 180, 180, 2, dark ? "#3B4651" : C.hair);
}

function footer(slide, source, n, dark = false) {
  const col = dark ? "#AEB7BE" : C.muted;
  rule(slide, 64, 674, 1052, 1, dark ? "#36414C" : C.hair);
  text(slide, source, 64, 684, 900, 20, { size: 9.5, color: col });
  text(slide, String(n).padStart(2, "0"), 1160, 682, 56, 24, {
    size: 13,
    color: col,
    align: "right",
    bold: true,
  });
}

function tag(slide, label, left, top, width, color, fill) {
  rect(slide, left, top, width, 28, fill, color, 1);
  text(slide, label, left + 8, top + 5, width - 16, 18, {
    size: 12,
    bold: true,
    color,
    align: "center",
  });
}

function note(slide, left, top, width, height, title, body, color = C.exp) {
  rect(slide, left, top, 4, height, color, color, 0);
  text(slide, title, left + 16, top + 8, width - 24, 22, {
    size: 14,
    bold: true,
    color,
  });
  text(slide, body, left + 16, top + 34, width - 24, height - 44, {
    size: 15,
    color: C.muted,
  });
}

function metric(slide, left, top, width, height, label, value, body, color) {
  rect(slide, left, top, width, height, C.paper, color, 1);
  text(slide, label.toUpperCase(), left + 16, top + 14, width - 32, 18, {
    size: 10.5,
    color: C.muted,
    bold: true,
  });
  text(slide, value, left + 16, top + 38, width - 32, 36, {
    title: true,
    size: 30,
    color,
    bold: true,
  });
  text(slide, body, left + 16, top + 82, width - 32, height - 96, {
    size: 13,
    color: C.muted,
  });
}

function vBars(slide, items, left, top, width, height, opts = {}) {
  const max = opts.max ?? 1;
  const gap = opts.gap ?? 28;
  const barW = (width - gap * (items.length - 1)) / items.length;
  rule(slide, left, top + height, width, 1, C.hair);
  for (let i = 0; i < items.length; i += 1) {
    const item = items[i];
    const x = left + i * (barW + gap);
    const h = Math.max(2, item.value / max * height);
    rect(slide, x, top + height - h, barW, h, item.color);
    text(slide, item.valueLabel ?? item.value.toFixed(2), x, top + height - h - 25, barW, 18, {
      size: 13,
      bold: true,
      color: item.color,
      align: "center",
    });
    text(slide, item.label, x - 6, top + height + 10, barW + 12, 34, {
      size: 12,
      color: C.muted,
      align: "center",
    });
  }
}

function hBars(slide, items, left, top, width, rowH, max) {
  for (let i = 0; i < items.length; i += 1) {
    const item = items[i];
    const y = top + i * rowH;
    text(slide, item.label, left, y + 3, 100, 24, { size: 14, color: C.ink, bold: item.label === "Combined" });
    rule(slide, left + 130, y + 15, width, 1, C.hair);
    const w = item.value / max * width;
    rect(slide, left + 130, y + 4, w, 22, item.color);
    text(slide, item.value.toFixed(2), left + 138 + w, y + 5, 58, 18, { size: 12.5, bold: true, color: item.color });
    text(slide, item.delta, left + 210 + w, y + 5, 80, 18, { size: 11, color: C.muted });
  }
}

function groupedRobustness(slide, rows, left, top, width, height, includeCombined) {
  const groupW = width / rows.length;
  const barW = includeCombined ? 36 : 48;
  const colors = includeCombined ? [C.kgw, C.exp, C.combo] : [C.exp, C.kgw];
  const keys = includeCombined ? ["kgw", "exp", "combo"] : ["exp", "kgw"];
  const labels = includeCombined ? ["KGW", "EXP", "Combined"] : ["EXP", "KGW"];
  rule(slide, left, top + height, width, 1, C.hair);
  for (let i = 0; i < rows.length; i += 1) {
    const gx = left + i * groupW;
    const totalW = keys.length * barW + (keys.length - 1) * 10;
    const start = gx + (groupW - totalW) / 2;
    for (let j = 0; j < keys.length; j += 1) {
      const val = rows[i][keys[j]];
      const h = Math.max(2, val * height);
      const x = start + j * (barW + 10);
      rect(slide, x, top + height - h, barW, h, colors[j]);
      text(slide, val.toFixed(2), x - 4, top + height - h - 22, barW + 8, 16, {
        size: 11,
        color: colors[j],
        bold: true,
        align: "center",
      });
    }
    text(slide, rows[i].condition, gx + 8, top + height + 12, groupW - 16, 32, {
      size: 12.5,
      color: C.ink,
      bold: i === 0,
      align: "center",
    });
  }
  const lx = left + width - (includeCombined ? 255 : 170);
  for (let j = 0; j < keys.length; j += 1) {
    rect(slide, lx + j * 86, top - 44, 12, 12, colors[j]);
    text(slide, labels[j], lx + 18 + j * 86, top - 48, 76, 20, { size: 11.5, color: C.muted });
  }
}

function tableCell(slide, value, left, top, width, height, opts = {}) {
  rect(slide, left, top, width, height, opts.fill ?? C.paper, opts.lineFill ?? C.hair, 1);
  text(slide, value, left + 10, top + 8, width - 20, height - 12, {
    size: opts.size ?? 12.5,
    color: opts.color ?? C.ink,
    bold: opts.bold ?? false,
    align: opts.align ?? "left",
    valign: "middle",
  });
}

function slide01(p) {
  const slide = p.slides.add();
  bg(slide);
  text(slide, "LLM Watermarking for\nIntellectual Property Protection", 68, 82, 760, 138, {
    title: true, size: 56, bold: true, color: C.ink,
  });
  text(slide, "Revised comparative presentation: KGW, EXP, and Combined KGW-EXP", 70, 232, 760, 30, { size: 20, color: C.muted });
  text(slide, "Guo Yutai, Zhong Yimin, Wang Yanzhi, Ouyang Zhaohong,\nQin Xiaoyu, Yao Hanbin | June 22, 2026", 70, 285, 760, 44, { size: 15, color: C.muted });

  rect(slide, 790, 86, 1, 500, C.hair);
  text(slide, "Revision focus", 830, 88, 220, 24, { size: 13, bold: true, color: C.muted });
  text(slide, "The paper now adds a fusion method that embeds both KGW and EXP signals in one generation pipeline.", 830, 122, 330, 72, { size: 19, color: C.ink });
  tag(slide, "KGW", 830, 236, 100, C.kgw, C.kgwLight);
  text(slide, "Green-list\nlogit bias", 952, 229, 210, 48, { size: 18, bold: true });
  rule(slide, 830, 304, 340, 1);
  tag(slide, "EXP", 830, 342, 100, C.exp, C.expLight);
  text(slide, "Exponential\nresampling", 952, 335, 210, 48, { size: 18, bold: true });
  rule(slide, 830, 410, 340, 1);
  tag(slide, "Combined", 830, 448, 112, C.combo, C.comboLight);
  text(slide, "Two-stage embedding\n+ Fisher detection", 960, 439, 220, 56, { size: 18, bold: true });

  rect(slide, 70, 390, 650, 150, C.paper2, C.hair, 1);
  text(slide, "Presentation route", 94, 410, 180, 20, { size: 13, bold: true, color: C.muted });
  text(slide, "Evaluation metrics first, then experimental result analysis, then summary.", 94, 442, 570, 62, { size: 22, bold: true });
  text(slide, "The cover intentionally avoids outcome metrics.", 94, 512, 380, 20, { size: 14, color: C.muted });
  footer(slide, "Source: revised paper title page and introduction.", 1);
}

function slide02(p) {
  const slide = p.slides.add();
  bg(slide);
  header(slide, "Revision frame", "The revision turns a two-method comparison into a fusion-method study.", 2);
  const items = [
    ["Original baseline", "KGW and EXP are reproduced under unified conditions."],
    ["Observed bottleneck", "Single token-level watermarks degrade under synonym substitution and deletion."],
    ["New contribution", "Combined KGW-EXP overlays KGW logit bias with EXP resampling."],
    ["Joint decision", "Fisher's method fuses KGW and EXP p-values for the final detector."],
  ];
  for (let i = 0; i < items.length; i += 1) {
    const y = 210 + i * 90;
    const col = i < 2 ? (i === 0 ? C.exp : C.warn) : C.combo;
    rect(slide, 78, y, 62, 62, i === 1 ? C.warnLight : (i === 0 ? C.expLight : C.comboLight), col, 1);
    text(slide, String(i + 1), 78, y + 13, 62, 32, { title: true, size: 28, bold: true, color: col, align: "center" });
    text(slide, items[i][0], 164, y, 300, 24, { size: 18, bold: true });
    text(slide, items[i][1], 164, y + 30, 500, 40, { size: 15, color: C.muted });
    if (i < items.length - 1) rule(slide, 108, y + 66, 1, 24);
  }
  rect(slide, 735, 220, 370, 284, C.paper, C.hair, 1);
  rect(slide, 735, 220, 8, 284, C.combo);
  text(slide, "What changed in the story", 765, 248, 280, 30, { size: 22, bold: true });
  text(slide, "The central claim is no longer only that EXP beats KGW under benign settings. The updated claim is that complementary token-level signals can materially improve robustness.", 765, 296, 275, 108, { size: 16, color: C.ink });
  rule(slide, 765, 426, 260, 1);
  text(slide, "This shifts the deck's result emphasis from pairwise comparison to the Combined method's robustness-quality balance.", 765, 450, 270, 50, { size: 15, color: C.muted });
  footer(slide, "Source: revised paper, Sections 1.3-1.4 and 2.1.", 2);
}

function slide03(p) {
  const slide = p.slides.add();
  bg(slide);
  header(slide, "Evaluation metrics", "One metric lens is used for KGW, EXP, and Combined.", 3);
  const cols = [
    ["Effectiveness", C.exp, C.expLight, "Can the detector identify watermarked text while controlling false positives?", ["TPR under no attack", "FPR on natural text", "Score separation: z-score or p-value"]],
    ["Imperceptibility", C.combo, C.comboLight, "Does embedding preserve fluency and avoid suspicious statistical drift?", ["PPL by GPT-2", "Delta PPL versus natural baseline", "Quality stability under parameters"]],
    ["Robustness", C.warn, C.warnLight, "Does the watermark survive meaning-preserving word-level edits?", ["TPR after Word-S", "TPR after Word-D", "Robustness coefficient"]],
  ];
  for (let i = 0; i < cols.length; i += 1) {
    const x = 72 + i * 384;
    rect(slide, x, 210, 330, 324, C.paper, C.hair, 1);
    rect(slide, x, 210, 330, 8, cols[i][1]);
    rect(slide, x + 24, 240, 52, 52, cols[i][2], cols[i][1], 1);
    text(slide, String(i + 1), x + 24, 251, 52, 30, { title: true, size: 26, bold: true, color: cols[i][1], align: "center" });
    text(slide, cols[i][0], x + 92, 241, 210, 26, { size: 21, bold: true });
    text(slide, cols[i][3], x + 92, 272, 210, 54, { size: 14.5, color: C.muted });
    rule(slide, x + 24, 342, 282, 1);
    for (let j = 0; j < cols[i][4].length; j += 1) {
      const y = 364 + j * 48;
      rect(slide, x + 28, y + 6, 7, 7, cols[i][1]);
      text(slide, cols[i][4][j], x + 48, y, 238, 34, { size: 13.5 });
    }
  }
  text(slide, "The revised paper keeps the original metric order, but expands every result slide to include the fusion method.", 74, 574, 940, 28, { size: 18, bold: true });
  footer(slide, "Source: revised paper, Section 2.3.", 3);
}

function slide04(p) {
  const slide = p.slides.add();
  bg(slide);
  header(slide, "Combined method", "KGW-EXP embeds two token-level signals before making one Fisher decision.", 4);
  const steps = [
    ["KGW bias", C.kgw, C.kgwLight, "Split vocabulary into green/red lists and add delta = 2.0 to green-list logits."],
    ["Temperature + softmax", C.combo, C.comboLight, "Scale biased logits and compute the adjusted candidate distribution."],
    ["EXP resampling", C.exp, C.expLight, "Sample the token maximizing the prefix-keyed exponential resampling score."],
    ["Fisher detector", C.combo, C.comboLight, "Convert KGW z-score and EXP statistic to p-values, then combine them."],
  ];
  for (let i = 0; i < steps.length; i += 1) {
    const x = 80 + i * 284;
    rect(slide, x, 238, 218, 132, C.paper, C.hair, 1);
    rect(slide, x, 238, 218, 8, steps[i][1]);
    text(slide, steps[i][0], x + 18, 262, 178, 24, { size: 18, bold: true, color: C.ink });
    text(slide, steps[i][3], x + 18, 300, 178, 54, { size: 14.2, color: C.muted });
    if (i < steps.length - 1) {
      rule(slide, x + 226, 302, 42, 2, C.hair);
      rect(slide, x + 266, 298, 8, 10, C.hair);
    }
  }
  rect(slide, 132, 452, 430, 110, C.paper2, C.hair, 1);
  text(slide, "Detection statistic", 156, 474, 170, 20, { size: 14, bold: true, color: C.muted });
  text(slide, "chi-square = -2 * (ln p_KGW + ln p_EXP)", 156, 506, 360, 26, { size: 22, bold: true, color: C.ink });
  rect(slide, 658, 452, 390, 110, C.paper2, C.hair, 1);
  text(slide, "Decision rule", 682, 474, 170, 20, { size: 14, bold: true, color: C.muted });
  text(slide, "score = -log10(p_combined)\nwatermarked if score > 2.2", 682, 506, 326, 46, { size: 18, bold: true, color: C.ink });
  footer(slide, "Source: revised paper, Sections 3.4.1-3.4.4.", 4);
}

function slide05(p) {
  const slide = p.slides.add();
  bg(slide);
  header(slide, "Experimental protocol", "The revision evaluates three methods under shared metrics but two experiment scales.", 5);
  const cards = [
    ["Methods", "KGW, EXP, and Combined KGW-EXP"],
    ["Model", "Qwen2.5-1.5B-Instruct via MarkLLM"],
    ["Baseline set", "20 news prompts, 200-token generations"],
    ["Robustness set", "50 C4 prompts, 100-token generations"],
    ["Attacks", "Word-S 20% substitution; Word-D 30% deletion"],
  ];
  for (let i = 0; i < cards.length; i += 1) {
    const x = 70 + i * 222;
    rect(slide, x, 232, 176, 120, C.paper, C.hair, 1);
    rect(slide, x, 232, 176, 6, i === 0 ? C.combo : (i === 4 ? C.warn : C.exp));
    text(slide, cards[i][0], x + 14, 258, 146, 24, { size: 16, bold: true, align: "center" });
    text(slide, cards[i][1], x + 14, 296, 146, 40, { size: 12.5, color: C.muted, align: "center" });
  }
  rect(slide, 120, 430, 430, 116, C.paper2, C.hair, 1);
  text(slide, "Combined configuration", 146, 452, 240, 24, { size: 18, bold: true });
  text(slide, "KGW gamma 0.5, delta 2.0; EXP prefix k=4, threshold 1e-4; Fisher theta 2.2; shared hash key 15485863.", 146, 488, 340, 42, { size: 14.5, color: C.muted });
  rect(slide, 646, 430, 410, 116, C.paper2, C.hair, 1);
  text(slide, "Interpretation guardrail", 672, 452, 240, 24, { size: 18, bold: true });
  text(slide, "The n=50 C4 robustness table is the main evidence for Combined; the n=20 news table explains why fusion was needed.", 672, 488, 320, 42, { size: 14.5, color: C.muted });
  footer(slide, "Source: revised paper, Sections 4.1 and Table 2.", 5);
}

function slide06(p) {
  const slide = p.slides.add();
  bg(slide);
  header(slide, "Experimental result analysis", "EXP and Combined reach perfect benign detection; KGW remains weaker.", 6);
  text(slide, "TPR under no attack", 110, 220, 300, 22, { size: 14, bold: true, color: C.muted });
  vBars(slide, data.detection.map(d => ({ label: d.label, value: d.value, color: d.color, valueLabel: d.value.toFixed(3) })), 110, 282, 430, 210, { max: 1, gap: 42 });
  metric(slide, 612, 242, 176, 124, "All FPR", "0.000", "False positives remain controlled across reported configurations.", C.exp);
  note(slide, 850, 242, 300, 92, "EXP separation", "Watermarked p-values cluster near zero; natural p-values remain approximately uniform.", C.exp);
  note(slide, 850, 372, 300, 92, "Combined detector", "Fisher fusion lets two weaker p-value signals become one stronger decision statistic.", C.combo);
  text(slide, "Caveat: Combined benign detection is reported through the revised all-method comparison and robustness experiment, while baseline EXP/KGW detection uses the n=20 news setup.", 612, 512, 520, 50, { size: 14, color: C.muted });
  footer(slide, "Source: revised paper, Tables 3, 8, and 9.", 6);
}

function slide07(p) {
  const slide = p.slides.add();
  bg(slide);
  header(slide, "Experimental result analysis", "Combined sits closest to natural text quality instead of drifting high or low.", 7);
  text(slide, "Mean PPL by GPT-2; natural baseline = 19.63", 90, 220, 500, 22, { size: 14, color: C.muted, bold: true });
  hBars(slide, data.ppl, 90, 274, 430, 70, 26);
  rule(slide, 90 + 130 + 19.63 / 26 * 430, 250, 1, 260, C.combo);
  text(slide, "natural\n19.63", 90 + 130 + 19.63 / 26 * 430 + 8, 246, 64, 36, { size: 10.5, color: C.combo });
  rect(slide, 742, 240, 340, 250, C.paper, C.hair, 1);
  rect(slide, 742, 240, 8, 250, C.combo);
  text(slide, "Quality interpretation", 770, 268, 240, 30, { size: 22, bold: true });
  text(slide, "KGW raises PPL above baseline, suggesting visible distribution distortion. EXP pushes PPL far below baseline, suggesting reduced lexical diversity.", 770, 318, 248, 76, { size: 15.5, color: C.ink });
  rule(slide, 770, 414, 238, 1);
  text(slide, "Combined PPL = 17.26, with delta PPL = -2.37, making it closest to natural output in this table.", 770, 438, 248, 44, { size: 15, color: C.muted });
  footer(slide, "Source: revised paper, Table 4 and Section 5.3.", 7);
}

function slide08(p) {
  const slide = p.slides.add();
  bg(slide);
  header(slide, "Experimental result analysis", "Baseline sensitivity explains why neither single method is the final answer.", 8);
  rect(slide, 82, 218, 480, 326, C.paper, C.hair, 1);
  text(slide, "KGW strength delta", 112, 244, 240, 24, { size: 19, bold: true });
  text(slide, "TPR rises only when delta becomes aggressive, which can distort quality.", 112, 276, 345, 38, { size: 14.5, color: C.muted });
  vBars(slide, data.kgwDelta.map(d => ({ label: d.label, value: d.value, color: d.label === "2.0" ? C.warn : C.kgw, valueLabel: d.value.toFixed(2) })), 118, 360, 360, 112, { max: 1, gap: 20 });
  rect(slide, 640, 218, 480, 326, C.paper, C.hair, 1);
  text(slide, "EXP threshold tau", 670, 244, 240, 24, { size: 19, bold: true });
  text(slide, "EXP is stable over threshold choices, but still degrades when token positions are deleted.", 670, 276, 345, 38, { size: 14.5, color: C.muted });
  vBars(slide, data.expTau.map(d => ({ label: d.label, value: d.value, color: C.exp, valueLabel: d.value.toFixed(2) })), 682, 360, 320, 112, { max: 1, gap: 38 });
  note(slide, 416, 566, 430, 58, "Fusion logic", "Combined does not replace the baselines; it exploits their complementary failure modes.", C.combo);
  footer(slide, "Source: revised paper, Tables 5-6 and Section 4.4.", 8);
}

function slide09(p) {
  const slide = p.slides.add();
  bg(slide);
  header(slide, "Experimental result analysis", "Single-method attacks expose the robustness gap that motivated fusion.", 9);
  text(slide, "Baseline n=20 attack TPR", 92, 230, 260, 22, { size: 14, bold: true, color: C.muted });
  groupedRobustness(slide, data.baselineAttacks, 104, 300, 680, 190, false);
  rect(slide, 865, 238, 280, 250, C.paper, C.hair, 1);
  rect(slide, 865, 238, 8, 250, C.warn);
  text(slide, "Failure mode", 895, 266, 180, 26, { size: 22, bold: true });
  text(slide, "Both KGW and EXP are token-level watermarks. Word-S changes token identities; Word-D removes scored positions.", 895, 314, 205, 80, { size: 15.5 });
  rule(slide, 895, 412, 190, 1);
  text(slide, "The old conclusion was fragility. The revised paper asks whether two fragile signals can become a stronger joint detector.", 895, 436, 214, 48, { size: 14.5, color: C.muted });
  note(slide, 104, 560, 690, 64, "Baseline result", "Word-S reduces both EXP and KGW to 0.050 TPR; Word-D leaves EXP at 0.000 and KGW at 0.100.", C.warn);
  footer(slide, "Source: revised paper, Table 7 and Section 4.5.", 9);
}

function slide10(p) {
  const slide = p.slides.add();
  bg(slide);
  header(slide, "Experimental result analysis", "Combined improves robustness most where deletion weakens EXP.", 10);
  text(slide, "n=50 C4 robustness TPR; FPR = 0.000 in all conditions", 92, 224, 520, 22, { size: 14, bold: true, color: C.muted });
  groupedRobustness(slide, data.hybridRobustness, 92, 306, 760, 210, true);
  metric(slide, 910, 240, 210, 132, "Word-S gain", "+4 pp", "Combined over EXP: 0.90 vs. 0.86.", C.combo);
  metric(slide, 910, 398, 210, 132, "Word-D gain", "+32 pp", "Combined over EXP: 0.68 vs. 0.36.", C.warn);
  text(slide, "Why the gain is asymmetric", 96, 566, 250, 22, { size: 15, bold: true, color: C.muted });
  text(slide, "Deletion shortens the sequence and hurts EXP's joint p-value. KGW still scores surviving tokens, and Fisher fusion can lift marginal cases above the threshold.", 96, 596, 770, 38, { size: 16, color: C.ink });
  footer(slide, "Source: revised paper, Table 8 and Sections 4.6, 5.4.", 10);
}

function slide11(p) {
  const slide = p.slides.add();
  bg(slide);
  header(slide, "Experimental result analysis", "Combined is the recommended adversarial-deployment option.", 11);
  const x = 64;
  const y = 220;
  const widths = [190, 205, 205, 230, 190];
  const rows = [
    ["Dimension", "EXP", "KGW", "Combined", "Verdict"],
    ["No-attack detection", "TPR 1.00, FPR 0.00", "TPR 0.80, FPR 0.00", "TPR 1.00, FPR 0.00", "EXP / Combined"],
    ["Text quality", "PPL 11.17", "PPL 25.05", "PPL 17.26, near baseline", "Combined"],
    ["Word-S robustness", "TPR 0.86", "TPR 0.20", "TPR 0.90", "Combined"],
    ["Word-D robustness", "TPR 0.36", "TPR 0.12", "TPR 0.68", "Combined"],
    ["Complexity", "Moderate", "Simple", "Highest", "Trade-off"],
  ];
  for (let r = 0; r < rows.length; r += 1) {
    let cx = x;
    for (let c = 0; c < rows[r].length; c += 1) {
      tableCell(slide, rows[r][c], cx, y + r * 52, widths[c], 52, {
        fill: r === 0 ? C.dark : (r % 2 === 0 ? C.paper2 : C.paper),
        color: r === 0 ? C.bg : (c === 4 && rows[r][c] === "Combined" ? C.combo : C.ink),
        bold: r === 0 || c === 0 || c === 4,
        align: c === 4 ? "center" : "left",
        size: r === 0 ? 12 : 12.4,
      });
      cx += widths[c];
    }
  }
  note(slide, 990, 550, 180, 70, "Deployment read", "Use Combined when paraphrasing or deletion attacks are realistic.", C.combo);
  footer(slide, "Source: revised paper, Table 9 and Section 5.5.", 11);
}

function slide12(p) {
  const slide = p.slides.add();
  bg(slide, C.dark);
  text(slide, "Summary", 72, 68, 380, 56, { title: true, size: 48, bold: true, color: C.bg });
  text(slide, "The revised contribution is statistical fusion: two token-level signals, one stronger detector.", 72, 132, 780, 42, { size: 23, color: "#DDE3E8" });
  rule(slide, 72, 196, 180, 2, C.combo);
  const takeaways = [
    ["01", C.exp, "EXP remains the stronger standalone baseline.", "It keeps perfect benign detection and low PPL, but deletion still reduces TPR to 0.36 in the larger robustness table."],
    ["02", C.kgw, "KGW provides interpretable residual signal.", "Its green-list z-score is fragile alone, yet surviving tokens can still contribute under deletion."],
    ["03", C.combo, "Combined delivers the best overall balance.", "TPR reaches 1.00 without attack, 0.90 under Word-S, 0.68 under Word-D, and PPL stays close to natural baseline."],
    ["04", C.warn, "The open problem is not solved completely.", "Combined raises the robustness bar, but Word-D TPR of 0.68 leaves room for semantic-level watermarking."],
  ];
  for (let i = 0; i < takeaways.length; i += 1) {
    const y = 238 + i * 92;
    rect(slide, 76, y, 52, 52, takeaways[i][1]);
    text(slide, takeaways[i][0], 76, y + 14, 52, 24, { size: 18, bold: true, color: C.bg, align: "center" });
    text(slide, takeaways[i][2], 154, y - 2, 720, 26, { size: 22, bold: true, color: C.bg });
    text(slide, takeaways[i][3], 154, y + 34, 760, 34, { size: 15.5, color: "#BFC7CE" });
  }
  rect(slide, 942, 280, 210, 208, C.dark2, "#3B4651", 1);
  text(slide, "Future direction", 968, 306, 150, 28, { size: 20, bold: true, color: C.bg });
  text(slide, "Move beyond exact token identity toward semantic or structural watermark signals that survive meaning-preserving transformations.", 968, 356, 150, 92, { size: 15, color: "#DDE3E8" });
  footer(slide, "Source: revised paper, Conclusion.", 12, true);
}

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

async function main() {
  await fs.mkdir(PREVIEW, { recursive: true });
  await fs.mkdir(LAYOUT, { recursive: true });
  await fs.mkdir(path.dirname(OUT), { recursive: true });

  const p = Presentation.create({ slideSize: { width: 1280, height: 720 } });
  [
    slide01,
    slide02,
    slide03,
    slide04,
    slide05,
    slide06,
    slide07,
    slide08,
    slide09,
    slide10,
    slide11,
    slide12,
  ].forEach(fn => fn(p));

  const previewPaths = [];
  for (let i = 0; i < p.slides.items.length; i += 1) {
    const slide = p.slides.items[i];
    const stem = `slide-${String(i + 1).padStart(2, "0")}`;
    const png = await p.export({ slide, format: "png", scale: 1 });
    const pngPath = path.join(PREVIEW, `${stem}.png`);
    await writeBlob(pngPath, png);
    previewPaths.push(pngPath);
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(path.join(LAYOUT, `${stem}.layout.json`), await layout.text(), "utf8");
  }

  const montage = await p.export({ format: "webp", montage: true, scale: 1 });
  await writeBlob(path.join(PREVIEW, "deck-montage.webp"), montage);

  const pptx = await PresentationFile.exportPptx(p);
  await pptx.save(OUT);

  const stat = await fs.stat(OUT);
  await fs.writeFile(path.join(WORK, "manifest.json"), JSON.stringify({
    output: OUT,
    outputBytes: stat.size,
    slideCount: p.slides.items.length,
    previewPaths,
  }, null, 2), "utf8");
  console.log(JSON.stringify({ output: OUT, outputBytes: stat.size, slideCount: p.slides.items.length }, null, 2));
}

main().catch((error) => {
  console.error(error.stack || error.message || String(error));
  process.exitCode = 1;
});
