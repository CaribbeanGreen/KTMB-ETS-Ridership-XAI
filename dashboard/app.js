// ETS ridership surge predictor - hand-written front end (no framework)
// Reads window.ETS_DATA (06_prepare_dashboard.py: 2026 holdout with actual figures) and, if present,
// window.ETS_FORECAST (07_forecast.py: future dates, no actual figures yet).
(function () {
  "use strict";
  const D = window.ETS_DATA;
  const FC = window.ETS_FORECAST || null;
  const $ = (id) => document.getElementById(id);
  if (!D) {
    document.querySelector("main").innerHTML =
      '<p>No data found. Run <strong>python 06_prepare_dashboard.py</strong> ' +
      "first; it creates dashboard/data.js next to this page.</p>";
    return;
  }

  const DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];
  const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December"];
  const fmt = (n) => Math.round(n).toLocaleString("en-MY");
  const signed = (n) => (n >= 0 ? "+" : "−") + fmt(Math.abs(n));

  // ---- index rows: key "district|date" -> row object
  const F = D.rowFields;
  const V = D.valueFields;
  const rows = D.rows.map((r) => {
    const o = {};
    F.forEach((k, i) => (o[k] = r[i]));
    o.v = {};
    V.forEach((k, i) => (o.v[k] = o.values[i]));
    o.base = D.baseValue;
    o.forecast = false;
    return o;
  });
  // forecast rows: same layout, but no actual ridership and their own model baseline
  if (FC) {
    const idx = new Map(D.districts.map((d, i) => [d.name, i]));
    FC.rows.forEach((r) => {
      const o = {};
      FC.rowFields.forEach((k, i) => (o[k] = r[i]));
      o.district = idx.get(o.districtName);
      if (o.district === undefined) return;
      o.actual = null;
      o.v = {};
      V.forEach((k, i) => (o.v[k] = o.values[i]));
      o.base = FC.baseValue;
      o.forecast = true;
      rows.push(o);
    });
  }
  const byKey = new Map(rows.map((r) => [r.district + "|" + r.date, r]));
  const dates = [...new Set(rows.map((r) => r.date))].sort();
  const minDate = dates[0];
  const maxDate = dates[dates.length - 1];
  const networkByDate = new Map();
  rows.forEach((r) => {
    const n = networkByDate.get(r.date) || { actual: 0, pred: 0 };
    n.actual = r.actual === null || n.actual === null ? null : n.actual + r.actual;
    n.pred += r.pred;
    networkByDate.set(r.date, n);
  });

  // ---- plain-language description of each calendar / weather condition
  function describe(feature, r) {
    const v = r.v;
    switch (feature) {
      case "day_of_week": return DAYS[v.day_of_week];
      case "is_weekend": return v.is_weekend ? "Weekend" : "Weekday";
      case "month": return MONTHS[v.month - 1];
      case "is_public_holiday": return v.is_public_holiday ? "Public holiday: " + r.holiday : "Not a public holiday";
      case "is_holiday_eve": return v.is_holiday_eve ? "Day before a public holiday" : "Not the day before a holiday";
      case "days_to_festival":
        if (v.days_to_festival === 0) return "First day of " + (r.festival || "a festival");
        if (v.days_to_festival >= 30) return "No festival in the next 30 days";
        return v.days_to_festival + (v.days_to_festival === 1 ? " day" : " days") + " before " + (r.festival || "a festival");
      case "days_since_festival":
        if (v.days_since_festival >= 30) return "No festival in the past 30 days";
        if (v.days_since_festival === 0) return "Festival day";
        return v.days_since_festival + (v.days_since_festival === 1 ? " day" : " days") + " after " + (r.festival || "a festival");
      case "is_festive_period": return v.is_festive_period ? "Festive travel period" + (r.festival ? " (" + r.festival + ")" : "") : "Outside festive travel periods";
      case "is_ramadan": return v.is_ramadan ? "Ramadan" : "Not Ramadan";
      case "temp_mean_c": return (r.wx === 2 ? "Typical " + MONTHS[v.month - 1] + " temperature " : "Average temperature ") +
        v.temp_mean_c.toFixed(1) + " °C";
      case "rain_mm":
        if (r.wx === 2) return "Typical " + MONTHS[v.month - 1] + " rainfall " + v.rain_mm.toFixed(1) + " mm";
        return v.rain_mm < 0.1 ? "Dry day" : "Rainfall " + v.rain_mm.toFixed(1) + " mm" + (r.wx === 1 ? " (forecast)" : "");
      default: return feature;
    }
  }

  function shortDate(iso) {
    const d = new Date(iso + "T00:00:00");
    return d.getDate() + " " + MONTHS[d.getMonth()].slice(0, 3) + " " + d.getFullYear();
  }
  function longDate(iso) {
    const d = new Date(iso + "T00:00:00");
    return DAYS[(d.getDay() + 6) % 7] + ", " + d.getDate() + " " + MONTHS[d.getMonth()] + " " + d.getFullYear();
  }
  function shiftDate(iso, days) {
    const d = new Date(iso + "T00:00:00");
    d.setDate(d.getDate() + days);
    return d.getFullYear() + "-" + String(d.getMonth() + 1).padStart(2, "0") + "-" + String(d.getDate()).padStart(2, "0");
  }
  function changeStatus(pct) {
    if (pct >= 15) return ["Surge expected", "surge"];
    if (pct >= 5) return ["Busier than usual", "busy"];
    if (pct <= -5) return ["Quieter than usual", "quiet"];
    return ["Normal demand", ""];
  }

  // ---- controls
  const sel = $("district");
  const byState = {};
  D.districts.forEach((d, i) => (byState[d.state] = byState[d.state] || []).push([i, d.name]));
  Object.keys(byState).sort().forEach((state) => {
    const g = document.createElement("optgroup");
    g.label = state;
    byState[state].forEach(([i, name]) => {
      const o = document.createElement("option");
      o.value = i;
      o.textContent = name.replace("W.P. ", "");
      g.appendChild(o);
    });
    sel.appendChild(g);
  });
  const dateIn = $("date");
  dateIn.min = minDate;
  dateIn.max = maxDate;
  // ---- pipeline status (report Figure 3.5: input & processing layer)
  const DS = D.dataset || {};
  (DS.sources || ["KTMB ETS ridership", "Open-Meteo weather", "DOSM", "Public holidays"]).forEach((name) => {
    const li = document.createElement("li");
    li.textContent = name;
    $("sources").appendChild(li);
  });
  if (DS.tuned && DS.params) {
    $("pipe-status").textContent = "Cleaned & partitioned (80% train / 20% test). Tuned with " + DS.cvFolds +
      "-fold time-series cross-validation inside the 80% training part: " + DS.params.n_trees + " trees, max depth " +
      DS.params.max_depth + ", min samples per leaf " + DS.params.min_samples_leaf + ".";
  }
  if (DS.studyRows) {
    $("pipe-note").textContent = "Joined dataset: " + DS.studyRows.toLocaleString("en-MY") +
      " district-days from " + DS.districts + " districts, " + DS.features + " input features. Split by date: 80% train (" +
      DS.trainPeriod[0] + " to " + DS.trainPeriod[1] + "), 20% test (" + DS.testPeriod[0] + " to " + DS.testPeriod[1] +
      "). The " + DS.trees + "-tree final model was then retrained on all of 2023 to 2025." +
      (FC ? " Forecast model: same settings, retrained on " + FC.trainRows.toLocaleString("en-MY") + " district-days (" +
        FC.trainedOn[0] + " to " + FC.trainedOn[1] + "); weather from the Open-Meteo forecast for the next 16 days, monthly averages after that." : "");
  }

  const lastActual = rows.filter((r) => !r.forecast).reduce((m, r) => (r.date > m ? r.date : m), minDate);
  $("range-note").textContent = FC
    ? "Dates up to " + shortDate(lastActual) + " show the actual figures beside the prediction; later dates, to " +
      shortDate(maxDate) + ", are forecasts."
    : "Dates from " + shortDate(minDate) + " to " + shortDate(maxDate) + ", never used to train the model.";

  // ---- render
  function render() {
    const k = Number(sel.value);
    const date = dateIn.value;
    const r = byKey.get(k + "|" + date);
    const dist = D.districts[k];
    $("b-district").textContent = dist.name.replace("W.P. ", "");
    $("b-state").textContent = dist.state;
    $("b-date").textContent = date ? longDate(date) : "—";
    const board = $("board");
    if (!r) {
      board.classList.add("empty");
      ["b-pred", "b-actual", "b-change"].forEach((id) => ($(id).textContent = "—"));
      $("b-error").textContent = "";
      $("b-status").textContent = "No data for this day";
      $("b-status").className = "status";
      $("b-network").textContent = "Choose a date between " + longDate(minDate) + " and " + longDate(maxDate) + ".";
      $("why-lead").textContent = "";
      $("baseline").textContent = "";
      $("factors").innerHTML = "";
      $("factor-total").textContent = "";
      $("inputs").innerHTML = "";
      $("actions").innerHTML = "";
      drawTrend(k, date);
      return;
    }
    board.classList.remove("empty");
    $("b-pred").textContent = fmt(r.pred);
    if (r.forecast) {
      $("b-actual").textContent = "—";
      $("b-error").textContent = "not happened yet";
    } else {
      $("b-actual").textContent = fmt(r.actual);
      const err = r.pred - r.actual;
      $("b-error").textContent = (err >= 0 ? "over" : "under") + " by " + fmt(Math.abs(err)) +
        " (" + (r.actual ? Math.abs(err / r.actual * 100).toFixed(1) : "—") + "%)";
    }
    const pct = (r.pred / r.normalSameWeekday - 1) * 100;
    $("b-compare-label").textContent = "Compared with a normal " + DAYS[r.v.day_of_week];
    $("b-change").textContent = (pct >= 0 ? "+" : "−") + Math.abs(pct).toFixed(0) + "%";
    const [label, cls] = changeStatus(pct);
    $("b-status").textContent = label;
    $("b-status").className = "status " + cls;
    const n = networkByDate.get(date);
    $("b-network").textContent = "Whole ETS network that day: " + fmt(n.pred) + " trips predicted" +
      (n.actual === null ? "" : ", " + fmt(n.actual) + " actual") + ". Weather: " +
      (r.forecast ? FC.weatherLabels[r.wx] : "observed (Open-Meteo archive)") + ".";
    renderWhy(r, dist);
    renderActions(r, dist, pct);
    drawTrend(k, date);
  }

  function renderWhy(r, dist) {
    const items = D.temporal.map((f, i) => ({ f, value: r.shap[i] }))
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
    const top = items.slice(0, 6);
    const rest = items.slice(6).reduce((s, it) => s + it.value, 0);
    const list = top.map((it) => ({ name: describe(it.f, r), value: it.value }));
    list.push({ name: "Other conditions (" + (items.length - 6) + ")", value: rest });
    const up = top.filter((it) => it.value > 0).slice(0, 3).map((it) => describe(it.f, r));
    $("why-lead").textContent = up.length
      ? "Main reasons for extra trips: " + up.join("; ") + "."
      : "No condition on this day adds trips; the prediction sits at or below the district's normal level.";
    $("baseline").textContent = "Normal level for " + dist.name.replace("W.P. ", "") +
      " (set by its population and income): " + fmt(r.districtLevel) + " trips";
    $("factors").innerHTML = shapChart(list);
    const NAMES = { day_of_week: "Day of week", is_weekend: "Weekend", month: "Month", is_public_holiday: "Public holiday",
      is_holiday_eve: "Holiday eve", days_to_festival: "Days to next festival", days_since_festival: "Days since last festival",
      is_festive_period: "Festive period", is_ramadan: "Ramadan", temp_mean_c: "Average temperature", rain_mm: "Rainfall" };
    const tb = $("inputs");
    tb.innerHTML = "";
    const addRow = (a, b, c) => {
      const tr = document.createElement("tr");
      [a, b, c].forEach((t) => { const td = document.createElement("td"); td.textContent = t; tr.appendChild(td); });
      tb.appendChild(tr);
    };
    D.temporal.forEach((f, i) => addRow(NAMES[f] || f, describe(f, r), signed(r.shap[i])));
    if (dist.population_k !== undefined) {
      addRow("District population", fmt(dist.population_k * 1000), "");
      addRow("Median household income", "RM " + fmt(dist.income), signed(r.districtLevel - r.base) + " (both)");
    }
    const total = r.shap.reduce((s, v) => s + v, 0);
    $("factor-total").textContent = fmt(r.districtLevel) + " normal level " + signed(total) +
      " from today's conditions = " + fmt(r.pred) + " trips predicted";
  }

  // ---- TreeSHAP bar chart: inline SVG, red = adds trips, black = removes trips
  const esc = (t) => String(t).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  function shapChart(list) {
    const W = 720, rowH = 28, labelW = 320, padR = 70, mid = labelW + (W - labelW - padR) / 2;
    const half = (W - labelW - padR) / 2 - 10;
    const H = list.length * rowH + 10;
    const maxAbs = Math.max(...list.map((x) => Math.abs(x.value)), 1);
    let svg = '<svg class="chart" viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="TreeSHAP contributions in trips" ' +
      'font-family="sans-serif" font-size="13">';
    svg += '<line x1="' + mid + '" x2="' + mid + '" y1="0" y2="' + H + '" stroke="#000000" stroke-width="1"></line>';
    list.forEach((x, i) => {
      const yTop = 5 + i * rowH, yMid = yTop + rowH / 2;
      const w = Math.max(Math.abs(x.value) / maxAbs * half, 1);
      const up = x.value >= 0;
      const bx = up ? mid : mid - w;
      svg += '<text x="0" y="' + (yMid + 4) + '" fill="#000000">' + esc(x.name) + "</text>";
      svg += '<rect x="' + bx.toFixed(1) + '" y="' + (yTop + 5) + '" width="' + w.toFixed(1) + '" height="' + (rowH - 10) +
        '" fill="' + (up ? "#c8102e" : "#000000") + '"><title>' + esc(x.name) + ": " + signed(x.value) + " trips</title></rect>";
      const tx = up ? bx + w + 6 : bx - 6;
      svg += '<text x="' + tx.toFixed(1) + '" y="' + (yMid + 4) + '" text-anchor="' + (up ? "start" : "end") + '" fill="#000000">' +
        signed(x.value) + "</text>";
    });
    return svg + "</svg>";
  }

  // ---- recommended actions (report Algorithm 3.2, GENERATE_RECOMMENDATIONS):
  //      rules on the surge level plus the strongest TreeSHAP factors for that day
  const FESTIVE = ["days_to_festival", "days_since_festival", "is_festive_period", "is_holiday_eve", "is_public_holiday"];
  function recommend(r, dist, pct) {
    const place = dist.name.replace("W.P. ", "");
    const v = r.v;
    const out = [];
    if (pct >= 15) {
      out.push("Add coaches or extra ETS services on departures from " + place + ", and roster extra ticketing and platform staff.");
    } else if (pct >= 5) {
      out.push("Keep standby coaches ready and add counter staff for the peak departure hours at " + place + ".");
    } else if (pct <= -5) {
      out.push("Run the normal timetable; a suitable day for planned maintenance or staff leave at " + place + ".");
    } else {
      out.push("Run the normal timetable and staffing at " + place + ".");
    }
    const shap = {};
    D.temporal.forEach((f, i) => (shap[f] = r.shap[i]));
    const festivePush = FESTIVE.reduce((s, f) => s + Math.max(shap[f], 0), 0);
    const fest = r.festival || "the festival";
    if (festivePush >= 0.05 * r.pred && pct >= 5) {
      if (v.days_to_festival <= 7 && v.days_to_festival <= v.days_since_festival) {
        out.push("Outbound festive travel before " + fest + ": prioritise departures in the days before it and expect a return wave after it.");
      } else if (v.days_since_festival <= 7) {
        out.push("Return travel after " + fest + ": plan extra capacity on return journeys.");
      } else if (v.is_holiday_eve) {
        out.push("Eve of " + (r.holiday || "a public holiday") + ": expect the heaviest departures in the afternoon and evening.");
      } else if (v.is_public_holiday) {
        out.push("Public holiday (" + r.holiday + "): expect leisure trips spread through the day.");
      }
    }
    if (v.rain_mm >= 20) {
      out.push("Heavy rain expected (" + v.rain_mm.toFixed(0) + " mm): prepare for crowded sheltered platforms and possible delays; check flood alerts.");
    }
    if (v.is_ramadan && shap.is_ramadan < 0) {
      out.push("Ramadan lowers demand; expect it to build sharply in the final week before Hari Raya.");
    }
    if (r.forecast && r.wx === 2) {
      out.push("This far ahead the weather is a monthly average; press Update forecast within 16 days of the date for a real weather forecast.");
    }
    return out;
  }
  function renderActions(r, dist, pct) {
    const ul = $("actions");
    ul.innerHTML = "";
    recommend(r, dist, pct).forEach((t) => {
      const li = document.createElement("li");
      li.textContent = t;
      ul.appendChild(li);
    });
  }

  // ---- trend chart: 21 days centred on the selected date (inline SVG)
  function drawTrend(k, date) {
    const box = $("trend");
    if (!date) { box.innerHTML = ""; return; }
    const pts = [];
    for (let i = -10; i <= 10; i++) {
      const d = shiftDate(date, i);
      const r = byKey.get(k + "|" + d);
      if (r) pts.push({ i, d, actual: r.actual, pred: r.pred });
    }
    if (!pts.length) { box.innerHTML = "<p>No data around this date.</p>"; return; }
    const W = 720, H = 300, m = { l: 60, r: 14, t: 12, b: 30 };
    const vals = pts.flatMap((p) => (p.actual === null ? [p.pred] : [p.actual, p.pred]));
    let lo = Math.min(...vals), hi = Math.max(...vals);
    const pad = Math.max((hi - lo) * 0.12, 5);
    lo = Math.max(0, lo - pad); hi = hi + pad;
    const rawStep = (hi - lo) / 4;
    const mag = Math.pow(10, Math.floor(Math.log10(rawStep)));
    const tick = [1, 2, 2.5, 5, 10].map((k) => k * mag).find((k) => k >= rawStep);
    lo = Math.floor(lo / tick) * tick; hi = Math.ceil(hi / tick) * tick;
    const x = (i) => m.l + (i + 10) / 20 * (W - m.l - m.r);
    const y = (v) => m.t + (1 - (v - lo) / (hi - lo)) * (H - m.t - m.b);
    const line = (key) => {
      let d = "", pen = false;
      pts.forEach((p) => {
        if (p[key] === null) { pen = false; return; }
        d += (pen ? "L" : "M") + x(p.i).toFixed(1) + "," + y(p[key]).toFixed(1);
        pen = true;
      });
      return d;
    };
    let svg = '<svg class="chart" viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="Actual and predicted trips for 21 days" ' +
      'font-family="sans-serif" font-size="12">';
    svg += '<g fill="#000000">';
    for (let v = lo; v <= hi + 1e-9; v += tick) {
      svg += '<line x1="' + m.l + '" x2="' + (W - m.r) + '" y1="' + y(v) + '" y2="' + y(v) + '" stroke="#000000" stroke-opacity="0.15"></line>';
      svg += '<text x="' + (m.l - 8) + '" y="' + (y(v) + 4) + '" text-anchor="end">' + fmt(v) + "</text>";
    }
    [-10, -5, 0, 5, 10].forEach((i) => {
      const d = new Date(shiftDate(date, i) + "T00:00:00");
      const anchor = i === -10 ? "start" : i === 10 ? "end" : "middle";
      svg += '<text x="' + x(i) + '" y="' + (H - 8) + '" text-anchor="' + anchor + '">' + d.getDate() + " " +
        MONTHS[d.getMonth()].slice(0, 3) + "</text>";
    });
    svg += "</g>";
    svg += '<line x1="' + x(0) + '" x2="' + x(0) + '" y1="' + m.t + '" y2="' + (H - m.b) + '" stroke="#000000" stroke-width="1" stroke-dasharray="4 4"></line>';
    svg += '<path d="' + line("actual") + '" fill="none" stroke="#000000" stroke-width="2"></path>';
    svg += '<path d="' + line("pred") + '" fill="none" stroke="#c8102e" stroke-width="2"></path>';
    const sel0 = pts.find((p) => p.i === 0);
    if (sel0) svg += '<circle cx="' + x(0) + '" cy="' + y(sel0.pred) + '" r="5" fill="#c8102e"></circle>';
    const step = (W - m.l - m.r) / 20;
    pts.forEach((p) => {
      svg += '<rect class="hit" data-d="' + p.d + '" x="' + (x(p.i) - step / 2) + '" y="' + m.t + '" width="' + step +
        '" height="' + (H - m.t - m.b) + '" fill="transparent" cursor="pointer"><title>' + longDate(p.d) + ": " +
        (p.actual === null ? "forecast " + fmt(p.pred) : "actual " + fmt(p.actual) + ", predicted " + fmt(p.pred)) +
        " trips (click to select)</title></rect>";
    });
    svg += "</svg>";
    const known = pts.filter((p) => p.actual !== null);
    let note;
    if (!known.length) {
      note = "Forecast only: these days have not happened yet, so there is no actual line to compare.";
    } else {
      const errAvg = known.reduce((a, p) => a + Math.abs(p.pred - p.actual), 0) / known.length;
      const actAvg = known.reduce((a, p) => a + p.actual, 0) / known.length;
      note = "Over the " + known.length + " days with actual figures the prediction was off by " + fmt(errAvg) +
        " trips a day on average (" + (errAvg / actAvg * 100).toFixed(1) + "% of actual demand)." +
        (known.length < pts.length ? " The rest are forecasts." : "");
    }
    box.innerHTML = svg + "<p>" + note + "</p>";

    box.querySelectorAll(".hit").forEach((rect) => {
      const p = pts.find((q) => q.d === rect.dataset.d);
      rect.addEventListener("click", () => { dateIn.value = p.d; render(); });
    });
  }

  // ---- biggest predicted surges
  function renderPeaks() {
    const avg = {};
    rows.forEach((r) => ((avg[r.district] = avg[r.district] || []).push(r.pred)));
    const big = new Set(Object.keys(avg).filter((k) => avg[k].reduce((a, b) => a + b, 0) / avg[k].length >= 100).map(Number));
    const pool = FC ? rows.filter((r) => r.forecast && r.date >= FC.issued) : rows;
    if (FC) {
      $("peaks-title").textContent = "Biggest upcoming surge in each district";
      $("peaks-hint").textContent = "From " + shortDate(FC.issued) + " onward, ranked by extra trips above a normal day. " +
        "Select one to open it.";
    }
    const ranked = pool.filter((r) => big.has(r.district))
      .map((r) => ({ r, pct: (r.pred / r.normalSameWeekday - 1) * 100, extra: r.pred - r.normalSameWeekday }))
      .sort((a, b) => b.extra - a.extra)
      .filter((x, i, all) => all.findIndex((y) => y.r.district === x.r.district) === i)   // biggest per district
      .slice(0, 12);
    const ol = $("peaks");
    ranked.forEach(({ r, pct, extra }) => {
      const li = document.createElement("li");
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = D.districts[r.district].name.replace("W.P. ", "") + ": +" + fmt(extra) + " trips (+" +
        pct.toFixed(0) + "%), " + longDate(r.date) + (r.festival ? ", " + r.festival : "");
      b.addEventListener("click", () => {
        sel.value = r.district;
        dateIn.value = r.date;
        render();
        $("board").scrollIntoView({ behavior: "smooth", block: "start" });
      });
      li.appendChild(b);
      ol.appendChild(li);
    });
    return ranked[0];
  }

  // ---- wire up
  const predictBtn = $("predict");
  const markStale = () => { predictBtn.classList.add("stale"); $("stale-note").textContent = "Inputs changed: press Predict."; };
  const predict = () => { predictBtn.classList.remove("stale"); $("stale-note").textContent = ""; render(); };
  sel.addEventListener("change", markStale);
  dateIn.addEventListener("change", markStale);
  predictBtn.addEventListener("click", predict);
  $("prev").addEventListener("click", () => { if (dateIn.value > minDate) { dateIn.value = shiftDate(dateIn.value, -1); render(); } });
  $("next").addEventListener("click", () => { if (dateIn.value < maxDate) { dateIn.value = shiftDate(dateIn.value, 1); render(); } });

  const first = renderPeaks();
  const kl = D.districts.findIndex((d) => d.name.includes("Kuala Lumpur"));
  sel.value = first ? first.r.district : kl;
  dateIn.value = first ? first.r.date : minDate;
  render();

  const M = D.metrics;
  $("footer").textContent = "Model: Random Forest (" + (DS.trees || 100) + " trees) implemented in NumPy, trained on ETS ridership from " +
    "2023 to 2025 with weather, public holidays, festivals and DOSM district data. Explanations: TreeSHAP. " +
    "On the unseen 2026 data the network daily R² is " + M.network_r2 + " and the festive-day error is " +
    M.festive_mape + "%, against " + M.naive_festive_mape + "% for a simple same-weekday average. From May 2026 it ran " +
    "about 12% below actual demand as ridership grew, so treat forecasts as a conservative estimate. Data prepared " +
    D.generated + (FC ? "; forecast issued " + FC.generated : "") + ".";

  // ---- light / dark mode (styling in style.css under [data-theme="dark"]; the choice is remembered per browser)
  const modeBtn = $("mode");
  const showMode = () => {
    const dark = document.documentElement.getAttribute("data-theme") === "dark";
    modeBtn.textContent = dark ? "Light mode" : "Dark mode";
    modeBtn.setAttribute("aria-pressed", String(dark));
  };
  if (modeBtn) {
    showMode();
    modeBtn.addEventListener("click", () => {
      const next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      try { localStorage.setItem("ets-mode", next); } catch (e) { /* storage blocked: mode still applies now */ }
      showMode();
    });
  }

  // ---- one-click data update (needs the dashboard opened through serve.py / Start Dashboard.bat)
  const U = { status: $("update-status"), prompt: $("update-prompt"), progress: $("update-progress"),
    log: $("update-log"), fc: $("update-forecast"), full: $("update-full") };
  const setButtons = (on) => { U.fc.disabled = !on; U.full.disabled = !on; };
  function showStatus(st) {
    const s = st.state || {};
    U.status.textContent = "Actual data to " + shortDate(lastActual) + ". " +
      (FC ? "Forecast issued " + shortDate(FC.issued) + "." : "No forecast yet.") +
      (s.last ? " Last updated here " + s.last + "." : "");
    const stale = !FC || FC.issued < st.today;
    U.prompt.hidden = !(stale && !st.job.running);
    if (stale) {
      U.prompt.textContent = FC ? "The forecast is from " + shortDate(FC.issued) + ". Update it for today's weather."
        : "No forecast yet. Run Update forecast.";
    }
  }
  const mins = (sec, pct) => (sec == null || (pct === 0 && !sec) ? "estimating time" :
    sec >= 90 ? "about " + Math.round(sec / 60) + " min left" : sec > 15 ? "under 2 min left" : "almost done");
  function showJob(j) {
    const run = $("update-run");
    if (!j.name) { run.hidden = true; $("update-details").hidden = true; return; }
    run.hidden = false;
    $("update-details").hidden = false;
    const bar = $("update-bar");
    bar.value = Math.round((j.progress || 0) * 100);
    $("update-pct").textContent = bar.value + "%";
    if (j.running) {
      $("update-eta").textContent = mins(j.eta, bar.value);
      U.progress.textContent = (j.name === "full" ? "Full data update" : "Updating forecast") + ": " +
        (j.phase || "working") + (j.steps > 1 ? " (step " + j.step + " of " + j.steps + ")" : "") + ".";
    } else {
      $("update-eta").textContent = j.ok ? "done" : "stopped";
      U.progress.textContent = j.ok ? "Updated at " + j.finished + "." :
        "The update stopped with an error at " + j.finished + ". The previous data is still shown; open Show details to see why.";
      if (!j.ok) $("update-details").open = true;
    }
    U.log.textContent = j.lines.join("\n");
    U.log.scrollTop = U.log.scrollHeight;
  }
  let polling = null;
  function poll() {
    fetch("/api/status", { cache: "no-store" }).then((r) => r.json()).then((st) => {
      showStatus(st);
      showJob(st.job);
      setButtons(!st.job.running);
      if (st.job.running && !polling) polling = setInterval(poll, 2000);
      if (!st.job.running && !st.job.ok && st.job.name) $("update-run").hidden = false;
      if (!st.job.running && polling) {
        clearInterval(polling);
        polling = null;
        if (st.job.ok) { U.progress.textContent = "Done. Loading the new data..."; setTimeout(() => location.reload(), 1500); }
      }
    }).catch(() => {
      U.status.textContent = "The update server is not responding. Close this page and start it again with Start Dashboard.bat.";
      setButtons(false);
    });
  }
  function start(mode) {
    if (mode === "full" && !window.confirm("A full data update downloads new ETS ridership and weather, rebuilds the " +
      "dataset and retrains the forecast model. It takes about 30 to 60 minutes. Start now?")) return;
    setButtons(false);
    U.prompt.hidden = true;
    fetch("/api/update?mode=" + mode, { method: "POST" }).then((r) => r.json()).then((res) => {
      if (res.error) U.progress.textContent = res.error;
      poll();
    });
  }
  U.fc.addEventListener("click", () => start("forecast"));
  U.full.addEventListener("click", () => start("full"));
  if (location.protocol === "file:") {
    setButtons(false);
    U.status.textContent = "Updates are off because this page was opened as a file. Open it with Start Dashboard.bat to update.";
    if (FC && FC.issued < new Date().toISOString().slice(0, 10)) {
      U.prompt.hidden = false;
      U.prompt.textContent = "The forecast is from " + shortDate(FC.issued) + ".";
    }
  } else {
    poll();
  }
})();
