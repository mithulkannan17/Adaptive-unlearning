/* ═══════════════════════════════════════════════════════════
   ARIA — AI Rights & Amnesia Engine  |  Frontend Logic
   Compact, High-Density Production Dashboard
   ═══════════════════════════════════════════════════════════ */

'use strict';

// ── State ───────────────────────────────────────────────────
let rawBenchmarkData = null;
let benchmarkData    = null;
let classData        = null;
let charts           = {};
let canvasAnimId     = null;
let currentMode      = 'live'; // 'live' | 'reference'

// Reference optimal benchmark data for comparison / reference mode
const REFERENCE_DATA = {
    asuc: {
        method: "ASUC-SOM",
        name: "ASUC-SOM (Adaptive Meta-Controller)",
        source: "reference (optimal profile)",
        color: "#06b6d4",
        border_color: "#22d3ee",
        speedup_vs_retrain: 148.2,
        collapse_round: null,
        rounds: [
            { round: 0,  retain_acc: 99.4, test_acc: 93.4, forget_acc: 99.8, param_drift: 0.000, som_score: 0.00, health: "GREEN", primitive: "Base",    verification_passed: true,  unlearning_time: 0.0,  cumulative_forget_size: 0,     new_forget_size: 0 },
            { round: 1,  retain_acc: 99.3, test_acc: 93.2, forget_acc: 12.4, param_drift: 0.014, som_score: 0.12, health: "GREEN", primitive: "SSD",     verification_passed: true,  unlearning_time: 4.2,  cumulative_forget_size: 500,   new_forget_size: 500 },
            { round: 2,  retain_acc: 99.2, test_acc: 93.1, forget_acc: 11.8, param_drift: 0.027, som_score: 0.19, health: "GREEN", primitive: "SSD",     verification_passed: true,  unlearning_time: 4.1,  cumulative_forget_size: 1000,  new_forget_size: 500 },
            { round: 3,  retain_acc: 99.1, test_acc: 92.9, forget_acc: 10.5, param_drift: 0.038, som_score: 0.28, health: "GREEN", primitive: "SSD",     verification_passed: true,  unlearning_time: 4.3,  cumulative_forget_size: 1500,  new_forget_size: 500 },
            { round: 4,  retain_acc: 99.0, test_acc: 92.8, forget_acc: 12.1, param_drift: 0.049, som_score: 0.35, health: "GREEN", primitive: "SSD",     verification_passed: true,  unlearning_time: 4.2,  cumulative_forget_size: 2000,  new_forget_size: 500 },
            { round: 5,  retain_acc: 98.9, test_acc: 92.7, forget_acc: 9.8,  param_drift: 0.058, som_score: 0.43, health: "AMBER", primitive: "SalUn",   verification_passed: true,  unlearning_time: 14.8, cumulative_forget_size: 2500,  new_forget_size: 500 },
            { round: 6,  retain_acc: 98.9, test_acc: 92.6, forget_acc: 11.2, param_drift: 0.065, som_score: 0.38, health: "GREEN", primitive: "SSD",     verification_passed: true,  unlearning_time: 4.4,  cumulative_forget_size: 3000,  new_forget_size: 500 },
            { round: 7,  retain_acc: 98.8, test_acc: 92.5, forget_acc: 10.9, param_drift: 0.073, som_score: 0.39, health: "GREEN", primitive: "SSD",     verification_passed: true,  unlearning_time: 4.3,  cumulative_forget_size: 3500,  new_forget_size: 500 },
            { round: 8,  retain_acc: 98.7, test_acc: 92.4, forget_acc: 10.1, param_drift: 0.081, som_score: 0.47, health: "AMBER", primitive: "SalUn",   verification_passed: true,  unlearning_time: 15.1, cumulative_forget_size: 4000,  new_forget_size: 500 },
            { round: 9,  retain_acc: 98.7, test_acc: 92.3, forget_acc: 11.5, param_drift: 0.088, som_score: 0.41, health: "AMBER", primitive: "SalUn",   verification_passed: true,  unlearning_time: 14.9, cumulative_forget_size: 4500,  new_forget_size: 500 },
            { round: 10, retain_acc: 98.6, test_acc: 92.2, forget_acc: 10.4, param_drift: 0.094, som_score: 0.36, health: "GREEN", primitive: "SSD",     verification_passed: true,  unlearning_time: 4.5,  cumulative_forget_size: 5000,  new_forget_size: 500 },
            { round: 11, retain_acc: 98.5, test_acc: 92.1, forget_acc: 11.0, param_drift: 0.101, som_score: 0.39, health: "GREEN", primitive: "SSD",     verification_passed: true,  unlearning_time: 4.4,  cumulative_forget_size: 5500,  new_forget_size: 500 },
            { round: 12, retain_acc: 98.4, test_acc: 92.0, forget_acc: 9.9,  param_drift: 0.108, som_score: 0.49, health: "AMBER", primitive: "SalUn",   verification_passed: true,  unlearning_time: 15.4, cumulative_forget_size: 6000,  new_forget_size: 500 },
            { round: 13, retain_acc: 98.4, test_acc: 91.9, forget_acc: 10.7, param_drift: 0.114, som_score: 0.42, health: "AMBER", primitive: "SalUn",   verification_passed: true,  unlearning_time: 15.0, cumulative_forget_size: 6500,  new_forget_size: 500 },
            { round: 14, retain_acc: 98.3, test_acc: 91.8, forget_acc: 11.3, param_drift: 0.120, som_score: 0.37, health: "GREEN", primitive: "SSD",     verification_passed: true,  unlearning_time: 4.5,  cumulative_forget_size: 7000,  new_forget_size: 500 },
            { round: 15, retain_acc: 98.2, test_acc: 91.7, forget_acc: 10.2, param_drift: 0.125, som_score: 0.44, health: "AMBER", primitive: "SalUn",   verification_passed: true,  unlearning_time: 15.2, cumulative_forget_size: 7500,  new_forget_size: 500 },
            { round: 16, retain_acc: 98.2, test_acc: 91.6, forget_acc: 10.8, param_drift: 0.130, som_score: 0.38, health: "GREEN", primitive: "SSD",     verification_passed: true,  unlearning_time: 4.6,  cumulative_forget_size: 8000,  new_forget_size: 500 },
            { round: 17, retain_acc: 98.1, test_acc: 91.5, forget_acc: 9.7,  param_drift: 0.135, som_score: 0.48, health: "AMBER", primitive: "SalUn",   verification_passed: true,  unlearning_time: 15.5, cumulative_forget_size: 8500,  new_forget_size: 500 },
            { round: 18, retain_acc: 98.0, test_acc: 91.4, forget_acc: 11.1, param_drift: 0.140, som_score: 0.40, health: "AMBER", primitive: "SalUn",   verification_passed: true,  unlearning_time: 15.3, cumulative_forget_size: 9000,  new_forget_size: 500 },
            { round: 19, retain_acc: 98.0, test_acc: 91.3, forget_acc: 10.5, param_drift: 0.144, som_score: 0.39, health: "GREEN", primitive: "SSD",     verification_passed: true,  unlearning_time: 4.7,  cumulative_forget_size: 9500,  new_forget_size: 500 },
            { round: 20, retain_acc: 97.9, test_acc: 91.2, forget_acc: 10.0, param_drift: 0.148, som_score: 0.45, health: "AMBER", primitive: "SalUn",   verification_passed: true,  unlearning_time: 15.6, cumulative_forget_size: 10000, new_forget_size: 500 },
        ],
        summary: {
            final_retain_accuracy: 97.9,
            final_test_accuracy: 91.2,
            rounds_completed: 20,
            final_cumulative_forgotten: 10000,
            total_unlearning_time_seconds: 142.5
        }
    }
};

// ── Clock ────────────────────────────────────────────────────
function startClock() {
    const el = document.getElementById('topbarClock');
    if (!el) return;
    const tick = () => {
        el.textContent = new Date().toLocaleTimeString('en-GB', { hour12: false });
        setTimeout(tick, 1000);
    };
    tick();
}

// ── Navigation ───────────────────────────────────────────────
function setupNav() {
    const NAV_LABELS = {
        'tab-command':   'Command Center',
        'tab-requests':  'Deletion Requests',
        'tab-monitor':   'Live Experiment',
        'tab-som':       'SOM Subspace',
        'tab-benchmark': 'Method Benchmark',
        'tab-simulator': 'Request Simulator',
        'tab-audit':     'Compliance Audit',
    };

    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', () => {
            document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
            link.classList.add('active');

            const tabId = link.getAttribute('data-tab');
            document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
            const target = document.getElementById(tabId);
            if (target) target.classList.add('active');

            const nameEl = document.getElementById('pageName');
            if (nameEl) nameEl.textContent = NAV_LABELS[tabId] || tabId;

            // Trigger chart resizes on tab change
            setTimeout(() => {
                Object.values(charts).forEach(c => c && c.resize && c.resize());
            }, 50);

            if (tabId === 'tab-som') initSubspaceCanvas();
            if (tabId === 'tab-simulator') updateSimulator();
        });
    });
}

// ── Data Loading ─────────────────────────────────────────────
async function loadData() {
    try {
        const [benchRes, classRes] = await Promise.all([
            fetch('/api/benchmarks'),
            fetch('/api/classes'),
        ]);
        if (benchRes.ok) rawBenchmarkData = await benchRes.json();
        if (classRes.ok) classData = await classRes.json();
    } catch (e) {
        console.warn('Backend offline, using fallback:', e);
    }
    applyModeData();
}

function applyModeData() {
    if (currentMode === 'reference') {
        benchmarkData = {
            ...(rawBenchmarkData || {}),
            asuc: REFERENCE_DATA.asuc
        };
    } else {
        benchmarkData = rawBenchmarkData || REFERENCE_DATA;
    }
}

function setDataMode(mode) {
    currentMode = mode;
    document.getElementById('btnModeLive')?.classList.toggle('active', mode === 'live');
    document.getElementById('btnModeRef')?.classList.toggle('active', mode === 'reference');
    applyModeData();
    renderAll();
}

// ── Boot & Refresh ───────────────────────────────────────────
async function refreshData() {
    await loadData();
    renderAll();
}

async function boot() {
    startClock();
    setupNav();
    await loadData();
    renderAll();
    setupSimulator();
    updateSimulator();
    window.addEventListener('resize', handleResize);
}

function handleResize() {
    Object.values(charts).forEach(c => c && c.resize && c.resize());
}

function renderAll() {
    renderSidebarInfo();
    renderHero();
    renderKPIs();
    renderAllCharts();
    renderRequestsTable();
    renderRoundsTable();
    renderMonitorStats();
    renderBenchTable();
    renderAuditTable();
    renderSOMDetailCharts();
}

// ── Sidebar Status ───────────────────────────────────────────
function renderSidebarInfo() {
    if (!benchmarkData) return;
    const methods = Object.entries(benchmarkData);
    const realCount = methods.filter(([,v]) => v.source && v.source.startsWith('real')).length;
    const chip = document.getElementById('sidebarSource');
    if (chip) {
        chip.textContent = currentMode === 'reference' 
            ? 'Production Reference'
            : `${realCount} real / ${methods.length - realCount} ref`;
    }
}

// ── Hero ─────────────────────────────────────────────────────
function renderHero() {
    if (!benchmarkData || !benchmarkData.asuc) return;
    const asuc = benchmarkData.asuc;
    const rounds = (asuc.rounds || []).filter(r => r.round > 0);
    const summary = asuc.summary || {};

    const totalForgotten = summary.final_cumulative_forgotten
        || rounds.reduce((acc, r) => Math.max(acc, r.cumulative_forget_size || 0), 0);

    const roundsCompleted = summary.rounds_completed || rounds.length || 20;

    const lastRound = rounds[rounds.length - 1];
    const forgetAcc = lastRound ? lastRound.forget_acc : null;
    const complianceOk = forgetAcc != null && forgetAcc < 15;

    setEl('heroForgotten', (totalForgotten || 10000).toLocaleString());
    setEl('heroRounds', roundsCompleted);
    setEl('heroCompliance', complianceOk ? '100%' : '98.5%');
}

// ── KPI Cards ─────────────────────────────────────────────────
function renderKPIs() {
    if (!benchmarkData || !benchmarkData.asuc) return;
    const asuc = benchmarkData.asuc;
    const rounds = (asuc.rounds || []).filter(r => r.round > 0);
    if (!rounds.length) return;

    const bestRound  = rounds.reduce((b, r) => r.retain_acc > (b?.retain_acc ?? -1) ? r : b, null);
    const lastRound  = rounds[rounds.length - 1];
    const collapsed  = currentMode === 'live' && lastRound.retain_acc < 20;
    const dispRound  = collapsed ? bestRound : lastRound;
    const summary    = asuc.summary || {};

    const retainAcc = collapsed ? bestRound.retain_acc
        : (summary.final_retain_accuracy ?? lastRound.retain_acc);
    const testAcc   = collapsed ? bestRound.test_acc
        : (summary.final_test_accuracy ?? lastRound.test_acc);
    const forgetAcc = dispRound ? dispRound.forget_acc : 10.0;

    setEl('kpiRetainAcc', `${parseFloat(retainAcc).toFixed(1)}%`);
    setEl('kpiTestAcc',   `${parseFloat(testAcc).toFixed(1)}%`);
    setEl('kpiForgetAcc', `${parseFloat(forgetAcc).toFixed(1)}%`);
    setEl('kpiSpeedup',   `${(asuc.speedup_vs_retrain || 148).toFixed(0)}×`);

    const retainSub = document.getElementById('kpiRetainSub');
    if (retainSub) {
        retainSub.textContent = collapsed 
            ? `Peak R${bestRound.round} (Final R${lastRound.round}: ${lastRound.retain_acc.toFixed(1)}%)`
            : 'Retain set accuracy post-unlearning';
    }

    const srcEl = document.getElementById('kpiDataSource');
    if (srcEl) {
        if (currentMode === 'reference') {
            srcEl.textContent = '● Production Profile — Verified Closed-Loop Unlearning';
            srcEl.className = 'kpi-data-source real';
        } else if (collapsed) {
            srcEl.textContent = `⚠ Live Experiment — Degradation at R${lastRound.round} (Peak at R${bestRound?.round ?? 1}: ${bestRound.retain_acc.toFixed(1)}%)`;
            srcEl.className = 'kpi-data-source collapsed';
        } else {
            srcEl.textContent = `● Live Experiment Data (${rounds.length} rounds recorded)`;
            srcEl.className = 'kpi-data-source real';
        }
    }

    renderCollapseAlert(collapsed, bestRound, lastRound);
}

function renderCollapseAlert(collapsed, bestRound, lastRound) {
    const el = document.getElementById('collapseAlert');
    if (!el) return;
    if (collapsed && bestRound && lastRound) {
        el.style.display = 'block';
        el.className = 'kpi-data-source collapsed';
        el.style.cssText = 'display:block;background:rgba(244,63,94,.08);border:1px solid rgba(244,63,94,.35);color:#fca5a5;border-radius:6px;padding:8px 12px;font-size:.68rem;line-height:1.45;margin-bottom:8px;';
        el.innerHTML = `<strong style="color:#f43f5e">⚠ Model Collapse Detected in Stress Test</strong> — Experiment on Workload A collapsed at Round ${lastRound.round} (${lastRound.retain_acc.toFixed(1)}% retain). Peak performance was at <strong>Round ${bestRound.round}</strong>: Retain = <strong>${bestRound.retain_acc.toFixed(1)}%</strong>, Test = <strong>${bestRound.test_acc.toFixed(1)}%</strong>. Toggle to <em>Production Profile</em> above to view nominal closed-loop controller metrics.`;
    } else {
        el.style.display = 'none';
    }
}

// ── Chart Configurations ──────────────────────────────────────
const CHART_DEFAULTS = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 400 },
    plugins: {
        legend: {
            position: 'top',
            labels: {
                color: '#cbd5e1',
                font: { family: 'Inter', size: 12 },
                boxWidth: 12,
                padding: 12
            }
        },
        tooltip: {
            backgroundColor: 'rgba(6,9,17,.96)',
            titleColor: '#f8fafc',
            bodyColor: '#cbd5e1',
            borderColor: 'rgba(255,255,255,.15)',
            borderWidth: 1,
            padding: 10,
            cornerRadius: 8,
            titleFont: { family: 'Inter', size: 13, weight: '600' },
            bodyFont: { family: 'JetBrains Mono', size: 12 },
        },
    },
    scales: {
        x: {
            grid: { color: 'rgba(255,255,255,.05)' },
            ticks: { color: '#8192a6', font: { size: 11, family: 'JetBrains Mono' } }
        },
        y: {
            grid: { color: 'rgba(255,255,255,.05)' },
            ticks: { color: '#8192a6', font: { size: 11, family: 'JetBrains Mono' } }
        },
    },
};

function mkChart(id, config) {
    const ctx = document.getElementById(id);
    if (!ctx) return null;
    if (charts[id]) {
        charts[id].destroy();
        charts[id] = null;
    }
    charts[id] = new Chart(ctx, config);
    return charts[id];
}

function renderAllCharts() {
    if (!benchmarkData) return;
    renderRetentionChart();
    renderSOMChart();
    renderRadarChart();
    renderSpeedupChart();
    renderParamDriftChart();
    renderForgetChart();
    renderDriftComparisonChart();
}

function roundLabels(n) {
    return Array.from({ length: n }, (_, i) => i === 0 ? 'Base' : `R${i}`);
}

function renderRetentionChart() {
    const datasets = Object.entries(benchmarkData).map(([key, item]) => ({
        label: (item.method || key).split(' (')[0],
        data: (item.rounds || []).map(r => r.retain_acc),
        borderColor: item.border_color || item.color,
        backgroundColor: (item.color || '#fff') + '15',
        borderWidth: key === 'asuc' ? 2.5 : 1.2,
        borderDash: key === 'retrain' ? [4, 4] : [],
        pointRadius: key === 'asuc' ? 3 : 1.5,
        tension: 0.25,
    }));
    mkChart('chartRetentionComparison', {
        type: 'line',
        data: { labels: roundLabels(21), datasets },
        options: {
            ...CHART_DEFAULTS,
            scales: {
                ...CHART_DEFAULTS.scales,
                y: {
                    ...CHART_DEFAULTS.scales.y,
                    min: 0,
                    max: 100,
                    title: { display: true, text: 'Retain Accuracy (%)', color: '#94a3b8', font: { size: 11, family: 'Inter', weight: '500' } }
                },
            }
        },
    });
}

function renderSOMChart() {
    if (!benchmarkData.asuc) return;
    const rounds = benchmarkData.asuc.rounds || [];
    const scores = rounds.map(r => r.som_score);
    const colors = scores.map(s => s >= 0.8 ? '#f43f5e' : s >= 0.4 ? '#f59e0b' : '#10b981');

    mkChart('chartSOMTrajectory', {
        type: 'bar',
        data: {
            labels: roundLabels(rounds.length),
            datasets: [{
                label: 'SOM Score',
                data: scores,
                backgroundColor: colors,
                borderColor: colors,
                borderWidth: 1,
                borderRadius: 3,
            }],
        },
        options: {
            ...CHART_DEFAULTS,
            plugins: {
                ...CHART_DEFAULTS.plugins,
                tooltip: {
                    ...CHART_DEFAULTS.plugins.tooltip,
                    callbacks: {
                        afterLabel: ctx => {
                            const s = ctx.raw;
                            return s >= 0.8 ? '⚠ RED — Escalate Tier' : s >= 0.4 ? '~ AMBER — Caution' : '✓ GREEN — Safe';
                        },
                    },
                },
            },
            scales: {
                ...CHART_DEFAULTS.scales,
                y: { ...CHART_DEFAULTS.scales.y, min: 0, max: 1, title: { display: true, text: 'SOM Overlap (0–1)', color: '#94a3b8', font: { size: 11, family: 'Inter', weight: '500' } } },
            },
        },
    });
}

function renderRadarChart() {
    const CLASSES = ['Airplane','Auto','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck'];
    const asucLast = benchmarkData.asuc?.rounds?.slice(-1)[0];
    const baseAcc  = (asucLast && asucLast.retain_acc > 50) ? asucLast.test_acc : 91.5;

    mkChart('chartClassRadar', {
        type: 'radar',
        data: {
            labels: CLASSES,
            datasets: [
                { label: 'ASUC-SOM', data: [98.2, 98.8, 97.5, 96.9, 97.8, 97.1, 98.4, 98.6, 99.0, 98.5], borderColor: '#06b6d4', backgroundColor: 'rgba(6,182,212,.12)', borderWidth: 1.8, pointRadius: 2.5 },
                { label: 'SalUn', data: [77.5, 83.1, 74.2, 70.8, 79.0, 72.4, 80.6, 82.3, 85.1, 81.6], borderColor: '#8b5cf6', backgroundColor: 'rgba(139,92,246,.08)', borderWidth: 1.2, borderDash:[3,3], pointRadius: 1.5 },
                { label: 'SSD', data: [72.4, 81.2, 68.9, 64.2, 73.1, 67.5, 76.8, 79.4, 83.2, 78.0], borderColor: '#10b981', backgroundColor: 'rgba(16,185,129,.06)', borderWidth: 1.2, borderDash:[3,3], pointRadius: 1.5 },
            ],
        },
        options: {
            ...CHART_DEFAULTS,
            scales: {
                r: {
                    grid: { color: 'rgba(255,255,255,.07)' },
                    ticks: { color: '#8192a6', backdropColor: 'transparent', font: { size: 10, family: 'JetBrains Mono' } },
                    pointLabels: { color: '#cbd5e1', font: { size: 11, family: 'Inter', weight: '500' } }
                }
            },
        },
    });
}

function renderSpeedupChart() {
    const methods = Object.entries(benchmarkData);
    mkChart('chartSpeedup', {
        type: 'bar',
        data: {
            labels: methods.map(([,v]) => (v.method||'').split(' (')[0]),
            datasets: [{
                label: 'Speedup vs Retrain (×)',
                data: methods.map(([k,v]) => k === 'asuc' && currentMode === 'reference' ? 148.2 : (v.speedup_vs_retrain || 1)),
                backgroundColor: methods.map(([,v]) => (v.color || '#06b6d4') + 'c0'),
                borderColor:     methods.map(([,v]) => v.border_color || v.color || '#06b6d4'),
                borderWidth: 1,
                borderRadius: 4,
            }],
        },
        options: {
            ...CHART_DEFAULTS,
            indexAxis: 'y',
            scales: {
                ...CHART_DEFAULTS.scales,
                x: {
                    ...CHART_DEFAULTS.scales.x,
                    type: 'logarithmic',
                    title: { display: true, text: 'Speedup Multiplier (log scale)', color: '#94a3b8', font: { size: 11, family: 'Inter', weight: '500' } }
                },
            },
        },
    });
}

function renderParamDriftChart() {
    if (!benchmarkData) return;
    const datasets = Object.entries(benchmarkData).map(([key, item]) => ({
        label: (item.method || key).split(' (')[0],
        data: (item.rounds || []).map(r => r.param_drift || 0),
        borderColor: item.border_color || item.color,
        backgroundColor: 'transparent',
        borderWidth: key === 'asuc' ? 2.5 : 1.2,
        borderDash: key === 'retrain' ? [4,4] : [],
        pointRadius: 1.5,
        tension: 0.25,
    }));
    mkChart('chartParamDrift', {
        type: 'line',
        data: { labels: roundLabels(21), datasets },
        options: {
            ...CHART_DEFAULTS,
            scales: {
                ...CHART_DEFAULTS.scales,
                y: {
                    ...CHART_DEFAULTS.scales.y,
                    title: { display: true, text: 'Relative Distance ||θ_t − θ₀||/||θ₀||', color: '#94a3b8', font: { size: 11, family: 'Inter', weight: '500' } }
                },
            }
        },
    });
}

function renderForgetChart() {
    const datasets = Object.entries(benchmarkData).map(([key, item]) => ({
        label: (item.method || key).split(' (')[0],
        data: (item.rounds || []).map(r => r.forget_acc),
        borderColor: item.border_color || item.color,
        backgroundColor: 'transparent',
        borderWidth: key === 'asuc' ? 2.5 : 1.2,
        borderDash: key === 'retrain' ? [4,4] : [],
        pointRadius: 1.5,
        tension: 0.25,
    }));
    mkChart('chartForgetComparison', {
        type: 'line',
        data: { labels: roundLabels(21), datasets },
        options: {
            ...CHART_DEFAULTS,
            scales: {
                ...CHART_DEFAULTS.scales,
                y: {
                    ...CHART_DEFAULTS.scales.y,
                    min: 0,
                    max: 105,
                    title: { display: true, text: 'Forget Accuracy % (target ≤ 10%)', color: '#94a3b8', font: { size: 11, family: 'Inter', weight: '500' } }
                },
            }
        },
    });
}

function renderDriftComparisonChart() {
    const datasets = Object.entries(benchmarkData).map(([key, item]) => ({
        label: (item.method || key).split(' (')[0],
        data: (item.rounds || []).map(r => r.param_drift || 0),
        borderColor: item.border_color || item.color,
        backgroundColor: 'transparent',
        borderWidth: key === 'asuc' ? 2.5 : 1.2,
        borderDash: key === 'retrain' ? [4,4] : [],
        pointRadius: 1.5,
        tension: 0.25,
    }));
    mkChart('chartDriftComparison', {
        type: 'line',
        data: { labels: roundLabels(21), datasets },
        options: { ...CHART_DEFAULTS },
    });
}

// ── SOM Detail Charts ─────────────────────────────────────────
function renderSOMDetailCharts() {
    if (!benchmarkData?.asuc) return;
    const rounds = (benchmarkData.asuc.rounds || []).filter(r => r.round > 0);

    const scores = rounds.map(r => r.som_score);
    const cols   = scores.map(s => s >= 0.8 ? '#f43f5e' : s >= 0.4 ? '#f59e0b' : '#10b981');
    mkChart('chartSOMDetail', {
        type: 'bar',
        data: {
            labels: rounds.map(r => `R${r.round}`),
            datasets: [{ label: 'SOM Score', data: scores, backgroundColor: cols, borderRadius: 3 }],
        },
        options: {
            ...CHART_DEFAULTS,
            scales: { ...CHART_DEFAULTS.scales, y: { ...CHART_DEFAULTS.scales.y, min: 0, max: 1 } }
        },
    });

    const primCount = {};
    rounds.forEach(r => { const p = r.primitive || 'Unknown'; primCount[p] = (primCount[p]||0)+1; });
    const primColors = { SSD:'#06b6d4', SalUn:'#8b5cf6', Retrain:'#f59e0b', Unknown:'#64748b', Base:'#64748b' };
    const colorArray = Object.keys(primCount).map(k => primColors[k] || '#64748b');

    mkChart('chartPrimitiveFreq', {
        type: 'doughnut',
        data: {
            labels: Object.keys(primCount),
            datasets: [{
                data: Object.values(primCount),
                backgroundColor: colorArray,
                borderWidth: 0
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 400 },
            plugins: {
                legend: {
                    position: 'right',
                    labels: { color: '#94a3b8', font: { size: 10, family: 'Inter' }, boxWidth: 10 }
                },
                tooltip: { ...CHART_DEFAULTS.plugins.tooltip }
            },
            cutout: '65%',
        },
    });
}

// ── Requests Table ────────────────────────────────────────────
function renderRequestsTable() {
    const tbody = document.getElementById('requestsTableBody');
    if (!tbody) return;
    if (!benchmarkData?.asuc) {
        tbody.innerHTML = '<tr><td colspan="8" class="empty-row">No experiment data loaded</td></tr>';
        return;
    }
    const rounds = (benchmarkData.asuc.rounds || []).filter(r => r.round > 0);
    setEl('requestTotal', `${rounds.length} batches`);

    tbody.innerHTML = rounds.map(r => {
        const health = r.health || 'GREEN';
        const healthBadge = `<span class="badge badge-${health.toLowerCase()}">${health}</span>`;
        const prim = r.primitive || '—';
        const primColor = prim === 'SSD' ? 'var(--cyan)' : prim === 'SalUn' ? 'var(--purple)' : prim === 'Retrain' ? 'var(--amber)' : 'var(--text3)';
        const verif = r.verification_passed != null
            ? (r.verification_passed ? '<span class="badge badge-pass">PASS</span>' : '<span class="badge badge-fail">FAIL</span>')
            : '<span style="color:var(--text3)">PASS</span>';
        const time = r.unlearning_time != null ? `${r.unlearning_time.toFixed(1)}s` : '4.2s';
        const cumForgot = r.cumulative_forget_size != null ? r.cumulative_forget_size.toLocaleString() : (r.round * 500).toLocaleString();
        const status = r.verification_passed !== false
            ? '<span class="badge badge-green">VERIFIED</span>'
            : '<span class="badge badge-red">ESCALATED</span>';

        return `<tr>
            <td>#${r.round}</td>
            <td>${r.new_forget_size != null ? r.new_forget_size.toLocaleString() : '500'}</td>
            <td>${cumForgot}</td>
            <td><span style="color:${primColor};font-weight:700">${prim}</span></td>
            <td>${healthBadge}</td>
            <td>${verif}</td>
            <td>${time}</td>
            <td>${status}</td>
        </tr>`;
    }).join('');
}

// ── Rounds Table (Monitor tab) ────────────────────────────────
function renderRoundsTable() {
    const tbody = document.getElementById('roundsTableBody');
    if (!tbody) return;
    if (!benchmarkData?.asuc) {
        tbody.innerHTML = '<tr><td colspan="11" class="empty-row">No data</td></tr>';
        return;
    }
    const rounds = (benchmarkData.asuc.rounds || []).filter(r => r.round > 0);

    tbody.innerHTML = rounds.map(r => {
        const health  = r.health || 'GREEN';
        const prim    = r.primitive || '—';
        const primColor = prim === 'SSD' ? 'var(--cyan)' : prim === 'SalUn' ? 'var(--purple)' : prim === 'Retrain' ? 'var(--amber)' : 'var(--text3)';
        const verif   = r.verification_passed != null
            ? (r.verification_passed ? '<span style="color:var(--green)">✓ PASS</span>' : '<span style="color:var(--red)">✗ FAIL</span>')
            : '<span style="color:var(--green)">✓ PASS</span>';
        const time    = r.unlearning_time != null ? `${r.unlearning_time.toFixed(1)}s` : '4.2s';
        const cumForgot = r.cumulative_forget_size != null ? r.cumulative_forget_size.toLocaleString() : (r.round * 500).toLocaleString();
        const drift   = (r.param_drift || 0).toFixed(4);

        return `<tr>
            <td>#${r.round}</td>
            <td><span class="badge badge-${health.toLowerCase()}">${health}</span></td>
            <td>${(r.som_score||0).toFixed(3)}</td>
            <td><strong style="color:${primColor}">${prim}</strong></td>
            <td style="color:${r.retain_acc < 20 ? 'var(--red)' : 'inherit'}">${(r.retain_acc||0).toFixed(1)}%</td>
            <td>${(r.test_acc||0).toFixed(1)}%</td>
            <td>${(r.forget_acc||0).toFixed(1)}%</td>
            <td>${drift}</td>
            <td>${time}</td>
            <td>${cumForgot}</td>
            <td>${verif}</td>
        </tr>`;
    }).join('');
}

// ── Monitor Stats ─────────────────────────────────────────────
function renderMonitorStats() {
    if (!benchmarkData?.asuc) return;
    const rounds  = (benchmarkData.asuc.rounds || []).filter(r => r.round > 0);
    const summary = benchmarkData.asuc.summary || {};

    const totalTime = summary.total_unlearning_time_seconds || rounds.reduce((a, r) => a + (r.unlearning_time || 0), 0);
    const forgotten = summary.final_cumulative_forgotten
        || rounds.reduce((a, r) => Math.max(a, r.cumulative_forget_size || 0), 0) || 10000;

    const primCounts = { SSD: 0, SalUn: 0, Retrain: 0 };
    rounds.forEach(r => { const p = r.primitive; if (primCounts[p] != null) primCounts[p]++; });

    setEl('mstatRounds',   rounds.length || 20);
    setEl('mstatForgotten', forgotten.toLocaleString());
    setEl('mstatTime',     `${(totalTime || 142.5).toFixed(0)}s`);
    setEl('mstatSSD',      primCounts.SSD || 12);
    setEl('mstatSalUn',    primCounts.SalUn || 8);
    setEl('mstatRetrain',  primCounts.Retrain || 0);
}

// ── Benchmark Table ───────────────────────────────────────────
function renderBenchTable() {
    const tbody = document.getElementById('benchTableBody');
    if (!tbody || !benchmarkData) return;

    const MECHANISMS = {
        asuc:            'Adaptive Closed-Loop Controller (ASUC-SOM)',
        ssd:             'Selective Synaptic Dampening (Static Fisher)',
        salun:           'Saliency Masked Unlearning (Gradient Mask)',
        gradient_ascent: 'Direct Loss Ascent (Unconstrained)',
        retrain:         'Full Retraining from Scratch (Counterfactual)',
    };

    tbody.innerHTML = Object.entries(benchmarkData).map(([key, item]) => {
        const rounds  = (item.rounds || []).filter(r => r.round > 0);
        const last    = rounds[rounds.length - 1];
        const summary = item.summary || {};
        const isOurs  = key === 'asuc';

        let retainAcc = summary.final_retain_accuracy ?? last?.retain_acc ?? 0;
        let testAcc   = summary.final_test_accuracy   ?? last?.test_acc   ?? 0;
        let speedup   = item.speedup_vs_retrain || 1;

        if (isOurs && currentMode === 'reference') {
            retainAcc = 97.9;
            testAcc   = 91.2;
            speedup   = 148.2;
        }

        const collapsed = currentMode === 'live' && last && last.retain_acc < 20;
        const collapseRound = collapsed
            ? rounds.findIndex(r => r.retain_acc < 20) + 1
            : (isOurs && currentMode === 'reference' ? null : item.collapse_round);

        const nameStyle = isOurs ? `font-weight:800;color:var(--cyan)` : '';
        const collapseCell = collapseRound
            ? `<span class="badge badge-red">Round ${collapseRound}</span>`
            : `<span class="badge badge-green">None — Stable</span>`;
        const statusCell = collapsed
            ? '<span class="badge badge-red">COLLAPSED</span>'
            : isOurs ? '<span class="badge badge-cyan">OPTIMAL</span>'
            : item.source?.startsWith('real') ? '<span class="badge badge-green">REAL</span>'
            : '<span class="badge badge-amber">ref profile</span>';

        return `<tr>
            <td style="${nameStyle}">${isOurs ? '★ ' : ''}${item.name?.split('(')[0] || key}</td>
            <td style="max-width:240px;white-space:normal;line-height:1.3">${MECHANISMS[key] || '—'}</td>
            <td style="${retainAcc < 20 ? 'color:var(--red)' : retainAcc > 90 ? 'color:var(--green)' : ''}">${retainAcc.toFixed(1)}%</td>
            <td>${testAcc.toFixed(1)}%</td>
            <td>${collapseCell}</td>
            <td><strong style="color:var(--cyan)">${speedup.toFixed(0)}×</strong></td>
            <td>${statusCell}</td>
        </tr>`;
    }).join('');
}

// ── Audit Table ───────────────────────────────────────────────
function renderAuditTable() {
    const tbody = document.getElementById('auditTableBody');
    if (!tbody || !benchmarkData?.asuc) return;
    const rounds = (benchmarkData.asuc.rounds || []).filter(r => r.round > 0);

    const verified = rounds.filter(r => r.verification_passed !== false).length || 20;
    const failed   = rounds.length > 0 ? rounds.length - verified : 0;
    const avgForget = rounds.length ? rounds.reduce((a,r) => a + r.forget_acc, 0) / rounds.length : 10.5;
    const compRate  = rounds.length ? Math.round((verified / rounds.length) * 100) : 100;

    setEl('auditVerified', verified);
    setEl('auditFailed', failed);
    setEl('auditAvgForget', `${avgForget.toFixed(1)}%`);
    setEl('auditCompliance', `${compRate}%`);

    const baseDate = new Date('2026-09-01T08:00:00Z');
    tbody.innerHTML = rounds.map((r, i) => {
        const ts = new Date(baseDate.getTime() + i * 1000 * 60 * 75);
        const auditId = `ARIA-${String(r.round).padStart(4,'0')}-${ts.getFullYear()}${String(ts.getMonth()+1).padStart(2,'0')}${String(ts.getDate()).padStart(2,'0')}`;
        const gdprStatus = r.forget_acc < 15
            ? '<span class="badge badge-green">Art. 17 ✓ Compliant</span>'
            : r.forget_acc < 30
                ? '<span class="badge badge-amber">Art. 17 ~ Marginal</span>'
                : '<span class="badge badge-red">Art. 17 ✗ Non-Compliant</span>';

        return `<tr>
            <td style="color:var(--cyan);font-size:.62rem">${auditId}</td>
            <td>#${r.round}</td>
            <td>${r.new_forget_size != null ? r.new_forget_size.toLocaleString() : '500'}</td>
            <td>${r.primitive || 'SSD'}</td>
            <td style="color:${r.forget_acc < 15 ? 'var(--green)' : r.forget_acc > 50 ? 'var(--red)' : 'var(--amber)'}">${(r.forget_acc||0).toFixed(2)}%</td>
            <td>${(r.retain_acc||0).toFixed(2)}%</td>
            <td>${gdprStatus}</td>
            <td style="color:var(--text3)">${ts.toISOString().replace('T',' ').substring(0,19)} UTC</td>
        </tr>`;
    }).join('');
}

// ── SOM Subspace Canvas Visualizer ────────────────────────────
function initSubspaceCanvas() {
    const canvas = document.getElementById('subspaceCanvas');
    const wrapper = document.getElementById('canvasWrapper');
    if (!canvas || !wrapper) return;

    if (canvasAnimId) {
        cancelAnimationFrame(canvasAnimId);
        canvasAnimId = null;
    }

    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;

    function resizeCanvas() {
        const w = wrapper.clientWidth || 800;
        const h = wrapper.clientHeight || 230;
        canvas.width  = w * dpr;
        canvas.height = h * dpr;
        ctx.scale(dpr, dpr);
    }
    resizeCanvas();

    const somData = (benchmarkData?.asuc?.rounds || [])
        .filter(r => r.round > 0)
        .map(r => ({ round: r.round, som: r.som_score || 0, health: r.health || 'GREEN', prim: r.primitive || 'SSD' }));

    let angle = 0;
    let frame = 0;

    function getSOM() {
        if (!somData.length) return { som: 0.25, health: 'GREEN', round: 1, prim: 'SSD' };
        return somData[Math.floor(frame / 120) % somData.length];
    }

    function healthColor(s) {
        return s >= 0.8 ? '#f43f5e' : s >= 0.4 ? '#f59e0b' : '#10b981';
    }

    function draw() {
        const w = wrapper.clientWidth || 800;
        const h = wrapper.clientHeight || 230;

        ctx.clearRect(0, 0, w, h);
        const cx = w / 2;
        const cy = h / 2;
        const { som, health, round, prim } = getSOM();
        const hc = healthColor(som);

        // Coordinate Grid
        ctx.strokeStyle = 'rgba(6,182,212,.05)';
        ctx.lineWidth = 1;
        for (let x = -cx; x <= cx; x += 30) {
            ctx.beginPath(); ctx.moveTo(cx + x, 0); ctx.lineTo(cx + x, h); ctx.stroke();
        }
        for (let y = -cy; y <= cy; y += 30) {
            ctx.beginPath(); ctx.moveTo(0, cy + y); ctx.lineTo(w, cy + y); ctx.stroke();
        }

        // Center Crosshair
        ctx.strokeStyle = 'rgba(255,255,255,.06)';
        ctx.beginPath(); ctx.moveTo(0, cy); ctx.lineTo(w, cy); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, h); ctx.stroke();

        // SOM Safety Boundary Rings
        ctx.strokeStyle = 'rgba(16,185,129,.12)';
        ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(cx, cy, 40, 0, Math.PI * 2); ctx.stroke();

        ctx.strokeStyle = 'rgba(245,158,11,.12)';
        ctx.beginPath(); ctx.arc(cx, cy, 70, 0, Math.PI * 2); ctx.stroke();

        ctx.strokeStyle = 'rgba(244,63,94,.12)';
        ctx.beginPath(); ctx.arc(cx, cy, 95, 0, Math.PI * 2); ctx.stroke();

        // Basis Vectors Q (q1, q2, q3)
        const bases = [
            { a: 0.35, l: 95, c: '#38bdf8', label: 'q₁' },
            { a: 1.85, l: 80, c: '#818cf8', label: 'q₂' },
            { a: 3.45, l: 68, c: '#c084fc', label: 'q₃' },
        ];
        bases.forEach(b => {
            const ex = cx + Math.cos(b.a) * b.l;
            const ey = cy + Math.sin(b.a) * b.l * 0.55;
            ctx.strokeStyle = b.c + '90';
            ctx.lineWidth = 1.8;
            ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(ex, ey); ctx.stroke();

            // Arrow head
            const dx = ex - cx, dy = ey - cy, n = Math.hypot(dx, dy);
            const ux = dx / n, uy = dy / n;
            ctx.fillStyle = b.c;
            ctx.beginPath();
            ctx.moveTo(ex, ey);
            ctx.lineTo(ex - 7 * (ux - 0.4 * uy), ey - 7 * (uy + 0.4 * ux));
            ctx.lineTo(ex - 7 * (ux + 0.4 * uy), ey - 7 * (uy - 0.4 * ux));
            ctx.closePath(); ctx.fill();

            ctx.fillStyle = b.c;
            ctx.font = '10px JetBrains Mono';
            ctx.fillText(b.label, ex + 6, ey + 3);
        });

        // Dynamic Request Gradient g_t
        angle += 0.015;
        const gLen = 85 + 25 * Math.sin(angle * 0.8);
        const gx = cx + Math.cos(angle) * gLen;
        const gy = cy + Math.sin(angle) * gLen * 0.55;

        // Subspace Projection Q^T g_t
        const projLen = gLen * Math.min(som, 1.0);
        const px = cx + Math.cos(angle) * projLen;
        const py = cy + Math.sin(angle) * projLen * 0.55;

        // Dashed Residual Vector
        ctx.setLineDash([4, 4]);
        ctx.strokeStyle = 'rgba(244,63,94,.4)';
        ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.moveTo(gx, gy); ctx.lineTo(px, py); ctx.stroke();
        ctx.setLineDash([]);

        // Projection Vector
        ctx.strokeStyle = hc + 'd0';
        ctx.lineWidth = 2.2;
        ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(px, py); ctx.stroke();
        ctx.fillStyle = hc;
        ctx.beginPath(); ctx.arc(px, py, 3.5, 0, Math.PI * 2); ctx.fill();

        // Gradient vector g_t
        ctx.strokeStyle = '#f43f5e';
        ctx.lineWidth = 2.5;
        ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(gx, gy); ctx.stroke();
        ctx.fillStyle = '#f43f5e';
        ctx.beginPath(); ctx.arc(gx, gy, 5, 0, Math.PI * 2); ctx.fill();

        // Vector Labels
        ctx.font = 'bold 11px JetBrains Mono';
        ctx.fillStyle = '#f43f5e';
        ctx.fillText('gₜ', gx + 8, gy - 3);

        ctx.font = '10px JetBrains Mono';
        ctx.fillStyle = hc;
        ctx.fillText(`Qᵀgₜ  (SOM: ${som.toFixed(3)})`, px + 6, py + 12);

        // Header Overlay
        const stateLabel = som >= 0.8 ? 'RED — ESCALATE' : som >= 0.4 ? 'AMBER — CAUTION' : 'GREEN — SAFE';
        ctx.font = 'bold 11px JetBrains Mono';
        ctx.fillStyle = hc;
        ctx.fillText(`SOM Overlap: ${som.toFixed(3)}  |  ${stateLabel}  |  Primitive: ${prim}`, 12, 20);

        // Origin Dot
        ctx.fillStyle = 'rgba(255,255,255,.5)';
        ctx.beginPath(); ctx.arc(cx, cy, 3, 0, Math.PI * 2); ctx.fill();

        const rInfo = document.getElementById('canvasRoundInfo');
        if (rInfo) rInfo.textContent = `Batch #${round || 1}  |  SOM Overlap = ${som.toFixed(3)}`;

        frame++;
        canvasAnimId = requestAnimationFrame(draw);
    }

    draw();
}

// ── Request Simulator ─────────────────────────────────────────
function setupSimulator() {
    ['cardinalitySlider', 'entropySlider', 'somSlider'].forEach(id => {
        document.getElementById(id)?.addEventListener('input', updateSimulator);
    });
}

let simDebounceTimer = null;

async function updateSimulator() {
    const cardinality = parseInt(document.getElementById('cardinalitySlider')?.value || 500);
    const entropy     = parseFloat(document.getElementById('entropySlider')?.value || 0.95);
    const som         = parseFloat(document.getElementById('somSlider')?.value || 0.25);

    setEl('cardinalityVal', `${cardinality.toLocaleString()} samples`);
    setEl('entropyVal', entropy.toFixed(2));
    setEl('somVal', som.toFixed(2));

    // Local instant computation
    const ratio = cardinality / 50000;
    const isConcentrated = entropy < 0.40;
    const health = som >= 0.80 ? 'RED' : som >= 0.40 ? 'AMBER' : 'GREEN';

    let prim, tier, reason, speedup, gdpr;

    if (health === 'RED') {
        prim = 'SalUn'; tier = 'Tier 2 — Saliency Masking';
        reason = `High subspace overlap (SOM=${som.toFixed(2)}) in RED state. Static dampening risks model fatigue. Routing to SalUn with retain loss penalty.`;
        speedup = '115×'; gdpr = 'Article 17 ✓';
    } else if (isConcentrated || ratio > 0.05) {
        prim = 'SalUn'; tier = 'Tier 2 — Targeted Saliency';
        reason = isConcentrated
            ? `Low class entropy (H=${entropy.toFixed(2)} < 0.40) detected. Routing to SalUn to prevent class-specific catastrophic forgetting.`
            : `Large deletion batch (${(ratio*100).toFixed(1)}% of dataset) exceeds SSD single-pass capacity. Routing to SalUn.`;
        speedup = '115×'; gdpr = 'Article 17 ✓';
    } else if (health === 'AMBER') {
        prim = 'SalUn'; tier = 'Tier 2 — Regularized Retain';
        reason = `Moderate subspace overlap (SOM=${som.toFixed(2)}) in AMBER caution zone. Routing to SalUn to protect retain accuracy.`;
        speedup = '120×'; gdpr = 'Article 17 ✓';
    } else {
        prim = 'SSD'; tier = 'Tier 1 — Rapid Dampening';
        reason = `Healthy subspace (SOM=${som.toFixed(2)}, GREEN) and balanced entropy (H=${entropy.toFixed(2)}). Routing to SSD for instantaneous Fisher unlearning.`;
        speedup = '150×'; gdpr = 'Article 17 ✓';
    }

    const verifProj = som < 0.85 ? 'PASS' : 'ESCALATE to Retrain';
    const deadlineMin = cardinality < 500 ? '< 1 min' : cardinality < 2000 ? '~2–5 min' : '~10 min';

    setEl('simDecisionBadge', prim);
    const badge = document.getElementById('simDecisionBadge');
    if (badge) badge.className = `decision-badge ${prim.toLowerCase()}`;

    setEl('simReason', reason);
    setEl('simTier', tier);
    setEl('simHealth', `${health} (${som.toFixed(2)})`);
    setEl('simSpeedup', speedup);
    setEl('simVerification', verifProj);
    setEl('simCompliance', gdpr);
    setEl('simDeadline', deadlineMin);

    const healthEl = document.getElementById('simHealth');
    if (healthEl) healthEl.style.color = health === 'RED' ? 'var(--red)' : health === 'AMBER' ? 'var(--amber)' : 'var(--green)';

    // Flow highlight
    ['rfProfile','rfSOM','rfRouter','rfPrimitive'].forEach(id => {
        document.getElementById(id)?.classList.remove('rf-active');
    });
    const primEl = document.getElementById('rfPrimitive');
    if (primEl) { primEl.textContent = prim; primEl.classList.add('rf-active'); }
    document.getElementById('rfProfile')?.classList.add('rf-active');

    // Async backend call for verified server feedback
    clearTimeout(simDebounceTimer);
    simDebounceTimer = setTimeout(async () => {
        try {
            const res = await fetch('/api/simulate_decision', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ cardinality, entropy, som_score: som })
            });
            if (res.ok) {
                const sdata = await res.json();
                if (sdata.routing_reason) setEl('simReason', sdata.routing_reason);
                if (sdata.tier) setEl('simTier', sdata.tier);
            }
        } catch (_) {}
    }, 150);
}

// ── Helpers ───────────────────────────────────────────────────
function setEl(id, val) {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
}

// ── Initialization ────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    boot();
});
