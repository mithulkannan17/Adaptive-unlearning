/**
 * ASUC-SOM Interactive Visualizer Frontend Application
 * Handles Chart.js visualization, real-time Subspace Canvas animation,
 * interactive request profiling, and benchmark comparative analytics.
 */

// Global state
let benchmarkData = null;
let classData = null;
let activeMethod = 'asuc';
let charts = {};
let canvasAnimationId = null;

// DOM Ready
document.addEventListener('DOMContentLoaded', async () => {
    setupNavigation();
    setupSimulatorControls();
    await loadInitialData();
    initSubspaceCanvas();
    renderOverviewKPIs();
    renderAllCharts();
    renderRoundsTable();
});

// Setup sidebar navigation
function setupNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            navItems.forEach(i => i.classList.remove('active'));
            item.classList.add('active');

            const tabId = item.getAttribute('data-tab');
            document.querySelectorAll('.tab-view').forEach(view => {
                view.classList.remove('active');
            });
            const targetView = document.getElementById(tabId);
            if (targetView) targetView.classList.add('active');

            // Trigger chart resize on view change
            Object.values(charts).forEach(c => c && c.resize && c.resize());
        });
    });
}

// Fetch benchmark and class data
async function loadInitialData() {
    try {
        const benchRes = await fetch('/api/benchmarks');
        if (benchRes.ok) {
            benchmarkData = await benchRes.json();
        }
        const classRes = await fetch('/api/classes');
        if (classRes.ok) {
            classData = await classRes.json();
        }
    } catch (e) {
        console.warn('Backend API offline, falling back to bundled dataset:', e);
    }
}

// Setup Interactive Simulator controls
function setupSimulatorControls() {
    const cardSlider = document.getElementById('cardinalitySlider');
    const cardVal = document.getElementById('cardinalityVal');
    const entSlider = document.getElementById('entropySlider');
    const entVal = document.getElementById('entropyVal');
    const somSlider = document.getElementById('somSlider');
    const somVal = document.getElementById('somVal');

    function updateSimulation() {
        const cardinality = parseInt(cardSlider.value);
        const entropy = parseFloat(entSlider.value);
        const somScore = parseFloat(somSlider.value);

        cardVal.innerText = `${cardinality.toLocaleString()} samples`;
        entVal.innerText = entropy.toFixed(2);
        somVal.innerText = somScore.toFixed(2);

        simulateDecision(cardinality, entropy, somScore);
    }

    if (cardSlider && entSlider && somSlider) {
        cardSlider.addEventListener('input', updateSimulation);
        entSlider.addEventListener('input', updateSimulation);
        somSlider.addEventListener('input', updateSimulation);
        updateSimulation();
    }
}

// Client-side & API-synced dynamic decision simulator
async function simulateDecision(cardinality, entropy, somScore) {
    let result = null;
    try {
        const res = await fetch('/api/simulate_decision', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ cardinality, entropy, som_score: somScore })
        });
        if (res.ok) {
            result = await res.json();
        }
    } catch (e) {}

    // Fallback simulation logic if offline
    if (!result) {
        const totalRemaining = 49500;
        const ratio = cardinality / totalRemaining;
        const isConcentrated = entropy < 0.40;
        let healthState = 'GREEN';
        let healthColor = '#10b981';

        if (somScore >= 0.80) {
            healthState = 'RED';
            healthColor = '#f43f5e';
        } else if (somScore >= 0.40) {
            healthState = 'AMBER';
            healthColor = '#f59e0b';
        }

        let chosen = 'SSD';
        let tier = 'Tier 1 (Rapid Dampening)';
        let reason = `Healthy subspace (SOM = ${somScore.toFixed(2)}, ${healthState}) and balanced entropy (${entropy.toFixed(2)}). Routing to Tier 1 SSD.`;

        if (healthState === 'RED') {
            chosen = 'SalUn';
            tier = 'Tier 2 (Regularized Retain)';
            reason = 'Critical SOM subspace overlap (RED State >= 0.80). Routing to SalUn with retain loss penalty to prevent capacity collapse.';
        } else if (isConcentrated) {
            chosen = 'SalUn';
            tier = 'Tier 2 (Targeted Saliency)';
            reason = `Request is class-concentrated (Entropy H(c) = ${entropy.toFixed(2)} < 0.40). SSD would induce class-specific collapse; routing to SalUn.`;
        } else if (ratio > 0.05) {
            chosen = 'SalUn';
            tier = 'Tier 2 (Large Request)';
            reason = `Large deletion ratio (${(ratio * 100).toFixed(1)}% > 5%). Routing to SalUn.`;
        }

        result = {
            health_state: healthState,
            health_color: healthColor,
            cardinality,
            cardinality_ratio: ratio,
            class_entropy: entropy,
            is_class_concentrated: isConcentrated,
            som_score: somScore,
            chosen_primitive: chosen,
            tier,
            routing_reason: reason,
            projected_verification: somScore < 0.85 ? 'PASS' : 'ESCALATE',
            estimated_speedup: chosen === 'SSD' ? '150x' : '115x'
        };
    }

    // Render decision UI
    const badge = document.getElementById('simDecisionBadge');
    const tierElem = document.getElementById('simTier');
    const reasonElem = document.getElementById('simReason');
    const healthBadge = document.getElementById('simHealthBadge');
    const speedupElem = document.getElementById('simSpeedup');
    const verifElem = document.getElementById('simVerification');

    if (badge) {
        badge.className = `decision-badge-large badge-${result.chosen_primitive.toLowerCase()}`;
        badge.innerHTML = `<span>▶ Primitive:</span> ${result.chosen_primitive}`;
    }
    if (tierElem) tierElem.innerText = result.tier;
    if (reasonElem) reasonElem.innerText = result.routing_reason;
    if (healthBadge) {
        healthBadge.innerText = `${result.health_state} (${result.som_score.toFixed(2)})`;
        healthBadge.style.color = result.health_color;
    }
    if (speedupElem) speedupElem.innerText = result.estimated_speedup;
    if (verifElem) verifElem.innerText = result.projected_verification;
}

// Render Overview KPI Stat Cards
function renderOverviewKPIs() {
    if (!benchmarkData || !benchmarkData.asuc) return;
    const asucRounds = benchmarkData.asuc.rounds;
    const latest = asucRounds[asucRounds.length - 1];

    document.getElementById('kpiRetainAcc').innerText = `${latest.retain_acc.toFixed(1)}%`;
    document.getElementById('kpiTestAcc').innerText = `${latest.test_acc.toFixed(1)}%`;
    document.getElementById('kpiForgetAcc').innerText = `${latest.forget_acc.toFixed(1)}%`;
    document.getElementById('kpiSpeedup').innerText = `${benchmarkData.asuc.speedup_vs_retrain.toFixed(0)}x`;
}

// Chart.js initialization & rendering
function renderAllCharts() {
    if (!benchmarkData) return;

    renderRetentionComparisonChart();
    renderSOMTrajectoryChart();
    renderParameterDriftChart();
    renderClassRadarChart();
    renderSpeedupChart();
}

// 1. Retention & Test Accuracy Comparison
function renderRetentionComparisonChart() {
    const ctx = document.getElementById('chartRetentionComparison');
    if (!ctx) return;

    const rounds = Array.from({ length: 21 }, (_, i) => `Round ${i}`);
    const datasets = Object.keys(benchmarkData).map(key => {
        const item = benchmarkData[key];
        return {
            label: item.name.split(' (')[0],
            data: item.rounds.map(r => r.retain_acc),
            borderColor: item.border_color,
            backgroundColor: item.color + '22',
            borderWidth: key === 'asuc' ? 3 : 2,
            borderDash: key === 'retrain' ? [5, 5] : [],
            pointRadius: key === 'asuc' ? 4 : 2,
            pointHoverRadius: 6,
            tension: 0.25
        };
    });

    if (charts.retention) charts.retention.destroy();
    charts.retention = new Chart(ctx, {
        type: 'line',
        data: { labels: rounds, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: { position: 'top', labels: { color: '#94a3b8', font: { family: 'Inter', size: 11 } } },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.95)',
                    titleColor: '#fff',
                    bodyColor: '#cbd5e1',
                    borderColor: 'rgba(255, 255, 255, 0.1)',
                    borderWidth: 1,
                    padding: 10
                }
            },
            scales: {
                x: { grid: { color: 'rgba(255, 255, 255, 0.04)' }, ticks: { color: '#64748b' } },
                y: {
                    min: 50,
                    max: 100,
                    grid: { color: 'rgba(255, 255, 255, 0.04)' },
                    ticks: { color: '#64748b', callback: v => `${v}%` }
                }
            }
        }
    });
}

// 2. SOM Metric Trajectory & Health Thresholds
function renderSOMTrajectoryChart() {
    const ctx = document.getElementById('chartSOMTrajectory');
    if (!ctx) return;

    const rounds = Array.from({ length: 21 }, (_, i) => `R${i}`);
    const datasets = [
        {
            label: 'ASUC-SOM (Adaptive)',
            data: benchmarkData.asuc.rounds.map(r => r.som_score),
            borderColor: '#06b6d4',
            backgroundColor: 'rgba(6, 182, 212, 0.15)',
            borderWidth: 3,
            fill: true,
            tension: 0.3
        },
        {
            label: 'SSD (Static Collapse Drift)',
            data: benchmarkData.ssd.rounds.map(r => r.som_score),
            borderColor: '#f43f5e',
            borderWidth: 2,
            borderDash: [4, 4],
            tension: 0.3
        }
    ];

    if (charts.som) charts.som.destroy();
    charts.som = new Chart(ctx, {
        type: 'line',
        data: { labels: rounds, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'top', labels: { color: '#94a3b8', font: { family: 'Inter', size: 11 } } }
            },
            scales: {
                x: { grid: { color: 'rgba(255, 255, 255, 0.04)' }, ticks: { color: '#64748b' } },
                y: {
                    min: 0,
                    max: 1.05,
                    grid: { color: 'rgba(255, 255, 255, 0.04)' },
                    ticks: { color: '#64748b' }
                }
            }
        }
    });
}

// 3. Parameter Drift vs Rounds
function renderParameterDriftChart() {
    const ctx = document.getElementById('chartParamDrift');
    if (!ctx) return;

    const rounds = Array.from({ length: 21 }, (_, i) => `R${i}`);
    const datasets = Object.keys(benchmarkData).map(key => {
        const item = benchmarkData[key];
        return {
            label: item.name.split(' (')[0],
            data: item.rounds.map(r => r.param_drift),
            borderColor: item.border_color,
            borderWidth: key === 'asuc' ? 3 : 2,
            tension: 0.25
        };
    });

    if (charts.drift) charts.drift.destroy();
    charts.drift = new Chart(ctx, {
        type: 'line',
        data: { labels: rounds, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'top', labels: { color: '#94a3b8', font: { family: 'Inter', size: 11 } } }
            },
            scales: {
                x: { grid: { color: 'rgba(255, 255, 255, 0.04)' }, ticks: { color: '#64748b' } },
                y: { grid: { color: 'rgba(255, 255, 255, 0.04)' }, ticks: { color: '#64748b' } }
            }
        }
    });
}

// 4. Radar Chart of Class-Wise Retention
function renderClassRadarChart() {
    const ctx = document.getElementById('chartClassRadar');
    if (!ctx || !classData) return;

    if (charts.radar) charts.radar.destroy();
    charts.radar = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: classData.class_names.map(c => c.toUpperCase()),
            datasets: [
                {
                    label: 'ASUC-SOM (Adaptive)',
                    data: classData.class_retention.asuc,
                    borderColor: '#06b6d4',
                    backgroundColor: 'rgba(6, 182, 212, 0.2)',
                    borderWidth: 2
                },
                {
                    label: 'SSD (Static)',
                    data: classData.class_retention.ssd,
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    borderWidth: 1.5
                },
                {
                    label: 'Exact Retraining',
                    data: classData.class_retention.retrain,
                    borderColor: '#38bdf8',
                    backgroundColor: 'rgba(56, 189, 248, 0.1)',
                    borderDash: [3, 3],
                    borderWidth: 1.5
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    angleLines: { color: 'rgba(255, 255, 255, 0.08)' },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    pointLabels: { color: '#94a3b8', font: { family: 'Inter', size: 10 } },
                    min: 50,
                    max: 100,
                    ticks: { color: '#64748b', backdropColor: 'transparent' }
                }
            },
            plugins: {
                legend: { position: 'top', labels: { color: '#94a3b8', font: { family: 'Inter', size: 11 } } }
            }
        }
    });
}

// 5. Speedup Comparison
function renderSpeedupChart() {
    const ctx = document.getElementById('chartSpeedup');
    if (!ctx || !benchmarkData) return;

    const labels = Object.keys(benchmarkData).map(k => benchmarkData[k].name.split(' (')[0]);
    const speeds = Object.keys(benchmarkData).map(k => benchmarkData[k].speedup_vs_retrain);
    const colors = Object.keys(benchmarkData).map(k => benchmarkData[k].color);

    if (charts.speedup) charts.speedup.destroy();
    charts.speedup = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Speedup Factor vs Retrain',
                data: speeds,
                backgroundColor: colors,
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { grid: { display: false }, ticks: { color: '#94a3b8' } },
                y: {
                    type: 'logarithmic',
                    grid: { color: 'rgba(255, 255, 255, 0.04)' },
                    ticks: { color: '#64748b', callback: v => `${v}x` }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}

// Render Round Breakdown Table
function renderRoundsTable() {
    const tbody = document.getElementById('roundsTableBody');
    if (!tbody || !benchmarkData || !benchmarkData.asuc) return;

    tbody.innerHTML = '';
    benchmarkData.asuc.rounds.forEach(r => {
        const tr = document.createElement('tr');
        const badgeClass = r.health === 'GREEN' ? 'badge-green' : (r.health === 'AMBER' ? 'badge-amber' : 'badge-red');
        tr.innerHTML = `
            <td>#${r.round}</td>
            <td><span class="badge-tag ${badgeClass}">${r.health}</span></td>
            <td>${r.som_score.toFixed(3)}</td>
            <td><strong style="color: var(--accent-cyan);">${r.primitive}</strong></td>
            <td>${r.retain_acc.toFixed(1)}%</td>
            <td>${r.test_acc.toFixed(1)}%</td>
            <td>${r.forget_acc.toFixed(1)}%</td>
            <td>${(r.param_drift * 100).toFixed(2)}%</td>
            <td>${r.selected_params.toLocaleString()}</td>
            <td>${r.mia_auc.toFixed(3)}</td>
        `;
        tbody.appendChild(tr);
    });
}

// Interactive Animated 2D/3D Subspace Canvas Simulator
function initSubspaceCanvas() {
    const canvas = document.getElementById('subspaceCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    function resizeCanvas() {
        canvas.width = canvas.parentElement.clientWidth;
        canvas.height = canvas.parentElement.clientHeight;
    }
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();

    let angle = 0;

    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const cx = canvas.width / 2;
        const cy = canvas.height / 2;

        // Draw Subspace Grid Plane
        ctx.strokeStyle = 'rgba(6, 182, 212, 0.12)';
        ctx.lineWidth = 1;
        for (let i = -150; i <= 150; i += 30) {
            ctx.beginPath();
            ctx.moveTo(cx + i, cy - 100);
            ctx.lineTo(cx + i, cy + 100);
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(cx - 150, cy + i);
            ctx.lineTo(cx + 150, cy + i);
            ctx.stroke();
        }

        // Draw Historical Orthonormal Basis Q vectors
        const qVectors = [
            { angle: 0.3, len: 120, label: 'q1 (SOM Basis 1)', color: '#38bdf8' },
            { angle: 1.8, len: 100, label: 'q2 (SOM Basis 2)', color: '#818cf8' },
            { angle: 3.4, len: 90, label: 'q3 (SOM Basis 3)', color: '#c084fc' }
        ];

        qVectors.forEach(v => {
            const vx = cx + Math.cos(v.angle) * v.len;
            const vy = cy + Math.sin(v.angle) * v.len * 0.5;

            ctx.strokeStyle = v.color;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(cx, cy);
            ctx.lineTo(vx, vy);
            ctx.stroke();

            // Arrowhead
            ctx.fillStyle = v.color;
            ctx.beginPath();
            ctx.arc(vx, vy, 4, 0, Math.PI * 2);
            ctx.fill();

            ctx.fillStyle = '#94a3b8';
            ctx.font = '10px JetBrains Mono';
            ctx.fillText(v.label, vx + 6, vy);
        });

        // Current request gradient vector g_t (oscillating/rotating)
        angle += 0.015;
        const gx = cx + Math.cos(angle) * 140;
        const gy = cy + Math.sin(angle) * 80;

        // Projection on subspace plane
        const projX = cx + Math.cos(angle) * 140;
        const projY = cy + Math.sin(angle) * 40;

        // Projection dashed line
        ctx.strokeStyle = 'rgba(244, 63, 94, 0.4)';
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(gx, gy);
        ctx.lineTo(projX, projY);
        ctx.stroke();
        ctx.setLineDash([]);

        // Projection vector
        ctx.strokeStyle = 'rgba(16, 185, 129, 0.8)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(projX, projY);
        ctx.stroke();

        // Incoming gradient g_t
        ctx.strokeStyle = '#f43f5e';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(gx, gy);
        ctx.stroke();

        ctx.fillStyle = '#f43f5e';
        ctx.beginPath();
        ctx.arc(gx, gy, 5, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = '#f43f5e';
        ctx.font = 'bold 11px JetBrains Mono';
        ctx.fillText('g_t (Request Gradient)', gx + 8, gy);

        canvasAnimationId = requestAnimationFrame(draw);
    }

    draw();
}
