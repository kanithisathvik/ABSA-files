// Advanced ABSA Frontend JavaScript
// ============================================================================

// Global state
let currentResults = null;
let analysisHistory = [];
let batchData = [];
let chartInstances = {}; // Store chart instances for updates

// Configuration
const CONFIG = {
    AUTO_SAVE: true,
    ANIMATION_DURATION: 500,
    MAX_HISTORY: 50,
    CHART_THEME: 'dark'
};

// Utility Functions
function destroyChart(chartId) {
    if (chartInstances[chartId]) {
        chartInstances[chartId].destroy();
        delete chartInstances[chartId];
    }
}

function animateValue(element, start, end, duration = 1000) {
    const range = end - start;
    const increment = range / (duration / 16);
    let current = start;
    
    const timer = setInterval(() => {
        current += increment;
        if ((increment > 0 && current >= end) || (increment < 0 && current <= end)) {
            current = end;
            clearInterval(timer);
        }
        element.textContent = current.toFixed(2);
    }, 16);
}

function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    notification.style.cssText = `
        position: fixed; top: 20px; right: 20px; z-index: 10000;
        padding: 16px 24px; border-radius: 8px; color: white;
        background: ${type === 'success' ? '#00CC96' : type === 'error' ? '#EF553B' : '#636EFA'};
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
        animation: slideIn 0.3s ease-out;
    `;
    document.body.appendChild(notification);
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease-in';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// DOM Elements - Single Analysis
const productCategory = document.getElementById('productCategory');
const reviewInput = document.getElementById('reviewText');
const aspectsInput = document.getElementById('aspectsInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const clearBtn = document.getElementById('clearBtn');
const sampleBtn = document.getElementById('exampleBtn');
const loadingMsg = document.getElementById('loadingOverlay');
const resultsSection = document.getElementById('resultsSection');

// Add smooth scroll behavior
document.documentElement.style.scrollBehavior = 'smooth';

// Tab switching
document.querySelectorAll('.tab-button').forEach(btn => {
    btn.addEventListener('click', () => {
        const tab = btn.dataset.tab;
        
        // Update buttons
        document.querySelectorAll('.tab-button').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        
        // Update content
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        document.getElementById(tab).classList.add('active');
        
        // Load tab-specific content
        if (tab === 'insights') loadInsights();
    });
});

// Check server status on load
async function checkStatus() {
    try {
        const statusDot = document.querySelector('.status-dot');
        statusDot.classList.add('loading');
        
        const response = await fetch('/status');
        const data = await response.json();
        
        if (data.status === 'ready') {
            statusDot.classList.remove('loading');
            statusDot.classList.add('active');
            document.getElementById('statusText').textContent = 
                `Ready • ${data.categories_supported} categories • ${data.model}`;
            showNotification('Server ready!', 'success');
        }
    } catch (error) {
        const statusDot = document.querySelector('.status-dot');
        statusDot.classList.remove('loading');
        statusDot.classList.add('error');
        document.getElementById('statusText').textContent = 'Server offline';
        console.error('Status check failed:', error);
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    checkStatus();
    loadFromLocalStorage();
    
    // Add input animations
    const inputs = document.querySelectorAll('input, textarea, select');
    inputs.forEach(input => {
        input.addEventListener('focus', function() {
            this.parentElement.style.transform = 'scale(1.01)';
        });
        input.addEventListener('blur', function() {
            this.parentElement.style.transform = 'scale(1)';
        });
    });
    
    // Preload chart library for better performance
    if (window.Plotly) {
        console.log('Plotly loaded successfully');
    }
});

// Suggest aspects based on category
async function suggestAspects() {
    const category = productCategory.value;
    try {
        const response = await fetch('/categories');
        const data = await response.json();
        const categoryData = data.categories.find(c => c.id === category);
        
        if (categoryData && categoryData.common_aspects) {
            aspectsInput.value = categoryData.common_aspects.slice(0, 7).join(', ');
        }
    } catch (error) {
        console.error('Failed to load aspects:', error);
    }
}

// Show error message
function showError(message) {
    errorMsg.textContent = message;
    errorMsg.classList.remove('hidden');
    setTimeout(() => errorMsg.classList.add('hidden'), 8000);
}

// Clear form
function clearForm() {
    reviewInput.value = '';
    aspectsInput.value = '';
    resultsSection.style.display = 'none';
    showNotification('Form cleared', 'success');
}

// Load sample review
function loadSample() {
    const samples = [
        {
            category: 'smartphones',
            review: 'The battery life is absolutely amazing and lasts all day. Camera quality is very good in daylight but not great in low light. The screen is incredibly bright and crisp. Performance is extremely fast for gaming. However, the price is way too high for what you get.',
            aspects: 'battery, camera, screen, performance, price'
        },
        {
            category: 'laptops',
            review: 'Keyboard is comfortable for long typing sessions. Display quality is excellent with accurate colors. Battery life is decent but not exceptional. Build quality feels premium and solid. Performance is good for everyday tasks but struggles with heavy workloads. Price is reasonable for what you get.',
            aspects: 'keyboard, display, battery life, build quality, performance, price'
        },
        {
            category: 'headphones',
            review: 'Sound quality is phenomenal with deep bass and clear highs. Noise cancellation works incredibly well on flights. Comfort is excellent even after hours of use. Battery lasts forever, easily 30+ hours. However, they are quite expensive. Microphone quality is just okay for calls.',
            aspects: 'sound quality, noise cancellation, comfort, battery, price, microphone'
        }
    ];
    
    const sample = samples[Math.floor(Math.random() * samples.length)];
    productCategory.value = sample.category;
    reviewInput.value = sample.review;
    aspectsInput.value = sample.aspects;
}

// Main analysis function
async function analyzeReview() {
    const review = reviewInput.value.trim();
    const category = productCategory.value;
    const aspectsText = aspectsInput.value.trim();
    
    if (!review) {
        showNotification('Please enter a review to analyze', 'error');
        return;
    }
    
    // Parse aspects
    const aspects = aspectsText ? 
        aspectsText.split(',').map(a => a.trim()).filter(a => a) : 
        [];
    
    // Show loading overlay
    loadingMsg.style.display = 'flex';
    resultsSection.style.display = 'none';
    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '<span class="btn-icon">⏳</span> Analyzing...';
    
    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                review: review,
                category: category,
                aspects: aspects.length > 0 ? aspects : null
            })
        });
        
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }
        
        const data = await response.json();
        currentResults = data;
        
        // Hide loading, show results
        loadingMsg.style.display = 'none';
        resultsSection.style.display = 'block';
        
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<span class="btn-icon">🔍</span> Analyze Review';
        
        // Display results with animation
        displayResults(data);
        showNotification('Analysis completed successfully!', 'success');
        
        // Smooth scroll to results
        setTimeout(() => {
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }, 100);
        
        // Add to history
        analysisHistory.push({
            timestamp: new Date().toISOString(),
            review: review,
            category: category,
            results: data
        });
        
        // Limit history size
        if (analysisHistory.length > CONFIG.MAX_HISTORY) {
            analysisHistory.shift();
        }
        
        // Auto-save if enabled
        if (CONFIG.AUTO_SAVE) {
            saveToLocalStorage();
        }
        
    } catch (error) {
        loadingMsg.style.display = 'none';
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<span class="btn-icon">🔍</span> Analyze Review';
        showNotification(`Analysis failed: ${error.message}`, 'error');
        console.error('Analysis error:', error);
    }
}

// Local storage functions
function saveToLocalStorage() {
    try {
        localStorage.setItem('absa_history', JSON.stringify(analysisHistory));
        localStorage.setItem('absa_config', JSON.stringify(CONFIG));
    } catch (e) {
        console.warn('Failed to save to localStorage:', e);
    }
}

function loadFromLocalStorage() {
    try {
        const history = localStorage.getItem('absa_history');
        const config = localStorage.getItem('absa_config');
        
        if (history) {
            analysisHistory = JSON.parse(history);
        }
        if (config) {
            Object.assign(CONFIG, JSON.parse(config));
        }
    } catch (e) {
        console.warn('Failed to load from localStorage:', e);
    }
}

// Display results
function displayResults(results) {
    if (!results || results.length === 0) {
        showError('No results returned');
        return;
    }
    
    // Update summary cards
    updateSummaryCards(results);
    
    // Render visualizations
    renderCharts(results);
    
    // Render table
    renderTable(results);
    
    // Show results section
    resultsSection.classList.remove('hidden');
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Update summary cards
function updateSummaryCards(results) {
    const avgScore = (results.reduce((sum, r) => sum + (r['Score (1-10)'] || 0), 0) / results.length).toFixed(1);
    const positiveCount = results.filter(r => r.Sentiment === 'Positive').length;
    const negativeCount = results.filter(r => r.Sentiment === 'Negative').length;
    const avgConfidence = (results.reduce((sum, r) => sum + (r['Confidence (%)'] || 0), 0) / results.length).toFixed(1);
    
    document.getElementById('avgScore').textContent = avgScore;
    document.getElementById('positiveCount').textContent = positiveCount;
    document.getElementById('negativeCount').textContent = negativeCount;
    document.getElementById('avgConfidence').textContent = `${avgConfidence}%`;
}

// Render all charts
function renderCharts(results) {
    renderBarChart(results);
    renderRadarChart(results);
    renderPieChart(results);
    renderHeatmap(results);
}

// Bar chart
function renderBarChart(results) {
    const aspects = results.map(r => r.Aspect);
    const scores = results.map(r => r['Score (1-10)'] || 0);
    const sentiments = results.map(r => r.Sentiment);
    
    const colors = sentiments.map(s => {
        if (s === 'Positive') return '#10B981';
        if (s === 'Negative') return '#EF4444';
        return '#8B5CF6';
    });
    
    const data = [{
        x: aspects,
        y: scores,
        type: 'bar',
        marker: { 
            color: colors,
            line: {
                color: 'rgba(255, 255, 255, 0.2)',
                width: 2
            }
        },
        text: scores.map(s => s.toFixed(1)),
        textposition: 'outside',
        textfont: { color: '#E8EAED', size: 12, weight: 'bold' },
        hovertemplate: '<b>%{x}</b><br>Score: %{y:.2f}<extra></extra>'
    }];
    
    const layout = {
        title: {
            text: 'Sentiment Scores by Aspect',
            font: { color: '#E8EAED', size: 18, weight: 'bold' }
        },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#E8EAED', family: 'Inter, sans-serif' },
        xaxis: { 
            gridcolor: '#2D3139',
            title: 'Aspects',
            showgrid: false,
            color: '#9CA3AF'
        },
        yaxis: { 
            gridcolor: '#2D3139',
            range: [0, 10],
            title: 'Score',
            color: '#9CA3AF'
        },
        margin: { t: 50, b: 80, l: 60, r: 20 },
        transition: {
            duration: CONFIG.ANIMATION_DURATION,
            easing: 'cubic-in-out'
        }
    };
    
    const config = {
        responsive: true,
        displayModeBar: false,
        displaylogo: false
    };
    
    Plotly.newPlot('barChart', data, layout, config);
}

// Radar chart
function renderRadarChart(results) {
    const aspects = results.map(r => r.Aspect);
    const scores = results.map(r => r['Score (1-10)'] || 0);
    
    const data = [{
        r: scores,
        theta: aspects,
        fill: 'toself',
        type: 'scatterpolar',
        line: { 
            color: '#3B82F6',
            width: 3
        },
        fillcolor: 'rgba(59, 130, 246, 0.3)',
        marker: {
            color: '#3B82F6',
            size: 10,
            line: {
                color: '#E8EAED',
                width: 2
            }
        },
        name: 'Sentiment Score',
        hovertemplate: '<b>%{theta}</b><br>Score: %{r:.2f}<extra></extra>'
    }];
    
    const layout = {
        title: {
            text: 'Overall Sentiment Profile',
            font: { color: '#E8EAED', size: 18, weight: 'bold' }
        },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#E8EAED', family: 'Inter, sans-serif' },
        polar: {
            bgcolor: 'rgba(0,0,0,0)',
            radialaxis: {
                visible: true,
                range: [0, 10],
                gridcolor: '#2D3139',
                tickfont: { color: '#9CA3AF' }
            },
            angularaxis: {
                gridcolor: '#2D3139',
                tickfont: { color: '#E8EAED', size: 11, weight: '600' }
            }
        },
        margin: { t: 50, b: 40, l: 60, r: 60 },
        transition: {
            duration: CONFIG.ANIMATION_DURATION,
            easing: 'cubic-in-out'
        }
    };
    
    const config = {
        responsive: true,
        displayModeBar: false,
        displaylogo: false
    };
    
    Plotly.newPlot('radarChart', data, layout, config);
}

// Pie chart
function renderPieChart(results) {
    const sentimentCounts = {};
    results.forEach(r => {
        const sentiment = r.Sentiment;
        sentimentCounts[sentiment] = (sentimentCounts[sentiment] || 0) + 1;
    });
    
    const data = [{
        values: Object.values(sentimentCounts),
        labels: Object.keys(sentimentCounts),
        type: 'pie',
        hole: 0.4,
        marker: {
            colors: Object.keys(sentimentCounts).map(s => {
                if (s === 'Positive') return '#10B981';
                if (s === 'Negative') return '#EF4444';
                return '#8B5CF6';
            }),
            line: {
                color: '#1C1F26',
                width: 3
            }
        },
        textinfo: 'label+percent',
        textfont: {
            size: 14,
            color: '#E8EAED',
            weight: 'bold'
        },
        hovertemplate: '<b>%{label}</b><br>Count: %{value}<extra></extra>'
    }];
    
    const layout = {
        title: {
            text: 'Sentiment Distribution',
            font: { color: '#E8EAED', size: 18, weight: 'bold' }
        },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#E8EAED', family: 'Inter, sans-serif' },
        margin: { t: 50, b: 40, l: 40, r: 40 },
        transition: {
            duration: CONFIG.ANIMATION_DURATION,
            easing: 'cubic-in-out'
        },
        showlegend: true,
        legend: {
            font: { color: '#E8EAED' }
        }
    };
    
    const config = {
        responsive: true,
        displayModeBar: false,
        displaylogo: false
    };
    
    Plotly.newPlot('pieChart', data, layout, config);
}
// Heatmap showing probabilities
function renderHeatmap(results) {
    const aspects = results.map(r => r.Aspect);
    const sentiments = ['Positive', 'Neutral', 'Negative'];
    
    const zData = sentiments.map(sentiment => 
        results.map(r => (r.Probabilities && r.Probabilities[sentiment]) || 0)
    );
    
    const data = [{
        z: zData,
        x: aspects,
        y: sentiments,
        type: 'heatmap',
        colorscale: [
            [0, '#1C1F26'],
            [0.3, '#8B5CF6'],
            [0.6, '#3B82F6'],
            [1, '#10B981']
        ],
        colorbar: {
            thickness: 15,
            len: 0.7,
            tickfont: { color: '#E8EAED', size: 10 },
            title: {
                text: 'Probability',
                side: 'right',
                font: { color: '#E8EAED', size: 12 }
            }
        },
        hovertemplate: '<b>%{y}</b><br>%{x}<br>Probability: %{z:.3f}<extra></extra>'
    }];
    
    const layout = {
        title: {
            text: 'Sentiment Probabilities Heatmap',
            font: { color: '#E8EAED', size: 18, weight: 'bold' }
        },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#E8EAED', family: 'Inter, sans-serif' },
        xaxis: { 
            title: 'Aspects',
            gridcolor: '#2D3139',
            tickfont: { color: '#9CA3AF' }
        },
        yaxis: { 
            title: 'Sentiment',
            gridcolor: '#2D3139',
            tickfont: { color: '#E8EAED' }
        },
        margin: { t: 50, b: 80, l: 80, r: 20 },
        transition: {
            duration: CONFIG.ANIMATION_DURATION,
            easing: 'cubic-in-out'
        }
    };
    
    const config = {
        responsive: true,
        displayModeBar: false,
        displaylogo: false
    };
    
    Plotly.newPlot('heatmapChart', data, layout, config);
}


// Render results table
function renderTable(results) {
    const tbody = document.getElementById('resultsTableBody');
    tbody.innerHTML = results.map(r => {
        const sentiment = r.Sentiment || 'Unknown';
        const score = (r['Score (1-10)'] || 0).toFixed(2);
        const confidence = (r['Confidence (%)'] || 0).toFixed(1);
        const probs = r.Probabilities || { Positive: 0, Neutral: 0, Negative: 0 };
        const details = r.Details || {};
        
        const sentimentClass = sentiment.toLowerCase();
        const sentimentSymbol = sentiment === 'Positive' ? '✓' : sentiment === 'Negative' ? '✗' : '○';
        
        return `
            <tr>
                <td><strong>${r.Aspect}</strong></td>
                <td class="sentiment-${sentimentClass}">${sentimentSymbol} ${sentiment}</td>
                <td><span class="score-badge">${score}</span></td>
                <td>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${confidence}%"></div>
                        <span>${confidence}%</span>
                    </div>
                </td>
                <td>
                    <div class="probs-mini">
                        <div class="prob-item">
                            <span class="prob-label">Pos:</span>
                            <span class="prob-value">${probs.Positive.toFixed(3)}</span>
                        </div>
                        <div class="prob-item">
                            <span class="prob-label">Neu:</span>
                            <span class="prob-value">${probs.Neutral.toFixed(3)}</span>
                        </div>
                        <div class="prob-item">
                            <span class="prob-label">Neg:</span>
                            <span class="prob-value">${probs.Negative.toFixed(3)}</span>
                        </div>
                    </div>
                </td>
                <td>
                    <button class="btn-mini" onclick="showDetailsModal(${JSON.stringify(details).replace(/"/g, '&quot;')})">
                        View Details
                    </button>
                </td>
            </tr>
        `;
    }).join('');
}

// Show details modal
function showDetailsModal(details) {
    const detailsHtml = Object.entries(details).map(([key, value]) => 
        `<div class="detail-row"><strong>${key}:</strong> ${JSON.stringify(value)}</div>`
    ).join('');
    
    alert(`Analysis Details:\n\n${JSON.stringify(details, null, 2)}`);
}

// Export functions
function exportJson() {
    if (!currentResults) return;
    
    const dataStr = JSON.stringify(currentResults, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    downloadBlob(dataBlob, 'absa_results.json');
}

function exportCsv() {
    if (!currentResults) return;
    
    const csv = Papa.unparse(currentResults.map(r => ({
        Aspect: r.Aspect,
        Sentiment: r.Sentiment,
        Score: r['Score (1-10)'],
        Confidence: r['Confidence (%)'],
        Positive_Prob: r.Probabilities?.Positive,
        Neutral_Prob: r.Probabilities?.Neutral,
        Negative_Prob: r.Probabilities?.Negative
    })));
    
    const dataBlob = new Blob([csv], { type: 'text/csv' });
    downloadBlob(dataBlob, 'absa_results.csv');
}

function copyResults() {
    if (!currentResults) return;
    
    const text = currentResults.map(r => 
        `${r.Aspect}: ${r.Sentiment} (${r['Score (1-10)']}/10) - ${r['Confidence (%)']}% confidence`
    ).join('\n');
    
    navigator.clipboard.writeText(text).then(() => {
        alert('Results copied to clipboard!');
    });
}

function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Load insights
async function loadInsights() {
    try {
        const response = await fetch('/status');
        const data = await response.json();
        
        document.getElementById('modelInfo').innerHTML = `
            <p><strong>Model:</strong> ${data.model_name || 'Not loaded'}</p>
            <p><strong>Status:</strong> ${data.model_loaded ? '✅ Loaded' : '❌ Not loaded'}</p>
            <p><strong>Categories:</strong> ${data.product_categories?.length || 0}</p>
        `;
        
        document.getElementById('configInfo').innerHTML = `
            <p><strong>Product Categories:</strong> 14 specialized configs</p>
            <p><strong>Features:</strong> Negation detection, intensity modifiers, fuzzy matching</p>
            <p><strong>Fallback Models:</strong> 4 models with automatic fallback</p>
        `;
        
        document.getElementById('statsInfo').innerHTML = `
            <p><strong>Total Analyses:</strong> ${analysisHistory.length}</p>
            <p><strong>Session Start:</strong> ${new Date().toLocaleString()}</p>
            <p><strong>Average Analysis Time:</strong> ~2-3 seconds</p>
        `;
        
    } catch (error) {
        console.error('Failed to load insights:', error);
    }
}

// Event Listeners
analyzeBtn.addEventListener('click', analyzeReview);
clearBtn.addEventListener('click', clearForm);
sampleBtn.addEventListener('click', loadSample);
document.getElementById('exportJsonBtn')?.addEventListener('click', exportJson);
document.getElementById('exportCsvBtn')?.addEventListener('click', exportCsv);
document.getElementById('copyBtn')?.addEventListener('click', copyResults);

// Allow Enter key in inputs
aspectsInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') analyzeReview();
});

// Load history from localStorage
const savedHistory = localStorage.getItem('analysisHistory');
if (savedHistory) {
    try {
        analysisHistory = JSON.parse(savedHistory);
    } catch (e) {
        console.error('Failed to load history:', e);
    }
}

// Tab switching functionality
document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
        const tabName = tab.getAttribute('data-tab');
        
        // Remove active class from all tabs and panes
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
        
        // Add active class to clicked tab and corresponding pane
        tab.classList.add('active');
        document.getElementById(`${tabName}-tab`).classList.add('active');
    });
});

// Initialize
checkStatus();
setInterval(checkStatus, 30000); // Check status every 30 seconds
