const API_BASE = 'http://localhost:8000/api';

// State
let currentPage = 1;
let currentLimit = 20;
let currentQuestionId = null;
let currentSearch = '';

// Elements
const questionList = document.getElementById('questionList');
const searchInput = document.getElementById('searchInput');
const prevPageBtn = document.getElementById('prevPage');
const nextPageBtn = document.getElementById('nextPage');
const pageInfo = document.getElementById('pageInfo');

const emptyState = document.getElementById('emptyState');
const detailView = document.getElementById('detailView');

// Detail Elements
const questionText = document.getElementById('questionText');
const metaTags = document.getElementById('metaTags');
const choicesList = document.getElementById('choicesList');
const contextBox = document.getElementById('contextBox');
const contextText = document.getElementById('contextText');
const hintBox = document.getElementById('hintBox');
const hintText = document.getElementById('hintText');

// Inference Elements
const runInferenceBtn = document.getElementById('runInferenceBtn');
const inferenceResult = document.getElementById('inferenceResult');
const inferenceLoading = document.getElementById('inferenceLoading');
const inferenceError = document.getElementById('inferenceError');
const answerPosterior = document.getElementById('answerPosterior');
const latentPosteriors = document.getElementById('latentPosteriors');
const rawJson = document.getElementById('rawJson');

// Init
document.addEventListener('DOMContentLoaded', () => {
    fetchQuestions();

    // Event Listeners
    searchInput.addEventListener('input', debounce((e) => {
        currentSearch = e.target.value;
        currentPage = 1;
        fetchQuestions();
    }, 300));

    prevPageBtn.addEventListener('click', () => {
        if (currentPage > 1) {
            currentPage--;
            fetchQuestions();
        }
    });

    nextPageBtn.addEventListener('click', () => {
        currentPage++;
        fetchQuestions();
    });

    runInferenceBtn.addEventListener('click', runInference);
});

// Fetch Questions
async function fetchQuestions() {
    questionList.innerHTML = '<div class="loading">Loading...</div>';

    try {
        let url = `${API_BASE}/questions?page=${currentPage}&limit=${currentLimit}`;
        if (currentSearch) {
            url += `&search=${encodeURIComponent(currentSearch)}`;
        }

        const res = await fetch(url);
        const data = await res.json();

        renderQuestionList(data.items);
        updatePagination(data);
    } catch (err) {
        questionList.innerHTML = '<div class="error">Failed to load questions</div>';
        console.error(err);
    }
}

function renderQuestionList(items) {
    questionList.innerHTML = '';

    if (items.length === 0) {
        questionList.innerHTML = '<div class="loading">No questions found</div>';
        return;
    }

    items.forEach(item => {
        const div = document.createElement('div');
        div.className = `question-item ${currentQuestionId === item.id ? 'active' : ''}`;
        div.onclick = () => selectQuestion(item.id);

        div.innerHTML = `
            <div class="q-id">${item.id}</div>
            <div class="q-text">${item.question}</div>
        `;
        questionList.appendChild(div);
    });
}

function updatePagination(data) {
    pageInfo.textContent = `Page ${data.page}`;
    prevPageBtn.disabled = data.page === 1;
    // Simple check: if we got less than limit, we are at end
    nextPageBtn.disabled = data.items.length < currentLimit;
}

// Select Question
async function selectQuestion(id) {
    currentQuestionId = id;

    // Update active state in list
    document.querySelectorAll('.question-item').forEach(el => {
        el.classList.remove('active');
        if (el.querySelector('.q-id').textContent === String(id)) {
            el.classList.add('active');
        }
    });

    // Show loading detail?
    emptyState.classList.add('hidden');
    detailView.classList.remove('hidden');

    // Reset Inference UI
    inferenceResult.classList.add('hidden');
    inferenceError.classList.add('hidden');
    inferenceLoading.classList.add('hidden');
    runInferenceBtn.disabled = false;

    try {
        const res = await fetch(`${API_BASE}/questions/${id}`);
        const data = await res.json();
        renderDetail(data);
    } catch (err) {
        console.error(err);
        alert("Failed to load question details");
    }
}

function renderDetail(data) {
    const ex = data.raw_example;

    // Meta
    metaTags.innerHTML = `
        <span class="tag">${ex.subject}</span>
        <span class="tag">${ex.topic}</span>
        <span class="tag">${ex.grade}</span>
    `;

    questionText.textContent = ex.question;

    // Choices
    choicesList.innerHTML = '';
    ex.choices.forEach((choice, idx) => {
        const div = document.createElement('div');
        div.className = 'choice-item';
        // If dataset has answer index, we could highlight it, but let's keep it hidden 
        // until inference? Or show it? Usually ScienceQA has 'answer' index (0-based).
        // Let's show the correct answer for reference if available.
        if (ex.answer === idx) {
            div.classList.add('correct');
        }
        div.textContent = `${idx}. ${choice}`;
        choicesList.appendChild(div);
    });

    // Context/Hint
    if (ex.hint) {
        hintBox.classList.remove('hidden');
        hintText.textContent = ex.hint;
    } else {
        hintBox.classList.add('hidden');
    }

    if (ex.context) {
        contextBox.classList.remove('hidden');
        contextText.textContent = ex.context;
    } else {
        contextBox.classList.add('hidden');
    }
}

// Run Inference
async function runInference() {
    if (!currentQuestionId) return;

    runInferenceBtn.disabled = true;
    inferenceLoading.classList.remove('hidden');
    inferenceResult.classList.add('hidden');
    inferenceError.classList.add('hidden');

    try {
        const res = await fetch(`${API_BASE}/infer/${currentQuestionId}`, {
            method: 'POST'
        });

        if (!res.ok) {
            const errData = await res.json();
            throw new Error(errData.detail || 'Inference failed');
        }

        const result = await res.json();
        renderInferenceResult(result);
    } catch (err) {
        inferenceError.textContent = err.message;
        inferenceError.classList.remove('hidden');
    } finally {
        runInferenceBtn.disabled = false;
        inferenceLoading.classList.add('hidden');
    }
}

function renderInferenceResult(result) {
    inferenceResult.classList.remove('hidden');

    // Raw JSON
    rawJson.textContent = JSON.stringify(result, null, 2);

    // Answer Posterior
    const ansProbs = result.answer_posterior.option_probabilities;
    const selected = result.answer_posterior.selected_answer;

    answerPosterior.innerHTML = '';
    // Sort by key or value? Usually keys are options.
    // Let's iterate keys.
    for (const [opt, prob] of Object.entries(ansProbs)) {
        const pct = (prob * 100).toFixed(1);
        const isSelected = opt === selected;

        const row = document.createElement('div');
        row.className = 'bar-container';
        row.innerHTML = `
            <div class="bar-label">${opt.substring(0, 10)}</div>
            <div class="bar-track">
                <div class="bar-fill" style="width: ${pct}%; background-color: ${isSelected ? 'var(--success)' : 'var(--accent)'}"></div>
                <div class="bar-value">${pct}%</div>
            </div>
        `;
        answerPosterior.appendChild(row);
    }

    // Latent Posteriors
    latentPosteriors.innerHTML = '';
    const latents = result.latent_posteriors;
    for (const [name, data] of Object.entries(latents)) {
        const div = document.createElement('div');
        div.className = 'latent-item';

        // Build bars for states
        let barsHtml = '';
        for (const [state, prob] of Object.entries(data.state_probabilities)) {
            const pct = (prob * 100).toFixed(1);
            barsHtml += `
                <div class="bar-container" style="margin-top: 4px;">
                    <div class="bar-label" style="width: 80px; font-weight: normal; font-size: 0.8rem;">${state}</div>
                    <div class="bar-track" style="height: 16px;">
                        <div class="bar-fill" style="width: ${pct}%;"></div>
                        <div class="bar-value" style="font-size: 0.7rem;">${pct}%</div>
                    </div>
                </div>
            `;
        }

        div.innerHTML = `
            <div class="latent-name">${name}</div>
            ${barsHtml}
            <div class="latent-justification">${data.justification}</div>
        `;
        latentPosteriors.appendChild(div);
    }
}

// Utils
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}
