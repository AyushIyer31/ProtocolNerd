const baseURL = window.env.FRONTEND_FLOW.API_URL;
const disclaimer = window.env.FRONTEND_FLOW.DISCLAIMER;

// ------------------------------
// General helpers
// ------------------------------
function $(id) {
    return document.getElementById(id);
}

function safeJsonParse(value, fallback = null) {
    try {
        return JSON.parse(value);
    } catch {
        return fallback;
    }
}

function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str ?? "";
    return div.innerHTML;
}

function markdownToHtml(text) {
    if (!text) return "";
    let html = escapeHtml(text);
    // Bold: **text**
    html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    // Italic: *text*
    html = html.replace(/(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)/g, "<em>$1</em>");
    // Numbered lists: lines starting with 1. 2. etc.
    html = html.replace(/^(\d+)\.\s+(.+)$/gm, '<li value="$1">$2</li>');
    html = html.replace(/((?:<li[^>]*>.*<\/li>\n?)+)/g, '<ol>$1</ol>');
    // Bullet lists: lines starting with - or *
    html = html.replace(/^[-*]\s+(.+)$/gm, "<li>$1</li>");
    html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, function (match) {
        if (match.includes('<li value=')) return match;
        return "<ul>" + match + "</ul>";
    });
    // Paragraphs from double newlines
    html = html.replace(/\n{2,}/g, "</p><p>");
    // Single newlines to <br> (but not inside lists)
    html = html.replace(/\n/g, "<br>");
    // Clean up empty tags
    html = html.replace(/<p><\/p>/g, "");
    return "<p>" + html + "</p>";
}

function humanizeExecutionStrategy(strategy) {
    const map = {
        agentic: "Agentic",
        prompt_based: "Prompt-based",
    };
    return map[strategy] || strategy || "Unknown";
}

// ------------------------------
// Progress log
// ------------------------------
function appendProgressLine(line) {
    const log = $("progress-log");
    if (!log) return;
    const entry = document.createElement("div");
    entry.className = "progress-entry";
    entry.textContent = line;
    log.appendChild(entry);
    log.scrollTop = log.scrollHeight;
}

function clearProgress() {
    const log = $("progress-log");
    if (log) log.innerHTML = "";
}

// ------------------------------
// Reset
// ------------------------------
function resetUI() {
    const results = $("results");
    const output = $("analysis-output");
    const meta = $("run-metadata");
    const progressLog = $("progress-log");
    const pdfButton = $("generate-pdf-button");

    if (results) {
        results.classList.remove("hidden");
        results.classList.add("visible");
    }
    if (output) output.innerHTML = "";
    if (meta) meta.innerHTML = "";
    if (progressLog) {
        progressLog.classList.remove("hidden");
        progressLog.innerHTML = "";
    }
    localStorage.removeItem("localAnalysisResult");
    localStorage.removeItem("rawOutput");
    if (pdfButton) pdfButton.classList.add("hidden");
}

// ------------------------------
// Render analysis result as HTML
// ------------------------------
function statusBadge(status) {
    const map = {
        compliant: { cls: "badge-compliant", label: "Compliant" },
        non_compliant: { cls: "badge-noncompliant", label: "Non-Compliant" },
        unclear: { cls: "badge-unclear", label: "Unclear" },
    };
    const info = map[status] || map.unclear;
    return `<span class="status-badge ${info.cls}">${info.label}</span>`;
}

function riskBadge(risk) {
    const map = {
        low: { cls: "badge-compliant", label: "Low" },
        medium: { cls: "badge-unclear", label: "Medium" },
        high: { cls: "badge-noncompliant", label: "High" },
    };
    const info = map[risk] || { cls: "badge-unclear", label: risk || "Unknown" };
    return `<span class="status-badge ${info.cls}">${info.label}</span>`;
}

function renderAnalysisResult(result) {
    const output = $("analysis-output");
    const meta = $("run-metadata");
    const pdfButton = $("generate-pdf-button");
    const progressLog = $("progress-log");

    if (!output) return;

    // Hide progress log once results arrive
    if (progressLog) progressLog.classList.add("hidden");

    const fo = result?.final_output || {};
    const html = [];

    // --- Synthesis (final report) ---
    if (fo.synthesis) {
        html.push(`
            <section class="report-section report-synthesis">
                <h3><i class="fas fa-gavel"></i> Final Report</h3>
                <div class="report-body">${markdownToHtml(fo.synthesis)}</div>
            </section>
        `);
    }

    // --- Overall risk + counts ---
    if (fo.overall_risk) {
        const counts = fo.status_counts || {};
        html.push(`
            <section class="report-section report-risk">
                <h3><i class="fas fa-shield-alt"></i> Overall Risk: ${riskBadge(fo.overall_risk)}</h3>
                <div class="status-counts">
                    <span class="count-chip count-compliant">${counts.compliant ?? 0} compliant</span>
                    <span class="count-chip count-noncompliant">${counts.non_compliant ?? 0} non-compliant</span>
                    <span class="count-chip count-unclear">${counts.unclear ?? 0} unclear</span>
                </div>
            </section>
        `);
    }

    // --- Target summary ---
    if (fo.summary) {
        html.push(`
            <section class="report-section report-summary">
                <h3><i class="fas fa-file-alt"></i> Target Document Summary</h3>
                <div class="report-body">${markdownToHtml(fo.summary)}</div>
            </section>
        `);
    }

    // --- Findings ---
    if (Array.isArray(fo.findings) && fo.findings.length > 0) {
        const findingsHtml = fo.findings
            .map((f, i) => {
                const evidenceHtml = Array.isArray(f.context_evidence) && f.context_evidence.length > 0
                    ? f.context_evidence
                          .filter((e) => e.quote)
                          .map((e) => `<div class="evidence-item"><span class="evidence-source">${escapeHtml(e.source_file)} (${escapeHtml(e.chunk_id)})</span><blockquote>${escapeHtml(e.quote)}</blockquote></div>`)
                          .join("")
                    : "";
                return `
                    <div class="finding-card">
                        <div class="finding-header">
                            <span class="finding-number">#${i + 1}</span>
                            ${statusBadge(f.status)}
                            <span class="finding-title">${escapeHtml(f.issue || "Untitled")}</span>
                        </div>
                        ${f.explanation ? `<div class="finding-explanation">${markdownToHtml(f.explanation)}</div>` : ""}
                        ${evidenceHtml ? `<div class="finding-evidence"><strong>Evidence:</strong>${evidenceHtml}</div>` : ""}
                        ${f.recommendation ? `<div class="finding-recommendation"><strong>Recommendation:</strong> ${escapeHtml(f.recommendation)}</div>` : ""}
                    </div>
                `;
            })
            .join("");

        html.push(`
            <section class="report-section report-findings">
                <h3><i class="fas fa-search"></i> Detailed Findings (${fo.findings.length})</h3>
                ${findingsHtml}
            </section>
        `);
    }

    // --- Gaps / Notes ---
    if (Array.isArray(fo.gaps) && fo.gaps.length > 0) {
        html.push(`
            <section class="report-section">
                <h3><i class="fas fa-exclamation-circle"></i> Gaps</h3>
                <ul>${fo.gaps.map((g) => `<li>${escapeHtml(g)}</li>`).join("")}</ul>
            </section>
        `);
    }

    // --- Fallback for unexpected schema ---
    if (html.length === 0) {
        html.push(`
            <section class="report-section">
                <h3>Analysis Output</h3>
                <pre class="raw-json">${escapeHtml(JSON.stringify(fo, null, 2))}</pre>
            </section>
        `);
    }

    output.innerHTML = html.join("");

    // --- Metadata sidebar ---
    if (meta) {
        meta.innerHTML = `
            <div class="meta-item"><i class="fas fa-cog"></i> <strong>Analysis:</strong> ${escapeHtml(result.analysis_mode || "compliance")}</div>
            <div class="meta-item"><i class="fas fa-sitemap"></i> <strong>Execution:</strong> ${escapeHtml(humanizeExecutionStrategy(result.execution_strategy || "agentic"))}</div>
            <div class="meta-item"><i class="fas fa-file"></i> <strong>Target:</strong> ${escapeHtml(result.target_file || "Unknown")}</div>
            <div class="meta-item"><i class="fas fa-folder"></i> <strong>Context:</strong> ${escapeHtml((result.context_files || []).join(", "))}</div>
            <div class="meta-item"><i class="fas fa-cubes"></i> <strong>Chunks:</strong> ${result.retrieved_chunk_count ?? "?"}</div>
            <div class="meta-item"><i class="fas fa-clock"></i> <strong>Runtime:</strong> ${result.runtime_seconds ?? "?"}s</div>
        `;
    }

    // Store for PDF export
    localStorage.setItem("localAnalysisResult", JSON.stringify(result));
    localStorage.setItem("rawOutput", output.innerText);

    if (pdfButton) pdfButton.classList.remove("hidden");
}

// ------------------------------
// Custom prompts management
// ------------------------------
const customPrompts = [];

function toggleCustomPromptsSection() {
    const strategy = $("execution-strategy")?.value;
    const section = $("custom-prompts-section");
    if (!section) return;
    if (strategy === "prompt_based") {
        section.classList.remove("hidden");
    } else {
        section.classList.add("hidden");
    }
}

function renderPromptList() {
    const listEl = $("prompt-list");
    const itemsEl = $("prompt-items");
    const countEl = $("prompt-count");
    if (!listEl || !itemsEl || !countEl) return;

    if (customPrompts.length === 0) {
        listEl.classList.add("hidden");
        return;
    }

    listEl.classList.remove("hidden");
    countEl.textContent = customPrompts.length;
    itemsEl.innerHTML = customPrompts
        .map((p, i) => `
            <li class="prompt-item">
                <span class="prompt-item-number">${i + 1}.</span>
                <span class="prompt-item-text">${escapeHtml(p)}</span>
                <button class="prompt-item-remove" title="Remove" onclick="removePrompt(${i})">
                    <i class="fas fa-times"></i>
                </button>
            </li>
        `)
        .join("");
}

function addPrompt(text) {
    const trimmed = (text || "").trim();
    if (!trimmed) return false;
    customPrompts.push(trimmed);
    renderPromptList();
    return true;
}

function removePrompt(index) {
    customPrompts.splice(index, 1);
    renderPromptList();
}

function clearPrompts() {
    customPrompts.length = 0;
    renderPromptList();
}

async function parsePromptsFile(file) {
    const text = await file.text();
    let parsed = [];

    if (file.name.endsWith(".json")) {
        try {
            const json = JSON.parse(text);
            if (Array.isArray(json)) {
                parsed = json.filter((x) => typeof x === "string" && x.trim());
            } else {
                appendProgressLine("Prompts file: JSON must be an array of strings.");
                return;
            }
        } catch {
            appendProgressLine("Prompts file: could not parse JSON.");
            return;
        }
    } else {
        // .txt — one prompt per non-empty line
        parsed = text.split("\n").map((l) => l.trim()).filter(Boolean);
    }

    if (parsed.length === 0) {
        appendProgressLine("Prompts file: no prompts found.");
        return;
    }

    parsed.forEach((p) => customPrompts.push(p));
    renderPromptList();
}

function initializePromptsPanel() {
    const addBtn = $("add-prompt-btn");
    const promptInput = $("prompt-input");
    const clearBtn = $("clear-prompts-btn");
    const promptsFile = $("prompts-file");

    if (addBtn && promptInput) {
        addBtn.addEventListener("click", () => {
            if (addPrompt(promptInput.value)) promptInput.value = "";
        });
        promptInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                if (addPrompt(promptInput.value)) promptInput.value = "";
            }
        });
    }

    if (clearBtn) {
        clearBtn.addEventListener("click", clearPrompts);
    }

    if (promptsFile) {
        promptsFile.addEventListener("change", async () => {
            if (promptsFile.files?.[0]) {
                await parsePromptsFile(promptsFile.files[0]);
                promptsFile.value = "";
            }
        });
    }
}

// ------------------------------
// File upload display
// ------------------------------
function updateFileList(inputEl, listEl) {
    const files = Array.from(inputEl.files || []);
    if (files.length === 0) {
        listEl.innerHTML = "";
        listEl.classList.add("empty-state");
        listEl.textContent = inputEl.multiple
            ? "No context files selected yet."
            : "No target file selected yet.";
        return;
    }
    listEl.classList.remove("empty-state");
    listEl.innerHTML = files
        .map(
            (f) =>
                `<span class="file-chip"><i class="fas fa-file"></i> ${escapeHtml(f.name)} <small>(${(f.size / 1024).toFixed(0)} KB)</small></span>`
        )
        .join("");
}

function initializeFileInputs() {
    const contextInput = $("context-files");
    const contextList = $("context-files-list");
    const targetInput = $("target-file");
    const targetList = $("target-file-list");

    if (contextInput && contextList) {
        contextInput.addEventListener("change", () => updateFileList(contextInput, contextList));
    }
    if (targetInput && targetList) {
        targetInput.addEventListener("change", () => updateFileList(targetInput, targetList));
    }
}

// ------------------------------
// UI behavior
// ------------------------------
function triggerSubmitAnimation() {
    const submitButton = $("submit");
    if (!submitButton) return;
    submitButton.classList.add("submitting");
    setTimeout(() => submitButton.classList.remove("submitting"), 600);
}

function setSubmitDisabled(disabled) {
    const submitButton = $("submit");
    if (!submitButton) return;
    submitButton.disabled = disabled;
    submitButton.style.opacity = disabled ? "0.6" : "1";
    submitButton.style.cursor = disabled ? "not-allowed" : "pointer";
}

// ------------------------------
// PDF generation
// ------------------------------
function generatePDF() {
    try {
        if (typeof window.jspdf === "undefined") {
            alert("PDF generation library not loaded. Please refresh and try again.");
            return;
        }
        const stored = safeJsonParse(localStorage.getItem("localAnalysisResult"), null);
        const rawOutput = localStorage.getItem("rawOutput");
        const question = $("question")?.value?.trim() || "Untitled request";

        if (!stored || !rawOutput) {
            alert("No analysis is available to export yet.");
            return;
        }

        const { jsPDF } = window.jspdf;
        const doc = new jsPDF();
        const margin = 15;
        const maxWidth = 180;
        const pageHeight = doc.internal.pageSize.height;
        const lineHeight = 7;
        let y = 20;

        function addWrappedText(text, fontSize = 12, isBold = false) {
            doc.setFont("helvetica", isBold ? "bold" : "normal");
            doc.setFontSize(fontSize);
            const lines = doc.splitTextToSize(String(text ?? ""), maxWidth);
            lines.forEach((line) => {
                if (y > pageHeight - 20) { doc.addPage(); y = 20; }
                doc.text(line, margin, y);
                y += lineHeight;
            });
        }

        doc.setFont("helvetica", "bold");
        doc.setFontSize(14);
        doc.text("ProtocolsNerd Analysis Report", margin, y);
        y += 10;

        const now = new Date();
        doc.setFont("helvetica", "normal");
        doc.setFontSize(10);
        doc.text(`Generated: ${now.toLocaleDateString()} ${now.toLocaleTimeString()}`, margin, y);
        y += 12;

        addWrappedText("Question:", 14, true);
        addWrappedText(question, 12, false);
        y += 4;
        doc.line(margin, y, margin + maxWidth, y);
        y += 10;

        addWrappedText("Analysis:", 14, true);
        addWrappedText(rawOutput, 11, false);

        const timestamp = now.toISOString().replace(/[:.]/g, "-").slice(0, 19);
        doc.save(`analysis_${timestamp}.pdf`);
    } catch (error) {
        console.error("PDF generation error:", error);
        alert("Error generating PDF. Please try again.");
    }
}

// ------------------------------
// Backend communication
// ------------------------------
async function startLocalAnalysis({
    userQuery,
    contextFiles,
    targetFile,
    analysisMode = "compliance",
    executionStrategy = "agentic",
    prompts = [],
}) {
    const formData = new FormData();
    formData.append("user_query", userQuery);
    formData.append("analysis_mode", analysisMode);
    formData.append("execution_strategy", executionStrategy);
    for (const file of contextFiles) formData.append("context_files", file);
    formData.append("target_file", targetFile);
    if (prompts.length > 0) {
        formData.append("custom_prompts", JSON.stringify(prompts));
    }

    const response = await fetch(`${baseURL}/process_local_rag_analysis`, {
        method: "POST",
        body: formData,
        mode: "cors",
    });

    if (!response.ok) {
        const text = await response.text().catch(() => "");
        throw new Error(text || `HTTP error! status: ${response.status}`);
    }
    return response.json();
}

function listenToSession(sessionId) {
    return new Promise((resolve, reject) => {
        const eventSource = new EventSource(`${baseURL}/sse?session_id=${encodeURIComponent(sessionId)}`);
        eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (data.update) appendProgressLine(data.update);
                if (data.final_output) { eventSource.close(); resolve(data.final_output); }
                if (data.error) { eventSource.close(); reject(new Error(data.error)); }
            } catch (err) {
                console.error("Failed to parse SSE payload:", err);
            }
        };
        eventSource.onerror = () => {
            eventSource.close();
            reject(new Error("Streaming connection failed."));
        };
    });
}

// ------------------------------
// Main submit flow
// ------------------------------
async function handleSubmit() {
    const question = $("question")?.value?.trim() || "";
    const contextFiles = Array.from($("context-files")?.files || []);
    const targetFile = $("target-file")?.files?.[0] || null;
    const executionStrategy = $("execution-strategy")?.value || "agentic";

    resetUI();
    triggerSubmitAnimation();

    if (!question) { appendProgressLine("Please enter a question."); return; }
    if (contextFiles.length === 0) { appendProgressLine("Please upload at least one context document."); return; }
    if (!targetFile) { appendProgressLine("Please upload a target document."); return; }

    setSubmitDisabled(true);
    appendProgressLine("Connecting to backend...");

    try {
        const promptsToSend = executionStrategy === "prompt_based" ? [...customPrompts] : [];
        const { session_id: sessionId } = await startLocalAnalysis({
            userQuery: question,
            contextFiles,
            targetFile,
            analysisMode: "compliance",
            executionStrategy,
            prompts: promptsToSend,
        });

        appendProgressLine(`Session started: ${sessionId}`);
        const finalResult = await listenToSession(sessionId);
        renderAnalysisResult(finalResult);
    } catch (error) {
        console.error("Error processing query:", error);
        appendProgressLine(`Error: ${error.message || error}`);
    } finally {
        setSubmitDisabled(false);
    }
}

// ------------------------------
// DOMContentLoaded init
// ------------------------------
document.addEventListener("DOMContentLoaded", function () {
    const questionInput = $("question");
    const submitButton = $("submit");
    const generatePdfButton = $("generate-pdf-button");
    const disclaimerEl = document.querySelector(".disclaimer");
    const executionStrategySelect = $("execution-strategy");

    document.title = window.env.FRONTEND_FLOW.SITE_NAME;
    const logoEl = $("site-logo");
    if (logoEl) logoEl.src = window.env.FRONTEND_FLOW.SITE_LOGO;
    const taglineEl = $("site-tagline");
    if (taglineEl) taglineEl.textContent = window.env.FRONTEND_FLOW.SITE_TAGLINE;
    if (disclaimerEl) disclaimerEl.textContent = disclaimer;
    if (questionInput) questionInput.placeholder = window.env.FRONTEND_FLOW.QUESTION_PLACEHOLDER;
    document.body.style.backgroundColor = window.env.FRONTEND_FLOW.STYLES.BACKGROUND_COLOR;
    document.body.style.fontFamily = window.env.FRONTEND_FLOW.STYLES.FONT_FAMILY;
    if (submitButton) submitButton.style.backgroundColor = window.env.FRONTEND_FLOW.STYLES.SUBMIT_BUTTON_BG;

    fetch(`${baseURL}/fetch_backend_mode`)
        .then((response) => response.json())
        .then((backendMode) => {
            if (executionStrategySelect && backendMode?.default_execution_strategy) {
                executionStrategySelect.value = backendMode.default_execution_strategy;
            }
        })
        .catch((error) => {
            console.warn("Could not fetch backend mode:", error);
        });

    initializeFileInputs();
    initializePromptsPanel();
    toggleCustomPromptsSection();

    if (executionStrategySelect) {
        executionStrategySelect.addEventListener("change", toggleCustomPromptsSection);
    }

    const updateSubmitState = () => {
        const hasText = (questionInput?.value || "").trim().length > 0;
        setSubmitDisabled(!hasText);
    };
    updateSubmitState();

    if (questionInput) {
        questionInput.addEventListener("input", updateSubmitState);
        questionInput.addEventListener("keydown", (event) => {
            if (event.key === "Enter" && (questionInput.value || "").trim() && !submitButton.disabled) {
                event.preventDefault();
                handleSubmit();
            }
        });
    }
    if (submitButton) submitButton.addEventListener("click", handleSubmit);
    if (generatePdfButton) generatePdfButton.addEventListener("click", () => generatePDF());
});
