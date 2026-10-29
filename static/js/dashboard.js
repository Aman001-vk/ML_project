(() => {
    "use strict";

    const STORAGE_KEY = "student-outcome-prediction-history-v1";
    const MAX_ENTRIES = 20;
    const validOutcomes = new Set(["Dropout", "Enrolled", "Graduate"]);
    const list = document.getElementById("history-list");
    const empty = document.getElementById("history-empty");
    const message = document.getElementById("history-message");

    function readHistory() {
        try {
            const raw = window.localStorage.getItem(STORAGE_KEY);
            if (raw === null) return [];
            const records = JSON.parse(raw);
            if (!Array.isArray(records)) throw new Error("History is not a list.");
            return records.filter((entry) => entry &&
                typeof entry.studentId === "string" &&
                validOutcomes.has(entry.outcome) &&
                typeof entry.eventId === "string" &&
                typeof entry.timestamp === "string");
        } catch (_error) {
            if (message) message.textContent = "Saved history is unavailable or invalid. Clear it to start a fresh local history.";
            return null;
        }
    }

    function renderHistory(records) {
        if (!list || !empty) return;
        list.replaceChildren();
        if (!records) {
            empty.hidden = true;
            return;
        }
        const newestFirst = [...records].sort((a, b) => b.timestamp.localeCompare(a.timestamp));
        empty.hidden = newestFirst.length > 0;
        newestFirst.forEach((entry) => {
            const item = document.createElement("li");
            item.className = "history-item";
            const details = document.createElement("div");
            const id = document.createElement("strong");
            id.textContent = `Student ID ${entry.studentId}`;
            const date = document.createElement("span");
            const parsedDate = new Date(entry.timestamp);
            date.textContent = Number.isNaN(parsedDate.getTime()) ? "Date unavailable" :
                new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(parsedDate);
            details.append(id, date);
            const outcome = document.createElement("span");
            outcome.className = `history-outcome ${entry.outcome.toLowerCase()}`;
            outcome.textContent = entry.outcome;
            item.append(details, outcome);
            list.append(item);
        });
    }

    let history = readHistory();
    if (history !== null) renderHistory(history);

    const resultEvent = document.getElementById("prediction-event");
    if (resultEvent && history !== null) {
        const { studentId, outcome, eventId } = resultEvent.dataset;
        if (studentId && /^[0-9]{1,20}$/.test(studentId) && validOutcomes.has(outcome) && eventId) {
            if (!history.some((entry) => entry.eventId === eventId)) {
                history.unshift({ studentId, outcome, timestamp: new Date().toISOString(), eventId });
                history = history.slice(0, MAX_ENTRIES);
                try {
                    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
                    if (message) message.textContent = "This successful prediction was saved in this browser.";
                    renderHistory(history);
                } catch (_error) {
                    if (message) message.textContent = "Prediction succeeded, but browser storage is unavailable. No history was saved.";
                }
            }
        }
    }

    const clearButton = document.getElementById("clear-history");
    if (clearButton) {
        clearButton.addEventListener("click", () => {
            try {
                window.localStorage.removeItem(STORAGE_KEY);
                history = [];
                if (message) message.textContent = "Local prediction history cleared.";
                renderHistory(history);
            } catch (_error) {
                if (message) message.textContent = "Browser storage could not be cleared.";
            }
        });
    }

    const form = document.getElementById("prediction-form");
    const button = document.getElementById("predict-button");
    const status = document.getElementById("submit-status");
    if (form && button) {
        form.addEventListener("submit", (event) => {
            if (!form.reportValidity()) {
                event.preventDefault();
                return;
            }
            if (button.disabled) {
                event.preventDefault();
                return;
            }
            button.disabled = true;
            button.classList.add("is-loading");
            button.setAttribute("aria-busy", "true");
            button.querySelector(".button-label").textContent = "Checking student record…";
            if (status) status.textContent = "Looking up the record and generating an estimate.";
        });
    }
})();
