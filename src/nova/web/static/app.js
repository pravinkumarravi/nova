// Nova Web UI - Client Application Logic with Real-time SSE Streaming

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const modeAgentBtn = document.getElementById("mode-agent-btn");
  const modeLmBtn = document.getElementById("mode-lm-btn");
  const agentSettings = document.getElementById("agent-settings");
  const lmSettings = document.getElementById("lm-settings");
  const currentModeBadge = document.getElementById("current-mode-badge");
  const modeBadgeLabel = document.getElementById("mode-badge-label");

  const tempSlider = document.getElementById("temp-slider");
  const tempVal = document.getElementById("temp-val");
  const topkSlider = document.getElementById("topk-slider");
  const topkVal = document.getElementById("topk-val");
  const threshSlider = document.getElementById("thresh-slider");
  const threshVal = document.getElementById("thresh-val");

  const chatForm = document.getElementById("chat-form");
  const userInput = document.getElementById("user-input");
  const sendBtn = document.getElementById("send-btn");
  const messagesViewport = document.getElementById("messages-viewport");
  const messagesList = document.getElementById("messages-list");
  const welcomeCard = document.getElementById("welcome-card");
  const clearChatBtn = document.getElementById("clear-chat-btn");
  const istClock = document.getElementById("ist-clock");

  const statLmParams = document.getElementById("stat-lm-params");
  const statIntentParams = document.getElementById("stat-intent-params");
  const statVocab = document.getElementById("stat-vocab");

  let currentMode = "agent"; // 'agent' or 'lm'
  let isStreaming = false;

  // 1. Live Indian Standard Time (IST) Clock
  function updateClock() {
    try {
      const now = new Date();
      const options = {
        timeZone: "Asia/Kolkata",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        hour12: true,
      };
      istClock.textContent = new Intl.DateTimeFormat("en-IN", options).format(now);
    } catch (e) {
      istClock.textContent = new Date().toLocaleTimeString();
    }
  }
  updateClock();
  setInterval(updateClock, 1000);

  // 2. Fetch System Metadata
  async function loadSystemInfo() {
    try {
      const res = await fetch("/api/info");
      if (res.ok) {
        const data = await res.json();
        if (data.language_model) {
          statLmParams.textContent = Number(data.language_model.parameters).toLocaleString();
          statVocab.textContent = `${data.language_model.vocab_size} tokens`;
        }
        if (data.intent_classifier) {
          statIntentParams.textContent = Number(data.intent_classifier.parameters).toLocaleString();
        }
      }
    } catch (err) {
      console.warn("Unable to fetch /api/info:", err);
    }
  }
  loadSystemInfo();

  // 3. Sliders & Mode Toggle
  tempSlider.addEventListener("input", (e) => (tempVal.textContent = Number(e.target.value).toFixed(2)));
  topkSlider.addEventListener("input", (e) => (topkVal.textContent = e.target.value));
  threshSlider.addEventListener("input", (e) => (threshVal.textContent = Number(e.target.value).toFixed(2)));

  function setMode(mode) {
    currentMode = mode;
    if (mode === "agent") {
      modeAgentBtn.classList.add("active");
      modeLmBtn.classList.remove("active");
      agentSettings.style.display = "flex";
      lmSettings.style.display = "none";
      modeBadgeLabel.textContent = "Active: Agent Mode (Router + Tools)";
    } else {
      modeLmBtn.classList.add("active");
      modeAgentBtn.classList.remove("active");
      lmSettings.style.display = "flex";
      agentSettings.style.display = "none";
      modeBadgeLabel.textContent = "Active: Decoder LM (Autoregressive Generation)";
    }
  }

  modeAgentBtn.addEventListener("click", () => setMode("agent"));
  modeLmBtn.addEventListener("click", () => setMode("lm"));

  // 4. Quick Prompt Pills
  document.querySelectorAll(".prompt-pill").forEach((pill) => {
    pill.addEventListener("click", () => {
      userInput.value = pill.getAttribute("data-prompt");
      userInput.focus();
      handleSendMessage();
    });
  });

  // 5. Clear Chat
  clearChatBtn.addEventListener("click", () => {
    messagesList.innerHTML = "";
    welcomeCard.style.display = "flex";
  });

  // 6. Handle Send Message
  chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    handleSendMessage();
  });

  async function handleSendMessage() {
    const text = userInput.value.trim();
    if (!text || isStreaming) return;

    // Hide welcome card
    welcomeCard.style.display = "none";

    // Append User Message
    appendMessage("user", text);
    userInput.value = "";
    userInput.focus();

    // Prepare Bot Message with streaming container
    const botContainer = createBotMessageContainer();
    messagesList.appendChild(botContainer.element);
    scrollToBottom();

    // Set streaming state
    isStreaming = true;
    sendBtn.disabled = true;

    const payload = {
      prompt: text,
      mode: currentMode,
      temperature: parseFloat(tempSlider.value),
      top_k: parseInt(topkSlider.value, 10),
      threshold: parseFloat(threshSlider.value),
    };

    try {
      const response = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`HTTP error ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop(); // keep trailing incomplete chunk

        for (const block of lines) {
          if (!block.trim()) continue;
          parseSSEBlock(block, botContainer);
        }
      }
    } catch (err) {
      botContainer.content.textContent = `Error connecting to Nova server: ${err.message}`;
    } finally {
      botContainer.cursor.remove();
      isStreaming = false;
      sendBtn.disabled = false;
      scrollToBottom();
    }
  }

  function parseSSEBlock(block, botContainer) {
    const lines = block.split("\n");
    let eventType = "message";
    let dataStr = "";

    for (const line of lines) {
      if (line.startsWith("event: ")) {
        eventType = line.slice(7).trim();
      } else if (line.startsWith("data: ")) {
        dataStr = line.slice(6).trim();
      }
    }

    if (!dataStr) return;

    try {
      const data = JSON.parse(dataStr);

      if (eventType === "step") {
        renderTelemetryStep(data, botContainer.stepsContainer);
        scrollToBottom();
      } else if (eventType === "token") {
        botContainer.content.textContent += data.text;
        scrollToBottom();
      } else if (eventType === "done") {
        if (botContainer.content.textContent === "" && data.final_response) {
          botContainer.content.textContent = data.final_response;
        }
      } else if (eventType === "error") {
        botContainer.content.textContent = `Error: ${data.message}`;
      }
    } catch (e) {
      console.error("SSE parse error:", e);
    }
  }

  function renderTelemetryStep(stepData, container) {
    const card = document.createElement("div");
    card.className = "step-card";

    if (stepData.step === "classification") {
      card.classList.add("intent-card");
      card.innerHTML = `
        <div class="step-title">
          <span>🧠 Intent:</span>
          <span class="step-tag intent-badge">${stepData.intent}</span>
        </div>
        <div class="conf-pill">
          Confidence: ${(stepData.confidence * 100).toFixed(1)}% ${stepData.passed ? "✓" : "⚠ (fallback)"}
        </div>
      `;
    } else if (stepData.step === "tool_execution") {
      card.classList.add("tool-card");
      const val = stepData.result ? stepData.result.formatted_value : "";
      card.innerHTML = `
        <div class="step-title">
          <span>⚙️ Tool Call:</span>
          <span class="step-tag"><code>${stepData.tool}</code></span>
        </div>
        <div class="conf-pill" style="color: var(--accent-emerald);">
          ${val ? `Result: ${val}` : "Executed"}
        </div>
      `;
    } else {
      card.innerHTML = `
        <div class="step-title">
          <span>🔍 ${stepData.title}</span>
        </div>
        <div class="conf-pill" style="color: var(--text-dim);">${stepData.detail || ""}</div>
      `;
    }

    container.appendChild(card);
  }

  function appendMessage(role, text) {
    const item = document.createElement("div");
    item.className = `message-item ${role}`;
    item.innerHTML = `
      <div class="message-avatar">${role === "user" ? "U" : "N"}</div>
      <div class="message-bubble">
        <div class="message-content">${escapeHTML(text)}</div>
      </div>
    `;
    messagesList.appendChild(item);
    scrollToBottom();
  }

  function createBotMessageContainer() {
    const item = document.createElement("div");
    item.className = "message-item bot";

    const stepsContainer = document.createElement("div");
    stepsContainer.className = "telemetry-steps";

    const content = document.createElement("span");
    content.className = "text-stream";

    const cursor = document.createElement("span");
    cursor.className = "streaming-cursor";

    const messageContent = document.createElement("div");
    messageContent.className = "message-content";
    messageContent.appendChild(content);
    messageContent.appendChild(cursor);

    const bubble = document.createElement("div");
    bubble.className = "message-bubble";
    bubble.appendChild(stepsContainer);
    bubble.appendChild(messageContent);

    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = "N";

    item.appendChild(avatar);
    item.appendChild(bubble);

    return {
      element: item,
      stepsContainer,
      content,
      cursor,
    };
  }

  function scrollToBottom() {
    messagesViewport.scrollTop = messagesViewport.scrollHeight;
  }

  function escapeHTML(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }
});
