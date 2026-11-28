"use strict";

const csrf = document.querySelector('meta[name="csrf-token"]').content;
const csrfHeaders = { "X-CSRF-Token": csrf };
const jsonHeaders = { ...csrfHeaders, "Content-Type": "application/json" };

function showError(node, message = "") {
  node.textContent = message;
  node.hidden = !message;
}

async function responseError(response, fallback) {
  try {
    const body = await response.json();
    return body.error?.message || fallback;
  } catch {
    return fallback;
  }
}

function refresh(name) {
  window.htmx.trigger(document.body, name);
}

const account = document.querySelector("#account-form");
account?.addEventListener("submit", async (event) => {
  event.preventDefault();
  const error = account.querySelector("[data-error]");
  const button = account.querySelector("button");
  const credentials = JSON.stringify(Object.fromEntries(new FormData(account)));
  showError(error);
  button.disabled = true;
  try {
    if (account.dataset.action === "register") {
      const result = await fetch("/auth/register", { method: "POST", headers: jsonHeaders, body: credentials });
      if (!result.ok) throw new Error(await responseError(result, "Could not create the account."));
    }
    const result = await fetch("/auth/login", { method: "POST", headers: jsonHeaders, body: credentials });
    if (!result.ok) throw new Error(await responseError(result, "Could not sign in."));
    location.assign("/library");
  } catch (failure) {
    showError(error, failure.message);
    button.disabled = false;
  }
});

document.querySelector("[data-logout]")?.addEventListener("click", async () => {
  try {
    const result = await fetch("/auth/logout", { method: "POST", headers: csrfHeaders });
    if (result.ok || result.status === 401) {
      location.assign("/login");
      return;
    }
    throw new Error(await responseError(result, "Could not sign out."));
  } catch (failure) {
    showError(document.querySelector("#session-error"), failure.message);
  }
});

const upload = document.querySelector("#upload-form");
upload?.addEventListener("submit", (event) => {
  event.preventDefault();
  const file = upload.elements.pdf.files[0];
  const error = upload.querySelector("[data-error]");
  const status = document.querySelector("#upload-status");
  const progress = document.querySelector("#upload-progress");
  const button = upload.querySelector("button");
  showError(error);
  status.textContent = "";
  if (!file) return;
  if (file.size > Number(upload.dataset.limit) * 1024 ** 2) {
    progress.hidden = true;
    showError(error, "This PDF exceeds the upload size limit.");
    return;
  }
  const params = new URLSearchParams({ filename: file.name });
  for (const key of ["document_id", "version"]) {
    const value = upload.elements[key].value.trim();
    if (value) params.set(key, value);
  }
  const request = new XMLHttpRequest();
  request.open("POST", "/documents?" + params);
  request.setRequestHeader("Content-Type", "application/pdf");
  request.setRequestHeader("X-CSRF-Token", csrf);
  request.responseType = "json";
  button.disabled = true;
  progress.hidden = false;
  progress.value = 0;
  status.textContent = "Uploading…";
  request.upload.onprogress = (transfer) => {
    if (transfer.lengthComputable) progress.value = 100 * transfer.loaded / transfer.total;
  };
  request.onloadend = () => {
    button.disabled = false;
    progress.hidden = true;
    status.textContent = "";
    if (request.status === 202) {
      upload.reset();
      status.textContent = "Upload received. Track indexing under Recent uploads.";
      refresh("uploadChanged");
    } else if (request.status === 401) {
      location.assign("/login");
    } else {
      showError(error, request.response?.error?.message || "Could not upload the PDF.");
    }
  };
  request.send(file);
});

let completed = document.querySelector("#uploads")?.dataset.completed;
document.body.addEventListener("htmx:afterSwap", (event) => {
  if (event.detail.target.id !== "uploads") return;
  const current = document.querySelector("#uploads")?.dataset.completed;
  if (current !== completed) {
    completed = current;
    refresh("libraryChanged");
  }
});

const deleteDialog = document.querySelector("#delete-dialog");
document.body.addEventListener("click", (event) => {
  const button = event.target.closest("[data-delete]");
  if (!button || !deleteDialog) return;
  deleteDialog.dataset.documentId = button.dataset.delete;
  document.querySelector("#delete-name").textContent = button.dataset.name;
  deleteDialog.returnValue = "";
  deleteDialog.showModal();
});
deleteDialog?.addEventListener("close", async () => {
  if (deleteDialog.returnValue !== "delete") return;
  const error = document.querySelector("#document-error");
  showError(error);
  try {
    const result = await fetch("/documents/" + encodeURIComponent(deleteDialog.dataset.documentId), {
      method: "DELETE", headers: csrfHeaders,
    });
    if (result.status === 401) {
      location.assign("/login");
      return;
    }
    if (!result.ok) throw new Error(await responseError(result, "Could not delete the document."));
    refresh("libraryChanged");
    refresh("uploadChanged");
  } catch (failure) {
    showError(error, failure.message);
  }
});

function addMessage(container, role, text) {
  const article = document.createElement("article");
  article.className = "message message-" + role;
  const label = document.createElement("h3");
  label.textContent = role === "user" ? "You" : "Answer";
  const content = document.createElement("p");
  content.className = "message-text";
  content.textContent = text;
  article.append(label, content);
  container.append(article);
  return { article, content };
}

function citationLinks(citations) {
  const links = document.createElement("div");
  links.className = "citations";
  links.setAttribute("aria-label", "Source pages");
  for (const citation of citations) {
    const version = encodeURIComponent(citation.document_id);
    const chunk = encodeURIComponent(citation.chunk_id);
    for (const page of citation.pages) {
      if (!Number.isInteger(page) || page < 1) continue;
      const link = document.createElement("a");
      link.href = "/citations/" + version + "/" + chunk + "/pages/" + page;
      link.textContent = "[" + citation.marker + "] Page " + page;
      link.target = "_blank";
      link.rel = "noopener";
      links.append(link);
    }
  }
  return links;
}

function rememberChat(form, chatId, title) {
  if (form.dataset.chatId) return;
  form.dataset.chatId = chatId;
  const href = "/chat?chat_id=" + encodeURIComponent(chatId);
  history.replaceState(null, "", href);
  const item = document.createElement("li");
  const link = document.createElement("a");
  link.href = href;
  link.textContent = title.slice(0, 100);
  link.setAttribute("aria-current", "page");
  item.append(link);
  document.querySelector("#chat-list").prepend(item);
  document.querySelector("#no-chats")?.remove();
}

async function readEvents(stream, consume) {
  const reader = stream.getReader();
  const decode = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += decode.decode(value, { stream: !done });
      const events = buffer.split(/\r?\n\r?\n/);
      buffer = events.pop();
      for (const event of events) {
        const lines = event.split(/\r?\n/);
        const type = lines.find((line) => line.startsWith("event:"))?.slice(6).trim();
        const data = lines.filter((line) => line.startsWith("data:")).map((line) => line.slice(5)).join("\n");
        if (type && data) consume(type, JSON.parse(data));
      }
      if (done) return;
    }
  } finally {
    reader.releaseLock();
  }
}

const question = document.querySelector("#question-form");
const conflictMessages = {
  prefer_newer: "Sources disagree. The answer uses the newer version.",
  refused: "Sources disagree. Their dates do not establish which version is newer.",
};
question?.addEventListener("submit", async (event) => {
  event.preventDefault();
  const error = question.querySelector("[data-error]");
  const button = question.querySelector("button");
  const formData = new FormData(question);
  const text = formData.get("question").trim();
  const versions = formData.getAll("version_ids");
  showError(error);
  if (!text || versions.length < 1 || versions.length > 20) {
    showError(error, "Enter a question and select between one and twenty source versions.");
    return;
  }
  const messages = document.querySelector("#messages");
  button.disabled = true;
  let pending, userMessage, streamed, answered = false, scrollPending = false;
  try {
    const result = await fetch("/ask", {
      method: "POST", headers: jsonHeaders,
      body: JSON.stringify({ question: text, version_ids: versions, chat_id: question.dataset.chatId || null }),
    });
    if (result.status === 401) {
      location.assign("/login");
      return;
    }
    if (!result.ok) throw new Error(await responseError(result, "Could not send the question."));
    const empty = document.querySelector("#empty-chat");
    if (empty) empty.hidden = true;
    userMessage = addMessage(messages, "user", text);
    pending = addMessage(messages, "assistant", "Checking sources…");
    messages.scrollTop = messages.scrollHeight;
    await readEvents(result.body, (type, data) => {
      if (type === "error") throw new Error(data.error.message);
      if (type === "token") {
        if (!streamed) {
          streamed = document.createTextNode("");
          pending.content.replaceChildren(streamed);
        }
        streamed.appendData(data.text);
        if (!scrollPending) {
          scrollPending = true;
          requestAnimationFrame(() => {
            messages.scrollTop = messages.scrollHeight;
            scrollPending = false;
          });
        }
        return;
      }
      if (type !== "answer") return;
      pending.content.textContent = data.answer;
      if (Object.hasOwn(conflictMessages, data.conflict_policy)) {
        const note = document.createElement("p");
        note.className = "conflict";
        note.setAttribute("role", "status");
        note.textContent = conflictMessages[data.conflict_policy];
        pending.content.before(note);
      }
      if (data.citations.length) pending.article.append(citationLinks(data.citations));
      rememberChat(question, data.chat_id, text);
      question.elements.question.value = "";
      messages.scrollTop = messages.scrollHeight;
      answered = true;
    });
    if (!answered) throw new Error("The answer stream ended early. Send the question again.");
  } catch (failure) {
    pending?.article.remove();
    userMessage?.article.remove();
    if (!messages.querySelector(".message")) {
      const empty = document.querySelector("#empty-chat");
      if (empty) empty.hidden = false;
    }
    showError(error, failure.message);
  } finally {
    button.disabled = false;
  }
});
