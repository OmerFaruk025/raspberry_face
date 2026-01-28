function sendCommand(command) {
    fetch(`/${command}`).then(() => setTimeout(updateStatus, 300));
}

async function updateStatus() {
    try {
        // Durum güncelle
        const resStatus = await fetch('/');
        const html = await resStatus.text();
        const parser = new DOMParser();
        const doc = parser.parseFromString(html, 'text/html');
        const status = doc.getElementById('status').innerText;
        const statusEl = document.getElementById('status');
        statusEl.innerText = status;
        statusEl.className = status === "Çalışıyor" ? "running" : "stopped";

        // Logları çek ve göster
        const resLogs = await fetch('/logs');
        const data = await resLogs.json();
        const logContainer = document.getElementById('log');

        logContainer.innerHTML = "";
        data.forEach(item => {
            const p = document.createElement("p");
            p.textContent = `${item.name}`;
            logContainer.appendChild(p);
        });

        // Scroll en alta (artık scrollbar yok ama içerik altta başlar)
        logContainer.scrollTop = logContainer.scrollHeight;
    } catch (err) {
        console.error("Log çekme hatası:", err);
    }
}

// İlk yüklemede ve her saniye güncelle
updateStatus();
setInterval(updateStatus, 1000);
