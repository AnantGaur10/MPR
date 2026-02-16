const powerButton = document.getElementById('power-button');
const statusText = document.getElementById('status-text');
const videoElement = document.getElementById('localVideo');
const videoSourceSelect = document.getElementById('video-source');
const currentWordDisplay = document.getElementById('current-word');
const currentSentenceDisplay = document.getElementById('current-sentence');
const authView = document.getElementById('auth-view');
const mainApp = document.getElementById('main-app');
const authForm = document.getElementById('auth-form');
const authTitle = document.getElementById('auth-title');
const authSubmit = document.getElementById('auth-submit');
const authError = document.getElementById('auth-error');
const authTabs = document.querySelectorAll('.auth-tab');
const profileModal = document.getElementById('profile-modal');
const showProfileBtn = document.getElementById('show-profile-btn');
const closeModalBtn = document.getElementById('close-modal');
const signsList = document.getElementById('signs-list');
const startRecBtn = document.getElementById('start-rec-btn');
let localStream = null;
let ws = null;
let recWs = null;
let captureInterval = null;
let statusInterval = null;
let isServiceRunning = false;
let isPaused = false;
let currentUser = null;
let authMode = 'signin';
const API_BASE = 'http://localhost:8080/api';
const WS_BASE = 'ws://localhost:8080/ws/ml';
const FRAME_RATE = 30;
async function checkAuth() {
  const token = localStorage.getItem('auth_token');
  if (!token) {
    showAuth();
    return;
  }
  try {
    const res = await fetch(`${API_BASE}/auth/me`, {
      headers: { 'Authorization': `Bearer ${token}` }
    });
    if (res.ok) {
      const data = await res.json();
      currentUser = data.user;
      showApp();
    } else {
      showAuth();
    }
  } catch (e) {
    console.error("Error in checkAuth:", e);
    showAuth();
  }
}
function showAuth() {
  authView.style.display = 'flex';
  mainApp.style.display = 'none';
  stopService();
}
function showApp() {
  authView.style.display = 'none';
  mainApp.style.display = 'flex';
  const display = document.getElementById('user-display');
  if (display) display.innerText = currentUser.name;
  loadSigns();
}
authTabs.forEach(tab => {
  tab.addEventListener('click', () => {
    authTabs.forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    authMode = tab.dataset.mode;
    authTitle.innerText = authMode === 'signin' ? 'Sign In' : 'Sign Up';
    authSubmit.innerText = authMode === 'signin' ? 'Sign In' : 'Create Account';
    document.getElementById('auth-name').style.display = authMode === 'signin' ? 'none' : 'block';
  });
});
authForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const email = document.getElementById('auth-email').value;
  const password = document.getElementById('auth-password').value;
  const name = document.getElementById('auth-name').value;
  const endpoint = authMode === 'signin' ? '/auth/signin' : '/auth/signup';
  const body = authMode === 'signin' ? { email, password } : { email, password, name };
  try {
    const res = await fetch(`${API_BASE}${endpoint}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    const data = await res.json();
    if (res.ok) {
      localStorage.setItem('auth_token', data.token);
      await checkAuth();
    } else {
      authError.innerText = data.error || 'Authentication failed';
    }
  } catch (e) {
    console.error("Error in authForm submit:", e);
    authError.innerText = 'Server connection failed';
  }
});
function logout() {
  localStorage.removeItem('auth_token');
  currentUser = null;
  showAuth();
  profileModal.style.display = 'none';
}
async function loadSigns() {
  const token = localStorage.getItem('auth_token');
  try {
    const res = await fetch(`${API_BASE}/signs/list`, {
      headers: { 'Authorization': `Bearer ${token}` }
    });
    if (res.ok) {
      const data = await res.json();
      signsList.innerHTML = data.signs.length ? '' : '<p class="no-signs">No custom signs recorded.</p>';
      data.signs.forEach(sign => {
        const item = document.createElement('div');
        item.className = 'sign-item';
        item.innerHTML = `
          <div class="sign-info">
            <span class="sign-name-tag">${sign.sign_name}</span>
            <span class="sign-samples">${sign.sample_count || 0} samples</span>
          </div>
          <button class="delete-btn" onclick="window.deleteSign('${sign.id}')">
            <svg viewBox="0 0 24 24" width="16" height="16"><path fill="currentColor" d="M19,4H15.5L14.5,3H9.5L8.5,4H5V6H19M6,19A2,2 0 0,0 8,21H16A2,2 0 0,0 18,19V7H6V19Z"/></svg>
          </button>`;
        signsList.appendChild(item);
      });
    }
  } catch (e) {
    console.error("Error in loadSigns:", e);
  }
} window.deleteSign = deleteSign;
document.getElementById('logout-btn')?.addEventListener('click', async () => {
  const token = localStorage.getItem('auth_token');
  if (token) {
    try {
      await fetch(`${API_BASE}/auth/logout`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` }
      });
    } catch (e) {
      console.error("Error in logout click:", e);
    }
  }
  localStorage.removeItem('auth_token');
  currentUser = null;
  location.reload();
});
async function deleteSign(id) {
  if (!confirm('Delete this sign?')) return;
  const token = localStorage.getItem('auth_token');
  try {
    await fetch(`${API_BASE}/signs/delete/${id}`, {
      method: 'DELETE',
      headers: { 'Authorization': `Bearer ${token}` }
    });
    loadSigns();
  } catch (e) {
    console.error("Error in deleteSign:", e);
  }
}
startRecBtn.addEventListener('click', async () => {
  const signName = document.getElementById('sign-name-input').value.trim().toUpperCase();
  if (!signName) return alert('Enter a sign name');
  const token = localStorage.getItem('auth_token');
  const recProgress = document.getElementById('rec-progress');
  const recFill = document.getElementById('rec-fill');
  recProgress.style.display = 'block';
  startRecBtn.disabled = true;
  if (!localStream) {
    try {
      localStream = await navigator.mediaDevices.getUserMedia({ video: { deviceId: videoSourceSelect.value ? { exact: videoSourceSelect.value } : undefined } });
      videoElement.srcObject = localStream;
    } catch (e) {
      console.error("Error in rec start (camera):", e);
      alert("Please check camera settings");
      startRecBtn.disabled = false;
      return;
    }
  }
  const canvas = document.createElement('canvas');
  canvas.width = 640; canvas.height = 480;
  const ctx = canvas.getContext('2d');
  recWs = new WebSocket(`${WS_BASE}?token=${token}`);
  recWs.onopen = () => {
    recWs.send(JSON.stringify({ type: 'RECORD_START', sign_name: signName }));
    const recInterval = setInterval(() => {
      if (recWs?.readyState === WebSocket.OPEN) {
        ctx.drawImage(videoElement, 0, 0, 640, 480);
        recWs.send(canvas.toDataURL('image/jpeg', 0.5));
      } else {
        clearInterval(recInterval);
      }
    }, 1000 / FRAME_RATE);
  };
  recWs.onmessage = (e) => {
    try {
      const msg = JSON.parse(e.data);
      if (msg.type === 'RECORD_STATUS') {
        if (msg.status === 'RECORDING') {
          recFill.style.width = `${(msg.count / msg.limit) * 100}%`;
        } else if (msg.status === 'COMPLETE') {
          alert(`Saved ${msg.sign}!`);
          recProgress.style.display = 'none';
          recFill.style.width = '0%';
          document.getElementById('sign-name-input').value = '';
          startRecBtn.disabled = false;
          loadSigns();
          setTimeout(() => document.getElementById('sign-name-input').focus(), 100);
          if (recWs) {
            recWs.close();
            recWs = null;
          }
        }
      }
    } catch (e) {
      console.error("Error in recWs.onmessage:", e);
    }
  };
  recWs.onerror = (e) => {
    console.error("WebSocket error in recording:", e);
    startRecBtn.disabled = false;
    recProgress.style.display = 'none';
  };
});
document.getElementById('train-btn').addEventListener('click', async () => {
  const token = localStorage.getItem('auth_token');
  const btn = document.getElementById('train-btn');
  btn.innerText = 'Training...';
  btn.disabled = true;
  try {
    const res = await fetch(`${API_BASE}/signs/train`, {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${token}` }
    });
    const data = await res.json();
    alert(data.message || data.error);
    setTimeout(() => document.getElementById('sign-name-input').focus(), 100);

    // If service is running, tell it to reload the model
    if (res.ok && ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'RELOAD_MODEL' }));
    }
  } catch (e) {
    console.error("Error in train-btn click:", e);
    alert('Training failed');
  }
  btn.innerText = 'Train Model';
  btn.disabled = false;
});
powerButton.addEventListener('click', () => {
  if (isServiceRunning) stopService(); else startService();
});
async function startService() {
  if (isServiceRunning) return;
  setUIState(true, "Connecting...");
  try {
    const token = localStorage.getItem('auth_token');
    const selectedDeviceId = videoSourceSelect.value;
    localStream = await navigator.mediaDevices.getUserMedia({ video: selectedDeviceId ? { deviceId: { exact: selectedDeviceId } } : true });
    videoElement.srcObject = localStream;
    const canvas = document.createElement('canvas');
    canvas.width = 640; canvas.height = 480;
    const ctx = canvas.getContext('2d');
    ws = new WebSocket(`${WS_BASE}?token=${token}`);
    ws.onopen = () => {
      setUIState(true, "Connected");
      captureInterval = setInterval(() => {
        if (ws?.readyState === WebSocket.OPEN && !isPaused) {
          ctx.drawImage(videoElement, 0, 0, 640, 480);
          ws.send(canvas.toDataURL('image/jpeg', 0.5));
        }
      }, 1000 / FRAME_RATE);
      statusInterval = setInterval(() => {
        if (ws?.readyState === WebSocket.OPEN && !isPaused) {
          ws.send(JSON.stringify({ type: 'GET_STATE' }));
        }
      }, 500);
    };
    ws.onmessage = async (event) => {
      try {
        if (typeof event.data === 'string') {
          const msg = JSON.parse(event.data);
          if (msg.type === 'STATE_UPDATE') {
            currentWordDisplay.textContent = msg.word || "---";
            currentSentenceDisplay.textContent = msg.sentence || "---";
          }
          return;
        }
        const audioUrl = URL.createObjectURL(event.data);
        const audio = new Audio(audioUrl);

        // --- Automated VB-CABLE Routing ---
        try {
          const devices = await navigator.mediaDevices.enumerateDevices();
          const cableInput = devices.find(d =>
            d.kind === 'audiooutput' &&
            d.label.toLowerCase().includes('cable input')
          );

          if (cableInput) {
            console.log(`Found VB-CABLE: ${cableInput.label}. Routing audio...`);
            await audio.setSinkId(cableInput.deviceId);
          } else {
            console.warn("VB-CABLE (CABLE Input) not found. Using default playback device.");
          }
        } catch (err) {
          console.error("Error setting VB-CABLE audio sink:", err);
        }

        audio.play().finally(() => {
          // Cleanup URL after some time to prevent memory leaks
          setTimeout(() => URL.revokeObjectURL(audioUrl), 10000);
        });
        // ---------------------------------
      } catch (e) {
        console.error("Error in ws.onmessage:", e);
      }
    };
    ws.onerror = (e) => {
      console.error("WebSocket error in service:", e);
      stopService();
    };
    ws.onclose = () => stopService();
  } catch (err) {
    console.error("Error in startService:", err);
    setUIState(false, "Error");
  }
}
function stopService() {
  clearInterval(captureInterval); clearInterval(statusInterval);
  if (ws) { ws.close(); ws = null; }
  if (localStream) { localStream.getTracks().forEach(t => t.stop()); localStream = null; }
  if (videoElement) videoElement.srcObject = null;
  setUIState(false, "Stopped");
}
function setUIState(isRunning, msg) {
  isServiceRunning = isRunning;
  statusText.innerText = msg;
  powerButton.classList.toggle('active', isRunning);
  statusText.classList.toggle('active', isRunning);
}
showProfileBtn.addEventListener('click', () => {
  profileModal.style.display = 'flex';
  loadSigns();
  setTimeout(() => document.getElementById('sign-name-input').focus(), 100);
});
document.getElementById('sign-name-input').addEventListener('click', (e) => e.target.focus());
closeModalBtn.addEventListener('click', () => { profileModal.style.display = 'none'; });
initializeApp();
checkAuth();
async function initializeApp() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter(d =>
      d.kind === 'videoinput' &&
      !d.label.toLowerCase().includes('integrated') &&
      !d.label.toLowerCase().includes('built-in')
    );
    videoSourceSelect.innerHTML = videoDevices.map(d => `<option value="${d.deviceId}">${d.label}</option>`).join('');
  } catch (e) {
    console.error("Error in initializeApp:", e);
  }
}

// Keyboard Shortcuts Listener
document.addEventListener('keydown', (e) => {
  // Don't trigger shortcuts if typing in any input field
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

  const key = e.key.toUpperCase();

  if (key === 'P') {
    isPaused = !isPaused;
    statusText.innerText = isPaused ? "Paused" : (isServiceRunning ? "Connected" : "Stopped");
    statusText.style.color = isPaused ? "#ffa500" : (isServiceRunning ? "#00ff7f" : "#ffffff");
    // Sync pause state with backend
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: isPaused ? 'PAUSE' : 'RESUME' }));
    }
  } else if (ws && ws.readyState === WebSocket.OPEN) {
    if (key === 'A') {
      ws.send(JSON.stringify({ type: 'AUTOCORRECT' }));
    } else if (key === ' ') {
      e.preventDefault(); // Prevent page scroll
      ws.send(JSON.stringify({ type: 'SPACE' }));
    } else if (key === 'ENTER') {
      ws.send(JSON.stringify({ type: 'SUBMIT_SENTENCE' }));
    } else if (key === 'BACKSPACE') {
      ws.send(JSON.stringify({ type: 'BACKSPACE' }));
    }
  }
});