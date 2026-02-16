!macro customInstall
    ; This macro runs after the main application files have been installed.
    
    ; --- Step 1: Install VB-CABLE Silently (No user interaction needed) ---
    DetailPrint "Installing VB-CABLE audio driver in the background..."
    
    ; Copy driver files to temp and install silently
    CreateDirectory "$TEMP\VBCableInstall"
    ${If} ${RunningX64}
        CopyFiles /SILENT "$INSTDIR\resources\app\build-resources\drivers\vbcable\*.*" "$TEMP\VBCableInstall"
        ExecWait '"$TEMP\VBCableInstall\VBCABLE_Setup_x64.exe" -i -h' $0
    ${Else}
        CopyFiles /SILENT "$INSTDIR\resources\app\build-resources\drivers\vbcable_x86\*.*" "$TEMP\VBCableInstall"
        ExecWait '"$TEMP\VBCableInstall\VBCABLE_Setup.exe" -i -h' $0
    ${EndIf}
    RMDir /r "$TEMP\VBCableInstall"
    
    ${If} $0 != 0
        DetailPrint "VB-CABLE installation failed with code $0"
        MessageBox MB_OK|MB_ICONEXCLAMATION "VB-CABLE audio driver installation failed.$\n$\nError code: $0$\n$\nThe application may not work correctly."
    ${Else}
        DetailPrint "VB-CABLE driver installed successfully (silent)"
    ${EndIf}
    
    ; --- Step 2: Install OBS Studio with GUI ---
    MessageBox MB_YESNO|MB_ICONQUESTION \
        "This application requires OBS Studio's Virtual Camera feature.$\n$\nWould you like to install OBS Studio now?$\n$\n(You'll need to complete the installation and enable the Virtual Camera)" \
        IDNO skip_obs
    
    DetailPrint "Launching OBS Studio installer..."
    
    ; Check if OBS is already installed
    ReadRegStr $2 HKLM "SOFTWARE\OBS Studio" ""
    ${If} $2 != ""
        MessageBox MB_YESNO|MB_ICONQUESTION "OBS Studio is already installed at:$\n$2$\n$\nInstall anyway?" IDNO skip_obs
    ${EndIf}
    
    ; Launch OBS installer WITHOUT /S flag (shows GUI)
    ExecWait '"$INSTDIR\resources\app\build-resources\drivers\OBS-Studio-32.0.2-Windows-x64-Installer.exe"' $1
    
    ${If} $1 != 0
        MessageBox MB_OK|MB_ICONEXCLAMATION "OBS Studio installation may have been cancelled or failed.$\n$\nReturn code: $1$\n$\nYou can install it manually later from obsproject.com"
        Goto skip_obs
    ${EndIf}
    
    ; --- Step 3: Create and Show Setup Instructions ---
    MessageBox MB_OK|MB_ICONINFORMATION \
        "OBS Studio installed successfully!$\n$\n⚠️ IMPORTANT: You must set up the Virtual Camera before using this app.$\n$\nClick OK to see the complete setup guide."
    
    ; Create the QuickStart HTML file
    FileOpen $3 "$INSTDIR\QuickStart.html" w
    FileWrite $3 '<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Sign Language Gestures to Audio - Setup Guide</title><style>*{margin:0;padding:0;box-sizing:border-box}body{font-family:"Segoe UI",Tahoma,Geneva,Verdana,sans-serif;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:#333;padding:20px;line-height:1.6}.container{max-width:900px;margin:0 auto;background:#fff;border-radius:15px;box-shadow:0 20px 60px rgba(0,0,0,.3);overflow:hidden}.header{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:#fff;padding:40px;text-align:center}.header h1{font-size:2.5em;margin-bottom:10px}.header p{font-size:1.2em;opacity:.9}.content{padding:40px}.alert{background:#fff3cd;border-left:5px solid #ffc107;padding:20px;margin:20px 0;border-radius:5px}.alert-danger{background:#f8d7da;border-left-color:#dc3545}.alert-success{background:#d4edda;border-left-color:#28a745}.section{margin:40px 0}.section h2{color:#667eea;font-size:2em;margin-bottom:20px;border-bottom:3px solid #667eea;padding-bottom:10px}.steps{counter-reset:step-counter}.step{background:#f8f9fa;padding:25px;margin:20px 0;border-radius:10px;border-left:5px solid #667eea;position:relative;padding-left:80px}.step::before{counter-increment:step-counter;content:counter(step-counter);position:absolute;left:20px;top:50%;transform:translateY(-50%);background:#667eea;color:#fff;width:45px;height:45px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:1.5em;font-weight:700}.step h3{color:#333;margin-bottom:10px;font-size:1.3em}.step p{color:#666;margin-bottom:10px}.step ul{margin:15px 0 10px 20px}.step li{margin:8px 0;color:#555}.code{background:#2d3436;color:#00ff7f;padding:3px 8px;border-radius:4px;font-family:"Courier New",monospace;font-size:.9em}.highlight{background:#ffeb3b;padding:2px 6px;border-radius:3px;font-weight:700}.daily-use{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:#fff;padding:30px;border-radius:10px;margin:30px 0}.daily-use h3{font-size:1.8em;margin-bottom:15px}.daily-use ol{padding-left:20px;font-size:1.1em}.daily-use li{margin:10px 0}.important-note{background:#e74c3c;color:#fff;padding:15px;border-radius:8px;margin:15px 0;font-weight:700}.footer{background:#2d3436;color:#fff;text-align:center;padding:20px}</style></head><body><div class="container"><div class="header"><h1>&#128245; Sign Language Gestures to Audio</h1><p>Complete Setup Guide</p></div><div class="content"><div class="alert alert-danger"><strong>&#128680; CRITICAL - RESTART REQUIRED!</strong><br><strong>You MUST restart your computer NOW!</strong> The VB-CABLE driver will NOT work until you restart.</div><div class="section"><h2>&#128640; OBS Virtual Camera Setup</h2><p style="font-size:1.1em;margin-bottom:20px">Follow these steps <strong>exactly</strong> to set up OBS:</p><div class="steps"><div class="step"><h3>Open OBS Studio</h3><p>Launch OBS Studio from your Start Menu or Desktop.</p><p class="important-note">&#9888; If you see "Auto-Configuration Wizard", click <span class="highlight">"I want to use the virtual camera only"</span> or "Cancel"</p></div><div class="step"><h3>Add Your Webcam</h3><p>In the <strong>Sources</strong> panel (bottom of OBS):</p><ul><li>Click the <span class="code">+</span> (plus) button</li><li>Select <span class="code">Video Capture Device</span></li><li>Click <span class="code">OK</span> when prompted</li></ul></div><div class="step"><h3>Select Your Physical Webcam</h3><p>In the Properties window:</p><ul><li>In <span class="code">Device</span> dropdown, select your real webcam (NOT "OBS Virtual Camera")</li><li>You should see yourself in the preview</li><li>Click <span class="code">OK</span></li></ul></div><div class="step"><h3>Stretch Video to Fill Screen</h3><p>Make your video fill the black preview area:</p><ul><li>Click the red border around your video to select it</li><li>Drag the corner handles to stretch it</li><li><strong>OR</strong> Right-click → <span class="code">Transform</span> → <span class="code">Fit to Screen</span></li></ul><p class="important-note">&#128161; Hold <strong>Shift</strong> while dragging to keep aspect ratio</p></div><div class="step"><h3>Start Virtual Camera</h3><p>Now start the Virtual Camera:</p><ul><li>In <strong>Controls</strong> panel (bottom-right), click <span class="code">Start Virtual Camera</span></li><li><strong>OR</strong> Menu bar: <span class="code">Tools</span> → <span class="code">Start Virtual Camera</span></li></ul><div class="important-note">&#9989; Button will change to "Stop Virtual Camera" and turn green/blue</div></div><div class="step"><h3>Minimize OBS</h3><p><strong>IMPORTANT:</strong> <span class="highlight">Minimize OBS - DO NOT CLOSE IT!</span></p><p>Virtual Camera only works when OBS is running.</p></div><div class="step"><h3>Launch Sign Language Gestures to Audio</h3><ul><li>Open the Sign Language Gestures to Audio app</li><li>Select <span class="code">OBS Virtual Camera</span> from dropdown</li><li>Click the power button to start</li></ul></div></div></div><div class="daily-use"><h3>&#128260; Daily Use (4 Quick Steps)</h3><ol><li>Open <strong>OBS Studio</strong></li><li>Click <strong>"Start Virtual Camera"</strong></li><li>Minimize OBS</li><li>Launch <strong>Sign Language Gestures to Audio</strong> and click power</li></ol><p style="margin-top:15px">&#9889; Takes less than 10 seconds!</p></div><div class="section" style="background:#f8f9fa;padding:30px;border-radius:10px"><h3 style="color:#e74c3c;margin-bottom:20px">&#10067; Troubleshooting</h3><div style="margin:15px 0;padding:15px;background:#fff;border-radius:5px;border-left:3px solid #e74c3c"><strong style="color:#e74c3c;display:block;margin-bottom:5px">Problem: Cant find + button in Sources</strong><p>Solution: Look at bottom of OBS window for "Sources" panel. If hidden: View → Docks → Sources</p></div><div style="margin:15px 0;padding:15px;background:#fff;border-radius:5px;border-left:3px solid #e74c3c"><strong style="color:#e74c3c;display:block;margin-bottom:5px">Problem: Webcam not in Device dropdown</strong><p>Solution: Make sure webcam is connected. Close other apps using camera (Zoom, Teams). Try unplugging/replugging webcam.</p></div><div style="margin:15px 0;padding:15px;background:#fff;border-radius:5px;border-left:3px solid #e74c3c"><strong style="color:#e74c3c;display:block;margin-bottom:5px">Problem: Cant find Start Virtual Camera</strong><p>Solution: Look in Controls panel (bottom-right). Or use menu: Tools → Start Virtual Camera</p></div><div style="margin:15px 0;padding:15px;background:#fff;border-radius:5px;border-left:3px solid #e74c3c"><strong style="color:#e74c3c;display:block;margin-bottom:5px">Problem: OBS Virtual Camera not in app dropdown</strong><p>Solution: 1) Make sure Virtual Camera is STARTED in OBS. 2) Close and reopen Sign Language Gestures to Audio app. 3) If still missing, restart computer.</p></div><div style="margin:15px 0;padding:15px;background:#fff;border-radius:5px;border-left:3px solid #e74c3c"><strong style="color:#e74c3c;display:block;margin-bottom:5px">Problem: No audio output</strong><p>Solution: Did you restart after installation? VB-CABLE requires restart. Check Windows Sound Settings for "CABLE Input".</p></div></div><div class="alert alert-success"><strong>&#9989; Youre All Set!</strong><br>After first-time setup, just: Start OBS Virtual Camera → Launch Sign Language Gestures to Audio</div></div><div class="footer"><p>Sign Language Gestures to Audio - Created by Anant Gaur</p><p style="margin-top:10px;opacity:.7">Click the Help button in the app to view this guide anytime</p></div></div></body></html>'
    FileClose $3
    
    ; Open the created HTML file
    ExecShell "open" "$INSTDIR\QuickStart.html"
    
    skip_obs:
    
    ; --- Final Message ---
    MessageBox MB_OK|MB_ICONINFORMATION \
        "Installation complete!$\n$\n⚠️ IMPORTANT REMINDERS:$\n$\n1. Please restart your computer for VB-CABLE driver to work$\n2. Follow the setup guide to enable OBS Virtual Camera$\n3. Click the Help button in the app anytime for instructions$\n$\nEnjoy using ASL to Audio!"
!macroend