# Demo Recording Instructions

## 🎬 How to Record the Demo GIF

### 1️⃣ **Setup Your Recording**
- Open the Streamlit app at: http://localhost:8504
- Switch to **Demo Mode** in the sidebar
- Click **▶ Start** to begin the demo engine
- Watch for chart updates and logs

### 2️⃣ **Recording Steps (Windows)**

**Option A: Xbox Game Bar (Built-in)**
1. Press `Win + Alt + R` to start recording
2. If not working, press `Win + G` to open Game Bar first
3. Click the record button or press `Win + Alt + R`
4. Record for 15-20 seconds showing:
   - Chart updating with price data
   - Logs updating in real-time
   - Starting/stopping the bot
5. Press `Win + Alt + R` again to stop

**Option B: OBS Studio (Professional)**
1. Download OBS Studio (free)
2. Create a new scene with Window Capture
3. Select your browser window
4. Start recording
5. Stop after 15-20 seconds

### 3️⃣ **Convert to GIF**
1. Go to https://ezgif.com/video-to-gif
2. Upload your recorded video
3. Trim to 15-20 seconds
4. Resize width to ~800px
5. Save as `demo.gif`

### 4️⃣ **What to Show in the Demo**
- ✅ Streamlit dashboard interface
- ✅ Demo mode selected in sidebar
- ✅ Start button click → engine starts
- ✅ Chart updating with live price data
- ✅ Logs showing trading activity
- ✅ Stop button to end demo

### 5️⃣ **Add to Repository**
1. Save the final `demo.gif` in this `/docs/` folder
2. The README.md is already updated to show it
3. Commit and push:
   ```bash
   git add docs/demo.gif
   git commit -m "Add demo GIF for README"
   git push
   ```

## 📝 Current Status
- ✅ Streamlit app running on http://localhost:8504
- ✅ README.md updated with demo section
- ⏳ **Waiting for demo.gif recording**

Replace this file with the actual `demo.gif` once recorded!
