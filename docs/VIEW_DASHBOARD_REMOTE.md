# Watching the robot dashboard from another laptop

The robots, the camera, Qwen and the orchestrator all run on the robot laptop. A second laptop
only opens the dashboard in a browser, to watch and screen-record. It needs nothing installed.

## On the robot laptop (whoever runs the arms)

1. Add `--host 0.0.0.0` to the runner command, for example:
   ```powershell
   .venv\Scripts\python.exe -m hfteama.runner --backend sorter --no-vlm --view --watch --host 0.0.0.0 `
     --arms vato_follower `
     --task tasks/sorter_boxes/boxes.json --poses tasks/sorter_boxes/poses_real.json
   ```
2. After you type `yes`, the runner prints two addresses:
   - `dashboard: http://127.0.0.1:8050/?token=...` is yours. Open exactly that one: the buttons
     (Go, Pause, Stop all, Plug) need its token.
   - `other laptops (view only): http://<IP>:8050/` is the one to send the viewer.
3. If Windows asks whether Python may accept connections, allow it on **Private networks**.
4. Tell the viewer when the run is up.

## On the viewer's laptop

You need the same Wi-Fi as the robot laptop, and Chrome or Edge.

1. Wait for the link. It looks like `http://10.x.x.x:8050/`; the numbers change, so use the
   one you're sent.
2. Open it **after** you're told the run is up. Opened too early, the page shows its built-in
   demo instead of the real robots.
3. Check the badge at the top of the page:
   - **Live**: you're seeing the real run. Start recording.
   - **Demo** or **Offline**: refresh the page (F5).
4. Go full screen (F11) for a cleaner recording.

The buttons won't work for you. That's on purpose: only the laptop next to the robots can start
or stop them.

## If the page doesn't load

1. Check that both laptops are on the same Wi-Fi.
2. Ask whoever runs the arms to check the Windows firewall prompt (step 3 above).
3. If it still fails, the Wi-Fi is probably blocking laptops from talking to each other. Start
   a phone hotspot, connect both laptops to it, and open the link again (the robot laptop's
   address changes: use the new one the runner prints).
