# lookat

Webcam attention detection. The screen shows one thing when nobody is looking
at it, and something else the moment someone does. Same code on a Windows or
Linux notebook and on a Raspberry Pi.

```
webcam ──► face mesh (478 points, incl. irises)
             │
             ├─► head pose      (solvePnP on 6 stable landmarks)
             ├─► iris offset    (pupil position inside the eye opening)
             └─► eyes open?     (blink blendshape / eye aspect ratio)
                     │
                     ▼
            gaze direction vs. the screen cone  ──►  score 0..1
                     │
              smoothing + hysteresis + dwell time
                     │
                     ▼
            "looking" / "away"  ──►  fullscreen scene swap (+ optional shell hooks)
```

It answers **"is someone looking at this screen?"**, not "which pixel are they
looking at". Screen-point gaze needs a per-user calibration routine and gives
you roughly a fist-sized error at desk distance — presence/attention is the
part that is robust enough to drive a display. See
[Going further](#going-further) if you need the gaze point too.

## Quick start on Windows

WSL cannot see your webcam (no `/dev/video*`, the WSL kernel has no UVC
driver), so run this on Windows directly. In **PowerShell**, from this folder:

```powershell
# one-time: installs Python if needed, creates .venv, downloads the model
powershell -ExecutionPolicy Bypass -File scripts\install_windows.ps1
```

If it installs Python, close and reopen PowerShell and run it once more. Then:

```powershell
.\.venv\Scripts\python.exe run.py --list-cameras     # find your camera index
.\.venv\Scripts\python.exe run.py --windowed --debug # see what it sees
.\.venv\Scripts\python.exe run.py                    # fullscreen, for real
```

`run.bat` is a shortcut for the same thing: `run.bat --windowed --debug`.

Press **m** for the settings menu, **d** for the debug overlay, **f** to
toggle fullscreen, **q** or **Esc** to quit.

## Quick start on a Mac

Apple Silicon is the smooth path. On an Intel Mac see the note below.

```bash
git clone https://github.com/gitmick/lookat.git ~/lookat
cd ~/lookat
./scripts/install_mac.sh
```

The script finds or installs Python 3.10+, builds a `.venv`, installs the
package, puts a `lookat` launcher in `~/.local/bin`, and downloads the face
model. Then:

```bash
lookat --version              # where config, images and data live
lookat --windowed --debug     # check it sees you
lookat                        # fullscreen, for real
```

**macOS will ask for camera permission the first time.** If you are never
asked, or the picture is black, enable your terminal under *System Settings >
Privacy & Security > Camera*. A denied camera looks exactly like a missing
one, so lookat prints that hint when it gets no frames.

**Intel Macs:** MediaPipe stopped publishing Intel wheels after 0.10.21, so
`pyproject.toml` pins that release on `x86_64` Darwin. It works, but it is an
older engine than Apple Silicon gets, and OpenCV needs macOS 14+.

### Updating

```bash
lookat --check-update     # is there anything new?
lookat --update           # pull it and reinstall
```

or press **m** and choose **Update**. It checks in the background at startup
and shows what it found in the menu.

The repository is public, so cloning and updating need no credentials. (If
you ever make it private again, run `gh auth login && gh auth setup-git` on
that Mac, or switch the remote to SSH.)

Updates can never clobber local work: your config, calibration, pictures and
enrolled faces all live **outside** the checkout (see `lookat --version`), and
`--update` refuses to run if the checkout has uncommitted changes.

## Quick start on a Raspberry Pi

Pi 4 or Pi 5, 64-bit Raspberry Pi OS (Bookworm). USB webcam or CSI camera
module both work.

```bash
git clone <this folder> ~/lookat && cd ~/lookat
./scripts/install_pi.sh
./.venv/bin/python run.py --windowed --debug
```

Autostart on boot:

```bash
sudo cp scripts/lookat.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now lookat
journalctl -u lookat -f
```

The unit file has notes for running on a bare console without a desktop
(`SDL_VIDEODRIVER=kmsdrm`).

## Quick start on a Linux notebook

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./scripts/fetch_model.sh
./.venv/bin/python run.py --windowed --debug
```

## Where your data lives

Nothing personal is kept in the checkout, so updates are always safe:

```
lookat --version
```

prints the config file, the images folder and the data directory. On a fresh
install that is `~/Library/Application Support/lookat` on macOS,
`%APPDATA%\lookat` on Windows and `~/.config/lookat` on Linux. If a
`config.yaml` already sits next to the code in a git clone, that one wins, so
an existing setup keeps working untouched.

`people.json` holds enrolled faces and is never committed.

## The menu

Press **m** while it is running. Up/Down to move, Enter to act, Esc to close.
Everything you normally need is there, so you rarely have to touch the config
file or the command line:

```
 lookat  -  settings

   Calibrate gaze                              +29 / +2
   Learn a new person                     anna, ben
   Remove a person                                    2
   Face recognition                                  on
   Camera                                      device 0
   Screen mode                                   images
   Fullscreen                                       off
   Save settings
   Update                                    up to date
   Quit
```

- **Calibrate gaze** — look at the screen for 8 seconds; it works out where
  the screen is relative to the camera and applies it immediately.
- **Learn a new person** — type a name, then look at the camera for 10
  seconds. The person is saved *and* given a screen of their own straight
  away, so you can see it working without editing anything.
- **Remove a person** — pick from a list, confirm, and their face data is
  deleted from `people.json` immediately. You pick from a list rather than
  typing, so a typo cannot delete the wrong person.
- **Camera** — scans for cameras in the background and cycles through the ones
  that work. Reverts if the new one will not open.
- **Screen mode** — switch between the text screens and your pictures.
- **Update** — pull the newest version and reinstall.
- **Fullscreen** — toggles, and tells you if the display refuses.
- **Save settings** — writes camera, fullscreen, recognition and the
  calibration into `config.yaml`, keeping your comments.

Calibration and enrolment run on the camera that is *already open* — most
webcams cannot be opened twice, so they could not work any other way. Changes
apply live; only **Save settings** makes them permanent.

## Two screen modes: text and pictures

`display.mode` decides what is drawn. Toggle it live from the menu under
**Screen mode**.

- **`text`** — the built-in coloured screens with a headline. Good for
  setting up and testing, because it shows at a glance which state you are in.
- **`images`** — your own pictures, full screen, nothing else.

Pictures go in the images folder (`lookat --version` prints the path; it is
created for you with a README inside). Files are found by name, in any of
`.png .jpg .jpeg .webp .bmp`:

```
images/
  idle.jpg          shown when nobody is looking
  attentive.jpg     shown when someone is looking
  person-anna.jpg   optional: shown when "anna" is recognised
```

A recognised person with no picture of their own falls back to
`attentive`. If a picture is missing entirely, the screen says so and names
the file it wanted rather than failing silently.

`display.images.fit` is `cover` (fill the screen, cropping if the aspect
ratio differs) or `contain` (show all of the picture, with bars).

## Changing what is shown

Everything lives in `config.yaml` under `display.scenes`. Two scenes, `idle`
and `attentive`, crossfaded into each other:

```yaml
display:
  scenes:
    idle:
      background: "#0d1117"
      text: "…"
      subtext: "nobody is watching"
      text_color: "#3d4551"
      image: null            # any image path; scaled to cover the screen
    attentive:
      background: "#7c1d3f"
      text: "Hello."
      subtext: "I noticed you"
      text_color: "#ffe9f0"
      image: posters/hello.jpg
```

Anything the built-in display cannot do, do with **hooks** — shell commands
fired on each state change:

```yaml
hooks:
  on_attentive: "mpv --fs --loop /home/pi/clip.mp4"
  on_idle: "pkill mpv"
  debounce_seconds: 2.0
```

Run with `--headless` to skip the window entirely and use only the hooks, e.g.
when the Pi is driving a kiosk browser, a relay or an MQTT topic.

## Tuning

Defaults work at a desk, roughly 0.4–1.2 m from the camera. To tune for your
actual setup:

Easiest is the in-app menu: press **m**, choose **Calibrate gaze**, then
**Save settings**. From the command line:

```bash
python run.py --auto-calibrate --write     # look at the screen for 8s
```

It counts down, samples while you look at the screen, and writes the offsets
and tolerances straight into `config.yaml` (comments and layout preserved).
Drop `--write` to only print them. No window required, so it also works over
SSH on a headless Pi.

For more control there is an interactive version:

```bash
python run.py --calibrate
```

Look straight at the screen and press **SPACE** a dozen times, moving your head
a little between presses. Then look away — at the wall, at your phone, past the
screen — and press **a** a dozen times. Press **s** and it prints a YAML block
to paste into `config.yaml`.

Both write `gaze.yaw_offset_deg` / `pitch_offset_deg`, which encode **where the
screen is relative to the camera**. They are only valid for that physical
arrangement: move the camera or your chair and you must calibrate again. If the
offsets come out larger than about 25°, aim the camera at your face instead —
software cannot recover the range you lose when people drift out of frame.

The knobs that matter, in order:

| Setting | Effect |
| --- | --- |
| `gaze.yaw_tolerance_deg` / `pitch_tolerance_deg` | How far off-axis still counts as looking. The single most important pair. |
| `gaze.pitch_offset_deg` | Where the screen sits relative to the camera. Camera on top of a monitor → people look slightly **down** → set about `-8`. |
| `gaze.eye_gain_yaw_deg` | How much the pupils count vs. the head. `0` = head direction only (very stable, ignores side-eye). |
| `attention.enter_frames` / `exit_frames` | Trigger-happiness. Raise to make it calmer. |
| `attention.min_state_seconds` | Hard floor on how fast the screen may flip. |
| `gaze.horizontal_fov_deg` | Your camera's field of view. Wrong values skew all the angles. Most webcams are 55–78°. |

Override anything without editing the file:

```bash
python run.py --set gaze.yaw_tolerance_deg=30 --set attention.enter_frames=5
```

### Reading the debug overlay

`--debug` shows the camera with the landmarks, the gaze arrow and:

- **head yaw/pitch** – where the face points. `yaw > 0` = towards the right of
  the image, `pitch > 0` = upwards.
- **gaze yaw/pitch** – head pose plus the iris deflection.
- **dev yaw/pitch** – the same, but relative to the screen, after the
  position compensation and your offsets. **This is what gets thresholded.**
  It should sit near `0, 0` when you look at the screen.
- **score** – 0..1 before smoothing.

If `dev` is not near zero while you look straight at the screen, put those
numbers into `gaze.yaw_offset_deg` / `pitch_offset_deg` (or just run
`--calibrate`).

## How it decides

1. **Head pose.** Six stable landmarks (nose, chin, both outer eye corners,
   both mouth corners) are matched against a generic 3D head with `solvePnP`.
   The rotation gives the direction the face points, in degrees off the camera
   axis. Verified to 0.0° against synthetic projections in
   `tests/test_geometry.py`.
2. **Iris offset.** The pupil's position inside the eye opening adds up to
   `eye_gain_yaw_deg` on top, so glancing sideways without moving your head is
   caught too.
3. **Position compensation.** Somebody standing at the edge of the frame has to
   turn their head to look at the camera. That expected angle is subtracted, so
   they are not misread as looking away.
4. **Eyes open.** Closed eyes never count as attention.
5. **Score.** 1.0 inside the tolerance cone, linear falloff over
   `softness_deg`, 0 outside.
6. **Stability.** An EMA, then separate enter/exit thresholds, then
   consecutive-frame counts, then a minimum dwell time. A blink or one bad
   frame cannot flip the screen.

## Telling specific people apart

Optional, off by default. It uses OpenCV's built-in SFace recogniser, so there
is **no extra Python dependency** — only a 37 MB model, fetched on first use.

Press **m** and pick **Learn a new person**, or use the command line:

```bash
python run.py --enroll alice                    # 10s from the webcam
python run.py --enroll bob --from photos/bob*.jpg
python run.py --people                          # who is enrolled
python run.py --forget bob
```

Then switch it on in `config.yaml` and give each person their own scene:

```yaml
identity:
  enabled: true

display:
  scenes:
    person:
      alice:
        background: "#1d3557"
        text: "Hi Alice"
      bob:
        background: "#5c2018"
        text: "Hello Bob"
```

A per-person scene inherits anything it does not define from `attentive`, so
you usually only write the two or three lines that differ. Somebody who is
looking but *not* recognised gets the plain `attentive` scene. Hooks can use
the name too:

```yaml
hooks:
  on_attentive: "/usr/local/bin/greet {person}"   # empty string if unknown
```

### How reliable is it

Each face becomes a 128-number template; recognition is a cosine similarity
against the enrolled templates. Measured on sample photos: **same person
0.74–0.99, different people 0.07–0.15**, so the 0.40 default threshold has a
wide margin. Two guards prevent confident mistakes:

- `identity.threshold` — the best match must clear this at all.
- `identity.margin` — it must also beat the runner-up by this much, so two
  similar-looking people produce "unknown" rather than a coin flip.
- On top of that, a majority vote over `vote_frames` recognitions means one
  bad frame cannot rename the person on screen.

`--enroll` refuses frames where the face is too small, too turned, or the eyes
are closed, and keeps a spread of distinct templates rather than N copies of
one pose. Enrol in the light you will actually use. If you enrol somebody who
already resembles an enrolled person, it warns you.

It will not reliably tell identical twins apart, and it is not an
authentication mechanism — a photo on a phone will fool it.

## Performance

The detector runs in its own thread at `detector.detection_fps` (default 12);
the display always renders at `display.fps`, so the screen stays smooth
regardless.

| Machine | ms/frame | Practical detection rate |
| --- | --- | --- |
| x86-64 laptop CPU | ~14 | 12 fps at ~15% of one core |
| Raspberry Pi 5 | ~25–40 | 12 fps comfortably |
| Raspberry Pi 4 | ~60–90 | set `detection_fps: 8` and `camera.width: 480` |
| Raspberry Pi Zero 2 W | very slow | use `detector.backend: haar`, expect presence-only |

If the Pi struggles: lower `camera.width`/`height` first (it costs the most),
then `detector.detection_fps`. The display stays at 30 fps either way.

Face recognition costs about as much again as the landmarks (measured: 14.4 ms
for landmarks + pose, 14.1 ms for recognition on x86-64). It only runs every
`identity.recognize_every` frames — at the default of 3 that is 19 ms per
frame instead of 28 ms. Raise it on a Pi.

## Troubleshooting

**"could not open camera"** — `camera.device: null` makes it pick one
automatically, and it falls back to scanning if the configured index dies.
Otherwise run `python run.py --list-cameras`, or press **m** and cycle
**Camera**. On Windows, close Teams/Zoom/the Camera app first; they lock the
device.

**Black window on a Pi with no desktop** — set `SDL_VIDEODRIVER=kmsdrm` (see
the systemd unit), and make sure the user is in the `video` and `render`
groups.

**It triggers when nobody is there** — lower the tolerances, raise
`attention.enter_frames`, and check `gaze.min_face_width_ratio` so faces far in
the background are ignored.

**It never triggers** — run with `--debug` and look at `dev yaw/pitch`. Those
should sit near `0, 0` when you look at the screen; if they do not, run
`--auto-calibrate --write`. A large `dev yaw` usually means the camera is aimed
off to one side of where people actually sit.

**It flickers** — raise `attention.min_state_seconds` and `exit_frames`, or
lower `attention.smoothing`.

**Poor detection in the dark** — this is a plain RGB camera; it needs light on
the face. For a reliable installation use an IR-illuminated camera (a Pi NoIR
plus an IR LED ring works well and is invisible to the viewer).

## Testing without a webcam

```bash
python tests/test_geometry.py             # pose math, scoring, hysteresis
python tests/test_pipeline.py             # threads, display, hooks, shutdown
python tests/test_pipeline.py face.jpg    # same, but feed a real photo through
python tests/test_identity.py a.jpg b.jpg # recognition, with two people's photos
python tests/test_menu.py face.jpg        # menu, plus in-app calibrate/enrol
```

You can also point the app at a video file instead of a camera:

```bash
python run.py --set camera.device=clip.mp4 --windowed --debug
```

## Layout

```
run.py / run.bat        entry points
config.yaml             all settings
lookat/
  camera.py             threaded grabber: OpenCV (Win/Linux) + Picamera2
  detector.py           face mesh -> head pose + iris -> score  (+ Haar fallback)
  attention.py          smoothing, hysteresis, dwell time
  display.py            fullscreen scenes with a crossfade
  overlay.py            debug drawing
  hooks.py              shell commands on state change
  menu.py               the on-screen settings menu
  paths.py              where config, pictures and face data live
  update.py             `--update` / the Update menu entry
  identity.py           face recognition (SFace) and the people database
  enroll.py             --enroll / --people / --forget
  calibrate.py          interactive and automatic tuning
  app.py                wiring and CLI
scripts/                setup for Windows / Pi, model download, systemd unit
tests/                  runnable without a camera
```

## Going further

**Where on the screen are they looking?** The pieces are already here: 
`Observation.dev_yaw` / `dev_pitch` is the gaze direction relative to the
screen centre. Multiply by the screen geometry and you get a coarse point —
expect 5–15 cm of error at desk distance without per-user calibration. For
something accurate you want a 9-point calibration screen mapping raw
(head pose, iris) features to screen coordinates with a small regression, and
ideally an IR camera.

**Several people at once.** `num_faces` is fixed at 1 in `detector.py`. Raising
it and scoring each face gives you "how many people are watching", and with
recognition, which of them.

## Privacy

Everything runs locally. No frames leave the machine, nothing is recorded, and
no images are written to disk unless you enable the debug overlay (which only
renders to the window). If you deploy this somewhere public, say so on a sign —
in the EU a camera in a public space is personal-data processing even when you
discard every frame.

Face recognition raises the stakes. `people.json` holds face templates, which
under the GDPR are biometric data processed for the purpose of uniquely
identifying someone — Article 9 special-category data, which generally needs
explicit consent. In practice: enrol only people who agreed to it, keep
`people.json` out of version control (it is gitignored) and off shared drives,
and delete it with `--forget` when someone asks. A template cannot be turned
back into a photograph, but it does identify a specific person, so treat the
file as personal data rather than as a hash.

## Licence notes

The MediaPipe face landmark model is downloaded at setup time from Google's
model store (Apache 2.0). The Haar cascades, if the fallback is used, come from
the OpenCV repository (BSD).
