"""
Daily themed reel, fully automated:
AI writes a hooky ~25-second script (natural 8-14 word sentences, calm closing line)
→ visuals are allocated by speaking time, one per clip (Cloudflare FLUX images +
trimmed Pixabay clips) → MoneyPrinterTurbo adds voice and captions → ffmpeg adds
background music, an outro (last frame fades to black while the music fades to
silence) and on-screen hook text → YouTube (scheduled publish) + Instagram.

Usage:
  python daily_reel.py              # full run with upload (safe to run several times a day)
  python daily_reel.py --no-upload  # test run, no posting
Rerunning on the same day reuses today's plan, visuals and finished video
(delete work/<date>/ to start the day fresh).
"""
import base64, datetime, json, math, os, pathlib, random, shutil, subprocess, sys, time
from zoneinfo import ZoneInfo
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ---------- setup ----------
ROOT = pathlib.Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
T = json.loads((ROOT / "theme.json").read_text(encoding="utf-8"))
TODAY = datetime.date.today().isoformat()
WORK = ROOT / "work" / TODAY
OUT = ROOT / "output"
HIST = ROOT / "history.json"
IG_TOKEN_FILE = ROOT / "ig_token.json"
CAL_FILE = ROOT / "calibration.json"
for d in (WORK, OUT):
    d.mkdir(parents=True, exist_ok=True)

MPT_API = os.getenv("MPT_API", "http://127.0.0.1:8080/api/v1")
MPT_BASE = MPT_API.split("/api/")[0]
MPT_DIR = pathlib.Path(os.path.expanduser(os.getenv("MPT_DIR", "~/MoneyPrinterTurbo")))
FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
FFPROBE = shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe"
WORDS_PER_SEC = 2.3          # starting guess at rate 1.0; replaced by measured speed after the first render
BANNED_OPENERS = ("did you know", "imagine", "in this video", "hey", "hi ", "hello", "welcome")
TEXT_MODEL = None
gem = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def log(msg):
    line = f"{datetime.datetime.now():%Y-%m-%d %H:%M} {msg}"
    print(line, flush=True)
    with open(ROOT / "reel.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def notify(msg):
    """macOS notification, so you only need to look when something went wrong."""
    subprocess.run(["osascript", "-e",
                    f'display notification {json.dumps(msg)} with title {json.dumps(T["channel"])}'],
                   capture_output=True)


def clip_seconds():
    return max(1, min(15, round(T["clip_seconds"])))   # MPT accepts whole seconds 1-15


# ---------- 1. AI text (Gemini first, Groq as backup) ----------
def pick_text_model():
    wanted = os.getenv("TEXT_MODEL", "").replace("models/", "")
    names = [m.name.replace("models/", "") for m in gem.models.list()
             if "generateContent" in (m.supported_actions or [])]
    if wanted in names:
        return wanted
    flash = sorted(n for n in names if "flash" in n and not any(
        x in n for x in ("image", "tts", "live", "audio", "preview", "exp", "lite")))
    if not flash:
        raise RuntimeError(f"No usable Flash model found. Available: {names}")
    log(f"TEXT_MODEL '{wanted}' unavailable, using {flash[-1]}")
    return flash[-1]


def llm_json(prompt):
    global TEXT_MODEL
    try:
        TEXT_MODEL = TEXT_MODEL or pick_text_model()
    except Exception as e:
        log(f"Gemini model lookup failed: {str(e)[:150]}")
    if TEXT_MODEL:
        for attempt in range(3):
            try:
                r = gem.models.generate_content(
                    model=TEXT_MODEL, contents=prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json"))
                return json.loads(r.text)
            except Exception as e:
                msg = str(e)
                log(f"Gemini attempt {attempt + 1} failed: {msg[:150]}")
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    break
                time.sleep(15 * (attempt + 1))
    if os.getenv("GROQ_API_KEY"):
        log("Using Groq instead")
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions", timeout=120,
            headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"},
            json={"model": os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
                  "messages": [{"role": "user", "content": prompt + "\nReply with JSON only."}],
                  "response_format": {"type": "json_object"}})
        if not r.ok:
            raise RuntimeError(f"Groq failed ({r.status_code}): {r.text[:500]}")
        return json.loads(r.json()["choices"][0]["message"]["content"])
    raise RuntimeError("No AI available: Gemini failed and no GROQ_API_KEY set")


# ---------- 1b. Script length & pacing ----------
def words_per_sec():
    """Real speaking speed (voice_rate included), learned from earlier renders."""
    try:
        cal = json.loads(CAL_FILE.read_text())
        if cal.get("voice_rate") == T["voice_rate"]:
            return cal["wps"]
    except (OSError, ValueError, KeyError):
        pass
    return WORDS_PER_SEC * T["voice_rate"]


def media_seconds(path):
    r = subprocess.run([FFPROBE, "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=nw=1:nk=1", str(path)], capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return None


def calibrate(video, narration):
    """Measure how fast the voice really spoke so the next script hits the target length."""
    secs = media_seconds(video)
    if not secs:
        log("Could not measure video length (is ffprobe installed?)")
        return
    measured = len(narration.split()) / secs
    wps = round(0.6 * measured + 0.4 * words_per_sec(), 3)
    CAL_FILE.write_text(json.dumps({"wps": wps, "voice_rate": T["voice_rate"]}))
    log(f"Length {secs:.1f}s + {T.get('outro_seconds', 1.5)}s outro (target {T['target_seconds']}s) | "
        f"voice {measured:.2f} words/s, next run uses {wps}")


def script_problems(ep, lo, hi):
    sents = ep.get("sentences") or []
    if not sents or not all(isinstance(s, dict) and s.get("text") for s in sents):
        return ['Return "sentences" as a list of objects with "text" and "visuals".']
    p = []
    total = sum(len(s["text"].split()) for s in sents)
    if total < lo:
        p.append(f"The script has only {total} words but needs {lo}-{hi}. Add a specific number, "
                 "comparison or consequence to sentences; no filler.")
    elif total > hi:
        p.append(f"The script has {total} words but must have {lo}-{hi}. Cut the weakest words.")
    long = [s["text"] for s in sents if len(s["text"].split()) > 18]
    if long:
        p.append(f"Split these sentences, they are over 18 words: {long}")
    if sum(len(s["text"].split()) < 6 for s in sents) > 2:
        p.append("Too many very short sentences; most should be 8-14 words.")
    if sents[0]["text"].strip().lower().startswith(BANNED_OPENERS):
        p.append("Sentence 1 starts with a banned opener; start with the shocking claim itself.")
    last = sents[-1]["text"].strip()
    if not last.endswith("."):
        p.append(f'The last sentence must be a calm closing statement ending with a period, '
                 f'not a question or exclamation: "{last}"')
    return p


def to_shots(sentences, per_shot):
    """One visual per clip_seconds of speech, allocated cumulatively so timing doesn't drift."""
    shots, spoken, slots = [], 0, 0
    for s in sentences:
        spoken += len(s["text"].split())
        need = max(1, round(spoken / per_shot) - slots)
        vis = [v for v in (s.get("visuals") or []) if isinstance(v, dict) and v.get("visual")][:need]
        if not vis:
            vis = [{"type": "ai", "visual": s["text"]}]
        while len(vis) < need:
            vis.append({"type": "ai", "visual": f"{vis[-1]['visual']}, different camera angle, closer view"})
        shots += vis
        slots += need
    # one spare visual so a slightly long voiceover never loops back to the hook image
    shots.append({"type": "ai", "visual": f"{shots[-1]['visual']}, wide shot, calm fading light"})
    return shots


def plan_episode(history):
    fmt = T["formats"][len(history) % len(T["formats"])]
    recent = [h["title"] for h in history[-120:]]
    wps = words_per_sec()
    words = round(T["target_seconds"] * wps)
    lo, hi = round(words * 0.9), round(words * 1.1)
    n_sent = max(5, round(words / 11))
    per_shot = clip_seconds() * wps
    mascot = (f'\nWhere it fits, include this recurring mascot in "ai" visuals: {T["mascot"]}.'
              if T.get("mascot") else "")
    prompt = f"""You are the head writer of "{T['channel']}", a short-video channel whose videos get millions of views.
Niche: {T['niche']}. Audience: {T['audience']}. Tone: {T['tone']}.
Today's format: {fmt}
Already covered (never repeat or closely resemble): {recent}

STEP 1 (think silently, do not output): brainstorm 8 topics for today's format.
Score each for SURPRISE (would a smart 20-year-old say "wait, what?!"), VISUAL potential
and SHAREABILITY. Pick the single best one.

STEP 2: write the voiceover as exactly {n_sent} sentences, {words} words in total
(it MUST be between {lo} and {hi} words; count them before answering).
- Most sentences 8-14 words. Vary the rhythm: a short 4-6 word punch after a long sentence
  is fine, but never more than two short ones, and never over 18 words.
- Write for the ear: natural spoken English, contractions, one idea per sentence,
  no lists, no parentheses, no semicolons.
- Sentence 1 = the hook, spoken in the first 2 seconds: a shocking claim, a specific number,
  or a direct "you" scenario. NEVER start with "Did you know", "Imagine", "In this video" or a greeting.
- Sentence 2 immediately raises the stakes or contradicts what people expect.
- Open a loop early (promise something crazier that is revealed near the end).
- Escalate: every sentence more surprising than the one before. No filler, no moralizing,
  no "follow", "subscribe" or "comment below".
- Use specific numbers, comparisons to everyday things, and "you" language.
- Second-to-last sentence: the biggest payoff/twist of the video.
- Last sentence: a calm closing line that clearly sounds like the end. A complete statement
  with a falling tone, 7-12 words, ending with a period. It sums up or puts the twist in
  perspective (it may echo a word from the hook). Not a question, not an exclamation,
  not a cliffhanger.
- Only well-established facts; when unsure of an exact number, use a safe approximation ("about", "more than").

STEP 3: visuals. Each visual is on screen for about {clip_seconds()} seconds, so give each sentence
one visual per ~{per_shot:.0f} words of that sentence (about {round(per_shot)} words = 1 visual,
{round(per_shot * 2)} words = 2 visuals, {round(per_shot * 3)} words = 3). Visuals inside one sentence show
successive moments of exactly what is being said, literally, not symbolically.
- "type": "ai" or "stock" (about {int(T['ai_share'] * 100)}% "ai"; the first 2 visuals MUST be "ai").
- "visual" for "ai": one sentence = subject + action + setting + camera angle (extreme close-up,
  aerial, low angle, wide shot) + lighting/mood. Make the first 2 visuals the most dramatic of the video.
  The final visual should be a calm, wide, slightly darker shot that suits a fade to black.
  Vertical composition, main subject centered, NO text, letters or numbers in the image.
- "visual" for "stock": a 2-4 word concrete search term for real footage.
Never show: {T['banned_visuals']}.{mascot}

Return JSON:
{{"title": "max 60 characters, curiosity-driven, honest",
  "hook_text": "max 6 words shown big on screen during the first {T['hook_seconds']} seconds; punchy, complements the spoken hook",
  "sentences": [{{"text": "...", "visuals": [{{"type": "ai", "visual": "..."}}]}}],
  "description": "2 sentences for the post"}}"""

    best, feedback = None, ""
    for attempt in range(3):
        ep = llm_json(prompt + feedback)
        problems = script_problems(ep, lo, hi)
        total = sum(len(s.get("text", "").split())
                    for s in ep.get("sentences") or [] if isinstance(s, dict))
        if best is None or abs(total - words) < abs(best[1] - words):
            best = (ep, total)
        if not problems:
            break
        log(f"Script attempt {attempt + 1} rejected: {'; '.join(problems)[:300]}")
        feedback = ("\n\nYOUR PREVIOUS ATTEMPT IS BELOW. Keep its topic and best lines, "
                    "but fix these problems:\n- " + "\n- ".join(problems) + "\n"
                    + json.dumps(ep, ensure_ascii=False))

    ep, total = best
    sentences = [s for s in ep.pop("sentences", []) if isinstance(s, dict) and s.get("text")]
    if not sentences:
        raise RuntimeError("AI returned no usable sentences")
    ep["narration"] = " ".join(s["text"].strip() for s in sentences)
    ep["shots"] = to_shots(sentences, per_shot)
    ep["format"] = fmt
    log(f"Script: {total} words, {len(sentences)} sentences, "
        f"{len(ep['shots'])} visuals, ≈{total / wps:.0f}s")
    return ep


# ---------- 2. Visuals ----------
def ai_image(visual, path):
    url = (f"https://api.cloudflare.com/client/v4/accounts/{os.environ['CF_ACCOUNT_ID']}"
           "/ai/run/@cf/black-forest-labs/flux-1-schnell")
    for _ in range(2):
        try:
            r = requests.post(url, timeout=120,
                              headers={"Authorization": f"Bearer {os.environ['CF_API_TOKEN']}"},
                              json={"prompt": f"{visual}. {T['art_style']}", "steps": 4})
            r.raise_for_status()
            path.write_bytes(base64.b64decode(r.json()["result"]["image"]))
            return path
        except Exception as e:
            log(f"FLUX failed ({str(e)[:150]}), retrying")
            time.sleep(10)
    return None


def stock_clip(term, path, used):
    """Download one Pixabay video for the search term (vertical preferred)."""
    try:
        r = requests.get("https://pixabay.com/api/videos/", timeout=30, params={
            "key": os.environ["PIXABAY_API_KEY"], "q": term[:100],
            "safesearch": "true", "per_page": 20})
        r.raise_for_status()
        candidates = []
        for h in r.json().get("hits", []):
            if h["id"] in used or h.get("duration", 0) < 3:
                continue
            for size in ("medium", "large"):
                v = h.get("videos", {}).get(size) or {}
                if v.get("url") and v.get("height", 0) >= 720:
                    candidates.append((v["height"] <= v["width"], h["id"], v["url"]))
                    break
        candidates.sort(key=lambda c: c[0])
        if candidates:
            _, vid, url = candidates[0]
            path.write_bytes(requests.get(url, timeout=180).content)
            used.add(vid)
            return path
    except Exception as e:
        log(f"Pixabay failed for '{term}': {str(e)[:150]}")
    return None


def trim_clip(src, dst, seconds):
    """Cut a stock clip to exactly one shot, so MPT uses it once and stays in sync."""
    r = subprocess.run([FFMPEG, "-y", "-ss", "0.5", "-i", str(src), "-t", str(seconds), "-an",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", str(dst)],
                       capture_output=True, text=True)
    src.unlink(missing_ok=True)
    if r.returncode != 0 or not dst.exists():
        log(f"Trim failed: {r.stderr[-200:]}")
        return None
    return dst


def make_visuals(shots):
    files, used, clip = [], set(), clip_seconds()
    for i, s in enumerate(shots, 1):
        existing = next((p for p in (WORK / f"{i:02d}.mp4", WORK / f"{i:02d}.png") if p.exists()), None)
        if existing:
            files.append(existing)
            continue
        log(f"Shot {i}/{len(shots)}: {s.get('type')} – {s['visual'][:60]}")
        p = None
        if s.get("type") == "stock" and i > 2:          # first 2 shots are always AI
            raw = stock_clip(s["visual"], WORK / f"raw_{i:02d}.mp4", used)
            if raw:
                p = trim_clip(raw, WORK / f"{i:02d}.mp4", clip)
        if p is None:
            p = ai_image(s["visual"], WORK / f"{i:02d}.png")
        if p is None and files:                         # keep the count so timing stays in sync
            log(f"Shot {i} failed, reusing previous visual")
            dup = WORK / f"{i:02d}{files[-1].suffix}"
            shutil.copy(files[-1], dup)
            p = dup
        if p:
            files.append(p)
    if len(files) < 5:
        raise RuntimeError(f"Only {len(files)} visuals created")
    log(f"Visuals: {len(files)} files")
    return files


# ---------- 3. MoneyPrinterTurbo ----------
def mpt_up():
    try:
        r = requests.get(f"{MPT_BASE}/openapi.json", timeout=5)
        return r.status_code == 200 and "paths" in r.json()
    except (requests.RequestException, ValueError):
        return False

def find_uv():
    """uv lives in different places depending on how it was installed; launchd's PATH is minimal."""
    for p in (os.getenv("UV_PATH"), shutil.which("uv"),
              os.path.expanduser("~/.local/bin/uv"), os.path.expanduser("~/.cargo/bin/uv"),
              "/opt/homebrew/bin/uv", "/usr/local/bin/uv"):
        if p and os.path.exists(p):
            return p
    raise RuntimeError("uv not found: run 'which uv' in Terminal and add UV_PATH=<that path> to .env")

def ensure_mpt():
    if mpt_up():
        return
    log("Starting MoneyPrinterTurbo API ...")
    uv = find_uv()
    logf = open(ROOT / "mpt_api.log", "a")
    subprocess.Popen([uv, "run", "python", "main.py"], cwd=MPT_DIR,
                     stdout=logf, stderr=subprocess.STDOUT)
    for _ in range(36):
        time.sleep(5)
        if mpt_up():
            log("MoneyPrinterTurbo API is up")
            return
    raise RuntimeError("MPT API did not start (see mpt_api.log)")


def pick_song():
    """Random background track from MPT's song folder (we mix it ourselves so it can fade out)."""
    songs = sorted((MPT_DIR / "resource" / "songs").glob("*.mp3"))
    if not songs:
        log("No songs found in MoneyPrinterTurbo/resource/songs, letting MPT add music")
        return None
    return random.choice(songs)


def render(ep, files, own_music=False):
    mat_dir = MPT_DIR / "storage" / "local_videos"     # the only folder MPT accepts
    mat_dir.mkdir(parents=True, exist_ok=True)
    materials, copied = [], []
    for f in files:
        dst = mat_dir / f"reel-{TODAY}-{f.name}"
        shutil.copy(f, dst)
        copied.append(dst)
        materials.append({"provider": "local", "url": str(dst), "duration": 0})

    payload = {
        "video_subject": ep["title"],
        "video_script": ep["narration"],
        "video_source": "local",
        "video_materials": materials,
        "video_aspect": "9:16",
        "video_fit_mode": "cover",
        "image_motion": "zoom",
        "video_concat_mode": "sequential",
        "video_clip_duration": clip_seconds(),
        "video_count": 1,
        "voice_name": T["voice"],
        "voice_rate": T["voice_rate"],
        "bgm_type": "" if own_music else "random",       # "" = voice only; music added in add_outro
        "bgm_volume": 0.0 if own_music else T.get("bgm_volume", 0.15),
        "subtitle_enabled": True,
        "subtitle_display_mode": "word_by_word",
        "subtitle_animation": "pop_spring",
        "subtitle_position": "custom",
        "custom_position": 65.0,
        "font_size": 75,
        "text_fore_color": "#FFFFFF",
        "stroke_color": "#000000",
        "stroke_width": 3.0,
    }
    r = requests.post(f"{MPT_API}/videos", json=payload, timeout=60)
    if not r.ok:
        raise RuntimeError(f"MPT rejected request ({r.status_code}): {r.text[:1500]}")
    task_id = r.json()["data"]["task_id"]
    log(f"MPT task {task_id} started (voice {T['voice']})")

    try:
        for _ in range(120):
            time.sleep(10)
            t = requests.get(f"{MPT_API}/tasks/{task_id}", timeout=30).json()["data"]
            print(f"   rendering… {t.get('progress', 0)}%", flush=True)
            if t.get("state") == -1:
                raise RuntimeError(f"MPT failed at '{t.get('failed_stage')}': {t.get('error')}")
            if t.get("videos"):
                src, dst = t["videos"][0], OUT / f"{TODAY}.mp4"
                url = src if src.startswith("http") else (MPT_BASE + src if src.startswith("/") else None)
                if url:
                    dst.write_bytes(requests.get(url, timeout=300).content)
                else:
                    shutil.copy(src, dst)
                return dst
        raise RuntimeError("MPT render timed out")
    finally:
        for c in copied:
            c.unlink(missing_ok=True)


# ---------- 3b. Music + outro (fade to black, music fades to silence) ----------
def add_outro(video, song):
    dur = media_seconds(video)
    if not dur:
        log("Could not read video length, skipping outro")
        return video
    tail = float(T.get("outro_seconds", 1.5))
    total = dur + tail
    fade_v = min(1.0, tail)                              # image fade length
    black = min(0.3, max(0.0, tail - fade_v))            # fully black at the very end
    v = (f"[0:v]tpad=stop_mode=clone:stop_duration={tail},"
         f"fade=t=out:st={total - fade_v - black:.2f}:d={fade_v}[v]")
    # tiny fade on the voice track so the last word never ends in a click, then silence
    voice = f"[0:a]afade=t=out:st={max(0, dur - 0.12):.2f}:d=0.12,apad=whole_dur={total:.2f}"
    if song:
        vol = T.get("bgm_volume", 0.2)
        music_fade = tail + 0.3                          # starts just before the voice ends
        fc = (f"{v};{voice}[voice];"
              f"[1:a]volume={vol},afade=t=in:d=0.3,"
              f"afade=t=out:st={total - music_fade:.2f}:d={music_fade:.2f}[m];"
              f"[voice][m]amix=inputs=2:duration=first:normalize=0[a]")
        inputs = ["-i", str(video), "-stream_loop", "-1", "-i", str(song)]
    else:
        fc = f"{v};{voice}[a]"
        inputs = ["-i", str(video)]
    tmp = video.with_name(video.stem + "_outro.mp4")
    r = subprocess.run([FFMPEG, "-y", *inputs, "-filter_complex", fc,
                        "-map", "[v]", "-map", "[a]", "-t", f"{total:.2f}",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                        "-c:a", "aac", "-b:a", "192k", str(tmp)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        log(f"Outro failed (video kept without it): {r.stderr[-400:]}")
        return video
    tmp.replace(video)
    log(f"Outro added: {tail}s fade to black" + (f", music: {song.name}" if song else ""))
    return video


# ---------- 3c. On-screen hook text ----------
def add_hook(video, hook):
    """Big hook text for the first seconds: drawn with Pillow, overlaid with ffmpeg
    (works even when ffmpeg was built without the drawtext filter)."""
    if not hook:
        return video
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        log("Pillow not installed (pip install pillow), skipping hook text")
        return video
    font_path = next((f for f in ("/System/Library/Fonts/Supplemental/Arial Black.ttf",
                                  "/System/Library/Fonts/Supplemental/Arial Bold.ttf")
                      if os.path.exists(f)), None)
    if not font_path:
        log("No bold font found, skipping hook text")
        return video

    lines, cur = [], ""
    for w in hook.upper().split():                     # wrap to ~14 characters per line
        if cur and len(cur) + len(w) + 1 > 14:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)

    r = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height", "-of", "csv=p=0", str(video)],
                       capture_output=True, text=True)
    try:
        vw, vh = map(int, r.stdout.strip().split(",")[:2])
    except ValueError:
        log("Could not read video size, skipping hook text")
        return video

    size = round(vw * 100 / 1080)                      # 100 px at 1080 wide
    stroke = max(4, round(size * 0.09))
    font = ImageFont.truetype(font_path, size)
    ascent, descent = font.getmetrics()
    line_h = ascent + descent + round(size * 0.14)
    img = Image.new("RGBA", (vw, vh), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    y = round(vh * 0.13)
    for line in lines:
        left, _, right, _ = draw.textbbox((0, 0), line, font=font, stroke_width=stroke)
        draw.text(((vw - (right - left)) / 2 - left, y), line, font=font, fill="white",
                  stroke_width=stroke, stroke_fill="black")
        y += line_h
    png = WORK / "hook.png"
    img.save(png)

    tmp = video.with_name(video.stem + "_hook.mp4")
    secs = T.get("hook_seconds", 3)
    r = subprocess.run([FFMPEG, "-y", "-i", str(video), "-i", str(png), "-filter_complex",
                        f"[0:v][1:v]overlay=0:0:enable='lt(t,{secs})'[v]",
                        "-map", "[v]", "-map", "0:a", "-c:a", "copy",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", str(tmp)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        log(f"Hook text failed (video kept without it): {r.stderr[-300:]}")
        return video
    tmp.replace(video)
    log(f"Hook text added: {' / '.join(lines)}")
    return video


# ---------- 4a. YouTube ----------
def youtube_status():
    """Schedule the video for publish_time (in publish_timezone) if that is still ahead today."""
    status = {"selfDeclaredMadeForKids": False,
              "containsSyntheticMedia": T.get("ai_disclosure", True)}
    when = T.get("publish_time")
    if when:
        tz = ZoneInfo(T.get("publish_timezone", "America/New_York"))
        h, m = map(int, when.split(":"))
        now = datetime.datetime.now(tz)
        at = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if at > now + datetime.timedelta(minutes=15):
            status["privacyStatus"] = "private"          # required for scheduled videos
            status["publishAt"] = at.astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            return status, f"scheduled for {at:%Y-%m-%d %H:%M} {tz.key}"
    status["privacyStatus"] = T.get("privacy", "public")
    return status, status["privacyStatus"]


def upload_youtube(video, ep):
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    scopes = ["https://www.googleapis.com/auth/youtube.upload"]
    tok = ROOT / "token.json"
    creds = Credentials.from_authorized_user_file(str(tok), scopes) if tok.exists() else None
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not sys.stdin.isatty():
                raise RuntimeError("YouTube login needed: run 'python daily_reel.py' once in Terminal")
            creds = InstalledAppFlow.from_client_secrets_file(
                str(ROOT / "client_secret.json"), scopes).run_local_server(port=0)
        tok.write_text(creds.to_json())
    yt = build("youtube", "v3", credentials=creds)
    status, label = youtube_status()
    body = {
        "snippet": {"title": f"{ep['title']} #shorts"[:100],
                    "description": f"{ep['description']}\n\n{' '.join(T['hashtags'])}",
                    "categoryId": T["youtube_category"]},
        "status": status,
    }
    res = yt.videos().insert(part="snippet,status", body=body,
                             media_body=MediaFileUpload(str(video), resumable=True)).execute()
    log(f"YouTube ({label}): https://youtube.com/shorts/{res['id']}")


# ---------- 4b. Instagram (optional) ----------
def ig_token():
    if IG_TOKEN_FILE.exists():
        data = json.loads(IG_TOKEN_FILE.read_text())
    else:
        data = {"token": os.getenv("IG_TOKEN", ""), "refreshed": TODAY}
    if not data["token"]:
        return ""
    age = (datetime.date.today() - datetime.date.fromisoformat(data["refreshed"])).days
    if age >= 30:
        r = requests.get("https://graph.instagram.com/refresh_access_token", timeout=30,
                         params={"grant_type": "ig_refresh_token", "access_token": data["token"]})
        if r.ok:
            data = {"token": r.json()["access_token"], "refreshed": TODAY}
            log("Instagram token refreshed")
        else:
            log(f"Instagram token refresh failed: {r.text[:300]}")
    IG_TOKEN_FILE.write_text(json.dumps(data))
    return data["token"]


def upload_instagram(video, ep):
    ig_id = os.getenv("IG_USER_ID")
    if not ig_id:
        return
    token = ig_token()
    if not token:
        return
    G = os.getenv("IG_GRAPH", "https://graph.instagram.com/v23.0")
    caption = f"{ep['title']}\n\n{ep['description']}\n\n{' '.join(T['hashtags'])}"
    c = requests.post(f"{G}/{ig_id}/media", timeout=60, data={
        "media_type": "REELS", "upload_type": "resumable",
        "caption": caption, "access_token": token}).json()
    if "id" not in c or "uri" not in c:
        raise RuntimeError(f"Container creation failed: {c}")
    with open(video, "rb") as f:
        requests.post(c["uri"], data=f, timeout=600, headers={
            "Authorization": f"OAuth {token}", "offset": "0",
            "file_size": str(os.path.getsize(video))}).raise_for_status()
    for _ in range(60):
        s = requests.get(f"{G}/{c['id']}", timeout=30,
                         params={"fields": "status_code", "access_token": token}).json()
        if s.get("status_code") == "FINISHED":
            break
        if s.get("status_code") == "ERROR":
            raise RuntimeError(s)
        time.sleep(10)
    requests.post(f"{G}/{ig_id}/media_publish", timeout=60, data={
        "creation_id": c["id"], "access_token": token}).raise_for_status()
    log("Instagram: published")


# ---------- housekeeping ----------
def cleanup(work_days=7, output_days=30):
    now = time.time()
    for d in (ROOT / "work").iterdir():
        if d.is_dir() and d.stat().st_mtime < now - work_days * 86400:
            shutil.rmtree(d, ignore_errors=True)
    for f in OUT.glob("*.mp4"):
        if f.stat().st_mtime < now - output_days * 86400:
            f.unlink(missing_ok=True)


# ---------- main ----------
if __name__ == "__main__":
    history = json.loads(HIST.read_text(encoding="utf-8")) if HIST.exists() else []
    uploading = "--no-upload" not in sys.argv
    if uploading and any(h.get("date") == TODAY for h in history):
        log("Already posted today, nothing to do")
        sys.exit(0)
    try:
        plan_file = WORK / "plan.json"
        if plan_file.exists():
            ep = json.loads(plan_file.read_text(encoding="utf-8"))
            log(f"Reusing today's plan: {ep['title']}")
        else:
            ep = plan_episode(history)
            log(f"Episode ({ep['format']}): {ep['title']}  |  hook: {ep.get('hook_text')}")
            plan_file.write_text(json.dumps(ep, indent=2, ensure_ascii=False), encoding="utf-8")

        video = OUT / f"{TODAY}.mp4"
        if ep.get("video_done") and video.exists():
            log(f"Reusing today's finished video: {video}")
        else:
            files = make_visuals(ep["shots"])
            ensure_mpt()
            song = pick_song()
            voices = [T["voice"]] + T.get("fallback_voices", [])
            for attempt in range(4):
                T["voice"] = voices[attempt % len(voices)]
                try:
                    video = render(ep, files, own_music=song is not None)
                    break
                except RuntimeError as e:
                    if "audio" in str(e) and attempt < 3:
                        log(f"Voice failed ({T['voice']}), retrying in 3 minutes with next voice")
                        time.sleep(180)
                    else:
                        raise
            calibrate(video, ep["narration"])            # measure before the outro adds time
            video = add_outro(video, song)
            video = add_hook(video, ep.get("hook_text"))
            ep["video_done"] = True
            plan_file.write_text(json.dumps(ep, indent=2, ensure_ascii=False), encoding="utf-8")
            log(f"Rendered: {video}")

        if not uploading:
            log("Test run: upload skipped")
        else:
            try:
                upload_youtube(video, ep)
            except Exception as e:                       # not marked as posted → next run retries
                raise RuntimeError(f"YouTube upload failed: {str(e)[:300]}")
            try:
                upload_instagram(video, ep)
            except Exception as e:
                log(f"Instagram failed: {str(e)[:300]}")
                notify("Instagram upload failed – see reel.log")
            history.append({"date": TODAY, "title": ep["title"], "format": ep["format"]})
            HIST.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
            notify(f"Posted: {ep['title']}")
        cleanup()
    except Exception as e:
        log(f"FAILED: {e}")
        notify(f"Daily reel failed: {str(e)[:120]}")