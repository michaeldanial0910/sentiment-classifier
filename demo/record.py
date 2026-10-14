"""(Used to make demo/sentiment_demo.mp4.) Drive the real Streamlit app in a visible Chromium on Xvfb and log timestamps of each scene.
Run under:  xvfb-run -s "-screen 0 1920x1080x24" python record.py  (ffmpeg x11grab started by this script)
"""
import json, os, subprocess, sys, time
from playwright.sync_api import sync_playwright

OUT = os.path.dirname(os.path.abspath(__file__))
URL = "http://localhost:8501"
marks = []
T0 = None

def mark(name):
    marks.append({"name": name, "t": round(time.time() - T0, 2)})
    print(f"{marks[-1]['t']:7.2f}  {name}", flush=True)

CURSOR_JS = """
(() => {
  if (window.__cursor) return;
  const c = document.createElement('div');
  c.innerHTML = `<svg width="26" height="26" viewBox="0 0 24 24"><path d="M4 2 L4 19 L8.6 14.8 L11.6 21.4 L14.4 20.2 L11.4 13.6 L17.6 13.4 Z"
     fill="#1C2333" stroke="white" stroke-width="1.6" stroke-linejoin="round"/></svg>`;
  Object.assign(c.style, {position:'fixed', left:'0px', top:'0px', zIndex: 2147483647, pointerEvents:'none',
                          transform:'translate(-4px,-2px)', transition:'transform 60ms'});
  const ring = document.createElement('div');
  Object.assign(ring.style, {position:'fixed', width:'34px', height:'34px', borderRadius:'50%', border:'2px solid #2A6FD6',
     zIndex: 2147483646, pointerEvents:'none', opacity:'0', transform:'translate(-17px,-17px) scale(0.4)',
     transition:'opacity 350ms, transform 350ms'});
  const add = () => { document.body.appendChild(ring); document.body.appendChild(c); };
  if (document.body) add(); else document.addEventListener('DOMContentLoaded', add);
  document.addEventListener('mousemove', e => { c.style.left = e.clientX+'px'; c.style.top = e.clientY+'px';
                                               ring.style.left = e.clientX+'px'; ring.style.top = e.clientY+'px'; }, true);
  document.addEventListener('mousedown', () => { ring.style.transition='none'; ring.style.opacity='0.9';
      ring.style.transform='translate(-17px,-17px) scale(0.4)'; requestAnimationFrame(() => { ring.style.transition='opacity 450ms, transform 450ms';
      ring.style.opacity='0'; ring.style.transform='translate(-17px,-17px) scale(1.3)'; }); }, true);
  window.__cursor = true;
})();
"""

def main():
    global T0
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "x11grab", "-framerate", "30",
                           "-video_size", "1920x1080", "-draw_mouse", "0", "-i", os.environ["DISPLAY"],
                           "-c:v", "libx264", "-preset", "veryfast", "-crf", "14", "-pix_fmt", "yuv420p",
                           f"{OUT}/raw.mp4"], stdin=subprocess.PIPE)
    T0 = time.time()
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path="/opt/pw-browsers/chromium", headless=False,
                              args=["--kiosk", "--window-position=0,0", "--window-size=1920,1080",
                                    "--force-device-scale-factor=1.25", "--hide-scrollbars",
                                    "--disable-infobars", "--no-first-run"])
        ctx = b.new_context(no_viewport=True)
        ctx.add_init_script(CURSOR_JS)
        pg = ctx.new_page()
        pg.goto(URL)
        pg.wait_for_selector("text=Read a guest review", timeout=60000)
        pg.wait_for_selector(".verdict", timeout=60000)
        pg.wait_for_timeout(2500)          # fonts, charts settle
        pg.evaluate(CURSOR_JS)
        m = pg.mouse
        m.move(760, 500)
        mark("ready")
        main_el = "section.stMain, [data-testid='stMain']"

        def glide(locator, steps=28, click=True, dx=0, dy=0):
            box = locator.bounding_box()
            x, y = box["x"] + box["width"] / 2 + dx, box["y"] + box["height"] / 2 + dy
            m.move(x, y, steps=steps)
            pg.wait_for_timeout(180)
            if click:
                m.click(x, y)

        def scroll_to(px, dur_ms=1600):
            pg.evaluate(f"""(() => {{ const el = document.querySelector("{main_el}") || document.scrollingElement;
               const start = el.scrollTop, end = {px}, t0 = performance.now(), d = {dur_ms};
               function step(t) {{ const k = Math.min(1, (t - t0) / d), e = k<.5 ? 2*k*k : 1-Math.pow(-2*k+2,2)/2;
                  el.scrollTop = start + (end - start) * e; if (k < 1) requestAnimationFrame(step); }}
               requestAnimationFrame(step); }})()""")
            pg.wait_for_timeout(dur_ms + 150)

        def settle(sel=None, ms=900):
            if sel:
                pg.wait_for_selector(sel, timeout=60000)
            pg.wait_for_timeout(ms)

        # ---- Try it ----
        mark("try_start")
        pg.wait_for_timeout(900)
        glide(pg.get_by_text("Complaint, Arabic", exact=True))
        settle(ms=1500)
        mark("try_complaint")
        glide(pg.get_by_text("Contrast, English", exact=True))
        settle(ms=1500)
        for attempt in range(8):
            mark(f"pull_click_{attempt}")
            glide(pg.get_by_role("button", name="Pull a random test review"), steps=22)
            settle("text=What all three models said", ms=1500)
            txt = pg.locator(".slip .text").first.inner_text()
            ok = len(txt) < 260
            mark(f"pulled_{attempt}_{'ok' if ok else 'long'}")
            if ok:
                break
        scroll_to(330, 1100)
        pg.wait_for_timeout(1300)
        mark("try_end")

        # ---- Results ----
        glide(pg.get_by_role("link", name="Results"))
        settle("text=Macro-F1 with 95% bootstrap intervals", 300)
        pg.wait_for_selector(".vega-embed", timeout=60000)
        pg.wait_for_timeout(1300)
        mark("results_start")
        scroll_to(250, 1100)
        pg.wait_for_timeout(1200)
        mark("results_forest")
        scroll_to(1040, 1300)
        pg.wait_for_timeout(500)
        glide(pg.get_by_role("radio", name="English").last)
        settle(ms=1300)
        mark("results_cm")
        scroll_to(1520, 1200)
        tv = pg.locator('[data-testid="stSliderThumbValue"]').first.bounding_box()
        tx, ty = tv["x"] + tv["width"] / 2, tv["y"] + tv["height"] + 13
        m.move(tx, ty, steps=22)
        pg.wait_for_timeout(200)
        m.down()
        m.move(tx - 150, ty, steps=36)
        pg.wait_for_timeout(500)
        m.move(tx + 50, ty, steps=36)
        m.up()
        settle(ms=1800)
        mark("results_end")

        # ---- Mistakes ----
        glide(pg.get_by_role("link", name="Mistakes"))
        settle("text=Six that explain the pattern", 300)
        pg.wait_for_selector(".vega-embed", timeout=60000)
        pg.wait_for_timeout(1000)
        mark("mistakes_start")
        scroll_to(560, 1700)
        pg.wait_for_timeout(700)
        scroll_to(1150, 1600)
        pg.wait_for_timeout(700)
        mark("mistakes_end")

        # ---- The data ----
        glide(pg.get_by_role("link", name="The data"))
        settle("text=Arabic reviews are much shorter", 300)
        pg.wait_for_selector(".vega-embed", timeout=60000)
        pg.wait_for_timeout(1000)
        mark("data_start")
        scroll_to(330, 1400)
        pg.wait_for_timeout(1100)
        scroll_to(1050, 1500)
        pg.wait_for_timeout(1100)
        mark("data_end")

        # ---- Batch ----
        glide(pg.get_by_role("link", name="Batch"))
        settle("text=Download sorted reviews", 1300)
        mark("batch_start")
        scroll_to(300, 1300)
        pg.wait_for_timeout(1000)
        glide(pg.get_by_role("button", name="Download sorted reviews (CSV)"), click=False)
        pg.wait_for_timeout(1400)
        mark("batch_end")
        b.close()
    ff.communicate(b"q")
    json.dump(marks, open(f"{OUT}/marks.json", "w"), indent=1)

main()
