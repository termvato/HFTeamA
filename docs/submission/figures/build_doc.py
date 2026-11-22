"""Build the Orchestra technical document (Word).

python build_doc.py <figures dir> <out.docx>
"""

import sys

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

FIG, OUT = sys.argv[1], sys.argv[2]
INK = RGBColor(0x16, 0x21, 0x2B)
INK2 = RGBColor(0x52, 0x51, 0x4E)
ACCENT = RGBColor(0x1C, 0x5C, 0xAB)
FONT = "Arial"

doc = Document()
sec = doc.sections[0]
sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
for side in ("left_margin", "right_margin"):
    setattr(sec, side, Cm(2.0))
sec.top_margin, sec.bottom_margin = Cm(1.8), Cm(1.8)
TEXT_W = 21.0 - 4.0  # cm

st = doc.styles
normal = st["Normal"]
normal.font.name, normal.font.size, normal.font.color.rgb = FONT, Pt(11), INK
normal.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
normal.paragraph_format.space_after = Pt(4)
normal.paragraph_format.line_spacing = 1.08
for name, size, before, after in (("Heading 1", 15, 12, 4), ("Heading 2", 12, 8, 2)):
    h = st[name]
    h.font.name, h.font.size, h.font.bold, h.font.color.rgb = FONT, Pt(size), True, ACCENT
    rf = h.element.rPr.rFonts
    rf.set(qn("w:eastAsia"), FONT)
    for theme in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        if rf.get(qn(theme)) is not None:
            del rf.attrib[qn(theme)]  # the theme font would override Arial
    h.paragraph_format.space_before, h.paragraph_format.space_after = Pt(before), Pt(after)
    h.paragraph_format.keep_with_next = True


def para(text="", bold_lead=None, size=None, color=None, align=None, after=None, italic=False):
    p = doc.add_paragraph()
    if bold_lead:
        r = p.add_run(bold_lead)
        r.bold = True
    r = p.add_run(text)
    r.italic = italic
    if size:
        for run in p.runs:
            run.font.size = Pt(size)
    if color:
        for run in p.runs:
            run.font.color.rgb = color
    if align is not None:
        p.alignment = align
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    return p


def bullet(text, lead=None):
    p = doc.add_paragraph(style="List Bullet")
    if lead:
        p.add_run(lead).bold = True
    p.add_run(text)
    p.paragraph_format.space_after = Pt(2)
    return p


def figure(path, width_cm, caption, n):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(path, width=Cm(width_cm))
    cap(caption, n)


def cap(caption, n, kind="Figure"):
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = c.add_run(f"{kind} {n}. ")
    r.bold = True
    r.font.size = Pt(9)
    r.font.color.rgb = INK2
    r2 = c.add_run(caption)
    r2.font.size = Pt(9)
    r2.font.color.rgb = INK2
    c.paragraph_format.space_after = Pt(8)


def shade(cell, hex_fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


def cell_text(cell, text, bold=False, size=10, color=None, align=None):
    cell.text = ""
    p = cell.paragraphs[0]
    r = p.add_run(text)
    r.bold, r.font.size = bold, Pt(size)
    if color:
        r.font.color.rgb = color
    if align is not None:
        p.alignment = align
    p.paragraph_format.space_after = Pt(0)


def widths(table, cms):
    table.autofit = False
    tblPr = table._tbl.tblPr
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    tblPr.append(layout)
    grid = table._tbl.tblGrid
    for gc, w in zip(grid.findall(qn("w:gridCol")), cms, strict=True):
        gc.set(qn("w:w"), str(int(w * 567)))
    for row in table.rows:
        for c, w in zip(row.cells, cms, strict=True):
            c.width = Cm(w)


def borders(table, color="D6D5D0"):
    tbl = table._tbl
    tblPr = tbl.tblPr
    b = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "single")
        e.set(qn("w:sz"), "4")
        e.set(qn("w:color"), color)
        b.append(e)
    tblPr.append(b)


def page_number_footer():
    p = sec.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Orchestra · Plug and Produce · technical documentation · page ")
    r.font.size, r.font.color.rgb = Pt(8.5), INK2
    run = p.add_run()
    for kind, text in (("begin", None), (None, "PAGE"), ("end", None)):
        if kind:
            f = OxmlElement("w:fldChar")
            f.set(qn("w:fldCharType"), kind)
            run._r.append(f)
        else:
            t = OxmlElement("w:instrText")
            t.set(qn("xml:space"), "preserve")
            t.text = text
            run._r.append(t)
    run.font.size, run.font.color.rgb = Pt(8.5), INK2


page_number_footer()

# ------------------------------------------------------------------ title
t = para("Orchestra: Plug and Produce", size=24, color=INK, after=0)
t.runs[0].bold = True
para(
    "Tidy the desk with SO-101 arms. In a hurry? Plug in another one.",
    size=13,
    color=ACCENT,
    after=2,
)
para(
    'Technical documentation · HackLab Berlin 2026, Hugging Face track "Desk Hero" · '
    "Team Orchestra · 21-22 November 2026",
    size=9.5,
    color=INK2,
    after=8,
)

box = doc.add_table(rows=1, cols=1)
box.alignment = WD_TABLE_ALIGNMENT.CENTER
shade(box.cell(0, 0), "EEF4FB")
c = box.cell(0, 0)
c.width = Cm(TEXT_W)
cell_text(c, "In one paragraph. ", bold=True, size=10.5, color=ACCENT)
c.paragraphs[0].add_run(
    "A camera looks at a messy desk; a vision-language model names each object and decides where "
    "it goes; a planner splits the steps across however many SO-101 arms are plugged in and "
    "predicts when the job will be done. Plug in another arm and it is found by its USB serial, "
    "joins in under a second, and the remaining work is re-planned live: the predicted finish time "
    "comes down and the screen says, in words, what limits further speed-up. No training per "
    "object and no re-programming per arm. Two real arms tidy five boxes in 106 s; in simulation "
    "three arms join one after another and build a 14-box pyramid with no contact between arms."
).font.size = Pt(10.5)
c.paragraphs[0].paragraph_format.space_after = Pt(0)
para("", after=2)

# ------------------------------------------------------------------ 1. problem and idea
doc.add_heading("1. The problem and the idea", level=1)
para(
    "A desk robot today does one trained thing with the objects it was shown. An SO-101 arm costs "
    "about $100; teaching it is the expensive part (one independent test on this arm: 30-40% "
    "success after 50 demonstrations). A second arm means new data or re-programming, and more "
    "arms are never N times faster: published multi-arm systems reach 1.5-3x, without saying why."
)
para(
    "Orchestra turns this around. The arms run known skills (pick, place) behind one contract, and "
    "the intelligence sits in three moves: ",
    after=2,
).add_run("see the desk, split the work, re-split when the set of arms changes.").bold = True
figure(
    f"{FIG}/flow.png",
    16.5,
    "The system in one loop. Steps 1-5 run once per batch; an arm "
    "joining, leaving or failing feeds back into the planner, which re-plans only what has not "
    "started.",
    1,
)

# ------------------------------------------------------------------ 2. how it works
doc.add_heading("2. How it works", level=1)
para(
    "Everything is task-agnostic: a job is a folder with its steps, their order and the shared "
    "areas they need (a bin, a bowl, a pyramid). The core never knows which job it runs.",
    after=4,
)
bullet(
    " A top-down camera and Qwen3-VL point at every object; ArUco markers on the table map "
    "pixels to metres, so no depth camera is needed. The model names objects and picks their "
    "bin, then a validator turns its answer into pick and place steps. In a first test (one "
    "laptop GPU, a synthetic image) it found 6 of 6 objects with 0.6 cm mean error "
    "(8.5 s per scan); after each "
    "place a fresh frame confirms it (2.8 s per check). In the real-arm runs the boxes carry "
    "AprilTags, which give their positions directly.",
    "See.",
)
bullet(
    " Each step is offered only to arms that can reach it. Step-time estimates are replaced "
    "by each arm's measured times as the run goes on. HEFT plans live; CP-SAT (OR-Tools) proves "
    "plans optimal at 4-8 arms in under 2 s. Every re-plan computes a lower bound, finish >= "
    "max(work / arms, longest chain of ordered steps, busiest shared area), and shows the "
    "binding term as the limit: arms, order or a named resource.",
    "Split.",
)
bullet(
    " The executor dispatches steps as a partial order, keeps one arm at a time in each shared "
    "area (a runtime lock), retries failed grasps and watches each arm's heartbeat: a silent arm "
    "is dropped and its step returns to the pool. Running steps are never moved; a new plan is "
    "adopted only if 10% better, unless arms changed or a step failed.",
    "Run.",
)
bullet(
    " Real arms (LeRobot), simulated arms (MuJoCo, the same SO-101 model and IK) and fake arms "
    "all implement execute(task) -> ok | failed | timeout | arm_down. The executor never "
    "branches on which kind it talks to, and a learned policy fits the same call.",
    "Arms.",
)
bullet(
    " A watcher polls the USB serials of the known followers. A new one starts its worker, "
    "checks its state and joint limits, homes slowly and joins; the executor re-plans the "
    "pending work at once.",
    "Plug in.",
)
figure(
    f"{FIG}/architecture.png",
    17.0,
    "Architecture as built: perception, planning and "
    "execution, the arm contract with its three backends, and the observe-and-control layer. "
    "The append-only event log is the only thing the screen reads.",
    2,
)

# ------------------------------------------------------------------ 3. the screen
doc.add_heading("3. What the operator sees", level=1)
para(
    "The dashboard (FastAPI with a live event stream) shows each arm and its state, the "
    "countdown, a what-if row for 1-4 arms, plan against actual and an activity feed. Figure 3 "
    "is the moment that matters: one arm running, then a second one plugged in."
)
figure(
    f"{FIG}/fig_dashboard.png",
    16.5,
    "The live screen (Orchestra dashboard, fake arms). "
    "Top: one arm, 1:01 left. Bottom, about 6 s after the second arm was plugged in: 0:39 left, "
    "and the new limit named: the shared bowl.",
    3,
)
para(
    "Why speed-up stops is no mystery to the operator: for a 12-object sort the planner "
    "predicts 120 s with one arm and 64 s with two; a third arm helps less and a fourth adds "
    "nothing, because every arm queues at the shared input bin (Figure 4). That is 2.2x with "
    "three arms, inside the published 1.5-3x range, and the screen names the binding limit."
)
figure(
    f"{FIG}/chart_speedup.png",
    12.0,
    "Planner output for sorting 12 objects with 1-4 arms "
    "against the ideal 1/N, with the binding limit under each pair (planned, not measured).",
    4,
)

# ------------------------------------------------------------------ 4. results
doc.add_heading("4. Results", level=1)
doc.add_heading("4.1 Two real arms", level=2)
para(
    "Two SO-101 leader/follower pairs were calibrated on a shared table with a C922 overhead "
    "camera, AprilTag boxes and a paper bowl as the shared drop area. On Sunday morning both "
    "arms picked and placed by camera and re-split live; in the join run, the second arm was "
    "plugged in 32 s into the batch and joined in under a second."
)
figure(
    f"{FIG}/fig_real_rig.png",
    14.0,
    "The real rig: two SO-101 followers, each holding an "
    "AprilTag box, above the shared paper bowl; the ArUco markers on the table map the camera "
    "to table coordinates.",
    5,
)
figure(
    f"{FIG}/chart_real_run.png",
    12.0,
    "Five boxes with two real arms: the best measured "
    "batch (106 s) against the computed bounds. The gap is unbalanced work (one arm finishes "
    "about 36 s early) and turns at the bowl (0-21 s).",
    6,
)

kt = doc.add_table(rows=1, cols=3)
kt.alignment = WD_TABLE_ALIGNMENT.CENTER
borders(kt)
rows = [
    ("Measure", "Value", "Where"),
    ("Best two-arm batch, 5 boxes", "106 s", "real arms, 22 Nov 09:20"),
    ("Second arm plugged in mid-run: joins", "under 1 s", "real arms, 09:25"),
    ("Planned speed-up, 3 arms (12 objects)", "2.2x", "planner output"),
    ("Sort 12 objects, 3 simulated arms", "24/24 steps", "MuJoCo, grasp retry"),
    ("Pyramid, arms join after 3 and 7 boxes", "28/28 steps, ~120 s", "MuJoCo, 3 runs"),
    ("Contact between arms (pyramid runs)", "none, closest 6.1 cm", "checked at 50 Hz"),
    ("Automated tests (CI on every push)", "about 440", "pytest + ruff"),
]
for k, (a, b, cc) in enumerate(rows):
    r = kt.rows[0] if k == 0 else kt.add_row()
    for j, txt in enumerate((a, b, cc)):
        cell_text(r.cells[j], txt, bold=(k == 0), size=9.5, color=INK if k else ACCENT)
    if k == 0:
        for cl in r.cells:
            shade(cl, "EEF4FB")
widths(kt, [7.6, 4.4, 5.0])
cap(
    "Key numbers. Real-arm results come from the run logs; planned numbers are planner output.",
    1,
    kind="Table",
)

doc.add_heading("4.2 Three arms in simulation", level=2)
para(
    "Only two arms fit on our table, so the same planner and executor ran three simulated arms "
    "on a different job: stacking 14 boxes into a three-layer pyramid. Arm 1 starts alone; arm 2 "
    "is plugged in after 3 boxes and arm 3 after 7, through the same call a USB plug-in makes. "
    "MuJoCo simulates; Isaac Sim renders the same scene from every body's pose. Each arm picks in "
    "its own wedge and the pyramid is the one locked shared area. Three checked runs completed "
    "28/28 steps in about 120 s with no contact between arms (closest 6.1 cm); in the recorded "
    "run every box ended within 2.9 mm of its slot."
)
figure(
    f"{FIG}/fig_sim.png",
    17.0,
    "Isaac Sim render of the MuJoCo run: arm 1 alone (left), "
    "all three arms working (middle), the finished pyramid (right). Video: "
    "docs/media/pyramid_isaac_sim.mp4.",
    7,
)

doc.add_heading("4.3 Learned policies on the same contract", level=2)
para(
    "The arms run scripted IK skills today. To handle objects without known coordinates, a "
    "learned policy can stand behind the same execute(task) call, one copy per arm, with the "
    "orchestrator still splitting the objects. A recording studio with live top and wrist "
    "cameras produced a public single-arm dataset (20 episodes) and a two-arm dataset "
    "(41 episodes). SmolVLA was trained with early stopping on held-out episodes, and X-VLA "
    "(0.9 B parameters) on the two-arm demos in 1 h 42 min, reaching a training loss of 0.003 "
    "and 0.9 degrees mean open-loop error on recorded frames. That shows the model fits the "
    "demonstrations; its success rate on the arms is not measured yet."
)
lp = doc.add_table(rows=1, cols=2)
lp.alignment = WD_TABLE_ALIGNMENT.CENTER
widths(lp, [8.5, 8.5])
for k, (img, w) in enumerate(
    ((f"{FIG}/fig_real_front_12.png", 6.4), (f"{FIG}/chart_xvla_loss.png", 6.4))
):
    cl = lp.cell(0, k)
    cl.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cl.paragraphs[0].add_run().add_picture(img, width=Cm(w))
cap(
    "Left: a front-camera frame from the two-arm dataset, 41 teleoperated demonstrations "
    "(diegoloayza789/hfteam_box_middle). Right: X-VLA training loss on it (log scale).",
    8,
)

# ------------------------------------------------------------------ 5. safety, limits, next
doc.add_heading("5. Safety", level=1)
para(
    "Nothing moves a real arm without the person at the rig confirming it. Targets are checked "
    "against joint limits and the table; two arms never share an area (runtime lock); a stop is "
    "slow (freeze, lower, then torque off), never with the arm raised; follower motion is capped "
    "per step; calibration is never rewritten. The emergency stop is the follower's 12 V supply."
)

doc.add_heading("6. Limits and next steps", level=1)
bullet(
    " A place step locks the bowl for its whole 15 s. Locking it only for the drop, or one drop "
    "spot per arm, removes most of the waiting.",
    "Turns at the shared bowl:",
)
bullet(
    " When every box is reachable by only one arm, a new arm adds its own boxes instead of "
    "taking a share, and the prediction rises. Objects in the band both arms reach make the "
    "finish time drop.",
    "Layout matters:",
)
bullet(
    " Both arms' motion loops share one Python process today; a few runs slowed 3-4x. One "
    "process per arm is the fix.",
    "One process per arm:",
)
bullet(
    " Measure SmolVLA and X-VLA success on the arms, then run them behind the same contract "
    "with the scripted skill as fallback.",
    "Learned policies:",
)

doc.add_heading("7. Build and run", level=1)
para(
    "Code: github.com/termvato/HFTeamA (public repository, Python 3.10+). Run it without "
    'hardware, then open http://127.0.0.1:8050, press Go and "Plug in next arm":',
    size=10,
    after=2,
)
for line in (
    'pip install -e ".[dev,sched,view]"',
    "python -m hfteama.demo --backend fake --plugged rakesh_follower",
):
    cp = doc.add_paragraph()
    cp.paragraph_format.left_indent = Cm(0.6)
    cp.paragraph_format.space_after = Pt(1)
    cr = cp.add_run(line)
    cr.font.name, cr.font.size = "Courier New", Pt(9.5)

doc.save(OUT)
print("saved", OUT)
