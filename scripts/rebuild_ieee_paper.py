"""Exact IEEE Two-Column Format Research Paper Rebuilder for ASUC-SOM.
Preserves the exact human-written academic paper text, IEEE section hierarchy,
equations, Algorithm 1 box, Table I (Routing), Table II (Results), Table III (Comparison),
and IEEE reference bibliography in genuine IEEE two-column conference/transactions layout.
"""
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from pathlib import Path
import os

def set_cell_border(cell, **kwargs):
    """
    Set cell borders: top, bottom, left, right.
    Usage: set_cell_border(cell, top={"sz": 4, "val": "single", "color": "000000"})
    """
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = tcPr.first_child_found_in("w:tcBorders")
    if tcBorders is None:
        tcBorders = OxmlElement('w:tcBorders')
        tcPr.append(tcBorders)
    for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        edge_data = kwargs.get(edge)
        if edge_data:
            tag = 'w:{}'.format(edge)
            element = tcBorders.find(qn(tag))
            if element is None:
                element = OxmlElement(tag)
                tcBorders.append(element)
            for key, attr in [("val", "w:val"), ("color", "w:color"), ("sz", "w:sz"), ("space", "w:space")]:
                if key in edge_data:
                    element.set(qn(attr), str(edge_data[key]))

def build_ieee_paper():
    doc = docx.Document()
    
    # ── 1. Page & Section Setup ──────────────────────────────────────
    # Section 1: Title and Authors (Single Column spanning page)
    sec1 = doc.sections[0]
    sec1.top_margin = Inches(0.75)
    sec1.bottom_margin = Inches(1.0)
    sec1.left_margin = Inches(0.625)
    sec1.right_margin = Inches(0.625)
    sec1.header_distance = Inches(0.5)
    sec1.footer_distance = Inches(0.5)
    
    # Helper to style runs
    def style_run(run, font_name="Times New Roman", size_pt=10, bold=False, italic=False):
        run.font.name = font_name
        run.font.size = Pt(size_pt)
        run.bold = bold
        run.italic = italic

    # ── Title ───────────────────────────────────────────────────────
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(12)
    run_title = p_title.add_run("ASUC-SOM: Subspace-Aware Adaptive Routing for Sequential Machine Unlearning under GDPR Article 17")
    style_run(run_title, size_pt=18, bold=True)

    # ── Authors ─────────────────────────────────────────────────────
    p_auth = doc.add_paragraph()
    p_auth.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_auth.paragraph_format.space_after = Pt(14)
    
    run_a1 = p_auth.add_run("Mithul Kannan\n")
    style_run(run_a1, size_pt=10, bold=True)
    
    run_affil = p_auth.add_run("Department of Computer Science and Engineering\nAutonomous Systems and Machine Learning Research Group\n")
    style_run(run_affil, size_pt=9, italic=True)
    
    run_email = p_auth.add_run("mithulkannan@ieee.org")
    style_run(run_email, size_pt=9)

    # ── 2. Create Section 2 for Two-Column Body ─────────────────────
    sec2 = doc.add_section(docx.enum.section.WD_SECTION.CONTINUOUS)
    sec2.top_margin = Inches(0.75)
    sec2.bottom_margin = Inches(1.0)
    sec2.left_margin = Inches(0.625)
    sec2.right_margin = Inches(0.625)
    
    # Enable 2-column layout in sec2 XML
    sectPr = sec2._sectPr
    cols = sectPr.xpath('./w:cols')
    if cols:
        cols[0].set(qn('w:num'), '2')
        cols[0].set(qn('w:space'), '360') # 0.25 in (360 twips) column gap
    else:
        new_cols = parse_xml(r'<w:cols {} w:num="2" w:space="360"/>'.format(nsdecls('w')))
        sectPr.append(new_cols)

    # Helper functions for IEEE paragraphs
    def add_p(text="", align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_after=4, indent=14.4):
        p = doc.add_paragraph()
        p.alignment = align
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.line_spacing = 1.0
        if indent > 0:
            p.paragraph_format.first_line_indent = Pt(indent)
        if text:
            run = p.add_run(text)
            style_run(run, size_pt=10)
        return p

    def add_h1(title_text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(title_text)
        style_run(run, size_pt=10, bold=True)
        return p

    def add_h2(subtitle_text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(subtitle_text)
        style_run(run, size_pt=10, italic=True, bold=True)
        return p

    def add_eq(eq_text, eq_num=""):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.first_line_indent = Pt(0)
        run_eq = p.add_run(f"\t{eq_text}\t{eq_num}")
        style_run(run_eq, font_name="Times New Roman", size_pt=9.5, italic=True)
        return p

    def add_fig(img_path, caption_text, width_in=3.25):
        if os.path.exists(img_path):
            p_img = doc.add_paragraph()
            p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_img.paragraph_format.space_before = Pt(6)
            p_img.paragraph_format.space_after = Pt(2)
            p_img.paragraph_format.first_line_indent = Pt(0)
            run_img = p_img.add_run()
            run_img.add_picture(img_path, width=Inches(width_in))
            
            p_cap = doc.add_paragraph()
            p_cap.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p_cap.paragraph_format.space_before = Pt(0)
            p_cap.paragraph_format.space_after = Pt(6)
            p_cap.paragraph_format.first_line_indent = Pt(0)
            p_cap.paragraph_format.line_spacing = 1.0
            run_cap = p_cap.add_run(caption_text)
            style_run(run_cap, size_pt=8, italic=False)

    # ── Abstract & Index Terms ──────────────────────────────────────
    p_abs = doc.add_paragraph()
    p_abs.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_abs.paragraph_format.space_after = Pt(4)
    p_abs.paragraph_format.first_line_indent = Pt(14.4)
    p_abs.paragraph_format.line_spacing = 1.0
    
    r_abs_label = p_abs.add_run("Abstract—")
    style_run(r_abs_label, size_pt=9, bold=True, italic=True)
    r_abs_text = p_abs.add_run(
        "Machine unlearning is usually studied one request at a time: a model is trained, a forget set is named, an algorithm removes it, and the evaluation stops. "
        "Regulation does not work that way. Under GDPR Article 17 and the CCPA, erasure requests keep arriving for as long as a system stays in service, and each one lands on a model that earlier requests have already altered. "
        "We argue that this sequential setting is where fast unlearning methods such as SSD and SalUn break down, with retain accuracy sliding toward chance within two to five consecutive requests. "
        "We present ASUC-SOM, a controller that treats a stream of deletions as a closed-loop control problem. A Subspace Overlap Monitor (SOM) keeps a low-rank orthonormal basis of directions that earlier unlearning steps have already changed, "
        "and scores how much of a new deletion gradient falls inside it, which gives a warning before any weights are touched. "
        "A tiered router then chooses between SSD, SalUn with retain regularization, and exact retraining, using the SOM score and the shape of the request. "
        "A verification gate checks forget accuracy, retain accuracy and membership-inference risk, and if any check fails the checkpoint is rolled back and the request is escalated one tier. "
        "The two fast tiers run about 150× and 115× faster than retraining. A compliance layer wraps the controller and records every batch in a hash-chained audit trail."
    )
    style_run(r_abs_text, size_pt=9, bold=False)

    p_idx = doc.add_paragraph()
    p_idx.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_idx.paragraph_format.space_after = Pt(10)
    p_idx.paragraph_format.first_line_indent = Pt(14.4)
    r_idx_label = p_idx.add_run("Index Terms—")
    style_run(r_idx_label, size_pt=9, bold=True, italic=True)
    r_idx_text = p_idx.add_run("machine unlearning, sequential deletion, GDPR Article 17, right to be forgotten, selective synaptic dampening, saliency, subspace monitoring, membership inference.")
    style_run(r_idx_text, size_pt=9, italic=True)

    # ── Section I: Introduction ──────────────────────────────────────
    add_h1("I. INTRODUCTION")
    add_p("A trained network is, in a practical sense, a compressed copy of its training data. When someone invokes the right to erasure in Article 17 of the GDPR [1], or the deletion rights in the CCPA, the operator of a model has to decide what that request means for the weights. The cleanest answer is to retrain from scratch without that person’s records. It is also, for any model of useful size, an answer nobody wants to pay for every time a request comes in. Machine unlearning grew out of this tension [2], [3]. The question is whether the influence of a forget set Df can be removed faster than retraining, and whether anyone can show that it worked.")
    add_p("The last few years have produced several good answers to the single-request version of that question. Selective Synaptic Dampening (SSD) rescales parameters that matter more to the forget set than to the rest of the data, and needs no gradient steps at all [4]. SalUn confines fine-tuning to weights with high gradient saliency on the forget set [5]. Gradient ascent, Fisher-based noise injection [7] and teacher-based approaches [10] fill out the picture. Nearly all of this work is evaluated the same way. Take one trained model, delete one set, report forget and retain accuracy, and stop.")
    add_p("That protocol is a poor match for how deletion happens in production. Requests arrive over weeks and months, in different sizes, and from different corners of the label space. The model that receives the twentieth request is not the model that was trained; it is the product of nineteen earlier edits. Three things go wrong when a static method is applied repeatedly. First, updates accumulate error, and retain accuracy often falls to roughly random guessing after two to five rounds. We call this parameter fatigue. Second, no single method suits every request. Removing 50 random samples and removing 2,000 samples of a single class are geometrically very different jobs, yet a fixed algorithm treats them identically. Third, updates are applied blind. Nothing asks what earlier edits did to the same weights beforehand, and nothing confirms afterwards that the job was done.")
    add_p("This paper describes ASUC-SOM, a controller built around those three observations. Our contributions are as follows. (i) We pose sequential unlearning as a control problem with per-round constraints, and introduce the Subspace Overlap Monitor, a cheap geometric score that predicts fatigue before an update is applied. (ii) We design a three-tier router that matches each request to SSD, SalUn or exact retraining based on the SOM score and on request statistics. (iii) We add a closed-loop verification gate with rollback and automatic escalation, so that no model is released until it passes. (iv) We package these pieces in a compliance layer that produces a tamper-evident audit record for each batch. The rest of the paper covers related work (Section II), the problem setting (Section III), the framework (Section IV), the evaluation design (Section V), discussion of results (Section VI), limitations (Section VII) and conclusions (Section VIII).")

    # ── Section II: Related Work ─────────────────────────────────────
    add_h1("II. RELATED WORK")
    add_h2("A. Exact and approximate unlearning")
    add_p("Cao and Yang introduced the term for statistical-query learners [2], and Bourtoule et al. made exact unlearning practical for deep models by sharding the training data and retraining only the affected shard (SISA) [3]. Ginart et al. studied deletion for clustering [14], and Sekhari et al. gave generalization-based guarantees for what can be deleted [15]. Approximate methods trade guarantees for speed. Golatkar et al. scrub weights using Fisher information [7], Graves et al. undo the updates that a batch contributed [13], and Chundawat et al. distil the forgetting behaviour from a deliberately incompetent teacher [10]. SSD [4] and SalUn [5] are the two fast methods that we use as building blocks. Thudi et al. caution that the metrics used to judge approximate unlearning can be misleading [8], a point we return to in Section VII.")
    
    add_h2("B. Sequential and adaptive deletion")
    add_p("Most closely related in spirit is the work of Gupta et al. on adaptive machine unlearning [12], which examines deletion sequences in which later requests may depend on earlier model outputs and gives guarantees for certain learners. Their analysis is theoretical and centres on what an adversary can do with a stream of requests. Our concern is the engineering side of the same setting: what happens to the accuracy of a deep network when a fast heuristic is applied twenty times in a row, and what a controller can do about it.")

    add_h2("C. Catastrophic forgetting and subspace methods")
    add_p("Unlearning repeated many times resembles continual learning run in the wrong direction. Elastic weight consolidation limits drift in parameters that earlier tasks relied on [6], and gradient projection memory keeps a basis of important subspaces and projects new steps to be orthogonal to it [16]. SOM borrows the idea of maintaining a low-rank basis but uses it differently. It never constrains the update. It acts as a sensor, and its reading selects which unlearning primitive is safe to use.")

    add_h2("D. Auditing unlearning")
    add_p("Membership inference [9], [17] is the standard tool for asking whether a model still behaves differently on data it was supposed to forget. We use it as one of three checks in the verification gate, not as the only one, for reasons discussed later.")
    add_p("Outside the machine-learning literature, the practical question is what a regulator or a customer would accept as evidence. Whatever the answer turns out to be, an operator will want a record of what was checked and when. That is why we speak of audit trails in this paper and avoid the word certification.")

    # ── Section III: Problem Formulation ─────────────────────────────
    add_h1("III. PROBLEM FORMULATION")
    add_p("Let a model θ0 be trained on a dataset D with C classes. Over time, a stream of deletion requests Df(1), …, Df(T) arrives. After t requests the retain set is Dr(t) = D ∖ (Df(1) ∪ … ∪ Df(t)). At each round the operator chooses one of three unlearning primitives at ∈ {1, 2, 3}, and the model is updated as")
    add_eq("θt = Uat(θt−1, Df(t), Dr(t))", "(1)")
    add_p("A round is acceptable only if three conditions hold:")
    add_eq("Accf(θt) ≤ 1/C,    Accr(θt−1) − Accr(θt) ≤ ε,    |AUCt − 0.5| ≤ δ", "(2)")
    add_p("The first asks for forget-set accuracy at or below chance, the second bounds the loss in retain accuracy relative to the previous verified checkpoint, and the third asks that a membership-inference attacker is no better than a coin flip. We use ε = 3 percentage points and, for CIFAR-10 [11], 1/C = 10%. The operator’s goal is to satisfy these conditions in every round while keeping total compute low. Exact retraining always meets the goal and sets the upper bound on cost. The interesting question is how rarely it can be used.")
    add_p("We assume the operator keeps the retained data and can identify the forget set for each request, that requests are processed in arrival order and in batches, and that a checkpoint can be stored before each round. We make no legal claim about whether model weights are themselves personal data, which is still debated. Article 17 requires erasure without undue delay [1], and that timing pressure is the reason cost matters here at all.")

    # ── Section IV: Framework ────────────────────────────────────────
    add_h1("IV. THE ASUC-SOM FRAMEWORK")
    add_h2("A. Overview")
    add_p("The system processes each deletion request through a closed-loop control pipeline. A profiler summarizes the request, SOM measures how it interacts with earlier edits, the router picks a primitive, the primitive is applied, and the verification gate decides whether the result may leave the system. Failure sends the request back to the router one tier higher, after restoring the saved checkpoint.")

    add_h2("B. Request profiling")
    add_p("The profiler records two numbers. One is the cardinality n = |Df|. The other is the normalized class entropy of the forget set. If pc is the fraction of Df that belongs to class c, then")
    add_eq("h = − Σc pc log pc / log C  ∈ [0, 1]", "(3)")
    add_p("A value near 1 means the request is spread evenly across classes, which is typical of a handful of users deleting their own records. A value near 0 means it is concentrated, as when an entire category is withdrawn. Concentrated requests are harder because the dampening or fine-tuning has to act on features that the retained classes may share.")

    add_h2("C. Subspace Overlap Monitor")
    add_p("Scalar drift such as ‖θt − θ0‖ tells us how far the weights have moved, but not where. Two deletions can move the weights equally far, one into untouched directions and the other back into directions that previous deletions already disturbed. Only the second is a fatigue risk. SOM is designed to separate them.")
    add_p("SOM maintains a matrix Qk ∈ ℝd×k with orthonormal columns, spanning the parameter-space directions modified by earlier unlearning steps. When a new request produces a forget-loss gradient gt, the monitor reports")
    add_eq("SOM(gt, Qk) = ‖QkT gt‖2 / ‖gt‖2  ∈ [0, 1]", "(4)")
    add_p("This is the cosine of the angle between gt and its projection onto the subspace, so its square is the fraction of gradient energy that lies inside the region already edited. A score near 0 says the request is pushing into fresh territory. A score near 1 says it will mostly hit weights that have already absorbed damage. Because it is computed from the gradient and not from the updated weights, the score is available before the update is applied, which is what lets the router act on it.")
    add_p("After each verified round, the realized change Δ = θt − θt−1 is folded into the basis. The component of Δ outside the current span is orthogonalized by Gram–Schmidt and appended as a new column. If the number of columns would exceed a rank cap kmax, we take a truncated SVD of the weighted basis together with Δ and keep the leading kmax directions. Scoring costs O(dk) and the update costs O(dk2), which is small next to a training epoch. For very large networks the same monitor can be run on a subset of layers or on a random projection of the gradient; we regard that as an implementation choice and do not evaluate it here.")

    add_h2("D. Tiered adaptive router")
    add_p("The router (ASUC) maps the pair (s, h) to a tier, where s is the SOM score from (4). With default thresholds τl = 0.40, τh = 0.80 and τH = 0.40, the rule is")
    add_eq("a = 3 if s ≥ τh;   2 if τl ≤ s < τh or h < τH;   1 otherwise", "(5)")
    add_p("The conditions are checked in that order. Table I lists each tier together with its speedup relative to exact retraining.")

    # ── Table I: Routing policy ──
    p_t1_cap = doc.add_paragraph()
    p_t1_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_t1_cap.paragraph_format.space_before = Pt(6)
    p_t1_cap.paragraph_format.space_after = Pt(2)
    r_t1_cap = p_t1_cap.add_run("TABLE I\nROUTING POLICY")
    style_run(r_t1_cap, size_pt=9, bold=True)

    t1 = doc.add_table(rows=4, cols=4)
    t1.alignment = WD_TABLE_ALIGNMENT.CENTER
    t1_data = [
        ["Tier", "Trigger", "Primitive", "Speedup"],
        ["1", "s < 0.40 and h ≥ 0.40", "SSD [4]", "≈150×"],
        ["2", "0.40 ≤ s < 0.80, or h < 0.40", "SalUn + retain loss [5]", "≈115×"],
        ["3", "s ≥ 0.80, or escalation", "Exact retraining", "1×"]
    ]
    for r_i, row in enumerate(t1_data):
        for c_i, val in enumerate(row):
            cell = t1.cell(r_i, c_i)
            cell.text = val
            p_c = cell.paragraphs[0]
            p_c.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_c.paragraph_format.space_before = Pt(2)
            p_c.paragraph_format.space_after = Pt(2)
            style_run(p_c.runs[0], size_pt=8.5, bold=(r_i == 0))
            set_cell_border(cell, top={"sz": 4, "val": "single", "color": "000000"},
                                  bottom={"sz": 4, "val": "single", "color": "000000"})

    add_p("Tier 1 is for the easy case: a balanced request that avoids previously edited directions. SSD needs no gradient steps and is the cheapest option. Tier 2 handles the middle ground. When the request overlaps earlier edits, or is concentrated on a few classes, we use SalUn with an added retain-loss term so that the update is anchored to the data we want to keep. Tier 3 is reserved for the case where the SOM score says that a fast update would damage weights that are already compromised. There, exact retraining on Dr(t) is the only option that does not trade away utility.")
    add_p("The thresholds are easier to interpret in terms of energy. Because SOM is a cosine, the squared score is the share of the gradient lying in the edited subspace. At s = 0.40 that share is 16%, and at s = 0.80 it is 64%. So Tier 1 is used while fewer than about one-sixth of the gradient’s energy overlaps earlier edits, and Tier 3 is chosen once nearly two-thirds of it does. These cut-offs are design defaults and we treat their sensitivity as an experimental question (Section V), not a settled one.")
    add_p("We chose SSD and SalUn as the fast tiers because they sit at opposite ends of the cost and care spectrum. SSD changes weights in one pass by dampening, which makes it extremely cheap but gives it nothing to lean on when the forget set overlaps heavily with retained knowledge. SalUn takes gradient steps, so it can include a retain-loss term and is slower, but it protects utility better under overlap. The router is agnostic to these particular choices; any primitive could be slotted into a tier, provided its cost and behaviour under overlap are known.")

    add_h2("E. Closed-loop verification and rollback")
    add_p("Before the model is released, three checks are run. (a) Forget-set accuracy must not exceed chance, 10% on CIFAR-10. (b) Retain-set accuracy must not fall by more than 3 percentage points relative to the checkpoint taken at the start of the round. (c) The area under the ROC curve of a membership-inference attack [9] against the forget samples must be close to 0.50. If any check fails, the controller restores the checkpoint, increments the tier, and tries again. A request that fails even at Tier 3 is not released and is flagged for human review. The extra cost is a few evaluation passes and one attack run per round, which is small beside the cost of any training-based tier.")

    add_h2("F. Audit trail")
    add_p("The deployment layer (ARIA) writes one record per request containing the request identifier, |Df|, h, the SOM score, the tier finally used, the number of escalations, the three verification measurements and a digest of the released checkpoint. Each record includes a timestamp and the hash of the preceding record, so that editing a past entry breaks the chain. A live view of the SOM trace and tier decisions is available to operators. We want to be clear about what these records are. They document that specific empirical checks were run and passed at a specific time. They are not a mathematical certificate of erasure.")

    add_h2("G. A worked routing example")
    add_p("The following sequence is illustrative and its numbers are invented to show the mechanics, not measured. Suppose the first request removes 50 random samples. The basis is empty, so s = 0, and the entropy is high, so the router picks Tier 1. SSD runs, the gate passes, and the update direction is stored in Q. A second small random request arrives and has s = 0.15; it also goes to Tier 1. The third request removes most of one class. Its entropy is low, so it goes to Tier 2 even though s = 0.30. Suppose SalUn leaves the retain accuracy 4 points lower, which exceeds the bound. The gate rejects the result, the checkpoint is restored, and the request is rerun at Tier 3. After the exact retrain the basis is cleared and the stream continues on a clean footing. No record is lost along the way: the audit entry for the third request would show one escalation and the final tier.")

    add_h2("H. Procedure")
    add_p("Algorithm 1 gives the complete procedure for one request. Note that after an exact retrain the accumulated edits are discarded, so the basis is cleared; after any other tier it is extended with the verified update.")

    # ── Algorithm 1 Box ──
    t_alg = doc.add_table(rows=1, cols=1)
    t_alg.alignment = WD_TABLE_ALIGNMENT.CENTER
    c_alg = t_alg.cell(0, 0)
    set_cell_border(c_alg, top={"sz": 6, "val": "single", "color": "000000"},
                           bottom={"sz": 6, "val": "single", "color": "000000"},
                           left={"sz": 4, "val": "none", "color": "auto"},
                           right={"sz": 4, "val": "none", "color": "auto"})
    p_alg = c_alg.paragraphs[0]
    p_alg.paragraph_format.space_before = Pt(2)
    p_alg.paragraph_format.space_after = Pt(2)
    p_alg.paragraph_format.line_spacing = 1.05
    alg_text = (
        "Algorithm 1. ASUC-SOM, one deletion request\n"
        "Input: model θ, basis Q, forget set Df, retain data Dr\n"
        "Params: τl = 0.40, τh = 0.80, τH = 0.40\n"
        "1:  θprev ← θ   (checkpoint)\n"
        "2:  n ← |Df|;  h ← normalized class entropy of Df\n"
        "3:  g ← forget-loss gradient of θ on Df\n"
        "4:  s ← ‖QTg‖ / ‖g‖   (s ← 0 if Q is empty)\n"
        "5:  a ← Route(s, h)   ∈ {1, 2, 3}\n"
        "6:  repeat\n"
        "7:  θ ← Apply(a, θprev, Df, Dr)\n"
        "8:  if Verify(θ) passes then break\n"
        "9:  θ ← θprev;  a ← a + 1   (rollback, escalate)\n"
        "10: until a > 3  (then flag for human review)\n"
        "11: Q ← UpdateBasis(Q, θ − θprev);  reset Q if a = 3\n"
        "12: append hash-chained audit record;  return θ, Q"
    )
    p_alg.text = alg_text
    style_run(p_alg.runs[0], font_name="Courier New", size_pt=8.5)

    # ── Section V: Evaluation Design ─────────────────────────────────
    add_h1("V. EVALUATION DESIGN")
    add_p("Data and model. We use CIFAR-10 [11], which has C = 10 classes, with a ResNet-18 (11,173,962 parameters) trained to 93.44% test accuracy and 99.40% train retain accuracy. Request stream. Each experiment runs T = 20 sequential rounds. The stream mixes small balanced requests (for example 500 random samples per round across all classes in Workload A, totaling 10,000 deleted images) with class-concentrated ones (Workload B, 500 samples per class per round, H(c) = 0.0), and high-loss requests (Workload C), in a fixed pseudo-random order shared by all methods. Baselines. Gradient ascent, SSD and SalUn are each applied statically at every round, with identical hyper-parameters across rounds. Exact retraining is the reference for cost and for the ideal outcome.")
    add_p("Metrics. We report forget accuracy, retain accuracy, test accuracy and the membership-inference AUC after every round, plus wall-clock time and speedup relative to retraining. For the sequential setting we add two summary numbers: the number of rounds before retain accuracy falls to within 5 points of chance, and the tier histogram of the controller including how often escalation occurred. All results are averaged over 3 seeds. Training settings are SGD with momentum 0.9, weight decay 5e-4, initial learning rate 0.1 cosine-annealed over 200 epochs, and batch size 128.")
    add_p("Ablations and sensitivity. To see what each component contributes we remove them one at a time. Without SOM, the router uses only |Df| and h. Without the router, every request goes to a single tier. Without the gate, updates are released unchecked. We also sweep τl and τh over a grid around the defaults and report how the Tier 3 fraction and the final retain accuracy move. Finally, we compare SOM against a plain drift norm ‖θt − θ0‖ as the routing signal, which tests the claim in Section IV-C that direction matters and not only distance.")
    add_p("Reproducibility. Every run logs the request order, the seeds, the SOM score and tier for each round, and the verification measurements, so that any row of Table II can be traced back to a specific stream. Full code, dataset loaders, and raw logs are available at https://github.com/mithulkannan17/Adaptive-unlearning.")

    # ── Section VI: Results and Discussion ───────────────────────────
    add_h1("VI. RESULTS AND DISCUSSION")
    add_h2("A. Cost")
    add_p("Against exact retraining, SSD (Tier 1) gives a speedup of about 150× and SalUn with retain regularization (Tier 2) about 115×. Tier 3 costs as much as retraining by definition, so the benefit of the controller over a stream depends on how often it is invoked. If f1, f2, f3 are the fractions of rounds that end at each tier, and we ignore verification overhead, the mean cost per round in units of one retraining is")
    add_eq("c̄ = f1/150 + f2/115 + f3,    speedup = 1/c̄", "(6)")
    add_p("This is worth pausing on because it sets expectations honestly. If one round in twenty ends in a retrain (f3 = 0.05) and the rest are split between the fast tiers, the stream-level speedup is about 17×, not 100×. The Tier 3 fraction dominates the result. A good SOM is therefore one that is cautious enough to prevent collapse but not so cautious that it sends everything to retraining.")

    add_h2("B. Behaviour over a request stream")
    add_p("The central comparison is in Table II. Static baselines follow the pattern described in Section I: static SSD collapses catastrophically by Round 3 (dropping from 99.53% retain accuracy to 49.42% at Round 2 and collapsing to 9.96% at Round 3). Gradient Ascent fails to forget (99.80% forget accuracy). In contrast, ASUC-SOM sustains 97.19% retain accuracy and 90.89% test accuracy across all 20 rounds, verifying that closed-loop dynamic tier routing prevents catastrophic representational collapse.")

    # ── Table II: Results Table ──
    p_t2_cap = doc.add_paragraph()
    p_t2_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_t2_cap.paragraph_format.space_before = Pt(6)
    p_t2_cap.paragraph_format.space_after = Pt(2)
    r_t2_cap = p_t2_cap.add_run("TABLE II\nSEQUENTIAL UNLEARNING, T = 20 ROUNDS (WORKLOAD A)")
    style_run(r_t2_cap, size_pt=9, bold=True)

    t2 = doc.add_table(rows=6, cols=6)
    t2.alignment = WD_TABLE_ALIGNMENT.CENTER
    t2_data = [
        ["Method", "Rounds to collapse", "Final retain acc.", "Forget acc.", "MIA AUC", "Total time"],
        ["GA", "None (Failed to forget)", "99.47%", "99.80%", "0.998", "70.7 s"],
        ["SSD", "Round 3 (Collapse)", "9.96%", "10.00%", "0.500", "73.2 s"],
        ["SalUn", "None (>20 Stable)", "95.03%", "10.40%", "0.538", "303.5 s"],
        ["Retrain", "—", "99.10%", "9.20%", "0.512", "~21,100 s"],
        ["ASUC-SOM", "None (>20 Stable)", "97.19%", "10.20%", "0.546", "15,655.4 s*"]
    ]
    for r_i, row in enumerate(t2_data):
        for c_i, val in enumerate(row):
            cell = t2.cell(r_i, c_i)
            cell.text = val
            p_c = cell.paragraphs[0]
            p_c.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_c.paragraph_format.space_before = Pt(2)
            p_c.paragraph_format.space_after = Pt(2)
            style_run(p_c.runs[0], size_pt=8.5, bold=(r_i == 0 or r_i == 5))
            set_cell_border(cell, top={"sz": 4, "val": "single", "color": "000000"},
                                  bottom={"sz": 4, "val": "single", "color": "000000"})

    add_fig("paper_figures/fig1_sequential_accuracy.png",
            "Fig. 1. Sequential retain and test accuracy trajectories across 20 unlearning rounds on CIFAR-10 ResNet-18 (Workload A). "
            "Static SSD exhibits catastrophic representational collapse at Round 3, while ASUC-SOM sustains >97% retain utility throughout all 20 rounds.")

    add_fig("paper_figures/fig2_som_dynamics_drift.png",
            "Fig. 2. Subspace Overlap Metric (SOM) trajectory and parameter drift ||Δθ|| over 20 rounds. "
            "Green (s < 0.40), Amber (0.40 ≤ s < 0.80), and Red (s ≥ 0.80) regions dictate dynamic tier dispatch.")

    add_p("Table III summarizes the design differences that drive that outcome.")

    # ── Table III: Design Comparison ──
    p_t3_cap = doc.add_paragraph()
    p_t3_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_t3_cap.paragraph_format.space_before = Pt(6)
    p_t3_cap.paragraph_format.space_after = Pt(2)
    r_t3_cap = p_t3_cap.add_run("TABLE III\nDESIGN COMPARISON")
    style_run(r_t3_cap, size_pt=9, bold=True)

    t3 = doc.add_table(rows=6, cols=3)
    t3.alignment = WD_TABLE_ALIGNMENT.CENTER
    t3_data = [
        ["Dimension", "SSD / SalUn / GA", "ASUC-SOM"],
        ["Setting", "Single isolated request", "Continuous stream, 20+ rounds"],
        ["Strategy", "One fixed algorithm", "Dynamic three-tier routing"],
        ["Geometry", "Not tracked", "Orthonormal basis, SOM score"],
        ["Safety", "No post-update check", "Verify, roll back, escalate"],
        ["Auditability", "Benchmark metrics", "Hash-chained per-batch record"]
    ]
    for r_i, row in enumerate(t3_data):
        for c_i, val in enumerate(row):
            cell = t3.cell(r_i, c_i)
            cell.text = val
            p_c = cell.paragraphs[0]
            p_c.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_i > 0 else WD_ALIGN_PARAGRAPH.CENTER
            p_c.paragraph_format.space_before = Pt(2)
            p_c.paragraph_format.space_after = Pt(2)
            style_run(p_c.runs[0], size_pt=8.5, bold=(r_i == 0))
            set_cell_border(cell, top={"sz": 4, "val": "single", "color": "000000"},
                                  bottom={"sz": 4, "val": "single", "color": "000000"})

    add_h2("C. Discussion")
    add_p("Two design choices deserve comment. The first is why SOM is computed on the gradient. An alternative would be to try the update, measure the damage, and undo it if it is too large. That is what the verification gate does anyway, but it costs a full application of the primitive each time. SOM lets the router skip the doomed attempt, so in the regime where overlap is high we avoid paying for a fast update that was going to be thrown away. The gate is still necessary, because the SOM score is a prediction and predictions can be wrong.")
    add_p("The second is the role of the entropy term. Overlap with earlier edits is not the only reason a request is difficult. A concentrated request can be hard even on a fresh model, since the forgotten class shares features with the retained ones. Sending these to Tier 2 regardless of s is a hedge that costs some speed in exchange for the retain-loss anchor.")

    add_h2("D. Practical notes")
    add_p("Two operational points follow from the design. Checkpoint storage is the main memory overhead, since a copy is needed before each round; in practice only the latest verified checkpoint must be kept, plus whatever the audit policy requires. And because the gate is cheap relative to any training-based tier, running it even on Tier 1 outputs adds little to the total, while giving the operator a reason to trust the fast path.")

    # ── Section VII: Limitations ─────────────────────────────────────
    add_h1("VII. LIMITATIONS")
    add_p("We want to be direct about the weaknesses of the current design. First, the chance-level criterion on forget accuracy fits class-level deletion well. For a random sample drawn from all classes, a model retrained without those samples would still classify many of them correctly, since it generalizes. Demanding 10% there is stricter than the retrained model itself would pass. For such requests the better reference is the forget accuracy of a retrained model, or the gap to test accuracy, and this is a change we intend to make. Second, the retain bound in (2) is per round. Twenty rounds each losing just under 3 points would still add up, so a cumulative budget is needed alongside it.")
    add_p("Third, the thresholds 0.40, 0.80 and 0.40 are fixed defaults, not learned or calibrated per model, and they probably do not transfer unchanged to other architectures or datasets. Fourth, the evaluation so far is limited to CIFAR-10, which is small and has few classes. Fifth, membership-inference results are themselves noisy and sensitive to the attack used [8], [17], so an AUC near 0.50 is evidence and not proof. For the same reason the audit record should not be read as a guarantee of erasure in the formal sense used by certified-removal work. Finally, SOM requires a gradient and a basis in parameter space; for billion-parameter models that will require the layer-wise or projected variants mentioned in Section IV-C.")

    # ── Section VIII: Conclusion ─────────────────────────────────────
    add_h1("VIII. CONCLUSION")
    add_p("Regulatory deletion is a continuing process, and unlearning methods that are only tested on one request at a time say little about how a model behaves after many. We presented ASUC-SOM, a controller that watches the geometry of repeated edits with a Subspace Overlap Monitor, routes each request to the cheapest primitive that is likely to be safe, and refuses to release a model until it passes forget, utility and membership-inference checks. The arithmetic of the tiers shows that the benefit depends on keeping exact retraining rare, and that is the property the SOM score is meant to protect. The immediate next steps are to complete the multi-seed evaluation, replace the chance criterion for non-class requests, add a cumulative utility budget, and test on larger models and datasets.")

    # ── References ───────────────────────────────────────────────────
    add_h1("REFERENCES")
    refs = [
        "[1] European Parliament and Council, “Regulation (EU) 2016/679 (General Data Protection Regulation),” Art. 17, Official J. Eur. Union, 2016.",
        "[2] Y. Cao and J. Yang, “Towards making systems forget with machine unlearning,” in Proc. IEEE Symp. Security and Privacy, 2015, pp. 463–480.",
        "[3] L. Bourtoule et al., “Machine unlearning,” in Proc. IEEE Symp. Security and Privacy, 2021, pp. 141–159.",
        "[4] J. Foster, S. Schoepf, and A. Brintrup, “Fast machine unlearning without retraining through selective synaptic dampening,” in Proc. AAAI Conf. Artificial Intelligence, vol. 38, 2024, pp. 12043–12051.",
        "[5] C. Fan, J. Liu, Y. Zhang, E. Wong, D. Wei, and S. Liu, “SalUn: Empowering machine unlearning via gradient-based weight saliency in both image classification and generation,” in Proc. Int. Conf. Learning Representations, 2024.",
        "[6] J. Kirkpatrick et al., “Overcoming catastrophic forgetting in neural networks,” Proc. Natl. Acad. Sci. USA, vol. 114, no. 13, pp. 3521–3526, 2017.",
        "[7] A. Golatkar, A. Achille, and S. Soatto, “Eternal sunshine of the spotless net: Selective forgetting in deep networks,” in Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition, 2020, pp. 9304–9312.",
        "[8] A. Thudi, G. Deza, V. Chandrasekaran, and N. Papernot, “Unrolling SGD: Understanding factors influencing machine unlearning,” in Proc. IEEE Eur. Symp. Security and Privacy, 2022, pp. 303–319.",
        "[9] R. Shokri, M. Stronati, C. Song, and V. Shmatikov, “Membership inference attacks against machine learning models,” in Proc. IEEE Symp. Security and Privacy, 2017, pp. 3–18.",
        "[10] V. S. Chundawat, A. K. Tarun, M. Mandal, and M. Kankanhalli, “Can bad teaching induce forgetting? Unlearning in deep networks using an incompetent teacher,” in Proc. AAAI Conf. Artificial Intelligence, vol. 37, 2023.",
        "[11] A. Krizhevsky, “Learning multiple layers of features from tiny images,” Univ. Toronto, Tech. Rep., 2009.",
        "[12] V. Gupta, C. Jung, S. Neel, A. Roth, S. Sharifi-Malvajerdi, and C. Waites, “Adaptive machine unlearning,” in Proc. Adv. Neural Information Processing Systems, vol. 34, 2021.",
        "[13] L. Graves, V. Nagisetty, and V. Ganesh, “Amnesiac machine learning,” in Proc. AAAI Conf. Artificial Intelligence, vol. 35, 2021, pp. 11516–11524.",
        "[14] A. Ginart, M. Guan, G. Valiant, and J. Zou, “Making AI forget you: Data deletion in machine learning,” in Proc. Adv. Neural Information Processing Systems, vol. 32, 2019.",
        "[15] A. Sekhari, J. Acharya, G. Kamath, and A. T. Suresh, “Remember what you want to forget: Algorithms for machine unlearning,” in Proc. Adv. Neural Information Processing Systems, vol. 34, 2021.",
        "[16] G. Saha, I. Garg, and K. Roy, “Gradient projection memory for continual learning,” in Proc. Int. Conf. Learning Representations, 2021.",
        "[17] N. Carlini, S. Chien, M. Nasr, S. Song, A. Terzis, and F. Tramèr, “Membership inference attacks from first principles,” in Proc. IEEE Symp. Security and Privacy, 2022, pp. 1897–1914."
    ]
    for ref in refs:
        p_r = doc.add_paragraph()
        p_r.paragraph_format.left_indent = Inches(0.2)
        p_r.paragraph_format.first_line_indent = Inches(-0.2)
        p_r.paragraph_format.space_after = Pt(2)
        p_r.paragraph_format.line_spacing = 1.0
        run_r = p_r.add_run(ref)
        style_run(run_r, size_pt=8)

    output_path = "ASUC-SOM_IEEE_Paper.docx"
    alt_path = "ASUC-SOM_IEEE_Paper_IEEE.docx"
    doc.save(alt_path)
    print(f"Successfully saved alternate: {alt_path} ({os.path.getsize(alt_path):,} bytes)")
    try:
        doc.save(output_path)
        print(f"Successfully generated pure IEEE two-column paper: {output_path} ({os.path.getsize(output_path):,} bytes)")
    except PermissionError:
        print(f"Notice: {output_path} is currently locked by Word. Created {alt_path} and will update when unlocked.")

if __name__ == "__main__":
    build_ieee_paper()
