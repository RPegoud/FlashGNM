# Feedback on `output/talk_full.mp4` (134 s), with a revision plan

This is a copy of `talk_feedback.md` with a **Plan** under each section, saying what we'll change in response to the feedback. Proposed new order:

> connectome (A, without the hook) → **the algorithm, new** → the speedup (B) → benchmark bars → cohort cost with the 52,020 breakdown → three paper re-tests (rewritten text) → **TL;DR and next steps** (replaces E)

**Order:** talk A → talk B → benchmark bars from clip 6 (beat `02_B_impls`) → talk C → talk D → talk E.

**Timestamps** are cumulative beat durations (±0.5 s). The source file lets you re-render one beat (`node scripts/render.mjs <project>`). The captions are the bottom-line text as it appears on screen.

**How to use:** write under **Feedback** for each slide. Leave it empty if the slide is fine.

---

## General

- **Overall pacing / length:**
- **Story arc** (connectome → GPU → speed → re-tests → what it enables):
- **Visual style / readability:**
- **Missing:** for example the toolbox maintainer correspondence, PRs, the TL;DR slide:
- **Anything to cut:**

---

## 1 · Brain connectomes (talk A, `talkA_connectome`)

> **Plan**
> - **Drop the hook (1.1).** The talk opens on 1.2, the regions slide, which explains what a connectome is and how it's measured. The "6.8 days" number moves to the cohort slide (3.3), after the audience knows what is being fitted.
> - **Keep 1.2–1.4 unchanged.**
> - **1.5: swap the first two bullets.** "Generative models ask which simple rules grow a brain-like network" comes first, then "Wiring differs between people, through development and in disorders".
> - **1.5: replace the third bullet** (an implementation fact) with why the field cares about the fitted parameters. Proposed: *"The fitted parameters have been linked to age, cognition and schizophrenia"*. Liu et al. 2023's introduction says parameter variation "has been found to associate with aging (Betzel et al., 2016), cognitive performance (Akarca et al., 2021), and schizophrenia (Zhang et al., 2021)". The 52,020-networks fact moves to 3.3, where it explains the cost.

### 1.1 · 0:00–0:05 · Hook

- **Caption:** "Fitting a brain model to one person at 1,000 regions: 6.8 days →"
- **Source:** `output/talkA_connectome/01_A_open.mp4`
- **Feedback: Unnecessary slide. Audience doesn't even have context on what this is yet.**

### 1.2 · 0:05–0:10 · Regions

- **Caption:** "MRI divides the brain into regions: here 1,000"
- **Source:** `output/talkA_connectome/02_B_regions.mp4`
- **Feedback: Important slide, good to start with. We want to explain what the connectome is / how it's gathered.**

### 1.3 · 0:10–0:18 · Fibres

- **Captions:** "Diffusion MRI traces the white-matter fibres between them", then "Regions + fibres: a connectome, the brain's wiring diagram"
- **Source:** `output/talkA_connectome/03_C_fibres.mp4`
- **Feedback: good**

### 1.4 · 0:18–0:23 · Matrix

- **Caption:** "The same network as a 1,000 × 1,000 matrix: a dot per connection"
- **Source:** `output/talkA_connectome/04_D_matrix.mp4`
- **Feedback: good**

### 1.5 · 0:23–0:33 · Why connectomes matter

- **On screen:** wiring differs between people, through development and in disorders · generative models ask which simple rules grow a brain-like network · fitting one person means simulating 52,020 networks
- **Source:** `output/talkA_connectome/05_E_why.mp4`
- **Feedback: "**Wiring differs between people, through development and in disorders" makes sense; "Generative models ask which simple rules grow a brain-like network" also; but swap the order of the first two; the third bullet is an implementation fact, not really an answer - consider a better one, perhaps from the papers

---

## 2 · Why the GPU makes it possible (talk B, `talkB_parallel`)

> **Plan**
> - **Add a new "the algorithm" segment before any implementation detail.** It explains the model on its own terms, using the clips we already made for the original toolbox:
>   - **Initialisation and growth:** start from an empty graph (or a seed network); add one edge per step.
>   - **Wiring rule:** P(i, j) ∝ D_ij^η · K_ij^γ over non-edges, where K is the matching index (shared neighbours, normalised by degree). Source: `clip1_wiring` (beats growth, distance, matching, sample).
>   - **Termination:** stop when the edge count reaches the subject's own number of edges, T.
>   - **Energy:** E = max(KS_degree, KS_clustering, KS_betweenness, KS_edge length) against the subject's connectome.
>   - **Fit:** sweep (η, γ) and keep the lowest energy. This is **new**: no existing beat shows the energy formula or the fit loop.
>   - **Cost:** the reference toolbox recomputes and samples all n² entries every step. Source: `clip2_naive`.
> - **Then talk B covers how we speed it up,** with the algorithmic change and the kernel separated:
>   - **Algorithm:** two-stage sampling (row, then column), and patching rows and columns i and j instead of recomputing (B steps 1–2, optionally `clip3_chain`).
>   - **Kernel:** fused kernel, P not stored, narrow types, coalesced loads, batching (B 3–4).
> - **Remove the toolbox-vs-kernel split** from 2.1's first frame. It belongs in the speedup part, not the algorithm part.

### 2.1 · 0:33–0:40 · Step 1, sampling

- **Caption:** "Step 1: draw the next connection in proportion to P" (toolbox reads all n² entries; our kernel picks a row, then a column)
- **Source:** `output/talkB_parallel/01_A_sampling.mp4`
- **Feedback: This slide mixes the algorithm with the implementation. we need a slide with the algo with the formulas for energy and probabilities for sampling new edges, termination conditions (edge count) and graph init etc. We can do that using diagrams we made for the original toolbox. Only after we show how we speed it up with the kernel and algorithmic change.**

### 2.2 · 0:40–0:48 · Step 2, update

- **Captions:** "Step 2: add the connection and update P", then "Only rows and columns i and j change, so the row sums are patched, not recomputed"
- **Source:** `output/talkB_parallel/02_B_update.mp4`
- **Feedback:**

### 2.3 · 0:48–0:57 · Kernel tricks

- **Caption:** "What makes each simulation small and fast"
- **On screen:** one fused kernel · P recomputed, never stored · narrow types (3 bytes per pair) · coalesced loads · exact, drift-free sampling
- **Source:** `output/talkB_parallel/03_C_tricks.mp4`
- **Feedback:**

### 2.4 · 0:57–1:04 · Side by side

- **Captions:** "Small simulations fit side by side on one GPU", then "Same memory, 16× more simulations, a whole parameter sweep in one launch (n = 1,000)"
- **Source:** `output/talkB_parallel/04_D_side_by_side.mp4`
- **Feedback:**

---

## 3 · Benchmarks and what the speedup buys

> **Plan**
> - **3.1: keep as is.**
> - **3.2: cut the memory slide.**
> - **3.3: add the breakdown of where 52,020 comes from,** before the "1.3 years → 1.5 h" reveal. Three steps:
>   - **One fit:** a 51 × 51 grid of (η, γ) × 20 repeats per grid point = **52,020 simulations per person**.
>   - **One person at 1,000 regions:** toolbox about 6.8 days, against about 1.3 minutes for us. This is where the old hook number goes.
>   - **The whole study:** × 70 people, 1.3 years → 1.5 h.
>
>   Sources: `results/landscape_griffa_full/meta.json` (grid 51 × 51, 20 repeats) and `summary.csv` (91 min for all 70 subjects at n = 1000).
> - **Footnote on 3.3, for fairness:** the toolbox time is extrapolated from timed steps and covers simulation only. Ours includes evaluation.

### 3.1 · 1:04–1:18 · Implementation bars

- **Captions:** "n = 1000, B = 1024: every implementation on the same H100", then "From ~0.089 to 2,217 simulations per second"
- **Source:** `output/clip6_bench/02_B_impls.mp4` (reused from clip 6)
- **Feedback: excellent**

### 3.2 · 1:18–1:27 · Memory

- **Captions:** "Each simulation needs about 15× less GPU memory", then "So far more simulations fit on one GPU at once" (at 2,000 regions the toolbox runs out of memory at 1,024 simulations; we run 4,096)
- **Source:** `output/talkC_payoff/01_A_memory.mp4`
- **Feedback: not an important slide; can skip**

### 3.3 · 1:27–1:33 · Cohort cost

- **On screen:** "A whole study: 70 people × 1,000 regions × 52,020 networks each: 1.3 years → 1.5 h"
- **Caption:** "What previously took over a year now runs over lunch"
- **Source:** `output/talkC_payoff/02_B_cohort.mp4`
- **Feedback: this is good but need to break down why 52k networks each, i.e. 51x51 hyperparam grid with 20 samples per configuration perhaps?**

---

## 4 · Re-testing published findings (talk D, `talkD_papers`)

> **Plan.** Each paper slide gets a plain-language headline, plus one line on what the speedup made possible.
> - **4.1: keep.**
> - **4.2 Akarca, reframed as a "reverse test":**
>   - **Headline:** *"Akarca et al. fit two parameters to each child and find they correlate across people. We ran it in reverse: pick one fixed pair of parameters, grow 70 synthetic 'twins', fit them the same way, and the same correlation appears."*
>   - **Takeaway:** *"At 68 regions the correlation comes from the method (the model trades distance against clustering, and the energy scores exactly those), not from individuals. Only at 1,000 regions do real people spread more than their twins."*
>   - **Speedup line:** this needs 70 full replicated landscapes for the twins as well as the real subjects.
>   - **Wording caveat:** keep "comes from the method", and avoid "tautological" on the slide. Our data, density and conventions differ from Akarca's, and the "why" (distance vs clustering in both the rule and the energy) is our interpretation, not something we tested directly.
> - **4.3 Oldham 2022, clearer:**
>   - **Headline:** *"Their search picks the best of 10,000 single runs. Re-running that chosen setting 100 times scores much worse, for 56 of 70 people: the reported fit was a lucky draw."*
>   - **Speedup line:** *"With 20 runs per grid point we don't need luck, and every fit gets an error bar."*
>   - **Accuracy:** this is about the search's internal minimum (the paper's headline numbers are cross-validated), tested at 68 and 219 regions.
> - **4.4 Oldham, Fornito & Ball 2025, clearer and framed as "done for them":**
>   - **Headline:** *"They found models miss long-range connections, but limited their analysis to 200 regions to reduce compute. We ran it at 1,000 regions, whole brain: the result holds, and the best model is at chance at every distance, while other people's brains predict up to 90× better."*
>   - **Quote for the slide** (Methods, "Empirical Brain Networks"): *"We primarily used data for the left hemisphere, that is, 200 cortical regions in total as to: (a) reduce computational complexity (more nodes/connections in the network increases the time needed to run the GNM)…"*
>   - **Second quote, optional:** Oldham et al. 2022 limited themselves to 100 participants *"because of the computational burden of running multiple different models for each participant's network"*.
>   - **Number check:** the slide's "up to 90×" matches `src/data/science_s3.json` (100–120 mm bin at n = 1000). `FINDINGS.md` reports 30–125× from a different summary, so use one source consistently.
>   - **Scope:** only the classic rule was run at 1,000 regions; their own rule form was run up to 219.

### 4.1 · 1:33–1:37 · Intro

- **Caption:** "We re-ran three published analyses with every fit replicated 20×, on 70 people"
- **Source:** `output/talkD_papers/01_A_intro.mp4`
- **Feedback: good**

### 4.2 · 1:37–1:46 · Akarca et al. 2021: individual parameters

- **Captions:** "Akarca et al.: individual wiring parameters" → "At 68 regions, the fitting method alone produces this pattern" → "Only at 1,000 regions do real people spread more than their twins"
- **Source:** `output/talkD_papers/02_B_akarca.mp4`
- **Check:** our data, density and conventions differ from Akarca's (`reproductions.md`). The twins show the pattern even with 20 runs per grid point, because each target is one random network.
- **Feedback: nice slide but it's not entirely related to the speedup, more a criticism of the GNM science overall. If I had to summarize: the Akarca paper fits the two parameters to individuals and shows they correlate. But with our sampler we can consider the opposite: what if we pick some parameters, generate graphs, then fit parameters to those. we see the same correlation then. this suggests the correlation is tautological since we're using a graph construction method that trades-off edge length with homofily which creates clusters, then we fit parameters based on an energy function that combines clustering scores with edge lengths.**

### 4.3 · 1:46–1:53 · Oldham et al. 2022: single-run search

- **Caption:** "Oldham et al.: fitting with a single-run search"
- **On screen:** its best fit was luck. For 56 of 70 people, fresh networks at that point score worse than the replicated grid's best.
- **Source:** `output/talkD_papers/03_C_oldham2022.mp4`
- **Check:** Oldham 2022's headline numbers are cross-validated over 20 runs, so "luck" applies to the search's internal minimum. This was tested at 68 and 219 regions only.
- **Feedback: the text is correct but hard to understand. we need to make it clearer, something like: the paper's grid search was lucky as rerunning their approach 100 times usually results in much worse fits. with our faster implementation we don't need to be lucky and can properly measure significance of the fits.**

### 4.4 · 1:53–2:00 · Oldham, Fornito & Ball 2025: long range

- **Captions:** "Oldham, Fornito & Ball: long-range connections", then "The models match network statistics, not the actual wiring" (at 1,000 regions, the best-fitting model is at chance at every distance; other people's brains do up to 90× better)
- **Source:** `output/talkD_papers/04_D_oldham2025.mp4`
- **Check:** `FINDINGS.md` gives 30–125× above chance at 1,000 regions for the group consensus. Confirm where "90×" comes from.
- **Feedback: ok, but again the text is too concise and hard to decipher. basically we're saying we reproduced their results but the paper (find quote pls) says they did not have the means to test at 1000 node networks. we did it for them with the faster code**

---

## 5 · What this enables (talk E, `talkE_enables`)

> **Plan**
> - **Replace 5.1 with a TL;DR slide,** which is also the final slide (no separate close). It has three parts:
>   - **Speed:** one headline number, about 24,900× more simulations per second than the toolbox at n = 1000, B = 1024, on one H100. The bench clip's end frame already uses this, so keep the same number.
>   - **Three results, one line each:**
>     1. Akarca: at 68 regions, synthetic twins with identical parameters show the same η–γ correlation as real people.
>     2. Oldham 2022: single-run searches report lucky fits; with replicates, 56/70 people get a better fit.
>     3. Oldham 2025: models miss long-range wiring even at 1,000 regions, the resolution they couldn't afford.
>   - **Next steps:**
>     - **Pull request:** open a PR on the GenerativeNetworkModels toolbox repo on GitHub. Planned, so write "PR in preparation" until it's open.
>     - **Maintainers:** we've already corresponded with the toolbox authors.
>     - **Their suggestions:** dense probabilistic connectomes, per-subject uncertainty, and better similarity measures (Dendorfer et al. 2026).
> - **Before naming anyone or quoting the emails** on the slide, ask Francesco. Until then, write "in contact with the toolbox maintainers".
> - **Drop 5.2,** the "Every brain…" close.

### 5.1 · 2:00–2:10 · New questions

- **Caption:** "Questions that used to be too expensive to ask"
- **On screen:** 1.5 h: whole studies, every fit replicated · 13 s: dense connectomes · per person: honest uncertainty · every rule: fair model comparison
- **Source:** `output/talkE_enables/01_A_enables.mp4`
- **Feedback: this is cringe. we can just tldr. main speedup number and the three paper results summarized. next steps need to be listed: we'll make a PR on the Generative Network Models Toolbox repo on github and we've already corresponded with its authors**

### 5.2 · 2:10–2:14 · Close

- **On screen:** "Every brain. The whole landscape. Replicated." · "52,020 simulations per person, in about a minute" · "Thank you"
- **Source:** `output/talkE_enables/02_B_close.mp4`
- **Feedback: cringe, don't need this. we'll just close with the previous slide**
