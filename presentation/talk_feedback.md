# Feedback on `output/talk_full.mp4` (134 s)

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

### 5.1 · 2:00–2:10 · New questions

- **Caption:** "Questions that used to be too expensive to ask"
- **On screen:** 1.5 h: whole studies, every fit replicated · 13 s: dense connectomes · per person: honest uncertainty · every rule: fair model comparison
- **Source:** `output/talkE_enables/01_A_enables.mp4`
- **Feedback: this is cringe. we can just tldr. main speedup number and the three paper results summarized. next steps need to be listed: we'll make a PR on the Generative Network Models Toolbox repo on github and we've already corresponded with its authors**

### 5.2 · 2:10–2:14 · Close

- **On screen:** "Every brain. The whole landscape. Replicated." · "52,020 simulations per person, in about a minute" · "Thank you"
- **Source:** `output/talkE_enables/02_B_close.mp4`
- **Feedback: cringe, don't need this. we'll just close with the previous slide**
