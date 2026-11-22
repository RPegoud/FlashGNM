# Feedback on `output/talk_full.mp4`: revision plan v2

This updates `draft_talk_feedback.md` after reviewing the rendered contact sheets, the
current script, the benchmark data, and the reproduction notes.

## Proposed story

> connectome → how one model grows → how a person is fitted → why the reference is slow →
> our algorithm and kernel → benchmark → cost of a replicated study → three scientific
> results → TL;DR and next steps

Proposed rendered order:

1. `talkA_connectome`, without `01_A_open`
2. **new algorithm segment**, adapted from `clip1_wiring`
3. `talkB_parallel`, rewritten to cover only the speedup
4. `clip6_bench/02_B_impls`
5. rewritten cohort-cost beat from `talkC_payoff`; cut its memory beat
6. rewritten `talkD_papers`
7. one new TL;DR / next-steps beat replacing all of `talkE_enables`

The current visual language works well: black background, restrained green, progressive
reveals, and one principal object per frame. New slides should reuse it. Avoid solving
clarity problems by adding paragraphs: several existing frames are already near the limit
once a paper image and chart share the screen.

---

## 1 · Brain connectomes (`talkA_connectome`)

### 1.1 · Opening hook

**Feedback:** Cut `01_A_open`. The large `6.8 days → 76 s` reveal is visually strong, but
the audience cannot yet know what is being fitted or why it matters.

**Plan:** Open directly on `02_B_regions`. The point-cloud-to-brain animation is visually
strong enough to be an opening. Introduce the central question verbally by the end of the
fibres beat:

> “We represent the brain as regions joined by white-matter fibres. We then ask whether a
> simple growth rule can recreate this wiring.”

Move `6.8 days → 76 s` to the cohort-cost section, after the fitting procedure and the
52,020 simulations have been explained.

### 1.2–1.4 · Regions, fibres, matrix

**Feedback:** Keep. These are among the clearest visuals in the talk. The progressive
point-cloud, fibre and matrix transformations explain one object in three representations
without requiring much text.

**Plan:** Keep the rendered beats and captions. Do not add more explanatory labels. The
matrix beat can be shortened if timing becomes tight, but should not be removed by default:
it prepares the audience for the matrices in the implementation section.

### 1.5 · Why study them?

**Feedback:** The left-brain/right-list composition is readable. The third bullet is the
problem: “52,020 networks” is a computational fact, not an answer to the title.

**Plan:** Use this order:

1. **Generative models ask which simple rules can grow brain-like wiring.**
2. **Wiring differs between people, through development and in disorders.**
3. **Fitted parameters have been associated with age, cognition and schizophrenia.**

End verbally with: “But first we need to fit the model.” This should transition directly
to the new algorithm segment.

---

## 2 · How the model and fit work (new segment)

The missing conceptual explanation should use the existing `clip1_wiring` visual system.
Its small graph, highlighted candidate edge, distance curve, matching-index neighbourhood,
and progressive growth are much more legible than introducing the model through the large
probability matrices in `talkB_parallel`.

Do this in **two beats**, not one formula-heavy slide.

### 2.1 · How one network grows

**Visual plan:** Adapt `clip1_wiring`:

1. show the initial seed network;
2. highlight a missing edge;
3. reveal the probability rule;
4. separately highlight distance and matching index;
5. sample one edge and continue growing until `T` edges.

Use

> `P(i,j) ∝ D(i,j)^η · M(i,j)^γ`

where `M` denotes matching index. Do not use `K` here because the next beat contains KS
statistics. Resolve “empty graph or seed network” against the implementation before
rendering; do not present them as interchangeable.

On-screen language:

- **η controls the cost of distance**
- **γ controls preference for shared neighbours**
- **sample one missing edge; stop at the subject's edge count `T`**

The conceptual sentence to say is:

> “The two parameters trade off short connections against connections between regions
> with similar neighbours.”

### 2.2 · How one person is fitted

**Visual plan:** Keep the target connectome on the left. On the right, show a small
`η × γ` landscape building up, rather than another detailed network animation.

Reveal in three steps:

1. grow a simulated network for one `(η, γ)` pair;
2. compare its degree, clustering, betweenness and edge-length distributions with the
   subject;
3. define `E` as the largest of the four KS distances, sweep the landscape, and retain the
   lowest **mean** energy over repeats.

On screen, keep the energy compact:

> `E = max(KS degree, KS clustering, KS betweenness, KS edge length)`

Do not show initialization, growth, termination, energy and grid search simultaneously.
The visuals should leave the audience with two ideas: one network is stochastic, and a fit
therefore requires repeated simulations across a parameter landscape.

---

## 3 · How we make it fast (`talkB_parallel`)

### 3.1 · Sampling

**Feedback:** The side-by-side probability matrices animate well, but the current beat
mixes the scientific algorithm with its implementation. After the new algorithm segment,
this can focus entirely on computation.

**Plan:** Change the heading from “Step 1” to **“Avoid scanning all n² pairs”**. Preserve
the visual comparison:

- reference: sample from all `n²` entries;
- ours: sample a row from row sums, then sample a column within it;
- same distribution, about `2n` reads rather than `n²`.

Avoid presenting `36` versus `324` toy reads as the result; use those counters only as the
animation, with `n² → 2n` as the explanatory endpoint.

### 3.2 · Updating

**Feedback:** The changing rows and columns are visually clear once the audience already
understands the matrix.

**Plan:** Change the heading to **“Update only what changed”**. Say:

> “Adding edge `(i,j)` only changes rows and columns `i` and `j`, so we patch those rows
> and the row sums instead of rebuilding the whole matrix.”

Keep the highlighted cross. It communicates the optimization better than additional text.

### 3.3 · Kernel details

**Feedback:** The existing list is too dense and too implementation-specific for the main
story. The small grey explanatory lines will be difficult to read in a room.

**Plan:** Keep at most three large items:

- **one fused kernel per simulation**;
- **probabilities recomputed, never stored**;
- **compact state: 3 bytes per node pair**.

Move coalesced loads and exact fp64 row sums to backup/Q&A. Mention batching verbally or
in the following beat.

### 3.4 · Simulations side by side

**Feedback:** Keep. The `1,024` versus `16,384` filled rectangles are simple and visually
effective.

**Plan:** End on:

> **Same GPU memory: 16× more simulations at once**

Use “a whole parameter sweep in one launch” verbally. This beat explains capacity; do not
also ask it to explain the overall 24,900× throughput result.

---

## 4 · Benchmark and study cost

### 4.1 · Implementation bars (`clip6_bench/02_B_impls`)

**Feedback:** Keep. The repeated rescaling makes the size of the gain tangible. It is the
best benchmark visual in the deck.

**Plan:** Preserve the final headline, but qualify it precisely:

> **~24,900× higher simulation throughput**  
> `n = 1,000`, `B = 1,024`, same H100

This is `2,217 / 0.089` simulations per second. It is not the end-to-end study speedup.
The reference time is extrapolated from timed steps, so retain that footnote.

### 4.2 · Memory

**Feedback:** Cut `talkC_payoff/01_A_memory` from the main talk. It is a good backup slide,
but it repeats the capacity point already made by the side-by-side simulation beat.

### 4.3 · Why one fit contains 52,020 simulations

**Visual plan:** Replace the current immediate cohort reveal with a short staged count:

> `51 η values × 51 γ values × 20 repeats = 52,020 simulations per person`

Represent the `51 × 51` search as a grid or landscape and stack/echo it 20 times. Avoid a
sentence-sized equation at the top of an otherwise empty frame.

Then reveal:

> **one person at 1,000 regions: 6.8 days → 76 seconds**

Finally zoom out to:

> **70 people: 1.3 years → 1.5 hours**

Footnote: toolbox time is extrapolated and covers simulation; ours includes evaluation.
The end-to-end improvement is about 7,500×, distinct from the 24,900× throughput result.

---

## 5 · Re-testing published findings (`talkD_papers`)

### 5.1 · Intro

**Feedback:** Keep. The three paper covers establish that these are concrete re-tests.

**Plan:** Caption:

> **Three published analyses, re-run on 70 people with 20 simulations per grid point**

Do not imply that the datasets or conventions exactly reproduce each original paper.

### 5.2 · Akarca et al. 2021

**Feedback:** The paper-plus-scatter layout is good, but the current explanatory paragraph
is already close to the readable limit. The revision should use a short headline and let
the chart carry the comparison.

**Plan:** Frame it as a null test, not as a direct refutation and not as “tautological”.

Headline:

> **Can the parameter correlation appear without individual differences?**

Chart setup:

> **70 synthetic “twins”, generated from one shared parameter pair**

Result:

> **At 68 regions, fitting noise reproduces a strong η–γ correlation.**  
> **Only at 1,000 regions do real people vary more than the twins.**

Say verbally that distance and matching enter both the growth rule and the fitted network
statistics, which may explain the valley. Treat that as an interpretation, not something
directly tested. The original paper's correlation and ours have opposite signs, and the
data, density and conventions differ, so do not say “the same correlation” without a
caveat. The defensible conclusion is that a correlation by itself is not evidence of
individual differences.

Speedup connection: this null requires 70 full synthetic landscapes in addition to the 70
real-subject landscapes.

### 5.3 · Oldham et al. 2022

**Feedback:** The bar chart is readable, but the labels “reported”, “same point” and
“replicated grid” require too much inference. Make the winner's-curse sequence explicit.

**Plan:** Use a two-step headline:

> **A single-run search selects the luckiest simulation.**  
> **Generate 100 fresh networks there, and the apparent minimum disappears.**

Then show the three bars with plainer labels:

- **selected single run**;
- **100 fresh runs at the selected parameters**;
- **best replicated fit**.

Result line:

> **For 56/70 people, the replicated grid finds a better expected fit.**

Check that `56/70` is tied to the exact rule and resolution shown. The broader reproduction
summary varies from `51–64/70` across rules and scales. Say “uncertainty” or “error bars”,
not “significance”. Also state verbally that this concerns the search's internal minimum;
the paper's headline results used cross-validation.

### 5.4 · Oldham, Fornito & Ball 2025

**Feedback:** The paper-plus-distance-bars composition is visually effective. The current
text is too compressed, but adding a long quotation to the main frame would make it worse.

**Plan:** Use the paper's computational limitation as the setup, paraphrased on screen:

> **Their main analysis used 200 left-hemisphere regions partly to reduce compute.**

Then:

> **We extended the test to 1,000 regions, whole brain.**

Result:

> **The model remains at chance across distance; other people's brains predict the wiring
> up to 90× better.**

Keep the exact quotation in speaker notes or a backup slide, not beside the chart. State
that only the classic rule was run at 1,000 regions; their additive rule was tested up to
219. Use one source consistently for the multiplier: the current slide's `90×` comes from
`science_s3.json`, while another summary reports a `3–125×` range.

Speedup connection: this is the resolution and whole-brain test that was previously
computationally impractical.

---

## 6 · Final TL;DR and next steps

**Feedback:** Replace both `talkE_enables` beats. The current list contains good ideas but
feels detached from the evidence, and the slogan close adds no information.

**Visual plan:** One final held frame. Keep the hierarchy strong and the text short.

Top headline:

> **~24,900× higher simulation throughput**

Supporting scale:

> **70 replicated fits at 1,000 regions: 1.3 years → 1.5 hours**

Three results, one line each:

1. **Parameter correlations can arise from fitting uncertainty.**
2. **Single-run searches select lucky minima.**
3. **Long-range failure persists at 1,000 regions.**

Next steps:

- **PR to the GenerativeNetworkModels toolbox — in preparation**;
- **dense probabilistic connectomes**;
- **per-subject uncertainty**;
- **better similarity measures, including DeltaCon**.

Use “in contact with the toolbox maintainers” as a small footer or spoken point. Do not
name or quote private correspondents without Francesco's approval.

End on this frame; no separate thank-you slogan is needed.

---

## Timing and cuts

The new algorithm explanation is essential and will add time. Recover it by:

- cutting the original hook;
- cutting the main-talk memory beat;
- reducing kernel tricks from five to three;
- replacing both conclusion beats with one;
- keeping each paper to one setup, one result and one caveat spoken aloud.

If more time is needed, shorten the matrix representation beat and kernel-details beat.
Do not cut the model-growth explanation, the 52,020 breakdown, or the distinction between
single-run and replicated fitting: those are now the causal spine of the talk.

## Accuracy checklist before rendering

- Confirm the exact initial seed/initialization used by the implementation.
- Use `M` for matching index so it is not confused with KS statistics.
- Keep `24,900×` attached to simulation throughput under matched benchmark conditions.
- Keep `6.8 days → 76 s` and `1.3 years → 1.5 h` attached to the extrapolation caveat.
- Verify which rule and scale produce the displayed `56/70` result.
- Describe the Akarca result as a null showing the correlation **can** arise from fitting;
  do not claim an exact replication or established causal mechanism.
- State that only the classic rule reached 1,000 regions in the long-range analysis.
- Keep `90×` sourced to the plotted `science_s3.json` analysis, or change both plot and
  wording together if a different summary is chosen.
