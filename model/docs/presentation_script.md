# Presentation Script
**What Can I Cook Tonight? — A Bayesian LDA Recipe Recommender**

> ~5 minutes. Brackets are slide cues.

---

## 1 · The problem

Hi everyone. Our problem is an everyday one: you open the fridge, look at the few
ingredients you have, and ask — what can I actually cook tonight? We built *What Can I Cook
Tonight*: you type in your ingredients, and it returns the Top-5 recipes you can **actually make**.

---

## 2 · Why this model

The key insight: a recipe is like a "document", an ingredient like a "word", and a cuisine
or flavor is a latent **topic** — so we use LDA. We want more than "this dish fits your taste";
we want to say *how confident* we are, so **uncertainty** is central. But there's a hard
constraint: a fully-Bayesian NUTS posterior over 50k recipes simply **does not run**. Our answer
is a pragmatic **hybrid**: fit φ (topic→ingredient) as a **point estimate** with sklearn on the
**full corpus**, approximate φ's uncertainty by **Bootstrap**, and keep the real Bayesian fidelity
where it matters most — the **user-side** topic posterior.

---

## 3 · Detail 1: point estimate + Bootstrap

First detail. We fit φ **once** as a point estimate φ̂ (full 53k corpus, ~25 s). Where does
uncertainty come from? **Bootstrap**: resample recipes with replacement, warm-start from φ̂, refit
50 times, and treat those 50 φ̂_b as pseudo-samples of φ. Every downstream quantity — the user
profile, each recipe's profile, their alignment — is computed *per sample* and averaged, so the
"recommendation with uncertainty" story is fully preserved.

---

## 4 · Detail 2: aligning topic labels

Second — a pitfall we must handle. LDA topics are **exchangeable**: "topic 0" from one
Bootstrap refit has nothing to do with "topic 0" from another (label switching). So before treating
the 50 fits as comparable samples, we re-order each fit's topics back onto φ̂ with the **Hungarian
algorithm** on cosine similarity. After alignment, `phi_samples[:, k, :]` really is "the same topic
k" across draws, which is what makes the downstream KL meaningful.

---

## 5 · Detail 3: choosing K (an honest result)
> (show fig1, model selection)

Third: choosing the number of topics K. Because we can use the whole corpus, we sweep
K∈{4,6,8,10,12} by **held-out perplexity** (train 90%, score 10%). The honest result: perplexity
is **monotone in K**, picking the smallest, K=4 — this corpus genuinely favors few, coarse flavor
topics. We don't pad it for looks: K is a one-line override, and because coverage and rating
dominate the score, the recommendations are robust to K.

---

## 6 · Detail 4: propagating uncertainty
> (show fig2 topic φ + fig3 user posterior)

Fourth — where the Bayes really lands. Note a detail: on 53k recipes, φ is actually **very
well determined** — the Bootstrap spread is tiny (per-element std ≈ 5e-4). So we honestly call it
resampling **stability**, not posterior uncertainty. The genuinely uncertain, genuinely Bayesian
part is the **user side**: from your handful of ingredients we apply Bayes' theorem *per φ-sample*
to get P(topic | ingredients). The nice illustration: the Italian pantry's posterior collapses
*deterministically* onto one topic, while the baker's pantry splits across two — same machinery,
different certainty.

---

## 7 · Detail 5: the composite score
> (show fig4, recommendation uncertainty)

Finally, the ranking score — a product of four terms: **coverage squared** (a dish you can't
make is useless), times an **absolute-overlap bonus** (so a 3-ingredient dish can't game the
ratio to a perfect 1.0), times **flavor alignment** (a KL similarity on the posteriors), times a
**Bayesian-shrunk rating** (sparsely-rated dishes shrink toward the global mean). Each term
targets a concrete failure mode — the score is designed, not guessed.

---

## 8 · Takeaway

In one line: we turned an everyday question into a *methodologically honest* system — φ as a
point estimate to buy full-corpus scale and speed, uncertainty approximated by Bootstrap, and the
real Bayesian fidelity kept where it belongs: the user posterior inferred from a few ingredients,
carried all the way to every recommendation you see. Thank you.

---

## Appendix: likely Q&A

**Q: Why not learn φ with fully-Bayesian NUTS?** It doesn't scale to 50k recipes (a few hundred
already take tens of minutes), and we want the whole corpus. Point estimate + Bootstrap gives us
both full scale and the uncertainty-propagation story.

**Q: Is this still "Bayesian"?** Yes — and the Bayes is where it counts: the user-side topic
posterior is a genuine Bayesian update applied *per φ-sample*. φ is highly determined on a large
corpus, so its small Bootstrap spread is **by design**, and we honestly call it "stability".

**Q: Isn't K=4 too few?** Held-out perplexity honestly supports only a few topics on this
corpus; for finer flavor tags, raise `K` or `vocab_top_n` — it's a real signal of data size, not a bug.

**Q: How do you keep Bootstrap topics comparable?** After each refit we align topics back to the
point estimate φ̂ with the Hungarian algorithm, so "topic k" is consistent across samples — which is
what makes the KL and the uncertainty meaningful.
