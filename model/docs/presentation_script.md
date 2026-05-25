# 演讲稿 / Presentation Script
**今晚能做什么菜?— 贝叶斯 LDA 食谱推荐器 / What Can I Cook Tonight? — A Bayesian LDA Recipe Recommender**

> 约 5 分钟,中英双语对照(每段先中文、后英文)。括号里是放图提示。
> ~5 minutes, bilingual (Chinese then English per beat). Brackets are slide cues.

---

## 1 · 问题 / The problem

🇨🇳 大家好。我们的问题很日常:打开冰箱,看着手里这点配料,今晚到底能做什么菜?我们做了一个叫
"今晚能做什么菜"的推荐器——你输入手头的配料,它返回你**真正做得出来**的 Top-5 食谱。

🇬🇧 Hi everyone. Our problem is an everyday one: you open the fridge, look at the few
ingredients you have, and ask — what can I actually cook tonight? We built *What Can I Cook
Tonight*: you type in your ingredients, and it returns the Top-5 recipes you can **actually make**.

---

## 2 · 为什么这样选模型 / Why this model

🇨🇳 关键洞察是:一份食谱就像一篇"文档",一种配料就像一个"词",而菜系或风味就是背后的**潜在主题**。
所以我们用 LDA 来学"风味主题"。我们想要的不只是"这道菜配你口味",还要说出"我们有多确定"——所以
**不确定性**是核心。但有个现实约束:对 50k 食谱做全贝叶斯 NUTS 后验**根本跑不动**。我们的方案是
一个**务实的混合模型**:用 sklearn 在**全量语料**上把 φ(主题→配料)学成**点估计**,再用
**Bootstrap** 近似 φ 的不确定性,并把真正的贝叶斯保真度放在最该放的地方——**用户侧**的主题后验。

🇬🇧 The key insight: a recipe is like a "document", an ingredient like a "word", and a cuisine
or flavor is a latent **topic** — so we use LDA. We want more than "this dish fits your taste";
we want to say *how confident* we are, so **uncertainty** is central. But there's a hard
constraint: a fully-Bayesian NUTS posterior over 50k recipes simply **does not run**. Our answer
is a pragmatic **hybrid**: fit φ (topic→ingredient) as a **point estimate** with sklearn on the
**full corpus**, approximate φ's uncertainty by **Bootstrap**, and keep the real Bayesian fidelity
where it matters most — the **user-side** topic posterior.

---

## 3 · 技术细节一:点估计 + Bootstrap 伪后验 / Detail 1: point estimate + Bootstrap

🇨🇳 第一个技术细节。φ 我们只拟合**一次**得到点估计 φ̂(全量 53k 食谱,约 25 秒)。那不确定性从哪
来?**Bootstrap**:对食谱有放回重采样,从 φ̂ 暖启动重拟合 50 次,这 50 个 φ̂_b 就当作 φ 的"伪后验
样本"。下游每个量——用户画像、食谱画像、两者对齐度——都**逐样本**算完再平均,所以"带不确定性的推荐"
这条主线完整保留。

🇬🇧 First detail. We fit φ **once** as a point estimate φ̂ (full 53k corpus, ~25 s). Where does
uncertainty come from? **Bootstrap**: resample recipes with replacement, warm-start from φ̂, refit
50 times, and treat those 50 φ̂_b as pseudo-samples of φ. Every downstream quantity — the user
profile, each recipe's profile, their alignment — is computed *per sample* and averaged, so the
"recommendation with uncertainty" story is fully preserved.

---

## 4 · 技术细节二:主题标签对齐 / Detail 2: aligning topic labels

🇨🇳 第二点,一个必须处理的坑。LDA 主题是**可交换**的:每次 Bootstrap 重拟合出来的"主题 0"和上一次
的"主题 0"毫无关系(label switching)。所以在把这 50 次拟合当成可比的样本之前,我们用**匈牙利算法**
在余弦相似度上把每次的主题重新排回 φ̂。对齐之后,`phi_samples[:, k, :]` 跨样本才真的是"同一个主题
k",后面的 KL 对齐才有意义。

🇬🇧 Second — a pitfall we must handle. LDA topics are **exchangeable**: "topic 0" from one
Bootstrap refit has nothing to do with "topic 0" from another (label switching). So before treating
the 50 fits as comparable samples, we re-order each fit's topics back onto φ̂ with the **Hungarian
algorithm** on cosine similarity. After alignment, `phi_samples[:, k, :]` really is "the same topic
k" across draws, which is what makes the downstream KL meaningful.

---

## 5 · 技术细节三:用留出 perplexity 选 K(诚实结论)/ Detail 3: choosing K (an honest result)
> (放 fig1 模型选择图 / show fig1, model selection)

🇨🇳 第三点:主题数 K 怎么选。因为能上全量,我们在全语料上用**留出 perplexity**(训练 90%、打分 10%)
扫 K∈{4,6,8,10,12}。诚实的结果是:perplexity 对 K **单调递增**,选出最小的 K=4——这点语料本身就只
偏好少而粗的风味主题。我们没有为了好看硬塞更多主题:K 可以一行覆盖,而且因为覆盖率和评分主导分数,
推荐结果对 K 很稳健。

🇬🇧 Third: choosing the number of topics K. Because we can use the whole corpus, we sweep
K∈{4,6,8,10,12} by **held-out perplexity** (train 90%, score 10%). The honest result: perplexity
is **monotone in K**, picking the smallest, K=4 — this corpus genuinely favors few, coarse flavor
topics. We don't pad it for looks: K is a one-line override, and because coverage and rating
dominate the score, the recommendations are robust to K.

---

## 6 · 技术细节四:不确定性传播 / Detail 4: propagating uncertainty
> (放 fig2 主题 φ + fig3 用户后验 / show fig2 topic φ + fig3 user posterior)

🇨🇳 第四点,也是贝叶斯的真正落点。注意一个细节:在 53k 食谱上,φ 其实被**确定得很好**——Bootstrap
的离散度很小(每元素标准差约 5e-4)。所以我们诚实地把它叫"重采样**稳定性**",而不是后验不确定性。
真正不确定、真正贝叶斯的地方是**用户侧**:你只给几个配料,我们对**每个 φ 样本**用贝叶斯定理算
P(主题|配料)。漂亮的现象是:意式菜篮的后验**确定地**坍缩到单一主题,而烘焙菜篮在两个主题之间分裂
——同一套机制,确定性不同。

🇬🇧 Fourth — where the Bayes really lands. Note a detail: on 53k recipes, φ is actually **very
well determined** — the Bootstrap spread is tiny (per-element std ≈ 5e-4). So we honestly call it
resampling **stability**, not posterior uncertainty. The genuinely uncertain, genuinely Bayesian
part is the **user side**: from your handful of ingredients we apply Bayes' theorem *per φ-sample*
to get P(topic | ingredients). The nice illustration: the Italian pantry's posterior collapses
*deterministically* onto one topic, while the baker's pantry splits across two — same machinery,
different certainty.

---

## 7 · 技术细节五:综合打分 / Detail 5: the composite score
> (放 fig4 推荐不确定性图 / show fig4, recommendation uncertainty)

🇨🇳 最后是排序分数。它是四项相乘:**coverage 的平方**(做不出来的菜没用)× **绝对重合度奖励**
(防止只有 3 个配料的小菜靠比例刷满分)× **风味对齐**(后验上的 KL 相似度)× **贝叶斯收缩评分**
(评分数少的菜向全局均值收缩)。每一项都对应一个具体的失败模式——这套打分不是拍脑袋,而是逐个
问题设计出来的。

🇬🇧 Finally, the ranking score — a product of four terms: **coverage squared** (a dish you can't
make is useless), times an **absolute-overlap bonus** (so a 3-ingredient dish can't game the
ratio to a perfect 1.0), times **flavor alignment** (a KL similarity on the posteriors), times a
**Bayesian-shrunk rating** (sparsely-rated dishes shrink toward the global mean). Each term
targets a concrete failure mode — the score is designed, not guessed.

---

## 8 · 收尾 / Takeaway

🇨🇳 一句话总结:我们把一个日常问题,做成了一个**方法上诚实**的系统——φ 用点估计换来全语料的规模与
速度,不确定性用 Bootstrap 近似,而真正的贝叶斯保真度留在最该有的地方:从几个配料推断的用户后验,
并一路传到你看到的每一条推荐。谢谢。

🇬🇧 In one line: we turned an everyday question into a *methodologically honest* system — φ as a
point estimate to buy full-corpus scale and speed, uncertainty approximated by Bootstrap, and the
real Bayesian fidelity kept where it belongs: the user posterior inferred from a few ingredients,
carried all the way to every recommendation you see. Thank you.

---

## 附:可能的提问 / Appendix: likely Q&A

🇨🇳 **问:为什么不用全贝叶斯 NUTS 学 φ?** 答:它撑不到 50k 食谱(几百篇就要几十分钟),而我们想要
全语料。点估计 + Bootstrap 让我们既上全量又保住不确定性传播这条主线。
🇬🇧 **Q: Why not learn φ with fully-Bayesian NUTS?** It doesn't scale to 50k recipes (a few hundred
already take tens of minutes), and we want the whole corpus. Point estimate + Bootstrap gives us
both full scale and the uncertainty-propagation story.

🇨🇳 **问:那这还算"贝叶斯"吗?** 答:算,而且贝叶斯用在了刀刃上——用户侧的主题后验是对**每个 φ 样本**
做的真正贝叶斯更新。φ 在大语料上确定性很高,所以它的 Bootstrap 离散度小是**设计内**的,我们也如实
叫它"稳定性"。
🇬🇧 **Q: Is this still "Bayesian"?** Yes — and the Bayes is where it counts: the user-side topic
posterior is a genuine Bayesian update applied *per φ-sample*. φ is highly determined on a large
corpus, so its small Bootstrap spread is **by design**, and we honestly call it "stability".

🇨🇳 **问:K=4 是不是太少?** 答:留出 perplexity 在这点语料上诚实地只支持很少的主题;想要更细的风味
标签,加大 `K` 或 `vocab_top_n` 即可——这是数据量的真实信号,不是 bug。
🇬🇧 **Q: Isn't K=4 too few?** Held-out perplexity honestly supports only a few topics on this
corpus; for finer flavor tags, raise `K` or `vocab_top_n` — it's a real signal of data size, not a bug.

🇨🇳 **问:Bootstrap 怎么保证主题可比?** 答:每次重拟合后用匈牙利算法把主题对齐回点估计 φ̂,所以跨
样本的"主题 k"是一致的,KL 和不确定性才有意义。
🇬🇧 **Q: How do you keep Bootstrap topics comparable?** After each refit we align topics back to the
point estimate φ̂ with the Hungarian algorithm, so "topic k" is consistent across samples — which is
what makes the KL and the uncertainty meaningful.
