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
所以我们用 LDA 主题模型来学习"风味主题"。但我们刻意把它做成**全贝叶斯**的——用 NUTS / MCMC
保留 phi 和 theta 的**完整后验**,而不是像常见的 LDA 那样只取一个点估计。为什么?因为这个系统
真正的卖点是"带不确定性的推荐":我们不只说"这道菜配你的口味",还能说出"我们对此有多确定"。
点估计做不到这件事。

🇬🇧 The key insight: a recipe is like a "document", an ingredient like a "word", and a cuisine
or flavor is a latent **topic**. So we use LDA to learn flavor topics. But we deliberately made
it **fully Bayesian** — NUTS/MCMC keeping the *full posterior* of phi and theta, instead of the
usual point-estimate LDA. Why? Because the real value of this system is recommendation *with
uncertainty*: we don't just say "this dish fits your taste", we can say *how confident* we are.
A point estimate can't do that.

---

## 3 · 技术细节一:让 LDA 适配 NUTS / Detail 1: making LDA work with NUTS

🇨🇳 第一个技术细节。经典 LDA 给每个词采一个**离散**主题 z;但 NUTS 是基于梯度的,走不了离散变量。
我们的解法是把 z **解析地积分掉**:每个配料就变成一个概率向量为 `theta @ phi` 的 Categorical。
这和原模型**完全等价**,只是把 z 积掉,只剩下连续的单纯形留给 NUTS——既严谨,又能用梯度采样。

🇬🇧 First technical detail. Classic LDA samples a *discrete* topic z per word; but NUTS is
gradient-based and can't traverse discrete variables. Our fix is to **marginalize z
analytically**: each ingredient becomes a Categorical with probability vector `theta @ phi`.
This is *exactly the same model* with z integrated out, leaving only continuous simplices for
NUTS — rigorous, and gradient-friendly.

---

## 4 · 技术细节二:样本外选 K(最关键的方法论)/ Detail 2: choosing K out-of-sample (the key one)
> (放 fig1 模型选择图 / show fig1, model selection)

🇨🇳 第二点,也是我们最想强调的:怎么选主题数 K。一开始我们用 WAIC,结果它一路把 K 选到网格上限
——因为 WAIC 是**样本内**准则,主题越多越能拟合训练数据,根本没有内部最优;而且 `p_waic` 大得
离谱,说明复杂度惩罚已经不可信。于是我们改成**样本外**:留出 15% 的配料 token,在其余 token 上
拟合,选**留出预测似然**最高的 K。结果非常说明问题:样本内的 WAIC/LOO 想要 K=10,而留出预测在
K=2 就到顶了。这种背离就是过拟合的直接证据,也正是我们坚持样本外选择的理由。

🇬🇧 Second — and the one we most want to highlight: choosing the number of topics K. We started
with WAIC, and it pushed K straight to the top of our grid — because WAIC is *in-sample*: more
topics always fit the training data better, so there's no interior optimum; and a huge `p_waic`
showed the penalty was no longer trustworthy. So we switched to *out-of-sample*: hold out 15% of
ingredient tokens, fit on the rest, and pick the K with the best **held-out predictive
likelihood**. The result is telling: in-sample WAIC/LOO want K=10, while held-out predictive
peaks at K=2. That divergence is direct evidence of overfitting — and exactly why we select
out-of-sample.

---

## 5 · 技术细节三:不确定性传播 / Detail 3: propagating uncertainty
> (放 fig2 主题后验 + fig3 用户后验 / show fig2 topic posterior + fig3 user posterior)

🇨🇳 第三点:不确定性怎么一路传到推荐。因为我们保留的是后验**样本**,所以用户的风味画像、每道菜
的画像、以及两者的对齐度,都是**逐样本**算完再平均的。于是每条推荐都自带一个
`posterior_uncertainty`。一个很漂亮的现象:意式菜篮的后验会**确定地**坍缩到单一主题,而烘焙菜篮
会在两个主题之间分裂、带很宽的可信区间——同一套机制,确定性却不同。

🇬🇧 Third: how uncertainty reaches the recommendation. Because we keep posterior *samples*, the
user's flavor profile, each recipe's profile, and their alignment are all computed *per sample*
and then averaged. So every recommendation carries a `posterior_uncertainty`. A nice
illustration: the Italian pantry's posterior collapses *deterministically* onto one topic, while
the baker's pantry splits across two with wide credible intervals — same machinery, different
certainty.

---

## 6 · 技术细节四:综合打分 / Detail 4: the composite score
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

## 7 · 收尾 / Takeaway

🇨🇳 一句话总结:我们把一个日常问题,做成了一个**方法上诚实**的贝叶斯系统——主题用 NUTS 严格推断,
K 用样本外标准选,不确定性从后验一路带到了你看到的每一条推荐。谢谢。

🇬🇧 In one line: we turned an everyday question into a *methodologically honest* Bayesian system —
topics inferred rigorously with NUTS, K chosen out-of-sample, and uncertainty carried from the
posterior all the way to every recommendation you see. Thank you.

---

## 附:可能的提问 / Appendix: likely Q&A

🇨🇳 **问:为什么不用更快的变分 LDA(如 gensim)?** 答:那只给点估计,拿不到我们要传播的后验不确定性;
我们的核心贡献正是不确定性,所以值得用 MCMC。
🇬🇧 **Q: Why not faster variational LDA (e.g. gensim)?** It only gives a point estimate, not the
posterior uncertainty we propagate — and uncertainty is our core contribution, so MCMC is worth it.

🇨🇳 **问:K=2 是不是太少?** 答:留出准则在这点语料上诚实地只支持 ~2 个主题;想要更细的风味标签,
就加大 `n_train`/`vocab_size` 让样本外准则能支撑更大的 K——这是数据量的真实信号,不是 bug。
🇬🇧 **Q: Isn't K=2 too few?** On this corpus the held-out criterion honestly supports only ~2
topics; for finer flavor tags, grow `n_train`/`vocab_size` so out-of-sample evidence can justify a
larger K — it's a real signal of data size, not a bug.

🇨🇳 **问:ESS 看起来偏低?** 答:跨链 ESS 被 label switching 压低了;我们报告的是保留链的**链内** ESS,
因为主题标签只在单链内一致,而 WAIC/LOO/留出只依赖标签不变的 `theta@phi`,可以汇总所有链。
🇬🇧 **Q: ESS looks low?** Cross-chain ESS is deflated by label switching; we report the *within-chain*
ESS of the kept chain, since topic labels are only coherent within a chain — while WAIC/LOO/held-out
depend only on the label-invariant `theta @ phi` and can pool all chains.
