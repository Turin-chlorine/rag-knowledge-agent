检索增强⽣成
检索增强⽣成（英语：Retrieval-augmented generation，缩写：RAG）是⼀种使⼤语⾔模型
（LLM）能够从外部数据源中检索并集成新信息的技术。[1] 在RAG框架下，⼤语⾔模型⾸先查阅
指定的⽂档集合，再对⽤⼾查询作出回应。这些⽂档补充了模型预训练数据中已有的信息[2]，使模
型能够利⽤训练数据中未包含的领域特定信息和/或最新信息[2]。例如，该技术可帮助基于⼤语⾔模
型的聊天机器⼈访问企业内部数据，或依据权威来源⽣成回答。
RAG通过在⽣成回答前引⼊信息检索环节来提升⼤语⾔模型的性能[3]。与仅依赖静态训练数据的模
型不同，RAG能够从数据库、上传⽂档或⽹络资源中提取相关⽂本[1]。据《Ars Technica》报道：
“RAG本质上是将⼤语⾔模型的处理流程与⽹⻚搜索或其他⽂档查询过程相结合，从⽽帮助模型更
准确地遵循事实，以此提升其表现。”该⽅法有助于减少⼈⼯智能幻觉[3]——此类问题曾导致聊天机
器⼈描述并不存在的政策，或向寻求判例⽀持的律师推荐虚构的法律案例[4]。
RAG还降低了为纳⼊新数据⽽重新训练⼤语⾔模型的需求，从⽽节省计算资源与资⾦成本[1]。除提
升效率外，RAG还使模型能在回答中附带信息来源，便于⽤⼾核查引⽤内容。这种机制增强了透明
度，⽤⼾可通过⽐对检索到的原始内容来验证回答的准确性与相关性。
RAG这⼀术语最早出现于2020年发表的⼀篇研究论⽂[3]。
RAG与LLM的局限性
⼤语⾔模型可能提供错误信息。例如，⾕歌⾸次演⽰其⼤语⾔模型⼯具 Google Bard（后更名为
Gemini）时，该模型就詹姆斯·⻙布空间望远镜提供了错误信息。这⼀失误导致该公司的市值蒸发
约1000亿美元。[4] RAG可⽤于预防此类错误，但并不能解决所有问题。例如，即使检索来源本⾝
事实准确，若⼤语⾔模型误解了上下⽂，仍可能⽣成错误信息。《⿇省理⼯科技评论》举了⼀个例
⼦：某AI⽣成的回答称“美国曾有⼀位穆斯林总统，即巴拉克·侯赛因·奥巴⻢”。该模型检索⾃⼀本
学术著作，其修辞性书名为Barack Hussein Obama: America's First Muslim President?，但⼤
语⾔模型未能理解书名中的疑问语⽓，从⽽⽣成了错误陈述。[2]
采⽤RAG的⼤语⾔模型被设计为优先使⽤新检索到的信息。这种技术被称为“提⽰词填充”
（prompt stuffing）。在没有提⽰词填充的情况下，⼤语⾔模型的输⼊仅由⽤⼾⽣成；⽽采⽤提⽰
词填充时，系统会在此输⼊中额外添加相关上下⽂以引导模型⽣成回答。该⽅法在提⽰词开头即提
供关键信息，促使模型优先采⽤所供给的数据，⽽⾮依赖预训练知识。[5]
流程
RAG通过引⼊信息检索机制来增强⼤语⾔模型的能⼒，使模型能够访问并利⽤原始训练集之外的额
外数据。Ars Technica指出：“当新信息出现时，⽆需重新训练模型，只需⽤更新后的信息扩充模
型的外部知识库即可”（即“增强”）。[4] IBM表⽰：“在⽣成阶段，⼤语⾔模型会结合增强后的提⽰
词及其对训练数据的内部表征来综合⽣成回答”。[1]
关键阶段
通常，待引⽤的数据会被转换为⼤语⾔模型的嵌⼊向量
（embeddings），即以⾼维向量空间形式表⽰的数值化表
征。RAG可应⽤于⾮结构化（通常为⽂本）、半结构化或结
构化数据（例如知识图谱）。这些嵌⼊向量随后被存储在向
量数据库中，以⽀持⾼效的⽂档检索。
当⽤⼾提出查询时，系统⾸先调⽤⽂档检索器，选取与查询
最相关的⽂档⽤于增强原始输⼊。[2][3] 此类相关性⽐对可
采⽤多种⽅法实现，具体取决于所使⽤的索引类型。[1] 模
型通过提⽰⼯程对⽤⼾原始查询进⾏增强，将检索到的相关
信息⼀并输⼊⼤语⾔模型。较新的实现⽅案（截⾄2023年）
还可集成专⽤的增强模块，具备将查询扩展⾄多个领域、利 RAG流程概览：将外部⽂档与⽤⼾输⼊结
⽤记忆机制及⾃我改进能⼒从历史检索中持续学习等功能。 合形成⼤语⾔模型提⽰词，以⽣成定制化
输出
最终，⼤语⾔模型综合查询内容与检索到的⽂档⽣成输出结
果。[2][6] 部分模型还引⼊额外优化步骤以提升输出质量，
例如对检索结果进⾏重排序（re-ranking）、上下⽂筛选以及微调（finetuning）。
改进
上述基础流程可在RAG⼯作流的不同阶段进⾏优化。
编码器
这些⽅法聚焦于将⽂本编码为稠密向量或稀疏向量。稀疏向量⽤于编码词语⾝份，通常具有词典⻓
度，且⼤部分元素为零。稠密向量则⽤于编码语义，结构更为紧凑且零值较少。多种增强技术可改
进向量存储（数据库）中相似度的计算⽅式。[7]
通过优化向量相似度计算可提升性能。点积可增强相似度评分，⽽近似最近邻搜索
[8]
（ANN）相⽐K近邻算法（KNN）能提⾼检索效率。
晚交互（Late Interactions）技术可在检索后更精确地进⾏词语⽐对，有助于优化⽂档排序并提
[9]
升搜索相关性，从⽽提⾼准确性。
混合向量⽅法可将稠密向量表⽰与稀疏独热编码向量相结合，利⽤稀疏点积在计算效率上优于稠
[7]
密向量运算的优势。
其他检索技术通过优化⽂档选择⽅式来提升准确性。部分⽅法将SPLADE等稀疏表⽰与查询扩展
[10]
策略相结合，以提⾼搜索准确率与召回率。
以检索器为中⼼的⽅法
这些⽅法旨在提升向量数据库中⽂档检索的质量：
通过“逆完形填空任务”（ICT）对检索器进⾏预训练，该技术通过预测⽂档中被遮蔽的⽂本来帮
[11]
助模型学习检索模式。
有监督的检索器优化使检索概率与⽣成模型的似然分布对⻬。具体做法是：针对给定提⽰检索前
k个向量，评估⽣成回答的困惑度，并通过最⼩化检索器选择结果与模型似然分布之间的KL散
度来优化检索效果。[12]
重排序技术可在训练过程中优先选择最相关的检索⽂档，从⽽优化检索器性能。[13]
语⾔模型
RAG专⽤的Retro语⾔模型。每个Retro模块由注
意⼒层、分块交叉注意⼒层和前馈⽹络层组成。⿊
⾊⽂字框表⽰正在被处理的数据，蓝⾊⽂字表⽰执
⾏处理的算法。
通过针对检索器重新设计语⾔模型，⼀个规模缩⼩25倍的⽹络即可获得与更⼤模型相当的困惑度表
现。[14] 由于该⽅法（Retro）需从头开始训练，因此产⽣了原始RAG⽅案所避免的⾼昂训练成
本。其假设是：在训练过程中注⼊领域知识后，Retro⽆需过多关注领域细节，可将有限的参数资
源专注于语⾔语义建模。重新设计的语⾔模型结构如图所⽰。
有报告指出Retro难以复现， 因此研究者对其进⾏了改进以提升可复现性。改进后的版本称为
Retro++，并集成了上下⽂内RAG能⼒。[15]
分块策略
分块涉及将数据切分为向量的各种策略，以便检索器能够从中定位细节信息。
不同数据格式具有特定模式，合理的分块策略可充分利⽤这些模式。
三种主要分块策略包括：
固定⻓度重叠分块：实现简单⾼效。相邻分块间的重叠有助于维持跨块的语义连贯性。
基于语法的分块：可将⽂档按句⼦边界切分。spaCy或NLTK等库可辅助实现。
基于⽂件格式的分块：特定⽂件类型具有天然分块结构，应予以尊重。例如，代码⽂件宜以完整
函数或类为单位进⾏分块与向量化；HTML⽂件应保持<table>标签或base64编码的<img>元素
完整；PDF⽂件也需考虑类似因素。Unstructured或LangChain等库可协助实现此类分块。
混合搜索
有时仅依靠向量数据库搜索可能遗漏回答⽤⼾问题所需的关键事实。⼀种缓解策略是：先执⾏传统
⽂本搜索，将结果与向量搜索返回的⽂本块合并，再将融合后的混合⽂本输⼊语⾔模型进⾏⽣成。
评估与基准测试
RAG系统通常使⽤专⻔设计的基准测试进⾏评估，以检验可检索性、检索准确率与⽣成质量。常⽤
数据集包括BEIR（涵盖多领域信息检索任务的测试包）以及Natural Questions或Google QA
（⽤于开放域问答评估）。
挑战
RAG并不能完全防⽌⼤语⾔模型产⽣幻觉。据Ars Technica报道：“这（RAG）并⾮直接解决⽅
案，因为⼤语⾔模型在回应中仍可能围绕检索到的源材料产⽣幻觉。”[4]
尽管RAG提升了⼤语⾔模型（LLM）的准确性，但它并未消除所有挑战。其局限性之⼀在于：虽然
RAG减少了频繁重新训练模型的需求，但并未完全消除这⼀需求。此外，⼤语⾔模型可能难以识别
⾃⾝是否缺乏⾜够信息来提供可靠回答。若未经专⻔训练，模型即便在应当表明不确定性的情况下
仍可能⽣成答案。据IBM指出，当模型缺乏评估⾃⾝知识局限性的能⼒时，此类问题便可能出
现。[1]
RAG投毒
RAG系统可能检索到事实正确但具有误导性的来源，从⽽导致解读错误。在某些情况下，⼤语⾔模
型可能从来源中提取语句⽽未考虑其上下⽂，进⽽得出错误结论。此外，当⾯对相互⽭盾的信息
时，RAG模型可能难以判断哪个来源更为准确。此类局限性的最坏结果是，模型可能融合多个来源
的细节，⽣成将过时信息与更新信息以误导性⽅式混合的回答。据《⿇省理⼯科技评论》指出，这
些问题的产⽣源于RAG系统可能对其检索到的数据产⽣误读。[2]
参考
1. What is retrieval-augmented generation?.
IBM. 22 August 2023 [7 March 2025].
2. Why Google's AI Overviews gets things
wrong. MIT Technology Review. 31 May
2024 [7 March 2025].
3. Lewis, Patrick; Perez, Ethan; Piktus, 9. Khattab, Omar; Zaharia, Matei. ColBERT:
Aleksandra; Petroni, Fabio; Karpukhin, Efficient and Effective Passage Search via
Vladimir; Goyal, Naman; Küttler, Heinrich; Contextualized Late Interaction over BERT
Lewis, Mike; Yih, Wen-tau; Rocktäschel, . Proceedings of the 43rd International
Tim; Riedel, Sebastian; Kiela, Douwe. ACM SIGIR Conference on Research and
Retrieval-augmented generation for Development in Information Retrieval.
knowledge-intensive NLP tasks. 2020: 39‒48. ISBN 978-1-4503-8016-4.
International Conference on Neural doi:10.1145/3397271.3401075.
Information Processing Systems. Red 10. Wang, Yup; Conroy, John M.; Molino, Neil;
Hook, NY, USA: Curran Associates Inc. 6 Yang, Julia; Green, Mike. Laboratory for
December 2020 [9 December 2025]. Analytic Sciences in TREC 2024 Retrieval
ISBN 978-1-7138-2954-6. Augmented Generation Track. NIST TREC
4. Can a technology called RAG keep AI 2024. 2024 [15 March 2025].
models from making stuff up?. Ars 11. Lee, Kenton; Chang, Ming-Wei; Toutanova,
Technica. 6 June 2024 [7 March 2025]. Kristina. "Latent Retrieval for Weakly
5. Mitigating LLM hallucinations in text Supervised Open Domain Question
summarisation. BBC. 20 June 2024 Answering" (PDF). 2019.
[7 March 2025]. 12. Shi, Weijia; Min, Sewon; Yasunaga,
6. Lewis, Patrick; Perez, Ethan; Piktus, Michihiro; Seo, Minjoon; James, Rich;
Aleksandra; Petroni, Fabio; Karpukhin, Lewis, Mike; Zettlemoyer, Luke; Yih, Wen-
Vladimir; Goyal, Naman; Küttler, Heinrich; tau. REPLUG: Retrieval-Augmented Black-
Lewis, Mike; Yih, Wen-tau; Rocktäschel, Box Language Models. Proceedings of the
Tim; Riedel, Sebastian; Kiela, Douwe. 2024 Conference of the North American
Retrieval-Augmented Generation for Chapter of the Association for
Knowledge-Intensive NLP Tasks. Advances Computational Linguistics: Human
in Neural Information Processing Systems Language Technologies (Volume 1: Long
(Curran Associates, Inc.). 2020, 33: 9459‒ Papers). June 2024: 8371‒8384 [16 March
9474. arXiv:2005.11401 . 2025]. arXiv:2301.12652 .
7. Luan, Yi; Eisenstein, Jacob; Toutanova, doi:10.18653/v1/2024.naacl-long.463.
Kristina; Collins, Michael. Sparse, Dense, 13. Ram, Ori; Levine, Yoav; Dalmedigos, Itay;
and Attentional Representations for Text Muhlgay, Dor; Shashua, Amnon; Leyton-
Retrieval. Transactions of the Association Brown, Kevin; Shoham, Yoav. In-Context
for Computational Linguistics. 26 April Retrieval-Augmented Language Models.
2021, 9: 329‒345 [15 March 2025]. Transactions of the Association for
arXiv:2005.00181 . Computational Linguistics. 2023, 11: 1316‒
doi:10.1162/tacl_a_00369. 1331 [16 March 2025]. arXiv:2302.00083 .
8. Information retrieval. Microsoft. 10 doi:10.1162/tacl_a_00605.
January 2025 [15 March 2025]. 14. Borgeaud, Sebastian; Mensch, Arthur.
Improving language models by retrieving
from trillions of tokens(PDF). 2021.
15. Wang, Boxin; Ping, Wei; Xu, Peng; McAfee,
Lawrence; Liu, Zihan; Shoeybi,
Mohammad; Dong, Yi; Kuchaiev, Oleksii; Li,
Bo; Xiao, Chaowei; Anandkumar, Anima;
Catanzaro, Bryan. Shall We Pretrain
Autoregressive Language Models with
Retrieval? A Comprehensive Study.
Proceedings of the 2023 Conference on
Empirical Methods in Natural Language
Processing. 2023: 7763‒7786.
doi:10.18653/v1/2023.emnlp-main.482 .
参⻅
⼤型语⾔模型
推理语⾔模型
⽣成式⼈⼯智能
取⾃“https://zh.wikipedia.org/w/index.php?title=檢索增強⽣成&oldid=92761178”