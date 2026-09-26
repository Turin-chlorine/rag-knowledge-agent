⼤型语⾔模型
⼤型语⾔模型（英语：large language model，LLM），也称⼤语⾔模型，简称⼤模型，是⼀种基
于⼈⼯神经⽹络的语⾔模型。其名称中的“⼤型”指模型具有庞⼤的参数量（通常在数⼗亿以上，
如GPT-3含1750亿参数）以及巨⼤的训练数据规模。⼤语⾔模型通常采⽤⾃监督机器学习，能够在
海量⽆标注的⽂本上训练。⼤语⾔模型专为⾃然语⾔处理任务⽽设计，尤其是语⾔⽣成。[1][2]其中
包含Gemini和GPT-4o在内的部分模型具有多模态功能，能够同时处理⽂字、图⽚、⾳频和视频等
不同输⼊形式。⼤多数LLM采⽤了⽣成式预训练变换器(GPT) ，它们为ChatGPT、Gemini和
Claude等聊天机器⼈提供了核⼼功能。这些模型能够预测⼈类语⾔语料库中固有的句法、语义和本
体信息[3]，且展⽰出相当多训练期间“记住”的关于世界的常识。但它们也继承了训练数据中存在
的误差和偏差。[4]
由于⼤语⾔模型的记忆和泛化能⼒强⼤，其通常能⽤作通⽤模型：即使在没有针对特定任务（例如
情感分析、命名实体识别、⽂本翻译、摘要⽣成或数学推理）进⾏训练的情况下，LLM往往也能够
在这些任务中表现出⾊。 [4]⽽这些功能以往通常需要定制系统才能实现。[5] 此外，基于其跨任务泛
化能⼒，也可以针对特定任务对LLM进⾏微调，或通过提⽰⼯程进⾏引导，[6]从⽽在极少量特定任
务数据下实现或增强特定功能，如对话代理、代码⽣成、知识检索和⾃动推理等功能。
LLM源于早期的统计语⾔模型、神经概率语⾔模型和循环神经⽹络。2017年推出的Transformer
架构⽤⾃注意⼒机制取代了循环神经⽹络结构，从⽽实现了⾼效的并⾏化、更⻓的上下⽂处理能⼒
以及在前所未有的数据量上进⾏可扩展的训练。 [7]这项创新催⽣了GPT、BERT等模型，这些模型
展现出了⼤规模涌现⾏为，例如少样本学习和组合推理。[8]
为了扩展训练数据以增强模型能⼒，⼤部分LLM训练过程中存在利⽤⽹络上⼤量未经授权的公开艺
术家作品、⽤⼾与模型的交互以及其他公司LLM的输出（知识蒸馏）等数据训练AI的作法，在全
球引发了⼴泛的关于版权诉讼、道德争议和隐私保护的讨论。
历史
20世纪90年代，IBM对⻬模型开创了统计语⾔建模。2001年，⼀个基于3亿个单词进⾏训练的平
滑n-gram模型达到了当时最优的困惑度。[9] 在21世纪，随着互联⽹的普及，⼀些研究⼈员构建了
互联⽹规模的语⾔数据集（“⽹络语料库”[10]），并在此基础上训练统计语⾔模型。[11][12] 2009
年，在⼤多数语⾔处理任务中，统计语⾔模型优于符号语⾔模型，因为它们可以有效地消化⼤型数
据集。[13]
在 2012 年左右神经⽹络在图像处理领域占据主导地位后[14]，它们也被应⽤于语⾔建模。⾕歌于
2016 年将其翻译服务转换为神经机器翻译。就像在Transformer架构出现之前的语⾔模型⼀样，
它由seq2seq深度LSTM⽹络完成。
在 2017 年 NeurIPS 会议上，⾕歌研究⼈员在他们的⾥程碑式论⽂《Attention Is All You Need》
中介绍了Transformer架构。这篇论⽂的⽬标是改进 2014 年的 seq2seq 技术，[7] 并且主要基于
Bahdanau 等⼈在 2014 年开发的注意⼒机制。[15]2018 年，BERT被引⼊后迅速变得“⽆处不
在”。[16]虽然原始的 Transformer 同时具有编码器和解码器块，但 BERT 是⼀个仅编码器的模
型。随着仅解码器模型（如 GPT）通过提⽰解决任务的能⼒迅速提⾼，BERT 在学术和研究中的使
⽤率在 2023 年开始下降。[17]
仅解码器模型GPT-1于2018年推出，但2019年推出的GPT-2才引起了⼴泛关注，因为OpenAI最
初认为它过于强⼤，⽆法公开发布，因为担⼼被恶意使⽤。[18] 2020 年的GPT-3则更进⼀步，⾃
2024年起仅通过API提供，不提供下载模型以在本地执⾏。2022 年⾯向消费者的基于浏览器的
ChatGPT 吸引了普通⺠众的想象⼒，并引起了⼀些媒体炒作和在线热议。[19] 2023年的GPT-4因
其准确性的提⾼⽽受到称赞，并因其多模态功能⽽被称为“圣杯”。[20] OpenAI没有透露GPT-4的
⾼级架构和参数数量。ChatGPT的发布导致计算机科学的⼏个研究⼦领域的LLM使⽤率上升，包
括机器⼈技术、软件⼯程和⼀些有社会影响的⼯作。[21]与其竞争的语⾔模型在很⼤程度上试图与
GPT系列相提并论，⾄少在参数数量⽅⾯是这样。[22]
⾃2022年以来，开源模型越来越受欢迎，尤其是最初的BLOOM和LLaMA，尽管两者在使⽤领域
都有限制。Mistral AI的模型Mistral 7B和Mixtral 8x7b拥有更宽松的Apache许可证。截⾄
2024年6⽉，根据LMSYS Chatbot Arena排⾏榜，Llama 3的700亿参数模型的指令微调变体是
最强⼤的开放LLM，强于GPT-3.5但不如GPT-4。[23] 2025年1⽉，DeepSeek发布了 DeepSeek-
R1，这是⼀个拥有6710亿个参数的开放权重模型，其性能与OpenAI o1相当，但成本却低得
多。[24]
⾃2023年以来，许多LLM已被训练为多模态，能够处理或⽣成其他类型的数据，例如图像或⾳
频。这些LLM也称为⼤型多模态模型 (LMM)。[25]
截⾄2024年，最⼤、功能最强⼤的模型均基于Transformer架构。最近的⼀些实现基于其他架
构，例如循环神经⽹络变体和Mamba（状态空间模型）。[26][27][28]
⾃2023年以来，开放权重的⼤语⾔模型已⽇益成为⼈⼯智能领域的重要组成部分，有助于更⼴泛地
参与⼈⼯智能开发，并提⾼模型评估的透明度。Vake 等⼈ (2025) 的研究表明，社区驱动的开放权
重模型贡献能够显著提⾼其效率和性能，⽤⼾参与度在Hugging Face等协作平台上迅速增⻓。[29]
Paris 等⼈ (2025) 进⼀步指出，⼈⼯智能的开放性不应仅限于发布模型代码或权重，还应涵盖⼈⼯
智能研究和部署中的包容性、问责制和伦理责任。[30] 总⽽⾔之，这些研究强调，开放权重逻辑模
型能够加速创新，增强科学可重复性，同时促进⼈⼯智能⽣态系统的透明化和参与性。
数据集预处理
词元化（tokenization）
由于机器学习算法处理的是数字⽽不是⽂本，因此必须将⽂本转换为数字表⽰的词元（token）。该
过程称为词元化（tokenization），是数据预处理中的⼀个关键步骤。[31]
词元化⾸先需要确定⼀个词汇表，然后为每个词汇表条⽬任意但唯⼀地分配整数索引，最后将嵌⼊
与整数索引关联。确定词汇表的算法包括字节对编码（BPE）和WordPiece（BERT）。不同的算法
下，平均每个单词需要的词元个数也有所不同。该信息也取决于数据集的语⾔等因素。由于每个词
元可以表⽰多个字符的，标记器还能够压缩数据集。[32][33]
词汇表中，通常会设计⼀些特殊词元⽤作控制字符，例如 [MASK] 表⽰掩码标记（如 BERT 中使⽤
的），[UNK]（“未知”）表⽰未出现在词汇表中的字符。此外，⼀些特殊符号⽤于表⽰特殊的⽂本格
式。例如，“Ġ”表⽰ RoBERTa 和 GPT 中的前⼀个空格。“##”表⽰ BERT 中前⼀个单词的延
续。[34]
例如，GPT-3（旧版）使⽤的 BPE 标记器会将标记器：
tokenizer: texts -> series of numerical
拆分为
"tokens"
tokenizer: texts ->series of numerical "tokens"
由于 LLM 通常要求输⼊是⼀个整⻬的⾼维数组，因此当并发地使⽤多个⽂本进⾏训练时，必须先
“填充”较短的⽂本（padding），直到它们与最⻓⽂本的⻓度匹配。
字节对编码
作为⽰例，考虑基于字节对编码的标记器。在第⼀步中，所有唯⼀字符（包括空格和标点符号）都
被视为⼀组初始的 n-gram（即⼀组初始的 uni-gram）。随后，最常⻅的⼀对相邻字符合并为⼀个
⼆元组，并⽤它替换该对的所有实例。然后，将最常⼀起出现的相邻对（先前合并的）n-gram 再
次合并为更⻓的 n-gram，直到获得规定⼤⼩的词汇表（对于 GPT-3，⼤⼩为 50257）。[35] 训练标
记器后，任何⽂本都可以被它标记，只要它不包含未出现在初始 uni-gram 集中的字符。[36]
问题
基于从主要英语语料库中提取的频率的标记词汇表对⼀个普通英语单词使⽤尽可能少的标记。然
⽽，由这种针对英语优化的标记器编码的另⼀种语⾔的普通单词被分成次优数量的标记。对于某些
语⾔，例如缅甸掸语，GPT-2 标记器每个单词最多可以使⽤ 15 倍的标记。与英语相⽐，葡萄⽛语
和德语等更⼴泛使⽤的语⾔也“溢价 50%”。[37]
贪⼼标记化还会导致⽂本补全出现微妙的问题。[38]
数据清洗
在训练 LLM 的背景下，数据集通常通过删除低质量、重复或有害数据来清理。[39] 清理后的数据集
可以提⾼训练效率并提⾼下游性能。[40][41]训练过的 LLM 可⽤于清理数据集以训练进⼀步的
LLM。[42]
随着⽹络上 LLM ⽣成内容的⽐例不断增加，未来的数据清理可能包括过滤掉此类内容。如果内容
与⼈类⽂本相似（使过滤变得困难）但质量较低（降低在其上训练的模型的性能），则 LLM ⽣成的
内容可能会带来问题。[43]
合成数据
训练最⼤的语⾔模型可能需要⽐⾃然可⽤的更多的语⾔数据，或者⾃然发⽣的数据质量不够。在这
些情况下，可能会使⽤合成数据。微软的 Phi 系列LLM采⽤另⼀LLM⽣成的类似教科书的数据进
⾏训练。[44]
架构
注意⼒机制和上下⽂窗⼝
为了找出上下⽂窗⼝范围内哪些 token 彼此相关，注意⼒机制会使⽤多个注意⼒头为每个 token
（更准确地说是其嵌⼊）计算“软”权重，每个注意⼒头都有⾃⼰的“相关性”来计算⾃⼰的软权
重。例如，⼩型（即 1.17亿参数⼤⼩）GPT-2 模型有 12 个注意⼒头和⼀个只有 1000 个 token 的
上下⽂窗⼝。[46] 在其中等版本中，它有 3.45 亿个参数，包含 24 层，每层有 12 个注意⼒头。对于
梯度下降的训练，使⽤的批处理⼤⼩为 512。[47]
最⼤的模型，例如 2024 年 2 ⽉推出的 Google Gemini
1.5，可以有⼀个⼤⼩⾼达 100 万的上下⽂窗⼝（1000 万的
上下⽂窗⼝也“成功测试”）。[48] 其他具有⼤上下⽂窗⼝的
模型包括 Anthropic 的 Claude 2.1，其上下⽂窗⼝最多有
20 万个 token。[49] 请注意，此最⼤值指的是输⼊ token
的数量，输出 token 的最⼤数量与输⼊不同，并且通常较
⼩。例如，GPT-4 Turbo 模型的最⼤输出为 4096 个
token。[50]
模型在⽣成下⼀个答案时可以考虑的对话⻓度也受到上下⽂
窗⼝⼤⼩的限制。如果对话的⻓度（例如与 ChatGPT 的对
话）⻓于其上下⽂窗⼝，则在⽣成下⼀个答案时只会考虑上
下⽂窗⼝内的部分，或者模型需要应⽤某种算法来总结对话
中太远的部分。
使上下⽂窗⼝变⼤的缺点包括计算成本更⾼，并且可能削弱
对局部上下⽂的关注，⽽使上下⽂窗⼝变⼩可能会导致模型
错过重要的⻓距离依赖关系。平衡它们是⼀个实验和特定领
域的考虑问题。
当每个注意⼒头根据⾃⼰的标准计算其他
标记与“it_”标记的相关程度时，注意到
模型可以预先训练，以预测⽚段如何继续，或者在给定训练
由第⼆列表⽰的第⼆个注意⼒头主要关注
数据集中的⽚段的情况下预测⽚段中缺少什么。[51] 它可以 前两⾏，即标记“The”和“animal”，
是 ⽽第三列主要关注下⾯两⾏，即
“tired”，它已被标记为两个标记。[45]
⾃回归的（即预测⽚段如何继续，就像 GPT 所做的那
样）：例如，给定⼀个⽚段“我喜欢吃”，模型会预测
“冰淇淋”或“寿司”。
填空式的（即填充⽚段中缺失的部分，就像“BERT”[52] 所做的那样）：例如，给定⼀个⽚段
“我喜欢 [__] [__] 淇淋”，模型会预测“吃”和“冰”作为缺失的内容。
模型可以在辅助任务上进⾏训练，以测试它们对数据分布的理解，例如下⼀句预测 (NSP)，其中呈
现成对的句⼦，模型必须预测它们是否连续出现在训练语料库中。[53] 在训练期间，正则化损失也
⽤于稳定训练。然⽽，正则化损失通常不⽤于测试和评估。
混合专家模型
最⼤的 LLM 可能过于昂贵，⽆法直接训练和使⽤。对于此类模型，可以应⽤专家混合 (MoE)，这
是⾕歌研究⼈员⾃ 2017 年以来⼀直进⾏的研究⽅向，⽤于训练多达 1 万亿个参数的模型。[54][55]
参数数量
通常，LLM 使⽤单精度或半精度浮点数（float32和float16）进⾏训练。⼀个float16值有16位，
即2字节，因此10亿个参数需要2 GB的空间。最⼤的模型通常拥有超过1000亿个参数，这超出了
⼤多数消费电⼦产品的容量范围。[56]
量化
训练后量化[57]旨在通过降低已训练模型参数的精度来减少空间需求，同时尽可能保留其性能。量化
可以进⼀步分为静态量化和动态量化。静态量化是指量化参数预先确定（通常在校准阶段），⽽动态
量化是指在推理过程中应⽤量化。最简单的量化形式是将所有参数截断为给定的⽐特数：这适⽤于
静态量化和动态量化，但会损失⼤量精度。动态量化允许每层使⽤不同的量化码本，可以是值查找
表或线性映射（缩放因⼦和偏置），但代价是放弃了使⽤低精度运算可能带来的速度提升。
量化后的模型通常被视为已冻结，权重修改（例如微调）仅应⽤于原始模型。可以使⽤低秩⾃适应
（low-rank adaptation, LoRA）来微调量化后的模型。[58]
扩展性
提⽰⼯程
以前⼤多数只能通过（昂贵的）微调才能实现的结果，都可以通过提⽰⼯程（prompt
engineering）实现，尽管仅限于单个对话的范围（更准确地说，仅限于上下⽂窗⼝的范围）。[59]
指令调优
指令调优（Instruction Tuning）是⼀种微调技术，通过在包含（指令，输出）对的数据集上以监
督学习⽅式进⼀步训练⼤型语⾔模型，使其更好地理解和执⾏⼈类指令。这种⽅法弥合了⼤型语⾔
模型的下⼀个词预测⽬标与⽤⼾希望模型遵循⼈类指令之间的差距[60]。
检索增强⽣成
检索增强⽣成（RAG）是⼀种通过将LLM与⽂档检索系统集成来增强其性能的⽅法。给定⼀个查
询，调⽤⽂档检索器来检索最相关的⽂档。这通常是通过将查询和⽂档编码成向量来实现的，然后
找到向量（通常存储在向量数据库中）与查询向量最相似的⽂档。之后，LLM 基于查询和从检索到
的⽂档中包含的上下⽂⽣成输出。[61][62]
基于⼈类反馈的强化学习
近端策略优化等基于⼈类反馈的强化学习算法被⼴泛⽤于进⼀步微调⼀个⼤语⾔模型[63]，语⾔模型
能够根据⼈类专家对回答做出的反馈，提供与⼈类价值观⼀致的答案，同时拒绝不适当或超出模型
知识空间的问题。
推理模型
2024 年末，LLM 开发出现了⼀个新⽅向，即专⻔为复杂推理任务设计的模型。这些“推理模型”
经过训练，在提供最终答案之前会花费更多时间⽣成分步解决⽅案，类似于⼈类解决问题的过
程。[64] OpenAI 于 2024年9⽉通过其 o1 模型引⼊了这⼀趋势，随后于2024年12⽉推出了o3。
与传统 LLM 相⽐，这些模型在数学、科学和编码任务⽅⾯表现出显着的改进。例如，在国际数学
奥林匹克资格考试问题上，GPT-4o的准确率达到 13%，⽽o1的准确率达到 83%。[65][66] 2025 年
1 ⽉，中国公司深度求索（DeepSeek）发布了DeepSeek-R1，这是⼀个 6710亿参数的开放权重推
理模型，其性能与 OpenAI 的 o1 相当，但运⾏成本明显更低。与 OpenAI 的专有模型不同，
DeepSeek-R1 的开放权重特性允许研究⼈员研究和构建算法，但其训练数据仍保持私密。[67] 与传
统的 LLM 相⽐，这些推理模型通常需要每个查询更多的计算资源，因为它们执⾏更⼴泛的处理来
逐步解决问题。然⽽，它们在需要结构化逻辑思维的领域表现出了卓越的能⼒，例如数学、科学研
究和计算机编程。[68]
训练成本
“⼤型语⾔模型”中的限定词“⼤型”本质上是模糊的，因
为没有明确的阈值来定义“⼤型”所需的参数数量。随着时
间的推移，以前被认为是“⼤型”的东西可能会演变。2018
年的 GPT-1 通常被认为是第⼀个 LLM，尽管它只有 1.17 亿
个参数。在⼤型语⾔模型列表中可以看到向⼤型模型发展的
趋势。
⾃ 2020 年以来，软件和硬件的进步⼤⼤降低了成本，以⾄
部分模型的训练成本估计
于在 2023 年，训练⼀个 120 亿参数的 LLM 的计算成本为
72,300 A100-GPU ⼩时，⽽在 2020 年，训练⼀个 15 亿参
数的 LLM（⽐ 2020 年最先进的 LLM ⼩两个数量级）的成本在 80,000 美元到 1,600,000 美元之
间。[69][70][71]⾃ 2020 年以来，⼤量资⾦投⼊到越来越⼤的模型中。例如，2019 年训练 GPT-2
（即 15 亿个参数的模型）花费了 5 万美元，⽽ 2022 年训练 PaLM（即 5400 亿个参数的模型）花
费了 800 万美元，⽽ Megatron-Turing NLG 530B（2021 年）花费了约 1100 万美元。[72]
对于基于 Transformer 的 LLM，训练成本远⾼于推理成本。训练⼀个 token 需要每个参数 6 次
FLOP，⽽推理⼀个 token 需要每个参数 1 到 2 次 FLOP。[73]
输⼊输出形式
多模态模型
多模态模型（英语：Large Multimodal Model，LMM），意味着“具有多种模态”，⽽“模态”是
指⼀种输⼊或输出类型，例如视频、图像、⾳频、⽂本、本体感受等。[74] 已经有许多专⻔训练过
的 AI 模型来摄取⼀种模态并输出另⼀种模态，例如⽤于图像到标签的 AlexNet[75]、⽤于图像⽂本
到⽂本的视觉问答[76]、以及⽤于语⾳到⽂本的语⾳识别。
从 LLM 创建多模态模型的常⽤⽅法是“标记”经过训练的编码器的输出。具体来说，可以构建⼀
个可以理解图像的 LLM，如下所⽰：采⽤经过训练的 LLM，并采⽤经过训练的图像编码器 。制
作⼀个⼩的多层感知器 这样对于任何图像 ，后处理向量 具有与编码标记相同的尺⼨。这
是⼀个“图像标记”。然后，可以交错⽂本标记和图像标记。然后在图像⽂本数据集上对复合模型进
⾏微调。可以更复杂地应⽤这种基本构造来改进模型。可以冻结图像编码器以提⾼稳定性。[77]
Flamingo 证明了标记化⽅法的有效性，对⼀对预训练的语⾔模型和图像编码器进⾏了微调，使其
在视觉问答⽅⾯的表现优于从头开始训练的模型。[78] 使⽤标记化⽅法将 Google PaLM 模型微调为
多模态模型 PaLM-E，并应⽤于机器⼈控制。[6] LLaMA 模型也已使⽤标记化⽅法转变为多模态，
以允许图像输⼊[79] 和视频输⼊。[80]
GPT-4 可以使⽤⽂本和图像作为输⼊[81]（尽管视觉组件直到 GPT-4V[82]] 才向公众发布）；Google
DeepMind 的 Gemini 也是多模态的。[83] Mistral 于 2024 年 9 ⽉推出了⾃⼰的多态号 Pixtral
12B。[84]
⾮⾃然语⾔
LLM处理编程语⾔的⽅式与处理⾃然语⾔的⽅式类似。由于代码和⼈类语⾔⼀样，都是以纯⽂本形
式表⽰的，因此⽆需对词法单元的处理⽅式进⾏特殊更改。LLM可以根据⽤⾃然语⾔编写的问题或
指令⽣成代码。它们还可以⽤⾃然语⾔描述代码，或将其翻译成其他编程语⾔。LLM最初被⽤作代
码补全⼯具，但随着技术的进步，它们已发展成为⾃动编程⼯具。诸如GitHub Copilot之类的服务
提供经过专⻔训练、微调或提⽰的LLM，⽤于编程。
偏差和局限性
⼤语⾔模型偏差和局限性是⾃然语⾔处理（NLP）领域正在进⾏的研究。虽然ChatGPT等⼤语⾔模
型在⽣成类⼈⽂本⽅⾯表现出了卓越的能⼒，但它们很容易受到算法偏⻅影响，继承和放⼤训练数
据中存在的偏⻅。这可能表现为对不同⼈⼝统计数据的歪曲表述或不公平待遇，例如基于种族[85]、
性别[86]、语⾔[87]和⽂化群体[87]的不同观点与态度。此外，这些模型通常⾯临事实准确性的限制。
技术取向导致局限
机器学习和⼈⼯智能⽅⾯的专家杨⽴昆在GTC2025上的“炉边对话”环节提出观点，认为仅仅依靠
语⾔和⽂字训练出来的 AI 系统，永远⽆法逼近⼈类的理解⼒[88]。他也提到了世界模型（World
Models）这⼀概念。他认为，学术界开发AI系统需要基于不同于当前token预测架构的新路径。
其中⼀个原因是：Token具有离散的性质。“在典型的NLP任务中，token的选择范围通常在⼏千
个左右。因此当你训练⼀个系统去预测下⼀个token，它并不能精确地预测出确切的token，⽽是
只能基于字典中的所有可能选项⽣成⼀个概率分布。”杨利昆描述到。他⼜说，现实世界中⼈类⾯对
的是⾼维、连续的数据。现在的有些AI通过像素精度的视频进⾏（⾏为或者规则）的预测，这种⽅
法在构建认知模型⽅⾯的效果却⽋佳。[89]
幻觉
幻觉指的是⼤语⾔模型输出与客观事实不符或具有误导性的内容，其可能由模型本⾝或⽤⼾引导产
⽣。[90]
偏差
语⾔偏差
语⾔偏差是指与语⾔相关的⼀种统计抽样偏差，也就是说在信息抽样中，查询语⾔导致的系统偏差
会使其⽆法准确呈现数据中的各种不同主题和观点。当前的⼤型语⾔模型主要是根据英语数据进⾏
训练的，因此通常将英语观点视为真实可靠的观点，⽽系统地将⾮英语观点视为不相关、错误或噪
⾳。当被问到诸如“什么是⾃由主义？”之类的政治意识形态的问题时，ChatGPT以英美⻆度为中
⼼，⽽对例如说越南的“反对国家⼲预个⼈和经济⽣活”与中国的“限制政府权⼒”等视⽽不⻅。
同样，回复中也没有⽇本、韩国、法国和德国语料库中的主流政治观点。[87]
性别偏差
性别偏差是指这些模型产⽣的结果倾向于对⼀种性别产⽣不公平的偏⻅。这种偏差通 常源于训练这
些模型的数据。例如，⼤型语⾔模型通常根据传统的性别规范来分配⻆⾊和特征；它可能会将护⼠
或秘书主要与⼥性联系起来，将⼯程师或⾸席执⾏官与男性联系起来。[85][91]
政治偏差
政治偏差是指算法系统地倾向于某些政治观点、意识形态或结果，也可能表现出政治偏⻅。由于训
练数据包含⼴泛的政治观点和覆盖范围，因此模型可能会⽣成倾向于特定政治意识形态或观点的响
应，具体取决于数据中这些观点的普遍程度。[92][93]
⽂化偏差
⽂化偏⻅是指⼤语⾔模型对特定的⽂化实践、信仰或传统持有偏⻅，由于受到训练数据中⽂化信息
的不均衡、误导性或歧视性影响。例如，若模型的训练数据中某种⽂化的观点被过度代表，模型就
继承这种偏差形成⼀定的偏⻅。[94]
地域偏差
地域偏差是指⼤语⾔模型根据地理位置或国籍对⼈们的⾏为、习惯或特征做出偏⻅性的假设。这种
偏差可能导致对特定地区的知识、成就、问题、潜⼒等⽅⾯的误解、低估或过度放⼤。[95]
年龄偏差
年龄偏差是指⼤语⾔模型在处理或⽣成与年龄相关的话题时，根据年龄做出刻板印象化的假设，例
如认为年⻓者不懂技术或年轻⼈缺乏责任感。[96]
职业偏差
职业偏差是指⼤语⾔模型对特定职业持有刻板印象，将某些职业视为⽐其他职业更有价值或重要，
或对特定职业的⼈群做出性格或能⼒上的假设。[97]
参⻅
⼤型语⾔模型列表
聊天机器⼈
语⾔模型
⾃然语⾔⽣成
聊天机器⼈精神病
参考资料
1. Bommasani, Rishi; Hudson, Drew A.; Adeli, 6. Brown, Tom B.; Mann, Benjamin; Ryder,
Ehsan; Altman, Russ; Arora, Simran; von Nick; Subbiah, Melanie; Kaplan, Jared;
Arx, Matthew; Bernstein, Michael S.; Bohg, Dhariwal, Prafulla; Neelakantan, Arvind;
Jeannette; Bosselut, Antoine; Brunskill, Shyam, Pranav; Sastry, Girish; Askell,
Emma. On the Opportunities and Risks of Amanda; Agarwal, Sandhini; Herbert-Voss,
Foundation Models. 2021. Ariel; Krueger, Gretchen; Henighan, Tom;
arXiv:2108.07258 . Child, Rewon; Ramesh, Aditya; Ziegler,
2. Brown, Tom B.; Mann, Benjamin; Ryder, Daniel M.; Wu, Jeffrey; Winter, Clemens;
Nick; Subbiah, Melanie; Kaplan, Jared; Hesse, Christopher; Chen, Mark; Sigler,
Dhariwal, Prafulla; Neelakantan, Arvind; Eric; Litwin, Mateusz; Gray, Scott; Chess,
Shyam, Pranav; Sastry, Girish; Askell, Benjamin; Clark, Jack; Berner, Christopher;
Amanda. Language Models are Few-Shot McCandlish, Sam; Radford, Alec; Sutskever,
Learners. 2020. arXiv:2005.14165 [cs.CL]. Ilya; Amodei, Dario. Larochelle, H.;
Ranzato, M.; Hadsell, R.; Balcan, M.F.; Lin,
3. Fathallah, Nadeen; Das, Arunav; De
H. , 编. Language Models are Few-Shot
Giorgis, Stefano; Poltronieri, Andrea;
Learners(PDF). Advances in Neural
Haase, Peter; Kovriguina, Liubov. NeOn-
Information Processing Systems (Curran
GPT: A Large Language Model-Powered
Associates, Inc.). Dec 2020, 33: 1877‒1901
Pipeline for Ontology Learning(PDF).
[2023-03-14]. arXiv:2005.14165 .
Extended Semantic Web Conference 2024.
doi:10.1145/3582269.3615599. （原始内容
Hersonissos, Greece. 2024-05-26.
存档(PDF)于2023-11-17）.
4. Manning, Christopher D.Human Language
7. Vaswani, Ashish; Shazeer, Noam; Parmar,
Understanding & Reasoning. Daedalus.
Niki; Uszkoreit, Jakob; Jones, Llion;
2022, 151 (2): 127‒138 [2023-06-08].
Gomez, Aidan N; Kaiser, Łukasz;
S2CID 248377870.
Polosukhin, Illia. Attention is All you Need
doi:10.1162/daed_a_01905. （原始内容存
(PDF). Advances in Neural Information
档于2023-03-09）.
Processing Systems (Curran Associates,
5. Kaplan, Jared; McCandlish, Sam;
Inc.). 2017, 30[2024-01-21]. （原始内容存
Henighan, Tom; Brown, Tom B.; Chess,
档(PDF)于2024-02-21）.
Benjamin; Child, Rewon; Gray, Scott;
8. Devlin, Jacob; Chang, Ming-Wei; Lee,
Radford, Alec; Wu, Jeffrey; Amodei, Dario.
Kenton; Toutanova, Kristina. BERT: Pre-
Scaling Laws for Neural Language Models.
training of Deep Bidirectional
2020. arXiv:2001.08361 [cs.LG].
Transformers for Language Understanding.
2018. arXiv:1810.04805 [cs.CL].
9. Goodman, Joshua, A Bit of Progress in
Language Modeling, 2001-08-09,
Bibcode:2001cs........8005G,
arXiv:cs/0108005
10. Kilgarriff, Adam; Grefenstette, Gregory. 16. Rogers, Anna; Kovaleva, Olga; Rumshisky,
Introduction to the Special Issue on the Anna. A Primer in BERTology: What We
Web as Corpus. Computational Linguistics. Know About How BERT Works.
September 2003, 29 (3): 333‒347 Transactions of the Association for
[2025-01-20]. ISSN 0891-2017. Computational Linguistics. 2020, 8: 842‒
doi:10.1162/089120103322711569. （原始 866 [2024-01-21]. S2CID 211532403.
内容存档于2024-06-16）. arXiv:2002.12327 .
11. Banko, Michele; Brill, Eric. Scaling to very doi:10.1162/tacl_a_00349. （原始内容存档
very large corpora for natural language 于2022-04-03）.
disambiguation. Proceedings of the 39th 17. Movva, Rajiv; Balachandar, Sidhika; Peng,
Annual Meeting on Association for Kenny; Agostini, Gabriel; Garg, Nikhil;
Computational Linguistics - ACL '01 Pierson, Emma. Topics, Authors, and
(Morristown, NJ, USA: Association for Institutions in Large Language Model
Computational Linguistics). 2001: 26‒33 Research: Trends from 17K arXiv Papers.
[2025-01-20]. Proceedings of the 2024 Conference of the
doi:10.3115/1073012.1073017. （原始内容 North American Chapter of the Association
存档于2024-09-22）. for Computational Linguistics: Human
12. Resnik, Philip; Smith, Noah A. The Web as a Language Technologies (Volume 1: Long
Parallel Corpus. Computational Papers). 2024: 1223‒1243 [2024-12-08].
Linguistics. September 2003, 29 (3): 349‒ arXiv:2307.10700 .
380 [2024-06-07]. ISSN 0891-2017. doi:10.18653/v1/2024.naacl-long.67. （原
doi:10.1162/089120103322711578 . （原始 始内容存档于2025-04-12）.
内容存档于2024-06-07）. 18. Hern, Alex. New AI fake text generator may
13. Halevy, Alon; Norvig, Peter; Pereira, be too dangerous to release, say creators.
Fernando. The Unreasonable Effectiveness The Guardian. 2019-02-14 [2024-01-20].
of Data. IEEE Intelligent Systems. March （原始内容存档于2019-02-14）.
2009, 24 (2): 8‒12 [2025-01-20]. ISSN 1541- 19. ChatGPT a year on: 3 ways the AI chatbot
1672. doi:10.1109/MIS.2009.36. （原始内容 has completely changed the world in 12
存档于2024-10-04）. months. Euronews. 2023-11-30
14. Chen, Leiyu; Li, Shaobo; Bai, Qiang; Yang, [2024-01-20]. （原始内容存档于2024-01-
Jing; Jiang, Sanlong; Miao, Yanming. 14）.
Review of Image Classification Algorithms 20. Heaven, Will. GPT-4 is bigger and better
Based on Convolutional Neural Networks. than ChatGPT―but OpenAI won't say why.
Remote Sensing. 2021, 13 (22): 4712. MIT Technology Review. 2023-03-14
Bibcode:2021RemS...13.4712C. [2024-01-20]. （原始内容存档于2023-03-
doi:10.3390/rs13224712 . 17）.
15. Bahdanau, Dzmitry; Cho, Kyunghyun;
Bengio, Yoshua. Neural Machine
Translation by Jointly Learning to Align
and Translate. 2014. arXiv:1409.0473
[cs.CL].
21. Movva, Rajiv; Balachandar, Sidhika; Peng, 29. Vake, Domen; Šinik, Bogdan; Vičič, Jernej;
Kenny; Agostini, Gabriel; Garg, Nikhil; Tošić, Aleksandar. Is Open Source the
Pierson, Emma. Topics, Authors, and Future of AI? A Data-Driven Approach.
Institutions in Large Language Model Applied Sciences. 2025-03-05, 15 (5): 2790.
Research: Trends from 17K arXiv Papers. ISSN 2076-3417. doi:10.3390/app15052790
Proceedings of the 2024 Conference of the （英语）.
North American Chapter of the Association 30. Paris, Tamara; Moon, AJung; Guo, Jin L.C.
for Computational Linguistics: Human Opening the Scope of Openness in AI.
Language Technologies (Volume 1: Long Proceedings of the 2025 ACM Conference
Papers). 2024: 1223‒1243 [2024-12-08]. on Fairness, Accountability, and
arXiv:2307.10700 . Transparency. Association for Computing
doi:10.18653/v1/2024.naacl-long.67. （原 Machinery: 1293‒1311. 2025-06-23.
始内容存档于2025-04-12）. doi:10.1145/3715275.3732087 .
22. Parameters in notable artificial intelligence 31. 赵鑫，李军毅，周昆，唐天⼀，⽂继荣. ⼤语
systems. ourworldindata.org. 2023-11-30 ⾔模型. 北京: ⾼等教育出版社. 2024.
[2024-01-20]. （原始内容存档于2024-10-
32. Yennie Jun. All languages are NOT created
06）.
(tokenized) equal. Language models cost
23. LMSYS Chatbot Arena Leaderboard. much more in some languages than
huggingface.co. [2024-06-12]. （原始内容存 others. 2023-05-03 [2023-08-17]. （原始内
档于2024-06-10）. 容存档于2023-08-17）. "In other words, to
24. Sharma, Shubham. Open-source express the same sentiment, some
DeepSeek-R1 uses pure reinforcement languages require up to 10 times more
learning to match OpenAI o1 ― at 95% less tokens."
cost. VentureBeat. 2025-01-20 33. Petrov, Aleksandar; Malfa, Emanuele La;
[2025-01-26]. （原始内容存档于2025-01- Torr, Philip; Bibi, Adel. Language Model
25） （美国英语）. Tokenizers Introduce Unfairness Between
25. Zia, Dr Tehseen. Unveiling of Large Languages. NeurIPS. 2023-06-23
Multimodal Models: Shaping the [2023-09-16]. arXiv:2305.15425 . （原始内
Landscape of Language Models in 2024. 容存档于2023-12-15） ‒通过
Unite.AI. 2024-01-08 [2024-12-28]. （原始内 openreview.net.
容存档于2024-12-04） （美国英语）. 34. Kaushal, Ayush; Mahowald, Kyle, What do
26. Peng, Bo; et al. RWKV: Reinventing RNNS tokens know about their characters and
for the Transformer Era. 2023. how do they know it?, 2022-06-06,
arXiv:2305.13048 [cs.CL]. arXiv:2206.02608
27. Merritt, Rick. What Is a Transformer 35. OpenAI API. platform.openai.com.
Model?. NVIDIA Blog. 2022-03-25 [2023-04-30]. （原始内容存档于2023-04-
[2023-07-25]. （原始内容存档于2023-11- 23）.
17）.
28. Gu, Albert; Dao, Tri, Mamba: Linear-Time
Sequence Modeling with Selective State
Spaces, 2023-12-01, arXiv:2312.00752
36. Paaß, Gerhard; Giesselbach, Sven. Pre- 42. Lin, Zhenghao; Gou, Zhibin; Gong, Yeyun;
trained Language Models. Foundation Liu, Xiao; Shen, Yelong; Xu, Ruochen; Lin,
Models for Natural Language Processing. Chen; Yang, Yujiu; Jiao, Jian. Rho-1: Not All
Artificial Intelligence: Foundations, Theory, Tokens Are What You Need. 2024-04-11.
and Algorithms. 2022: 19‒78 [2023-08-03]. arXiv:2404.07965 [cs.CL].
ISBN 9783031231902. doi:10.1007/978-3- 43. Brown, Tom B.; et al. Language Models are
031-23190-2_2. （原始内容存档于2023-08- Few-Shot Learners. 2020. arXiv:2005.14165
03）. [cs.CL].
37. Petrov, Aleksandar; Emanuele La Malfa; 44. Abdin, Marah; Jacobs, Sam Ade; Awan,
Torr, Philip H. S.; Bibi, Adel. Language Ammar Ahmad; Aneja, Jyoti; Awadallah,
Model Tokenizers Introduce Unfairness Ahmed; Awadalla, Hany; Bach, Nguyen;
Between Languages. 2023. Bahree, Amit; Bakhtiari, Arash. Phi-3
arXiv:2305.15425 [cs.CL]. Technical Report: A Highly Capable
38. Lundberg, Scott. The Art of Prompt Design: Language Model Locally on Your Phone.
Prompt Boundaries and Token Healing. 2024-04-23. arXiv:2404.14219 [cs.CL].
Medium. 2023-12-12 [2024-08-05]. （原始 45. Allamar, Jay. Illustrated transformer.
内容存档于2024-08-05） （英语）. [2023-07-29]. （原始内容存档于2023-07-
39. Dodge, Jesse; Sap, Maarten; Marasović, 25）.
Ana; Agnew, William; Ilharco, Gabriel; 46. Allamar, Jay. The Illustrated GPT-2
Groeneveld, Dirk; Mitchell, Margaret; (Visualizing Transformer Language
Gardner, Matt. Documenting Large Webtext Models). [2023-08-01]. （原始内容存档于
Corpora: A Case Study on the Colossal 2019-08-13）.
Clean Crawled Corpus. 2021.
47. Paaß, Gerhard; Giesselbach, Sven. Pre-
arXiv:2104.08758 [cs.CL].
trained Language Models. Foundation
40. Lee, Katherine; Ippolito, Daphne; Nystrom, Models for Natural Language Processing.
Andrew; Zhang, Chiyuan; Eck, Douglas; Artificial Intelligence: Foundations, Theory,
Callison-Burch, Chris; Carlini, Nicholas. and Algorithms. 2022: 19‒78 [2023-08-03].
Deduplicating Training Data Makes ISBN 9783031231902. doi:10.1007/978-3-
Language Models Better(PDF). Proceedings 031-23190-2_2. （原始内容存档于2023-08-
of the 60th Annual Meeting of the 03）.
Association for Computational Linguistics.
48. Our next-generation model: Gemini 1.5.
May 2022,. 1: Long Papers: 8424‒8445
Google. 2024-02-15 [2024-02-18]. （原始内
[2025-02-07]. doi:10.18653/v1/2022.acl-
容存档于2024-02-18）.
long.577. （原始内容存档(PDF)于2024-09-
49. Long context prompting for Claude 2.1.
30）.
2023-12-06 [2024-01-20]. （原始内容存档于
41. Li, Yuanzhi; Bubeck, Sébastien; Eldan,
2024-08-27）.
Ronen; Del Giorno, Allie; Gunasekar,
50. Rate limits. openai.com. [2024-01-20]. （原
Suriya; Lee, Yin Tat, Textbooks Are All You
始内容存档于2024-02-02）.
Need II: phi-1.5 technical report, 2023-09-
11, arXiv:2309.05463
51. Zaib, Munazza; Sheng, Quan Z.; Emma 58. Mittal, Aayush Mittal. LoRa, QLoRA and QA-
Zhang, Wei. A Short Survey of Pre-trained LoRA: Efficient Adaptability in Large
Language Models for Conversational AI-A Language Models Through Low-Rank
New Age in NLP. Proceedings of the Matrix Factorization. Unite.AI. 2023-10-24
Australasian Computer Science Week [2025-11-16]（美国英语）.
Multiconference. 2020-02-04: 1‒4. 59. Wei, Jason; Tay, Yi; Bommasani, Rishi;
ISBN 9781450376976. S2CID 211040895. Raffel, Colin; Zoph, Barret; Borgeaud,
arXiv:2104.10810 . Sebastian; Yogatama, Dani; Bosma,
doi:10.1145/3373017.3373028. Maarten; Zhou, Denny; Metzler, Donald;
52. Jurafsky, Dan; Martin, James H. Speech Chi, Ed H.; Hashimoto, Tatsunori; Vinyals,
and Language Processing(PDF) 3rd edition Oriol; Liang, Percy; Dean, Jeff; Fedus,
draft. 2023-01-07 [2022-05-24]. （原始内容 William. Emergent Abilities of Large
存档(PDF)于2023-03-23）. Language Models. Transactions on
53. Jurafsky, Dan; Martin, James H. Speech Machine Learning Research. 2022-08-31
and Language Processing(PDF) 3rd edition [2023-03-19]. ISSN 2835-8856. （原始内容
draft. 2023-01-07 [2022-05-24]. （原始内容 存档于2023-03-22）.
存档(PDF)于2023-03-23）. 60. What is instruction tuning?. IBM.
54. Shazeer, Noam; Mirhoseini, Azalia; Maziarz, [2024-12-09]. （原始内容存档于2024-12-
Krzysztof; Davis, Andy; Le, Quoc; Hinton, 09）.
Geoffrey; Dean, Jeff. Outrageously Large 61. Lewis, Patrick; Perez, Ethan; Piktus,
Neural Networks: The Sparsely-Gated Aleksandra; Petroni, Fabio; Karpukhin,
Mixture-of-Experts Layer. 2017-01-01. Vladimir; Goyal, Naman; Küttler, Heinrich;
arXiv:1701.06538 [cs.LG]. Lewis, Mike; Yih, Wen-tau; Rocktäschel,
55. Lepikhin, Dmitry; Lee, HyoukJoong; Xu, Tim; Riedel, Sebastian; Kiela, Douwe.
Yuanzhong; Chen, Dehao; Firat, Orhan; Retrieval-Augmented Generation for
Huang, Yanping; Krikun, Maxim; Shazeer, Knowledge-Intensive NLP Tasks. Advances
Noam; Chen, Zhifeng. GShard: Scaling in Neural Information Processing Systems
Giant Models with Conditional (Curran Associates, Inc.). 2020, 33: 9459‒
Computation and Automatic Sharding. 9474 [2023-06-12]. arXiv:2005.11401 . （原
2021-01-12. arXiv:2006.16668 [cs.CL]. 始内容存档于2023-06-12）.
56. Mann, Tobias. How to run an LLM locally 62. Kiela, Douwe; Riedel, Sebastian; Lewis,
on your PC in less than 10 minutes. Patrick; Piktus, Aleksandra. Retrieval
www.theregister.com. [2024-05-17]. Augmented Generation: Streamlining the
creation of intelligent natural language
57. Nagel, Markus; Amjad, Rana Ali; Baalen,
processing models. Meta. 2020-09-28.
Mart Van; Louizos, Christos; Blankevoort,
Tijmen. Up or Down? Adaptive Rounding
for Post-Training Quantization.
Proceedings of the 37th International
Conference on Machine Learning (PMLR).
2020-11-21: 7197‒7206 [2023-06-14]. （原
始内容存档于2023-06-14）.
63. Ouyang, Long; Wu, Jeff; Jiang, Xu; Almeida, 71. Biderman, Stella; Schoelkopf, Hailey;
Diogo; Wainwright, Carroll L.; Mishkin, Anthony, Quentin; Bradley, Herbie; Khan,
Pamela; Zhang, Chong; Agarwal, Sandhini; Mohammad Aflah; Purohit, Shivanshu;
Slama, Katarina; Ray, Alex; Schulman, Prashanth, USVSN Sai. Pythia: A Suite for
John; Hilton, Jacob; Kelton, Fraser; Miller, Analyzing Large Language Models Across
Luke; Simens, Maddie; Askell, Amanda; Training and Scaling. April 2023.
Welinder, Peter; Christiano, Paul; Leike, arXiv:2304.01373 [cs.CL].
Jan; Lowe, Ryan. Training language 72. Maslej, Nestor; Fattorini, Loredana;
models to follow instructions with human Brynjolfsson, Erik; Etchemendy, John;
feedback. 2022. arXiv:2203.02155 [cs.CL]. Ligett, Katrina; Lyons, Terah; Manyika,
64. Introducing OpenAI o1-preview. OpenAI. James; Ngo, Helen; Niebles, Juan Carlos,
2024-09-12 [2025-02-03]. （原始内容存档于 Artificial Intelligence Index Report 2023,
2024-11-26）. 2023-10-05, arXiv:2310.03715
65. Introducing OpenAI o1-preview. OpenAI. 73. Section 2.1 and Table 1, Kaplan, Jared;
2024-09-12 [2025-02-03]. （原始内容存档于 McCandlish, Sam; Henighan, Tom; Brown,
2024-11-26）. Tom B.; Chess, Benjamin; Child, Rewon;
66. Metz, Cade. OpenAI Unveils New A.I. That Gray, Scott; Radford, Alec; Wu, Jeffrey;
Can 'Reason' Through Math and Science Amodei, Dario. Scaling Laws for Neural
Problems. The New York Times. 2024-12-20 Language Models. 2020. arXiv:2001.08361
[2025-02-03]. （原始内容存档于2025-02- [cs.LG].
09）. 74. Kiros, Ryan; Salakhutdinov, Ruslan; Zemel,
67. Gibney, Elizabeth. China's cheap, open AI Rich. Multimodal Neural Language Models.
model DeepSeek thrills scientists. Nature. Proceedings of the 31st International
2025-01-30 [2025-02-03]. （原始内容存档于 Conference on Machine Learning (PMLR).
2025-01-29）. 2014-06-18: 595‒603 [2023-07-02]. （原始
内容存档于2023-07-02）.
68. Metz, Cade. OpenAI Unveils New A.I. That
Can 'Reason' Through Math and Science 75. Krizhevsky, Alex; Sutskever, Ilya; Hinton,
Problems. The New York Times. 2024-12-20 Geoffrey E. ImageNet Classification with
[2025-02-03]. （原始内容存档于2025-02- Deep Convolutional Neural Networks.
09）. Advances in Neural Information Processing
Systems (Curran Associates, Inc.). 2012, 25
69. Wiggers, Kyle. The emerging types of
[2023-07-02]. （原始内容存档于2023-07-
language models and why they matter.
02）.
TechCrunch. 2022-04-28 [2023-03-09]. （原
始内容存档于2023-03-16）. 76. Antol, Stanislaw; Agrawal, Aishwarya; Lu,
Jiasen; Mitchell, Margaret; Batra, Dhruv;
70. Sharir, Or; Peleg, Barak; Shoham, Yoav. The
Zitnick, C. Lawrence; Parikh, Devi. VQA:
Cost of Training NLP Models: A Concise
Visual Question Answering. ICCV. 2015:
Overview. 2020. arXiv:2004.08900 [cs.CL].
2425‒2433 [2023-07-02]. （原始内容存档于
2023-07-02）.
77. Li, Junnan; Li, Dongxu; Savarese, Silvio; 85. Kotek, Hadas; Dockum, Rikker; Sun, David.
Hoi, Steven. BLIP-2: Bootstrapping Gender bias and stereotypes in Large
Language-Image Pre-training with Frozen Language Models. Proceedings of The ACM
Image Encoders and Large Language Collective Intelligence Conference. CI '23
Models. 2023-01-01. arXiv:2301.12597 (New York, NY, USA: Association for
[cs.CV]. Computing Machinery). 2023-11-05.
78. Alayrac, Jean-Baptiste; Donahue, Jeff; Luc, ISBN 979-8-4007-0113-9.
Pauline; Miech, Antoine; Barr, Iain; Hasson, doi:10.1145/3582269.3615599.
Yana; Lenc, Karel; Mensch, Arthur; Millican, 86. Davidson, Thomas; Bhattacharya,
Katherine; Reynolds, Malcolm; Ring, Debasmita; Weber, Ingmar. Roberts, Sarah
Roman; Rutherford, Eliza; Cabi, Serkan; T.; Tetreault, Joel; Prabhakaran,
Han, Tengda; Gong, Zhitao. Flamingo: a Vinodkumar; Waseem, Zeerak , 编. Racial
Visual Language Model for Few-Shot Bias in Hate Speech and Abusive Language
Learning. Advances in Neural Information Detection Datasets. Proceedings of the
Processing Systems. 2022-12-06, 35: Third Workshop on Abusive Language
23716‒23736 [2023-07-02]. Online (Florence, Italy: Association for
arXiv:2204.14198 . （原始内容存档于2023- Computational Linguistics). 2019-08.
07-02）. doi:10.18653/v1/W19-3504.
79. Liu, Haotian; Li, Chunyuan; Wu, Qingyang; 87. Queenie Luo; Michael J. Puett; Michael D.
Lee, Yong Jae. Visual Instruction Tuning. Smith. A Perspectival Mirror of the
2023-04-01. arXiv:2304.08485 [cs.CV]. Elephant: Investigating Language Bias on
80. Zhang, Hang; Li, Xin; Bing, Lidong. Video- Google, ChatGPT, Wikipedia, and YouTube.
LLaMA: An Instruction-tuned Audio-Visual arXiv. （原始内容存档于2024-04-16）.
Language Model for Video Understanding. 88. 杨⽴昆：“AGI即将到来”完全是⽆稽之谈，
2023-06-01. arXiv:2306.02858 [cs.CL]. 真正的智能要建⽴在世界模型之上. ⿇省理⼯
81. OpenAI. GPT-4 Technical Report. 2023-03- 科技评论中⽂版. 2025-03-28 [2025-04-20]
27. arXiv:2303.08774 [cs.CL]. （中⽂（中国⼤陆））.
82. OpenAI. GPT-4V(ision) System Card(PDF). 89. 苏霍伊；甲⼦光年. 杨⽴昆GTC对话实录：
2023-09-25 [2025-02-11]. （原始内容存档 “AGI即将到来”完全是⽆稽之谈｜甲⼦光年
(PDF)于2023-09-25）. . 澎湃新闻. 2025-03-24 [2025-04-20]（中⽂
（中国⼤陆））.
83. Pichai, Sundar, Google Keynote (Google
I/O '23), timestamp 15:31, 2023-05-10 90. Lei Huang; Weijiang Yu; Weitao Ma. A
[2023-07-02] Survey on Hallucination in Large Language
Models: Principles, Taxonomy, Challenges,
84. Wiggers, Kyle. Mistral releases Pixtral 12B,
and Open Questions. arXiv. （原始内容存档
its first multimodal model. TechCrunch.
于2024-11-28）.
2024-09-11 [2024-09-14]. （原始内容存档于
2024-09-14）.
91. Yucong Duan; Fuliang Tang; Zhendong 95. Yucong Duan; Fuliang Tang; Kunguang Wu;
Guo; Yingtian Mei; Yuxing Wang; Kunguang Zhendong Guo; Shuaishuai Huang;
Wu; Zeyu Yang; Shuaishuai Huang; Yingtian Mei; Yuxing Wang; Zeyu Yang;
Shiming Gong. Global Large Language Shiming Gong. "Ranking of Large
Model EQ and IQ Bias Evaluation -Released Language Model (LLM) Regional Bias" --
by DIKWP -AC Research Group. DIKWP Research Group International
ResearchGate. 2023. Standard Evaluation. ResearchGate. 2024.
doi:10.13140/RG.2.2.12894.61762 ‒通过 doi:10.13140/RG.2.2.10019.63529 ‒通过
ResearchGate （英语）. ResearchGate.
92. Zhou, Karen; Tan, Chenhao. Bouamor, 96. Yucong Duan; Fuliang Tang; Kunguang Wu;
Houda; Pino, Juan; Bali, Kalika , 编. Entity- Zhendong Guo; Shuaishuai Huang;
Based Evaluation of Political Bias in Yingtian Mei; Yuxing Wang; Zeyu Yang;
Automatic Summarization. Findings of the Shiming Gong. "The Large Language
Association for Computational Linguistics: Model (LLM) Bias Evaluation (Age Bias)" --
EMNLP 2023 (Singapore: Association for DIKWP Research Group International
Computational Linguistics). 2023-12 Standard Evaluation. ResearchGate. 2024.
[2023-12-26]. doi:10.13140/RG.2.2.26397.12006 ‒通过
doi:10.18653/v1/2023.findings-emnlp.696. ResearchGate.
（原始内容存档于2024-04-24）. 97. Yucong Duan; Fuliang Tang; Kunguang Wu;
93. Waight, Hannah; Yang, Eddie; Yuan, Yin; Zhendong Guo; Shuaishuai Huang;
Messing, Solomon; Roberts, Margaret E.; Yingtian Mei; Yuxing Wang; Zeyu Yang;
Stewart, Brandon M.; Tucker, Joshua A. Shiming Gong. "The Large Language
State media control impacts the output of Model (LLM) Bias Evaluation (Occupational
U.S.-based LLMs. Good Authority. Bias)" --DIKWP Research Group
[2026-06-12]（美国英语）. International Standard Evaluation.
94. Yucong Duan; Fuliang Tang; Kunguang Wu; ResearchGate. 2024.
Zhendong Guo; Shuaishuai Huang; doi:10.13140/RG.2.2.23041.67689 ‒通过
Yingtian Mei; Yuxing Wang; Zeyu Yang; ResearchGate.
Shiming Gong. "Ranking of Large
Language Model (LLM) Cultural Bias" --
DIKWP Research Group International
Standard Evaluation. ResearchGate. 2024.
doi:10.13140/RG.2.2.26652.67200 ‒通过
ResearchGate.
外部链接
Open LLM Leaderboard（开放LLM排⾏榜旨在跟踪、排名和评估开放LLM和聊天机器⼈） (htt
ps://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) （⻚⾯存档备份 (http
s://web.archive.org/web/20231222103236/https://huggingface.co/spaces/HuggingFaceH4/
open_llm_leaderboard)，存于互联⽹档案馆）
取⾃“https://zh.wikipedia.org/w/index.php?title=⼤型语⾔模型&oldid=94347548”