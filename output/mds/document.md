# Extracted Content from document.pdf

![Image associated with caption: Figure 1: The MiniRAG employs a streamlined workflow built on the key components: heterogeneous
graph indexing and lightweight graph-based knowledge retrieval. This architecture addresses the
unique challenges faced by on-device RAG systems, optimizing for both efficiency and effectiveness.](..\images\document\page_1_figure_0.jpg)

*Figure 1: The MiniRAG employs a streamlined workflow built on the key components: heterogeneous
graph indexing and lightweight graph-based knowledge retrieval. This architecture addresses the
unique challenges faced by on-device RAG systems, optimizing for both efficiency and effectiveness.*

mance degradation where accuracy drops, or complete system failure where certain advanced RAG frameworks become entirely inoperable when transitioning from LLMs to SLMs.

To address these fundamental challenges, we propose MiniRAG, a novel RAG system that reimagines the information retrieval and generation pipeline with a focus on extreme simplicity and computational efficiency. Our design is motivated by three fundamental observations about Small Language Models (SLMs): (1) while they struggle with sophisticated semantic understanding, they excel at pattern matching and localized text processing; (2) explicit structural information can effectively compensate for limited semantic capabilities; and (3) decomposing complex RAG operations into simpler, welldefined steps can maintain system robustness without requiring advanced reasoning capabilities. These insights lead us to prioritize structural knowledge representation over semantic complexity, marking a significant departure from traditional LLM-centric RAG architectures.

Our MiniRAG introduces two key technical innovations that leverage these insights: (1) a semanticaware heterogeneous graph indexing mechanism that systematically combines text chunks and named entities in a unified structure, reducing reliance on complex semantic understanding, and (2) a lightweight topology-enhanced retrieval approach that utilizes graph structures and heuristic search patterns for efficient knowledge discovery. Through careful design choices and architectural optimization, these components work synergistically to enable robust RAG functionality even with limited model capabilities, fundamentally reimagining how RAG systems can operate within the constraints of SLMs while leveraging their strengths.

Through extensive experimentation across datasets and Small Language Models, we demonstrate MiniRAG’s exceptional performance: compared to existing lightweight RAG systems, MiniRAG achieves 1.3-2.5× higher effectiveness while using only 25% of the storage space. When transitioning from LLMs to SLMs, our system maintains remarkable robustness, with accuracy reduction ranging from merely 0.8% to 20% across different scenarios. Most notably, MiniRAG consistently achieves state-of-the-art performance across all evaluation settings, including tests on two comprehensive datasets with four different SLMs, while maintaining a lightweight footprint suitable for resourceconstrained environments such as edge devices and privacy-sensitive applications. To facilitate further research in this direction, we also introduce LiHuaWorld, a comprehensive benchmark dataset specifically designed for evaluating lightweight RAG systems under realistic on-device scenarios such as personal communication and local document retrieval.

## 2 THE MINIRAG FRAMEWORK

In this section, we present the detailed architecture of our proposed MiniRAG framework. As illustrated in Fig.1, MiniRAG consists of two key components: (1) heterogeneous graph indexing (Sec.2.1), which creates a semantic-aware knowledge representation, and (2) lightweight graph-based knowledge retrieval (Sec.2.2), which enables efficient and accurate information retrieval.


--- End of Page 1 ---



|  | 0 | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- | --- |
| 0 |  | NaiveRAG | GraphRAG | LightRAG | MiniRAG |
| 1 | LiHuaWorld |  |  |  |  |
| 2 |  | acc↑\nerr↓ | acc↑\nerr↓ | acc↑\nerr↓ | acc↑\nerr↓ |
| 3 | Phi-3.5-mini-instruct | 41.22%\n23.20% | /\n/ | 39.81%\n25.39% | 53.29%\n23.35% |
| 4 | GLM-Edge-1.5B-Chat | 42.79%\n24.76% | /\n/ | 35.74%\n25.86% | 52.51%\n25.71% |
| 5 | Qwen2.5-3B-Instruct | 43.73%\n24.14% | /\n/ | 39.18%\n28.68% | 48.75%\n26.02% |
| 6 | MiniCPM3-4B | 43.42%\n17.08% | /\n/ | 35.42%\n21.94% | 51.25%\n21.79% |
| 7 | gpt-4o-mini | 46.55%\n19.12% | 35.27%\n37.77% | 56.90%\n20.85% | 54.08%\n19.44% |
| 8 |  | NaiveRAG | GraphRAG | LightRAG | MiniRAG |
| 9 | MultiHop-RAG |  |  |  |  |
| 10 |  | acc↑\nerr↓ | acc↑\nerr↓ | acc↑\nerr↓ | acc↑\nerr↓ |
| 11 | Phi-3.5-mini-instruct | 42.72%\n31.34% | /\n/ | 27.03%\n11.78% | 49.96%\n28.44% |
| 12 | GLM-Edge-1.5B-Chat | 44.44%\n24.26% | /\n/ | /\n/ | 51.41%\n23.44% |
| 13 | Qwen2.5-3B-Instruct | 39.48%\n31.69% | /\n/ | 21.91%\n13.73% | 48.55%\n33.10% |
| 14 | MiniCPM3-4B | 39.24%\n31.42% | /\n/ | 19.48%\n10.41% | 47.77%\n26.88% |
| 15 | gpt-4o-mini | 53.60%\n27.19% | 60.92%\n16.86% | 64.91%\n19.37% | 68.43%\n19.41% |


*Table 1: Performance evaluation using accuracy (acc) and error (err) rates, measured as percentages
(%). Higher accuracy and lower error rates indicate better RAG performance. Results compare
various baseline methods against our MiniRAG across multiple datasets. Bold values indicate best
performance, while “/” denotes cases where methods failed to generate effective responses.*

vulnerabilities in their architectures. Advanced LLM-based RAG methods exhibit severe performance degradation, with LightRAG’s accuracy plummeting from 56.90% to 35.42% during LLM to SLM transition, while GraphRAG experiences complete system failure due to its inability to generate high-quality content. While basic retrieval systems like NaiveRAG show resilience, they suffer from significant limitations, being restricted to basic functionality and lacking advanced reasoning capabilities. This performance analysis highlights a critical challenge: existing advanced systems’ over-reliance on sophisticated language capabilities leads to fundamental operational failures when using simpler models, creating a significant barrier to widespread adoption in resource-constrained environments, where high-end language models may not be available or practical to deploy.

MiniRAG’s Unique Advantages. These innovations enable MiniRAG to maintain strong performance even with simpler language models, making it particularly suitable for resource-constrained environments while preserving the core functionalities of RAG systems.

i) Semantic-Aware Graph Indexing for Reduced Model Dependency. MiniRAG fundamentally reimagines the indexing process through a dual-node heterogeneous graph structure. Instead of relying on powerful text generation capabilities, the system focuses on basic entity extraction and heterogeneous relationship mapping. This design combines text chunk nodes for preserving raw contextual information with entity nodes for capturing key semantic elements, creating a robust knowledge representation that remains effective even with limited language model capabilities.

ii) Topology-Enhanced Retrieval for Balanced Performance. MiniRAG employs a lightweight graph-based retrieval mechanism that balances multiple information signals through a systematic process. Beginning with query-driven path discovery, the system integrates embeddingbased matching with structural graph patterns and entity-specific relevance scores. Through topology-aware search and optimized efficiency, it achieves robust retrieval quality without requiring advanced language understanding, making it particularly effective for on-device deployment.

These innovations enable MiniRAG to maintain strong performance with simpler language models, making it ideal for resource-constrained environments while preserving core RAG functionalities.

Storage Efficiency While Maintaining Performance. MiniRAG demonstrates exceptional storage efficiency while preserving high accuracy levels. Empirical evaluations show that MiniRAG achieves competitive accuracy while requiring only 25% of the storage space compared to baselines like LightRAG

![Image associated with caption: Figure 3: Accuracy vs.
Storage
Efficiency: Comparative analysis
of three RAG systems - MiniRAG,
LightRAG, and GraphRAG. ective for on-device deployment.](..\images\document\page_2_figure_10.jpg)

*Figure 3: Accuracy vs.
Storage
Efficiency: Comparative analysis
of three RAG systems - MiniRAG,
LightRAG, and GraphRAG. ective for on-device deployment.*


--- End of Page 2 ---

... LiHua: The intensity of Kendall’s journey really keeps us on edge! It’s hard not to root for him despite everything, especially when you see how hurt he is.

Kieran: I really think it’s a mix of both! On one hand, he craves that power and validation, but on the other, he seems desperate to carve out his own identity separate from Logan’s shadow. It’s such an interesting storyline, watching him fight that internal battle.

... EmilyBurnett: Absolutely! The suspense makes it so much more thrilling. Plus, with all the character complexities, there’s never a dull moment.

Event Generation with Human Oversight. Events serve as conversation catalysts, functioning as carefully crafted scripts that guide character interactions and dialogue topics. While GPT-4-mini occasionally provides creative inspiration, our team primarily authors these events through deliberate human curation to ensure narrative coherence and authenticity. The conversation generation process is powered by AgentScope (Gao et al., 2024a), which transforms these event scripts into natural dialogues. Below is a representative excerpt of events from a typical week:



|  | 0 | 1 | 2 |
| --- | --- | --- | --- |
| 0 | Time | Participants | Case |
| 1 | 20260818_10:00 | Li Hua and Thane Cham-\nbers | Thane Chambers asks Li Hua which character\nin the game Witcher 3 Li Hua likes the best and\nwhy. |
| 2 | 20260819_10:00 | Li Hua and Jake Watson | Li Hua messages Jake Watson asking Jake if he\nhas some time during the weekend to help Li\nHua improve his dribbling skills. |
| 3 | 20260820_14:00 | Li Hua, Emily, and\nOthers in TVfan group | Emily Burnett creates a poll\nfor\nthe group to\nvote on their favorite HBO series of all time. |


Query Set Design. Our query set has two dimensions: event-based content and reasoning complexity. The event-based dimension encompasses six categories (When, Where, Who, What, How, and Yes/No questions), while the reasoning complexity distinguishes between single-hop and multi-hop queries based on required inferential steps. The following examples illustrate these diverse query types:



|  | 0 | 1 | 2 |
| --- | --- | --- | --- |
| 0 | Time | Participants | Case |
| 1 | 20260818_10:00 | Li Hua and Thane Cham- | Thane Chambers asks Li Hua which character |
| 2 |  | bers | in the game Witcher 3 Li Hua likes the best and |
| 3 |  |  | why. |
| 4 | 20260819_10:00 | Li Hua and Jake Watson | Li Hua messages Jake Watson asking Jake if he |
| 5 |  |  | has some time during the weekend to help Li |
| 6 |  |  | Hua improve his dribbling skills. |
| 7 | 20260820_14:00 | Li Hua, Emily, and | Emily Burnett creates a poll\nfor\nthe group to |
| 8 |  | Others in TVfan group | vote on their favorite HBO series of all time. |



--- End of Page 3 ---

