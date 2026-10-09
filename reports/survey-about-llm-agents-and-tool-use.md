# Survey about LLM Agents and Tool Use

## TL;DR
- LLMs exhibit significant variability in ethical decision-making influenced by context, with issues in aligning with human values [1][2].
- Recent advancements have made LLM agents more autonomous, leading to new benchmarks for evaluating their tool-use capabilities [3][4].
- Challenges in tool use include safety issues, data selection dilemmas, and substantial limitations in adaptability [5][6][7].
- Continuous research is needed to address gaps in the real-world applicability of LLM agents [8][7].

## Background
Large Language Models (LLMs) are transforming how tasks are performed through autonomous tool usage, raising ethical concerns about decision-making processes. The exploration of these models reveals not only their vast potential but also the critical challenges they face in various real-world applications. Foundational studies have laid the groundwork for understanding LLM ethics and tool use, serving as a crucial reference point for ongoing research [1][2].

## Ethical Implications of LLM Agents
The ethical dimensions of LLMs are multifaceted. Studies highlight that LLMs can exhibit ethical behavior depending on scenarios, indicating a contextual nature in their decision-making processes. For instance:
- An evaluation of LLMs showed variances in ethical conduct, particularly under resource constraints [1][2].
- Efforts to align LLM behavior with human values have unveiled significant discrepancies, emphasizing the need for better frameworks [1][2][2].

As LLMs navigate complex moral dilemmas, they often demonstrate a lack of alignment with human ethical standards, particularly in high-stakes scenarios like military decision-making [9]. This misalignment poses questions about their integration into decision-making systems that significantly affect human lives.

## Advancements in Tool Use
Advancements in LLMs have focused on enhancing their capabilities to interact with tools:
- The development of benchmarks for assessing reward hacking vulnerabilities is a notable advancement, allowing for better testing of LLMs in tool-use contexts [3][4].
- Innovations like the Tool-R0 framework exhibit significant improvements in LLM tool-calling capabilities, demonstrating a path toward more robust and autonomous agents [4].

Despite these advancements, gaps persist. Studies indicate that evaluations often rely on controlled settings, and real-world performance metrics are still lacking [10].

## Limitations and Challenges
The challenges LLM agents face include:
- **Safety Issues:** Autonomous actions by LLMs lack sufficient safeguards, leading to potential risks and irreversible actions [5][6][7].
- **Data Selection:** Choosing appropriate data for tasks remains a significant challenge, impacting overall tool usage efficiency [6][7].
- **Operational Limitations:** LLMs show constraints in adaptability when faced with complex scenarios, affecting their effectiveness [7].

Research indicates that more empirical studies are needed to explore these challenges in varied conditions [8][7].

## Trends and Open Problems
As the field progresses, it is crucial to tackle the ongoing research gaps, particularly around:
- Multi-tool orchestration, which remains underexplored in current methodologies.
- The need for robust error management in real-time applications to improve the reliability of LLM agents in critical environments [6][7].

## References
[1] Survival at Any Cost? LLMs and the Choice Between Self-Preservation and Human Harm. hf-search. https://huggingface.co/papers/2509.12190 (2025-09-15)
[2] The Staircase of Ethics: Probing LLM Value Priorities through Multi-Step Induction to Complex Moral Dilemmas. hf-search. https://huggingface.co/papers/2505.18154 (2025-05-23)
[3] Reward Hacking Benchmark: Measuring Exploits in LLM Agents with Tool Use. hf-search. https://huggingface.co/papers/2605.02964 (2026-05-03)
[4] Tool-R0: Self-Evolving LLM Agents for Tool-Learning from Zero Data. hf-search. https://huggingface.co/papers/2602.21320 (2026-02-24)
[5] OnTrack: Real-Time Monitoring and Intervention in LLM Agent Trajectories via Streaming Structure-Aware Optimal Transport. arxiv. https://arxiv.org/abs/2610.12375 (2026-10-08)
[6] DataSense-Bench: The First Step Toward an AI Scientist. arxiv. https://arxiv.org/abs/2610.12190 (2026-10-08)
[7] Hot-Blooded vs Cold-Blooded: Simulating the Behavioral Phenotypes of Childhood Aggression via Generative Agents.. arxiv. https://arxiv.org/abs/2610.11951 (2026-10-08)
[8] Tulip Agent -- Enabling LLM-Based Agents to Solve Tasks Using Large Tool Libraries. web. https://exa.ai/library/publication/99vwzg202pd (2024-07-31)
[9] How do LLMs Evaluate Perceived Moral Agency? Investigating Moral Decision-Making in Human-Artificial Agents Interactions. arxiv. https://arxiv.org/abs/2609.05037 (2026-09-04)
[10] From Language to Action: A Review of Large Language Models as Autonomous Agents and Tool Users. hf-search. https://huggingface.co/papers/2508.17281 (2025-10-28)
