In our ACL paper "[Define, Evaluate, and Improve Task-Oriented Cognitive Capabilities for Instruction Generation Models](https://arxiv.org/abs/2301.05149)", we formulate task-oriented cognitive capabilities: (i) the ability to quickly generate good candidate utterances (the search capability) (ii) the ability to predict how a listener interprets those utterances and choose the most appropriate one (the pragmatic capability). We design an evaluation scheme for comparing these capabilities of a language model with those of a human. Applying this scheme to examine various models in a navigation instruction generation problem, we find that their pragmatic capability is severely lacking. This insight leads us to augment them with better models of the listener and obtain a significant boost of 11% in success rate in guiding real humans.

NEWS (Dec 19, 2022): We release the human evaluation interface (`human_eval_interface/`).

NEWS (Jun 2, 2023): We also release the pragmatic speaker codes with pretrained models (`pragmatic_speaker/`).

The instructions are placed under each folder's README.md.

![image info](./pragmatic_speaker/bounded_pragmatic.png)
