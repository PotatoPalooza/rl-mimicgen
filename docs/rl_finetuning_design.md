# MuJoCo RL Fine-Tuning First Pass

## Recommendation

Use a demo-augmented on-policy actor-critic loop as the first online fine-tuning method.
In this repo that means:

- load the robomimic checkpoint directly as the actor
- preserve the robomimic observation pipeline, including image encoders and BC-RNN state
- add a separate value network for online updates
- optimize a PPO-style clipped policy loss plus a decaying BC loss on the original demo dataset

This is closest in spirit to DAPG while staying simpler to implement correctly with robomimic checkpoints than natural policy gradient or offline-to-online Q-learning.

## Methods Considered

- DAPG: strong fit for sparse long-horizon manipulation with demonstrations, but the original method uses natural policy gradient machinery that is heavier than needed for a first pass.
- PPO: simple and reliable on-policy baseline, but too easy to drift away from the BC initialization without an explicit demo anchor.
- AWAC: attractive for offline-to-online fine-tuning, but adding replay, bootstrapped critics, and stable image / recurrent support is more work than a first pass warrants.
- SAC / CQL / IQL variants: sample-efficient in continuous control, but they are a worse plug-and-play fit for robomimic BC-RNN image policies and require more invasive actor / critic changes.

## Why This Choice

- MimicGen tasks are long-horizon and typically trained from demonstration-rich robomimic checkpoints.
- The actor already contains the right visual encoder and recurrent structure; replacing it would throw away the main asset.
- On-policy updates avoid replay-buffer design questions for image observations and recurrent state.
- The extra BC loss reduces catastrophic drift under sparse rewards.
- Saving the updated actor back into robomimic checkpoint format keeps the result usable with existing MimicGen tooling.

## Main Challenges

- Sparse rewards make pure PPO brittle; the BC anchor is there to keep exploration near successful demonstrations.
- BC-RNN and image policies are stateful and expensive, so the first pass uses single-env sequential rollouts and full-batch updates instead of a highly optimized vectorized trainer.
- robomimic GMM policies are not naturally bounded by a tanh transform, so action clipping is still a rough edge for future cleanup.
- The critic is separate from the actor for simplicity, which is correct but not the most sample-efficient design.

## Papers Considered

- Mandlekar et al., "MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations", CoRL 2023. https://arxiv.org/abs/2310.17596
- Mandlekar et al., "What Matters in Learning from Offline Human Demonstrations for Robot Manipulation", CoRL 2021. https://arxiv.org/abs/2108.03298
- Rajeswaran et al., "Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations", RSS 2018. https://arxiv.org/abs/1709.10087
- Schulman et al., "Proximal Policy Optimization Algorithms", 2017. https://arxiv.org/abs/1707.06347
- Nair et al., "Accelerating Online Reinforcement Learning with Offline Datasets", 2020. https://arxiv.org/abs/2006.09359
