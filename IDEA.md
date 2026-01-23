# https://arxiv.org/pdf/2212.04488v2
In Custom Diffusion, the text encoder is frozen.
Only:
- Cross-attention key/value projection matrices in the U-Net, and
- The embedding of the new modifier token (e.g., V*)
# https://arxiv.org/pdf/2311.12052
#### Stage 1: Appearance Control Pretraining
*   **Goal:** Teach the Appearance Model how to inject identity features via the MSSA module.
*   **Setup:**
    *   **Pose ControlNet:** Disabled/Not included.
    *   **Task:** Reconstruction. The model tries to reconstruct the Reference Image $I_R$ given $I_R$ as the condition.
    *   **Why:** This forces the Main UNet to rely entirely on the Appearance Model's keys/values to get the details right.

#### Stage 2: Appearance-Disentangled Pose Control
*   **Goal:** Teach the Pose ControlNet to control structure *without* overriding the appearance.
*   **Setup:**
    *   **Pose ControlNet:** Enabled and trainable.
    *   **Appearance Model:** Enabled and fine-tuned.
    *   **Task:** Given Reference Image $I_R$ and a Target Pose $I_C$, generate the target image.
*   **Why:** Since the Appearance Model is already trained (from Stage 1), the Pose ControlNet doesn't need to learn "what a face looks like." It only needs to learn "where the head is turned." This effectively disentangles motion from identity.


#### Apply CFG too

During inference, they use a modified Classifier-Free Guidance (CFG) to balance the conditioning.

Standard CFG usually toggles the text prompt. MagicPose toggles the **Reference Image**.
*   **Unconditional:** The Reference Image input is replaced with a null/empty tensor.
*   **Conditional:** The Reference Image is provided.

The final noise prediction is:
$$\epsilon = \epsilon_{uncond} + s \cdot (\epsilon_{cond} - \epsilon_{uncond})$$

Where $s$ is the guidance scale. They found that increasing this scale significantly improves identity preservation (Face-Cos score) without breaking the image structure.

