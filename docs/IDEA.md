# Technical Concepts and Implementation Ideas

This document outlines key technical concepts and training strategies derived from recent research in personalized generation and human pose control.

---

## 1. Custom Diffusion: Multi-Concept Fine-tuning
**Reference:** [Custom Diffusion (ArXiv: 2212.04488)](https://arxiv.org/abs/2212.04488)

In the Custom Diffusion framework, the text encoder remains frozen to preserve linguistic knowledge. The optimization focuses on a minimal set of parameters:
*   **Cross-Attention Projections:** Only the key ($K$) and value ($V$) projection matrices within the U-Net's cross-attention layers are updated.
*   **Modifier Token Embeddings:** The embedding for a new modifier token (e.g., `V*`) is learned to represent the specific concept/object.

---

## 2. HyperHuman: Multi-Stage Training Strategy
**Reference:** [HyperHuman (ArXiv: 2311.12052)](https://arxiv.org/abs/2311.12052)

To effectively disentangle identity from pose, a two-stage training approach is employed:

### Stage 1: Appearance Control Pretraining
*   **Objective:** Train the Appearance Model to effectively inject identity features via the **Macro-to-micro Spatial Self-Attention (MSSA)** module.
*   **Strategy:** The Pose ControlNet is disabled. The model performs **Identity Reconstruction**, where it attempts to reconstruct a reference image $I_R$ using $I_R$ itself as the condition.
*   **Rationale:** This setup mandates that the Main U-Net relies exclusively on the Appearance Model’s keys and values for fine-grained structural and texture details.

### Stage 2: Appearance-Disentangled Pose Control
*   **Objective:** Train the Pose ControlNet to govern structural elements without interfering with the established appearance.
*   **Strategy:** Both the Pose ControlNet and the Appearance Model are enabled. The model is trained to generate a target image given a Reference Image $I_R$ and a Target Pose $I_C$.
*   **Rationale:** Since the Appearance Model has already mastered identity preservation in Stage 1, the ControlNet can focus solely on spatial configuration (e.g., orientation and limb positioning), effectively disentangling motion from identity.

---

## 3. Inference Techniques: Modified CFG
During the inference phase, a modified version of **Classifier-Free Guidance (CFG)** is utilized to improve identity fidelity.

While standard CFG typically alternates between a conditional and a null text prompt, this implementation (inspired by MagicPose) toggles the **Reference Image**:

*   **Unconditional Branch:** The Reference Image input is replaced with a null/empty tensor ($\phi$).
*   **Conditional Branch:** The actual Reference Image is provided.

The final noise prediction is calculated as:
$$\epsilon_{\theta}(x_t, c, I_R) = \epsilon_{\theta}(x_t, c, \phi) + s \cdot (\epsilon_{\theta}(x_t, c, I_R) - \epsilon_{\theta}(x_t, c, \phi))$$

Where $s$ represents the guidance scale. Empirical results suggest that increasing $s$ significantly enhances identity preservation (measured via Face-Cosine similarity) without compromising the overall image structure.

