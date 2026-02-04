# DreamBooth Implementation and Extension Task

## 1. DreamBooth Paper Reproduction
**Source:** [DreamBooth Project Page](https://dreambooth.github.io/)

*   **Setup and Reproduce:** Environment setup and reproduction of key experiments outlined in the paper (selecting a few representative parts is sufficient).
*   **Deep Dive:** Develop a comprehensive understanding of the model architecture, implementation details, and training procedures.

## 2. Extending DreamBooth with Pose Control
The current DreamBooth implementation focuses on text-to-image conditioning. The goal is to extend this functionality to incorporate pose-based conditioning.

*   **Requirement:** Integrate 2-3 reference images of a specific subject (e.g., a human) with both a text prompt (e.g., "a man standing on a mountain") and a human pose reference.
*   **Technical Approach:** Explore related research and architectures to integrate pose conditioning into the existing DreamBooth framework.
*   **Target Output Example:** Given reference images of a Unitree robot, use a pose reference from a different robot model to generate an image of the Unitree robot performing the corresponding activity.
