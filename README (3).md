# Efficient LLMs via Switchable and Dynamic Quantization

This project explores quantization and precision-switching strategies for efficient fine-tuning of GPT-2 
on the SQuAD dataset. We evaluate different quantization profiles (4-bit, 6-bit, 8-bit), cyclic precision 
training (CPT), and robustness under adversarial typo attacks. LoRA adapters are used to enable 
profile-specific adaptations under a strict training budget of 1,000 steps.

## Repository Structure
- `EIC_LLM_SDQ.ipynb` — main Jupyter notebook with all experiments.
- `step4_barchart.png`, `step5_cpt_loss.png`, `step6_robustness.png` — figures used in the report.
- Report.pdf — the actual report
## Requirements
- Python 3.10+
- PyTorch 2.0+
- Hugging Face Transformers
- Datasets (SQuAD)
- Matplotlib, Pandas, Numpy

## Running the Project
1. Open `EIC_LLM_SDQ.ipynb` in Jupyter or VSCode.
2. Run all cells to reproduce training runs and figures.
3. Figures will be saved as `.png` files for inclusion in the LaTeX report.

## Notes
- Training is limited to 1,000 steps as specified in the assignment.
- Results focus on relative comparisons between quantization strategies, not absolute accuracy.
- The fixed 6-bit profile was used as the baseline for adversarial robustness evaluation, 
  even though 8-bit performed best
