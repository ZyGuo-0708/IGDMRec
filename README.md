# IGDMRec (TMM)

Pytorch implementation for  
**IGDMRec: Behavior Conditioned Item Graph Diffusion for Multimodal Recommendation**  
*IEEE Transactions on Multimedia (TMM), accepted.*

- **arXiv version**: https://arxiv.org/abs/2512.19983  
- **Authors**: Ziyuan Guo, Jie Guo, Zhenghao Chen, Bin Song, Fei Richard Yu  

This implementation is built upon the **MMRec** framework.  
We sincerely thank the authors for their excellent work.

The repository focuses on **diffusion-based denoising for multimodal recommendation**.

---

## How to Run

1. Put the dataset (e.g. `baby`) under the `datasets/` directory.
2. Construct the behavior-aware item–item graph:
```bash
python build_iib_graph.py
```
3. Train IGDMRec:
```bash
python main.py
```

Hyper-parameters can be modified via config files.

---

## Environment

- Python 3.10.14  
- PyTorch 2.3.0  

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{guo2025igdmrec,
  title   = {IGDMRec: Behavior Conditioned Item Graph Diffusion for Multimodal Recommendation},
  author  = {Guo, Ziyuan and Guo, Jie and Chen, Zhenghao and Song, Bin and Yu, Fei Richard},
  journal = {arXiv preprint arXiv:2512.19983},
  year    = {2025}
}
```
