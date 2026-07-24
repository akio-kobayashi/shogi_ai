# Third-party references

## T²MLR

The implementation in `models/t2mlr.py` is an experiment-specific adaptation of:

- Ziyang Cai, Xingyu Zhu, Yihe Dong, Yinghui He, and Sanjeev Arora,
  “T²MLR: Transformer with Temporal Middle-Layer Recurrence,” arXiv:2607.15178.
- Official implementation: <https://github.com/princeton-pli/T2MLR>
- Official implementation license: Apache License 2.0.

The original Hugging Face wrapper is not copied verbatim. The recurrence equations,
gated fusion, exact recurrent decoding path, and Jacobi-style batch approximation are
reimplemented for the small decoder-only shogi model. Source references and the
experiment-specific simplifications are documented directly in `models/t2mlr.py`.
