# Code for Submission #9107

To split the data heterogeneously onto 6 local workers and preprocess it, run:
```
python preprocessing.py --data covtype --cond 1e-30 --it_max 5000 --n 6
``` 
Then, run DIANA+ (quant+)
```
python main.py --data covtype  --alg SD-DIANA-plus --n 6
```

We support 11 algorithms in total:

Baselines:
- SD-DCGD: DCGD (Khirirat et al. [2018]) + standard quantization (Alistarh et al. [2017])
- SD-DIANA: DIANA (Mishchenko et al. [2019]) + standard quantization (Alistarh et al. [2017])
- BL-DCGD: DCGD (Khirirat et al. [2018]) + block quantization (Alistarh et al. [2017])
- BL-DIANA: DIANA (Mishchenko et al. [2019]) + block quantization (Alistarh et al. [2017])

Proposed Algorithms:
- SD-DCGD-plus-fnl: DCGD+ (Algorithm 1) + standard quantization (Alistarh et al. [2017])
- SD-DCGD-plus: DCGD+ (Algorithm 1) + quantization with varying steps (Section 5)
- SD-DIANA-plus-fnl: DIANA+ (Algorithm 2) + standard quantization (Alistarh et al. [2017])
- SD-DIANA-plus: DIANA+ (Algorithm 2) + quantization with varying steps (Section 5)
- BL-DCGD-plus: DCGD+ (Algorithm 1) + block quantization (Section 4)
- BL-DIANA-plus: DIANA+ (Algorithm 2) + block quantization (Section 4)
- BL-DIANA-plus-fnl: DIANA+ (Algorithm 2) + block quantization (Alistarh et al. [2017])




#### Prerequisites:
- Python 3.6+
- MPI4PY

To reproduce the figures on covtype dataset in our paper, run:
```
bash covtype.sh
```

