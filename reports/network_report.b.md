# Network Report — B

* Date: 2026-09-24
* Machine: B
* OS: Windows
* Hardware profile: C (no discrete GPU)
* Python environment: Conda aigcfr, Python 3.11.16
* Probe method: HTTP request, intended timeout 5 s

## Connectivity Results

|Endpoint|Purpose|Result|Time|
|-|-|-:|-:|
|https://pypi.tuna.tsinghua.edu.cn/simple/|pip mirror|HTTP 200|35447 ms|
|https://mirrors.aliyun.com/pytorch-wheels/|PyTorch wheel mirror|HTTP 200|3467 ms|
|https://download.pytorch.org/whl/cu128|PyTorch official index|HTTP 200|771 ms|
|https://hf-mirror.com|HuggingFace mirror|HTTP 200|1063 ms|
|https://www.modelscope.cn|ModelScope|HTTP 200|3394 ms|
|https://zenodo.org|Synthetic dataset hosting|HTTP 200|2128 ms|
|https://github.com|Code / Releases|HTTP 200|4313 ms|
|https://arxiv.org|Papers|HTTP 200|1172 ms|

## Configuration

* pip index: https://pypi.tuna.tsinghua.edu.cn/simple
* pip trusted host: pypi.tuna.tsinghua.edu.cn
* HF\_ENDPOINT: https://hf-mirror.com

## Conclusion

All 8 tested endpoints were reachable on machine B, so the normal download routes can be used and the fallback plan in docs/MIRRORS.md §5 is not currently required.

The Tsinghua PyPI endpoint returned HTTP 200 but was unusually slow in this probe (35447 ms). Actual pip behavior should therefore be checked during package installation rather than treating this single HTTP timing as representative.

