# Tổng hợp `wake_benchmark`

EPE trung bình [px] (trung bình trên mọi ppp × SNR × seed của trường; với `cai_*`: trên các cặp ảnh test). In = vùng top-10 % |ω| của trường gốc; thời gian = s/cặp trên CPU.

| trường | ofpiv | ofpiv_warp | crosscorr | raftpiv_v1 | raftpiv | raftpiv_ls |
|---|---|---|---|---|---|---|
| wake_base | 0.176 (in 0.272) | 0.191 (in 0.299) | **0.131 (in 0.237)** | 0.154 (in 0.224) | 0.140 (in 0.209) | 0.158 (in 0.237) |
| wake_recirc | 0.225 (in 0.246) | 0.241 (in 0.271) | **0.191 (in 0.226)** | 0.222 (in 0.255) | 0.213 (in 0.241) | 0.226 (in 0.257) |
| wake_far | 0.282 (in 0.320) | 0.308 (in 0.352) | **0.277 (in 0.309)** | 0.309 (in 0.332) | 0.303 (in 0.330) | 0.318 (in 0.352) |
| cai_cylinder | 0.097 (in 0.104) | 0.112 (in 0.131) | **0.049 (in 0.058)** | 0.099 (in 0.128) | 0.098 (in 0.125) | 0.105 (in 0.132) |
| cai_backstep | 0.079 (in 0.090) | 0.105 (in 0.119) | **0.032 (in 0.049)** | 0.056 (in 0.063) | 0.055 (in 0.059) | 0.103 (in 0.116) |
| thời gian [s] | 2.29 | 6.11 | 0.87 | 0.89 | 0.88 | 6.56 |

Bước sóng cắt MTF [px] (trường sin u = 0.5·sin(2πx/λ), trung bình trên ppp × SNR; NaN = không đạt):

| phương pháp | lambda_c_0.9 | lambda_c_0.5 |
|---|---|---|
| ofpiv | 32.9 | 16.5 |
| ofpiv_warp | 37.5 | 16.1 |
| crosscorr | 42.1 | 23.3 |
| raftpiv_v1 | nan | 97.5 |
| raftpiv | 203.3 | 45.5 |
| raftpiv_ls | 57.0 | 30.6 |
