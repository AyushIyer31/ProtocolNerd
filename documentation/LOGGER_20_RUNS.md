# Logger isolation: 20 paired runs

Each fresh process imports main before collecting logger tests. Baseline fails4/5; fixedtestfixturepasses5/5. NoAWS/modelcalls; fakeS3 only. Oneunrelatedrankingfailure remainsoutside thisfive-testset.

| Run | Before | After |
|---|---|---|
| 1 | 4 failed, 1 passed, 1 warning in 0.12s | 5 passed, 1 warning in 0.02s |
| 2 | 4 failed, 1 passed, 1 warning in 0.09s | 5 passed, 1 warning in 0.01s |
| 3 | 4 failed, 1 passed, 1 warning in 0.09s | 5 passed, 1 warning in 0.01s |
| 4 | 4 failed, 1 passed, 1 warning in 0.09s | 5 passed, 1 warning in 0.01s |
| 5 | 4 failed, 1 passed, 1 warning in 0.09s | 5 passed, 1 warning in 0.01s |
| 6 | 4 failed, 1 passed, 1 warning in 0.09s | 5 passed, 1 warning in 0.02s |
| 7 | 4 failed, 1 passed, 1 warning in 0.09s | 5 passed, 1 warning in 0.02s |
| 8 | 4 failed, 1 passed, 1 warning in 0.10s | 5 passed, 1 warning in 0.01s |
| 9 | 4 failed, 1 passed, 1 warning in 0.09s | 5 passed, 1 warning in 0.02s |
| 10 | 4 failed, 1 passed, 1 warning in 0.10s | 5 passed, 1 warning in 0.01s |
| 11 | 4 failed, 1 passed, 1 warning in 0.09s | 5 passed, 1 warning in 0.02s |
| 12 | 4 failed, 1 passed, 1 warning in 0.12s | 5 passed, 1 warning in 0.01s |
| 13 | 4 failed, 1 passed, 1 warning in 0.10s | 5 passed, 1 warning in 0.01s |
| 14 | 4 failed, 1 passed, 1 warning in 0.09s | 5 passed, 1 warning in 0.01s |
| 15 | 4 failed, 1 passed, 1 warning in 0.12s | 5 passed, 1 warning in 0.01s |
| 16 | 4 failed, 1 passed, 1 warning in 0.09s | 5 passed, 1 warning in 0.01s |
| 17 | 4 failed, 1 passed, 1 warning in 0.09s | 5 passed, 1 warning in 0.01s |
| 18 | 4 failed, 1 passed, 1 warning in 0.09s | 5 passed, 1 warning in 0.02s |
| 19 | 4 failed, 1 passed, 1 warning in 0.10s | 5 passed, 1 warning in 0.02s |
| 20 | 4 failed, 1 passed, 1 warning in 0.09s | 5 passed, 1 warning in 0.02s |
