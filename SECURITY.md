# Security & Dataset Privacy Policy

## 1. Codebase Security Vulnerability Reporting
If you discover a security issue in the `mmregcheck` evaluation toolkit or GitHub Actions leaderboard workflow, please **do not** open a public issue. Instead, use **GitHub Private Vulnerability Reporting** under the [**Security** tab](https://github.com/khanhtq29vn/MM-RegCheck/security).

## 2. Dataset Privacy & Takedown Requests (`MedReg-MM`)
All advertisements and facility records in `MedReg-MM` were collected from publicly accessible sources and statutory registries in accordance with institutional ethical guidelines. If you are a facility owner or individual and wish to request anonymization or removal of a specific record ID, please open a confidential request or contact the corresponding authors (`ngtbinh@hcmus.edu.vn`, `khanhtq@nus.edu.vn`).

## 3. Collaborator Guardrails Against Accidental Data Leaks
Never commit unencrypted dataset splits (`*.csv`, `*.parquet`, `*ground_truth*.jsonl`) or API keys (`.env`) to this repository. Our CI pipeline (`tests/test_data_leak_guard.py`) automatically rejects any Pull Request containing unencrypted dataset files larger than 100 KB.
