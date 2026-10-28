# Scout golden-set eval — 2026-10-28T23:37:03+00:00

> Scout answers research questions with 100% of claims verified against sources, in 30 s (median), for £0.024 per report; 68% of golden-set questions answered correctly.

| Metric | Overall | web | docs | both | unanswerable |
|---|---|---|---|---|---|
| Cases scored | 50 | 20 | 12 | 10 | 8 |
| Task success (judge) | 68% | 65% | 100% | 10% | 100% |
| Key-point coverage | 81% | 81% | 100% | 57% | – |
| Tool selection | 100% | 100% | 100% | 100% | 100% |
| Claim support (final report) | 100% | 100% | 100% | 100% | 100% |
| Claim support (first check) | 87% | 88% | 94% | 78% | 83% |
| Citation accuracy (judge) | 66% | 74% | 76% | 54% | 45% |
| Avg steps | 45.7 | 47.5 | 40.5 | 46.6 | 47.6 |
| Avg cost per report | £0.0237 | £0.0278 | £0.0174 | £0.0231 | £0.0235 |
| Latency p50 | 30s | 39s | 24s | 27s | 31s |
| Latency p95 | 49s | 54s | 26s | 45s | 42s |
| Errors | 0 | 0 | 0 | 0 | 0 |
| Skipped | 0 | 0 | 0 | 0 | 0 |

Spend: research £1.1844 + judges £0.4484.
Commit `1ee463a` · cases file `8f958e15ec01` · judge prompts {'judge_task': 'v1:8059926b', 'judge_citation': 'v1:c5ee7e40'}

## Cases that did not pass

- **web-minimum-wage-start** (scored, passed=False): The report states most provisions took effect on 1 April 1999, covering point 1. It explicitly says it could not find the GBP 3.60 main adult rate, so point 2 is not covered.
- **web-boe-independence** (scored, passed=False): The report states the May 1997 announcement (6 May 1997) and ties the Bank of England Act (Royal Assent 1998) to the formal legal basis. It says the Act gave a statutory basis for the MPC and that the MPC formulates and implements monetary policy, but never explicitly says the MPC sets interest rates, so point 3 is only implied and counted as not covered.
- **web-speed-of-light** (scored, passed=False): The report says the available material does not give the value of c or explain why it is exact. The value 299792458 appears only in a source title in the limitations section, and the report never states it as an answer. It does not mention the 1983 metre definition as the reason either, so neither key point is covered.
- **web-bletchley-declaration** (scored, passed=False): The report gives the summit date (1-2 November 2023) and the frontier AI risk focus, but it does not mention Bletchley Park by name as the venue, and it says outright that the number of signatories is not known, so point 2 is not covered. Nothing in it contradicts the key.
- **web-transformer-paper** (scored, passed=False): The report names 'Attention Is All You Need' and dates it to 2017 (arXiv 12 Jun 2017, NIPS 2017) with Vaswani et al. as authors. It never says the authors were at Google, so the second key point is only partly covered.
- **web-ben-nevis-height** (scored, passed=False): The report states Ben Nevis is 1,345m, matching key point 1 (feet not required to match since metres given). It never says it is the highest mountain in the UK (only the source title mentions Great Britain's tallest, not in body), so point 2 is not covered.
- **web-turing-test-paper** (scored, passed=False): The report says the 1950 paper's proposal is based on 'The Imitation Game' (point 3), but it explicitly says it lacks the title and journal. Point 1 is not covered, and point 2 is only partly covered because the year 1950 is given without the journal Mind.
- **both-ax7-battery-voltage** (scored, passed=False): The report states the AX-7 uses a LiFePO4 pack (key point 1) but explicitly says the single-cell nominal voltage is not available, so the ~3.2 V figure (key point 2) is not covered.
- **both-mileage-vs-hmrc** (scored, passed=False): The report confirms Quillmere's 45p/25p rates but says HMRC's rate rose to 55p for 2026-27, so the rates no longer match. That conflicts with the key's statement that HMRC's rates are 45p/25p and match.
- **both-daventry-county** (scored, passed=False): The report identifies the Daventry fulfilment site as the outage location but says the county cannot be determined, so it never states Northamptonshire (key point 2).
- **both-ax7-ip-rating** (scored, passed=False): The report states IP54 and explains that 5 means dust protected, but it says the meaning of 4 (splashing water) could not be established, so key point 3 is not covered.
- **both-singapore-cabin-class** (scored, passed=False): The report states the policy rule (economy for 6 hours or less, premium economy for more than 6 hours). It does not state that the London to Singapore flight takes about 13 hours or conclude that premium economy is allowed, and instead says the cabin class cannot be determined.
- **both-ax7-speed-vs-walking** (scored, passed=False): The report states the 1.2 m/s shared-zone limit (point 1), but says no typical human walking speed was found and gives no comparison, so point 2 is not covered.
- **both-company-size** (scored, passed=False): The report gives the GBP 23.4m forecast and 212 headcount, and concludes Quillmere does not meet the small thresholds (£15m, 50 employees). It never says the company is medium-sized, and it does not mention the April 2025 change in the threshold, so key point 2 is only partly covered.
- **both-leeds-population** (scored, passed=False): The report identifies Leeds as the site but says the 2021 census population could not be found, so the roughly 812,000 figure is missing. It does not contradict the key.
- **both-coventry-river** (scored, passed=False): The report correctly says the AX-7 is assembled in Coventry but says the river could not be determined, so it does not state the River Sherbourne. It does not contradict the key.
