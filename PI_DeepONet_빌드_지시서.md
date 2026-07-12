# PI-DeepONet 빌드 지시서 (물리정규화 surrogate)

> **브랜치: `els-hedging-pideeponet`** 에서만 작업 (v2 이론가 baseline 보존).
> **목표:** 기존 데이터기반 DeepONet(이론가 R² 0.937)에 **다자산 Black-Scholes PDE 잔차를 soft penalty로 추가**한 PI-DeepONet을 만들어, **순수 DeepONet 대비 (a) 이론가 정확도, (b) 데이터효율(적은 MC 라벨로도 잘 되는지)** 이 개선되는지 검증한다.
>
> **전제 (중요):**
> - **MC 재실행 금지.** 기존 `data/cache/product_mc.parquet` / dataset 캐시 그대로 사용 (MC 라벨 = 데이터손실 앵커).
> - **재현성:** 모든 torch 학습함수 시작부 `torch.manual_seed(seed)` 재시딩 (v2 픽스와 동일 관례). fold별 재시딩.
> - **독립 구현.** 팀원 `PI-DeepONet-ELS` repo 복붙 금지. 레퍼런스는 업로드된 **Wang·Perdikaris, "Learning the solution operator ... with physics-informed DeepONets" (Sci. Adv. 2021)** 논문.
> - 컨벤션(180일 vol·corr, KR NS금리, 필터 23,479)은 v1/v2와 동일.

---

## 0. 핵심 아이디어
- 이론가 V는 위험중립 GBM 하에서 **다자산 BS PDE**를 만족한다. 이 PDE 잔차를 학습 손실에 넣어 **물리적으로 일관된 surrogate**를 만든다.
- **autocall·KI(경로의존·불연속)는 물리로 직접 안 넣고 MC 데이터손실이 담당** → PDE 잔차는 **연속영역(상환/평가일 사이)의 매끄러운 BS 동역학만** 규제. (이게 물리정규화가 achievable한 이유.)
- 최종 비교: **plain DeepONet vs PI-DeepONet**, 그리고 **데이터효율 곡선**(MC 라벨 25/50/100%).

## 1. 아키텍처 재구성 (`pi_deeponet.py` 신규)
PINN 잔차를 계산하려면 출력이 **상태좌표의 함수**여야 하므로 trunk를 좌표입력으로 바꾼다.
- **Branch (operator 입력 u):** 계약+시장 요약 = 기존 trunk/branch 피처(`strk_0..11, B, coupon, tenor` + `u0..9, sig1..3, rho12/13/23, sig_eff, r`). → MLP/1D-CNN로 인코딩 → 계수벡터 `b_k`.
- **Trunk (좌표 입력 y):** **`(x1, x2, x3, τ)`** = 3자산 로그가격(or 정규화가격 Sᵢ) + 잔여시간. → MLP → 기저 `t_k(y)`.
- **출력:** `V(y; u) = Σ_k b_k(u)·t_k(y) + b_0`. (연속·미분가능하도록 trunk 활성함수는 tanh/GELU.)
- 학습대상 스케일: MC와 동일 스케일(정규화 원금=1). 좌표계는 **로그가격 x=lnS** 권장(BS PDE가 상수계수화되어 학습 안정).

## 2. 물리 — 다자산 BS PDE 잔차
정규화가격 S(=1 기준), 위험중립. 로그가격 x_i=ln S_i 좌표에서:

  ∂V/∂t + Σ_i (r − ½σ_i²) ∂V/∂x_i + ½ Σ_{i,j} ρ_ij σ_i σ_j ∂²V/∂x_i∂x_j − rV = 0

- σ_i, ρ_ij, r 은 **각 상품의 branch 입력에서 가져옴**(상품별 상수).
- 미분은 **torch.autograd**로 계산(1차 `∂V/∂x_i, ∂V/∂t`, 2차 `∂²V/∂x_i∂x_j` 헤시안). `create_graph=True`.
- **잔차** `f_pde = LHS` → `L_pde = mean(f_pde²)` (collocation점 위).

## 3. 손실 구성
```
L = L_data + λ_pde · L_pde + λ_term · L_term
```
- **L_data:** 각 상품의 **발행시점 좌표**(x=0 즉 S=1, τ=만기)에서 `V(that; u)` = MC 이론가. (MSE 또는 MAE — v2에서 MAE 우세였으니 MAE 기본, MSE도 비교.)
- **L_pde:** collocation점(§4)에서 BS PDE 잔차².
- **L_term:** 만기좌표 τ=0에서 `V = 최종 payoff`(worst-of ≥ 최종행사가 → 원금+쿠폰, KI이력·worst<B → worst·원금 등). payoff 함수로 직접 부여. (autocall·KI 경로의존 세부는 데이터가 담당하므로, terminal은 **만기 단일시점 조건만** 근사로 넣고 λ_term 작게 시작.)
- λ는 **[λ_pde, λ_term]** 를 그리드(예: λ_pde∈{0(=plain), 0.01, 0.1, 1})로 스캔 — **λ_pde=0이 plain DeepONet과 동일**해야 함(정합 체크).

## 4. Collocation 샘플링
- 상품별로 `(x, τ)` 를 도메인에서 무작위 샘플: x_i ∈ [ln0.3, ln1.7] 정도(배리어~ATM 커버), τ ∈ [0, 만기].
- 상품당 N_col(예: 64) 점. 미니배치마다 재샘플(SGD collocation).
- σ/ρ/r 은 해당 상품 값 사용.

## 5. 실험 설계 (핵심 비교)
1. **plain vs PI:** 동일 아키텍처·seed에서 λ_pde=0(plain) vs 최적 λ_pde(PI) → 이론가 R²/MAPE/bp/Spearman 비교. **개선 있나?**
2. **데이터효율:** MC 라벨을 **25% / 50% / 100%** 만 L_data에 사용(나머지는 PDE만) → plain vs PI 정확도 곡선. **물리가 저데이터에서 이득 주나?** (PI-DeepONet의 핵심 기대효과)
3. walk-forward 4-fold, OOS 평가는 v2와 동일 분할.

## 6. 재현성·평가
- 학습함수 시작부 `torch.manual_seed(seed)`; 같은 실행 2회 pooled R² 일치 확인.
- 지표: **R², MAPE, bp오차, MAE, RMSE, Spearman** (이론가=MC 기준). moneyness 구간별도 선택.
- PDE 잔차 자체의 수렴(L_pde 로그)도 기록.

## 7. 산출물 (`data/out/`)
- `pi_deeponet_comparison.csv` : plain vs PI(λ별) 지표표.
- `pi_deeponet_data_efficiency.csv` : 라벨 25/50/100% × {plain, PI} 지표.
- `pi_deeponet_reproducibility_check.csv` : 2회 실행 일치.
- 그림: 데이터효율 곡선, plain-vs-PI 산점도, L_pde 수렴.

## 8. 실행 순서 (MC 제외)
```
git checkout els-hedging-pideeponet
# 데이터/ MC 캐시 그대로 사용
python -m els_hedging_v1.pi_deeponet --lambda-pde 0     # plain 정합 체크 (=기존 DeepONet과 동일해야)
python -m els_hedging_v1.pi_deeponet --lambda-scan      # λ_pde 그리드 + 데이터효율
python -m els_hedging_v1.pi_deeponet --report           # 비교표·그림·재현성
```
> 환경: `KMP_DUPLICATE_LIB_OK=TRUE`, `PYTHONIOENCODING=utf-8`. MC 재실행 없음.

## 9. 성공 기준 / 정직한 리스크
- **최소 성공:** λ_pde=0이 plain과 정확히 일치(구현 정합) + PDE 잔차가 학습되며 수렴 + PI가 **저데이터(25/50%)에서 plain보다 우위**면 "물리 유효".
- **리스크(정직히):** 이론가가 이미 R² 0.937로 잘 되어 있어, **100% 데이터에선 PI 이득이 미미할 수 있음**(v2 잔차보정처럼). 그때 **가치는 "데이터효율"과 "물리 일관성"** 에 있음 — 그 프레이밍으로 결과 해석.
- terminal/autocall 근사가 불안정하면 **λ_term=0으로 두고 PDE-in-continuation + data만** 으로 축소(그래도 유효한 물리정규화).
