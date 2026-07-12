# ELS_hedging_V1 — 프로젝트 현황 · 이어하기 문서

> 나중에 (Cowork 앱이든 Claude Code CLI든) 이 프로젝트를 이어서 할 때 **이 파일 + 3개 지시서**를 먼저 읽으면 상태가 복원됨.

## 1. 한 줄
ELS(3-star worst-of 스텝다운 오토콜, 낙인) 공정가치 고속산출 — **이론가 중심** 2-stage(DeepONet Stage-1이 MC 이론가 재현 + Stage-2 마진모델). 팀 노션·피드백에 맞춰 **raw부터 독립 재구현**한 버전(V1).

## 2. 컨벤션 (확정)
- MC: **180일 역사적 변동성·상관 + KR FRED 금리(NS곡선) + 일별 KI 감시 + 10만경로**. (팀 현행과 동일)
- 필터: 3-star·worst-of·STEP·KRW·**낙인상품**·fair∈[0.70,1.05]·tenor∈[0.5,5] → **23,479건** (팀 23,151과 정합).
- Stage-2 = **MLP**(교수님 피드백). train/valid = **random shuffle**(test는 미래).

## 3. 현재 결과 (OOS 9,392, MAE 손실 대표본)
- **Stage-1 이론가 R² 0.937**(MAPE 0.47%, bp오차 44.9, Spearman 0.982) — 팀 목표(0.92~0.99) 달성. ← 핵심.
  (els-hedging-v2: torch 재현성 픽스 후 canonical 값. 이전 0.921은 매 fold `torch.manual_seed` 미재시딩으로
  다른 모델 학습 순서에 따라 값이 바뀌는 비재현적 결과였음 — `deeponet.py`/`margin_mlp.py` 학습함수 시작부
  재시딩으로 수정, 2회 독립 실행 R² 완전 일치 확인.)
- Stage-2 MLP R² −0.47, Final −0.01 — **MLP<tree, 부차**(교수님이 Final은 부차라 함).
- 속도(배치 forward): DeepONet 4.08µs/건, MC 18.2s/건 → **~445만배**(팀 130만배와 자릿수 일치; MC 상품당 시간 차이는 하드웨어).
- fair−MC −513(팀 −512), 저가 MAPE 13.9%→고가 1.7%(팀 12.5%→1.9%).
- 손실: **MAE > MAPE > MSE**(팀 관찰 일치).

## 4. 파일 위치
- **코드**: `els_hedging_v1/` (data_prep·market_data·mc_engine·features·datasets·deeponet·margin_mlp·benchmark·metrics·speed_benchmark·case_analysis·eda·splits·figures·run_all)
- **그림**: `data/out/figures/` (01 속도 · 02 fair−MC · 03 per-stage R² · 04 저가케이스 · 05 부호편향 · 06 fair산점도 · eda_00/01)
- **표(csv)**: `data/out/` (model_comparison · filter_cascade · calibration · moneyness · speed_benchmark_summary 등)
- **데이터**: `data/raw/`(원본 3 CSV) · `data/cache/`(MC parquet — **재실행 없이 재사용**)
- **지시서**: `V1_빌드_지시서.md` · `V1_피드백반영_지시서.md` · `V1_발표보완_지시서.md`

## 5. 실행법 (MC 재실행 불필요)
```
set KMP_DUPLICATE_LIB_OK=TRUE & set PYTHONIOENCODING=utf-8
python -m els_hedging_v1.deeponet --all-losses   # Stage-1 (MSE/MAE/MAPE)
python -m els_hedging_v1.margin_mlp               # Stage-2 MLP
python -m els_hedging_v1.metrics                   # %error·moneyness·calibration
python -m els_hedging_v1.speed_benchmark           # MC vs DeepONet (배치)
python -m els_hedging_v1.case_analysis · figures
```
> MC 전체는 `data/cache/product_mc.parquet` 재사용 (재계산 금지 — 오래 걸림).

## 6. git
- 원격: `github.com/SOOBEENKIM/PI-DeepONet_ELS`, 브랜치 **`els-hedging-v1`** (main=els_pricing v2는 별개, 안 건드림).
- 코드·지시서·그림·csv만 push. 데이터(raw·cache·parquet)는 gitignore(로컬 전용).

## 7. 다음 할 일 (후보)
- 발표(PPT) 정리 — 팀 덱과 정합(속도 배수 프레이밍, 필터 건수).
- (선택) 속도·필터를 팀 측정조건과 통일.
- (보류) 내재변동성(IV) 확보 → fair−MC 갭·저가오차 개선. 회사 데이터에 IV 없어 대기.
- (선택) issuer-조건 recent_margin으로 Stage-2 개선안 제안.
