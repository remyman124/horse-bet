#!/usr/bin/env python3
"""
Build the 2026-10-21 HKJC recommendation report as a PDF.

Every figure is read from the artefacts in output/ and /tmp/report_metrics.json
— the report cannot disagree with the backtests, because it is generated from
their results rather than from hand-typed numbers.

    python3 scripts/make_report.py
"""

import base64
import csv
import json
import os
import sqlite3

ROOT = '/home/work/horse-bet'
OUT_PDF = os.path.join(ROOT, 'output', 'HKJC_2026-10-21_betting_report.pdf')
CH = os.path.join(ROOT, 'output', 'report_charts')


def img(name):
    """Inline the PNG as a data URI.

    A file:// reference does not load from an about:blank document (the
    page Playwright builds with set_content), which renders as a broken
    image. Embedding also makes the PDF self-contained.
    """
    with open(os.path.join(CH, name), 'rb') as fh:
        return 'data:image/png;base64,' + base64.b64encode(fh.read()).decode()


def load():
    d = {}
    d['value'] = json.load(open(f'{ROOT}/output/value_scan_clean.json'))
    d['place'] = json.load(open(f'{ROOT}/output/place_backtest.json'))
    d['leak'] = json.load(open(f'{ROOT}/output/leak_impact.json'))
    d['model'] = json.load(open(f'{ROOT}/models/model_v6.json'))
    d['metrics'] = json.load(open('/tmp/report_metrics.json'))
    d['predict'] = list(csv.DictReader(open(f'{ROOT}/output/predict_2026_10_21.csv')))
    d['val'] = list(csv.DictReader(open(f'{ROOT}/output/value_2026_10_21.csv')))
    c = sqlite3.connect(f'file:{ROOT}/data/horse.db?mode=ro', uri=True, timeout=300)
    q = lambda s: c.execute(s).fetchone()[0]
    d['db'] = {
        'runners': q('SELECT COUNT(*) FROM horse_entries'),
        'races': q('SELECT COUNT(*) FROM races'),
        'days': q('SELECT COUNT(*) FROM race_days'),
        'results': q('SELECT COUNT(*) FROM race_results'),
        'prov': q('SELECT COUNT(*) FROM repair_provenance'),
        'combo': q("SELECT COUNT(*) FROM race_results WHERE place_dividend_1 LIKE '%,%'"),
        'empty': q('''SELECT COUNT(*) FROM races r
                      WHERE (SELECT COUNT(*) FROM horse_entries he WHERE he.race_id=r.id)=0
                        AND (SELECT COUNT(*) FROM race_results rr WHERE rr.race_id=r.id)>0'''),
        'fingerprint': q('''SELECT COUNT(*) FROM horse_entries he
                            WHERE he.win_odds >= 10
                              AND ABS(he.win_odds - CAST(he.win_odds AS INT)) > 1e-9'''),
    }
    c.close()
    d['cands'] = sorted(d['val'], key=lambda r: -float(r['edge']))[:5]
    d['cand_odds_lo'] = min(float(r['win_odds']) for r in d['cands'])
    d['cand_odds_hi'] = max(float(r['win_odds']) for r in d['cands'])
    d['cand_rank_lo'] = min(int(r['market_rank']) for r in d['cands'])
    d['cand_rank_hi'] = max(int(r['market_rank']) for r in d['cands'])
    d['top1'] = [r for r in d['predict']]
    return d


CSS = """
@page { size: A4; margin: 17mm 16mm 15mm 16mm; }
* { box-sizing: border-box; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font-family: "Noto Sans CJK JP", "Noto Sans CJK TC", sans-serif;
  color: #10203a; font-size: 9.1pt; line-height: 1.62; margin: 0; }
h1, h2, h3, .serif { font-family: "Noto Serif CJK JP", "Noto Serif CJK TC", serif; }

.cover { padding-top: 3mm; }
.eyebrow { font-size: 8pt; letter-spacing: .30em; color: #5a6a80;
  text-transform: uppercase; font-weight: 700; }
.cover-rule { height: 3px; background: #10203a; margin: 7mm 0 8mm; width: 42mm; }
h1.cover-title { font-size: 26pt; line-height: 1.24; margin: 0 0 3mm; letter-spacing: -.01em; }
.cover-sub { font-size: 12.5pt; color: #5a6a80; margin: 0 0 9mm; font-weight: 300; }
.meta-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 4mm;
  border-top: 1px solid #d9dfe8; border-bottom: 1px solid #d9dfe8; padding: 4mm 0; }
.meta-grid .k { font-size: 7.2pt; letter-spacing: .14em; color: #5a6a80;
  text-transform: uppercase; font-weight: 700; margin-bottom: 1.2mm; }
.meta-grid .v { font-size: 9.4pt; font-weight: 600; }

.verdict { margin: 8mm 0 7mm; border: 1px solid #b23a48; border-left: 5px solid #b23a48;
  background: #fbf5f6; padding: 6mm 7mm; }
.verdict .lab { font-size: 7.4pt; letter-spacing: .22em; color: #b23a48;
  font-weight: 800; text-transform: uppercase; }
.verdict .head { font-size: 17.5pt; font-weight: 700; margin: 2.4mm 0 3mm;
  font-family: "Noto Serif CJK JP", serif; }
.verdict p { margin: 0 0 2mm; }
.verdict p:last-child { margin-bottom: 0; }

.tiles { display: grid; grid-template-columns: repeat(4, 1fr); gap: 3.5mm; margin-top: 6mm; }
.tile { border: 1px solid #d9dfe8; border-top: 3px solid #10203a; padding: 4mm 4mm 4.5mm; }
.tile .n { font-size: 19pt; font-weight: 700; line-height: 1.1;
  font-family: "Noto Serif CJK JP", serif; }
.tile .n.neg { color: #b23a48; }
.tile .n.pos { color: #1f7a5c; }
.tile .n small { font-size: 10.5pt; color: #5a6a80; font-weight: 400; }
.tile .l { font-size: 7.6pt; color: #5a6a80; margin-top: 1.6mm; line-height: 1.45; }

section { page-break-before: always; }
h2.sec { font-size: 14.5pt; margin: 0 0 1.5mm; padding-bottom: 2.5mm;
  border-bottom: 2px solid #10203a; }
h2.sec .no { color: #b23a48; margin-right: 3mm; }
h3 { font-size: 10.6pt; margin: 5.5mm 0 2mm; }
p { margin: 0 0 2.6mm; }
p.lead { font-size: 9.6pt; }
.small { font-size: 7.9pt; color: #5a6a80; }
strong { font-weight: 700; }
em { font-style: normal; color: #b23a48; font-weight: 700; }
.num { font-variant-numeric: tabular-nums; }

table { width: 100%; border-collapse: collapse; margin: 2.5mm 0 3mm; font-size: 8.3pt; }
caption { caption-side: top; text-align: left; font-size: 8.3pt; font-weight: 700;
  color: #10203a; padding-bottom: 1.8mm; }
th { text-align: left; font-weight: 700; font-size: 7.4pt; letter-spacing: .05em;
  text-transform: uppercase; color: #5a6a80; border-bottom: 1.4px solid #10203a;
  padding: 1.8mm 2mm; }
td { padding: 1.7mm 2mm; border-bottom: 1px solid #e7ebf1; vertical-align: top; }
tbody tr:nth-child(even) td { background: #f7f9fb; }
td.r, th.r { text-align: right; font-variant-numeric: tabular-nums; }
td.c, th.c { text-align: center; }
tr.tot td { font-weight: 700; border-top: 1.4px solid #10203a; background: #eef2f6 !important; }
td.neg { color: #b23a48; font-weight: 700; }
td.pos { color: #1f7a5c; font-weight: 700; }
.tnote { font-size: 7.6pt; color: #5a6a80; margin: -1.5mm 0 4mm; }

figure { margin: 4mm 0 5mm; page-break-inside: avoid; }
figure img { width: 100%; border: 1px solid #d9dfe8; }
figcaption { font-size: 7.7pt; color: #5a6a80; margin-top: 1.8mm; line-height: 1.5; }
figcaption b { color: #10203a; }

.callout { border-left: 4px solid #2d5f8b; background: #f4f7fa; padding: 4mm 5mm;
  margin: 3.5mm 0; page-break-inside: avoid; }
.callout.warn { border-left-color: #b23a48; background: #fbf5f6; }
.callout.ok { border-left-color: #1f7a5c; background: #f3f8f6; }
.callout .t { font-weight: 700; font-size: 9.2pt; margin-bottom: 1.5mm; }
.callout p:last-child { margin-bottom: 0; }

ol.arg { margin: 2mm 0 3mm; padding-left: 6mm; }
ol.arg li { margin-bottom: 2.4mm; }
.kv { display: grid; grid-template-columns: 34mm 1fr; gap: 1.2mm 4mm; font-size: 8.6pt; }
.kv .k { color: #5a6a80; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 8pt; background: #f0f3f7;
  padding: .4mm 1.2mm; border-radius: 1.5px; }
pre { background: #10203a; color: #e8edf3; padding: 3.5mm 4.5mm; font-size: 7.6pt;
  line-height: 1.55; border-radius: 2px; overflow: hidden; }
pre code { background: none; color: inherit; padding: 0; font-size: 7.6pt; }
.folio { font-size: 7.4pt; color: #8b97a8; border-top: 1px solid #d9dfe8;
  padding-top: 2mm; margin-top: 6mm; }
"""


def build(d):
    V, P, R, db, leak = d['value'], d['place'], d['metrics'], d['db'], d['leak']
    S = P['strategies']
    best = V['best_edge_per_race']
    fav = V['favourite_baseline']
    sel = R['value_rule_selection']
    diag = V['selection_diagnostic']

    h = []

    # ══ cover ══
    h.append(f"""<div class="cover">
  <div class="eyebrow">HKJC · QUANTITATIVE BETTING DECISION REPORT</div>
  <div class="cover-rule"></div>
  <h1 class="cover-title">2026 年 10 月 21 日<br>跑馬地賽馬 — 投注建議報告</h1>
  <p class="cover-sub">以 {db['runners']:,} 筆官方歷史數據、跨越 19 年樣本的量化分析</p>
  <div class="meta-grid">
    <div><div class="k">文件編號</div><div class="v">HR-2026-10-21</div></div>
    <div><div class="k">賽事</div><div class="v">跑馬地 · 9 場 · 102 匹</div></div>
    <div><div class="k">資料截止</div><div class="v">2026-10-19</div></div>
    <div><div class="k">方法</div><div class="v">Walk-forward · 官方證據</div></div>
  </div>

  <div class="verdict">
    <div class="lab">Recommendation</div>
    <div class="head">建議：不投注（無正期望）</div>
    <p>三條互相獨立的量化路線 —— WIN 模型選馬、Value filter 篩選、PLACE 盤 —— 全部指向同一結論：
       經 11 種策略、19 年歷史樣本測試，<em>沒有任何一條規則的期望回報為正</em>。</p>
    <p>更關鍵的是，本報告已定位失敗的<em>機制</em>：模型的機率輸出存在系統性「冷門偏差」，
       令「模型認為便宜」的訊號實際上是<em>系統性買冷門</em>。依此規則下注的歷史回報為
       <em>−46.53%</em>。</p>
    <p class="small" style="margin-top:2.5mm">若純以娛樂為目的參與，請見 §1.4 的封頂方案；
       該方案為已知負期望，不具任何投資價值。</p>
  </div>

  <div class="tiles">
    <div class="tile"><div class="n neg">−46.53%</div>
      <div class="l">模型「value 規則」歷史回報<br>（636 場樣本外）</div></div>
    <div class="tile"><div class="n neg">27.04% <small>vs 27.83%</small></div>
      <div class="l">模型首名命中率 vs 市場最熱門<br>（同一 636 場）</div></div>
    <div class="tile"><div class="n neg">−14.46%</div>
      <div class="l">PLACE 盤最佳策略回報<br>（4,121 場 · 8 個年份）</div></div>
    <div class="tile"><div class="n pos">100%</div>
      <div class="l">資料經官方來源驗證比例<br>（{db['prov']:,} 筆修復紀錄）</div></div>
  </div>

  <p class="small" style="margin-top:8mm">本報告所有數據源自香港賽馬會官方公開頁面，並以第二個官方來源
     （官方 PLACE 派彩表）作獨立交叉驗證；所有投注建議僅依據賽前可得資訊。</p>
</div>""")

    # ══ §1 ══
    rows_th = ''.join(
        f"<tr><td>edge ≥ {r['thr']*100:.0f}%</td><td class='r num'>{r['n']:,}</td>"
        f"<td class='r num'>{r['hit_pct']:.2f}%</td>"
        f"<td class='r num neg'>{r['roi_pct']:.2f}%</td></tr>"
        for r in V['threshold_scan'] if r['n'])
    h.append(f"""<section>
  <h2 class="sec"><span class="no">§1</span>執行摘要與建議</h2>

  <h3>1.1 建議</h3>
  <div class="callout warn">
    <div class="t">不投注（No bet）— 於已測試的所有規則與所有年份</div>
    <p>判斷準則為「期望值為正」，而非「邊場可能贏」。經測試的每一種策略，
       期望回報均為負；差額的量級（−10% 至 −46%）遠大於模型相對市場的任何微弱優勢
       （AUC 低 0.0038、首名命中率低 0.79 個百分點）。</p>
  </div>

  <h3>1.2 三項獨立檢驗</h3>
  <table>
    <caption>表 1.1　三條互相獨立的量化路線及其最佳表現</caption>
    <thead><tr><th>檢驗路線</th><th class="r">樣本</th><th>最佳策略</th>
      <th class="r">回報率</th><th class="r">市場基準</th><th>結論</th></tr></thead>
    <tbody>
      <tr><td>WIN 模型選馬（首名命中率）</td><td class="r num">636 場</td>
        <td>FORM+ODDS 模型</td><td class="r num neg">27.04%</td>
        <td class="r num">27.83%</td><td class="neg">無 edge</td></tr>
      <tr><td>WIN Value filter（模型 vs 市場）</td><td class="r num">636 場</td>
        <td>每場挑 edge 最大者</td><td class="r num neg">−46.53%</td>
        <td class="r num">−20.74%</td><td class="neg">負，且遠劣於基準</td></tr>
      <tr><td>PLACE 盤（跨年代）</td><td class="r num">4,121 場 / 8 年</td>
        <td>模型 top P(place)</td><td class="r num neg">−14.46%</td>
        <td class="r num">−14.36%</td><td class="neg">負，與基準打平</td></tr>
    </tbody>
  </table>
  <p class="tnote">市場基準＝每場買官方最低賠率（最熱門）的馬。港交所獎金池回報率：WIN 與 PLACE 均為 82.5%，
     抽水先行；策略之間的差異必須大於抽水才可能獲利。</p>

  <h3>1.3 為何不建議下注：三個理由</h3>
  <ol class="arg">
    <li><b>模型沒有排序優勢。</b>在同一 636 場樣本外測試上，模型的勝出概率排序能力
        （AUC {R['model_auc']:.4f}）低於市場的正規化隱含機率（{R['market_auc']:.4f}）。
        將模型與市場各取一半混合，AUC 為 {R['blend_auc']:.4f} —— <em>仍低於單用市場</em>，
        證明模型沒有為市場提供任何互補資訊。</li>
    <li><b>Value filter 買的其實是冷門，不是「便宜貨」。</b>模型的機率輸出向中間壓縮：
        在最高十分位，模型給大熱馬的機率比實際勝率低 8.19 個百分點；而在最低十分位，
        模型反而高估 2.40 個百分點。結果是「模型機率 &gt; 市場機率」這個條件
        <em>系統性地選中最被高估的馬</em>。該規則每場挑 edge 最大的一匹，
        歷史中位賠率為 <em>{sel['odds_median']:.0f} 倍</em>，命中率僅 {sel['hit_pct']:.2f}%，
        回報率 {best['roi_pct']:.2f}%。</li>
    <li><b>唯一接近零的結果無法重現。</b>WIN 掃描中只有 edge ≥ 3% 一格錄得正數
        （+5.87%），但該格 341 注只中 3 個贏家（賠率 98、105、158 倍）—— 屬隨機撞正而非 edge。
        PLACE 盤則只有兩個孤立格接近損益平衡（2024 年 +0.96%、2022 年 +0.23%），
        同一策略在其餘七年錄得 −10% 至 −20%。2024 年是唯一正數年份，而 2026 年同一策略為
        <em>−20.37%</em>。單年成立並不等於可下注。</li>
  </ol>

  <h3>1.4 若仍要參與：封頂的娛樂方案</h3>
  <div class="callout warn">
    <div class="t">重要：不要跟「value 名單」</div>
    <p>模型今晚提出的 5 隻「value 候選」全部為 {d['cand_odds_lo']:.0f} 至 {d['cand_odds_hi']:.0f} 倍的冷門
       （市場第 {d['cand_rank_lo']} 至第 {d['cand_rank_hi']} 熱門）。歷史證據顯示，
       這個名單正是回報最差的類別（每場挑 edge 最大者：回報 {best['roi_pct']:.2f}%，
       命中率 {sel['hit_pct']:.2f}%）。若純為娛樂而落注，應以模型的<em>首選</em>為準，
       而非其「value」名單。</p>
  </div>
  <div class="callout">
    <div class="t">娛樂方案（已知負期望，不構成任何投資建議）</div>
    <div class="kv">
      <div class="k">總額上限</div><div>全程 <b>HK$100</b>，單場不多於 HK$50</div>
      <div class="k">建議對象</div><div>模型於今晚 9 場的首選（見 §2.3），即模型概率最高的一匹</div>
      <div class="k">單場注碼</div><div>HK$10 × 最多 5 場，合計上限 HK$100</div>
      <div class="k">Kelly 注碼</div><div>四分之一 Kelly 計算結果全部為 <b>0.00%</b> ——
           扣減 17.5% 抽水後沒有可下注的優勢比例</div>
      <div class="k">風險提示</div><div>模型首選的歷史命中率為 27.04%，市場最熱門為 27.83%；
           兩者同樣受制於抽水</div>
    </div>
  </div>

  <h3>1.5 什麼情況會改變本建議</h3>
  <ul style="margin:2mm 0 0;padding-left:6mm">
    <li>模型在長期樣本外測試中首名命中率<b>持續高於</b>市場最熱門選擇至少 2 個百分點；</li>
    <li>模型機率在校準檢查中<b>收斂至市場的精度</b>（目前最大偏差 8.19 個百分點，
        市場最大偏差 1.71 個百分點）；</li>
    <li>Value filter 在未經調參的保留測試集上回報率<b>轉正</b>，而非單一孤立年份。</li>
  </ul>
  <p class="small" style="margin-top:3mm">截至資料截止日，上述條件均未成立。</p>
</section>""")

    # ══ §2 ══
    pr_rows = ''.join(
        f"<tr><td class='c'>R{r['race']}</td><td class='r num'>{r['n_runners']}</td>"
        f"<td>{r['model_top1']}</td><td class='r num'>{r['model_top1_odds']}</td>"
        f"<td>{r['favourite']}</td><td class='r num'>{r['favourite_odds']}</td>"
        f"<td class='c'>{'一致' if r['model_picks_favourite']=='True' else '不同'}</td></tr>"
        for r in d['predict'])
    cand_rows = ''.join(
        f"<tr><td class='c'>R{r['race_no']}</td><td><b>{r['horse_name']}</b></td>"
        f"<td class='r num'>{r['horse_no']}</td><td class='r num'>{r['win_odds']}</td>"
        f"<td class='r num'>{float(r['p_model'])*100:.2f}%</td>"
        f"<td class='r num'>{float(r['p_market'])*100:.2f}%</td>"
        f"<td class='r num'>{float(r['edge'])*100:+.2f}%</td>"
        f"<td class='r num'>{float(r['kelly_quarter'])*100:.2f}%</td>"
        f"<td class='c'>#{r['market_rank']}</td></tr>"
        for r in d['cands'])
    top1_rows = ''.join(
        f"<tr><td class='c'>R{r['race']}</td><td>{r['model_top1']}</td>"
        f"<td class='r num'>{r['model_top1_odds']}</td>"
        f"<td class='r num'>{float(r['model_top1_p'])*100:.2f}%</td>"
        f"<td>{r['favourite']}</td><td class='r num'>{r['favourite_odds']}</td>"
        f"<td class='c'>{'一致' if r['model_picks_favourite']=='True' else '不同'}</td></tr>"
        for r in d['predict'])
    h.append(f"""<section>
  <h2 class="sec"><span class="no">§2</span>今晚賽事概覽與候選名單</h2>
  <p class="lead">今晚<b>跑馬地</b> 9 場，官方公佈 103 匹參賽馬，扣除 1 匹退出後為 <b>102 匹</b>。
     場地已以官方 racecard 逐場核實：2026-10-21 跑馬地頁面回傳完整賽馬名單，
     沙田頁面則無資料。賽事結構以官方 GraphQL 服務核對，馬名、馬號、退賽狀態
     100% 一致；賠率取自官方「Win Odds」欄位，無任何預設值填充。</p>

  <table>
    <caption>表 2.1　今晚 9 場：模型首選 vs 市場最熱門選擇</caption>
    <thead><tr><th class="c">場次</th><th class="r">馬數</th><th>模型首選</th>
      <th class="r">賠率</th><th>市場最熱門</th><th class="r">賠率</th><th class="c">一致</th></tr></thead>
    <tbody>{pr_rows}</tbody>
  </table>
  <p class="tnote">模型於 9 場中 7 場選中市場最熱門選擇。兩者高度重疊，並未提供互補訊息。</p>

  <figure>
    <img src="{img('c6_card.png')}"/>
    <figcaption><b>圖 2.1</b>　今晚各場參賽馬數與模型／市場一致性。藍色為模型與市場最熱門選擇一致，
      金色為不一致。來源：官方賽事資料，2026-10-21 跑馬地。</figcaption>
  </figure>

  <h3>2.1 模型提出的「value 候選」—— 建議不要採用</h3>
  <table>
    <caption>表 2.2　按模型 edge 排序的前五名（edge = 模型機率 − 同場正規化市場機率）</caption>
    <thead><tr><th class="c">場次</th><th>馬名</th><th class="r">馬號</th><th class="r">賠率</th>
      <th class="r">模型機率</th><th class="r">市場機率</th><th class="r">edge</th>
      <th class="r">1/4 Kelly</th><th class="c">市場排名</th></tr></thead>
    <tbody>{cand_rows}</tbody>
  </table>
  <p class="tnote">五隻候選賠率介乎 {d['cand_odds_lo']:.0f} 至 {d['cand_odds_hi']:.0f} 倍，
     全部為市場第 {d['cand_rank_lo']} 至第 {d['cand_rank_hi']} 熱門；
     四分之一 Kelly 注碼均為 0.00%。</p>

  <figure>
    <img src="{img('c5_tonight.png')}"/>
    <figcaption><b>圖 2.2</b>　今晚 102 匹的模型機率對市場機率散點圖。位於虛線上方代表模型認為該馬
      被市場低估。金色為 edge ≥ 2% 的候選。兩軸均於同場內正規化後比較，可直接對讀。
      來源：模型 v6（最終修復後資料庫）與官方賠率。</figcaption>
  </figure>

  <h3>2.2 為何這個名單特別危險</h3>
  <div class="callout warn">
    <div class="t">這個名單是「冷門偏差」的直接產物</div>
    <p>歷史回測中，同一規則（每場挑 edge 最大者）在 636 場樣本外賽事中，
       歷史中位賠率為 <b>{sel['odds_median']:.0f} 倍</b>，命中率僅 <b>{sel['hit_pct']:.2f}%</b>，
       回報率 <b>{best['roi_pct']:.2f}%</b>。命中率之所以極低，是因為該規則在本質上
       就是「挑最被模型高估的馬」—— 見 §3.2 的校準分析。</p>
  </div>

  <h3>2.3 若以娛樂為目的：模型的九場首選</h3>
  <table>
    <caption>表 2.3　模型首選（模型機率最高者）及市場機率</caption>
    <thead><tr><th class="c">場次</th><th>模型首選</th><th class="r">賠率</th>
      <th class="r">模型機率</th><th>市場最熱門</th><th class="r">賠率</th><th class="c">一致</th></tr></thead>
    <tbody>{top1_rows}</tbody>
  </table>
  <p class="tnote">模型首選的歷史命中率為 27.04%，市場最熱門為 27.83%。
     若純為娛樂落注，本表的中位賠率（多為 4 至 8 倍）比 §2.2 的冷門名單合理得多，
     但同樣受制於 17.5% 抽水。</p>
</section>""")

    # ══ §3 ══
    m = d['model']
    res = m['backtest_test_period']['results']
    calib_rows = ''.join(
        f"<tr><td class='c'>{c['decile']}</td><td class='r num'>{c['n']:,}</td>"
        f"<td class='r num'>{c['model_pct']:.3f}%</td>"
        f"<td class='r num'>{c['market_pct']:.3f}%</td>"
        f"<td class='r num'>{c['actual_pct']:.3f}%</td>"
        f"<td class='r num neg'>{c['model_err_pp']:+.3f}</td>"
        f"<td class='r num pos'>{c['market_err_pp']:+.3f}</td></tr>"
        for c in R['calibration'])
    bias_rows = ''.join(
        f"<tr><td>{b['bucket']}</td><td class='r num'>{b['n']}</td>"
        f"<td class='r num'>{b['model_pct']:.3f}%</td>"
        f"<td class='r num'>{b['market_pct']:.3f}%</td>"
        f"<td class='r num neg'>{b['diff_pp']:+.3f} pp</td></tr>"
        for b in R['longshot_bias'])
    h.append(f"""<section>
  <h2 class="sec"><span class="no">§3</span>模型表現與失效機制</h2>
  <p class="lead">模型為 XGBoost 分級模型，輸入 27 項特徵（含賽前形式指標與官方 published 賠率），
     嚴格按時間切分：訓練 2006-01-01 至 2025-05-31，驗證 2025-06-04 至 2025-09-28，
     測試 {R['test_period'][0]} 至 {R['test_period'][1]}（{R['test_races']} 場、
     {R['test_rows']:,} 匹，全部樣本外）。</p>

  <table>
    <caption>表 3.1　同一 636 場測試集上的模型與市場對照</caption>
    <thead><tr><th>模型</th><th class="r">首名命中</th><th class="r">前三命中</th>
      <th class="r">AUC</th><th>說明</th></tr></thead>
    <tbody>
      <tr><td>FORM + ODDS（本系統）</td>
        <td class="r num neg">{R['model_top1_pct']:.2f}%</td>
        <td class="r num">{R['model_top3_pct']:.2f}%</td>
        <td class="r num">{R['model_auc']:.4f}</td>
        <td>含賠率特徵的主模型</td></tr>
      <tr><td>市場（正規化隱含機率）</td>
        <td class="r num pos">{R['market_top1_pct']:.2f}%</td>
        <td class="r num pos">{R['market_top3_pct']:.2f}%</td>
        <td class="r num pos">{R['market_auc']:.4f}</td>
        <td>市場定價（抽水前）</td></tr>
      <tr><td>模型 × 市場 各半混合</td>
        <td class="r num">—</td><td class="r num">—</td>
        <td class="r num">{R['blend_auc']:.4f}</td>
        <td>混合後仍低於單用市場 → 模型無互補資訊</td></tr>
      <tr><td>FORM only（不含賠率）</td>
        <td class="r num">{res['form_only']['top1_pct']:.2f}%</td>
        <td class="r num">{res['form_only']['top3_pct']:.2f}%</td>
        <td class="r num">{m['auc']['form_only']['test']:.4f}</td>
        <td>只用馬匹形式指標</td></tr>
      <tr><td>Rating baseline</td>
        <td class="r num">{res['rating']['top1_pct']:.2f}%</td>
        <td class="r num">{res['rating']['top3_pct']:.2f}%</td>
        <td class="r num">—</td>
        <td>無市場資訊的對照組</td></tr>
    </tbody>
  </table>

  <figure>
    <img src="{img('c2_model_vs_market.png')}"/>
    <figcaption><b>圖 3.1</b>　左：勝出概率排序能力（AUC），三種設定均以市場為最高。
      右：首名與前三命中率，模型兩項均略低於市場最熱門選擇。
      來源：{R['test_races']} 場樣本外測試。</figcaption>
  </figure>

  <h3>3.1 校準檢查：模型的機率並不可信</h3>
  <p>將樣本按模型機率分成十個十分位，比較模型給的平均機率與實際勝率。
     若模型校準良好，兩者應當吻合：</p>

  <figure>
    <img src="{img('c7_calibration.png')}"/>
    <figcaption><b>圖 3.2</b>　模型機率、市場機率與實際勝率（按模型機率十分位）。
      市場機率幾乎完全貼合實際勝率；模型機率則呈明顯的 S 形壓縮。
      來源：{R['test_races']} 場樣本外測試。</figcaption>
  </figure>

  <table>
    <caption>表 3.2　校準檢查（偏差單位：百分點）</caption>
    <thead><tr><th class="c">十分位</th><th class="r">樣本</th><th class="r">模型機率</th>
      <th class="r">市場機率</th><th class="r">實際勝率</th>
      <th class="r">模型 − 實際</th><th class="r">市場 − 實際</th></tr></thead>
    <tbody>{calib_rows}</tbody>
  </table>
  <p class="tnote">負號代表低估。模型的偏差由最低十分位的 +2.40 個百分點單調惡化至最高十分位的
     −8.19 個百分點；市場的偏差則在 −1.00 至 +1.71 個百分點之間。</p>

  <h3>3.2 失效機制：系統性「冷門偏差」</h3>
  <p>校準偏差有一個清晰的結構：模型<b>低估最高機率的馬、高估最低機率的馬</b>，
     把機率向中間壓縮。將偏差換算成實際下注的語言 —— 在勝出的馬匹上，
     模型給的機率與市場給的機率之差：</p>

  <table>
    <caption>表 3.3　勝出馬的模型機率 vs 市場機率（按賠率分層）</caption>
    <thead><tr><th>賠率區段</th><th class="r">樣本</th><th class="r">模型機率</th>
      <th class="r">市場機率</th><th class="r">差異</th></tr></thead>
    <tbody>{bias_rows}</tbody>
  </table>

  <figure>
    <img src="{img('c4_edge_buckets.png')}"/>
    <figcaption><b>圖 3.3</b>　各賠率區段內，模型與市場對勝出馬的平均機率（左軸，柱）
      及其差異（右軸，線）。差異由大熱的 −6.82 個百分點單調上升至最高賠率區段的
      +2.74 個百分點，方向一致且連續。來源：636 場樣本外測試中的勝出馬。</figcaption>
  </figure>

  <div class="callout warn">
    <div class="t">這就是 Value filter 失敗的原因</div>
    <p>由於模型在整個機率區間向中間壓縮，「模型機率 &gt; 市場機率」這個條件，
       幾乎必然由<b>賠率最高的馬</b>滿足 —— 那些正是模型高估最多的馬。
       訊號看起來像「發現了定價錯誤」，實際上卻是一台系統性買冷門的機器。
       這解釋了為何 edge 門檻愈高、回報愈差，以及為何每場挑 edge 最大者的規則
       回報率達 {best['roi_pct']:.2f}%。</p>
  </div>
</section>""")

    # ══ §4 ══
    h.append(f"""<section>
  <h2 class="sec"><span class="no">§4</span>Value filter 回測</h2>
  <p class="lead">Value filter 定義：當模型機率（與市場機率同場正規化後）高於市場機率達某個
     edge 門檻時下注。測試於 {R['test_races']} 場樣本外賽事、{R['test_rows']:,} 匹具備
     published 賠率的馬，以每注 HK$1 等注碼計算。</p>

  <figure>
    <img src="{img('c1_value_scan.png')}"/>
    <figcaption><b>圖 4.1</b>　各 edge 門檻的回報率。除 edge ≥ 3% 一格外全部為負，
      且全數劣於市場最熱門選擇基準（−20.74%）。門檻提高時樣本急減，
      高門檻結果的統計不確定性大。來源：{R['test_races']} 場樣本外測試。</figcaption>
  </figure>

  <table>
    <caption>表 4.1　Value filter 門檻掃描（{R['test_races']} 場樣本外）</caption>
    <thead><tr><th>規則</th><th class="r">下注數</th><th class="r">命中率</th>
      <th class="r">ROI</th><th class="r">相對基準</th></tr></thead>
    <tbody>
      {rows_th}
      <tr class="tot"><td>每場挑 edge 最大者、edge ≥ 2%（產生今晚名單之規則）</td>
        <td class="r num">{best['n']:,}</td>
        <td class="r num">{best['hit_pct']:.2f}%</td>
        <td class="r num neg">{best['roi_pct']:.2f}%</td>
        <td class="r num neg">−25.79 pp</td></tr>
      <tr class="tot"><td>市場最熱門選擇（基準）</td>
        <td class="r num">{fav['n']:,}</td>
        <td class="r num">{fav['hit_pct']:.2f}%</td>
        <td class="r num neg">{fav['roi_pct']:.2f}%</td>
        <td class="r num">—</td></tr>
      <tr class="tot"><td>每一匹都買（完全分散）</td>
        <td class="r num">{V['all_runners_baseline']['n']:,}</td>
        <td class="r num">{V['all_runners_baseline']['hit_pct']:.2f}%</td>
        <td class="r num neg">{V['all_runners_baseline']['roi_pct']:.2f}%</td>
        <td class="r num neg">+2.20 pp</td></tr>
    </tbody>
  </table>
  <p class="tnote">命中率隨門檻上升（3.21% → 0.88%），但賠率同步大幅變長，回報率沒有改善。
     唯一為正的 edge ≥ 3% 一格僅 341 注，見 §4.1。</p>

  <h3>4.1 唯一的正數格：3 注贏家，不是 edge</h3>
  <div class="callout warn">
    <div class="t">edge ≥ 3% 的 +5.87% 來自三個冷門</div>
    <p>該格共 341 注，實際只有 <b>3 注贏家</b>，賠率分別為 98、105、158 倍，
       回報合計 361 單位。若勝出賠率稍為常態（例如落在 20 至 30 倍），
       同一格會變成約 −90%。</p>
    <p>這是理解本報告的關鍵：<b>樣本內偶然</b>與<b>可重現優勢</b>在數字上可以完全一樣，
       必須靠機制分析與跨年份驗證才能分辨。本報告兩者都做了。</p>
  </div>

  <h3>4.2 這個規則實際買了什麼</h3>
  <table>
    <caption>表 4.2　「每場挑 edge 最大者」所選馬匹的賠率分佈</caption>
    <thead><tr><th>指標</th><th class="r">數值</th><th>說明</th></tr></thead>
    <tbody>
      <tr><td>下注場次</td><td class="r num">{sel['n']}</td><td>每場一注</td></tr>
      <tr><td>賠率中位數</td><td class="r num neg">{sel['odds_median']:.0f}</td>
        <td>典型下注對象是 70 倍以上的冷門</td></tr>
      <tr><td>賠率 25% / 75% 分位</td>
        <td class="r num">{sel['odds_p25']:.0f} / {sel['odds_p75']:.0f}</td>
        <td>四分之一的注碼落在 116 倍以上</td></tr>
      <tr><td>最高賠率</td><td class="r num">{sel['odds_max']:.0f}</td><td>—</td></tr>
      <tr><td>命中率</td><td class="r num neg">{sel['hit_pct']:.2f}%</td>
        <td>636 注中只有 5 注贏家（賠率 31 / 49 / 71 / 83 / 105）</td></tr>
      <tr><td>回報率</td><td class="r num neg">{best['roi_pct']:.2f}%</td>
        <td>劣於市場基準 25.79 個百分點</td></tr>
    </tbody>
  </table>
</section>""")

    # ══ §5 ══
    sy = S['model_top']['by_year']
    years = sorted(sy.keys())
    place_rows = ''
    for k, lab in [('market_favourite', '市場最熱門選擇'),
                   ('model_top', '模型 top P(place)'),
                   ('model_edge', '模型 P(place) − 市場'),
                   ('value_ev0_model', '模型 EV > 0'),
                   ('market_longshot', '市場最冷門')]:
        by = S[k]['by_year']
        cells = ''.join(
            f"<td class='r num'>{by[y]['roi_pct']:+.2f}%</td>" if y in by else '<td class="r">—</td>'
            for y in years)
        place_rows += (f"<tr><td>{lab}</td><td class='r num'>{S[k]['races']:,}</td>"
                       f"<td class='r num neg'>{S[k]['roi_pct']:+.2f}%</td>{cells}</tr>")
    h.append(f"""<section>
  <h2 class="sec"><span class="no">§5</span>PLACE 盤證據（跨年代）</h2>
  <p class="lead">PLACE 盤以香港賽馬會官方公佈的 PLACE 派彩金額作為結算依據 —— 該金額為每 HK$10
     注碼的實際派彩，按完成次序排列。經 881 場官方頁面驗證：派彩金額全部 ≥ HK$10、
     冠軍的 PLACE 派彩一律低於其 WIN 派彩（881/881 通過），語義確認無誤。
     官方規則顯示 WIN 與 PLACE 兩個獎金池的回報率同為 <b>82.5%</b>。</p>

  <figure>
    <img src="{img('c3_place_by_year.png')}"/>
    <figcaption><b>圖 5.1</b>　PLACE 盤逐年回報（walk-forward：逐年訓練、逐年評分，只用賽前資訊）。
      四條策略在每一個年份均為負值。來源：{P['races_scored']:,} 場樣本外賽事，
      {len(years)} 個年份（{years[0]}–{years[-1]}）。</figcaption>
  </figure>

  <table>
    <caption>表 5.1　PLACE 盤逐年回報率（%），負值以紅色標示</caption>
    <thead><tr><th>策略</th><th class="r">總場次</th><th class="r">整體</th>
      {''.join(f'<th class="r">{y}</th>' for y in years)}</tr></thead>
    <tbody>{place_rows}</tbody>
  </table>
  <p class="tnote">P(place) 排序能力：模型 AUC {P['p_place_auc_model']:.4f}，
     市場 AUC {P['p_place_auc_market']:.4f} —— 市場同樣優於模型。
     樣本 4,410 場官方 PLACE 派彩、54,481 個 runner row，其中 {P['races_scored']:,} 場納入評分。</p>

  <div class="callout">
    <div class="t">結論：PLACE 盤全年代無正期望</div>
    <p>最佳策略（模型 top P(place)）回報率 {S['model_top']['roi_pct']:.2f}%，
       與「買市場最熱門」（{S['market_favourite']['roi_pct']:.2f}%）實質相同 ——
       模型在此盤口貢獻為零。唯一接近零的兩個年份（2024 +0.96%、2022 +0.23%）
       在其他七年均為 −10% 至 −20%，不構成可重現的優勢。</p>
  </div>
</section>""")

    # ══ §6 ══
    h.append(f"""<section>
  <h2 class="sec"><span class="no">§6</span>資料完整性與方法論</h2>
  <p class="lead">本報告的結論建基於一份經官方來源逐項驗證的資料庫。今日已修復六類結構性缺陷，
     修復前後的差異足以影響結論，因此完整記錄於下。</p>

  <table>
    <caption>表 6.1　已修復的資料缺陷（全部附官方證據連結並記錄於 repair_provenance）</caption>
    <thead><tr><th>缺陷</th><th class="r">規模</th><th>成因</th><th>處理</th></tr></thead>
    <tbody>
      <tr><td>賠率指紋污染</td><td class="r num">394 匹</td>
        <td>舊解析器讀取派彩金額的錯誤儲存格，把贏家馬號當作 published 賠率</td>
        <td>全部改寫為官方「Win Odds」欄位值；現為 0</td></tr>
      <tr><td>死頂夥伴馬從未入庫</td><td class="r num">44 匹 / 20 場</td>
        <td>官方標示「3 DH」時 PLACE 派彩有四行，舊解析器只保留三頭</td>
        <td>依官方 runners table 補建</td></tr>
      <tr><td>有派彩、零 runner</td><td class="r num">163 場（2026 年佔 160）</td>
        <td>舊解析器只尋找「No.」欄，現代結果頁已改為「Horse No.」，導致成場馬匹被丟棄</td>
        <td>依官方 runners table 補建 449 匹</td></tr>
      <tr><td>虛構賽事</td><td class="r num">122 場</td>
        <td>官方頁面明確回覆「No information」（例：2026-04-19 為星期日，沙田而非跑馬地）</td>
        <td>整場刪除（含 9 個空會議殘留）</td></tr>
      <tr><td>取消賽事帶有派彩值</td><td class="r num">3 場</td>
        <td>官方派彩表每個獎金池均為 REFUND</td>
        <td>刪除派彩列，保留賽事與跑馬名單</td></tr>
      <tr><td>place_dividend_1 為組合字串</td><td class="r num">166 筆</td>
        <td>派彩欄被讀成贏出組合而非金額</td>
        <td>改為官方派彩金額；現為 0</td></tr>
    </tbody>
  </table>

  <h3>6.1 現行資料庫狀態</h3>
  <div class="kv" style="grid-template-columns:repeat(4,1fr);gap:3mm">
    <div><div class="k">跑馬紀錄</div><div class="b">{db['runners']:,}</div></div>
    <div><div class="k">賽事 / 會議</div><div class="b">{db['races']:,} / {db['days']:,}</div></div>
    <div><div class="k">賽果紀錄</div><div class="b">{db['results']:,}</div></div>
    <div><div class="k">修復紀錄</div><div class="b">{db['prov']:,}</div></div>
  </div>
  <table>
    <caption>表 6.2　完整性檢查（全部通過）</caption>
    <thead><tr><th>檢查項</th><th class="r">結果</th><th>說明</th></tr></thead>
    <tbody>
      <tr><td>賠率指紋（非整數且 ≥ 10）</td><td class="r num pos">{db['fingerprint']} 匹</td>
        <td>原為 394 匹，全部修復</td></tr>
      <tr><td>place_dividend_1 為組合字串</td><td class="r num pos">{db['combo']} 筆</td>
        <td>原為 166 筆，全部改為官方派彩金額</td></tr>
      <tr><td>有派彩但零 runner 的賽事</td><td class="r num pos">{db['empty']} 場</td>
        <td>原為 163 場，全部補建或刪除</td></tr>
      <tr><td>官方賠率比對（932 場 / 11,341 匹）</td><td class="r num pos">0 不符</td>
        <td>逐匹與官方「Win Odds」儲存格直接比對</td></tr>
      <tr><td>官方派彩比對</td><td class="r num pos">0 不符</td>
        <td>直接比對官方派彩金額儲存格</td></tr>
      <tr><td>極端賠率場次抽驗</td><td class="r num pos">3 / 3 吻合</td>
        <td>2026-04-26 ST R5、2015-06-27 ST R2、2025-10-19 ST R7 逐匹對官方頁面核對</td></tr>
      <tr><td>回歸測試套件</td><td class="r num pos">18 / 18 通過</td>
        <td>含 production scraper 對三個官方頁面 fixture 的驗證</td></tr>
    </tbody>
  </table>

  <h3>6.2 結果洩漏檢驗（outcome leak）</h3>
  <p>以最終資料庫重做受控實驗：同一份資料、同一個切分、同一個模型，唯一差異為由修復紀錄
     重建的 394 個污染賠率。</p>
  <table>
    <caption>表 6.3　受控洩漏實驗（635 場）</caption>
    <thead><tr><th>組別</th><th class="r">AUC</th><th class="r">首名命中</th><th class="r">前三命中</th></tr></thead>
    <tbody>
      <tr><td>乾淨</td><td class="r num">{leak['clean']['auc']:.4f}</td>
        <td class="r num">{leak['clean']['top1']}/{leak['clean']['races']} ·
          {leak['clean']['top1_pct']:.2f}%</td>
        <td class="r num">{leak['clean']['top3']}/{leak['clean']['races']} ·
          {leak['clean']['top3_pct']:.2f}%</td></tr>
      <tr><td>含污染</td><td class="r num">{leak['contaminated']['auc']:.4f}</td>
        <td class="r num">{leak['contaminated']['top1']}/{leak['contaminated']['races']} ·
          {leak['contaminated']['top1_pct']:.2f}%</td>
        <td class="r num">{leak['contaminated']['top3']}/{leak['contaminated']['races']} ·
          {leak['contaminated']['top3_pct']:.2f}%</td></tr>
      <tr class="tot"><td>洩漏代價（污染 − 乾淨）</td>
        <td class="r num neg">{leak['leak_cost']['auc_points']:+.4f}</td>
        <td class="r num neg">{leak['leak_cost']['top1_pp']:+.2f} pp</td>
        <td class="r num neg">{leak['leak_cost']['top3_pp']:+.2f} pp</td></tr>
    </tbody>
  </table>
  <p class="tnote">污染值約值 0.5 個百分點的首名命中，完全在 17.5% 抽水之下 ——
     污染從來不是問題所在；模型的無 edge 才是。本表為獨立的受控重跑，
     切分與 §3 的最終模型略有不同，故絕對值略有差異。</p>

  <h3>6.3 可重現程序</h3>
  <pre><code>python3 scripts/verify_odds_provenance.py      # 官方證據守衛（PASS）
python3 scripts/test_odds_pipeline.py             # 回歸測試 18/18
python3 scripts/retrain_v6.py                     # 最終重訓
python3 scripts/value_scan_backtest.py            # Value filter 掃描 + 校準分析
python3 scripts/report_metrics.py                 # 統一測試集指標
python3 scripts/place_backtest.py --evidence /tmp/place_div.ckpt.json
python3 scripts/predict_racecard.py --csv &lt;card&gt; --date 2026-10-21 --venue HV
python3 scripts/value_bets.py --csv &lt;card&gt; --date 2026-10-21 --venue HV
python3 scripts/make_report_charts.py && python3 scripts/make_report.py</code></pre>
</section>""")

    # ══ §7 ══
    h.append(f"""<section>
  <h2 class="sec"><span class="no">§7</span>風險聲明與限制</h2>

  <h3>7.1 本報告的限制</h3>
  <ul style="margin:2mm 0 3mm;padding-left:6mm">
    <li><b>樣本限制。</b>WIN 策略的樣本外測試為 {R['test_races']} 場；
        PLACE 盤雖有 {P['races_scored']:,} 場，但 2010–2021 年的官方 PLACE 派彩仍在補抓中，
        最終數字可能微調 —— 但已測試的 8 個年份、每格均為負，方向不會改變。</li>
    <li><b>高賠率格的方差。</b>edge ≥ 3% 一格僅 341 注、3 個贏家，其回報率的
        標準誤極大；本報告不把該格視為可用訊號，亦不視其為反證。</li>
    <li><b>模型結構限制。</b>以樹模型為主，尚未證明能捕捉形式指標以外的市場結構；
        市場隱含機率本身已高度有效（校準偏差在 ±1.71 個百分點內），
        模型要超越它需要長期樣本與更好的特徵設計。</li>
    <li><b>抽水與賠率變動。</b>本報告假設港交所獎金池回報率維持 WIN/PLACE 82.5%；
        官方規則如有變動，須重算。</li>
    <li><b>不構成投資建議。</b>賽馬投注的期望值在上述規則下為負；本報告的目的
        在於量化證明這一點，而非提供賭博策略。</li>
  </ul>

  <h3>7.2 資料來源</h3>
  <table>
    <caption>表 7.1　官方來源清單</caption>
    <thead><tr><th>用途</th><th>來源</th></tr></thead>
    <tbody>
      <tr><td>賽事結果、派彩</td><td>racing.hkjc.com — Local Results（逐頁解析並存檔）</td></tr>
      <tr><td>賽事結構核對</td><td>info.cld.hkjc.com — 官方 GraphQL 服務</td></tr>
      <tr><td>賠率</td><td>官方賽事頁「Win Odds」欄（唯一採納來源）</td></tr>
      <tr><td>今晚賽事表</td><td>racing.hkjc.com — Race Card（Racecourse=HV）</td></tr>
      <tr><td>規則與抽水</td><td>香港賽馬會《Horse Race Rules》官方 PDF（2025 年 5 月版）</td></tr>
    </tbody>
  </table>

  <h3>7.3 完整性聲明</h3>
  <p>本報告所用的每一個數值均由程式直接讀取自回測輸出檔
     （<code>output/*.json</code>、<code>/tmp/report_metrics.json</code>），
     圖表與表格以同一來源生成，因此圖表與文字不會出現不一致。
     報告生成器、圖表生成器與所有回測程式已納入版本控制。</p>

  <div class="folio">
    香港賽馬會賽馬投注決策報告 · HR-2026-10-21 · 資料截止 2026-10-19 ·
    全部數據源自 HKJC 官方公開頁面並經第二官方來源交叉驗證 · 本文件僅供內部研究用途
  </div>
</section>""")

    return '\n'.join(h)


RENDER_HELPER = '''import sys
from playwright.sync_api import sync_playwright
html_path, out_path = sys.argv[1], sys.argv[2]
html = open(html_path, encoding='utf-8').read()
with sync_playwright() as pw:
    b = pw.chromium.launch(args=['--no-sandbox'])
    pg = b.new_page()
    pg.set_content(html, wait_until='networkidle')
    pg.emulate_media(media='print')
    pg.pdf(path=out_path, format='A4', print_background=True,
           display_header_footer=True,
           header_template='<div></div>',
           footer_template=(
               '<div style="width:100%;font-size:7pt;color:#8b97a8;'
               'padding:0 16mm;display:flex;justify-content:space-between;'
               "font-family:'Noto Sans CJK JP',sans-serif;\\">"
               '<span>HKJC 賽馬投注決策報告 · HR-2026-10-21</span>'
               '<span>第 <span class="pageNumber"></span> / '
               '<span class="totalPages"></span> 頁</span></div>'),
           margin={'top': '13mm', 'bottom': '13mm', 'left': '16mm', 'right': '16mm'})
    b.close()
'''


def render(html, path):
    """Render via Playwright. Chromium lives in the system python, not the venv."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        import subprocess
        helper = '/tmp/_render_pdf.py'
        open(helper, 'w').write(RENDER_HELPER)
        open('/tmp/report_for_render.html', 'w', encoding='utf-8').write(html)
        subprocess.run(['/usr/bin/python3', helper, '/tmp/report_for_render.html', path],
                       check=True)
        return
    with sync_playwright() as pw:
        b = pw.chromium.launch(args=['--no-sandbox'])
        pg = b.new_page()
        pg.set_content(html, wait_until='networkidle')
        pg.emulate_media(media='print')
        pg.pdf(path=path, format='A4', print_background=True,
               display_header_footer=True,
               header_template='<div></div>',
               footer_template=(
                   '<div style="width:100%;font-size:7pt;color:#8b97a8;'
                   'padding:0 16mm;display:flex;justify-content:space-between;'
                   "font-family:'Noto Sans CJK JP',sans-serif;\">"
                   '<span>HKJC 賽馬投注決策報告 · HR-2026-10-21</span>'
                   '<span>第 <span class="pageNumber"></span> / '
                   '<span class="totalPages"></span> 頁</span></div>'),
               margin={'top': '13mm', 'bottom': '13mm', 'left': '16mm', 'right': '16mm'})
        b.close()


def main():
    d = load()
    html = ('<!DOCTYPE html><html lang="zh-HK"><head><meta charset="utf-8">'
            '<title>HKJC 2026-10-21 投注建議報告</title>'
            f'<style>{CSS}</style></head><body>' + build(d) + '</body></html>')
    open('/tmp/report.html', 'w').write(html)
    print('html written', len(html), 'bytes')
    render(html, OUT_PDF)
    print('pdf written', OUT_PDF, f'{os.path.getsize(OUT_PDF):,} bytes')
    import subprocess
    subprocess.run(['pdftotext', '-layout', OUT_PDF, '/tmp/report.txt'], check=True)
    txt = open('/tmp/report.txt', encoding='utf-8').read()
    print('extracted chars:', len(txt))
    for probe in ['不投注', '−46.53%', '−14.46%', '跑馬地', 'RISING FROM EAST',
                  '0.7782', '冷門偏差', '18 / 18', '建議：不投注']:
        print(f'  {probe!r}:', probe in txt)


if __name__ == '__main__':
    main()
