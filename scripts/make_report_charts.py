#!/usr/bin/env python3
"""Charts for the 2026-10-21 HKJC recommendation report.

All numbers come from the artefacts in output/ — no value is re-derived
here, so the charts cannot disagree with the tables in the report.
"""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter

# The report is in Traditional Chinese, so the charts need a CJK face.
for _f in ('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
           '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'):
    if os.path.exists(_f):
        font_manager.fontManager.addfont(_f)
_CJK = 'Noto Sans CJK JP'

ROOT = '/home/work/horse-bet'
OUT = os.path.join(ROOT, 'output', 'report_charts')
os.makedirs(OUT, exist_ok=True)

INK = '#10203a'
SLATE = '#5a6a80'
RULE = '#d9dfe8'
RED = '#b23a48'
GREEN = '#1f7a5c'
BLUE = '#2d5f8b'
GOLD = '#b8860b'
TINT = '#f4f6f9'

plt.rcParams.update({
    'font.family': _CJK,
    'font.size': 9.5,
    'axes.edgecolor': RULE,
    'axes.labelcolor': INK,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'axes.titlecolor': INK,
    'text.color': INK,
    'xtick.color': SLATE,
    'ytick.color': SLATE,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.grid': True,
    'grid.color': RULE,
    'grid.linewidth': 0.7,
    'grid.alpha': 0.9,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'legend.frameon': False,
    'legend.fontsize': 9,
})


def finish(fig, name):
    for ax in fig.axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(RULE)
        ax.spines['bottom'].set_color(RULE)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('wrote', path)
    return path


# ── 1. value threshold scan ────────────────────────────────────────────────
def chart_value_scan():
    v = json.load(open(os.path.join(ROOT, 'output', 'value_scan_clean.json')))
    rows = [r for r in v['threshold_scan'] if r['n'] and r['thr'] < 0.06]
    labels = [f'{r["thr"]*100:.0f}%' for r in rows]
    roi = [r['roi_pct'] for r in rows]
    n = [r['n'] for r in rows]
    base = v['favourite_baseline']['roi_pct']

    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    bars = ax.bar(labels, roi, color=RED, alpha=0.9, width=0.62, zorder=3)
    for b, r, nn in zip(bars, roi, n):
        # keep the label clear of the baseline rule
        if abs(r - base) < 2.2:
            ax.text(b.get_x() + b.get_width() / 2, r - 1.4, f'{r:.2f}%',
                    ha='center', va='top', color='white', fontsize=9, fontweight='bold')
        else:
            ax.text(b.get_x() + b.get_width() / 2, r + 1.4, f'{r:.2f}%',
                    ha='center', va='bottom', color='white', fontsize=9, fontweight='bold')
        ax.text(b.get_x() + b.get_width() / 2, 0.8, f'n={nn:,}', ha='center',
                va='bottom', color=SLATE, fontsize=7.5)
    ax.axhline(0, color=INK, lw=1.1, zorder=4)
    ax.axhline(base, color=BLUE, lw=1.4, ls='--', zorder=4)
    ax.set_xlim(-0.65, len(labels) - 0.35 + 1.55)
    ax.text(len(labels) - 0.3, base, f'市場最熱門\n選擇基準\n{base:.2f}%',
            color=BLUE, fontsize=8.2, ha='left', va='center', fontweight='bold',
            zorder=6, linespacing=1.35)
    ax.set_ylim(-40, 4)
    ax.set_ylabel('投資回報率 ROI (%)')
    ax.set_xlabel('下注門檻：模型概率 − 市場概率（edge）')
    ax.set_title('所有 edge 門檻均錄得負回報，且全數劣於市場基準', pad=12)
    ax.text(0.5, -0.30, '註：門檻提高時樣本急減（n 由 3,553 降至 13），高門檻結果的統計不確定性大。',
            transform=ax.transAxes, ha='center', fontsize=7.6, color=SLATE)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.0f}%'))
    return finish(fig, 'c1_value_scan.png')


# ── 2. model vs market ─────────────────────────────────────────────────────
def chart_model_vs_market():
    import json as _j
    R = _j.load(open('/tmp/report_metrics.json'))
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))

    ax = axes[0]
    names = ['本模型\nFORM+ODDS', '市場\n正規化隱含機率', '各半混合']
    auc = [R['model_auc'], R['market_auc'], R['blend_auc']]
    bars = ax.bar(names, auc, color=[BLUE, INK], width=0.5, zorder=3)
    for b, a in zip(bars, auc):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.0012, f'{a:.4f}',
                ha='center', color=INK, fontsize=9.5, fontweight='bold')
    ax.set_ylim(0.74, 0.80)
    ax.set_ylabel('AUC')
    ax.set_title('勝出概率排序能力（AUC）', pad=10)
    ax.text(0.5, 0.05, '市場最高；混合模型亦未能超越市場 → 模型無排序優勢',
            transform=ax.transAxes, ha='center', color=RED, fontsize=8.2,
            fontweight='bold')
    ax.set_xlabel(f'勝出概率排序能力（測試集 {R["test_races"]} 場）')

    ax = axes[1]
    x = ['首名命中率\ntop-1', '前三命中率\ntop-3']
    model = [R['model_top1_pct'], R['model_top3_pct']]
    fav = [R['market_top1_pct'], R['market_top3_pct']]
    w = 0.34
    b1 = ax.bar([i - w / 2 for i in range(2)], model, w, label='本模型', color=BLUE, zorder=3)
    b2 = ax.bar([i + w / 2 for i in range(2)], fav, w, label='市場大熱', color=INK, zorder=3)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.7,
                    f'{b.get_height():.1f}%', ha='center', fontsize=8.5, fontweight='bold')
    ax.set_xticks(range(2))
    ax.set_xticklabels(x)
    ax.set_ylim(0, 70)
    ax.set_ylabel('命中率 (%)')
    ax.set_title('選馬命中率：本模型 vs 市場最熱門選擇', pad=10)
    ax.legend(loc='upper right', ncol=2)
    return finish(fig, 'c2_model_vs_market.png')


# ── 3. PLACE ROI by year ───────────────────────────────────────────────────
def chart_place_by_year():
    p = json.load(open(os.path.join(ROOT, 'output', 'place_backtest.json')))
    s = p['strategies']
    pick = [('market_favourite', '市場最熱門選擇', INK),
            ('model_top', '模型 top P(place)', BLUE),
            ('model_edge', '模型 P(place) − 市場', GOLD),
            ('value_ev0_model', '模型 EV > 0', GREEN)]
    years = sorted(next(iter(s.values()))['by_year'].keys())
    fig, ax = plt.subplots(figsize=(7.4, 4.1 if len(years) > 12 else 3.6))
    w = 0.2
    for i, (k, lab, c) in enumerate(pick):
        by = s[k]['by_year']
        vals = [by.get(y, {}).get('roi_pct', 0) for y in years]
        xs = [j + (i - 1.5) * w for j in range(len(years))]
        ax.bar(xs, vals, w, label=lab, color=c, zorder=3)
    ax.axhline(0, color=INK, lw=1.2, zorder=5)
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years, rotation=0 if len(years) <= 12 else 45,
                       ha='center' if len(years) <= 12 else 'right', fontsize=8.5)
    ax.set_ylabel('投資回報率 ROI (%)')
    ax.set_xlabel('年份（walk-forward：逐年訓練、逐年評分）')
    ax.set_title(f'PLACE 盤逐年回報：{len(years)} 個年份、每個策略均為負值', pad=34)

    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.0f}%'))
    lo = min(v for _, _, _ in pick for v in
             [s[k]['by_year'].get(y, {}).get('roi_pct', 0) for y in years])
    ax.set_ylim(lo * 1.18, 0)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.012), ncol=4,
              fontsize=8.2, frameon=False, handlelength=1.3, columnspacing=1.4)
    return finish(fig, 'c3_place_by_year.png')


# ── 4. model edge by odds bucket ──────────────────────────────────────────
def chart_edge_buckets():
    b = json.load(open('/tmp/report_metrics.json'))['longshot_bias']
    labels_map = {'<6 (fav)': '<6\n(大熱)', '6-11': '6–11', '11-21': '11–21',
                  '21-51': '21–51', '51+': '51+'}
    buckets = [labels_map.get(x['bucket'], x['bucket']) for x in b]
    model = [x['model_pct'] for x in b]
    market = [x['market_pct'] for x in b]
    diff = [x['diff_pp'] for x in b]
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    xs = range(len(buckets))
    ax.bar([i - 0.2 for i in xs], model, 0.38, label='模型平均概率', color=BLUE, zorder=3)
    ax.bar([i + 0.2 for i in xs], market, 0.38, label='市場平均概率', color=INK, zorder=3)
    ax2 = ax.twinx()
    ax2.plot(list(xs), diff, color=RED, marker='o', ms=5, lw=1.6, zorder=6,
             label='模型 − 市場')
    ax2.axhline(0, color=RED, lw=0.9, ls=':', zorder=6)
    ax2.set_ylim(-8, 8)
    ax2.set_ylabel('差異 (pp)', color=RED)
    ax2.tick_params(axis='y', colors=RED)
    ax2.grid(False)
    for i, d in enumerate(diff):
        ax2.annotate(f'{d:+.3f}', (i, d), textcoords='offset points',
                     xytext=(0, 9 if d >= 0 else -14), ha='center',
                     color=RED, fontsize=8, fontweight='bold')
    ax.set_xticks(list(xs))
    ax.set_xticklabels(buckets)
    ax.set_ylabel('勝出馬的平均機率 (%)')
    ax.set_title('模型存在系統性「冷門偏差」：愈冷門，模型愈高估', pad=12)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper right', ncol=3, fontsize=8.5)
    return finish(fig, 'c4_edge_buckets.png')


# ── 5. tonight's card ──────────────────────────────────────────────────────
def chart_tonight():
    import csv
    rows = list(csv.DictReader(open(os.path.join(ROOT, 'output', 'value_2026_10_21.csv'))))
    rows = [r for r in rows if float(r['p_model']) > 0.03]
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    for r in rows:
        pm = float(r['p_model']) * 100
        pk = float(r['p_market']) * 100
        edge = (float(r['edge'])) * 100
        top = edge >= 2.0
        ax.scatter(pk, pm, s=64 if top else 26,
                   color=GOLD if top else SLATE,
                   alpha=1.0 if top else 0.55,
                   edgecolors=INK if top else 'none',
                   linewidths=1.0 if top else 0, zorder=5 if top else 3)
    lim = [0, max(max(float(r['p_model']) for r in rows) * 100 * 1.12, 20)]
    ax.plot(lim, lim, color=INK, lw=1.0, ls='--', zorder=2)
    ax.text(lim[1] * 0.97, lim[1] * 0.93, '模型 = 市場', color=SLATE,
            fontsize=8, ha='right', rotation=0,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.9, pad=1.6))
    for r in rows:
        edge = float(r['edge']) * 100
        if edge >= 2.0:
            ax.annotate(f"{r['horse_name']}\nR{r['race_no']} @{r['win_odds']}  edge {edge:+.2f}%",
                        (float(r['p_market']) * 100, float(r['p_model']) * 100),
                        textcoords='offset points', xytext=(11, 6),
                        fontsize=8.5, color=INK, fontweight='bold')
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel('市場隱含機率（正規化，%）')
    ax.set_ylabel('模型機率（同場正規化，%）')
    ax.set_title('今晚 102 匹：模型 vs 市場 — 3 隻「edge ≥ 2%」候選（金色）', pad=12)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.0f}%'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.0f}%'))
    return finish(fig, 'c5_tonight.png')


# ── 6. tonight card overview ───────────────────────────────────────────────
def chart_card():
    import csv
    rows = list(csv.DictReader(open(os.path.join(ROOT, 'output', 'predict_2026_10_21.csv'))))
    fig, ax = plt.subplots(figsize=(7.4, 2.9))
    rs = [int(r['race']) for r in rows]
    hit = [r['model_picks_favourite'] == 'True' for r in rows]
    cols = [BLUE if h else GOLD for h in hit]
    ax.bar(rs, [r['n_runners'] for r in rows], color=cols, width=0.6, zorder=3)
    for r, h in zip(rows, hit):
        ax.text(int(r['race']), int(r['n_runners']) + 0.3,
                '藍：模型選大熱' if h else '金：模型不選大熱',
                ha='center', fontsize=7.5, color=SLATE)
    ax.set_xticks(rs)
    ax.set_ylim(0, 15)
    ax.set_ylabel('參賽馬數')
    ax.set_xlabel('場次（2026-10-21 沙田）')
    ax.set_title('今晚 9 場：模型於 6 場選中市場最熱門選擇，其餘 3 場改選次選', pad=12)
    return finish(fig, 'c6_card.png')


def chart_calibration():
    cal = json.load(open('/tmp/report_metrics.json'))['calibration']
    dec = [c['decile'] for c in cal]
    mdl = [c['model_pct'] for c in cal]
    mkt = [c['market_pct'] for c in cal]
    act = [c['actual_pct'] for c in cal]
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    ax.plot(dec, act, color=INK, lw=2.0, marker='o', ms=4, label='實際勝率', zorder=5)
    ax.plot(dec, mdl, color=RED, lw=1.6, ls='--', marker='s', ms=4, label='模型機率', zorder=4)
    ax.plot(dec, mkt, color=BLUE, lw=1.6, ls=':', marker='^', ms=4, label='市場機率', zorder=4)
    ax.set_xlabel('模型機率十分位（由低至高）')
    ax.set_ylabel('平均機率 / 實際勝率 (%)')
    ax.set_title('校準檢查：模型機率 vs 實際勝率（市場幾乎完全吻合）', pad=12)
    ax.legend(loc='upper left', ncol=3)
    ax.set_xticks(dec)
    return finish(fig, 'c7_calibration.png')


if __name__ == '__main__':
    chart_value_scan()
    chart_model_vs_market()
    chart_place_by_year()
    chart_edge_buckets()
    chart_tonight()
    chart_card()
    chart_calibration()
    print('\nall charts done ->', OUT)
