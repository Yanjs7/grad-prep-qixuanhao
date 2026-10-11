#!/usr/bin/env python3
"""Analyze only the fully completed frozen study; never tune from test predictions."""
import argparse, datetime, hashlib, json, sys
from pathlib import Path

ROOT = Path('/root/CTA')
sys.path.insert(0, str(ROOT / 'code/cta-ssl/src'))
sys.path.insert(0, str(ROOT / 'code/cta-ssl/tools'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import binomtest
from sklearn.metrics import roc_curve, precision_recall_curve
from cta_ssl.common import config, digest, save_json
from generate_experiment_report import load_complete, measures, exact_binomial_interval

PHASES = ['s0', 's0plus', 's1']
LABELS = {'s0': 'S0', 's0plus': 'S0+', 's1': 'S1'}
COLORS = {'s0': '#31688e', 's0plus': '#35b779', 's1': '#d95f02'}
REPETITIONS = 10000

def interval(values):
    return np.quantile(values, [.025, .975]).astype(float).tolist()

def primary_draws(rows, positive_draws, negative_draws):
    positive = rows[rows.label == 1].score.to_numpy()
    negative = rows[rows.label == 0].score.to_numpy()
    result = []
    for start in range(0, len(positive_draws), 500):
        a = positive[positive_draws[start:start + 500]][:, :, None]
        b = negative[negative_draws[start:start + 500]][:, None, :]
        result.append(((a > b) + .5 * (a == b)).mean(axis=(1, 2)))
    return np.concatenate(result)

def paired_primary(primary):
    ordered = {p: primary[primary.phase == p].sort_values('patient_id').reset_index(drop=True) for p in PHASES}
    assert all(ordered[p].patient_id.equals(ordered['s0'].patient_id) and ordered[p].label.equals(ordered['s0'].label) for p in PHASES)
    base = ordered['s0']; rng = np.random.default_rng(20261010)
    npos, nneg = int((base.label == 1).sum()), int((base.label == 0).sum())
    assert (npos, nneg) == (10, 19)
    ip = rng.integers(npos, size=(REPETITIONS, npos)); inn = rng.integers(nneg, size=(REPETITIONS, nneg))
    draws = {p: primary_draws(ordered[p], ip, inn) for p in PHASES}
    comparisons = []
    for a, b in [('s1', 's0plus'), ('s0plus', 's0'), ('s1', 's0')]:
        aa, bb = ordered[a], ordered[b]
        ca = aa.prediction.to_numpy() == aa.label.to_numpy(); cb = bb.prediction.to_numpy() == bb.label.to_numpy()
        a_only, b_only = int((ca & ~cb).sum()), int((cb & ~ca).sum())
        comparisons.append(dict(comparison=f'{LABELS[a]} minus {LABELS[b]}', primary=a == 's1' and b == 's0plus',
            metric='AortaSeg OOF AUROC', estimate=measures(aa)['auroc'] - measures(bb)['auroc'],
            ci95=interval(draws[a] - draws[b]), discordant_a_correct_b_wrong=a_only, discordant_b_correct_a_wrong=b_only,
            mcnemar_exact_p=float(binomtest(a_only, a_only + b_only, .5).pvalue) if a_only + b_only else 1.,
            method='Paired class-stratified scan bootstrap, 10000 draws; fixed OOF predictions and validation thresholds. McNemar tests classification correctness, not AUROC.',
            limitation='Conditional exploratory intervals omit retraining, fold-model dependence and unknown patient/pretraining overlap; secondary comparisons unadjusted.'))
    return {p: interval(draws[p]) for p in PHASES}, comparisons

def source_positive_cluster_interval(rows):
    positive = rows[rows.label == 1].copy()
    groups = positive.groupby('partition_unit_id').agg(successes=('prediction', 'sum'), size=('prediction', 'size'))
    if groups.empty: return None
    rng = np.random.default_rng(20261011)
    ids = rng.integers(len(groups), size=(REPETITIONS, len(groups)))
    values = groups.successes.to_numpy()[ids].sum(1) / groups['size'].to_numpy()[ids].sum(1)
    return interval(values)

def render_table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |',
                      *['| ' + ' | '.join(str(x) for x in row) + ' |' for row in rows]])

def number(value):
    return '不可计算' if value is None or pd.isna(value) else f'{float(value):.4f}'

def plot_files(out, pred, history, resources):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for phase in PHASES:
        rows = pred[(pred.phase == phase) & (pred.source == 'AortaSeg')]
        fpr, tpr, _ = roc_curve(rows.label, rows.score)
        precision, recall, _ = precision_recall_curve(rows.label, rows.score)
        axes[0].plot(fpr, tpr, label=LABELS[phase], color=COLORS[phase])
        axes[1].plot(recall, precision, label=LABELS[phase], color=COLORS[phase])
    axes[0].plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=.7)
    for ax, title, x, y in zip(axes, ['AortaSeg OOF ROC (N=29)', 'AortaSeg OOF PR (10 positive / 19 negative)'],
                              ['False positive rate', 'Recall'], ['True positive rate', 'Precision']):
        ax.set(title=title, xlabel=x, ylabel=y, xlim=(0, 1), ylim=(0, 1)); ax.legend()
    fig.tight_layout()
    for suffix in ['png', 'pdf']: fig.savefig(out / f'roc_pr.{suffix}', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(5, 3, figsize=(13, 15), squeeze=False)
    for fold in range(5):
        for phase in PHASES:
            rows = history[(history.fold == fold) & (history.phase == phase)].sort_values('step')
            axes[fold, 0].plot(rows.step, rows.supervised.rolling(15, min_periods=1).mean(), color=COLORS[phase], label=LABELS[phase])
            val = rows.dropna(subset=['validation_auroc'])
            axes[fold, 1].plot(val.step, val.validation_auroc, 'o-', color=COLORS[phase], label=LABELS[phase], markersize=3)
            if phase == 's1': axes[fold, 2].plot(rows.step, rows.consistency.rolling(15, min_periods=1).mean(), color=COLORS[phase])
        for col, title, ylabel in [(0, 'BCE (15-step mean)', 'Loss'), (1, 'AS validation AUROC (N=5)', 'AUROC'), (2, 'S1 consistency (15-step mean)', 'MSE')]:
            axes[fold, col].set(title=f'Fold {fold}: {title}', xlabel='Phase step', ylabel=ylabel)
        axes[fold, 1].set_ylim(-.03, 1.03)
    axes[0, 0].legend(); axes[0, 1].legend(); fig.tight_layout()
    for suffix in ['png', 'pdf']: fig.savefig(out / f'training_curves.{suffix}', dpi=160)
    plt.close(fig)
    trained = resources[resources.action == 'train'].groupby(['fold', 'phase']).agg(seconds=('seconds', 'sum'),
                        allocated=('peak_allocated_mib', 'max'), reserved=('peak_reserved_mib', 'max'))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ticks = np.arange(5)
    for i, phase in enumerate(PHASES):
        rows = trained.xs(phase, level='phase').sort_index()
        axes[0].bar(ticks + (i - 1) * .24, rows.seconds / 60, width=.24, color=COLORS[phase], label=LABELS[phase])
        axes[1].plot(ticks, rows.allocated / 1024, 'o-', color=COLORS[phase], label=f'{LABELS[phase]} allocated')
        axes[1].plot(ticks, rows.reserved / 1024, 's--', color=COLORS[phase], alpha=.65, label=f'{LABELS[phase]} reserved')
    axes[0].set(title='Recorded train invocation time', xlabel='Fold', ylabel='Minutes', xticks=ticks)
    axes[1].set(title='Own PyTorch peaks (not whole board)', xlabel='Fold', ylabel='GiB', xticks=ticks)
    for ax in axes: ax.legend(fontsize=8)
    fig.tight_layout()
    for suffix in ['png', 'pdf']: fig.savefig(out / f'time_memory.{suffix}', dpi=180)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True); args = parser.parse_args()
    freeze_dir = ROOT / 'reports/full-study-freeze-20261010'
    job = json.loads((freeze_dir / 'full-study-job.json').read_text())
    assert job['training_complete'] and job['evaluation_complete'], 'Report requires all planned training and evaluations'
    freeze = json.loads((freeze_dir / 'config-freeze-proof.json').read_text())
    assert digest(freeze['config']) == freeze['config_sha256']
    cfg = config(freeze['config']); pred, runs, manifest = load_complete(cfg)
    assert len(manifest) == 712 and len(pred) == 729 and pred.patient_id.nunique() == 243
    manifest['partition_unit_id'] = [str(row.leakage_group_id).strip() or str(pid) for pid, row in manifest.iterrows()]
    for col in ['label_basis', 'partition_unit_id', 'clinical_qc', 'technical_only']:
        if col in manifest: pred[col] = pred.patient_id.map(manifest[col])
    primary = pred[pred.source == 'AortaSeg'].copy()
    assert primary.groupby('phase').size().eq(29).all()
    assert primary[primary.phase == 's0'].partition_unit_id.nunique() == 29
    primary_ci, comparisons = paired_primary(primary)
    summaries = []
    for phase in PHASES:
        rows = pred[pred.phase == phase]
        for source in ['AortaSeg', 'ImageTAAD', 'ImageTBAD', 'mixed_source_secondary']:
            selected = rows if source == 'mixed_source_secondary' else rows[rows.source == source]
            m = dict(phase=phase, source=source, **measures(selected))
            lo, hi = exact_binomial_interval(m['tp'], m['tp'] + m['fn']); m.update(sensitivity_exact_ci_low=lo, sensitivity_exact_ci_high=hi)
            lo, hi = exact_binomial_interval(m['tn'], m['tn'] + m['fp']); m.update(specificity_exact_ci_low=lo, specificity_exact_ci_high=hi)
            if source == 'AortaSeg': m.update(auroc_ci_low=primary_ci[phase][0], auroc_ci_high=primary_ci[phase][1])
            if source != 'mixed_source_secondary': m['positive_partition_unit_bootstrap_ci95'] = source_positive_cluster_interval(selected)
            summaries.append(m)
    histories, resources, fold_metrics = [], [], []
    for fold in range(5):
        controls = json.loads((freeze_dir / f'fold{fold}-continuation-comparison-controls.json').read_text())
        assert all(controls[x] for x in ['supervised_visits_equal', 'preserved_parent_ema', 'seeds_equal']) and controls['steps'] == 400
        for phase in PHASES:
            directory = Path(cfg['output']) / 'mass' / f'seed{42 + fold}' / f'fold{fold}' / phase
            for line in (directory / 'train.jsonl').read_text().splitlines(): histories.append(dict(fold=fold, phase=phase, **json.loads(line)))
            for f in directory.glob('*-invocation-*.json'):
                inv = json.loads(f.read_text()); assert 'status' in inv
                resources.append(dict(invocation=str(f), **inv))
            metric = json.loads((directory / 'evaluation/metrics.json').read_text())
            fold_metrics.append(dict(fold=fold, phase=phase, **metric['by_source']['AortaSeg']))
    history, resource_df, folds = pd.DataFrame(histories), pd.DataFrame(resources), pd.DataFrame(fold_metrics)
    assert len(history) == 7000 and not history[['supervised', 'consistency', 'lr']].isna().any().any()
    out = Path(args.output); assert not out.exists(), 'Use a new output directory; preserve earlier analyses'; out.mkdir(parents=True)
    pred.to_csv(out / 'out_of_fold_predictions.csv', index=False)
    pred[pred.prediction != pred.label].to_csv(out / 'misclassified_cases.csv', index=False)
    pd.DataFrame(summaries).to_csv(out / 'metrics.csv', index=False)
    folds.to_csv(out / 'fold_metrics.csv', index=False); history.to_csv(out / 'training_steps.csv', index=False)
    resource_df.to_csv(out / 'own_process_resources.csv', index=False)
    manifest.to_csv(out / 'frozen_included_scans.csv')
    canonical = pd.read_csv(ROOT / 'data/manifests/cases_deduplicated_candidate.csv', keep_default_na=False)
    excluded = canonical[~canonical.patient_id.isin(manifest.index)].copy(); assert len(excluded) == 8
    excluded.to_csv(out / 'excluded_scans.csv', index=False)
    manifest.groupby(['source', 'label']).size().rename('scans').reset_index().to_csv(out / 'source_counts.csv', index=False)
    save_json(out / 'paired_comparisons.json', comparisons)
    save_json(out / 'frozen_protocol.json', freeze)
    save_json(out / 'completed_job_snapshot.json', job)
    plot_files(out, pred, history, resource_df)
    main_rows = [m for m in summaries if m['source'] == 'AortaSeg']
    delta = comparisons[0]; lo, hi = delta['ci95']
    conclusion = ('观察到S1的同来源AUROC提升，但尚不足以证明独立临床泛化。' if lo > 0 else
                  '观察到S1的同来源AUROC下降；本固定协议没有支持半监督改善。' if hi < 0 else
                  '配对AUROC差异区间跨越或包含0，本固定协议未得到明确半监督改善证据。')
    source_rows = [[m['source'], LABELS[m['phase']], m['positive'], number(m['sensitivity']),
                   '/'.join(number(x) for x in m['positive_partition_unit_bootstrap_ci95'])] for m in summaries
                   if m['source'] in ['ImageTAAD', 'ImageTBAD']]
    totals = resource_df.groupby(['action', 'phase']).agg(seconds=('seconds', 'sum'), peak_allocated_mib=('peak_allocated_mib', 'max'),
                         peak_reserved_mib=('peak_reserved_mib', 'max'), invocations=('status', 'size')).reset_index()
    totals.to_csv(out / 'time_memory_summary.csv', index=False)
    lines = [
        '# 公开CTA半监督研究完整实验报告', '',
        f'生成UTC：{datetime.datetime.now(datetime.timezone.utc).isoformat()}。固定5折、15个训练阶段及15个评估阶段均已完成；该报告是本协议的完整探索性实验交付。', '',
        '## 1. 研究问题与结果概要', '',
        '研究问题：在相同监督访问和续训预算下，诊断未知CTA的Mean Teacher一致性训练，能否优于继续监督训练？主要比较S1与S0+。MASS外部自监督编码器、注意力MIL聚合及EMA教师共同用于病例分数学习。', '',
        f'主要评估为同来源AortaSeg的29个OOF扫描（阳性10、阴性19）。S1−S0+ AUROC差异={delta["estimate"]:.4f}，配对95%区间[{lo:.4f}, {hi:.4f}]。{conclusion} 所有数值仅描述该便利样本、既定灰度网格及固定训练协议。', '',
        '## 2. 数据纳入、标签和有限QC', '',
        '725份源扫描完成头信息和体素读取审计；排除5份完全相同副本后720候选，保留既有8项暂缓，正式纳入712扫描。最终224来源阳性、19来源支持的阴性候选（训练label=0）、469诊断未知；扫描ID及707个保守划分单位均不能称为已确认独立患者。', '',
        render_table(['来源', '阳性', '阴性', '未知'], [['ImageTAAD',114,0,0],['ImageTBAD',100,0,0],['AortaSeg',10,19,0],['DB-SNet',0,0,469],['合计',224,19,469]]), '',
        '暂缓：Old01、TAAD2/39/84及4个既有DB条目，具体身份/依据保留excluded_scans.csv；没有根据测试性能选择纳入。正式类别来自canonical已有来源证据，不从图像重新赋诊断。Old/Young的19个阴性候选没有在本任务中逐例由专家确认排除夹层；表格中的阴性仅指冻结标签0，不等同于临床确诊阴性。未知DB始终-1，教师软目标不作为确认诊断。', '',
        '新TAAD/TBAD150、AS18、DB465共633精确当前绑定，复用旧TAAD62、AS11、TBAD2、DB4的既有精确批准。DB实际5225中心、927补充、430指定块13760深度；新150为2094中心430补充、42块1344深度；AS18为251中心45补充、3块96深度。', '',
        'QC范围为精确输入、ROI/provenance、预处理/NPZ/PNG快照、稀疏ROI、全部块中心及指定深度，不是每层/每体素、临床全壁或病灶覆盖认证。部分块存在网格外填充、心脏背景、强白器械样环/条纹或有限XY范围，逐例原观察保留；32深度被看不等于32层有完整居中主动脉。未进行新增临床诊断审阅。', '',
        '## 3. 模型与冻结参数', '',
        'MASS提供外部掩膜引导自监督预训练编码器。本实验加载mass_base.pth的ema_model编码器、输出512维特征并只微调最后阶段，没有重新执行MASS预训练。[MASS官方实现](https://github.com/Stanford-AIMI/MASS)', '',
        '每份扫描含多个32×96×96块，块特征结合三维归一化位置，通过softmax注意力加权聚合后分类。注意力是聚合权重，不是病灶标注或因果解释。[注意力MIL原论文](https://proceedings.mlr.press/v80/ilse18a.html)', '',
        'Mean Teacher学生以BCE及无标签病例软分数MSE反向更新，教师采用EMA α=0.99。教师弱/学生强视图仅灰度增益和噪声不同，几何坐标相同；一致性权重前200续训步由低至0.1，无硬伪标签筛选。[Mean Teacher原论文](https://arxiv.org/abs/1703.01780)', '',
        'Titan RTX显式float32、AMP关闭、headFP32、chunk32、激活检查点；学习率1e-4、AdamW权重衰减1e-4，编码器last_stage。预处理配置为名义1.5网格、[-991,500]映射范围、ROI zscore、8名义块和覆盖补充上限128。未知原始HU/mm、原生左右或起点没有被恢复，逐例原校准/几何假设仍有效。', '',
        '## 4. 划分和三组比较约束', '',
        '按真实来源/类别分层固定5折，外层测试覆盖全部243有标签扫描每阶段各一次；内层AS验证均5例（2阳性/3阴性），用于最佳EMA模型和Youden阈值，其他来源验证不决定模型选择。不同折模型的分数尺度可不同，故同时保留逐折指标。', '',
        'v4五组：TAAD1/111、120/121、6/118、9/17和DB d9bda0ac/d18b6dfa。同组训练/验证/测试/未知互斥。DB对反向裁剪中央样本相关0.999994、TAAD9/17有限平移最佳相关0.813724，后者仍未决而保守合组；不是确认五个患者，也不是完整独立性审计。', '',
        '每折S0从原MASS初始化完成600步；S0+/S1分别400步，共7000监督优化步。两延续共享同一AS验证最佳S0快照（匹配student、optimizer、EMA）、种子42+fold、监督访问序列、增强和学习率日程，父EMA原样保留。S0父选择步数可早于600，但S0计算预算全部执行；逐折父SHA和访问相等已实际断言。S1额外未知前向/反向计算另报。', '',
        '全部15训练阶段结束后才运行15项固定评估，无测试调参、阈值或模型选择。固定预算并不证明充分收敛；未根据最终测试结果延长训练。', '',
        '## 5. 同来源主要结果', '',
        render_table(['组别','N（阳性/阴性）','OOF AUROC（95% CI）','AUPRC','敏感度（精确95% CI）','特异度（精确95% CI）'],
         [[LABELS[m['phase']],f'{m["n"]} ({m["positive"]}/{m["negative"]})',f'{number(m["auroc"])} ({number(m["auroc_ci_low"])}–{number(m["auroc_ci_high"])})',number(m['auprc']),
           f'{number(m["sensitivity"])} ({number(m["sensitivity_exact_ci_low"])}–{number(m["sensitivity_exact_ci_high"])})',
           f'{number(m["specificity"])} ({number(m["specificity_exact_ci_low"])}–{number(m["specificity_exact_ci_high"])})'] for m in main_rows]), '',
        '![同来源OOF ROC和PR](roc_pr.png)', '',
        render_table(['组别','逐折AS AUROC均值','逐折AS AUROC样本标准差'], [[LABELS[p],number(folds[folds.phase==p].auroc.mean()),number(folds[folds.phase==p].auroc.std())] for p in PHASES]), '',
        'OOF AUROC区间为10000次类别分层扫描bootstrap；敏感度/特异度为Clopper–Pearson精确区间。区间条件于这些固定OOF预测，没有涵盖重训练、交叉验证模型相关、抽样偏倚或未知患者重叠。AUPRC依赖10/29阳性比例；逐折标准差不是独立训练重复的置信区间。', '',
        '## 6. 配对比较', '',
        render_table(['比较','AUROC差异','配对95% CI','A正确/B错误','B正确/A错误','McNemar精确p'], [[c['comparison'],number(c['estimate']),f'{number(c["ci95"][0])}–{number(c["ci95"][1])}',c['discordant_a_correct_b_wrong'],c['discordant_b_correct_a_wrong'],number(c['mcnemar_exact_p'])] for c in comparisons]), '',
        '主要比较为S1−S0+；其余探索性比较未做多重检验调整。同一AS扫描在三组间配对，AUROC差异使用同一bootstrap抽样索引；McNemar检验阈值分类正确性差异，不是AUROC检验。没有把区间跨0误写为两组等效。', '',
        '## 7. 其他来源与病例记录', '',
        render_table(['来源','组别','阳性数','敏感度','已知划分单位bootstrap 95% CI'],source_rows), '',
        'ImageTAAD/TBAD只有阳性，不能计算有意义的同来源AUROC或特异度。这里按已有v4保守单位重采样阳性敏感度，不能解决其他未知患者相关；混合243例结果仅次要，见metrics.csv，来源识别和阳性比例混杂可抬高该结果。', '',
        '逐扫描分数、预测、折、验证阈值、来源/类别依据及聚合权重见out_of_fold_predictions.csv；错误清单见misclassified_cases.csv。分数不视为已校准疾病概率，权重不用于重赋诊断或筛块。DB没有真值诊断指标。', '',
        '## 8. 真实耗时、显存和失败恢复', '',
        render_table(['动作','组别','记录调用累计分钟','本进程峰值分配MiB','本进程峰值预留MiB','调用数'], [[x.action,LABELS[x.phase],f'{x.seconds/60:.2f}',f'{x.peak_allocated_mib:.1f}',f'{x.peak_reserved_mib:.1f}',int(x.invocations)] for _,x in totals.iterrows()]), '',
        '![逐折真实时间与本进程显存](time_memory.png)', '',
        '时间来自每次phase调用记录，包括该调用预检及模型建立；训练循环时间另由completed.json记录。失败和恢复调用如存在均保留在own_process_resources.csv，不将最终恢复一次耗时冒充总耗时。上述显存来自本进程PyTorch allocator，预留与分配含义不同；整卡显存包含其他任务，不能归属本实验。用户已取消30%显存要求，未进行占位保活或干预其他进程。', '',
        '完整预处理465项于10月9日06:06:33成功退出，零错误；剩余人工限定QC及同步入账后才开训。同步曾因自动审批等待授权停留381绑定，84本地记录没有虚计；真人明确授权后正常部署/逐块绑定，全部465入账。脚本元数据/路径/标识符/args遮蔽失败、同名查看器保护及修复原记录保留，不改变CT、诊断或缓存，也不改实验预算。', '',
        '## 9. 训练曲线与解释范围', '',
        '![逐折损失、AS验证AUROC及一致性曲线](training_curves.png)', '',
        '曲线在每折内分别显示各阶段，不把不同折末尾与下一折起点连成同一曲线。AS内层验证只有5例，AUROC和最佳检查点选择很不稳定；一致性损失降低不等于诊断改善。S1与S0+同监督预算但计算成本不同。', '',
        '## 10. 局限及结论', '',
        f'{conclusion} 本结果不能确定MASS、MIL或Mean Teacher单独贡献，因为未设置去掉这些模块的独立消融。三者为既有方法，不能将组合直接声称为新算法优势。', '',
        '- 阴性候选只有19例，来自Old/Young便利来源且没有新增专家确诊排除，缺少真实临床疑似人群的非夹层病变，存在年龄、来源和类别混杂；同来源AS分析也未消除全部偏倚。',
        '- 已排除5个完全副本并设置5个合理保守组，其他同患者/不同裁剪重建及MASS外部预训练重叠仍未知；低相关不证明独立。',
        '- 首版29扫描开发预实验已有查看/训练，S1未观察提升、S0+/S1 BF16同分原因未知；完整队列与开发观察可能重叠，不能宣称完全未触碰的外部测试。',
        '- 探索性灰度/体素网格、腹部/下胸覆盖占比较高及根部/弓覆盖有限，强白/条纹、边缘填充、心脏上下文和局部ROI偏选限制病灶判断。有限技术QC不等于全壁或全体素医学审阅。',
        '- 单组5折种子、短固定步数、极小验证集和无独立外部医院测试不足以确定模型稳定性或临床效用。评估目标仅CTA夹层类别，不代表全部AAS或平扫CT验证。',
        '- 后续需要可靠患者身份、专家诊断、更多独立阴性和外部测试；如研究平扫迁移或临床多模态，应另行取得对应数据和标签。', '',
        '## 11. 完整交付与可复现性', '',
        '数据数量/纳入/筛除、冻结配置/划分、逐例OOF/错误、主次指标/配对区间、全部7000步曲线数据及本进程耗时显存随报告交付。原始输入/缓存及训练latest/best仍保留服务器；该报告包不包含CT、NPZ或模型权重，不是完整模型备份。', '',
        f'配置SHA `{freeze["config_sha256"]}`；manifestSHA `{freeze["manifest_sha256"]}`；foldsSHA `{freeze["folds_sha256"]}`；原MASS权重SHA `{freeze["original_mass_checkpoint_sha256"]}`。完整代码哈希见frozen_protocol.json，逐折父和比较控制保留服务器证据。', '',
        '- metrics.csv、fold_metrics.csv、paired_comparisons.json：主/次指标与配对分析。',
        '- out_of_fold_predictions.csv、misclassified_cases.csv：729行（三组×243扫描）配对结果。',
        '- frozen_included_scans.csv、excluded_scans.csv、source_counts.csv：712纳入、8暂缓和来源/标签依据。',
        '- training_steps.csv、own_process_resources.csv、time_memory_summary.csv：7000步日志与每次调用资源。',
        '- roc_pr、training_curves、time_memory：PNG及PDF图；frozen_protocol.json、completed_job_snapshot.json、report_proof.json：冻结与完成证据。', ''
    ]
    report = out / 'CTA半监督完整研究报告.md'; report.write_text('\n'.join(lines))
    save_json(out / 'report_proof.json', dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), all_planned_training_complete=True,
        all_planned_evaluation_complete=True, labeled_scans=243, phase_prediction_rows=729, training_step_rows=7000,
        analysis_seed=20261010, bootstrap_repetitions=REPETITIONS, primary_comparison='S1 minus S0+',
        primary_source='AortaSeg', no_model_or_threshold_selection_from_test=True, config_sha256=freeze['config_sha256'],
        generator_sha256=digest(__file__), files={str(p.relative_to(out)): dict(bytes=p.stat().st_size, sha256=digest(p)) for p in out.iterdir() if p.is_file()}))
    print('COMPLETE_FINAL_REPORT', report, flush=True)

if __name__ == '__main__':
    main()
