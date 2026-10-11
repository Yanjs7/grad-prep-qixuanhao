#!/usr/bin/env python3
"""Report complete out-of-fold predictions with source-specific caveats."""
import argparse
import datetime
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from scipy.stats import beta

from cta_ssl.common import config, digest, save_json


PHASES=['s0','s0plus','s1']


def exact_binomial_interval(successes,total):
    if not total:return None,None
    low=0. if successes==0 else float(beta.ppf(.025,successes,total-successes+1))
    high=1. if successes==total else float(beta.ppf(.975,successes+1,total-successes))
    return low,high


def measures(rows):
    y=rows.label.to_numpy(dtype=int);p=rows.prediction.to_numpy(dtype=int);s=rows.score.to_numpy(dtype=float)
    tp=int(((y==1)&(p==1)).sum());tn=int(((y==0)&(p==0)).sum())
    fp=int(((y==0)&(p==1)).sum());fn=int(((y==1)&(p==0)).sum())
    sens=tp/(tp+fn) if tp+fn else None;spec=tn/(tn+fp) if tn+fp else None
    both=set(y)=={0,1}
    return dict(n=len(y),positive=int((y==1).sum()),negative=int((y==0).sum()),
        auroc=float(roc_auc_score(y,s)) if both else None,
        auprc=float(average_precision_score(y,s)) if both else None,
        sensitivity=sens,specificity=spec,
        balanced_accuracy=(sens+spec)/2 if both else None,tp=tp,tn=tn,fp=fp,fn=fn)


def bootstrap(rows,repetitions,seed):
    rng=np.random.default_rng(seed);groups=[g.index.to_numpy() for _,g in rows.groupby('label')]
    sampled={k:[] for k in ['auroc','auprc','sensitivity','specificity','balanced_accuracy']}
    for _ in range(repetitions):
        ids=np.concatenate([rng.choice(g,size=len(g),replace=True) for g in groups])
        m=measures(rows.loc[ids])
        for key in sampled:
            if m[key] is not None:sampled[key].append(m[key])
    return {key:list(map(float,np.percentile(value,[2.5,97.5]))) if value else None for key,value in sampled.items()}


def load_complete(cfg):
    manifest=pd.read_csv(cfg['manifest'],keep_default_na=False).set_index('patient_id',verify_integrity=True)
    plan=json.loads(Path(cfg['folds']).read_text())
    result=[];runs=[]
    seen=set()
    mh,fh=digest(cfg['manifest']),digest(cfg['folds'])
    for fold in plan['folds']:
        number=int(fold['fold']);seed=int(cfg.get('seed',42))+number
        splitsets=[set(fold[k]) for k in ['train','val','test']]
        if any(splitsets[i]&splitsets[j] for i in range(3) for j in range(i)) or any(s&set(plan.get('unlabeled',[])) for s in splitsets):
            raise ValueError('Training, validation, test or unlabeled identities overlap')
        if seen&set(fold['test']):raise ValueError('Overlapping outer test folds')
        seen.update(fold['test'])
        base=Path(cfg['output'])/cfg['model']['encoder']/f'seed{seed}'/f'fold{number}'
        parent=None
        for phase in PHASES:
            folder=base/phase;ev=folder/'evaluation'
            for path in [folder/'completed.json',folder/'run.json',folder/'best.pt',ev/'metrics.json',ev/'predictions.csv']:
                if not path.is_file():raise ValueError(f'Full experiment incomplete: {path}')
            run=json.loads((folder/'run.json').read_text());done=json.loads((folder/'completed.json').read_text())
            meta=run['meta'];expected_steps=int(cfg['training']['initial_steps'] if phase=='s0' else cfg['training']['continuation_steps'])
            if run['config']!=cfg or meta['fold']!=number or meta['phase']!=phase or meta['seed']!=seed or meta['manifest_sha256']!=mh or meta['folds_sha256']!=fh or done['steps']!=expected_steps:
                raise ValueError('Run budget, identity or provenance differs from locked configuration')
            if any(set(meta.get(k+'_ids',[]))!=set(fold[k]) for k in ['train','val','test']):
                raise ValueError('Recorded training/validation/test identities differ from frozen folds')
            checkpoint=digest(folder/'best.pt')
            if phase=='s0':parent=checkpoint
            elif meta.get('parent_sha256')!=parent:raise ValueError('S0+ and S1 did not share the same S0 parent')
            evaluation=json.loads((ev/'metrics.json').read_text())
            if evaluation.get('checkpoint_sha256')!=checkpoint or evaluation.get('manifest_sha256')!=mh or evaluation.get('folds_sha256')!=fh:
                raise ValueError('Evaluation provenance missing or stale')
            predictions=pd.read_csv(ev/'predictions.csv')
            if predictions.patient_id.duplicated().any() or set(predictions.patient_id)!=set(fold['test']):
                raise ValueError('Prediction identities do not exactly match outer test fold')
            for _,row in predictions.iterrows():
                expected=manifest.loc[row.patient_id]
                if int(expected.label)!=int(row.label) or expected.source!=row.source:raise ValueError('Prediction labels or sources differ')
            if not np.isfinite(predictions.score).all() or not predictions.score.between(0,1).all() or not set(predictions.prediction)<=set([0,1]):
                raise ValueError('Invalid prediction values')
            threshold=evaluation['mixed_source_secondary']['threshold']
            if not np.array_equal(predictions.prediction.to_numpy(),(predictions.score>=threshold).astype(int).to_numpy()):raise ValueError('Predictions do not match saved validation threshold')
            predictions['fold']=number;predictions['phase']=phase;predictions['threshold']=threshold
            result.append(predictions)
            monitors=[]
            for path in sorted((folder/'monitoring').glob('*/summary.json')):
                monitors.append(dict(path=str(path),**json.loads(path.read_text())))
            runs.append(dict(fold=number,phase=phase,seed=seed,**done,monitoring=monitors))
    labeled=set(manifest.index[manifest.label!=-1])
    if cfg.get('study_scope') == 'exploratory_holdout':
        if len(plan['folds']) != 1:
            raise ValueError('Exploratory holdout requires one explicitly planned split')
        covered = set().union(*(set(plan['folds'][0][key]) for key in ['train','val','test']))
        if covered != labeled:
            raise ValueError('Holdout split must account for every included labeled scan')
    elif seen!=labeled:
        raise ValueError('Outer test folds do not cover every reviewed labeled case exactly once')
    return pd.concat(result,ignore_index=True),runs,manifest


def paired_difference(a,b,repetitions):
    a=a.set_index('patient_id').sort_index();b=b.set_index('patient_id').sort_index()
    if not a.index.equals(b.index) or not a.label.equals(b.label):raise ValueError('Paired comparison needs identical cases')
    rng=np.random.default_rng(20261007);groups=[np.flatnonzero(a.label.to_numpy()==y) for y in [0,1]]
    if any(not len(g) for g in groups):return None
    differences=[]
    for _ in range(repetitions):
        ids=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])
        differences.append(roc_auc_score(a.label.iloc[ids],a.score.iloc[ids])-roc_auc_score(b.label.iloc[ids],b.score.iloc[ids]))
    return dict(metric='AortaSeg AUROC: S1 minus S0+',estimate=measures(a)['auroc']-measures(b)['auroc'],
        ci95=list(map(float,np.percentile(differences,[2.5,97.5]))),
        method='Paired class-stratified case bootstrap; does not capture CV training-model uncertainty.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--bootstrap',type=int,default=2000);args=p.parse_args()
    if args.bootstrap<100:raise ValueError('Use at least 100 bootstrap resamples')
    cfg=config(args.config);pred,runs,manifest=load_complete(cfg)
    holdout = cfg.get('study_scope') == 'exploratory_holdout'
    prediction_name = 'holdout_predictions.csv' if holdout else 'out_of_fold_predictions.csv'
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    pred.to_csv(out/prediction_name,index=False)
    pred[pred.prediction!=pred.label].to_csv(out/'misclassified_cases.csv',index=False)
    manifest.groupby(['source','label']).size().rename('scans').reset_index().to_csv(out/'source_counts.csv',index=False)
    columns=[c for c in ['source','label','label_basis','intensity_scale','intensity_offset','intensity_basis',
        'geometry_basis','spacing_override_xyz','flip_x_direction','flip_y_direction','flip_z_direction','clinical_qc','prepared_qc'] if c in manifest]
    manifest[columns].to_csv(out/'input_assumptions.csv')
    summaries=[]
    for i,phase in enumerate(PHASES):
        phase_rows=pred[pred.phase==phase].reset_index(drop=True)
        for source in ['mixed_source_secondary',*sorted(phase_rows.source.unique())]:
            rows=phase_rows if source=='mixed_source_secondary' else phase_rows[phase_rows.source==source]
            record=dict(phase=phase,source=source,**measures(rows))
            low,high=exact_binomial_interval(record['tp'],record['tp']+record['fn'])
            record.update(sensitivity_exact_ci_low=low,sensitivity_exact_ci_high=high)
            low,high=exact_binomial_interval(record['tn'],record['tn']+record['fp'])
            record.update(specificity_exact_ci_low=low,specificity_exact_ci_high=high)
            if source=='AortaSeg':
                for key,bounds in bootstrap(rows,args.bootstrap,42+i).items():
                    record[key+'_ci_low']=None if bounds is None else bounds[0]
                    record[key+'_ci_high']=None if bounds is None else bounds[1]
            summaries.append(record)
    pd.DataFrame(summaries).to_csv(out/'metrics.csv',index=False)
    main_rows=pred[pred.source=='AortaSeg']
    delta=paired_difference(main_rows[main_rows.phase=='s1'],main_rows[main_rows.phase=='s0plus'],args.bootstrap)
    save_json(out/'paired_comparison.json',delta)
    save_json(out/'training_and_memory.json',runs)
    training_records, memory_records = [], []
    for run in runs:
        folder = Path(cfg['output']) / cfg['model']['encoder'] / f'seed{run["seed"]}' / f'fold{run["fold"]}' / run['phase']
        log = folder / 'train.jsonl'
        if log.is_file():
            for line in log.read_text().splitlines():
                training_records.append(dict(phase=run['phase'],fold=run['fold'],**json.loads(line)))
        for path in sorted((folder / 'monitoring').glob('*/gpu-memory.jsonl')):
            for line in path.read_text().splitlines():
                memory_records.append(dict(phase=run['phase'],fold=run['fold'],monitor=str(path),**json.loads(line)))
    if training_records:
        history = pd.DataFrame(training_records)
        history.to_csv(out/'training_steps.csv',index=False)
        fig,axes = plt.subplots(1,2,figsize=(10,4))
        for phase in PHASES:
            rows = history[history.phase==phase].sort_values(['fold','step'])
            axes[0].plot(rows.step,rows.supervised.rolling(10,min_periods=1).mean(),label=phase)
            if 'validation_auroc' in rows:
                selected = rows.dropna(subset=['validation_auroc'])
                axes[1].plot(selected.step,selected.validation_auroc,'o-',label=phase)
        axes[0].set(title='Supervised loss (10-step mean)',xlabel='Phase step',ylabel='BCE')
        axes[1].set(title='Validation AUROC (very small sample)',xlabel='Phase step',ylabel='AUROC',ylim=(-.05,1.05))
        for ax in axes:ax.legend()
        fig.tight_layout();fig.savefig(out/'training_curves.png',dpi=160);plt.close(fig)
    if memory_records:
        pd.DataFrame(memory_records).to_csv(out/'gpu_memory_samples.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for phase in PHASES:
        rows=main_rows[main_rows.phase==phase]
        if set(rows.label)!={0,1}:continue
        fpr,tpr,_=roc_curve(rows.label,rows.score);precision,recall,_=precision_recall_curve(rows.label,rows.score)
        axes[0].plot(fpr,tpr,label=phase);axes[1].plot(recall,precision,label=phase)
    axes[0].plot([0,1],[0,1],'k--',linewidth=.7)
    curve_titles = ['AortaSeg holdout ROC','AortaSeg holdout PR'] if holdout else ['AortaSeg OOF ROC','AortaSeg OOF PR']
    for ax,title,x,y in zip(axes,curve_titles,['False positive rate','Recall'],['True positive rate','Precision']):
        ax.set(title=title,xlabel=x,ylabel=y,xlim=(0,1),ylim=(0,1));ax.legend()
    fig.tight_layout();fig.savefig(out/'roc_pr.png',dpi=160);plt.close(fig)
    lines=['# CTA 半监督诊断实验报告','',f'生成时间（UTC）：{datetime.datetime.now(datetime.timezone.utc).isoformat()}','',
        '这是公开数据上的探索性主动脉夹层分类实验。以下为完成全部计划折后的实际结果，不代表临床验证或平扫 CT 诊断性能。','',
        '## 数据与划分','',f'已纳入扫描条目：{len(manifest)}；其中阳性 {(manifest.label==1).sum()}，阴性 {(manifest.label==0).sum()}，类别标签隐藏的训练池 {(manifest.label==-1).sum()}。扫描条目不能未经身份核验称为独立患者。逐来源数量见 source_counts.csv，逐例灰度/空间假设和标注依据见 input_assumptions.csv；原始候选清单及筛除原因需与冻结清单一起保留。','',
        '按来源及类别分层的外层测试折；阈值只在内层验证集 AortaSeg 上选择。全体测试条目在每阶段仅出现一次。未知跨来源患者身份仍可能造成泄漏。','',
        '## 方法与配置','',
        'S0 为初始监督训练；S0+ 与 S1 从同一 S0 最佳验证检查点继续，监督访问次数和继续步数相同。S1 额外使用 EMA 教师弱增强目标及学生强增强的一致性损失，计算量更大。','',
        '主动脉区域由外部模型依据输入影像预测；预处理 v3 使用缓存校验、自适应纵向块采样及实际未覆盖 ROI 的补充图像块。标注及人工质控依据见病例清单；技术批准不等于医生诊断。','',
        '```json',json.dumps(cfg,ensure_ascii=False,indent=2),'```','',
        f'病例清单 SHA256：`{digest(cfg["manifest"])}`；划分 SHA256：`{digest(cfg["folds"])}`。','',
        '## 同来源主要结果','',
        '| 阶段 | N（阳性/阴性） | AUROC（95% CI） | AUPRC | 敏感度（精确95% CI） | 特异度（精确95% CI） | 平衡准确率 |','|---|---|---|---|---|---|---|']
    def number(value):return '不可计算' if value is None or pd.isna(value) else f'{value:.4f}'
    for record in summaries:
        if record['source']!='AortaSeg':continue
        ci=f'{number(record.get("auroc_ci_low"))}–{number(record.get("auroc_ci_high"))}'
        sens_ci=f'{number(record.get("sensitivity_exact_ci_low"))}–{number(record.get("sensitivity_exact_ci_high"))}'
        spec_ci=f'{number(record.get("specificity_exact_ci_low"))}–{number(record.get("specificity_exact_ci_high"))}'
        lines.append(f'| {record["phase"]} | {record["n"]} ({record["positive"]}/{record["negative"]}) | {number(record["auroc"])} ({ci}) | {number(record["auprc"])} | {number(record["sensitivity"])} ({sens_ci}) | {number(record["specificity"])} ({spec_ci}) | {number(record["balanced_accuracy"])} |')
    lines+=['','![ROC 与 PR](roc_pr.png)','','S1 与 S0+ 的配对比较：','', '```json',json.dumps(delta,ensure_ascii=False,indent=2),'```','',
        '同来源区间按类别分层病例 bootstrap；不同折的训练模型相关，因此这些区间没有涵盖完整训练不确定性。AUPRC 依赖本数据的阳性比例。没有为单阳性来源计算 AUROC 或特异度。','',
        '## 其他来源和训练记录','',
        '完整来源指标、混淆矩阵和区间见 metrics.csv；混合来源指标为次要结果，可能反映来源识别。训练各次运行的耗时、PyTorch 峰值分配和真实任务显存监测见 training_and_memory.json。时间为各次已记录调用，不应把最终恢复调用耗时称为所有重启累计耗时。','',
        '显存监测是离散的整卡采样，可包含其他进程，不能保证连续占用超过 30%。监测缺失处不得推断为达标。失败、恢复和历史运行记录应一并核验。逐例错误见 misclassified_cases.csv，应结合原始影像由医生复核；本报告不依据注意力权重推断病变定位。','',
        '## 局限与后续研究','',
        '- 阴性对照仅来自 AortaSeg，样本极少且缺少真实疑似患者中的非夹层病变；不能推断临床特异度。','- ImageTAAD/TBAD 只有阳性，数据来源与标签混杂；主指标采用 AortaSeg 同来源结果仍不能消除小样本偏差。','- DB-SNet 使用目录身份作为扫描编号，类别标签隐藏；跨来源或重复访视的患者身份未知。','- 原始 HU、层距或方向缺失时采用的探索性校准/网格假设及排除病例必须在病例级记录中披露，不能称恢复了原始 DICOM 元数据。','- 输入模型主动脉覆盖不等于夹层病变覆盖；需医生复核错误病例，尤其短段病变和术后病例。','- 当前目标为 CTA 主动脉夹层分类，不覆盖全部 AAS，更不是平扫 CT 研究。','- 后续医院合作需独立的临床诊断标签、真正的阴性病例及患者级外部测试；取得配对平扫/CTA 后再做特征对齐或教师迁移，取得时间对齐的临床数据后再研究多模态。','',
        '## 文件','', '- out_of_fold_predictions.csv：逐扫描测试分数、预测、来源、折号和验证阈值。','- metrics.csv：主/次要结果、各来源及混淆矩阵。','- paired_comparison.json：同来源配对 AUROC 差异及探索性区间。','- training_and_memory.json：各次记录的训练耗时、显存及监测。','- roc_pr.png：主要结果曲线。','']
    if holdout:
        text = '\n'.join(lines)
        text = text.replace('# CTA 半监督诊断实验报告', '# CTA 半监督诊断首版预实验报告')
        text = text.replace('以下为完成全部计划折后的实际结果', '以下为完成单次预先固定留出划分的实际结果，完整队列实验另行保存')
        text = text.replace('按来源及类别分层的外层测试折；', '本次采用便利样本的一次固定训练/验证/测试划分，没有完成交叉验证；')
        text = text.replace('不同折的训练模型相关，因此这些区间没有涵盖完整训练不确定性。', '测试样本很少；bootstrap 区间可能退化，不能涵盖训练及便利抽样的不确定性。')
        text = text.replace('out_of_fold_predictions.csv', prediction_name)
        text += '\n\n## 首版预实验的额外限制\n\n- 本次阴性为 Young 队列的便利样本，缺少 Old 对照，存在年龄与形态混杂。\n- 验证仅两例，阈值和最佳检查点选择非常不稳定；同来源测试仅四例。即使高分也不能宣称临床有效或半监督显著改善。\n- 训练步数为较短的可行性预算，没有确定模型已充分收敛。完整队列和多折结果尚未完成。\n'
        text += '\n\n敏感度和特异度另给出 Clopper–Pearson 精确二项区间。仅2/2例正确时，其95%区间约15.8%–100%；AUROC 的小样本 bootstrap 即使退化为单点，也不代表已确定的临床性能。\n'
        text += '\n\n![真实训练曲线](training_curves.png)\n\n逐步日志见 training_steps.csv；显存采样见 gpu_memory_samples.csv，own_job 字段才是本实验进程的占用。\n'
        report_path = out/'首版预实验报告.md'
        report_path.write_text(text)
    else:
        report_path = out/'实验报告.md'
        report_path.write_text('\n'.join(lines))
    print('COMPLETE_EXPERIMENT_REPORT',report_path,flush=True)


if __name__=='__main__':main()
