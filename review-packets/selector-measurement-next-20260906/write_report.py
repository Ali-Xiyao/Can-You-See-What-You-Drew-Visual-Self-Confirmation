"""Render completed paired diagnostics with explicit populations and limitations."""
from common import *


def pct(x):return '—' if x is None else f'{100*x:.2f}%'
def pp(x):return '—' if x is None else f'{100*x:+.2f}pp'
def ci(x):
    return '无可判场景' if x['point'] is None else f"{pp(x['point'])} [{100*x['ci_low']:+.2f},{100*x['ci_high']:+.2f}]，n={x['n_scenes']}"


def main():
    r=read(SIDE/'results.json');v=read(SIDE/'results-validation.json')
    assert v['report_sha256']==sha(SIDE/'results.json')
    out=['# 固定图像三条件诊断结果','',
         '本轮测观察/选择随训练的变化；没有更新模型、生成新图或测新梯度。D*和D_g均未由本轮确认。','',
         'Naive训练臂是主分析，Gold训练臂是对照。主集合为此前冻结排除训练场景重叠后的14池；原16池完整结果并列提供。全部为已经看过的平衡开发池，不能称新的自然总体验证。','',
         '三个条件：①原描述和自绘措辞；②保留同一措辞，仅描述留空；③仅原题，整段自绘上下文移除。主要描述内容干预为②−①；③−①是较宽的上下文干预。题面始终含spec信息。','',
         '原评分仍是原题均分；扩展评分追加与原存在题同数的否定题后再均分。扩展版本没有参与历史训练。','']
    for cohort,name in [('primary_14','主分析：14池'),('full_16','补充：完整16池')]:
        s=r['summary'][cohort];c=r['contrasts'][cohort]
        out += [f'## {name}','', '实际选图收益相对同池全部候选均匀随机期望，按池等权。','',
                '| checkpoint | 带描述：原评分收益 | 空描述：原评分收益 | 仅题目：原评分收益 | 带描述：扩展评分收益 |','|---|---:|---:|---:|---:|']
        for tid,z in s.items():
            values=[z[cond]['selection'][variant]['actual_gain'] for cond,variant in [('prompt_on','original'),('prompt_blank','original'),('prompt_off','original'),('prompt_on','expanded')]]
            out += ['| '+tid+' | '+' | '.join(pp(x) for x in values)+' |']
        out += ['', '原题中“事实与请求冲突”子集上的事实准确率差。括号为探索性场景bootstrap 95%区间（单位pp）；n为有相应可判事实的场景数。','',
                '| checkpoint | 空描述−带描述 | 仅题目−带描述 | 空描述干预效应相对base的变化 |','|---|---|---|---|']
        for tid,z in c.items():
            a=z['prompt_blank_minus_on']['original_all_conflicting_fact_accuracy']
            b=z['prompt_off_minus_on']['original_all_conflicting_fact_accuracy']
            key='prompt_blank_effect_change_from_base_conflicting_original_fact_accuracy'
            out += [f"| {tid} | {ci(a)} | {ci(b)} | {ci(z[key]) if key in z else '共享起点'} |"]
        out += ['', '题级事实始终使用同一可判集合；先图内均题、再池内均图、最后池等权。以下为带描述条件。','',
                '| checkpoint | 原存在题：符合事实 | 计数题：符合事实 | 否定题：符合事实 | 原题：符合请求 |','|---|---:|---:|---:|---:|']
        for tid,z in s.items():
            f=z['prompt_on']['factual']
            values=[f[g][metric]['point'] for g,metric in [('existence','response_fact'),('count','response_fact'),('negative_existence','response_fact'),('original_all','response_request_all')]]
            out += ['| '+tid+' | '+' | '.join(pct(x) for x in values)+' |']
    out += ['', '## 完整性与范围','',
            '| checkpoint | 原题历史复现 | 错误/弃答（全部三条件） |','|---|---:|---:|']
    for tid,done in r['completion_audits'].items():
        rr=done['standard']['old_prompted_answer_reproduction']
        errs=sum(x['n_errors'] for x in done.values());abst=sum(x['n_abstained'] for x in done.values())
        out += [f"| {tid} | {rr['matches']}/{rr['total']} | {errs}/{abst} |"]
    out += ['',f"独立核验 {v['checks']['answers_checked']} 条回答的原始文本规范化及问题/图片/条件绑定，{v['checks']['pool_selections_checked']} 个池选择统计、{v['checks']['paired_contrasts_checked']} 组配对比较。所有被载入adapter的参数digest前后相同；原输入哈希不变。",'',
            '否定原子的CPU覆盖检查：在2733条可判新增问题中仅3条对应实际额外物体；共同79池理想选择前后不变。固定80图库新增事实199/202可判，全为不存在。真实模型加题后即使分数改变，也不能据此认定正确覆盖提高。','',
            '这些区间条件于已经看过的少量开发场景，不含独立训练种子不确定性，没有多重查看校正。全体场景、冲突子集和题级事实子集的分母均不同，不合并成总体结论。','',
            '下一版独立场景规模应依据实际测量方差及未决率规划；precision-planning.json给出多种变化率假设下的范围，不把200或800池当作已验证答案。', '',
            '完整逐池数据：[results.json](results.json)；独立验证：[results-validation.json](results-validation.json)；冻结设计：[PROTOCOL.md](PROTOCOL.md)与[补充](FRAME_ADDENDUM.md)。','']
    with (SIDE/'RESULTS.md').open('x',encoding='utf-8') as f:f.write('\n'.join(out))
    print('Wrote RESULTS.md from validated complete data.')


if __name__=='__main__':main()
