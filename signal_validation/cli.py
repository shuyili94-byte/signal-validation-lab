"""Research commands for externally supplied, arbitrary strategy signals."""
from __future__ import annotations

import argparse
import calendar as cal
import csv
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from .research import load_csv, phase1, walk_forward


def month_shift(day, months):
    value = date.fromisoformat(str(day))
    year, month = divmod(value.year * 12 + value.month - 1 + months, 12)
    month += 1
    return date(year, month, min(value.day, cal.monthrange(year, month)[1])).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def new_output(path):
    output = Path(path).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('output directory must be new or empty; existing research is never overwritten')
    output.mkdir(parents=True, exist_ok=True)
    return output


def create_demo(path):
    """Synthetic external 0/1 signals; deliberately contains no trading strategy."""
    rng = np.random.default_rng(20261106)
    days = []
    cursor = date(2019, 1, 1)
    while cursor < date(2025, 1, 1):
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor += timedelta(days=1)
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['date', 'symbol', 'close', 'signal_A', 'signal_B', 'signal_C', 'eligible'])
        prices = {'ASSET_A': 100.0, 'ASSET_B': 80.0}
        for day in days:
            common = rng.normal(0, .006)
            for symbol in prices:
                prices[symbol] *= float(np.exp(common + rng.normal(0, .007)))
                flags = (rng.random(3) < [.10, .15, .20]).astype(int).tolist()
                writer.writerow([day.isoformat(), symbol, f'{prices[symbol]:.8f}', *flags, 1])


def make_protocol(args, data):
    candidates = tuple(x.strip() for x in (args.validation_candidates or '').split(',') if x.strip()) or data.candidates
    if not candidates or len(set(candidates)) != len(candidates) or not set(candidates) <= set(data.candidates):
        raise ValueError('validation candidates must be unique names present in the input CSV')
    protocol = {
        'schema_version': 1,
        'data_kind': args.data_kind,
        'candidates': list(candidates),
        'phase1_start': args.start,
        'phase1_end': args.end,
        'wfa_start': month_shift(args.end, -args.train_months),
        'train_months': args.train_months,
        'oos_months': args.oos_months,
        'step_months': args.step_months,
        'gap_days': args.gap_days,
        'cooldown_sessions': args.cooldown_sessions,
        'min_signals': args.min_signals,
        'label_horizon_sessions': 3,
        'block_days': args.block_days,
        'seed': args.seed,
        'discovery_input_sha256': file_digest(args.input),
        'calendar_sha256': file_digest(args.calendar) if args.calendar else None,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'evidence_status': 'historical_research; untouched holdout and causal input signals are not certified',
    }
    protocol['protocol_sha256'] = digest(protocol)
    return protocol


def read_protocol(path):
    value = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    claimed = value.pop('protocol_sha256', None)
    if not claimed or digest(value) != claimed:
        raise ValueError('protocol hash mismatch; changing settings requires a new named experiment')
    if value.get('schema_version') != 1 or value.get('label_horizon_sessions') != 3:
        raise ValueError('unsupported protocol version or label horizon')
    if value['gap_days'] < 0 or value['train_months'] < 1 or value['step_months'] < value['oos_months']:
        raise ValueError('invalid protocol windows')
    value['protocol_sha256'] = claimed
    return value


def validation(data, protocol, args):
    end = args.end or (data.calendar[-1] + timedelta(days=1)).isoformat()
    result = walk_forward(data, candidates=protocol['candidates'],
                          start=protocol['wfa_start'], end=end,
                          train_months=protocol['train_months'], oos_months=protocol['oos_months'],
                          step_months=protocol['step_months'], gap_days=protocol['gap_days'],
                          cooldown_sessions=protocol['cooldown_sessions'], min_signals=protocol['min_signals'],
                          block_days=protocol['block_days'], seed=protocol['seed'],
                          bootstrap_samples=args.bootstrap_samples)
    for fold in result['folds']:
        if fold['oos_start'] < protocol['phase1_end']:
            raise ValueError('validation window overlaps the discovery period')
    return result


def percent(value):
    return '未计算' if value is None else f'{value:.2%}'


def metric_line(value):
    return (f"有效样本 {value.get('valid_count', 0)}；上涨 {value.get('wins', 0)}；"
            f"三日上涨比例 {percent(value.get('win_rate'))}；"
            f"未触发基准 {percent(value.get('baseline_rate'))}；"
            f"比例差 {percent(value.get('lift'))}")


def write_report(output, result):
    lines = ['# 任意策略信号的三日上涨比例验证', '',
             f"数据性质：**{result['data_kind']}**。模式：{result['mode']}。", '',
             '这里的上涨比例是信号日收盘到第3个市场交易日收盘的观察统计，',
             '不包含成交、资金、成本或仓位模拟，不等于净交易胜率或未来上涨承诺。', '',
             '候选信号由外部输入；引擎不生成策略，也不能证明输入信号未使用未来信息。', '']
    if result['data_kind'] == 'synthetic':
        lines += ['**本报告只验证软件流程，不是任何真实市场或策略的效果证据。**', '']
    first = result.get('phase1')
    if first:
        lines += ['## Phase 1：回顾性探索', '',
                  f"样本内入选候选：`{first.get('selected_candidate')}`。样本内选优不作为显著性验收。", '',
                  '| 候选 | 信号数 | 有效观察 | 三日上涨比例 | 未触发基准 | 差值 |',
                  '|---|---:|---:|---:|---:|---:|']
        for item in first['candidates']:
            lines.append(f"| {item['candidate']} | {item['signal_count']} | {item['valid_count']} | "
                         f"{percent(item['win_rate'])} | {percent(item['baseline_rate'])} | {percent(item['lift'])} |")
        pbo = first.get('pbo') or {}
        lines += ['', f"CSCV PBO：{percent(pbo.get('pbo'))}；状态：{pbo.get('status', 'not_computed')}。",
                  'PBO是指定候选族的回顾性选优风险诊断，不能替代未见数据验证。', '']
    second = result.get('walk_forward')
    if second:
        lines += ['## Phase 2：滚动历史验证', '', metric_line(second['aggregate']), '',
                  '| 训练开始 | 训练结束（不含） | OOS开始 | OOS结束（不含） | 入选候选 | 有效观察 | 三日上涨比例 |',
                  '|---|---|---|---|---|---:|---:|']
        for fold in second['folds']:
            oos = fold.get('oos') or {}
            lines.append(f"| {fold['train_start']} | {fold['train_end']} | {fold['oos_start']} | {fold['oos_end']} | "
                         f"{fold.get('selected_candidate')} | {oos.get('valid_count', 0)} | {percent(oos.get('win_rate'))} |")
        ci = second['aggregate'].get('bootstrap') or {}
        lines += ['', f"日期块重采样的比例差区间：[{percent(ci.get('ci_low'))}, {percent(ci.get('ci_high'))}]；"
                  f"状态：{ci.get('status', 'not_computed')}。",
                  '这是给定数据、候选族和区块长度下的描述性区间；未自动作出“策略有效”裁决。', '']
    lines += ['## 审计边界', '',
              '- `result.json`保留信号、端点、缺失及未成熟明细；缺失不算输，也不当作零收益。',
              '- 冷却只抑制同一标的相邻信号，不证明跨标的或时间上的统计独立。',
              '- 未触发基准使用同一 eligible 条件的原始0信号日期，未做因果识别或风险匹配。',
              '- 日历未显式提供时取CSV日期并集；整个市场缺失的交易日无法自动发现。',
              '- 所有日期区间为左闭右开。OOS标签不会使用本折结束之后的价格。',
              '- 配置哈希用于复现和发现误改，不是外部时间戳，也不能阻止研究者反复查看历史。', '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


def parser():
    root = argparse.ArgumentParser(description='Validate arbitrary external signals by their next-3-session upward frequency.')
    sub = root.add_subparsers(dest='command', required=True)
    for name in ('demo', 'phase1', 'walk-forward'):
        command = sub.add_parser(name)
        command.add_argument('--output', required=True)
        command.add_argument('--bootstrap-samples', type=int, default=500)
        if name != 'demo':
            command.add_argument('--input', required=True)
            command.add_argument('--calendar')
        if name == 'walk-forward':
            command.add_argument('--protocol', required=True)
            command.add_argument('--end', help='exclusive validation end; default one day after the last calendar date')
        else:
            command.add_argument('--start', required=name == 'phase1')
            command.add_argument('--end', required=name == 'phase1', help='exclusive discovery end')
            command.add_argument('--data-kind', choices=('user-supplied', 'synthetic'), default='user-supplied')
            command.add_argument('--validation-candidates', help='comma-separated preselected candidate fence; default all')
            command.add_argument('--train-months', type=int, default=18)
            command.add_argument('--oos-months', type=int, default=3)
            command.add_argument('--step-months', type=int, default=3)
            command.add_argument('--gap-days', type=int, default=5)
            command.add_argument('--cooldown-sessions', type=int, default=3)
            command.add_argument('--min-signals', type=int, default=10)
            command.add_argument('--block-days', type=int, default=5)
            command.add_argument('--seed', type=int, default=42)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.bootstrap_samples < 20 or args.bootstrap_samples > 10000:
            raise ValueError('bootstrap-samples must be between 20 and 10000')
        output = new_output(args.output)
        if args.command == 'demo':
            args.input = output / 'synthetic.csv'
            args.calendar = None
            args.data_kind = 'synthetic'
            args.start, args.end = '2019-01-01', '2020-07-01'
            create_demo(args.input)
        data = load_csv(args.input, calendar_path=args.calendar)
        result = {'mode': args.command, 'data_kind': getattr(args, 'data_kind', None),
                  'input_sha256': file_digest(args.input),
                  'calendar_sha256': file_digest(args.calendar) if args.calendar else None,
                  'version': '0.1.0', 'actual_trade_simulation': False}
        if args.command in ('phase1', 'demo'):
            if args.train_months < 1 or args.oos_months < 1 or args.step_months < args.oos_months:
                raise ValueError('positive train/OOS months and non-overlapping OOS steps are required')
            result['phase1'] = phase1(data, start=args.start, end=args.end,
                                      cooldown_sessions=args.cooldown_sessions, min_signals=args.min_signals,
                                      bootstrap_samples=args.bootstrap_samples, block_days=args.block_days, seed=args.seed)
            protocol = make_protocol(args, data)
            save_json(output / 'protocol.json', protocol)
        else:
            protocol = read_protocol(args.protocol)
            result['data_kind'] = protocol['data_kind']
            if (file_digest(args.calendar) if args.calendar else None) != protocol['calendar_sha256']:
                raise ValueError('calendar differs from the discovery protocol; create a new experiment')
            save_json(output / 'protocol.json', protocol)
        if args.command in ('walk-forward', 'demo'):
            if args.command == 'demo':
                args.end = None
            result['walk_forward'] = validation(data, protocol, args)
        result['protocol_sha256'] = protocol['protocol_sha256']
        save_json(output / 'result.json', result)
        write_report(output, result)
        print(json.dumps({'status': 'completed', 'mode': args.command, 'data_kind': result['data_kind'],
                          'report': str(output / 'report.md'), 'result': str(output / 'result.json')}, ensure_ascii=False))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f'Validation did not complete: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
