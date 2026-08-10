# @Time   : 2022/3/2
# @Author : Chen Yang
# @Email  : flust@ruc.edu.cn

"""
HBRBR 训练入口

用法:
    python run_hbrbr.py                       # 默认 HBRBR + zhilian
    python run_hbrbr.py -m HBRBR -d zhilian   # 显式指定
    python run_hbrbr.py -m HBRBR_NoBGNN       # 消融实验
"""

import argparse

from recbole_pjf.quick_start import run_recbole_pjf


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='HBRBR', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='zhilian', help='name of datasets')
    parser.add_argument('--config_files', type=str, default=None, help='config files')
    parser.add_argument('--prediction_head', '-p', type=str, default='elem_mlp',
                        help='prediction head: dot | bilinear | gated | elem_mlp | concat_mlp')

    args, _ = parser.parse_known_args()

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    config_dict = {'prediction_head': args.prediction_head}

    print(f"\n=== model={args.model}, dataset={args.dataset}, "
          f"prediction_head={args.prediction_head} ===")

    run_recbole_pjf(model=args.model, dataset=args.dataset,
                    config_file_list=config_file_list, config_dict=config_dict)
