# @Time   : 2022/3/4
# @Author : Chen Yang
# @Email  : flust@ruc.edu.cn

"""
recbole_pjf.trainer
########################
"""

from collections import OrderedDict, defaultdict

import torch
import numpy as np
# from recbole.trainer import Trainer
from recbole.trainer import Trainer
from recbole.trainer.trainer import NCLTrainer
from recbole.utils import calculate_valid_score


from recbole.data.interaction import Interaction
from recbole_pjf.data.dataloader import RFullSortEvalDataLoader, RNegSampleEvalDataLoader
from recbole.utils import (
    ensure_dir,
    get_local_time,
    early_stopping,
    calculate_valid_score,
    dict2str,
    EvaluatorType,
    KGDataLoaderState,
    get_tensorboard,
    set_color,
    get_gpu_usage,
    WandbLogger,
)
from tqdm import tqdm

def map_to_zeroone(flo, max_positive, min_negative):
    if flo > 0.5:
        return 0.5 + 0.5 * (flo - 0.5) / (max_positive - 0.5)
    else:
        return 0.5 - 0.5 * (0.5 - flo) / (0.5 - min_negative)


class MultiDirectTrainer(Trainer):
    def __init__(self, config, model):
        super(MultiDirectTrainer, self).__init__(config, model)
        setattr(self.eval_collector.register,'rec.items', True)

    @torch.no_grad()
    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False):
        # Overall look at the bilateral recommendation task

        if load_best_model:
            checkpoint_file = model_file or self.saved_model_file
            # checkpoint = torch.load(checkpoint_file, map_location = self.device)
            checkpoint = torch.load(checkpoint_file, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.load_other_parameter(checkpoint.get("other_parameter"))
            message_output = "Loading model structure and parameters from {}".format(
                checkpoint_file
            )
            self.logger.info(message_output)

        self.model.eval()

        if isinstance(eval_data[0], RFullSortEvalDataLoader):
            eval_func = self._full_sort_batch_eval
            self.item_tensor = eval_data[0].dataset.get_item_feature().to(self.device)
        else:
            eval_func = self._neg_sample_batch_eval
        struct_g = self.collect_bilateral_info(eval_data[0], eval_func, show_progress, 1)
        test_result_g = self.evaluator.evaluate(struct_g)
        if not self.config["single_spec"]:
            test_result_g = self._map_reduce(test_result_g, num_sample)

        self.config.change_direction()
        if isinstance(eval_data[1], RFullSortEvalDataLoader):
            eval_func = self._full_sort_batch_eval
            self.item_tensor = eval_data[1].dataset.get_item_feature().to(self.device)
        else:
            eval_func = self._neg_sample_batch_eval
        struct_j = self.collect_bilateral_info(eval_data[1], eval_func, show_progress, 2)

        test_result_j = self.evaluator.evaluate(struct_j)
        if not self.config["single_spec"]:
            test_result_j = self._map_reduce(test_result_j, num_sample)
        self.config.change_direction()

        # item_score_list = struct_g.get("rec.score").tolist()
        # real_item_score_list = [map_to_zeroone(x, max(item_score_list), min(item_score_list)) for i, x in
        #                         enumerate(item_score_list) if i % 2 == 0]
        # predict_label = [1 if x > 0.5 else 0 for x in real_item_score_list]
        # real_labels = [1 for _ in range(len(real_item_score_list))]
        #
        # job_score_list = struct_j.get("rec.score").tolist()
        # real_job_score_list = [map_to_zeroone(x, max(job_score_list), min(job_score_list)) for i, x in
        #                        enumerate(job_score_list) if i % 2 == 0]
        # predict_job_label = [1 if x > 0.5 else 0 for x in real_job_score_list]
        #
        # item_id_list = eval_data[0].dataset.inter_feat.interaction.get('user_id').tolist()
        # job_id_list = eval_data[0].dataset.inter_feat.interaction.get('job_id').tolist()
        # # 打开文件写入
        # columns = ["user_id", "job_id", "for_user_score", "user_real_labels", "for_user_predict_label", "for_job_score", "for_job_lable"]
        # with open("output.txt", "w") as file:
        #     file.write("\t".join(columns) + "\n")
        #     # 使用 zip 将四个列表按行组合
        #     for a, b, c, d, e, f, g in zip(item_id_list, job_id_list, real_item_score_list, real_labels, predict_label, real_job_score_list, predict_job_label):
        #         # 写入一行，列之间用空格分隔
        #         file.write(f"{a}\t{b}\t{c}\t{d}\t{e}\t{f}\t{g}\n")

        return test_result_g, test_result_j

    def _full_sort_batch_eval(self, batched_data):
        interaction, history_index, positive_u, positive_i = batched_data

        inter_len = len(interaction)
        new_inter = interaction.to(self.device).repeat_interleave(self.tot_item_num)
        batch_size = len(new_inter)
        new_inter.update(self.item_tensor.repeat(inter_len))
        if batch_size <= self.test_batch_size:
            scores = self.model.predict(new_inter)
        else:
            scores = self._spilt_predict(new_inter, batch_size)

        scores = scores.view(-1, self.tot_item_num)
        scores[:, 0] = -np.inf
        if history_index is not None:
            scores[history_index] = -np.inf
        return interaction, scores, positive_u, positive_i

    def collect_bilateral_info(self, eval_data, eval_func, show_progress, direct = 0):
        if self.config['eval_type'] == EvaluatorType.RANKING:
            self.tot_item_num = eval_data.dataset.item_num

        iter_data = (
            tqdm(
                eval_data,
                total=len(eval_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", "pink"),
            )
            if show_progress
            else eval_data
        )

        positive_u_list = []

        num_sample = 0
        for batch_idx, batched_data in enumerate(iter_data):
            num_sample += len(batched_data)
            interaction, scores, positive_u, positive_i = eval_func(batched_data[:4])
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(
                    set_color("GPU RAM: " + get_gpu_usage(self.device), "yellow")
                )

            self.eval_collector.eval_batch_collect(
                scores, interaction, positive_u, positive_i
            )

        self.eval_collector.model_collect(self.model)
        struct = self.eval_collector.get_data_struct()
        return struct

    def _valid_epoch(self, valid_data, show_progress=False):
        valid_result_all = self.evaluate(valid_data, load_best_model=False, show_progress=show_progress)
        valid_g_score = calculate_valid_score(valid_result_all[0], self.valid_metric)
        valid_j_score = calculate_valid_score(valid_result_all[1], self.valid_metric)
        valid_score = (valid_g_score + valid_j_score) / 2

        valid_result = OrderedDict()
        valid_result['For geek'] = valid_result_all[0]
        valid_result['\nFor job'] = valid_result_all[1]
        return valid_score, valid_result


    @torch.no_grad()
    def evaluate_per_user(self, eval_data, show_progress=False):
        """Return per-user full-sort scores and positive-item sets.

        Uses the same inference path as ``evaluate`` (full-sort, history masking,
        spilt_predict) but collects per-user results instead of aggregating.

        Returns: tuple of two dicts — (geek_per_user, job_per_user)
          Each dict maps user_id → {"scores": np.array(n_items,), "positives": set}
        """
        self.model.eval()
        results = []

        for direction_idx in (0, 1):
            loader = eval_data[direction_idx]

            if not isinstance(loader, RFullSortEvalDataLoader):
                raise TypeError(
                    "evaluate_per_user only supports RFullSortEvalDataLoader "
                    "(full-sort ranking). Got {}".format(type(loader))
                )

            self.tot_item_num = loader.dataset.item_num
            item_feat = loader.dataset.get_item_feature()
            # Some models (PJFNN, BPJFNN, etc.) need text features (idoc)
            # which are stored separately from get_item_feature().
            if loader.dataset.item_doc is not None:
                item_feat.update(loader.dataset.item_doc)
            self.item_tensor = item_feat.to(self.device)
            uid_field = loader.dataset.uid_field

            per_user_scores = defaultdict(list)
            per_user_positives = defaultdict(set)

            iter_data = (
                tqdm(loader, total=len(loader), ncols=100,
                     desc=set_color("Per-user eval", "pink"))
                if show_progress else loader
            )

            for batched_data in iter_data:
                # Reuse the exact same batch-eval logic (full-sort + history mask)
                interaction, scores, positive_u, positive_i = self._full_sort_batch_eval(
                    batched_data[:4]
                )
                scores_np = scores.cpu().numpy()
                pos_u = positive_u.cpu().numpy()
                pos_i = positive_i.cpu().numpy()

                user_ids = interaction[uid_field].cpu().tolist()

                for batch_idx, uid in enumerate(user_ids):
                    per_user_positives[uid] |= set(pos_i[pos_u == batch_idx].tolist())
                    per_user_scores[uid].append(scores_np[batch_idx])

            # Average score vectors if a user appeared in multiple batches
            per_user = {}
            for uid in per_user_scores:
                avg_scores = np.array(per_user_scores[uid]).mean(axis=0)
                # Re-apply padding mask at position 0 (belt-and-suspenders)
                avg_scores[0] = -np.inf
                per_user[uid] = {
                    "scores": avg_scores.tolist(),
                    "positives": sorted(per_user_positives[uid]),
                }

            results.append(per_user)

            if direction_idx == 0 and self.config.final_config_dict.get("biliteral", False):
                self.config.change_direction()

        # Restore original direction
        if self.config.final_config_dict.get("biliteral", False):
            self.config.change_direction()

        return results[0], results[1]

class MultiDirectNCLTrainer(NCLTrainer):
    """NCL 模型的双边评估 Trainer，继承 NCLTrainer 的 E-step 逻辑 + MultiDirectTrainer 的双边评估"""

    def __init__(self, config, model):
        super(MultiDirectNCLTrainer, self).__init__(config, model)
        setattr(self.eval_collector.register, 'rec.items', True)

    @torch.no_grad()
    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False):
        if load_best_model:
            checkpoint_file = model_file or self.saved_model_file
            checkpoint = torch.load(checkpoint_file, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.load_other_parameter(checkpoint.get("other_parameter"))
            message_output = "Loading model structure and parameters from {}".format(
                checkpoint_file
            )
            self.logger.info(message_output)

        self.model.eval()

        if isinstance(eval_data[0], RFullSortEvalDataLoader):
            eval_func = self._full_sort_batch_eval
            self.item_tensor = eval_data[0].dataset.get_item_feature().to(self.device)
        else:
            eval_func = self._neg_sample_batch_eval
        struct_g = self.collect_bilateral_info(eval_data[0], eval_func, show_progress, 1)
        test_result_g = self.evaluator.evaluate(struct_g)
        if not self.config["single_spec"]:
            test_result_g = self._map_reduce(test_result_g, num_sample)

        self.config.change_direction()
        if isinstance(eval_data[1], RFullSortEvalDataLoader):
            eval_func = self._full_sort_batch_eval
            self.item_tensor = eval_data[1].dataset.get_item_feature().to(self.device)
        else:
            eval_func = self._neg_sample_batch_eval
        struct_j = self.collect_bilateral_info(eval_data[1], eval_func, show_progress, 2)

        test_result_j = self.evaluator.evaluate(struct_j)
        if not self.config["single_spec"]:
            test_result_j = self._map_reduce(test_result_j, num_sample)
        self.config.change_direction()

        return test_result_g, test_result_j

    def _full_sort_batch_eval(self, batched_data):
        interaction, history_index, positive_u, positive_i = batched_data
        inter_len = len(interaction)
        new_inter = interaction.to(self.device).repeat_interleave(self.tot_item_num)
        batch_size = len(new_inter)
        new_inter.update(self.item_tensor.repeat(inter_len))
        if batch_size <= self.test_batch_size:
            scores = self.model.predict(new_inter)
        else:
            scores = self._spilt_predict(new_inter, batch_size)
        scores = scores.view(-1, self.tot_item_num)
        scores[:, 0] = -np.inf
        if history_index is not None:
            scores[history_index] = -np.inf
        return interaction, scores, positive_u, positive_i

    def collect_bilateral_info(self, eval_data, eval_func, show_progress, direct=0):
        if self.config['eval_type'] == EvaluatorType.RANKING:
            self.tot_item_num = eval_data.dataset.item_num
        iter_data = (
            tqdm(eval_data, total=len(eval_data), ncols=100,
                 desc=set_color(f"Evaluate   ", "pink"))
            if show_progress else eval_data
        )
        num_sample = 0
        for batch_idx, batched_data in enumerate(iter_data):
            num_sample += len(batched_data)
            interaction, scores, positive_u, positive_i = eval_func(batched_data[:4])
            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(
                    set_color("GPU RAM: " + get_gpu_usage(self.device), "yellow")
                )
            self.eval_collector.eval_batch_collect(
                scores, interaction, positive_u, positive_i
            )
        self.eval_collector.model_collect(self.model)
        struct = self.eval_collector.get_data_struct()
        return struct

    def _valid_epoch(self, valid_data, show_progress=False):
        valid_result_all = self.evaluate(valid_data, load_best_model=False, show_progress=show_progress)
        valid_g_score = calculate_valid_score(valid_result_all[0], self.valid_metric)
        valid_j_score = calculate_valid_score(valid_result_all[1], self.valid_metric)
        valid_score = (valid_g_score + valid_j_score) / 2
        valid_result = OrderedDict()
        valid_result['For geek'] = valid_result_all[0]
        valid_result['\nFor job'] = valid_result_all[1]
        return valid_score, valid_result

NCLTrainer = MultiDirectNCLTrainer  # 别名，使 get_trainer('GENERAL', 'NCL') 能匹配到
