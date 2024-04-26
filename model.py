from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from typing import Any, List, Dict

from feature_sets import *

from datamodule import Constants

import matplotlib.pyplot as plt

from scipy.stats import kendalltau, spearmanr

from train_cyrus import CyrusPassClassifier

act_fn_dict = {
    "relu": torch.nn.ReLU,
    "mish": torch.nn.Mish,
    "tanh": torch.nn.Tanh,
}

norm_layer_dict = {
    "batch": torch.nn.BatchNorm1d,
    "instance": torch.nn.InstanceNorm1d,
    None: torch.nn.Identity,
}

fset_dict = {
    "fset_1": fset_1,
    "fset_2": fset_2,
    "positional_fset": positional_fset,
}


class TargetPointGenerator(nn.Module):
    '''
    Sample pytorch MLP network, used to generate passes.

    Attributes
    ----------
    net : nn.Sequential
        MLP model.
    '''
    def __init__(self, input_size, output_size, bps_ckpt_path="./models/bps_model.ckpt") -> None:
        super().__init__()
        self.constants = Constants("cuda:0")

        self.bps_model = CyrusPassClassifier.load_from_checkpoint(
            bps_ckpt_path,
            strict=False,
        )
        self.bps_model.freeze()

        self.net = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, output_size),
            nn.Tanh(),
        )

    def get_generator_input(self, wm, raw_wm, bp_idx=None):
        bs = wm.shape[0]
        filtered_wm = raw_wm[:, self.constants._i_delta:]
        
        if bp_idx is None:
            tm_cartesian_pos = torch.stack(
                [torch.stack([filtered_wm[:, (i*6)+2], filtered_wm[:, (i*6)+3]]) for i in range(11)] #mudar para formatação gather
            ).permute(2,0,1) #x,y for each player
            tm_cartesian_pos /= self.constants.pos_norm_factor

            return wm, tm_cartesian_pos.reshape(bs, 22)
        else:
            tm_cartesian_pos = torch.gather(filtered_wm, 1, torch.stack([(bp_idx*6)+2, (bp_idx*6)+3], dim=1))
            tm_cartesian_pos /= self.constants.pos_norm_factor
            return wm, tm_cartesian_pos.reshape(bs, 2)

    def forward(self, x:torch.Tensor, raw_wm:torch.Tensor=None, bps_input:torch.Tensor=None, return_bps=False) -> torch.Tensor:
        '''
            Model's forward function.

            Parameters
            ----------
            x : torch.Tensor
                Tensor containing input data.

            Returns:
                Tensor containing the labeled result.
        '''

        bps_idx = torch.argmax(self.bps_model(bps_input), axis=1)

        generator_input, bps_pos = self.get_generator_input(x, raw_wm, bps_idx)

        delta = self.net(generator_input)
        target_point = (bps_pos + delta).float()
        if return_bps:
            return target_point, bps_pos
        else:
            return target_point


class PassEvaluator(nn.Module):
    '''
    Sample pytorch MLP network, used to generate passes.

    Attributes
    ----------
    net : nn.Sequential
        MLP model.
    '''
    def __init__(self, input_size, **hparams) -> None:
        super().__init__()

        self.hparams = hparams
        act_fn = hparams["act_fn"]
        norm_layer = hparams["norm_layer"]

        self.backbone = nn.Sequential(
            nn.Linear(input_size, 1024),
            norm_layer_dict[norm_layer](1024),
            act_fn_dict[act_fn](),
            nn.Linear(1024, 512),
            norm_layer_dict[norm_layer](512),
            act_fn_dict[act_fn](),
            nn.Linear(512, 256),
            norm_layer_dict[norm_layer](256),
            act_fn_dict[act_fn](),
            nn.Linear(256, 64),
            norm_layer_dict[norm_layer](64),
            act_fn_dict[act_fn](),
            nn.Linear(64, 32),
            norm_layer_dict[norm_layer](32),
            act_fn_dict[act_fn](),
        )

        self.eval_head = nn.Sequential(
            nn.Linear(32, 1),
        )


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        '''
            Model's forward function.

            Parameters
            ----------
            x : torch.Tensor
                Tensor containing input data.

            Returns:
                Tensor containing the labeled result.
        '''
        features = self.backbone(x)
        evaluation = self.eval_head(features)
        
        return evaluation


class SPO(pl.LightningModule):
    def __init__(
            self,
            hparams,
        ) -> None:
        '''
        Sample classifier training class. Derives from pl.LightingModule, which implements all the traning side tasks. Here you can create helper functions for anything, including training,
        testing, and validation steps.

        Attributes
        ----------
        model: TorchSampleNet
            Classification model.
        
        loss_fn: function
            Loss function that returs a torch.Tensor with the optimization graph.
        '''
        super().__init__()
    
        self.hparams.g_lr = 5.0e-4 if "g_lr" not in hparams else hparams["g_lr"]
        self.hparams.e_lr = 5.0e-4 if "e_lr" not in hparams else hparams["e_lr"]
        self.hparams.evaluator_act_fn = "relu" if "act_fn" not in hparams else hparams["evaluator"]["act_fn"]
        self.hparams.norm_layer = None if "act_fn" not in hparams else hparams["evaluator"]["norm_layer"]
        self.feature_set_function = "fset_2" if "feature_set_function" not in hparams else hparams["feature_set_function"]

        self.generator_evaluator_lambda = 1.0 if "evaluator_lambda" not in hparams["generator"] else hparams["generator"]["evaluator_lambda"]

        self.save_hyperparameters()

        self.generator = TargetPointGenerator(
            hparams["generator"]["input_size"], 
            hparams["generator"]["output_size"], 
            hparams["generator"]["bps_ckpt_path"],
        )

        self.evaluator = PassEvaluator(
            hparams["evaluator"]["input_size"], 
            act_fn=self.hparams.evaluator_act_fn,
            norm_layer=self.hparams.norm_layer,
        )

        self.get_feature_set = fset_dict[self.feature_set_function]
        self.evaluator_loss = nn.MSELoss()
        self.evaluator_val_loss = nn.MSELoss()

        self.constants = Constants("cuda:0")
  
    def get_evaluator_input(self, generated_action, wm):
        return wm
    
    def _risk_evaluator_train_step(self, optimizer_idx, wm, true_action, action_eval, action_risk):
        evaluator_input = self.get_evaluator_input(true_action, wm)

        predicted_risk = self.risk_evaluator(evaluator_input)

        evaluation_pred_loss = self.risk_evaluator_loss(predicted_risk, action_risk).mean()
        total_loss = evaluation_pred_loss

        self.log("risk_evaluator/pred_loss", float(evaluation_pred_loss))

        return total_loss
    
    def  _evaluator_train_step(self, optimizer_idx, wm, raw_wm, true_action, action_eval, action_risk):
        evaluator_input = self.get_evaluator_input(true_action, wm)

        pass_evaluation = self.evaluator(evaluator_input)

        evaluation_pred_loss = self.evaluator_loss(pass_evaluation, action_eval).mean()
        total_loss = evaluation_pred_loss

        self.log("evaluator/evaluator_pred_loss", float(evaluation_pred_loss))
        self.log("evaluator/total_loss", float(total_loss))

        return total_loss
    
    def _generator_train_step(self, optimizer_idx, wm, raw_wm, bps_row, true_action, action_eval, action_risk):
        bs = wm.shape[0]

        self.generated_action, best_player_pos = self.generator(wm, raw_wm, bps_row, return_bps=True)

        evaluator_input = self.get_feature_set(
            self.constants, 
            raw_wm, 
            self.generated_action.reshape((bs, 1, 2)),
            normalize=True, 
            normalize_candidate_point=False,
        ).float()

        final_eval = self.evaluator(evaluator_input)

        generator_loss = (1/(final_eval + 0.0001)).mean() # avoid dividing by 0
        total_loss = ( 
            self.generator_evaluator_lambda * generator_loss
        )

        self.log("generator/fake_pass_evaluation", float(generator_loss))
        self.log("generator/total_loss", float(total_loss))

        return total_loss

    def _joint_train_step(self, optimizer_idx, wm, raw_wm, bps_row, true_action, action_eval, action_risk):
        if optimizer_idx == 0: #generator training loop
            return self._generator_train_step(optimizer_idx, wm, raw_wm, bps_row, true_action, action_eval, action_risk)
        
        elif optimizer_idx == 1: #evaluator training loop
            return self._evaluator_train_step(optimizer_idx, wm, raw_wm, true_action, action_eval, action_risk)

    def training_step(self, batch:dict, batch_idx:int, optimizer_idx:int = 0) -> torch.Tensor:
        '''
        Implements the training_step for the model, here every operation required to train the model is used, inclusing logging functions and losses computation.
        This function will be called for each epoch, if multiple networks are used during training, this function will be called multiple times. Every network
        trained must be handled manually.

        Parameters
        ----------
        batch : dict
            Current epoch batch.
        
        batch_idx : int
            Integer containing the batch index.

        Retunrs
        -------
            Epoch loss.
        '''
        (
            wm,
            raw_wm,
            bps_row,
            true_action,
            action_eval,
            action_risk,
        ) = (
            batch["wm"], 
            batch["raw_wm"], 
            batch["bps_row"],
            batch["action"],
            batch["action_eval"],
            batch["action_risk"],
        )

        total_loss = self._joint_train_step(optimizer_idx, wm, raw_wm, bps_row, true_action, action_eval, action_risk)

        return total_loss
    
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        (
            wm,
            raw_wm,
            bps_row,
            true_action,
            action_eval,
        ) = (
            batch["wm"],
            batch["raw_wm"],
            batch["bps_row"],
            batch["action"],
            batch["action_eval"],
        )

        bs = wm.shape[0]
        evaluator_input = self.get_evaluator_input(true_action, wm)
        net_evaluation = self.evaluator(evaluator_input)
        evaluator_loss = self.evaluator_val_loss(net_evaluation, action_eval)
            
        generated_action = self.generator(wm, raw_wm, bps_row)
        fake_input = self.get_feature_set(
            self.constants, 
            raw_wm, 
            generated_action.reshape((bs, 1, 2)).detach(),
            normalize=True, 
            normalize_candidate_point=False,
        ).float()
        generated_action_eval = self.evaluator(fake_input)

        net_evaluation = net_evaluation.squeeze(1)
        action_eval = action_eval.squeeze(1)
        generated_action_eval = generated_action_eval.squeeze(1)

        top1_correct = torch.sum(torch.topk(net_evaluation, 1)[1] == torch.topk(action_eval, 1)[1])
        top5_correct = torch.sum(torch.topk(net_evaluation, 5)[1] == torch.topk(action_eval, 5)[1]) / 5
        top10_correct = torch.sum(torch.topk(net_evaluation, 10)[1] == torch.topk(action_eval, 10)[1]) / 10

        loss_dict = {
            "validation/evaluator/val_loss": evaluator_loss,
            "validation/evaluator/top1_correct": top1_correct,
            "validation/evaluator/top5_correct": top5_correct,
            "validation/evaluator/top10_correct": top10_correct,
            "generated_action_eval": generated_action_eval,
            "net_eval": net_evaluation,
        }

        return loss_dict

    def plot_passes(self, ball, target, pitch_scale=4):
        translated_ball = (ball * self.constants.pos_norm_factor).cpu().numpy()
        translated_target = (target * self.constants.pos_norm_factor).cpu().numpy()
        img = plt.imread("assets/sim2d_field.png")
        fig, ax = plt.subplots()
        xmax, ymax = self.constants._pitch_width_norm*pitch_scale, self.constants._pitch_height_norm*pitch_scale
        scale_delta = np.array([xmax/2.0, ymax/2.0])
        translated_ball = translated_ball * pitch_scale
        translated_target = translated_target * pitch_scale
        warp_fix = np.array([25.0, 12.0])
        
        dxdy = translated_target - translated_ball
        scaled_ball = translated_ball + scale_delta + warp_fix
        translated_target = translated_target + scale_delta

        ax.imshow(img, extent=[0, xmax+2*warp_fix[0], 0, ymax+2*warp_fix[1]])
        for i in range(scaled_ball.shape[0]):
            if target[i, 0] > 1 or target[i, 1] > 1 or target[i, 0] < -1 or target[i, 1] < -1:
                ax.arrow(scaled_ball[i, 0], scaled_ball[i, 1], dxdy[i, 0], dxdy[i, 1], head_width=5, head_length=5, color="r")
            else:
                ax.arrow(scaled_ball[i, 0], scaled_ball[i, 1], dxdy[i, 0], dxdy[i, 1], head_width=5, head_length=5, color="k")

        ax.scatter(scaled_ball[:, 0], scaled_ball[:, 1], marker=".", c="tab:orange")
        ax.axis("off")
        ax.set_xlim(0, xmax+2*warp_fix[0])
        ax.set_ylim(0, ymax+2*warp_fix[1])
        fig.tight_layout()
        fig.show()
        # self.logger.experiment.add_figure("passes", fig, global_step=self.global_step)
        
        plt.savefig(os.path.join(self.logger.log_dir, "pass.eps"), format="eps")
        ax.clear()
        fig.clear()
        plt.close(fig)

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT | List[EPOCH_OUTPUT]) -> None:
        loss_dict = {}
        for o in outputs:
            for k in o:
                if k not in loss_dict:
                    loss_dict[k] = [o[k]]
                else:
                    loss_dict[k].append(o[k])

        net_eval = torch.cat(loss_dict["net_eval"])
        generated_action_eval = torch.cat(loss_dict["generated_action_eval"])

        del loss_dict["net_eval"]
        del loss_dict["generated_action_eval"]

        for k in loss_dict:
            value = torch.stack(loss_dict[k]).float().mean()
            self.log(k, float(value))

        size = net_eval.shape[0]
        fake_selection_rate = (generated_action_eval > net_eval).sum()/size
        self.log("validation/fake_selection_rate", float(fake_selection_rate))

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        (
            wm,
            raw_wm,
            bps_row,
            true_action,
            action_eval,
            cycle,
            timehash,
        ) = (
            batch["wm"],
            batch["raw_wm"],
            batch["bps_row"],
            batch["action"],
            batch["action_eval"],
            batch["cycle"],
            batch["timehash"],
        )
        bs = wm.shape[0]
        net_evaluation = self.evaluator(
            self.get_evaluator_input(true_action, wm)
        )

        net_evaluation = net_evaluation.squeeze(1)
        action_eval = action_eval.squeeze(1)
        generated_action = self.generator(wm, raw_wm, bps_row)
        fake_input = self.get_feature_set(
            self.constants, 
            raw_wm, 
            generated_action.reshape((bs, 1, 2)).detach(),
            normalize=True, 
            normalize_candidate_point=False,
        ).float()
        generated_action_eval = self.evaluator(fake_input).squeeze(1)

        ball_pos = torch.gather(
            raw_wm, 
            1, 
            torch.tensor([2,3]).to(raw_wm.device).unsqueeze(0).repeat(raw_wm.shape[0], 1)
        ) / self.constants.pos_norm_factor

        loss_dict = {
            "net_eval": net_evaluation,
            "action_eval": action_eval,
            "true_action": true_action,
            "ball_pos": ball_pos,
            "generated_action": generated_action,
            "generated_action_eval": generated_action_eval,
            "cycle": cycle,
            "timehash": timehash,
        }

        return loss_dict
    
    def test_epoch_end(self, outputs) -> None:
        output_dict = {}
        for o in outputs:
            for k in o:
                if k not in output_dict:
                    output_dict[k] = [o[k]]
                else:
                    output_dict[k].append(o[k])

        net_eval = torch.cat(output_dict["net_eval"])
        action_eval = torch.cat(output_dict["action_eval"])
        generated_action_eval = torch.cat(output_dict["generated_action_eval"])
        cycle = torch.cat(output_dict["cycle"])
        timehash = torch.cat(output_dict["timehash"])
        ball = torch.cat(output_dict["ball_pos"])
        target_points = torch.cat(output_dict["generated_action"])

        rmse = torch.sqrt(F.mse_loss(net_eval, action_eval))
        size = action_eval.shape[0]
        eq_ranking = torch.sum(torch.topk(net_eval, size)[1] == torch.topk(action_eval, size)[1])/size

        fake_selection_rate = (generated_action_eval > net_eval).sum()/size

        net_eval = net_eval.cpu().numpy()
        action_eval = action_eval.cpu().numpy()
        fake_selection_rate = fake_selection_rate.cpu().numpy()
        kendall_tau = kendalltau(net_eval, action_eval, variant="b", alternative="greater")
        spearman_rho = spearmanr(net_eval, action_eval)

        self.plot_passes(ball[:50], target_points[:50])

        self.log("test/eq_rankings", float(eq_ranking))
        self.log("test/rmse", float(rmse))
        self.log("test/kendall_tau", float(kendall_tau.statistic))
        self.log("test/spearman_rho", float(spearman_rho.statistic))
        self.log("test/kendall_tau_pvalue", float(kendall_tau.pvalue))
        self.log("test/spearman_rho_pvalue", float(spearman_rho.pvalue))
        self.log("test/fake_selection_rate", float(fake_selection_rate))

    def configure_optimizers(self) -> torch.optim.Optimizer:
        '''
        Generate the optimizers used during traning, this function must return multiple optimizers if multiple networks are being trained. Read the docs for more
        information.

        Retunrs
        -------
            Model's optimizer.
        '''
        e_optimizer = torch.optim.Adam(self.evaluator.parameters(), lr=self.hparams.e_lr)
        g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.g_lr)

        return g_optimizer, e_optimizer    

