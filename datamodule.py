import os
import pytorch_lightning as pl
import numpy as np
import pandas as pd

from tqdm import tqdm

from torch.utils.data import DataLoader

from dataset import TabularDataset

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

from feature_sets import *

from sklearn.preprocessing import normalize

from params import set_input_size


class Constants():
    def __init__(self, device="cpu") -> None:
        self._pitch_width_norm = 105.0
        self._pitch_height_norm = 68.0
        self._player_max_vel_norm = 1.05
        self._goal_pos = np.array([self._pitch_width_norm/2, 0.0])
        self._goal_width = np.array([0, 14.02])
        self._i_delta = 6
        self._player_decay = 0.4

        self.pos_norm_factor = torch.tensor([[self._pitch_width_norm/2], [self._pitch_height_norm/2]], device=device).transpose(1,0)
        self.vel_norm_factor = torch.tensor([self._player_max_vel_norm], device=device)
        self.goal_pos1 = torch.tensor(self._goal_pos - self._goal_width/2, device=device)
        self.goal_pos2 = torch.tensor(self._goal_pos + self._goal_width/2, device=device)
        self.norm_goal_pos1 = torch.tensor(self._goal_pos - self._goal_width/2, device=device) / self.pos_norm_factor
        self.norm_goal_pos2 = torch.tensor(self._goal_pos + self._goal_width/2, device=device) / self.pos_norm_factor

    def pitchHalfLength(self):
        return self._pitch_width_norm/2
    
    def ourTeamGoalPos(self):
        pass


class DataModule(pl.LightningDataModule):
    '''
    Defines the DataModule. This pytorch lightning class contains all the required handling for the datasets: train, test, and validation.
    This class is used to ease the implementation and handling of all datasets by the training module.

    Attributes
    ----------
    train_dataset : TabularDataset
        A Dataset derived class to handle tabluar data during training.
    test_dataset : TabularDataset
        A Dataset derived class to handle tabluar data during testing.
    val_dataset : TabularDataset
        A Dataset derived class to handle tabluar data during validation.
    '''
    def __init__(
            self, 
            feature_set="fset1", 
            normalize_dset=False,
            normalize_evaluation=False,            
            batch_size=100, 
            eval_batch_size=100, 
            num_workers=4, 
            dataset_path="./db/pass-generator-dataset",
            dataset_nader_path="./db/cyrus-dataset",
            debug=False,
        ) -> None:
        super().__init__()

        self.num_workers = num_workers
        self.normalize_dset = normalize_dset
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.normalize_evaluation = normalize_evaluation
        action_dfs = []
        wm_dfs = []
        self.feature_set = feature_set
        self.constants = Constants()

        print("[DATAMODULE] Reading dataframes WORLDMODEL ACTION_DF")
        for unum in tqdm(range(1, 12)):
            action_file_path = os.path.join(dataset_path, "player_{}_action_dframe.csv".format(unum))
            wm_file_path = os.path.join(dataset_path, "player_{}_worldmodel_dframe.csv".format(unum))
            if os.path.exists(action_file_path) and os.path.exists(wm_file_path):
                try:
                    action_df = pd.read_csv(action_file_path, index_col=False, header=None)
                    wm_df = pd.read_csv(wm_file_path, index_col=False, header=None)

                    action_dfs.append(action_df)
                    wm_dfs.append(wm_df)
                except pd.errors.EmptyDataError:
                    print('empty data', action_file_path, wm_file_path)
        
        print("[DATAMODULE] Reading dataframes BEST PASS SELECTOR")
        list_csvs = [] 
        for dirpath, dirname, filenames in os.walk(dataset_nader_path):
            for filename in tqdm(filenames):
                if '.csv' in filename:
                    filepath = os.path.join(dirpath, filename)
                    unum = filename.split('_')[1]
                    df = pd.read_csv(filepath, index_col=False)
                    df['self_unum'] = [ unum for _ in range(df.shape[0]) ]
                    list_csvs.append(df)
        
        assert len(action_dfs) > 0 and len(wm_dfs) > 0, "Missing action dataframes."
        assert len(wm_dfs) > 0, "Missing wm dataframes."
        assert len(action_dfs) == len(wm_dfs), "Number of action and wm dataframes do not match."

        merged_action_df = pd.concat(action_dfs)
        merged_world_model_df = pd.concat(wm_dfs)

        if debug:
            merged_action_df = merged_action_df
            merged_world_model_df = merged_world_model_df
        #
        # use less memory
        del action_dfs
        del wm_dfs

        bps_df = pd.concat(list_csvs) # best pass selector
        del list_csvs

        bps_df = bps_df.drop(columns=['Unnamed: 782'])
        
        bps_df['timehash'] = bps_df['timehash'].astype(np.int64)
        bps_df['cycle'] = (bps_df['cycle']).astype(np.int64)
        bps_df['self_unum'] = bps_df['self_unum'].astype(np.int64)

        # merged_action_df = merged_action_df.iloc[:int(merged_action_df.shape[0] * 0.1)]
        # merged_world_model_df = merged_world_model_df.iloc[:int(merged_world_model_df.shape[0] * 0.1)]

        action_csv_columns = [
            "timehash",             #drop
            "cycle",                #drop
            "self_unum",            #drop
            "M_chain_count",        #drop
            "action_evaluation",    #drop
            "action_risk",          #drop
            "action_type",          #1st drop
            "series_index",         #1nd drop
            "s1_spend_time",        #drop
            "action_target_x",
            "action_target_y",
            "action_target_unum",
            "ball_holder_unum",     #drop
            "ball_holder_pos_x",    #drop
            "ball_holder_pos_y"     #drop
        ]

        worldmodel_csv_columns = [
            "timehash",
            "cycle",
            "self_unum",
            "ball_pos_x",
            "ball_pos_y",
            "ball_vel_x",
            "ball_vel_y",
        ]
        
        unum_fix = []
        for unum in range(1, 12):
            worldmodel_csv_columns.append("tm_player_%d_unum"%unum)
            worldmodel_csv_columns.append("tm_player_%d_pos_x"%unum)
            worldmodel_csv_columns.append("tm_player_%d_pos_y"%unum)
            worldmodel_csv_columns.append("tm_player_%d_vel_x"%unum)
            worldmodel_csv_columns.append("tm_player_%d_vel_y"%unum)
            worldmodel_csv_columns.append("tm_player_%d_body"%unum)
            unum_fix.append(("tm_player_%d_unum"%unum, unum))
        
        for unum in range(1, 12):
            worldmodel_csv_columns.append("opp_player_%d_unum"%unum)
            worldmodel_csv_columns.append("opp_player_%d_pos_x"%unum)
            worldmodel_csv_columns.append("opp_player_%d_pos_y"%unum)
            worldmodel_csv_columns.append("opp_player_%d_vel_x"%unum)
            worldmodel_csv_columns.append("opp_player_%d_vel_y"%unum)
            worldmodel_csv_columns.append("opp_player_%d_body"%unum)
            unum_fix.append(("opp_player_%d_unum"%unum, unum))
            
        merged_action_df.columns = action_csv_columns
        merged_world_model_df.columns = worldmodel_csv_columns
        
        merged_action_df = merged_action_df.drop(["action_type", "series_index"], axis=1, index=None)

        wm_bps_intersect = np.zeros(merged_world_model_df.shape[0], dtype=bool)
        for hash in set(bps_df.timehash):
            wm_bps_intersect |= (
                (merged_world_model_df.timehash == hash) \
                & (merged_world_model_df.cycle.isin(set(bps_df[bps_df.timehash == hash].cycle)))
            ).to_numpy()
        
        merged_world_model_df = merged_world_model_df[wm_bps_intersect]
        merged_action_df = merged_action_df[wm_bps_intersect]

        self.action_keys = [key for key in action_csv_columns if key != "action_evaluation"]
        self.action_eval_keys = ["action_evaluation", "action_risk"]

        action_eval_df = merged_action_df[self.action_eval_keys]
        dropped_action_df = merged_action_df.drop(self.action_eval_keys, axis=1)

        merged_world_model_df = merged_world_model_df.values
        dropped_action_df = dropped_action_df.values
        action_eval_df = action_eval_df.values

        if self.normalize_evaluation:
            action_eval_df = normalize(action_eval_df)

        indexes = np.array([i for i in range(dropped_action_df.shape[0])])
        train_split, test_split = train_test_split(indexes, test_size=0.3, random_state=199)
        test_split, val_split = train_test_split(test_split, test_size=0.5, random_state=199)

        world_model_train = merged_world_model_df[train_split]
        world_model_test = merged_world_model_df[test_split]
        world_model_val = merged_world_model_df[val_split]
        del merged_world_model_df

        action_train = dropped_action_df[train_split]
        action_test = dropped_action_df[test_split]
        action_val = dropped_action_df[val_split]
        del dropped_action_df

        eval_train = action_eval_df[train_split]
        eval_test = action_eval_df[test_split]
        eval_val = action_eval_df[val_split]
        action_eval_shape = action_eval_df.shape
        del action_eval_df

        stratification_clusters = np.zeros(action_eval_shape)[:,:1]
        cluster_train = stratification_clusters[train_split]
        cluster_test = stratification_clusters[test_split]
        cluster_val = stratification_clusters[val_split]

        if self.feature_set == "fset_1":
            fset_fn = fset_1
            processed_wm_train, action_train = get_featureset(self.constants, world_model_train, action_train, normalize=self.normalize_dset, fset_fn=fset_fn)
            processed_wm_test, action_test = get_featureset(self.constants, world_model_test, action_test, normalize=self.normalize_dset, fset_fn=fset_fn)
            processed_wm_val, action_val = get_featureset(self.constants, world_model_val, action_val, normalize=self.normalize_dset, fset_fn=fset_fn)
            processed_wm_train = processed_wm_train.cpu().numpy()
            processed_wm_test = processed_wm_test.cpu().numpy()
            processed_wm_val = processed_wm_val.cpu().numpy()
        elif self.feature_set == "fset_2":
            fset_fn = fset_2
            processed_wm_train, action_train = get_featureset(self.constants, world_model_train, action_train, normalize=self.normalize_dset, fset_fn=fset_fn)
            processed_wm_test, action_test = get_featureset(self.constants, world_model_test, action_test, normalize=self.normalize_dset, fset_fn=fset_fn)
            processed_wm_val, action_val = get_featureset(self.constants, world_model_val, action_val, normalize=self.normalize_dset, fset_fn=fset_fn)
            processed_wm_train = processed_wm_train.cpu().numpy()
            processed_wm_test = processed_wm_test.cpu().numpy()
            processed_wm_val = processed_wm_val.cpu().numpy()
        elif self.feature_set == "positional_fset":
            fset_fn = positional_fset
            processed_wm_train, action_train = get_featureset(self.constants, world_model_train, action_train, normalize=self.normalize_dset, fset_fn=fset_fn)
            processed_wm_test, action_test = get_featureset(self.constants, world_model_test, action_test, normalize=self.normalize_dset, fset_fn=fset_fn)
            processed_wm_val, action_val = get_featureset(self.constants, world_model_val, action_val, normalize=self.normalize_dset, fset_fn=fset_fn)
            processed_wm_train = processed_wm_train.cpu().numpy()
            processed_wm_test = processed_wm_test.cpu().numpy()
            processed_wm_val = processed_wm_val.cpu().numpy()
        elif self.feature_set == "fset2+positional_fset":
            fset_fn = fset_2
            processed_wm_train, _ = get_featureset(self.constants, world_model_train, action_train, normalize=self.normalize_dset, fset_fn=fset_fn)
            processed_wm_test, _ = get_featureset(self.constants, world_model_test, action_test, normalize=self.normalize_dset, fset_fn=fset_fn)
            processed_wm_val, _ = get_featureset(self.constants, world_model_val, action_val, normalize=self.normalize_dset, fset_fn=fset_fn)
            fset_fn = positional_fset
            processed_wm_train2, action_train = get_featureset(self.constants, world_model_train, action_train, normalize=self.normalize_dset, fset_fn=fset_fn)
            processed_wm_test2, action_test = get_featureset(self.constants, world_model_test, action_test, normalize=self.normalize_dset, fset_fn=fset_fn)
            processed_wm_val2, action_val = get_featureset(self.constants, world_model_val, action_val, normalize=self.normalize_dset, fset_fn=fset_fn)
            processed_wm_train = processed_wm_train.cpu().numpy()
            processed_wm_test = processed_wm_test.cpu().numpy()
            processed_wm_val = processed_wm_val.cpu().numpy()
            processed_wm_train2 = processed_wm_train2.cpu().numpy()
            processed_wm_test2 = processed_wm_test2.cpu().numpy()
            processed_wm_val2 = processed_wm_val2.cpu().numpy()

            processed_wm_train = np.concatenate([processed_wm_train, processed_wm_train2], axis=1)
            processed_wm_test = np.concatenate([processed_wm_test, processed_wm_test2], axis=1)
            processed_wm_val = np.concatenate([processed_wm_val, processed_wm_val2], axis=1)

        bps_df_dict = {}
        for cycle in set(bps_df.cycle):
            if cycle not in bps_df_dict:
                bps_df_dict[cycle] = {}
            
            cycle_df = bps_df[bps_df.cycle==cycle]
            for hash in set(cycle_df.timehash):
                if hash not in bps_df_dict[cycle]:
                    bps_df_dict[cycle][hash] = {}

                bps_df_dict[cycle][hash] = cycle_df[cycle_df.timehash==hash].drop(["timehash", "out_unum", "out_desc"], axis=1).to_numpy().astype(np.float32)[0]
        del bps_df
        
        set_input_size(processed_wm_train.shape[1]) # define neural_net input_sizes

        self.action_train_dataset = TabularDataset(
            action_train, 
            eval_train, 
            world_model_train, 
            processed_wm_train, 
            bps_df=bps_df_dict,
            fset_type=self.feature_set,
            clusters=cluster_train,
        )
        self.action_test_dataset = TabularDataset(
            action_test,
            eval_test, 
            world_model_test, 
            processed_wm_test, 
            bps_df=bps_df_dict,
            fset_type=self.feature_set,
            clusters=cluster_test,
        )
        self.action_val_dataset = TabularDataset(
            action_val, 
            eval_val, 
            world_model_val, 
            processed_wm_val, 
            bps_df=bps_df_dict,
            fset_type=self.feature_set,
            clusters=cluster_test,
        )

    def train_dataloader(self) -> DataLoader:
        '''
        Defines the train_dataloader get method, used by the pl.Trainer.fit function.

        Returns
        -------
        DataLoader to handle train_dataset.
        '''
        return DataLoader(self.action_train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        '''
        Defines the val_dataloader get method, used by the pl.Trainer.fit function.

        Returns
        -------
        DataLoader to handle val_dataset.
        '''
        return DataLoader(self.action_val_dataset, batch_size=self.eval_batch_size, num_workers=self.num_workers, persistent_workers=True, drop_last=True)

    def test_dataloader(self) -> DataLoader:
        '''
        Defines the test_dataloader get method, used by the pl.Trainer.fit function.

        Returns
        -------
        DataLoader to handle test_dataset.
        '''
        return DataLoader(self.action_test_dataset, batch_size=self.eval_batch_size, num_workers=self.num_workers, persistent_workers=True, drop_last=True)
