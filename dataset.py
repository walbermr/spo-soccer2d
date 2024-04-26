import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

class TabularDataset(Dataset):
    '''
    Defines the class to handle tabular data.

    Parameters
    ----------
    X : np.array
        Samples for the dataset, in np.array format.
    y : np.array
        Classes, or labels, for the dataset samples.
    wm : np.array
        Classes, or labels, for the dataset samples

    Attributes
    ----------
    X : np.array
        Samples for the dataset, in np.array format.
    y : np.array
        Classes, or labels, for the dataset samples.
    wm : np.array
        Classes, or labels, for the dataset samples
    '''
    def __init__(
        self, 
        actions: np.array, 
        evals: np.array, 
        raw_wm: np.array, 
        wm: np.array, 
        bps_df: pd.DataFrame=None, 
        fset_type=None, 
        clusters:np.array=None
    ) -> None:

        self.actions, self.evals, self.wm, self.raw_wm = actions, evals, wm, raw_wm
        self.fset_type = fset_type
        self.clusters = clusters
        self.bps_df = bps_df

        self.unum_encode = OneHotEncoder()
        self.unum_encode.fit(np.array([[i] for i in range(1,12)]))

        super().__init__()

    def __getitem__(self, index:int) -> dict:
        '''
        Function that returs the 'index' sample from the dataset. Required when using pytorch's DataLoader. Note that this function
        does not require a dictionary as return type, as this output will be handled by the programmer in the model's training function.

        Parameters
        ----------
        index : int
            Index to get sample

        Returns
        -------
        dictionary containing 'X' and 'y' as keys.
        '''

        action_row = self.actions[index]
        wm_row = self.wm[index]
        raw_wm_row = self.raw_wm[index]
        curr_timehash, curr_cycle, curr_self_unum = raw_wm_row[0].astype(np.int64), raw_wm_row[1].astype(np.int64), raw_wm_row[2].astype(np.int64)
        raw_wm_row = raw_wm_row[1:]
        
        bps_row = self.bps_df[curr_cycle][curr_timehash]
        if self.fset_type == "player_class_fset1":
            action_eval = self.evals[index]
            action_risk = np.array([0])

            return {
                "action": action_row.astype(np.float32),
                "action_eval": self.unum_encode.transform(action_eval[np.newaxis]).A[0].astype(np.float32),
                "action_risk": action_risk.astype(np.float32),
                "unum_label": action_eval[0].astype(np.float32),
                "wm": wm_row.astype(np.float32),
                "raw_wm": raw_wm_row,
                "bps_row": bps_row,
                "cycle": curr_cycle,
                "timehash": curr_timehash,
            } 
        else:
            action_eval = self.evals[index, :1]
            action_risk = self.evals[index, 1:]


        return {
            "action": action_row.astype(np.float32),
            "action_eval": action_eval.astype(np.float32),
            "action_risk": action_risk.astype(np.float32),
            "wm": wm_row.astype(np.float32),
            "raw_wm": raw_wm_row,
            "bps_row": bps_row,
            "cycle": curr_cycle,
            "timehash": curr_timehash,
        } 

    def __len__(self) -> int:
        '''
        Function that returs the size of the dataset. Required when using pytorch's DataLoader.

        Returns
        -------
        integer the dataset's size.
        '''
        return self.actions.shape[0]
