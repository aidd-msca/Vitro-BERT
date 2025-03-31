import os
import sys
# Add the source directory to Python path
sys.path.append('/scratch/work/masooda1/ToxBERT/src')
import logging
from typing import Tuple, Sequence, Any, Dict, Union, Optional
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
import torch
from molbert.utils.featurizer.molfeaturizer import SmilesIndexFeaturizer
from molbert.models.smiles import SmilesMolbertModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class MolBertFeaturizer:
    """
    This featurizer takes a molbert model and transforms the input data and
    returns the representation in the last layer (pooled output and sequence_output).
    """

    def __init__(
        self,
        model,
        featurizer,
        device: str = None,
        embedding_type: str = 'pooled',
        max_seq_len: Optional[int] = None,
        permute: bool = False,
    ) -> None:
        super().__init__()
        self.device = device or 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embedding_type = embedding_type
        self.max_seq_len = max_seq_len
        self.permute = permute
        self.featurizer = featurizer
        self.model = model

    def __getstate__(self):
        self.__dict__.update({'model': self.model.to('cpu')})
        self.__dict__.update({'device': 'cpu'})
        return self.__dict__

    def transform_single(self, smiles: str) -> Tuple[np.ndarray, bool]:
        features, valid = self.transform([smiles])
        return features, valid[0]

    def transform(self, molecules: Sequence[Any]) -> Tuple[Union[Dict, np.ndarray], np.ndarray]:
        input_ids, valid = self.featurizer.transform(molecules)
        input_ids = self.trim_batch(input_ids, valid)

        token_type_ids = np.zeros_like(input_ids, dtype=np.int64)
        attention_mask = np.zeros_like(input_ids, dtype=np.int64)
        attention_mask[input_ids != 0] = 1

        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=self.device)

        with torch.no_grad():
            outputs = self.model.model.bert(
                input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
            )

        sequence_output, pooled_output = outputs

        valid_tensor = torch.tensor(
            valid, dtype=sequence_output.dtype, device=sequence_output.device, requires_grad=False
        )

        pooled_output = pooled_output * valid_tensor[:, None]
        sequence_out = sequence_output * valid_tensor[:, None, None]

        sequence_out = sequence_out.detach().cpu().numpy()
        pooled_output = pooled_output.detach().cpu().numpy()
        out = pooled_output

        return out, valid

    @staticmethod
    def trim_batch(input_ids, valid):
        if any(valid):
            _, cols = np.where(input_ids[valid] != 0)
        else:
            cols = np.array([0])

        max_idx: int = int(cols.max().item() + 1)
        input_ids = input_ids[:, :max_idx]
        return input_ids

def main():

    # Load model configuration
    path_to_checkpoint = '/scratch/work/masooda1/ToxBERT/molbert_100epochs/checkpoints/last.ckpt'
    model_dir = os.path.dirname(os.path.dirname(path_to_checkpoint))
    hparams_path = os.path.join(model_dir, 'hparams.yaml')
    
    with open(hparams_path) as yaml_file:
        config_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
    config_dict["pretrained_model_path"] = path_to_checkpoint

    # Initialize and load model
    model = SmilesMolbertModel(config_dict)
    checkpoint = torch.load(config_dict["pretrained_model_path"], 
                          map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    device = "cuda"
    model.eval()
    model.freeze()
    model = model.to(device)

    # Initialize featurizer
    featurizer = SmilesIndexFeaturizer.bert_smiles_index_featurizer(126, permute=False)
    f = MolBertFeaturizer(model=model, featurizer=featurizer, device=device)

    # Load data
    invitro_data = pd.read_parquet("/scratch/work/masooda1/ToxBERT/data/pretraining_data/chembl20_selected_assays_with_normalzied_smiles.parquet")
    smiles_list = invitro_data.Normalized_SMILES.tolist()

    # Process in batches
    batch_size = 2000
    batches = [smiles_list[i:i+batch_size] for i in range(0, len(smiles_list), batch_size)]
    print(f"Total SMILES: {len(smiles_list)}")

    features_all, masks_all = [], []
    for batch_smiles in tqdm(batches):
        features, masks = f.transform(batch_smiles)  # Now capturing both features and masks
        torch.cuda.empty_cache()
        features_all.append(features)  # Store features for each batch
        masks_all.extend(masks.tolist())

    # Convert features list to numpy array and filter using masks
    features_all = np.vstack(features_all)  # Combine all features into one array
    filtered_features = features_all[masks_all]  # Filter features using masks

    # Filter original data
    filtered_invitro_data = invitro_data[masks_all].reset_index(drop=True)
    filtered_invitro_data = filtered_invitro_data.drop(["smiles"], axis = 1)
    filtered_invitro_data = filtered_invitro_data.rename(columns = {"Normalized_SMILES": "SMILES"})

    # Create features dataframe with filtered SMILES
    feature_columns = [f'feature_{i}' for i in range(filtered_features.shape[1])]
    features_df = pd.DataFrame(filtered_features, columns=feature_columns)
    features_df['Normalized_SMILES'] = filtered_invitro_data['Normalized_SMILES']

    # Save both dataframes separately
    data_dir = "/scratch/work/masooda1/ToxBERT/data/pretraining_data/"
    filtered_invitro_data.to_pickle(data_dir + 'Chembl20_filtered_for_MolBERT.pkl')
    features_df.to_pickle(data_dir + 'Chembl20_MolBERT_features.pkl')

    # Print statistics
    filtered_smiles = len(smiles_list) - sum(masks_all)
    print(f"We filtered {filtered_smiles} SMILES")
    print(f"Remaining SMILES: {sum(masks_all)}")
    print("Script Completed")

if __name__ == "__main__":
    main()