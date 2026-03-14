import os
import json
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr, spearmanr
from openai import OpenAI


CONDITION_MODE = 2 # 1 for model_name + scenario_name + region, 2 for model_fingerprints + scenario_desc + region
TRAINING_PATH = "/home/xuhw/others/cVAE_iam/data/scenario_desc_all_with_model_family_eff100_to_nan_remaining_drop_all_holdouts.xlsx"
DEV_PATH = "/home/xuhw/others/cVAE_iam/data/scenario_desc_all_with_model_family_eff100_to_nan_val_ground_truth.xlsx"
TEST_PATH = "/home/xuhw/others/cVAE_iam/data/scenario_desc_all_with_model_family_eff100_to_nan_test_ground_truth.xlsx"
MODEL_FINGERPRINTS_PATH = "/home/xuhw/others/cVAE_iam/data/model_fingerprint.xlsx"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SAVE_DIR = "/home/xuhw/others/cVAE_iam/save/cvae_condition{}".format(CONDITION_MODE)

train_hyper = json.load(open(os.path.join(SAVE_DIR, "training_hyperparameters.json"), 'r'))
BATCH_SIZE = train_hyper['batch_size']
HIDDEN_DIM = train_hyper['hidden_dim']
LATENT_DIM = train_hyper['latent_dim']


def read_model_fingerprints():
    df = pd.read_excel(MODEL_FINGERPRINTS_PATH)
    result = {}
    for index, row in df.iterrows():
        model_family = row['Model family']
        mitigation = row['Mitigation Preference']
        responds = row['Responds']
        result[model_family] = f"Mitigation Preference: {mitigation}\nResponds: {responds}"
    return result

def read_data(file_path=TRAINING_PATH, model_fingerprints=None, scen_name_desc_dict=None):
    if model_fingerprints is None:
        model_fingerprints = read_model_fingerprints()
        
    cache_dir = os.path.join(os.path.dirname(file_path), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    basename = os.path.basename(file_path)
    cache_file = os.path.join(cache_dir, f"{os.path.splitext(basename)[0]}.json")
    
    if os.path.exists(cache_file):
        print(f"Loading data from cache: {cache_file}")
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    print(f"Reading data from Excel: {file_path}")
    df = pd.read_excel(file_path)
    year_cols = [str(year) for year in range(2010, 2110, 10)]
    data_list = []
    
    n_rows = len(df)
    for i in tqdm(range(0, n_rows, 24)):
        chunk = df.iloc[i:i+24]
        if len(chunk) != 24:
            break
        
        if 'category_C' in chunk.columns:
            cat_val = str(chunk['category_C'].iloc[0]).strip()
        elif 'c_group' in chunk.columns:
            cat_val = str(chunk['c_group'].iloc[0]).strip()
        else:
            raise ValueError("No category column found")
            
        if cat_val in ['c1', 'c2', 'c3', 'c4', 'c1-c4']:
            category_C_mapped = 0
        elif cat_val in ['c5', 'c6', 'c5-c6']:
            category_C_mapped = 1
        elif cat_val in ['c7', 'c8', 'c7-c8']:
            category_C_mapped = 2
        else:
            category_C_mapped = -1
            
        model_family_val = str(chunk['model_family'].iloc[0])
        scenario_val = str(chunk['scenario'].iloc[0])
        scenario_desc_val = str(chunk['scenario_description'].iloc[0]) if scen_name_desc_dict is None else scen_name_desc_dict[scenario_val]
        region_val = str(chunk['region'].iloc[0])
        model_fp_str = model_fingerprints.get(model_family_val, "")
        
        data_list.append({
            "model_family": model_family_val,
            "scenario": scenario_val,
            "scenario_description": scenario_desc_val,
            "category_C": category_C_mapped,
            "region": region_val,
            "condition_text_1": f"{model_family_val} {scenario_val} {region_val}",
            "condition_text_2": f"{model_fp_str} {scenario_desc_val} {region_val}",
            "data": chunk[year_cols].fillna(-1).values.tolist(),
            "variable_list": chunk['variable'].tolist()
        })
        
    print(f"Saving data to cache: {cache_file}")
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)
        
    return data_list

def compute_condition_embedding(texts, model_name="text-embedding-3-large", api_key=OPENAI_API_KEY):
    # return a dictionary mapping text to embedding    
    cache_dir = os.path.join(os.path.dirname(TRAINING_PATH), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "condition_embedding.pkl")
    
    if os.path.exists(cache_file):
        print(f"Loading condition embeddings from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print(f"Computing condition embeddings using {model_name}...")
    client = OpenAI(api_key=api_key)
    unique_texts = list(set(texts))
    embeddings_dict = {}
    
    batch_size = 1000
    for i in tqdm(range(0, len(unique_texts), batch_size), desc="Computing Embeddings"):
        batch = unique_texts[i:i+batch_size]
        response = client.embeddings.create(input=batch, model=model_name)
        for j, data in enumerate(response.data):
            embeddings_dict[batch[j]] = data.embedding
            
    print(f"Saving condition embeddings to cache: {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(embeddings_dict, f)
        
    return embeddings_dict

class cVAEData(Dataset):
    def __init__(self, data_list, embeddings_dict, condition_mode=CONDITION_MODE):
        self.data_list = data_list
        self.embeddings_dict = embeddings_dict
        self.condition_mode = condition_mode

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        # Determine condition text based on condition mode
        if self.condition_mode == 1:
            condition_text = item["condition_text_1"]
        elif self.condition_mode == 2:
            condition_text = item["condition_text_2"]
        else:
            raise ValueError(f"Invalid condition_mode: {self.condition_mode}")
            
        # Get the embedding for the condition text
        embedding = self.embeddings_dict[condition_text]
        
        # Get the data and category
        data = item["data"]
        category = item["category_C"]
        
        # Convert to PyTorch tensors
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
        data_tensor = torch.tensor(data, dtype=torch.float32)
        category_tensor = torch.tensor(category, dtype=torch.float32)
        # Expand a new colum for data tensor, fill in with category_tensor
        data_tensor = torch.cat([data_tensor, torch.ones(1, data_tensor.shape[1]) * category_tensor], dim=0)
        return embedding_tensor, data_tensor, category_tensor


class Encoder(nn.Module):
    def __init__(self, input_dim, condition_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim + condition_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x, c):
        # Flatten x if it's 2D (batch, seq_len, features) -> (batch, seq_len * features)
        x_flat = x.view(x.size(0), -1)
        # Concatenate input and condition
        xc = torch.cat([x_flat, c], dim=1)
        h = self.relu(self.fc1(xc))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, condition_dim, hidden_dim, output_dim, output_shape):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim + condition_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.output_shape = output_shape

    def forward(self, z, c):
        zc = torch.cat([z, c], dim=1)
        h = self.relu(self.fc1(zc))
        out_flat = self.fc2(h)
        # Reshape to original data shape
        return out_flat.view(-1, *self.output_shape)

class CVAE(nn.Module):
    def __init__(self, input_dim, condition_dim, hidden_dim, latent_dim, output_shape):
        super(CVAE, self).__init__()
        self.encoder = Encoder(input_dim, condition_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, condition_dim, hidden_dim, input_dim, output_shape)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c):
        mu, logvar = self.encoder(x, c)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z, c)
        return recon_x, mu, logvar
        
    def generate(self, c, latent_dim):
        # Sample z from standard normal distribution
        z = torch.randn(c.size(0), latent_dim).to(c.device)
        return self.decoder(z, c)

def evaluate(model, dataloader, device, latent_dim, return_preds=False):
    # Retrieve variable names from the dataset. All items in the dataloader have the same variable_list.
    # The length of variable_names matches `n_variables`.
    variable_names = dataloader.dataset.data_list[0]["variable_list"]
    model.eval()
    all_preds = []
    all_trues = []
    
    with torch.no_grad():
        for embeddings, data, _ in dataloader:
            embeddings, data = embeddings.to(device), data.to(device)
            # Generate predictions using condition
            preds = model.generate(embeddings, latent_dim)
            all_preds.append(preds.cpu().numpy())
            all_trues.append(data.cpu().numpy())
            
    all_preds = np.concatenate(all_preds, axis=0) # Shape: (N, n_variables + 1, n_years)
    all_trues = np.concatenate(all_trues, axis=0)
    
    # Extract the true data variables and the C_variable (which is the last row we appended)
    # True data: [N, n_variables, n_years]
    preds_vars = all_preds[:, :-1, :]
    trues_vars = all_trues[:, :-1, :]
    
    preds_c_cat = all_preds[:, -1, :]
    trues_c_cat = all_trues[:, -1, :]
    
    # --- Calculate overall metrics on numerical variables ---
    flat_preds = preds_vars.flatten()
    flat_trues = trues_vars.flatten()
    
    mask = flat_trues != -1 # filter out padding
    flat_preds = flat_preds[mask]
    flat_trues = flat_trues[mask]
    
    mae = np.mean(np.abs(flat_preds - flat_trues))
    rmse = np.sqrt(np.mean((flat_preds - flat_trues)**2))
    
    # sMAPE
    numerator = np.abs(flat_preds - flat_trues)
    denominator = (np.abs(flat_preds) + np.abs(flat_trues)) / 2.0
    smape_mask = denominator != 0
    smape = np.mean(numerator[smape_mask] / denominator[smape_mask]) * 100
    
    # --- Dictionary Initialization ---
    results = {
        "overall": {}
    }
    
    # Calculate baseline metrics for each variable independently
    n_samples, n_variables, n_years = trues_vars.shape
    pearson_list = []
    spearman_list = []
    
    for v in range(n_variables):
        var_name = variable_names[v]
        v_preds = preds_vars[:, v, :].flatten()
        v_trues = trues_vars[:, v, :].flatten()
        
        v_mask = v_trues != -1
        
        # Initialize default metrics for this variable
        v_metrics = {
            'mae': None,
            'rmse': None,
            'smape_%': None,
            'pearson': None,
            'spearman': None,
            'c_category_accuracy': None
        }
        
        if np.sum(v_mask) > 0:
            v_preds_valid = v_preds[v_mask]
            v_trues_valid = v_trues[v_mask]
            
            # MAE, RMSE
            v_mae = np.mean(np.abs(v_preds_valid - v_trues_valid))
            v_rmse = np.sqrt(np.mean((v_preds_valid - v_trues_valid)**2))
            
            # sMAPE
            v_numer = np.abs(v_preds_valid - v_trues_valid)
            v_denom = (np.abs(v_preds_valid) + np.abs(v_trues_valid)) / 2.0
            v_smape_mask = v_denom != 0
            if np.sum(v_smape_mask) > 0:
                v_smape = np.mean(v_numer[v_smape_mask] / v_denom[v_smape_mask]) * 100
            else:
                v_smape = 0.0
                
            v_metrics['mae'] = float(v_mae)
            v_metrics['rmse'] = float(v_rmse)
            v_metrics['smape_%'] = float(v_smape)
            
            # Pearson, Spearman requires at least 2 points
            if np.sum(v_mask) > 1:
                p, _ = pearsonr(v_preds_valid, v_trues_valid)
                s, _ = spearmanr(v_preds_valid, v_trues_valid)
                
                if not np.isnan(p):
                    v_metrics['pearson'] = float(p)
                    pearson_list.append(p)
                if not np.isnan(s):
                    v_metrics['spearman'] = float(s)
                    spearman_list.append(s)
                    
        # Add to the output structure
        results[var_name] = v_metrics
                
    pearson = np.mean(pearson_list) if len(pearson_list) > 0 else 0
    spearman = np.mean(spearman_list) if len(spearman_list) > 0 else 0
    
    # --- Calculate Accuracy for c_variable ---
    # The true category is constant across years, shape: [N, n_years]
    # We'll take the first year's prediction and round it to nearest integer as class prediction
    pred_c_classes = np.round(preds_c_cat[:, 0])
    true_c_classes = trues_c_cat[:, 0]
    
    accuracy = np.mean(pred_c_classes == true_c_classes)
        
    # --- Populate Overall Metrics ---
    results["overall"] = {
        'mae': float(mae),
        'rmse': float(rmse),
        'smape_%': float(smape),
        'pearson': float(pearson),
        'spearman': float(spearman),
        'c_category_accuracy': float(accuracy)
    }
    
    if return_preds:
        return results, all_preds
    return results


if __name__ == "__main__":
    print(f"Reading data and loading model from: {SAVE_DIR}")
    
    # Load model fingerprints and dictionary mapping
    model_fingerprints = read_model_fingerprints()
    train_data = read_data(TRAINING_PATH)
    scen_name_desc_dict = {item["scenario"]: item["scenario_description"] for item in train_data}
    
    # Read test (or dev) data
    test_data = read_data(TEST_PATH, scen_name_desc_dict=scen_name_desc_dict)
    print("There are {} test samples for inference.".format(len(test_data)))
    
    # Gather text inputs
    all_texts = []
    for item in test_data:
        all_texts.append(item["condition_text_1"])
        all_texts.append(item["condition_text_2"])
        
    # Get the embedding using OpenAI (uses cache if available)
    embeddings = compute_condition_embedding(all_texts)
    
    test_dataset = cVAEData(test_data, embeddings)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model Setup
    sample_embedding, sample_data, _ = test_dataset[0]
    condition_dim = sample_embedding.shape[0]
    output_shape = sample_data.shape
    input_dim = np.prod(output_shape)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = CVAE(input_dim, condition_dim, HIDDEN_DIM, LATENT_DIM, output_shape).to(device)
    
    # Load Model Checkpoint
    model_path = os.path.join(SAVE_DIR, "best_cvae_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No checkpoint found at {model_path}")
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print("Starting Inference...")
    test_metrics, all_preds = evaluate(model, test_loader, device, LATENT_DIM, return_preds=True)
    
    print("\nInference Evaluation Overall Metrics:")
    for metric, value in test_metrics["overall"].items():
        if value is not None:
            print(f"{metric.upper()}: {value:.4f}")
        else:
            print(f"{metric.upper()}: None")

    # Optionally dump inference-specific metrics if desired
    with open(os.path.join(SAVE_DIR, "inference_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=4)
        
    print(f"Inference metrics saved to {os.path.join(SAVE_DIR, 'inference_metrics.json')}")
    
    # Save predictions to match format of TEST_PATH
    print("Formatting and saving predictions...")
    preds_vars = all_preds[:, :-1, :] # drop c_variable row
    n_samples, n_variables, n_years = preds_vars.shape
    year_cols = [str(year) for year in range(2010, 2110, 10)]
    
    output_rows = []
    
    for i, item in enumerate(test_data):
        # We need to construct 24 rows for each sample
        for v, var_name in enumerate(item["variable_list"]):
            row_dict = {
                "model_family": item["model_family"],
                "scenario": item["scenario"],
                "scenario_description": item["scenario_description"],
                "region": item["region"],
                "variable": var_name,
                "category_C": f"c{item['category_C'] + 1}" if item["category_C"] >= 0 else "unknown"  # Rough inverse mapping
            }
            # Add predicted years
            for y_idx, year_col in enumerate(year_cols):
                pred_val = float(preds_vars[i, v, y_idx])
                row_dict[year_col] = pred_val
                
            output_rows.append(row_dict)
            
    out_df = pd.DataFrame(output_rows)
    out_xlsx_path = os.path.join(SAVE_DIR, "inference_predictions.xlsx")
    out_df.to_excel(out_xlsx_path, index=False)
    print(f"Saved predictions to {out_xlsx_path}")
