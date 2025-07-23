import torch
import importlib
import os
import numpy as np
import config as base_config_module

# Get all model names defined in config.py (excluding BaseConfig)
all_config_names = [
    name
    for name in dir(base_config_module)
    if name.endswith("Config") and name != "BaseConfig"
]
model_names = [name.replace("Config", "") for name in all_config_names]

print(f"Found models: {', '.join(model_names)}\n")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# --- Load Common Pretrained Embeddings ---
try:
    pretrained_word_embedding = torch.from_numpy(
        np.load("./data/train/pretrained_word_embedding.npy")
    ).float()
    print("Loaded pretrained word embeddings.")
except FileNotFoundError:
    pretrained_word_embedding = None
    print("Pretrained word embeddings not found, using None.")
except Exception as e:
    print(f"Error loading word embeddings: {e}")
    pretrained_word_embedding = None


# --- DKN Specific Embeddings ---
# Attempt to load DKN specific embeddings only if needed, otherwise set to None
pretrained_entity_embedding = None
pretrained_context_embedding = None
if "DKN" in model_names:
    try:
        pretrained_entity_embedding = torch.from_numpy(
            np.load("./data/train/pretrained_entity_embedding.npy")
        ).float()
        print("Loaded DKN pretrained entity embeddings.")
    except FileNotFoundError:
        print("DKN pretrained entity embeddings not found, using None.")
    except Exception as e:
        print(f"Error loading DKN entity embeddings: {e}")

    try:
        pretrained_context_embedding = torch.from_numpy(
            np.load("./data/train/pretrained_context_embedding.npy")
        ).float()
        print("Loaded DKN pretrained context embeddings.")
    except FileNotFoundError:
        print("DKN pretrained context embeddings not found, using None.")
    except Exception as e:
        print(f"Error loading DKN context embeddings: {e}")

print("-" * 30)

# --- Calculate and Print Parameters ---
print(f"DEBUG: Models to process: {model_names}")
for model_name in model_names:
    print(f"Calculating parameters for: {model_name}")
    Model = None
    config = None
    try:
        # Dynamically import model and config
        model_module_path = f"model.{model_name}"
        config_module_path = "config"
        config_class_name = f"{model_name}Config"

        # --- Import Model ---
        model_module = None
        try:
            model_module = importlib.import_module(model_module_path)
            Model = getattr(model_module, model_name)
        except Exception as import_err:
            print(f"> Error DURING import of {model_module_path}: {import_err}")
            # Optionally re-raise or continue to the outer handler
            raise # Re-raise to be caught by the outer handler

        # --- Import Config ---
        config_module = importlib.import_module(config_module_path)
        config = getattr(config_module, config_class_name)

        # --- Instantiate Model ---
        if model_name == "DKN":
            model = Model(
                config,
                pretrained_word_embedding,
                pretrained_entity_embedding,
                pretrained_context_embedding,
            )
        else:
            # Most models expect config and word embeddings
            model = Model(config, pretrained_word_embedding)

        # Move model to device before calculating params (some layers might be device-dependent)
        # model.to(device) # Optional: usually not needed just for param count

        # Calculate trainable parameters
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Print result
        print(
            f"> {model_name}: {total_params:,} trainable parameters (~{total_params / 1e6:.2f} M)"
        )

    except ImportError as e:
        print(f"> Outer ImportError: Could not import module or class for {model_name}. Error: {e}. Skipping.")
    except AttributeError as e:
         print(f"> Outer AttributeError: Could not find class '{model_name}' or '{config_class_name}'. Error: {e}. Skipping.")
    except TypeError as e:
        print(f"> TypeError: Issue instantiating model {model_name}. Check constructor arguments. Error: {e}. Skipping.")
    except Exception as e:
        print(f"> An unexpected error occurred for model {model_name}: {e}. Skipping.")
    print("-" * 30)

print("Parameter calculation complete.")
