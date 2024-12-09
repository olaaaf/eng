import wandb
import torch
import numpy as np
from train.model import SimpleModel
import os


def load_model_version(model_id, version):
    run = wandb.init(project="mario_advanced_dqn", entity="olafercik")
    artifact = run.use_artifact(
        f"olafercik/mario_shpeed/advanced_model_checkpoint_{model_id}:v{version}",
        type="model",
    )
    artifact_dir = artifact.download()
    model_file = os.listdir(artifact_dir)[0]
    checkpoint = torch.load(
        f"{artifact_dir}/{model_file}", map_location=torch.device("cpu")
    )
    model = SimpleModel()
    model.load_state_dict(checkpoint["model_state_dict"])
    wandb.finish()
    return model


def compare_models(model1, model2):
    differences = {}
    for (name1, param1), (name2, param2) in zip(
        model1.named_parameters(), model2.named_parameters()
    ):
        if not torch.equal(param1, param2):
            diff = torch.mean(torch.abs(param1 - param2)).item()
            differences[name1] = diff
    return differences


def main():
    model_id = input("Enter model ID: ")
    start_version = int(input("Enter start version: "))
    end_version = int(input("Enter end version: "))

    models = {}
    for version in range(start_version, end_version + 1):
        print(f"Loading version {version}...")
        models[version] = load_model_version(model_id, version)

    print("\nComparing consecutive versions:")
    for v in range(start_version, end_version):
        diff = compare_models(models[v], models[v + 1])
        print(f"\nDifferences between v{v} and v{v+1}:")
        for layer, diff_value in diff.items():
            print(f"{layer}: {diff_value:.6f}")


if __name__ == "__main__":
    main()
