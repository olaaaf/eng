import os
import torch
import wandb
import matplotlib.pyplot as plt
from tqdm import tqdm
from tabulate import tabulate
from time import sleep

from game.runner import Runner
from train.model import SimpleModel
from train.helpers import ConfigFileReward
from util.logger import setup_logger
from util.db_handler import DBHandler
wandb.init()

def load_model_from_wandb(model_id: int, version: int) -> SimpleModel:
    artifact = wandb.use_artifact(
        f"olafercik/mario_shpeed/advanced_model_checkpoint_{model_id}:v{version}",
        type="model"
    )
    artifact_dir = artifact.download()
    checkpoint_file = os.listdir(artifact_dir)[0]
    checkpoint = torch.load(f"{artifact_dir}/{checkpoint_file}", map_location=device)
    
    model = SimpleModel(reward_handler)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

def evaluate_model(model: SimpleModel) -> dict:
    state = runner.reset()
    
    while not runner.done:
        with torch.no_grad():
            actions = model(state.view(1, -1))
            actions = actions.float().squeeze().cpu().tolist()
            state = runner.next(controller=actions)
            
    return {
        'max_x': max(runner.step.x_pos),
        'score': runner.step.score[-1],
        'time': runner.step.time,
        'finished': 1.0 if runner.alive else 0.0,
        'goomba_stomps': runner.step.goomba_stomps
    }

if __name__ == "__main__":
    # Initialize components
    db_handler = DBHandler()
    logger = setup_logger(db_handler, "model_evaluator")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_id = int(input("Enter model ID to evaluate: "))
    start_version = int(input("Enter starting version number: "))
    end_version = int(input("Enter ending version number: "))
    
    reward_handler = ConfigFileReward(logger, model_id, "rewards.json")
    runner = Runner(device, record=False)
    
    results = []
    print("\nEvaluating models...")
    
    for version in tqdm(range(start_version, end_version + 1)):
        try:
            model = load_model_from_wandb(model_id, version)
            metrics = evaluate_model(model)
            results.append({'version': version, **metrics})
        except Exception as e:
            print(f"\nError evaluating version {version}: {str(e)}")
    
    # Display results table
    headers = ['Version', 'Max X', 'Score', 'Time', 'Finished', 'Goombas']
    table_data = [
        [
            r['version'],
            f"{r['max_x']:.1f}",
            f"{r['score']:.0f}",
            f"{r['time']:.0f}",
            f"{r['finished']*100:.1f}%",
            r['goomba_stomps']
        ] 
        for r in results
    ]
    print("\nEvaluation Results:")
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    # Plot progression
    metrics_to_plot = ['max_x', 'score', 'finished']
    plt.figure(figsize=(12, 8))
    
    for i, metric in enumerate(metrics_to_plot, 1):
        plt.subplot(len(metrics_to_plot), 1, i)
        plt.plot([r['version'] for r in results], [r[metric] for r in results])
        plt.title(f'{metric.replace("_", " ").title()} vs Version')
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()