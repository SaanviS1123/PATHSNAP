"""Plots use saved measured model runs, never fabricated epoch curves."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/pathsnap-matplotlib')
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_model(name, rows, histories):
    out = Path('results/plots')
    out.mkdir(parents=True, exist_ok=True)
    cv = [row for row in rows if row['partition'] == 'cv']
    student = rows[-1]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].bar([str(row['fold']) for row in cv], [row['macro_f1'] for row in cv])
    axes[0].set(title=f'{name.upper()}: grouped CV', xlabel='Validation fold', ylabel='Candidate Macro-F1', ylim=(0, 1))
    axes[1].bar(['Top-1', 'Macro-F1'], [student['top1_acc'], student['macro_f1']], color=['#ce632f', '#3d8295'])
    axes[1].set(title='Student test', ylim=(0, 1))
    matrix = np.array(student['confusion_matrix'])
    axes[2].imshow(matrix, cmap='Blues')
    for (i, j), value in np.ndenumerate(matrix):
        axes[2].text(j, i, str(value), ha='center', va='center')
    axes[2].set(title='Student-test confusion matrix', xlabel='Predicted', ylabel='Actual', xticks=[0, 1], yticks=[0, 1])
    fig.tight_layout()
    fig.savefig(out / f'{name}_evaluation.png', dpi=150)
    plt.close(fig)
    if histories:
        fig, ax = plt.subplots(figsize=(7, 4))
        for run in sorted({row['run'] for row in histories}):
            history = [row for row in histories if row['run'] == run]
            ax.plot([row['epoch'] for row in history], [row['train_loss'] for row in history], label=run)
        ax.set(title='MLP: recorded training loss', xlabel='Epoch', ylabel='Training objective')
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / 'mlp_loss.png', dpi=150)
        plt.close(fig)
