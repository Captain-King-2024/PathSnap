"""Plot Random Forest feature importances and save to results/figures/.

Usage:
    python src/visualization/plot_rf_importance.py
"""
import json
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def main():
    summary_path = Path('models/RandomForest/training_summary.json')
    if not summary_path.exists():
        summary_path = Path('models/random_forest/training_summary.json')
    output_dir = Path('results/figures')
    output_dir.mkdir(parents=True, exist_ok=True)

    if not summary_path.exists():
        raise FileNotFoundError(
            f'{summary_path} not found. Please train the Random Forest model first.'
        )

    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    fi = summary['feature_importances']
    names = list(fi.keys())
    vals = [fi[n] for n in names]
    order = sorted(range(len(vals)), key=lambda i: vals[i], reverse=True)

    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(
            [names[i] for i in order[::-1]],
            [vals[i] for i in order[::-1]],
            color='steelblue',
        )
        ax.set_xlabel('Mean decrease in impurity (normalized)')
        ax.set_title('Random Forest Feature Importances')
        fig.tight_layout()
        out_file = output_dir / 'rf_feature_importances.png'
        fig.savefig(out_file, dpi=150)
        plt.close(fig)
        print(f'Saved → {out_file}')
    else:
        # Text fallback
        print('\nFeature importances (descending):')
        for i in order:
            print(f'  {names[i]:<25s} {vals[i]:.5f}')


if __name__ == '__main__':
    main()
