"""
Utility for logging and saving training results.
Saves training history, metrics, and configuration to JSON files.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any


class ResultsLogger:
    """Logs training results to JSON files for later analysis."""

    def __init__(self, output_dir: str, model_name: str):
        """
        Args:
            output_dir: Directory to save results
            model_name: Name of the model (e.g., 'text_specialist')
        """
        self.output_dir = output_dir
        self.model_name = model_name
        self.results = {
            'model_name': model_name,
            'start_time': datetime.now().isoformat(),
            'training_history': [],
            'best_metrics': {},
            'final_metrics': {},
            'config': {},
            'classification_reports': []
        }

    def log_config(self, config: Dict[str, Any]):
        """Log training configuration."""
        self.results['config'] = config

    def log_epoch(self, epoch: int, train_loss: float, train_f1: float,
                  val_loss: float, val_f1: float):
        """Log metrics for one epoch."""
        epoch_results = {
            'epoch': epoch,
            'train_loss': float(train_loss),
            'train_f1': float(train_f1),
            'val_loss': float(val_loss),
            'val_f1': float(val_f1)
        }
        self.results['training_history'].append(epoch_results)

    def log_best_metrics(self, epoch: int, val_f1: float, val_loss: float):
        """Log best validation metrics."""
        self.results['best_metrics'] = {
            'epoch': epoch,
            'val_f1': float(val_f1),
            'val_loss': float(val_loss)
        }

    def log_classification_report(self, epoch: int, report_dict: Dict):
        """Log classification report for an epoch."""
        self.results['classification_reports'].append({
            'epoch': epoch,
            'report': report_dict
        })

    def log_final_metrics(self, **kwargs):
        """Log final metrics and metadata."""
        self.results['final_metrics'] = {k: float(v) if isinstance(v, (int, float)) else v
                                         for k, v in kwargs.items()}
        self.results['end_time'] = datetime.now().isoformat()

    def save(self):
        """Save results to JSON file."""
        os.makedirs(self.output_dir, exist_ok=True)

        # Save full results
        results_path = os.path.join(self.output_dir, 'training_results.json')
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        # Save summary (for quick viewing)
        summary_path = os.path.join(self.output_dir, 'summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"{'='*60}\n")
            f.write(f"{self.model_name.upper()} - TRAINING SUMMARY\n")
            f.write(f"{'='*60}\n\n")

            f.write(f"Training Configuration:\n")
            for key, value in self.results['config'].items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

            f.write(f"Training History:\n")
            for epoch_data in self.results['training_history']:
                f.write(f"  Epoch {epoch_data['epoch']}: "
                       f"Train Loss={epoch_data['train_loss']:.4f}, "
                       f"Train F1={epoch_data['train_f1']:.4f}, "
                       f"Val Loss={epoch_data['val_loss']:.4f}, "
                       f"Val F1={epoch_data['val_f1']:.4f}\n")
            f.write("\n")

            if self.results['best_metrics']:
                f.write(f"Best Validation Metrics:\n")
                f.write(f"  Epoch: {self.results['best_metrics']['epoch']}\n")
                f.write(f"  Val F1: {self.results['best_metrics']['val_f1']:.4f}\n")
                f.write(f"  Val Loss: {self.results['best_metrics']['val_loss']:.4f}\n")
                f.write("\n")

            if self.results['final_metrics']:
                f.write(f"Final Metrics:\n")
                for key, value in self.results['final_metrics'].items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")

            # Add last classification report if available
            if self.results['classification_reports']:
                last_report = self.results['classification_reports'][-1]
                f.write(f"\nClassification Report (Epoch {last_report['epoch']}):\n")
                f.write(format_classification_report(last_report['report']))

        print(f"\n✅ Results saved to:")
        print(f"   - {results_path}")
        print(f"   - {summary_path}")


def format_classification_report(report_dict: Dict) -> str:
    """Format classification report dictionary as string."""
    lines = []

    # Header
    lines.append(f"{'':20} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}")
    lines.append("")

    # Per-class metrics
    for class_name, metrics in report_dict.items():
        if class_name in ['accuracy', 'macro avg', 'weighted avg']:
            continue
        if isinstance(metrics, dict):
            lines.append(f"{class_name:20} "
                        f"{metrics.get('precision', 0):10.2f} "
                        f"{metrics.get('recall', 0):10.2f} "
                        f"{metrics.get('f1-score', 0):10.2f} "
                        f"{metrics.get('support', 0):10.0f}")

    lines.append("")

    # Overall metrics
    if 'accuracy' in report_dict:
        lines.append(f"{'accuracy':20} {''} {''} {report_dict['accuracy']:10.2f} "
                    f"{report_dict.get('macro avg', {}).get('support', 0):10.0f}")

    for avg_type in ['macro avg', 'weighted avg']:
        if avg_type in report_dict:
            metrics = report_dict[avg_type]
            lines.append(f"{avg_type:20} "
                        f"{metrics.get('precision', 0):10.2f} "
                        f"{metrics.get('recall', 0):10.2f} "
                        f"{metrics.get('f1-score', 0):10.2f} "
                        f"{metrics.get('support', 0):10.0f}")

    return "\n".join(lines)
