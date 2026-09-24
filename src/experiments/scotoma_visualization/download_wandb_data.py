import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from textwrap import wrap

class WandbPlotter:
    def __init__(self):
        self.api = wandb.Api()
        self.entity = "ttdduu-cerco"
        self.project = "Scotoma_Sweeps"
        self.experiment_groups = ["atto-og-apr9-WS-S", "atto-og-apr9-LH-S"]
        self.plans = ['LH-S','WS-S']
        self.datasets = ['eth80','eth80-padded-circular']
        self.logpolar_apply = None
        self.df = None

    def collect_data(self, logpolar_apply=None):
        """Collect data from wandb and store in DataFrame"""
        print("Collecting data from wandb...")
        self.logpolar_apply = logpolar_apply
        all_runs_data = []
        for group in self.experiment_groups:
            # Create base filters
            filters = {"group": group}
            
            # Add logpolar_apply filter if specified
            if logpolar_apply is not None:
                filters["config.logpolar_apply"] = logpolar_apply
            
            runs = self.api.runs(
                f"{self.entity}/{self.project}",
                filters=filters  # Use the updated filters
            )
            
            # Debug information
            print(f"\nProcessing group: {group}")
            print(f"Filters applied: {filters}")
            
            for run in runs:
                run_data = {
                    'group': group,
                    'scotoma_radius': run.config.get('scotoma_radius'),
                    'scotoma_sharpness': run.config.get('scotoma_sharpness'),
                    'val_acc': run.summary.get('val_acc'),
                    'dataset': run.config.get('dataset'),
                    'logpolar_apply': run.config.get('logpolar_apply'),
                    'run_id': run.id  # Add run ID for debugging
                }
                all_runs_data.append(run_data)
            
            # Debug information for this group
            df_group = pd.DataFrame(all_runs_data)
            print(f"\nNumber of runs for {group}: {len(df_group)}")
            if not df_group.empty:
                print("\nData summary for this group:")
                print(df_group.groupby(['dataset', 'scotoma_radius', 'scotoma_sharpness', 'logpolar_apply']).size())
        
        self.df = pd.DataFrame(all_runs_data)
        print("\nFinal data shape:", self.df.shape)
        print("\nUnique combinations in final dataset:")
        print(self.df.groupby(['group', 'dataset', 'scotoma_radius', 'scotoma_sharpness', 'logpolar_apply']).size())

    def setup_plot_style(self, ax, title, xlabel, ylabel):
        """Apply consistent styling to plot axes"""
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.grid(True)

    def create_scotoma_plots(self, group_df, group):
        """Create plots for AS-S and LH-S groups"""
        if group == "bien-WS-S":
            group_name = "WS-S"
        elif group == "bien-LH-S":
            group_name = "LH-S"
        else:
            group_name = group.replace("bien-", "")  # fallback for other cases

        fig = plt.figure(figsize=(8*len(self.datasets), 12))

        radius_axes, sharpness_axes = [], []
        y_limits = {'radius': [float('inf'), float('-inf')],
                   'sharpness': [float('inf'), float('-inf')]}

        # Create plots
        for col, dataset in enumerate(self.datasets):
            dataset_df = group_df[group_df['dataset'] == dataset]
            if dataset_df.empty:
                print(f"No data found for {group} with dataset {dataset}")
                continue

            # Create and style plots
            axes = self._create_dataset_plots(dataset_df, col, dataset)
            radius_axes.append(axes['radius'])
            sharpness_axes.append(axes['sharpness'])
            
            # Update y-limits
            for plot_type in ['radius', 'sharpness']:
                y_limits[plot_type][0] = min(y_limits[plot_type][0], 
                                           dataset_df['val_acc'].min())
                y_limits[plot_type][1] = max(y_limits[plot_type][1], 
                                           dataset_df['val_acc'].max())

        # Apply consistent y-limits with padding
        self._apply_y_limits(radius_axes, sharpness_axes, y_limits)
        
        # Create shared legends
        self._create_shared_legends(radius_axes[-1], sharpness_axes[-1])

        # Adjust layout first
        plt.tight_layout()
        
        
        # Add centered title after layout adjustments
        wrapped_title = '\n'.join(wrap(f'Validation Accuracy Analysis for {group_name}', 
                                     width=90))
        fig.suptitle(wrapped_title, fontsize=20, y=1.02,ha='center')

        self._save_plot(group, '_analysis.png')

    def create_histograms(self, group_df, group):
        """Create histograms for LH-H group"""
        fig, axes = plt.subplots(2, 1, figsize=(8, 12))
        fig.suptitle(f'Validation Accuracy Histograms for {group.split("_")[1]}', 
                    fontsize=30, y=1.02)

        for ax, dataset in zip(axes, self.datasets):
            dataset_df = group_df[group_df['dataset'] == dataset]
            if not dataset_df.empty:
                ax.hist(dataset_df['val_acc'], bins=10, alpha=0.7, color='blue')
                self.setup_plot_style(ax, dataset, 'Validation Accuracy', 'Frequency')

        plt.tight_layout()
        self._save_plot(group, '_histograms.png')

    def _create_dataset_plots(self, dataset_df, col, dataset):
        """Create plots for a single dataset"""
        axes = {}
        
        # Debug information
        print(f"\nPlotting dataset: {dataset}")
        print(f"Number of data points: {len(dataset_df)}")
        print("\nUnique combinations:")
        print(dataset_df.groupby(['scotoma_radius', 'scotoma_sharpness']).size())
        
        # Radius plot
        ax1 = plt.subplot(2, len(self.datasets), col + 1)
        unique_sharpness = dataset_df['scotoma_sharpness'].unique()
        
        if len(unique_sharpness) == 1:
            # If only one sharpness value, plot both logpolar options on same plot
            colors = {'True': 'blue', 'False': 'red'}
            for logp in [True, False]:
                mask = (dataset_df['logpolar_apply'] == logp)
                data_to_plot = dataset_df[mask].sort_values('scotoma_radius')
                if not data_to_plot.empty:
                    ax1.plot(data_to_plot['scotoma_radius'], data_to_plot['val_acc'],
                            'o-', color=colors[str(logp)], 
                            label=f'LogPolar={logp}')
        else:
            # Original behavior for multiple sharpness values
            for sharpness in sorted(dataset_df['scotoma_sharpness'].unique()):
                mask = (dataset_df['scotoma_sharpness'] == sharpness)
                if self.logpolar_apply is not None:
                    mask = mask & (dataset_df['logpolar_apply'] == self.logpolar_apply)
                
                data_to_plot = dataset_df[mask].sort_values('scotoma_radius')
                ax1.plot(data_to_plot['scotoma_radius'], data_to_plot['val_acc'],
                        'o-', label=f'sharpness={sharpness:.3f}')
        
        self.setup_plot_style(ax1, dataset, 'Scotoma Radius', 'Validation Accuracy')
        axes['radius'] = ax1

        # Only create sharpness plot if there are multiple sharpness values
        if len(unique_sharpness) > 1:
            ax2 = plt.subplot(2, len(self.datasets), len(self.datasets) + col + 1)
            for radius in sorted(dataset_df['scotoma_radius'].unique()):
                mask = (dataset_df['scotoma_radius'] == radius)
                if self.logpolar_apply is not None:
                    mask = mask & (dataset_df['logpolar_apply'] == self.logpolar_apply)
                
                data_to_plot = dataset_df[mask].sort_values('scotoma_sharpness')
                ax2.plot(data_to_plot['scotoma_sharpness'], data_to_plot['val_acc'],
                        'o-', label=f'radius={radius:.3f}')
            self.setup_plot_style(ax2, dataset, 'Scotoma Sharpness', 'Validation Accuracy')
            axes['sharpness'] = ax2
        else:
            print(f"\nSkipping sharpness plot for {dataset} - only one sharpness value: {unique_sharpness[0]}")
            # Create an empty subplot to maintain grid structure
            ax2 = plt.subplot(2, len(self.datasets), len(self.datasets) + col + 1)
            ax2.set_visible(False)
            axes['sharpness'] = ax2

        return axes

    def _apply_y_limits(self, radius_axes, sharpness_axes, y_limits):
        """Apply consistent y-limits to all plots"""
        for plot_type, axes in [('radius', radius_axes), ('sharpness', sharpness_axes)]:
            padding = 0.05 * (y_limits[plot_type][1] - y_limits[plot_type][0])
            y_min = y_limits[plot_type][0] - padding
            y_max = y_limits[plot_type][1] + padding
            for ax in axes:
                ax.set_ylim(y_min, y_max)

    def _create_shared_legends(self, radius_ax, sharpness_ax):
        """Create shared legends for radius and sharpness plots"""
        radius_ax.legend(bbox_to_anchor=(1.05, 1.0),
                        loc='upper left',
                        fontsize=15, 
                        title='Sharpness Values', 
                        title_fontsize=15)
        
        sharpness_ax.legend(bbox_to_anchor=(1.05, 1.0),
                           loc='upper left',
                           fontsize=15, 
                           title='Radius Values', 
                           title_fontsize=15)
        
        plt.subplots_adjust(right=1)

    def _save_plot(self, group, suffix):
        """Save the current plot in both PNG and SVG formats"""
        # Remove any existing logpolar suffix from the group name
        base_filename = group.replace('_logpolar_True', '').replace('_logpolar_False', '')
        
        # Add logpolar suffix once
        if self.logpolar_apply is not None:
            base_filename += f"_logp_{self.logpolar_apply}"
            
        # Add the analysis suffix
        base_filename += suffix.replace('.png', '')
        
        # Save as PNG
        plt.savefig(base_filename + '.png', 
                   bbox_inches='tight', 
                   dpi=100)
        
        # Save as SVG with parameters for better editability
        plt.savefig(base_filename + '.svg', 
                   bbox_inches='tight', 
                   format='svg',
                   transparent=True,
                   bbox_extra_artists=None,
                   metadata={'Creator': '', 'Date': ''})  # Clear metadata for cleaner file

    def save_to_csv(self):
        """Save the collected data to a CSV file for each experiment group"""
        if self.df is None:
            print("No data to save. Please collect data first using collect_data().")
            return
        
        # Save separate CSV for each experiment group
        for group in self.experiment_groups:
            group_df = self.df[self.df['group'] == group]
            if group_df.empty:
                print(f"No data found for {group}")
                continue
            
            # Create filename with group and logpolar condition
            logp_str = f"_logp_{self.logpolar_apply}" if self.logpolar_apply is not None else ""
            csv_path = f'{group}{logp_str}.csv'
            
            # Save to CSV
            group_df.to_csv(csv_path, index=False)
            print(f"Saved data to: {csv_path}")

    def generate_all_plots(self, logpolar_apply=None):
        """Main function to generate all plots and save data"""
        # Collect all data regardless of logpolar_apply
        self.collect_data(logpolar_apply=None)
        
        # Save data to CSV
        self.save_to_csv()
        
        for idx, group in enumerate(self.experiment_groups):
            print(f"\nGenerating analysis for {group}")
            group_df = self.df[self.df['group'] == group]
            if group_df.empty:
                print(f"No data found for {group}")
                continue
                
            if group == "experiment_LH-H":
                self.create_histograms(group_df, group)
            else:
                self.create_scotoma_plots(group_df, group)

if __name__ == "__main__":
    plotter = WandbPlotter()
    # You can choose one of these options:
    plotter.generate_all_plots(logpolar_apply=True)  # Get all runs
    plotter.generate_all_plots(logpolar_apply=False)  # Get all runs
    # plotter.generate_all_plots(logpolar_apply=True)  # Get only runs with logpolar_apply=True
    # plotter.generate_all_plots(logpolar_apply=False)  # Get only runs with logpolar_apply=False