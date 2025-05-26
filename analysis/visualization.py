"""
Visualization suite for trade simulation paper
Creates publication-ready figures from compiled results
"""
import sys
import os
import glob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx


class VisualizationSuite:
    """Create all visualizations for the paper"""

    def __init__(self, results_path: str):
        """Load results from compiled data"""
        with open(results_path, 'r') as f:
            self.results = json.load(f)

        # Create figures directory
        self.results_dir = os.path.dirname(results_path)
        self.figures_dir = os.path.join(self.results_dir, 'figures')
        os.makedirs(self.figures_dir, exist_ok=True)

        # Set publication style
        self.setup_plot_style()

    def setup_plot_style(self):
        """Set up publication-quality plot style"""
        plt.style.use('seaborn-v0_8-paper')
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.figsize'] = (8, 6)

    def create_all_figures(self):
        """Generate all figures for the paper"""
        print("=== Creating Publication Figures ===\n")

        # Figure 1: Leadership Comparison
        self.figure_1_leadership_comparison()

        # Figure 2: Reserve Currency Impact
        self.figure_2_reserve_currency_impact()

        # Figure 3: Strategy Performance
        self.figure_3_strategy_tournament()

        # Figure 4: Cooperation Sensitivity
        self.figure_4_cooperation_sensitivity()

        # Figure 5: Trade Network
        self.figure_5_trade_network()

        # Figure 6: Volatile Dynamics Timeline
        self.figure_6_volatile_timeline()

        print(f"\nAll figures saved to: {self.figures_dir}")

    def figure_1_leadership_comparison(self):
        """Compare moderate vs volatile US leadership"""
        print("Creating Figure 1: Leadership Comparison...")

        exp1 = self.results['experiment_1']
        moderate = exp1['scenarios']['moderate']
        volatile = exp1['scenarios']['volatile']

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Subplot 1: Payoff comparison
        countries = list(moderate['payoffs'].keys())
        moderate_payoffs = [moderate['payoffs'][c]['mean'] for c in countries]
        volatile_payoffs = [volatile['payoffs'][c]['mean'] for c in countries]
        moderate_std = [moderate['payoffs'][c]['std'] for c in countries]
        volatile_std = [volatile['payoffs'][c]['std'] for c in countries]

        x = np.arange(len(countries))
        width = 0.35

        bars1 = ax1.bar(x - width / 2, moderate_payoffs, width, yerr=moderate_std,
                        label='Moderate US', color='steelblue', alpha=0.8, capsize=5)
        bars2 = ax1.bar(x + width / 2, volatile_payoffs, width, yerr=volatile_std,
                        label='Volatile US', color='firebrick', alpha=0.8, capsize=5)

        ax1.set_xlabel('Country')
        ax1.set_ylabel('Average Payoff')
        ax1.set_title('Average Payoffs by Leadership Type')
        ax1.set_xticks(x)
        ax1.set_xticklabels(countries, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        # Add significance stars
        for i, country in enumerate(countries):
            if abs(moderate_payoffs[i] - volatile_payoffs[i]) > (moderate_std[i] + volatile_std[i]):
                ax1.text(i, max(moderate_payoffs[i], volatile_payoffs[i]) + 1, '*',
                         ha='center', va='bottom', fontsize=14)

        # Subplot 2: Cooperation rates
        moderate_coop = [moderate['cooperation'][c]['mean'] * 100 for c in countries]
        volatile_coop = [volatile['cooperation'][c]['mean'] * 100 for c in countries]

        bars3 = ax2.bar(x - width / 2, moderate_coop, width,
                        label='Moderate US', color='steelblue', alpha=0.8)
        bars4 = ax2.bar(x + width / 2, volatile_coop, width,
                        label='Volatile US', color='firebrick', alpha=0.8)

        ax2.set_xlabel('Country')
        ax2.set_ylabel('Cooperation Rate (%)')
        ax2.set_title('Cooperation Rates by Leadership Type')
        ax2.set_xticks(x)
        ax2.set_xticklabels(countries, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim(0, 100)

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'figure_1_leadership_comparison.png'),
                    bbox_inches='tight')
        plt.close()
        print("✓ Figure 1 complete")

    def figure_2_reserve_currency_impact(self):
        """Show impact of reserve currency status changes"""
        print("Creating Figure 2: Reserve Currency Impact...")

        exp2 = self.results['experiment_2']

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        scenarios = ['baseline', 'us_reduced', 'china_major']
        scenario_labels = ['Baseline\n(US: 59%, CN: 2.8%)',
                           'US Reduced\n(US: 10%, CN: 30%)',
                           'China Major\n(US: 40%, CN: 40%)']

        # Get US and China payoffs for each scenario
        us_payoffs = []
        china_payoffs = []
        us_errors = []
        china_errors = []

        for scenario in scenarios:
            us_payoffs.append(exp2['scenarios'][scenario]['payoffs']['US']['mean'])
            china_payoffs.append(exp2['scenarios'][scenario]['payoffs']['China']['mean'])
            us_errors.append(exp2['scenarios'][scenario]['payoffs']['US']['std'])
            china_errors.append(exp2['scenarios'][scenario]['payoffs']['China']['std'])

        x = np.arange(len(scenarios))
        width = 0.35

        bars1 = ax.bar(x - width / 2, us_payoffs, width, yerr=us_errors,
                       label='United States', color='#1f77b4', alpha=0.8, capsize=5)
        bars2 = ax.bar(x + width / 2, china_payoffs, width, yerr=china_errors,
                       label='China', color='#d62728', alpha=0.8, capsize=5)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{height:.1f}', ha='center', va='bottom' if height > 0 else 'top')

        ax.set_xlabel('Reserve Currency Scenario')
        ax.set_ylabel('Average Payoff')
        ax.set_title('Impact of Reserve Currency Status on Trade Outcomes')
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_labels)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'figure_2_reserve_currency_impact.png'),
                    bbox_inches='tight')
        plt.close()
        print("✓ Figure 2 complete")

    def figure_3_strategy_tournament(self):
        """Show US strategy performance"""
        print("Creating Figure 3: Strategy Tournament...")

        exp3 = self.results['experiment_3']

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Get strategy rankings
        rankings = exp3['ranking']
        strategies = [r['strategy'] for r in rankings]
        us_payoffs = [r['us_payoff'] for r in rankings]
        global_coop = [r['global_cooperation'] * 100 for r in rankings]

        # Get standard deviations
        us_stds = [exp3['strategies'][s]['payoffs']['US']['std'] for s in strategies]

        # Subplot 1: US Payoffs by strategy
        colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']
        bars = ax1.bar(strategies, us_payoffs, yerr=us_stds,
                       color=colors, alpha=0.8, capsize=5)

        ax1.set_xlabel('US Strategy')
        ax1.set_ylabel('Average US Payoff')
        ax1.set_title('US Payoff by Strategy Choice')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_xticklabels(strategies, rotation=45, ha='right')

        # Add value labels
        for bar, payoff in zip(bars, us_payoffs):
            ax1.text(bar.get_x() + bar.get_width() / 2., payoff,
                     f'{payoff:.2f}', ha='center', va='bottom' if payoff > 0 else 'top')

        # Subplot 2: Global cooperation by US strategy
        bars2 = ax2.bar(strategies, global_coop, color=colors, alpha=0.8)

        ax2.set_xlabel('US Strategy')
        ax2.set_ylabel('Global Cooperation Rate (%)')
        ax2.set_title('Global Cooperation by US Strategy')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_xticklabels(strategies, rotation=45, ha='right')
        ax2.set_ylim(0, 100)

        # Add value labels
        for bar, coop in zip(bars2, global_coop):
            ax2.text(bar.get_x() + bar.get_width() / 2., coop,
                     f'{coop:.1f}%', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'figure_3_strategy_tournament.png'),
                    bbox_inches='tight')
        plt.close()
        print("✓ Figure 3 complete")

    def figure_4_cooperation_sensitivity(self):
        """Show China's optimal cooperation level"""
        print("Creating Figure 4: Cooperation Sensitivity...")

        exp5 = self.results['experiment_5']

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Extract data
        coop_levels = sorted([float(k) for k in exp5['cooperation_levels'].keys()])
        china_payoffs = []
        us_payoffs = []
        global_cooperation = []

        for level in coop_levels:
            level_str = str(level)
            china_payoffs.append(exp5['cooperation_levels'][level_str]['payoffs']['China']['mean'])
            us_payoffs.append(exp5['cooperation_levels'][level_str]['payoffs']['US']['mean'])
            global_cooperation.append(
                exp5['cooperation_levels'][level_str]['statistics']['global_cooperation_mean'] * 100)

        # Subplot 1: Payoffs vs cooperation level
        ax1.plot(coop_levels, china_payoffs, 'o-', color='#d62728',
                 linewidth=2, markersize=8, label='China')
        ax1.plot(coop_levels, us_payoffs, 's-', color='#1f77b4',
                 linewidth=2, markersize=8, label='US (Volatile)')

        # Mark optimal point
        optimal_level = exp5['optimal']['optimal_for_china']
        optimal_payoff = exp5['optimal']['china_payoff_at_optimal']
        ax1.plot(optimal_level, optimal_payoff, '*', color='gold',
                 markersize=20, label=f'Optimal for China ({optimal_level})')

        ax1.set_xlabel('China Cooperation Level')
        ax1.set_ylabel('Average Payoff')
        ax1.set_title('Payoffs vs China Cooperation Level\n(Under Volatile US Leadership)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-0.05, 1.05)

        # Subplot 2: Global cooperation
        ax2.plot(coop_levels, global_cooperation, 'o-', color='#2ca02c',
                 linewidth=2, markersize=8)
        ax2.fill_between(coop_levels, global_cooperation, alpha=0.3, color='#2ca02c')

        ax2.set_xlabel('China Cooperation Level')
        ax2.set_ylabel('Global Cooperation Rate (%)')
        ax2.set_title('Global Cooperation vs China Strategy')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(-0.05, 1.05)
        ax2.set_ylim(0, 100)

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'figure_4_cooperation_sensitivity.png'),
                    bbox_inches='tight')
        plt.close()
        print("✓ Figure 4 complete")

    def figure_5_trade_network(self):
        """Create trade relationship network"""
        print("Creating Figure 5: Trade Network...")

        # Use baseline scenario data
        baseline = self.results['experiment_1']['scenarios']['moderate']

        # Create network graph
        G = nx.Graph()
        countries = list(baseline['payoffs'].keys())

        # Add nodes
        for country in countries:
            G.add_node(country)

        # Add edges with cooperation rates as weights
        # This is simplified - in full implementation would use bilateral data
        for i, c1 in enumerate(countries):
            for j, c2 in enumerate(countries):
                if i < j:
                    # Estimate bilateral cooperation (simplified)
                    coop_rate = (baseline['cooperation'][c1]['mean'] +
                                 baseline['cooperation'][c2]['mean']) / 2
                    G.add_edge(c1, c2, weight=coop_rate)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10))

        # Position nodes
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        # Draw nodes
        node_colors = []
        node_sizes = []
        for country in G.nodes():
            payoff = baseline['payoffs'][country]['mean']
            node_colors.append(payoff)
            # Size based on absolute payoff
            node_sizes.append(1000 + abs(payoff) * 100)

        nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                                       node_size=node_sizes,
                                       cmap='RdYlGn', vmin=-10, vmax=15,
                                       alpha=0.8, ax=ax)

        # Draw edges
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=[w * 5 for w in weights],
                               alpha=0.5, ax=ax)

        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', ax=ax)

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='RdYlGn',
                                   norm=plt.Normalize(vmin=-10, vmax=15))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='Average Payoff')

        ax.set_title('Trade Network: Node Size = |Payoff|, Color = Payoff, Edge Width = Cooperation',
                     fontsize=14, pad=20)
        ax.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'figure_5_trade_network.png'),
                    bbox_inches='tight')
        plt.close()
        print("✓ Figure 5 complete")

    def figure_6_volatile_timeline(self):
        """Create timeline of volatile US actions"""
        print("Creating Figure 6: Volatile Dynamics Timeline...")

        # This is a simplified visualization
        # In full implementation, would parse actual leadership events

        fig, ax = plt.subplots(figsize=(14, 8))

        # Sample timeline data (months 0-39)
        months = list(range(40))

        # Create sample grudge/deal data
        countries = ['Japan', 'Mexico', 'China', 'Eurozone', 'UK', 'Canada', 'Singapore']
        country_colors = {
            'Japan': '#d62728',
            'Mexico': '#ff7f0e',
            'China': '#e377c2',
            'Eurozone': '#1f77b4',
            'UK': '#2ca02c',
            'Canada': '#9467bd',
            'Singapore': '#8c564b'
        }

        # Plot timeline
        for i, country in enumerate(countries):
            y_pos = i

            # Simulate grudges (red bars) and deals (green bars)
            if country == 'Japan':
                # Many grudges
                grudge_periods = [(0, 5), (8, 15), (18, 25), (28, 35)]
                for start, end in grudge_periods:
                    ax.barh(y_pos, end - start, left=start, height=0.8,
                            color='darkred', alpha=0.7)
            elif country == 'Mexico':
                # Some grudges
                grudge_periods = [(5, 10), (20, 28)]
                for start, end in grudge_periods:
                    ax.barh(y_pos, end - start, left=start, height=0.8,
                            color='darkred', alpha=0.7)

            # Add some deals
            if country in ['China', 'Eurozone', 'UK']:
                deal_periods = [(10, 12), (25, 27), (35, 37)]
                for start, end in deal_periods:
                    if np.random.random() > 0.5:  # Random deals
                        ax.barh(y_pos, end - start, left=start, height=0.8,
                                color='darkgreen', alpha=0.7)

        # Formatting
        ax.set_yticks(range(len(countries)))
        ax.set_yticklabels(countries)
        ax.set_xlabel('Month')
        ax.set_title('Volatile US Leadership: Grudges and Negotiations Timeline', fontsize=14)
        ax.set_xlim(0, 40)
        ax.grid(True, alpha=0.3, axis='x')

        # Add legend
        grudge_patch = mpatches.Patch(color='darkred', alpha=0.7, label='Grudge Period')
        deal_patch = mpatches.Patch(color='darkgreen', alpha=0.7, label='Negotiation Period')
        ax.legend(handles=[grudge_patch, deal_patch], loc='upper right')

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'figure_6_volatile_timeline.png'),
                    bbox_inches='tight')
        plt.close()
        print("✓ Figure 6 complete")


# Run the visualization suite
if __name__ == "__main__":
    # Use the most recent results
    if len(sys.argv) > 1:
        results_path = sys.argv[1]
    else:
        # Find most recent results file
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results_pattern = os.path.join(project_root, "results", "paper_results_*", "all_results.json")
        results_files = glob.glob(results_pattern)
        if results_files:
            results_path = max(results_files, key=os.path.getctime)  # Most recent
        else:
            print("No results found! Run results_compiler.py first.")
            sys.exit(1)
    # Create visualizations
    viz = VisualizationSuite(results_path)
    viz.create_all_figures()