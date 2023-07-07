import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from scipy import stats
from statannotations.Annotator import Annotator

px = 1 / plt.rcParams['figure.dpi']

sns.set_theme(
    style='darkgrid',
    font_scale=2,
    rc={
        'figure.figsize': (1920 * px, 1080 * px),
        'axes.labelsize': 48,
        'xtick.labelsize': 36,
        'ytick.labelsize': 36,
        'legend.fontsize': 36,
        'grid.linewidth': 2,
    }
)


date_code = {
    # 0: '2023-06-19-11-18-41',
    0: '2023-06-28-15-40-30',
    5: '2023-06-12-17-43-24',
    10: '2023-06-19-11-18-37',
    15: '2023-06-22-21-22-21',
}


def read_data(name: str) -> pd.DataFrame:
    df_list = []
    for num_obstacles in date_code:
        df = pd.read_csv(os.path.join('data', f'{name}_{date_code[num_obstacles]}.csv'))
        df['Obstacles'] = num_obstacles
        df_list.append(df)
    return pd.concat(df_list, ignore_index=True)


def plot_train_box():
    df = read_data('evaluation')
    df = df[(df['Users'] == 5) & (df['Obstacles'] == 5) & (df['Agent'] != 'Farthest')]
    plot = sns.boxplot(
        data=df,
        x='Day', y='Reward', hue='Agent',
    )
    plot.legend(facecolor='white', loc='lower right')

    plt.tight_layout()
    plt.show()


def plot_train_line():
    df = read_data('evaluation')
    df = df[(df['Users'] == 5) & (df['Obstacles'] == 5) & (df['Agent'] != 'Farthest')]
    plot = sns.lineplot(
        data=df,
        x='Day', y='Reward', hue='Agent', style='Agent',
        markers=True, markersize=24, linewidth=3,
    )
    plot.legend(facecolor='white', loc='lower right', markerscale=3)

    plt.tight_layout()
    plt.show()


def plot_stat():
    df = read_data('evaluation')
    df = df[(df['Users'] == 5) & (df['Obstacles'] == 5) & (df['Day'] == 5) & (df['Agent'] != 'Farthest')]
    plot = sns.boxplot(
        data=df,
        x='Reward', y='Agent',
        orient='h',
    )
    plot.set(ylabel=None)

    pairs = [
        ('Independent', 'FingerprintAttention'),
        ('WallNearest', 'FingerprintAttention'),
    ]
    annotator = Annotator(plot, data=df, x='Reward', y='Agent', pairs=pairs, orient='h')
    annotator.configure(test=None, text_format='star')
    annotator.set_pvalues([stats.wilcoxon(
        df[
            (df['Obstacles'] == 5) & (df['Users'] == 5) & (df['Agent'] == a)
        ]['Reward'],
        df[
            (df['Obstacles'] == 5) & (df['Users'] == 5) & (df['Agent'] == b)
        ]['Reward'],
        alternative='less'
    )[1] for a, b in pairs])
    annotator.annotate()

    plt.tight_layout()
    plt.show()


def plot_settings():
    sns.set_theme(
        style='darkgrid',
        font_scale=2,
        rc={
            'figure.figsize': (1920 * px, 1080 * px),
            'axes.labelsize': 36,
            'xtick.labelsize': 36,
            'ytick.labelsize': 36,
            'legend.fontsize': 36,
            'grid.linewidth': 2,
        }
    )

    df = read_data('evaluation')
    df = df[(df['Day'] == 5) & (df['Agent'] != 'Farthest')]
    plot = sns.catplot(
        data=df,
        x='Reward', y='Agent', col='Users', row='Obstacles',
        kind='box', margin_titles=True,
        sharex='col',
    )
    plot.set(ylabel=None)

    walls = [0, 5, 10, 15]
    users = [1, 3, 5, 7, 9]
    pairs = [
        ('Independent', 'FingerprintAttention'),
        ('WallNearest', 'FingerprintAttention'),
    ]
    for i in range(len(walls)):
        for j in range(len(users)):
            annotator = Annotator(plot.axes[i, j], data=df, x='Reward', y='Agent', pairs=pairs, orient='h')
            annotator.configure(test=None, text_format='star')
            annotator.set_pvalues([stats.wilcoxon(
                df[
                    (df['Obstacles'] == walls[i]) & (df['Users'] == users[j]) & (df['Agent'] == a)
                ]['Reward'],
                df[
                    (df['Obstacles'] == walls[i]) & (df['Users'] == users[j]) & (df['Agent'] == b)
                ]['Reward'],
                alternative='less'
            )[1] for a, b in pairs])
            annotator.annotate()

    plt.subplots_adjust(left=0.15, bottom=0.07, right=0.98, top=0.97, wspace=0.05, hspace=0.05)
    plt.show()


def plot_environment():
    sns.set_theme(
        style='darkgrid',
        font_scale=2,
        rc={
            'figure.figsize': (1920 * px, 1080 * px),
            'axes.labelsize': 36,
            'xtick.labelsize': 36,
            'ytick.labelsize': 36,
            'legend.fontsize': 36,
            'grid.linewidth': 2,
        }
    )

    df = read_data('evaluation')
    df = df[(df['Day'] == 5) & (df['Users'] == 5) & (df['Agent'] != 'Farthest')]
    plot = sns.catplot(
        data=df,
        x='Agent', y='Reward', row='Obstacles', col='Environment',
        kind='box', margin_titles=True, legend=True,
        sharey='row',
        showfliers=False,
    )
    plot.set_xticklabels([])
    plot.set(xlabel=None)

    environments = range(25)
    walls = [0, 5, 10, 15]
    pairs = [
        ('Independent', 'FingerprintAttention'),
        ('WallNearest', 'FingerprintAttention'),
    ]
    for i in range(len(walls)):
        plot.axes[i, 0].get_yaxis().set_label_coords(-1, 0.5)
        for j in environments:
            annotator = Annotator(plot.axes[i, j], data=df, x='Agent', y='Reward', pairs=pairs)
            annotator.configure(test=None, text_format='star')
            annotator.set_pvalues([stats.wilcoxon(
                df[
                    (df['Obstacles'] == walls[i]) & (df['Environment'] == j) & (df['Agent'] == a)
                    ]['Reward'],
                df[
                    (df['Obstacles'] == walls[i]) & (df['Environment'] == j) & (df['Agent'] == b)
                    ]['Reward'],
                alternative='less'
            )[1] for a, b in pairs])
            annotator.annotate()

    plt.subplots_adjust(left=0.05, bottom=0.01, right=0.98, top=1, wspace=0.08, hspace=0.2)
    plt.show()


def plot_training_time():
    df = read_data('time')
    plot = sns.boxplot(
        data=df[df['Train'] == True], y='Agent', x='Mean Training Time',
    )
    plot.set(xlabel="Training Time (s)")
    plot.set(ylabel=None)

    plt.tight_layout()
    plt.show()


def plot_time_settings():
    sns.set_theme(
        style='darkgrid',
        font_scale=2,
        rc={
            'figure.figsize': (1920 * px, 1080 * px),
            'axes.labelsize': 36,
            'xtick.labelsize': 36,
            'ytick.labelsize': 36,
            'legend.fontsize': 36,
            'grid.linewidth': 2,
        }
    )

    df = read_data('time')
    plot = sns.catplot(
        data=df[df['Train'] == False],
        x='Mean Selection Time', y='Agent', row='Users',
        # x='Agent', y='Mean Selection Time', col='Users',
        kind='box', margin_titles=True, #showfliers=False,
        sharex='col',
    )
    # plot.set_xticklabels(rotation=90)
    # plot.set(xlabel=None)
    plot.set(xlabel="Selection Time (s)")

    plt.xlim([0, 0.25])
    plt.subplots_adjust(left=0.295, bottom=0.08, right=0.94, top=0.99, wspace=0.2, hspace=0.1)
    plt.show()


if __name__ == '__main__':
    # plot_train_box()
    # plot_train_line()
    # plot_stat()
    plot_settings()
    # plot_environment()
    # plot_training_time()
    # plot_time_settings()
