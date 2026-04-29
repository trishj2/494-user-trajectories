import marimo

__generated_with = "0.19.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import seaborn as sns

    sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
    plt.rcParams['figure.dpi'] = 120

    TRAJ_PATH = 'sampled_user_month_traj.parquet'
    return TRAJ_PATH, mo, np, pd, plt, sns


@app.cell
def _(TRAJ_PATH, pd):
    traj = pd.read_parquet(TRAJ_PATH)
    print(f'Trajectory rows: {len(traj):,}  |  unique users: {traj["participantId"].nunique():,}')
    print(f'userMonth range: {traj["userMonth"].min()} – {traj["userMonth"].max()}')
    print(f'calendarDate range: {traj["calendarDate"].min()} – {traj["calendarDate"].max()}')
    traj.head(5)
    return (traj,)


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Activity Level Definition

    The `month_role` column classifies each user-month by their level of activity:
    - **not_active** – zero contributions that month
    - **single_post_requestor / single/double/triple_digit_requestor** – only note requests
    - **single_note_rater / single/double/triple/4_digit_rater** – only rated notes
    - **single_note_writer / single/double/triple/4_digit_writer** – wrote at least one note

    For aggregate analyses we collapse these into a simpler ordered activity tier.
    """)
    return


@app.cell
def _(pd, traj):
    _tier_map = {
        'not_active':              'Inactive',
        'single_post_requestor':   'Requestor (1)',
        'single_digit_requestor':  'Requestor (2–9)',
        'double_digit_requestor':  'Requestor (10–99)',
        'triple_digit_requestor':  'Requestor (100+)',
        'single_note_rater':       'Rater (1)',
        'single_digit_rater':      'Rater (2–9)',
        'double_digit_rater':      'Rater (10–99)',
        'triple_digit_rater':      'Rater (100–999)',
        '4_digit_rater':           'Rater (1000+)',
        'single_note_writer':      'Writer (1)',
        'single_digit_writer':     'Writer (2–9)',
        'double_digit_writer':     'Writer (10–99)',
        'triple_digit_writer':     'Writer (100–999)',
        '4_digit_writer':          'Writer (1000+)',
    }

    tier_order = [
        'Inactive',
        'Requestor (1)', 'Requestor (2–9)', 'Requestor (10–99)', 'Requestor (100+)',
        'Rater (1)', 'Rater (2–9)', 'Rater (10–99)', 'Rater (100–999)', 'Rater (1000+)',
        'Writer (1)', 'Writer (2–9)', 'Writer (10–99)', 'Writer (100–999)', 'Writer (1000+)',
    ]

    traj2 = traj.copy()
    traj2['activity_tier'] = pd.Categorical(
        traj2['month_role'].map(_tier_map),
        categories=tier_order, ordered=True
    )
    print(traj2['activity_tier'].value_counts())
    return tier_order, traj2


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Q1 — What proportion of notes, ratings, and requests come from users in each activity level?

    We aggregate `notesWritten`, `notesRated`, and `notesRequested` by activity tier across all active months.
    """)
    return


@app.cell
def _(plt, traj2):
    active = traj2.copy()

    _all_tiers = [t for t in active['activity_tier'].cat.categories if t != 'Inactive']

    _cmap   = plt.cm.get_cmap('tab20', len(_all_tiers))
    _colors = [_cmap(i) for i in range(len(_all_tiers))]

    _bar_keys   = ['all_notes', 'all_ratings', 'all_requests',
                   'ok_notes',  'ok_ratings',  'ok_requests']
    _bar_labels = ['Notes', 'Ratings', 'Requests', 'Notes', 'Ratings', 'Requests']

    _raw = {}
    for _g in _all_tiers:
        _s = active[active['activity_tier'] == _g]
        _raw[_g] = {
            'all_notes':    _s['notesWritten'].sum(),
            'all_ratings':  _s['notesRated'].sum(),
            'all_requests': _s['notesRequested'].sum(),
            'ok_notes':     _s['hits'].sum(),
            'ok_ratings':   (_s['correctHelpfuls'] + _s['correctNotHelpfuls']).sum(),
            'ok_requests':  _s['numRequestsResultingInCrh'].sum(),
        }

    _totals = {k: sum(_raw[g][k] for g in _all_tiers) for k in _bar_keys}
    _pcts   = {g: {k: _raw[g][k] / (_totals[k] + 1e-9) * 100
                   for k in _bar_keys}
               for g in _all_tiers}

    _x = [0, 1, 2, 4, 5, 6]

    _fig, _ax = plt.subplots(figsize=(11, 6))
    _bottoms = [0.0] * 6

    for _g, _c in zip(_all_tiers, _colors):
        _vals = [_pcts[_g][k] for k in _bar_keys]
        _ax.bar(_x, _vals, bottom=_bottoms, color=_c, label=_g, width=0.7, edgecolor='white', linewidth=0.4)
        for _xi, _v, _b in zip(_x, _vals, _bottoms):
            if _v >= 8:
                _ax.text(_xi, _b + _v / 2, f'{_v:.0f}%',
                         ha='center', va='center', fontsize=8, color='white', fontweight='bold')
        _bottoms = [b + v for b, v in zip(_bottoms, _vals)]

    _ax.set_xticks(_x)
    _ax.set_xticklabels(_bar_labels, fontsize=9)

    _ax.axvline(3, color='#222222', linewidth=1.5)

    # Subtle grey gridlines
    _ax.yaxis.grid(True, color='grey', linewidth=0.3, alpha=0.4)
    _ax.set_axisbelow(True)

    _ax.set_ylabel('% of Contributions', fontsize=10)
    _ax.set_ylim(0, 100)
    _ax.set_yticks(range(0, 101, 20))
    _ax.set_yticklabels([f'{v}%' for v in range(0, 101, 20)])
    _ax.set_title('')  # no title

    _ax.legend(title='Activity Level', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    _ax.spines['top'].set_visible(False)
    _ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('q1_stacked_bar.png', bbox_inches='tight')
    plt.show()

    vol = active.groupby('activity_tier', observed=True).agg(
        total_notes_written=('notesWritten',   'sum'),
        total_notes_rated  =('notesRated',     'sum'),
        total_requested    =('notesRequested', 'sum'),
    ).reset_index()
    vol = vol[vol['activity_tier'] != 'Inactive'].copy()
    vol['pct_notes']    = vol['total_notes_written'] / vol['total_notes_written'].sum() * 100
    vol['pct_ratings']  = vol['total_notes_rated']   / vol['total_notes_rated'].sum()   * 100
    vol['pct_requests'] = vol['total_requested']      / vol['total_requested'].sum()      * 100
    return active, vol


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Q2 — What proportion of helpful notes, ratings, and requests come from users in each activity level?

    - **Helpful notes**: notes that earned CRH status — `hits` column.
    - **Helpful/accurate ratings**: ratings matching the community verdict — `correctHelpfuls + correctNotHelpfuls`.
    - **Productive requests**: requests that resulted in a CRH note — `numRequestsResultingInCrh`.
    """)
    return


@app.cell
def _(active, plt, vol):
    _helpful = active.groupby('activity_tier', observed=True).agg(
        total_hits       =('hits',                      'sum'),
        total_correct_h  =('correctHelpfuls',           'sum'),
        total_correct_nh =('correctNotHelpfuls',        'sum'),
        total_crh_requests=('numRequestsResultingInCrh','sum'),
    ).reset_index()

    _helpful['total_correct_ratings'] = _helpful['total_correct_h'] + _helpful['total_correct_nh']
    _helpful = _helpful[_helpful['activity_tier'] != 'Inactive'].copy()
    _helpful['pct_helpful_notes']   = _helpful['total_hits']            / _helpful['total_hits'].sum()            * 100
    _helpful['pct_correct_ratings'] = _helpful['total_correct_ratings'] / _helpful['total_correct_ratings'].sum() * 100
    _helpful['pct_crh_requests']    = _helpful['total_crh_requests']    / (_helpful['total_crh_requests'].sum() + 1e-9) * 100

    _helpful = _helpful.merge(
        vol[['activity_tier', 'total_notes_written', 'total_notes_rated', 'total_requested']],
        on='activity_tier', how='left'
    )
    _helpful['hit_rate']      = _helpful['total_hits']            / (_helpful['total_notes_written'] + 1e-9) * 100
    _helpful['accuracy_rate'] = _helpful['total_correct_ratings'] / (_helpful['total_notes_rated']   + 1e-9) * 100
    _helpful['crh_req_rate']  = _helpful['total_crh_requests']    / (_helpful['total_requested']     + 1e-9) * 100

    _fig, _axes = plt.subplots(2, 3, figsize=(20, 10))

    for _ax, _col, _title, _color in [
        (_axes[0, 0], 'pct_helpful_notes',   'Share of Helpful Notes (CRH)',              'steelblue'),
        (_axes[0, 1], 'pct_correct_ratings',  'Share of Accurate Ratings',                 'seagreen'),
        (_axes[0, 2], 'pct_crh_requests',     'Share of Productive Requests (→CRH)',       'coral'),
        (_axes[1, 0], 'hit_rate',             'Hit Rate (CRH notes / notes written)',       'steelblue'),
        (_axes[1, 1], 'accuracy_rate',        'Accuracy Rate (correct ratings / ratings)',  'seagreen'),
        (_axes[1, 2], 'crh_req_rate',         'CRH Request Rate (CRH results / requests)', 'coral'),
    ]:
        _bars = _ax.barh(_helpful['activity_tier'].astype(str), _helpful[_col], color=_color, alpha=0.8)
        _ax.set_xlabel('% of Total' if _axes.flat[0] == _ax or _axes.flat[1] == _ax or _axes.flat[2] == _ax else '% (per-tier rate)')
        _ax.set_title(_title)
        for _bar, _val in zip(_bars, _helpful[_col]):
            if _val > 0.3:
                _ax.text(_val + 0.2, _bar.get_y() + _bar.get_height() / 2,
                         f'{_val:.1f}%', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig('q2_helpful_by_activity.png', bbox_inches='tight')
    plt.show()

    print("\n=== Helpful Notes by Activity Tier ===")
    print(_helpful[['activity_tier', 'total_hits', 'pct_helpful_notes', 'hit_rate']].to_string(index=False))
    print("\n=== Accurate Ratings by Activity Tier ===")
    print(_helpful[['activity_tier', 'total_correct_ratings', 'pct_correct_ratings', 'accuracy_rate']].to_string(index=False))
    print("\n=== Productive Requests (→CRH) by Activity Tier ===")
    print(_helpful[['activity_tier', 'total_crh_requests', 'pct_crh_requests', 'crh_req_rate']].to_string(index=False))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Q3 — How long do users stay before attrition?

    We define **attrition** as the user's last active month (i.e., the last `userMonth` where `activeMonth == True`). We then examine:
    1. **Tenure by entry cohort** (calendar year the user joined)
    2. **Tenure by entry activity level** (the user's `month_role` in `userMonth == 0`)
    """)
    return


@app.cell
def _(pd, traj2):
    entry_info = traj2[traj2['userMonth'] == 0][
        ['participantId', 'calendarDate', 'month_role', 'activity_tier']
    ].copy().rename(columns={
        'calendarDate':   'entry_date',
        'month_role':     'entry_role',
        'activity_tier':  'entry_tier',
    })
    entry_info['entry_year'] = pd.to_datetime(entry_info['entry_date']).dt.year

    last_active = (
        traj2[traj2['activeMonth']]
        .groupby('participantId')['userMonth']
        .max()
        .reset_index()
        .rename(columns={'userMonth': 'last_active_month'})
    )

    print(f'Users with at least one active month: {len(last_active):,}')

    user_attrition = entry_info.merge(last_active, on='participantId', how='left')
    user_attrition['last_active_month'] = user_attrition['last_active_month'].fillna(0)

    max_month = int(traj2['userMonth'].max())
    user_attrition['censored'] = user_attrition['last_active_month'] == max_month

    print(f'Total users: {len(user_attrition):,}')
    print(f'Censored (still active at end): {user_attrition["censored"].sum():,} ({user_attrition["censored"].mean()*100:.1f}%)')
    print(f'\nEntry year distribution:\n{user_attrition["entry_year"].value_counts().sort_index()}')
    return max_month, user_attrition


@app.cell
def _(mo):
    mo.md(r"""
    ### Q3a — Tenure by Entry Cohort (Calendar Year)

    Fraction of users still active at each userMonth, grouped by entry year.
    """)
    return


@app.cell
def _(max_month, np, plt, traj2, user_attrition):
    _cohort_years = sorted(user_attrition['entry_year'].unique())
    _months = np.arange(0, max_month + 1)

    _fig, _ax = plt.subplots(figsize=(12, 6))
    _palette = plt.cm.tab10(np.linspace(0, 0.9, len(_cohort_years)))

    for _color, _year in zip(_palette, _cohort_years):
        _cohort_users = user_attrition[user_attrition['entry_year'] == _year]['participantId'].values
        _n = len(_cohort_users)
        if _n < 10:
            continue
        _ctraj = traj2[traj2['participantId'].isin(_cohort_users)]
        _survival = []
        for _t in _months:
            _md = _ctraj[_ctraj['userMonth'] == _t]
            _survival.append(_md['activeMonth'].sum() / _n * 100 if len(_md) > 0 else 0)
        _ax.plot(_months, _survival, label=f'{_year} (n={_n})', color=_color, linewidth=1.8)

    _ax.set_xlabel('Months Since Joining')
    _ax.set_ylabel('% of Cohort Still Active')
    _ax.set_title('User Retention by Entry Cohort (Calendar Year)')
    _ax.legend(title='Entry Year', bbox_to_anchor=(1.02, 1), loc='upper left')
    _ax.set_xlim(0, max_month)
    _ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig('q3a_retention_by_cohort.png', bbox_inches='tight')
    plt.show()

    print("Median last active month by entry year:")
    print(user_attrition.groupby('entry_year')['last_active_month'].median().to_string())
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Q3b — Tenure by Entry Activity Level

    Survival curves grouped by the user's activity tier in their first month (`userMonth == 0`).
    """)
    return


@app.cell
def _(max_month, np, plt, tier_order, traj2, user_attrition):
    _entry_tiers = [t for t in tier_order if t != 'Inactive']
    _months = np.arange(0, max_month + 1)

    _fig, _axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left panel: all granular tiers
    _palette_tiers = plt.cm.tab20(np.linspace(0, 1, len(_entry_tiers)))
    for _color, _tier in zip(_palette_tiers, _entry_tiers):
        _cohort_users = user_attrition[user_attrition['entry_tier'] == _tier]['participantId'].values
        _n = len(_cohort_users)
        if _n < 30:
            continue
        _ctraj = traj2[traj2['participantId'].isin(_cohort_users)]
        _survival = []
        for _t in _months:
            _md = _ctraj[_ctraj['userMonth'] == _t]
            _survival.append(_md['activeMonth'].sum() / _n * 100 if len(_md) > 0 else 0)
        _axes[0].plot(_months, _survival, label=f'{_tier} (n={_n})', color=_color, linewidth=1.8)

    _axes[0].set_xlabel('Months Since Joining')
    _axes[0].set_ylabel('% of Group Still Active')
    _axes[0].set_title('Retention by Entry Activity Tier')
    _axes[0].legend(title='Entry Tier', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    _axes[0].set_xlim(0, max_month)
    _axes[0].set_ylim(0, 105)

    # Right panel: broad groups (Requestor / Rater / Writer)
    def _broad(tier):
        if 'Requestor' in str(tier): return 'Requestor'
        if 'Rater'      in str(tier): return 'Rater'
        if 'Writer'     in str(tier): return 'Writer'
        return 'Inactive'

    _ua2 = user_attrition.copy()
    _ua2['broad_entry'] = _ua2['entry_tier'].apply(_broad)

    for _broad_name, _color in [('Requestor', 'coral'), ('Rater', 'steelblue'), ('Writer', 'seagreen')]:
        _cohort_users = _ua2[_ua2['broad_entry'] == _broad_name]['participantId'].values
        _n = len(_cohort_users)
        if _n < 5:
            continue
        _ctraj = traj2[traj2['participantId'].isin(_cohort_users)]
        _survival = []
        for _t in _months:
            _md = _ctraj[_ctraj['userMonth'] == _t]
            _survival.append(_md['activeMonth'].sum() / _n * 100 if len(_md) > 0 else 0)
        _axes[1].plot(_months, _survival, label=f'{_broad_name} (n={_n})', color=_color, linewidth=2.5)

    _axes[1].set_xlabel('Months Since Joining')
    _axes[1].set_ylabel('% of Group Still Active')
    _axes[1].set_title('Retention by Entry Type (Broad)')
    _axes[1].legend(title='Entry Type')
    _axes[1].set_xlim(0, max_month)
    _axes[1].set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig('q3b_retention_by_entry_activity.png', bbox_inches='tight')
    plt.show()

    print("Median last active month by entry tier:")
    print(user_attrition.groupby('entry_tier')['last_active_month'].median().sort_values(ascending=False).to_string())

    user_attrition2 = user_attrition.copy()
    user_attrition2['broad_entry'] = user_attrition2['entry_tier'].apply(_broad)
    return (user_attrition2,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Q3c — Tenure by Entry Cohort AND Entry Activity Level (Combined)
    """)
    return


@app.cell
def _(plt, sns, user_attrition2):
    _plot_data = user_attrition2[user_attrition2['broad_entry'] != 'Inactive'].copy()

    _fig, _ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(
        data=_plot_data,
        x='entry_year', y='last_active_month',
        hue='broad_entry',
        palette={'Requestor': 'coral', 'Rater': 'steelblue', 'Writer': 'seagreen'},
        ax=_ax, width=0.6, flierprops=dict(markersize=2, alpha=0.3)
    )
    _ax.set_xlabel('Entry Year')
    _ax.set_ylabel('Last Active Month (months since joining)')
    _ax.set_title('Tenure Distribution by Entry Year and Entry Type')
    _ax.legend(title='Entry Type', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('q3c_tenure_combined.png', bbox_inches='tight')
    plt.show()

    print("Mean tenure by entry year × entry type:")
    print(
        _plot_data.groupby(['entry_year', 'broad_entry'])['last_active_month']
        .mean().unstack().round(1).to_string()
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Q3 Interpretation:**

    Users who joined earlier in the program's history naturally have longer *possible* tenures simply because the observation window is wider for them. Controlling for that, users who entered as **writers** (those who wrote a note in their first month) tend to stay active longer than pure **raters**, who in turn stay longer than users who only made **requests**. This makes intuitive sense: writing a note requires more investment and signals stronger commitment to the platform. Requestors who never rated or wrote notes have very short tenures — most are one-time visitors. Across all cohorts, the survival curves show a steep drop in the first few months, with a long tail of loyal power users who remain active for a year or more.
    """)
    return


if __name__ == "__main__":
    app.run()
