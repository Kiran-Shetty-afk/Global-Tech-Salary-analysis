import io
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score

sns.set_theme(style='whitegrid', palette='Spectral')

st.set_page_config(page_title='Global Tech Salary Hub', layout='wide')
st.title('💻 Global Tech Salary Hub')
st.markdown(
    'This app helps you explore global data science compensation trends, identify high-paying roles, and compare salary patterns by country, experience, company size, and work model.'
)

# ---------------------------
# Helpers
# ---------------------------

def currency(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 'N/A'
    return f'${value:,.2f}'


def percent(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 'N/A'
    return f'{value:.1f}%'


def int_format(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 'N/A'
    return f'{int(value):,}'


def safe_index(options, value):
    try:
        return options.index(value)
    except ValueError:
        return 0


def plot_pie(labels, sizes, title):
    fig, ax = plt.subplots(figsize=(6, 6))
    if not labels or len(labels) == 0 or sum(sizes) == 0:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=14)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        return fig
    colors = sns.color_palette('Spectral', len(labels))
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, textprops={'color': 'black'})
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axis('equal')
    plt.tight_layout()
    return fig


def plot_bar(df, x, y, title, xlabel, ylabel, horizontal=False, palette='Spectral'):
    fig, ax = plt.subplots(figsize=(10, 5))
    if df.empty or len(df[y]) == 0:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=14)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        return fig
    colors = sns.color_palette(palette, len(df))
    values = df[y].values
    if horizontal:
        bars = ax.barh(df[x], values, color=colors)
        for bar in bars:
            ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                    f'${bar.get_width():,.0f}', va='center', fontsize=9)
        ax.set_ylabel(xlabel)
        ax.set_xlabel(ylabel)
    else:
        bars = ax.bar(df[x], values, color=colors)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                    f'${bar.get_height():,.0f}', ha='center', fontsize=9)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.xticks(rotation=45, ha='right')
    ax.set_title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_heatmap(pivot, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(12, 6))
    if pivot.empty:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=14)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        return fig
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='Spectral', linewidths=.5, cbar_kws={'label': 'Median Salary (USD)'}, ax=ax)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    return fig


def get_percentile_rank(series, value):
    if len(series) == 0:
        return 0.0
    return float(np.searchsorted(np.sort(series), value, side='right') / len(series) * 100)


def get_available_excel_engine():
    try:
        import xlsxwriter  # noqa: F401
        return 'xlsxwriter'
    except ImportError:
        pass
    try:
        import openpyxl  # noqa: F401
        return 'openpyxl'
    except ImportError:
        return None


def map_years_to_experience(years):
    if years <= 2:
        return 'Entry'
    if years <= 5:
        return 'Mid'
    if years <= 10:
        return 'Senior'
    return 'Exec'


@st.cache_data
def load_data():
    df = pd.read_csv('data_science_salaries.csv')
    exp_map = {
        'Entry-level': 'Entry',
        'Mid-level': 'Mid',
        'Senior-level': 'Senior',
        'Executive-level': 'Exec'
    }
    df['experience_level'] = df['experience_level'].map(exp_map)
    df['job_title'] = df['job_title'].astype(str)
    df['company_location'] = df['company_location'].astype(str).str.strip()
    df['work_models'] = df['work_models'].astype(str)
    df['company_size'] = df['company_size'].astype(str)
    return df


@st.cache_resource
def load_model():
    try:
        return joblib.load('salary_predictor.pkl')
    except Exception as exc:
        st.error('Unable to load salary_predictor.pkl. Place it in the project root and reload the app.')
        raise exc


@st.cache_data
def prepare_dataset(df):
    df_copy = df.copy()
    top_jobs = df_copy['job_title'].value_counts().nlargest(20).index.tolist()
    top_locations = df_copy['company_location'].value_counts().nlargest(20).index.tolist()
    df_copy['job_title_grouped'] = df_copy['job_title'].where(df_copy['job_title'].isin(top_jobs), 'Other')
    df_copy['company_location_grouped'] = df_copy['company_location'].where(df_copy['company_location'].isin(top_locations), 'Other')
    return df_copy, top_jobs, top_locations


@st.cache_data
def compute_cv_scores(df):
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestRegressor

    df_copy, _, _ = prepare_dataset(df)
    features = ['job_title_grouped', 'experience_level', 'work_models', 'company_location_grouped', 'company_size']
    X = df_copy[features]
    y = df_copy['salary_in_usd']
    pipeline = Pipeline([
        ('preprocessor', ColumnTransformer(transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), features)
        ])),
        ('regressor', RandomForestRegressor(n_estimators=120, random_state=42, min_samples_split=6))
    ])
    cv_r2 = cross_val_score(pipeline, X, y, cv=5, scoring='r2', n_jobs=-1)
    cv_mae = -cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    return cv_r2, cv_mae


raw_df = load_data()
model = load_model()
prepared_df, top_jobs, top_locations = prepare_dataset(raw_df)
job_options = top_jobs + ['Other']
location_options = top_locations + ['Other']

all_countries = ['All'] + sorted(raw_df['company_location'].dropna().unique())
all_models = ['All'] + sorted(raw_df['work_models'].dropna().unique())
all_sizes = ['All'] + sorted(raw_df['company_size'].dropna().unique())
all_experience = ['All', 'Entry', 'Mid', 'Senior', 'Exec', 'Senior+']

company_size_info = (
    raw_df.groupby('company_size')['salary_in_usd']
    .agg(avg_salary='mean', count='count')
    .reset_index()
)

with st.sidebar:
    st.header('About This Data')
    st.write('Global salary records for data science and analytics roles from 2020 to 2024.')
    st.write('Use filters and preset buttons to explore high-paying roles, regions, and career progression.')
    st.markdown('---')
    st.subheader('Quick filter presets')
    if st.button('Remote Only'):
        st.session_state.selected_country = 'All'
        st.session_state.selected_model = 'Remote'
        st.session_state.selected_size = 'All'
        st.session_state.selected_experience = 'All'
    if st.button('Senior+'):
        st.session_state.selected_country = 'All'
        st.session_state.selected_model = 'All'
        st.session_state.selected_size = 'All'
        st.session_state.selected_experience = 'Senior+'
    if st.button('Large Companies'):
        st.session_state.selected_country = 'All'
        st.session_state.selected_model = 'All'
        st.session_state.selected_size = 'Large'
        st.session_state.selected_experience = 'All'
    if st.button('Reset Filters'):
        st.session_state.selected_country = 'All'
        st.session_state.selected_model = 'All'
        st.session_state.selected_size = 'All'
        st.session_state.selected_experience = 'All'
    st.markdown('---')
    st.write('The model uses the same dataset and provides directionally useful compensation estimates.')

filter_defaults = {
    'selected_country': ('All', all_countries),
    'selected_model': ('All', all_models),
    'selected_size': ('All', all_sizes),
    'selected_experience': ('All', all_experience)
}
for key, (default, options) in filter_defaults.items():
    if st.session_state.get(key) not in options:
        st.session_state[key] = default

# Tabs
tab1, tab2 = st.tabs(['📊 Global Analytics', '🤖 AI Salary Predictor'])

with tab1:
    st.markdown('### Explore compensation trends by location, experience, and work model')

    cols = st.columns(4)
    with cols[0]:
        selected_country = st.selectbox('Company Location', all_countries, index=safe_index(all_countries, st.session_state.selected_country))
    with cols[1]:
        selected_model = st.selectbox('Work Model', all_models, index=safe_index(all_models, st.session_state.selected_model))
    with cols[2]:
        selected_size = st.selectbox('Company Size', all_sizes, index=safe_index(all_sizes, st.session_state.selected_size))
    with cols[3]:
        selected_experience = st.selectbox('Experience Level', all_experience, index=safe_index(all_experience, st.session_state.selected_experience))

    st.session_state.selected_country = selected_country
    st.session_state.selected_model = selected_model
    st.session_state.selected_size = selected_size
    st.session_state.selected_experience = selected_experience

    def apply_filters(data):
        filtered = data.copy()
        if selected_country != 'All':
            filtered = filtered[filtered['company_location'] == selected_country]
        if selected_model != 'All':
            filtered = filtered[filtered['work_models'] == selected_model]
        if selected_size != 'All':
            filtered = filtered[filtered['company_size'] == selected_size]
        if selected_experience != 'All':
            if selected_experience == 'Senior+':
                filtered = filtered[filtered['experience_level'].isin(['Senior', 'Exec'])]
            else:
                filtered = filtered[filtered['experience_level'] == selected_experience]
        return filtered

    plot_df = apply_filters(raw_df)
    record_pct = len(plot_df) / len(raw_df) * 100 if len(raw_df) else 0
    st.markdown(f'**Showing {len(plot_df)} records ({record_pct:.1f}% of total) · Active filters: {sum([selected_country != "All", selected_model != "All", selected_size != "All", selected_experience != "All"]) }**')
    st.markdown('---')

    filtered_median = plot_df['salary_in_usd'].median() if len(plot_df) else 0
    filtered_mean = plot_df['salary_in_usd'].mean() if len(plot_df) else 0
    filtered_min = plot_df['salary_in_usd'].min() if len(plot_df) else 0
    filtered_max = plot_df['salary_in_usd'].max() if len(plot_df) else 0
    filtered_range = filtered_max - filtered_min if len(plot_df) else 0

    if plot_df.empty:
        st.warning('No records match the selected filters. Adjust filter values or reset filters to see visual analytics and tables.')
        st.stop()

    overall_avg = raw_df['salary_in_usd'].mean()
    overall_median = raw_df['salary_in_usd'].median()
    overall_range = raw_df['salary_in_usd'].max() - raw_df['salary_in_usd'].min()
    overall_growth = ((raw_df.groupby('work_year')['salary_in_usd'].median().iloc[-1] / raw_df.groupby('work_year')['salary_in_usd'].median().iloc[0]) - 1) * 100
    filtered_growth = 0
    if plot_df['work_year'].nunique() > 1:
        filtered_growth = ((plot_df.groupby('work_year')['salary_in_usd'].median().iloc[-1] / plot_df.groupby('work_year')['salary_in_usd'].median().iloc[0]) - 1) * 100

    row_metrics = st.columns(4)
    row_metrics[0].metric('Records', int_format(len(plot_df)), f'{record_pct:.1f}% of total')
    row_metrics[1].metric('Countries', int_format(plot_df['company_location'].nunique()), f'{int_format(raw_df["company_location"].nunique())} available')
    row_metrics[2].metric('Unique Roles', int_format(plot_df['job_title'].nunique()), f'{int_format(raw_df["job_title"].nunique())} available')
    row_metrics[3].metric('Median Salary', currency(filtered_median), f'{currency(filtered_median - overall_median)} vs overall')

    row_metrics2 = st.columns(4)
    row_metrics2[0].metric('Average Salary', currency(filtered_mean), f'{currency(filtered_mean - overall_avg)} vs overall')
    row_metrics2[1].metric('Salary Range', currency(filtered_range), f'{currency(filtered_range - overall_range)} vs overall')
    row_metrics2[2].metric('Growth Trend', percent(filtered_growth), f'{percent(filtered_growth - overall_growth)} vs overall')
    row_metrics2[3].metric('Top Salary', currency(filtered_max), f'{currency(filtered_max - raw_df["salary_in_usd"].max())} vs peak')

    st.markdown('---')
    with st.expander('Key Insights'):
        top_role = plot_df.groupby('job_title')['salary_in_usd'].median().idxmax() if len(plot_df) else 'N/A'
        common_role = plot_df['job_title'].mode().iloc[0] if len(plot_df) else 'N/A'
        top_country = plot_df.groupby('company_location')['salary_in_usd'].median().idxmax() if len(plot_df) else 'N/A'
        popular_model = plot_df['work_models'].mode().iloc[0] if len(plot_df) else 'N/A'
        st.write(f'- Highest-paying role: **{top_role}**')
        st.write(f'- Most common role: **{common_role}**')
        st.write(f'- Top paying country: **{top_country}**')
        st.write(f'- Most popular work model: **{popular_model}**')
        st.write(f'- Salary band: **{currency(filtered_min)} — {currency(filtered_max)}**')

    st.subheader('Visual Analytics')

    # Work model distribution shows the share of remote/hybrid/onsite records in the selected slice.
    # Top countries by record count highlights where most dataset observations come from.
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown('#### Work Model Share')
        model_counts = plot_df['work_models'].value_counts()
        fig = plot_pie(model_counts.index.tolist(), model_counts.values.tolist(), 'Work Model Share')
        st.pyplot(fig)
    with c2:
        st.markdown('#### Top Countries by Record Count')
        country_counts = plot_df['company_location'].value_counts().nlargest(10).reset_index()
        country_counts.columns = ['Country', 'Count']
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=country_counts, x='Country', y='Count', color='#2a9d8f', ax=ax)
        ax.set_title('Top Countries by Record Count', fontsize=16)
        ax.set_xlabel('Country')
        ax.set_ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

    # Salary distribution and trend reveal the shape of pay and how median pay changes over time.
    c3, c4 = st.columns(2)
    with c3:
        st.markdown('#### Salary Distribution')
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(plot_df['salary_in_usd'], bins=35, kde=True, color='#264653', ax=ax)
        ax.axvline(filtered_median, color='#e76f51', linestyle='--', linewidth=2, label='Median')
        ax.axvline(filtered_mean, color='#2a9d8f', linestyle=':', linewidth=2, label='Average')
        ax.set_title('Salary Distribution')
        ax.set_xlabel('Salary (USD)')
        ax.set_ylabel('Count')
        ax.legend()
        st.pyplot(fig)
    with c4:
        st.markdown('#### Yearly Median Salary Trend')
        trend = plot_df.groupby('work_year')['salary_in_usd'].median().reset_index()
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=trend, x='work_year', y='salary_in_usd', marker='o', color='#e76f51', ax=ax)
        ax.set_title('Median Salary Trend by Year')
        ax.set_xlabel('Year')
        ax.set_ylabel('Median Salary (USD)')
        st.pyplot(fig)

    # Compare median salary by experience and company size to identify pay drivers.
    c5, c6 = st.columns(2)
    with c5:
        st.markdown('#### Median Salary by Experience')
        exp_median = plot_df.groupby('experience_level')['salary_in_usd'].median().reset_index().sort_values('salary_in_usd', ascending=False)
        fig = plot_bar(exp_median, x='experience_level', y='salary_in_usd', title='Median Salary by Experience Level', xlabel='Experience Level', ylabel='Median Salary (USD)')
        st.pyplot(fig)
    with c6:
        st.markdown('#### Median Salary by Company Size')
        size_median = plot_df.groupby('company_size')['salary_in_usd'].median().reset_index().sort_values('salary_in_usd', ascending=False)
        fig = plot_bar(size_median, x='company_size', y='salary_in_usd', title='Median Salary by Company Size', xlabel='Company Size', ylabel='Median Salary (USD)')
        st.pyplot(fig)

    # Explore how experience and work model combine to affect median salary and spread.
    c7, c8 = st.columns(2)
    with c7:
        st.markdown('#### Experience vs Work Model Heatmap')
        pivot = plot_df.pivot_table(index='experience_level', columns='work_models', values='salary_in_usd', aggfunc='median').fillna(0)
        fig = plot_heatmap(pivot, 'Median Salary by Experience and Work Model', 'Work Model', 'Experience Level')
        st.pyplot(fig)
    with c8:
        st.markdown('#### Salary Range by Experience')
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=plot_df, x='experience_level', y='salary_in_usd', color='#2a9d8f', ax=ax)
        ax.set_title('Salary Spread by Experience Level')
        ax.set_xlabel('Experience Level')
        ax.set_ylabel('Salary (USD)')
        st.pyplot(fig)

    # Show top earning countries and the most common job titles in the current filter slice.
    c9, c10 = st.columns(2)
    with c9:
        st.markdown('#### Top Countries by Average Salary')
        country_avg = plot_df.groupby('company_location')['salary_in_usd'].mean().nlargest(10).reset_index()
        fig = plot_bar(country_avg, x='company_location', y='salary_in_usd', title='Top Countries by Average Salary', xlabel='Country', ylabel='Average Salary (USD)')
        st.pyplot(fig)
    with c10:
        st.markdown('#### Popular Job Titles by Count')
        title_counts = plot_df['job_title'].value_counts().nlargest(10).reset_index()
        title_counts.columns = ['Job Title', 'Count']
        fig = plot_bar(title_counts, x='Job Title', y='Count', title='Top Job Titles by Count', xlabel='Job Title', ylabel='Count')
        st.pyplot(fig)

    # Compare the filtered dataset composition by work model and company size.
    c11, c12 = st.columns(2)
    with c11:
        st.markdown('#### Work Model Distribution')
        work_counts = plot_df['work_models'].value_counts().reset_index()
        work_counts.columns = ['Work Model', 'Count']
        fig = plot_bar(work_counts, x='Work Model', y='Count', title='Work Model Distribution', xlabel='Work Model', ylabel='Count')
        st.pyplot(fig)
    with c12:
        st.markdown('#### Company Size Breakdown')
        size_counts = plot_df['company_size'].value_counts().reset_index()
        size_counts.columns = ['Company Size', 'Count']
        fig = plot_bar(size_counts, x='Company Size', y='Count', title='Company Size Breakdown', xlabel='Company Size', ylabel='Count')
        st.pyplot(fig)

    st.markdown('---')
    st.subheader('Dashboard tables')

    # Summary tables provide numeric context for the filtered dataset by experience, location, role, and work model.
    t1, t2 = st.columns(2)
    with t1:
        st.markdown('**Salary by Experience Level**')
        experience_table = (
            plot_df.groupby('experience_level')['salary_in_usd']
            .agg(records='count', average='mean', median='median', minimum='min', maximum='max')
            .reset_index()
        )
        experience_table[['average', 'median', 'minimum', 'maximum']] = experience_table[['average', 'median', 'minimum', 'maximum']].apply(lambda col: col.map(currency))
        st.dataframe(experience_table, width='stretch')
    with t2:
        st.markdown('**Top Countries by Average Salary**')
        country_table = (
            plot_df.groupby('company_location')['salary_in_usd']
            .agg(records='count', average='mean', median='median')
            .sort_values('average', ascending=False)
            .head(15)
            .reset_index()
        )
        country_table[['average', 'median']] = country_table[['average', 'median']].apply(lambda col: col.map(currency))
        st.dataframe(country_table, width='stretch')

    t3, t4 = st.columns(2)
    with t3:
        st.markdown('**Top Job Titles by Count**')
        jobs_table = (
            plot_df.groupby('job_title')['salary_in_usd']
            .agg(records='count', average='mean', median='median')
            .sort_values('records', ascending=False)
            .head(15)
            .reset_index()
        )
        jobs_table[['average', 'median']] = jobs_table[['average', 'median']].apply(lambda col: col.map(currency))
        st.dataframe(jobs_table, width='stretch')
    with t4:
        st.markdown('**Work Model Comparison**')
        model_table = (
            plot_df.groupby('work_models')['salary_in_usd']
            .agg(records='count', average='mean')
            .reset_index()
        )
        model_table['share'] = model_table['records'] / model_table['records'].sum() * 100
        model_table['avg_vs_overall'] = model_table['average'] / overall_avg * 100 - 100
        model_table_display = model_table.copy()
        model_table_display['average'] = model_table_display['average'].map(currency)
        model_table_display['share'] = model_table_display['share'].map(percent)
        model_table_display['avg_vs_overall'] = model_table_display['avg_vs_overall'].map(percent)
        st.dataframe(model_table_display, width='stretch')

    export_columns = ['job_title', 'experience_level', 'work_models', 'company_location', 'company_size', 'salary_in_usd', 'work_year']
    export_columns = [col for col in export_columns if col in plot_df.columns]
    export_df = plot_df[export_columns].copy()
    export_df = export_df.rename(columns={
        'job_title': 'Job Title',
        'experience_level': 'Experience Level',
        'work_models': 'Work Model',
        'company_location': 'Company Location',
        'company_size': 'Company Size',
        'salary_in_usd': 'Salary (USD)',
        'work_year': 'Work Year'
    })

    experience_export = experience_table.copy()
    country_export = country_table.copy()
    jobs_export = jobs_table.copy()
    model_export = model_table.copy()
    experience_export[['average', 'median', 'minimum', 'maximum']] = experience_export[['average', 'median', 'minimum', 'maximum']].round(2)
    country_export[['average', 'median']] = country_export[['average', 'median']].round(2)
    jobs_export[['average', 'median']] = jobs_export[['average', 'median']].round(2)
    model_export['share'] = model_export['share'].round(2)
    model_export['avg_vs_overall'] = model_export['avg_vs_overall'].round(2)

    excel_engine = get_available_excel_engine()
    if excel_engine is not None:
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine=excel_engine) as writer:
            export_df.to_excel(writer, sheet_name='Filtered Data', index=False)
            experience_export.to_excel(writer, sheet_name='Salary by Experience', index=False)
            country_export.to_excel(writer, sheet_name='Top Countries', index=False)
            jobs_export.to_excel(writer, sheet_name='Top Job Titles', index=False)
            model_export.to_excel(writer, sheet_name='Work Model Comparison', index=False)
        excel_data = excel_buffer.getvalue()
    else:
        excel_data = None

    if excel_data is not None:
        st.download_button(
            'Download filtered data + summaries as Excel',
            data=excel_data,
            file_name='filtered_salaries_with_summary.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    else:
        st.info('Excel export requires either xlsxwriter or openpyxl installed in this Python environment.')

with tab2:
    st.markdown('### Predict your market salary with model-backed insights')
    st.write('Provide your profile details and receive an estimate plus comparable salary context.')

    with st.form('predict_form'):
        c1, c2 = st.columns(2)
        with c1:
            selected_job = st.selectbox('Job Title', job_options, index=job_options.index('Data Scientist') if 'Data Scientist' in job_options else 0)
            selected_experience = st.selectbox('Experience Level', ['Entry', 'Mid', 'Senior', 'Exec'])
            exact_years = st.checkbox('I want to enter exact years of experience')
            experience_years = 0
            if exact_years:
                experience_years = st.slider('Years of experience', 0, 25, 3)
                mapped_experience = map_years_to_experience(experience_years)
                st.info(f'Mapped experience level: {mapped_experience}')
        with c2:
            selected_work_model = st.radio('Work Model', ['Remote', 'Hybrid', 'On-site'])
            selected_location = st.selectbox('Company Location', location_options, index=location_options.index('US') if 'US' in location_options else 0)
            selected_size = st.selectbox('Company Size', company_size_info['company_size'].tolist())
        submitted = st.form_submit_button('Predict Salary')

    if submitted:
        profile_experience = mapped_experience if exact_years else selected_experience
        user_record = pd.DataFrame([
            {
                'job_title_grouped': selected_job,
                'experience_level': profile_experience,
                'work_models': selected_work_model,
                'company_location_grouped': selected_location,
                'company_size': selected_size
            }
        ])
        try:
            prediction = model.predict(user_record)[0]
        except Exception:
            st.error('Prediction could not be generated. Please adjust the inputs and try again.')
            prediction = None

        if prediction is not None:
            percentile_rank = get_percentile_rank(raw_df['salary_in_usd'], prediction)
            if percentile_rank >= 75:
                badge_color = '#d4f5e9'
                label = 'Top tier market salary'
            elif percentile_rank >= 40:
                badge_color = '#fff3cd'
                label = 'Competitive market salary'
            else:
                badge_color = '#ffd6d6'
                label = 'Below median market salary'

            st.markdown(
                f"<div style='padding:20px;border-radius:12px;background:{badge_color};color:#111;font-family:Arial, sans-serif;'>"
                f"<h2 style='margin:0;color:#111;'>Estimated salary: {currency(prediction)}</h2>"
                f"<p style='margin:6px 0 0;color:#111;'>Percentile rank: <strong>{percentile_rank:.0f}th</strong></p>"
                f"<p style='margin:4px 0 0;color:#111;font-weight:600;'>{label}</p>"
                f"</div>",
                unsafe_allow_html=True
            )

            st.markdown('---')
            st.subheader('Profile summary')
            st.markdown(
                f'*Job:* **{selected_job}**  \n*Experience:* **{profile_experience}**  \n*Work model:* **{selected_work_model}**  \n*Location:* **{selected_location}**  \n*Company size:* **{selected_size}**'
            )

            similar_profiles = raw_df[
                (raw_df['job_title'].where(raw_df['job_title'].isin(job_options), 'Other') == selected_job)
                & (raw_df['experience_level'] == profile_experience)
                & (raw_df['work_models'] == selected_work_model)
                & (raw_df['company_location'] == selected_location)
                & (raw_df['company_size'] == selected_size)
            ]
            if similar_profiles.empty:
                similar_profiles = raw_df[
                    (raw_df['job_title'].where(raw_df['job_title'].isin(job_options), 'Other') == selected_job)
                    & (raw_df['experience_level'] == profile_experience)
                ]

            if not similar_profiles.empty:
                sample = similar_profiles.sort_values('salary_in_usd', ascending=False).head(5)[
                    ['job_title', 'experience_level', 'work_models', 'company_location', 'company_size', 'salary_in_usd']
                ]
                sample['salary_in_usd'] = sample['salary_in_usd'].map(currency)
                st.markdown('**Similar profiles from the dataset**')
                st.dataframe(sample, width='stretch')
            else:
                st.info('No close comparable profiles were found. The model estimate uses the most relevant patterns available.')

            st.markdown('### Market benchmarking')
            benchmark_job = raw_df[raw_df['job_title'] == selected_job]
            benchmark_exp = raw_df[raw_df['experience_level'] == profile_experience]
            benchmark_loc = raw_df[raw_df['company_location'] == selected_location]

            bm1, bm2, bm3 = st.columns(3)
            bm1.metric('Same role median', currency(benchmark_job['salary_in_usd'].median()))
            bm2.metric('Same experience median', currency(benchmark_exp['salary_in_usd'].median()))
            bm3.metric('Same location median', currency(benchmark_loc['salary_in_usd'].median()))

            st.markdown('### Career progression and salary potential')
            if selected_job != 'Other':
                progression = (
                    raw_df[raw_df['job_title'] == selected_job]
                    .groupby('experience_level')['salary_in_usd']
                    .median()
                    .reset_index()
                )
                progression['salary_in_usd'] = progression['salary_in_usd'].map(currency)
                st.dataframe(progression, width='stretch')
            else:
                st.write('Broader job title selection means career progression is more generalized.')

            with st.expander('Hiring and salary negotiation guidance'):
                st.write(
                    '- Highlight experience in high-value skills and business impact.\n'
                    '- Use the median salary for your experience level as the starting point for negotiations.\n'
                    '- Remote/hybrid options may increase your pay range depending on the region.\n'
                    '- Larger companies often pay more for the same role, but culture and fit matter as well.\n'
                )

            with st.expander('Model confidence & validation scores'):
                st.write('These scores are estimated from the available dataset and provide a sense of the model fit.')
                try:
                    cv_r2, cv_mae = compute_cv_scores(raw_df)
                    st.write(f'- 5-fold R²: **{np.mean(cv_r2):.2f}** ± {np.std(cv_r2):.2f}')
                    st.write(f'- 5-fold MAE: **{int(np.mean(cv_mae)):,} USD** ± {int(np.std(cv_mae)):,} USD')
                except Exception:
                    st.info('Cross-validation summary is temporarily unavailable.')
