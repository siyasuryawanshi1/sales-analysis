import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
from lifelines import KaplanMeierFitter


def load_and_prepare_data(df):
    """
    Prepare the data for analysis by handling dates based on order status
    """
    # First, let's add some logging to understand our data
    st.write("Initial data shape:", df.shape)

    # Convert date columns to datetime with proper error handling
    date_columns = ['Created Date', 'Next Action Date', 'DemoDate', 'QuoteSentDate']
    for col in date_columns:
        if col in df.columns:
            st.write(f"Processing column: {col}")
            st.write("Sample before conversion:", df[col].head(5))  # Log sample before conversion

            # Replace empty strings and invalid values with NaN
            df[col] = df[col].replace(['', 'NaN', 'nan', 'NaT', ' '], pd.NA)

            # Convert to datetime
            #df[col] = pd.to_datetime(df[col], format='%d/%m/%Y', errors='coerce')
            df[col] = pd.to_datetime(df[col], errors='coerce')

            # Log sample after conversion
            st.write("Sample after conversion:", df[col].head(5))

            # Log the number of valid dates for each column
            valid_dates = df[col].notna().sum()
            st.write(f"Valid dates in {col}: {valid_dates} out of {len(df)}")

            invalid_dates = df[df[col].isna()][col].unique()
            st.write(f"Invalid or problematic entries in {col}:", invalid_dates)
        else:
            st.write(f"Column {col} is not in the dataset.")

    # Create status-specific duration calculations
    df['ProcessDuration'] = pd.NA  # Initialize with pandas NA

    # Process each status separately
    for status in ['"Win"', '"Drop"', '"Ongoing"']:
        status_mask = df['Order Status'] == status
        status_count = status_mask.sum()
        st.write(f"\nProcessing {status} leads ({status_count} leads)")

        if status == '"Win"':
            df.loc[status_mask, 'ProcessDuration'] = calculate_win_duration(df[status_mask])
        elif status == '"Drop"':
            df.loc[status_mask, 'ProcessDuration'] = calculate_drop_duration(df[status_mask])
        else:  # Ongoing
            df.loc[status_mask, 'ProcessDuration'] = calculate_ongoing_duration(df[status_mask])

        # Log the number of calculated durations for this status
        valid_durations = df.loc[status_mask, 'ProcessDuration'].notna().sum()
        st.write(f"Calculated durations for {status}: {valid_durations} out of {status_count}")

    # Calculate stage durations with logging
    df['DemoStage'] = calculate_stage_duration(df, 'Created Date', 'DemoDate', 'Demo Stage')
    df['QuoteStage'] = calculate_stage_duration(df, 'DemoDate', 'QuoteSentDate', 'Quote Stage')

    return df


def calculate_win_duration(df_subset):
    """Calculate duration for won deals using the latest available date"""
    durations = pd.Series(index=df_subset.index, dtype='float64')

    for idx, row in df_subset.iterrows():
        if pd.isna(row['Created Date']):
            continue

        # Get all valid dates
        valid_dates = [
            d for d in [row['Next Action Date'], row['DemoDate'], row['QuoteSentDate']]
            if pd.notna(d)
        ]

        if valid_dates:
            durations[idx] = (max(valid_dates) - row['Created Date']).days

    return durations


def calculate_drop_duration(df_subset):
    """Calculate duration for dropped deals"""
    durations = pd.Series(index=df_subset.index, dtype='float64')
    current_date = pd.Timestamp.now()

    for idx, row in df_subset.iterrows():
        if pd.isna(row['Created Date']):
            continue

        end_date = row['Next Action Date'] if pd.notna(row['Next Action Date']) else current_date
        durations[idx] = (end_date - row['Created Date']).days

    return durations


def calculate_ongoing_duration(df_subset):
    """Calculate duration for ongoing deals"""
    durations = pd.Series(index=df_subset.index, dtype='float64')
    current_date = pd.Timestamp.now()

    for idx, row in df_subset.iterrows():
        if pd.notna(row['Created Date']):
            durations[idx] = (current_date - row['Created Date']).days

    return durations


def calculate_stage_duration(df, start_col, end_col, stage_name):
    """Calculate duration between two stages, handling missing dates"""
    # Create a mask for valid date pairs
    mask = df[start_col].notna() & df[end_col].notna()
    valid_pairs = mask.sum()
    st.write(f"\n{stage_name} duration calculation:")
    st.write(f"Valid date pairs: {valid_pairs} out of {len(df)}")

    # Initialize duration series
    durations = pd.Series(index=df.index, dtype='float64')

    # Calculate durations only for valid pairs
    if valid_pairs > 0:
        durations[mask] = (df.loc[mask, end_col] - df.loc[mask, start_col]).dt.days
        mean_duration = durations[mask].mean()
        st.write(f"Average {stage_name} duration: {mean_duration:.1f} days")
    else:
        st.write(f"No valid date pairs for {stage_name} duration calculation")

    return durations


def create_state_histogram(df):
    """Create an enhanced histogram showing lead states with duration information"""
    state_stats = df.groupby('Order Status').agg({
        'ProcessDuration': ['count', 'mean', 'median']
    }).round(1)

    fig = go.Figure()

    # Add bar for counts
    fig.add_trace(go.Bar(
        name='Count',
        x=state_stats.index,
        y=state_stats[('ProcessDuration', 'count')],
        text=state_stats[('ProcessDuration', 'count')],
        textposition='auto'
    ))

    # Add information about average duration
    fig.add_trace(go.Scatter(
        name='Avg Duration (days)',
        x=state_stats.index,
        y=state_stats[('ProcessDuration', 'mean')],
        yaxis='y2',
        mode='markers+text',
        text=state_stats[('ProcessDuration', 'mean')].round(1),
        textposition='top center'
    ))

    fig.update_layout(
        title='Lead States Distribution with Duration Analysis',
        yaxis=dict(title='Count'),
        yaxis2=dict(title='Average Duration (days)', overlaying='y', side='right'),
        barmode='group'
    )

    return fig


def create_duration_analysis(df):
    """Create detailed duration analysis with proper handling of missing values"""
    st.subheader('Duration Analysis')

    # Calculate statistics for each duration type
    duration_types = {
        'Process Duration': 'ProcessDuration',
        'Demo Stage': 'DemoStage',
        'Quote Stage': 'QuoteStage'
    }

    for label, column in duration_types.items():
        valid_data = df[column].dropna()
        if len(valid_data) > 0:
            st.write(f"\n{label} Statistics:")
            st.write(f"- Number of valid records: {len(valid_data)}")
            st.write(f"- Average duration: {valid_data.mean():.1f} days")
            st.write(f"- Median duration: {valid_data.median():.1f} days")
            st.write(f"- Min duration: {valid_data.min():.1f} days")
            st.write(f"- Max duration: {valid_data.max():.1f} days")
        else:
            st.write(f"\nNo valid data for {label}")

def create_funnel_analysis(df):
    """Create a funnel analysis showing conversion through stages"""
    total_leads = len(df)
    demo_completed = df['DemoStatus'].eq('Yes').sum()
    quote_sent = df['QuoteSentDate'].notna().sum()
    won_deals = (df['Order Status'] == '"Win"').sum()

    stages = ['Total Leads', 'Demo Completed', 'Quote Sent', 'Won']
    values = [total_leads, demo_completed, quote_sent, won_deals]

    fig = go.Figure(go.Funnel(
        y=stages,
        x=values,
        textinfo="value+percent initial"
    ))

    fig.update_layout(title='Sales Funnel Analysis')
    return fig


def create_assignee_performance_detailed(df):
    """Create detailed performance analysis by assignee"""
    assignee_stats = df.groupby('Created By').agg({
        'Form Id': 'count',
        'ProcessDuration': 'mean',
        'Order Status': lambda x: (x == '"Win"').mean()
    }).round(2)

    assignee_stats.columns = ['Total Leads', 'Avg Duration', 'Win Rate']

    fig = go.Figure()

    # Add bars for total leads
    fig.add_trace(go.Bar(
        name='Total Leads',
        x=assignee_stats.index,
        y=assignee_stats['Total Leads'],
        text=assignee_stats['Total Leads'],
        textposition='auto'
    ))

    # Add win rate line
    fig.add_trace(go.Scatter(
        name='Win Rate',
        x=assignee_stats.index,
        y=assignee_stats['Win Rate'],
        yaxis='y2',
        mode='lines+markers+text',
        text=(assignee_stats['Win Rate'] * 100).round(1).astype(str) + '%',
        textposition='top center'
    ))

    fig.update_layout(
        title='Assignee Performance Analysis',
        yaxis=dict(title='Total Leads'),
        yaxis2=dict(title='Win Rate', overlaying='y', side='right', tickformat='%'),
        barmode='group'
    )

    return fig, assignee_stats


def main():
    st.title('Sales Lead Analysis Dashboard')

    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            st.write("Data loaded successfully")

            # Process data
            df = load_and_prepare_data(df)

            # Overview metrics
            st.header('Overview Metrics')
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Leads", len(df))
            with col2:
                win_rate = (df['Order Status'] == '"Win"').mean() * 100
                st.metric("Win Rate", f"{win_rate:.1f}%")
            with col3:
                avg_duration = df['ProcessDuration'].mean()
                st.metric("Avg Duration (days)", f"{avg_duration:.1f}")
            with col4:
                active_leads = (df['Order Status'] == '"Ongoing"').sum()
                st.metric("Active Leads", active_leads)

            # Create tabs for different analyses
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                'Lead States',
                'Sales Funnel',
                'Assignee Performance',
                'Duration Analysis',
                'Detailed Data Analysis'
            ])

            with tab1:
                st.plotly_chart(create_state_histogram(df))

            with tab2:
                st.plotly_chart(create_funnel_analysis(df))

            with tab3:
                fig, stats = create_assignee_performance_detailed(df)
                st.plotly_chart(fig)
                st.subheader('Detailed Assignee Statistics')
                st.dataframe(stats.style.format({
                    'Win Rate': '{:.1%}',
                    'Avg Duration': '{:.1f}'
                }))

            with tab4:
                st.subheader('Stage Duration Analysis')
                duration_stats = df.agg({
                    'DemoStage': lambda x: x.mean(),
                    'QuoteStage': lambda x: x.mean(),
                    'ProcessDuration': lambda x: x.mean()
                }).round(1)

                st.write("Average duration (in days) for each stage:")
                st.write(f"- Time to Demo: {duration_stats['DemoStage']}")
                st.write(f"- Demo to Quote: {duration_stats['QuoteStage']}")
                st.write(f"- Total Process: {duration_stats['ProcessDuration']}")

            with tab5:
                # Create detailed duration analysis
                create_duration_analysis(df)

                # Display sample of processed data
                st.subheader('Sample of Processed Data')
                st.write(df[['Created Date', 'Next Action Date', 'DemoDate', 'QuoteSentDate',
                             'ProcessDuration', 'DemoStage', 'QuoteStage', 'Order Status']].head())

        except Exception as e:
            st.error(f"An error occurred while processing the data: {str(e)}")
            st.write("Please check your data format and try again.")


if __name__ == "__main__":
    main()