from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'data' / 'collision-2025.csv'
SEVERITY = {1: 'Fatal', 2: 'Serious', 3: 'Slight'}
SEVERITY_COLORS = {'Fatal': '#f05252', 'Serious': '#f5a524', 'Slight': '#22c997'}
FEATURES = ['number_of_vehicles', 'day_of_week', 'hour', 'speed_limit', 'first_road_class', 'road_type', 'light_conditions', 'weather_conditions', 'road_surface_conditions', 'urban_or_rural_area']
CAT_FEATURES = ['day_of_week', 'first_road_class', 'road_type', 'light_conditions', 'weather_conditions', 'road_surface_conditions', 'urban_or_rural_area']
NUM_FEATURES = ['number_of_vehicles', 'hour', 'speed_limit']

st.set_page_config(page_title='RoadSense | Collision Intelligence', page_icon='🛣️', layout='wide', initial_sidebar_state='expanded')
st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root{--ink:#eaf0fb;--muted:#8e9bb3;--panel:#111a2b;--line:#26334a;--accent:#5b8cff}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif} .stApp{background:radial-gradient(ellipse at 10% 0%,#172747 0%,#0b1220 42%,#080e19 100%);color:var(--ink)}
h1,h2,h3{font-family:'Space Grotesk',sans-serif;letter-spacing:-.035em} h1{font-size:2.45rem!important;color:#f4f7ff} p, label, .stMarkdown{color:#d8e1f0}
.block-container{padding-top:1.7rem;padding-bottom:3rem;max-width:1500px}.hero{padding:26px 30px;border:1px solid #2b3a54;border-radius:22px;background:linear-gradient(115deg,rgba(30,48,79,.96),rgba(16,27,45,.96));margin-bottom:20px;position:relative;overflow:hidden}.hero:after{content:' ';position:absolute;width:250px;height:250px;border-radius:50%;right:-70px;top:-125px;border:1px solid #4b6d9c;box-shadow:0 0 0 30px rgba(91,140,255,.04),0 0 0 60px rgba(91,140,255,.03)}.eyebrow{font-size:.74rem;letter-spacing:.17em;text-transform:uppercase;color:#8fb2ff;font-weight:700}.hero p{color:#aebbd0;margin-bottom:0}.metric{background:linear-gradient(145deg,rgba(20,31,51,.98),rgba(14,23,38,.98));border:1px solid #27364f;border-radius:17px;padding:18px 20px;min-height:124px}.metric .label{font-size:.78rem;text-transform:uppercase;letter-spacing:.08em;color:#94a4bd;font-weight:700}.metric .value{font-family:'Space Grotesk';font-size:1.85rem;font-weight:700;color:#f5f8ff;margin-top:8px}.metric .hint{font-size:.78rem;color:#91a0b8;margin-top:3px}.section-head{display:flex;align-items:center;justify-content:space-between;margin:24px 0 12px}.section-head h2{font-size:1.35rem!important;margin:0}.panel{background:rgba(17,26,43,.75);border:1px solid #27364d;border-radius:17px;padding:18px}.stTabs [data-baseweb="tab-list"]{gap:8px;background:#101a2b;padding:7px;border-radius:12px}.stTabs [data-baseweb="tab"]{border-radius:8px;color:#aebbd0}.stTabs [aria-selected="true"]{background:#263c63;color:white}.stButton>button{border-radius:10px;border:1px solid #4c73b8;background:#315b9d;color:white;font-weight:700}.stDownloadButton>button{border-radius:10px}.stSidebar{background:#0d1626}.stSidebar [data-testid="stMarkdownContainer"] p{color:#c3cfe0}.small-note{color:#92a0b6;font-size:.84rem}.stAlert{border-radius:12px}.stDataFrame{border:1px solid #27364d;border-radius:12px;overflow:hidden}
</style>''', unsafe_allow_html=True)

@st.cache_data(show_spinner='Loading collision records…')
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f'Dataset not found: {DATA_PATH}. Keep data/collision-2025.csv in the project.')
    df = pd.read_csv(DATA_PATH, low_memory=False)
    required = {'collision_severity','date','time','number_of_vehicles','number_of_casualties'}
    missing = required - set(df.columns)
    if missing: raise ValueError('Dataset is missing required columns: ' + ', '.join(sorted(missing)))
    df['collision_severity'] = pd.to_numeric(df['collision_severity'], errors='coerce')
    df = df[df['collision_severity'].isin([1,2,3])].copy()
    df['severity_label'] = df['collision_severity'].map(SEVERITY)
    df['date_parsed'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    df['hour'] = pd.to_datetime(df['time'], format='%H:%M', errors='coerce').dt.hour
    df['month'] = df['date_parsed'].dt.strftime('%b')
    df['month_num'] = df['date_parsed'].dt.month
    df['weekday_name'] = df['date_parsed'].dt.day_name()
    for col in ['longitude','latitude','number_of_vehicles','number_of_casualties','speed_limit','day_of_week','first_road_class','road_type','light_conditions','weather_conditions','road_surface_conditions','urban_or_rural_area']:
        if col in df: df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.replace({-1: np.nan, -1.0: np.nan})
    return df

@st.cache_resource(show_spinner='Training a local machine-learning model…')
def train_model(_cache_version='v1'):
    df = load_data()
    usable = df.dropna(subset=['collision_severity']).copy()
    # Use a reproducible sample so the dashboard trains quickly on modest hosting resources.
    if len(usable) > 30000:
        # Keep training bounded for free hosting; the subsequent split is stratified.
        usable = usable.sample(n=30000, random_state=42)
    usable['hour'] = usable['hour'].fillna(12)
    for c in FEATURES:
        if c not in usable: usable[c] = np.nan
    X = usable[FEATURES].copy()
    y = usable['collision_severity'].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42, stratify=y)
    prep = ColumnTransformer([
        ('numeric', SimpleImputer(strategy='median'), NUM_FEATURES),
        ('categorical', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), CAT_FEATURES)
    ])
    model = Pipeline([('preprocess', prep), ('classifier', RandomForestClassifier(n_estimators=70, max_depth=16, min_samples_leaf=3, class_weight='balanced_subsample', random_state=42, n_jobs=-1))])
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    metrics = {'accuracy': accuracy_score(y_test, pred), 'macro_f1': f1_score(y_test, pred, average='macro'), 'weighted_f1': f1_score(y_test, pred, average='weighted'), 'y_test': y_test, 'pred': pred, 'train_rows': len(x_train), 'test_rows': len(x_test), 'model': model}
    return metrics

try:
    df = load_data()
except Exception as exc:
    st.error(f'Could not load the project dataset. {exc}')
    st.stop()

with st.sidebar:
    st.markdown('## 🛣️ RoadSense')
    st.caption('COLLISION INTELLIGENCE')
    st.markdown('---')
    page = st.radio('NAVIGATION', ['Executive Overview','Explore Patterns','Collision Map','ML Predictions','Data Explorer'], label_visibility='visible')
    st.markdown('---')
    st.markdown('### Filter the dataset')
    severity_options = [x for x in ['Fatal','Serious','Slight'] if x in df['severity_label'].dropna().unique()]
    selected_severity = st.multiselect('Collision severity', severity_options, default=severity_options)
    years = sorted(df['collision_year'].dropna().astype(int).unique().tolist()) if 'collision_year' in df else []
    selected_years = st.multiselect('Year', years, default=years)
    if 'urban_or_rural_area' in df:
        area_vals = sorted(df['urban_or_rural_area'].dropna().unique().tolist())
        selected_area = st.multiselect('Area type code', area_vals, default=area_vals)
    else: selected_area = []
    st.caption('Filters apply to analytics and data exploration. The prediction model uses a held-out test split.')

filtered = df[df['severity_label'].isin(selected_severity)] if selected_severity else df.iloc[0:0]
if years and selected_years: filtered = filtered[filtered['collision_year'].isin(selected_years)]
if years and not selected_years: filtered = filtered.iloc[0:0]
if selected_area and 'urban_or_rural_area' in filtered: filtered = filtered[filtered['urban_or_rural_area'].isin(selected_area)]

st.markdown('<div class="hero"><div class="eyebrow">ROAD SAFETY • DATA SCIENCE</div><h1>RoadSense Collision Intelligence</h1><p>Explore collision patterns, identify risk signals, and evaluate a machine-learning model using real collision records.</p></div>', unsafe_allow_html=True)
st.caption(f'Current view: {len(filtered):,} of {len(df):,} records • Dataset: collision-2025.csv • Severity model: Random Forest classifier')

# chart styling helper
def style_fig(fig, height=330):
    fig.update_layout(template='plotly_dark', height=height, margin=dict(l=8,r=8,t=42,b=8), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family='DM Sans', color='#d9e4f5'), title_font=dict(size=16, color='#eef4ff'), legend_title_text='')
    fig.update_xaxes(gridcolor='rgba(120,145,180,.12)', zerolinecolor='rgba(120,145,180,.12)')
    fig.update_yaxes(gridcolor='rgba(120,145,180,.12)', zerolinecolor='rgba(120,145,180,.12)')
    return fig

def metric(label, value, hint):
    st.markdown(f'<div class="metric"><div class="label">{label}</div><div class="value">{value}</div><div class="hint">{hint}</div></div>', unsafe_allow_html=True)

if page == 'Executive Overview':
    st.markdown('<div class="section-head"><h2>Safety at a glance</h2></div>', unsafe_allow_html=True)
    a,b,c,d = st.columns(4)
    n = len(filtered); fatal = int((filtered['severity_label']=='Fatal').sum()); serious = int((filtered['severity_label']=='Serious').sum())
    casualties = int(filtered['number_of_casualties'].fillna(0).sum())
    with a: metric('Recorded collisions', f'{n:,}', 'Records after selected filters')
    with b: metric('Fatal collisions', f'{fatal:,}', f'{fatal/n*100:.1f}% of view' if n else 'No records in current view')
    with c: metric('Serious collisions', f'{serious:,}', f'{serious/n*100:.1f}% of view' if n else 'No records in current view')
    with d: metric('Reported casualties', f'{casualties:,}', 'Sum of casualties in filtered records')
    left,right = st.columns([1.05,1.4], gap='large')
    with left:
        sev = filtered['severity_label'].value_counts().reindex(['Fatal','Serious','Slight'], fill_value=0).rename_axis('Severity').reset_index(name='Collisions')
        fig=px.pie(sev, names='Severity', values='Collisions', hole=.68, color='Severity', color_discrete_map=SEVERITY_COLORS, title='Collision severity mix')
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(style_fig(fig,350), use_container_width=True)
    with right:
        if filtered['date_parsed'].notna().any():
            trend=filtered.dropna(subset=['date_parsed']).groupby(filtered['date_parsed'].dt.to_period('M').astype(str)).size().reset_index(name='Collisions').rename(columns={'date_parsed':'Month'})
            fig=px.area(trend, x='Month', y='Collisions', title='Monthly collision trend', markers=True)
            fig.update_traces(line_color='#6c9bff', fillcolor='rgba(91,140,255,.18)')
            st.plotly_chart(style_fig(fig,350), use_container_width=True)
        else: st.info('Date values are unavailable for the selected records.')
    left,right=st.columns(2,gap='large')
    with left:
        if 'speed_limit' in filtered:
            x=filtered.dropna(subset=['speed_limit']).groupby('speed_limit').size().reset_index(name='Collisions').sort_values('speed_limit')
            fig=px.bar(x,x='speed_limit',y='Collisions',title='Collisions by speed limit',labels={'speed_limit':'Speed limit','Collisions':'Records'}, color='Collisions',color_continuous_scale='Blues')
            st.plotly_chart(style_fig(fig,310),use_container_width=True)
    with right:
        if 'hour' in filtered:
            x=filtered.dropna(subset=['hour']).groupby('hour').size().reindex(range(24),fill_value=0).rename_axis('Hour').reset_index(name='Collisions')
            fig=px.line(x,x='Hour',y='Collisions',title='Time of day',markers=True)
            fig.update_traces(line_color='#29c6a3',line_width=3)
            st.plotly_chart(style_fig(fig,310),use_container_width=True)
    st.markdown('<p class="small-note">This dashboard describes recorded collisions; observed associations do not prove causation or forecast an individual collision.</p>',unsafe_allow_html=True)

elif page == 'Explore Patterns':
    st.markdown('## Explore collision patterns')
    x1,x2=st.columns(2,gap='large')
    with x1:
        if 'weekday_name' in filtered:
            order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
            v=filtered['weekday_name'].value_counts().reindex(order,fill_value=0).rename_axis('Weekday').reset_index(name='Collisions')
            st.plotly_chart(style_fig(px.bar(v,x='Weekday',y='Collisions',title='Collisions by weekday',color='Collisions',color_continuous_scale='Teal')),use_container_width=True)
        road=filtered['road_type'].value_counts(dropna=True).head(12).rename_axis('Road type code').reset_index(name='Collisions')
        st.plotly_chart(style_fig(px.bar(road,x='Road type code',y='Collisions',title='Road type distribution')),use_container_width=True)
    with x2:
        for col,title in [('weather_conditions','Weather condition code'),('road_surface_conditions','Road surface code')]:
            if col in filtered:
                v=filtered[col].value_counts(dropna=True).head(10).rename_axis('Condition code').reset_index(name='Collisions')
                st.plotly_chart(style_fig(px.bar(v,x='Collisions',y='Condition code',orientation='h',title=title)),use_container_width=True)
    st.info('Condition and road-type values are source dataset codes. Refer to the dataset documentation for the official code descriptions.')

elif page == 'Collision Map':
    st.markdown('## Collision map')
    st.caption('Locations are shown only when valid latitude and longitude coordinates are present. Points are sampled for browser performance.')
    if {'longitude','latitude'}.issubset(filtered.columns):
        geo=filtered.dropna(subset=['longitude','latitude'])
        geo=geo[geo['longitude'].between(-180,180)&geo['latitude'].between(-90,90)]
        if len(geo)>5000: geo=geo.sample(5000,random_state=42)
        if len(geo):
            fig=px.scatter_map(geo,lat='latitude',lon='longitude',color='severity_label',color_discrete_map=SEVERITY_COLORS,hover_data=[c for c in ['date','time','speed_limit','number_of_vehicles','number_of_casualties'] if c in geo],zoom=4,height=590,map_style='carto-darkmatter',title=f'Collision locations ({len(geo):,} points displayed)')
            fig.update_layout(margin=dict(l=0,r=0,t=45,b=0),paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig,use_container_width=True)
        else: st.warning('No valid coordinates are available for these filters.')
    else: st.warning('The dataset does not contain latitude and longitude columns.')

elif page == 'ML Predictions':
    st.markdown('## Machine-learning workbench')
    st.markdown('A Random Forest classifier estimates the recorded severity class from selected collision context fields. It is an educational analytical model, not a live safety system.')
    try:
        with st.spinner('Preparing model and held-out evaluation…'):
            result=train_model()
        a,b,c=st.columns(3)
        with a: metric('Test accuracy',f"{result['accuracy']:.1%}",'Held-out test records')
        with b: metric('Macro F1',f"{result['macro_f1']:.3f}",'Treats each class equally')
        with c: metric('Weighted F1',f"{result['weighted_f1']:.3f}",'Weighted by class frequency')
        st.caption(f"Model fitted on {result['train_rows']:,} records and evaluated on {result['test_rows']:,} held-out records. A reproducible sample is used to keep hosting requirements modest.")
        cm=confusion_matrix(result['y_test'],result['pred'],labels=[1,2,3])
        fig=px.imshow(cm,text_auto=True,labels=dict(x='Predicted class',y='Actual class',color='Records'),x=['Fatal','Serious','Slight'],y=['Fatal','Serious','Slight'],color_continuous_scale='Blues',title='Confusion matrix')
        st.plotly_chart(style_fig(fig,390),use_container_width=True)
        clf=result['model'].named_steps['classifier']; prep=result['model'].named_steps['preprocess']
        try:
            feature_names=prep.get_feature_names_out(); imp=clf.feature_importances_
            imp_df=pd.DataFrame({'Feature':feature_names,'Importance':imp}).sort_values('Importance',ascending=False).head(15)
            imp_df['Feature']=imp_df['Feature'].str.replace('numeric__','',regex=False).str.replace('categorical__','',regex=False)
            st.plotly_chart(style_fig(px.bar(imp_df.sort_values('Importance'),x='Importance',y='Feature',orientation='h',title='Top model feature signals'),440),use_container_width=True)
        except Exception: pass
        st.markdown('### Try a scenario')
        with st.form('prediction_form'):
            c1,c2,c3=st.columns(3)
            with c1: vehicles=st.number_input('Number of vehicles',1,20,2); hour=st.slider('Hour of day',0,23,17); speed=st.selectbox('Speed limit', [20,30,40,50,60,70], index=1)
            with c2: day=st.selectbox('Day of week code',list(range(1,8)),index=4); road=st.selectbox('Road class code',[1,2,3,4,5,6]); roadtype=st.selectbox('Road type code',[1,2,3,6,7,9])
            with c3: light=st.selectbox('Light condition code',[1,4,5,6,7]); weather=st.selectbox('Weather code',[1,2,3,4,5,6,7,8,9]); surface=st.selectbox('Road surface code',[1,2,3,4,5,6,7])
            area=st.selectbox('Area type code',[1,2,3],index=1)
            submitted=st.form_submit_button('Estimate severity',use_container_width=True)
        if submitted:
            row=pd.DataFrame([{'number_of_vehicles':vehicles,'day_of_week':day,'hour':hour,'speed_limit':speed,'first_road_class':road,'road_type':roadtype,'light_conditions':light,'weather_conditions':weather,'road_surface_conditions':surface,'urban_or_rural_area':area}])[FEATURES]
            prediction=int(result['model'].predict(row)[0]); probs=result['model'].predict_proba(row)[0]
            st.success(f"Model estimate: {SEVERITY[prediction]}")
            prob_df=pd.DataFrame({'Severity':[SEVERITY[int(x)] for x in result['model'].named_steps['classifier'].classes_],'Model probability':probs})
            st.plotly_chart(style_fig(px.bar(prob_df,x='Severity',y='Model probability',color='Severity',color_discrete_map=SEVERITY_COLORS,title='Model class probabilities'),280),use_container_width=True)
            st.warning('This is a statistical estimate from historical records. Do not use it to make real-world driving, emergency, insurance, or enforcement decisions.')
    except Exception as exc:
        st.error(f'Model training failed: {exc}')
        st.info('Install all dependencies from requirements.txt, then restart the dashboard.')

elif page == 'Data Explorer':
    st.markdown('## Data explorer')
    st.caption('Inspect the filtered source records and download the current view.')
    c1,c2,c3=st.columns([2,1,1])
    with c1: query=st.text_input('Search across visible text fields',placeholder='e.g. date, authority, road...')
    with c2: limit=st.selectbox('Rows to display',[100,500,1000,5000],index=1)
    with c3: show_missing=st.checkbox('Show missing-value summary',value=False)
    view=filtered.copy()
    if query:
        text_cols=[c for c in ['collision_index','local_authority_highway','local_authority_ons_district','date','time'] if c in view]
        if text_cols:
            mask=pd.Series(False,index=view.index)
            for col in text_cols: mask |= view[col].astype(str).str.contains(query,case=False,na=False)
            view=view[mask]
    display_cols=[c for c in ['collision_index','date','time','severity_label','number_of_vehicles','number_of_casualties','speed_limit','road_type','weather_conditions','road_surface_conditions','latitude','longitude','local_authority_highway'] if c in view]
    st.dataframe(view[display_cols].head(limit),use_container_width=True,height=430,hide_index=True)
    st.download_button('⬇ Download filtered records as CSV',data=view.to_csv(index=False).encode('utf-8'),file_name='roadsense_filtered_records.csv',mime='text/csv')
    if show_missing:
        miss=view.isna().sum().sort_values(ascending=False).head(20).rename_axis('Column').reset_index(name='Missing values')
        st.plotly_chart(style_fig(px.bar(miss.sort_values('Missing values'),x='Missing values',y='Column',orientation='h',title='Missing values by column'),420),use_container_width=True)

st.markdown('---')
st.markdown('<div style="display:flex;justify-content:space-between;gap:20px;flex-wrap:wrap;color:#8796ad;font-size:.78rem"><span>ROADSense • Road safety analytics portfolio project</span><span>Historical records · Local model training · No live predictions</span></div>',unsafe_allow_html=True)
