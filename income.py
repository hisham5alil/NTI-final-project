
import os

# تحديد عدد الأنوية المتاحة لـ OpenBLAS لتجنب استهلاك الذاكرة المفرط
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px

# ==========================================
# 1. تحميل وتنظيف البيانات
# ==========================================
df = pd.read_csv('census_income.csv')

# تنظيف المسافات الزائدة في النصوص
for col in df.select_dtypes(include=['object', 'string']).columns:
    df[col] = df[col].astype(str).str.strip()

# توحيد قيم عمود الدخل (إزالة النقطة الزائدة إن وجدت)
df['income'] = df['income'].str.replace('.', '', regex=False)

# استبدال '?' بـ 'Unknown' في الأعمدة النصية
df.replace('?', 'Unknown', inplace=True)

# ==========================================
# 2. إعداد تطبيق Dash
# ==========================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "لوحة تحكم تحليل الدخل والديموغرافيا"

# خيارات الفلاتر
income_options = [{'label': 'الكل', 'value': 'ALL'}] + [{'label': i, 'value': i} for i in sorted(df['income'].unique())]
sex_options = [{'label': 'الكل', 'value': 'ALL'}] + [{'label': s, 'value': s} for s in sorted(df['sex'].unique())]
workclass_options = [{'label': 'الكل', 'value': 'ALL'}] + [{'label': w, 'value': w} for w in sorted(df['workclass'].unique())]
education_options = [{'label': 'الكل', 'value': 'ALL'}] + [{'label': e, 'value': e} for e in sorted(df['education'].unique())]

min_age = int(df['age'].min())
max_age = int(df['age'].max())

# ==========================================
# 3. تصميم واجهة المستخدم (Layout)
# ==========================================
app.layout = dbc.Container([
    # العنوان الرئيسي
    dbc.Row([
        dbc.Col(
            html.Div([
                html.H1("📊 لوحة تحكم تحليل الدخل والخصائص السكانية (Census Income)", className="text-center text-primary fw-bold my-3"),
                html.P("تحليل تفاعلي للعلاقة بين مستوى التعليم، المهنة، ساعات العمل، ومستوى الدخل", className="text-center text-muted fs-5")
            ]),
            width=12
        )
    ]),

    # لوحة الفلاتر
    dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("فئة الدخل (Income):", className="fw-bold"),
                    dcc.Dropdown(id='income-filter', options=income_options, value='ALL', clearable=False)
                ], md=3),
                dbc.Col([
                    html.Label("النوع (Sex):", className="fw-bold"),
                    dcc.Dropdown(id='sex-filter', options=sex_options, value='ALL', clearable=False)
                ], md=3),
                dbc.Col([
                    html.Label("قطاع العمل (Workclass):", className="fw-bold"),
                    dcc.Dropdown(id='workclass-filter', options=workclass_options, value='ALL', clearable=False)
                ], md=3),
                dbc.Col([
                    html.Label("النطاق العمري (Age Range):", className="fw-bold"),
                    dcc.RangeSlider(
                        id='age-slider',
                        min=min_age,
                        max=max_age,
                        step=1,
                        value=[min_age, max_age],
                        marks={a: str(a) for a in range(15, max_age + 1, 15)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], md=3),
            ])
        ])
    ], className="mb-4 shadow-sm"),

    # كروت المؤشرات الرئيسية (KPI Cards)
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("إجمالي السجلات", className="card-subtitle text-muted"),
                html.H3(id="kpi-total", className="text-primary fw-bold mt-2")
            ])
        ], className="text-center shadow-sm"), md=3),

        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("نسبة ذوي الدخل العالي (>50K)", className="card-subtitle text-muted"),
                html.H3(id="kpi-high-income-pct", className="text-success fw-bold mt-2")
            ])
        ], className="text-center shadow-sm"), md=3),

        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("متوسط ساعات العمل أسبوعياً", className="card-subtitle text-muted"),
                html.H3(id="kpi-avg-hours", className="text-warning fw-bold mt-2")
            ])
        ], className="text-center shadow-sm"), md=3),

        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("متوسط العمر", className="card-subtitle text-muted"),
                html.H3(id="kpi-avg-age", className="text-info fw-bold mt-2")
            ])
        ], className="text-center shadow-sm"), md=3),
    ], className="mb-4"),

    # الرسوم البيانية - الصف الأول
    dbc.Row([
        dbc.Col(dcc.Graph(id='income-by-education-chart'), md=7),
        dbc.Col(dcc.Graph(id='income-pie-chart'), md=5),
    ], className="mb-4"),

    # الرسوم البيانية - الصف الثاني
    dbc.Row([
        dbc.Col(dcc.Graph(id='age-distribution-chart'), md=6),
        dbc.Col(dcc.Graph(id='hours-by-occupation-chart'), md=6),
    ], className="mb-4"),

    # الرسوم البيانية - الصف الثالث
    dbc.Row([
        dbc.Col(dcc.Graph(id='marital-income-chart'), md=6),
        dbc.Col(dcc.Graph(id='workclass-income-chart'), md=6),
    ], className="mb-4"),

], fluid=True, style={'backgroundColor': '#f8f9fa', 'padding': '20px'})


# ==========================================
# 4. التفاعل والتحديث التلقائي (Callbacks)
# ==========================================
@app.callback(
    [
        Output('kpi-total', 'children'),
        Output('kpi-high-income-pct', 'children'),
        Output('kpi-avg-hours', 'children'),
        Output('kpi-avg-age', 'children'),
        Output('income-by-education-chart', 'figure'),
        Output('income-pie-chart', 'figure'),
        Output('age-distribution-chart', 'figure'),
        Output('hours-by-occupation-chart', 'figure'),
        Output('marital-income-chart', 'figure'),
        Output('workclass-income-chart', 'figure'),
    ],
    [
        Input('income-filter', 'value'),
        Input('sex-filter', 'value'),
        Input('workclass-filter', 'value'),
        Input('age-slider', 'value')
    ]
)
def update_dashboard(selected_income, selected_sex, selected_workclass, age_range):
    # تصفية البيانات
    dff = df[
        (df['age'] >= age_range[0]) & 
        (df['age'] <= age_range[1])
    ]

    if selected_income != 'ALL':
        dff = dff[dff['income'] == selected_income]
    if selected_sex != 'ALL':
        dff = dff[dff['sex'] == selected_sex]
    if selected_workclass != 'ALL':
        dff = dff[dff['workclass'] == selected_workclass]

    # حساب المؤشرات
    total_records = len(dff)
    if total_records > 0:
        high_income_pct = f"{(dff['income'] == '>50K').mean() * 100:.1f}%"
        avg_hours = f"{dff['hours-per-week'].mean():.1f} hrs"
        avg_age = f"{dff['age'].mean():.1f} yrs"
    else:
        high_income_pct = "0%"
        avg_hours = "0"
        avg_age = "0"

    # 1. الدخل حسب المستوى التعليمي (Stacked Bar Chart)
    edu_income = dff.groupby(['education', 'income']).size().reset_index(name='count')
    # ترتيب التعليم حسب متوسط التعليم الرقمي إن أمكن
    edu_order = dff.groupby('education')['education-num'].mean().sort_values().index.tolist()
    fig_edu = px.bar(
        edu_income, x='education', y='count', color='income',
        title="توزيع فئات الدخل حسب المستوى التعليمي",
        labels={'education': 'المستوى التعليمي', 'count': 'العدد', 'income': 'فئة الدخل'},
        category_orders={'education': edu_order},
        barmode='group',
        color_discrete_map={'<=50K': '#3498db', '>50K': '#2ecc71'}
    )
    fig_edu.update_layout(template='plotly_white', xaxis_tickangle=-45)

    # 2. نسبة توزيع فئات الدخل (Donut Chart)
    fig_pie = px.pie(
        dff, names='income', title="نسبة الدخل (<=50K مقابل >50K)",
        hole=0.4, color='income',
        color_discrete_map={'<=50K': '#3498db', '>50K': '#2ecc71'}
    )

    # 3. توزيع الأعمار حسب الدخل (Histogram)
    fig_age = px.histogram(
        dff, x='age', color='income', barmode='overlay',
        title="توزيع الأعمار بحسب فئة الدخل",
        labels={'age': 'العمر', 'count': 'العدد', 'income': 'فئة الدخل'},
        color_discrete_map={'<=50K': '#3498db', '>50K': '#2ecc71'},
        opacity=0.7
    )
    fig_age.update_layout(template='plotly_white')

    # 4. متوسط ساعات العمل حسب المهن (Bar Chart)
    occ_hours = dff.groupby('occupation')['hours-per-week'].mean().reset_index().sort_values('hours-per-week', ascending=True)
    fig_occ = px.bar(
        occ_hours, x='hours-per-week', y='occupation', orientation='h',
        title="متوسط ساعات العمل الإسبوعية حسب المهنة",
        labels={'hours-per-week': 'ساعات العمل / أسبوع', 'occupation': 'المهنة'},
        color='hours-per-week', color_continuous_scale='Teal'
    )
    fig_occ.update_layout(template='plotly_white')

    # 5. الحالة الاجتماعية والدخل (Bar Chart)
    marital_inc = dff.groupby(['marital-status', 'income']).size().reset_index(name='count')
    fig_marital = px.bar(
        marital_inc, x='marital-status', y='count', color='income',
        title="توزيع الدخل حسب الحالة الاجتماعية",
        labels={'marital-status': 'الحالة الاجتماعية', 'count': 'العدد', 'income': 'فئة الدخل'},
        barmode='stack',
        color_discrete_map={'<=50K': '#3498db', '>50K': '#2ecc71'}
    )
    fig_marital.update_layout(template='plotly_white', xaxis_tickangle=-30)

    # 6. قطاع العمل والدخل (Bar Chart)
    work_inc = dff.groupby(['workclass', 'income']).size().reset_index(name='count')
    fig_work = px.bar(
        work_inc, x='workclass', y='count', color='income',
        title="توزيع الدخل حسب قطاع العمل",
        labels={'workclass': 'قطاع العمل', 'count': 'العدد', 'income': 'فئة الدخل'},
        barmode='group',
        color_discrete_map={'<=50K': '#3498db', '>50K': '#2ecc71'}
    )
    fig_work.update_layout(template='plotly_white', xaxis_tickangle=-30)

    return (
        f"{total_records:,}",
        high_income_pct,
        avg_hours,
        avg_age,
        fig_edu,
        fig_pie,
        fig_age,
        fig_occ,
        fig_marital,
        fig_work
    )

# ==========================================
# 5. تشغيل السيرفر
# ==========================================
if __name__ == '__main__':
    app.run(debug=True, port=8050)