import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px

# ==========================================
# 1. تحميل وتنظيف البيانات
# ==========================================
df = pd.read_csv('data.csv')

# تنظيف عمود السعر
df['price_clean'] = df['price'].astype(str).str.replace(',', '').str.replace(' ', '')
df['price_clean'] = pd.to_numeric(df['price_clean'], errors='coerce')

# تنظيف عمود الكيلومترات
df['km_clean'] = df['kilo_meter'].astype(str).str.replace(',', '').str.replace('KM', '', case=False).str.strip()
df['km_clean'] = pd.to_numeric(df['km_clean'], errors='coerce')

# تنظيف عمود السنة
df['year_clean'] = pd.to_numeric(df['year'], errors='coerce')

# استبعاد القيم الشاذة جداً لتحسين الرؤية البيانية
df_clean = df[(df['price_clean'] > 10000) & (df['price_clean'] <= 30000000) & (df['year_clean'] >= 1980)].copy()

# ==========================================
# 2. إعداد تطبيق Dash
# ==========================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "لوحة تحكم تحليل سوق السيارات"

# قائمة الماركات والوقود
brand_options = [{'label': 'الكل', 'value': 'ALL'}] + [{'label': b, 'value': b} for b in sorted(df_clean['brand'].dropna().unique())]
condition_options = [{'label': 'الكل', 'value': 'ALL'}] + [{'label': c, 'value': c} for c in df_clean['condition'].unique() if pd.notna(c)]
fuel_options = [{'label': 'الكل', 'value': 'ALL'}] + [{'label': f, 'value': f} for f in df_clean['fuel_type'].unique() if pd.notna(f)]

min_year = int(df_clean['year_clean'].min())
max_year = int(df_clean['year_clean'].max())

# ==========================================
# 3. تصميم واجهة المستخدم (Layout)
# ==========================================
app.layout = dbc.Container([
    # العنوان الرئيسي
    dbc.Row([
        dbc.Col(
            html.Div([
                html.H1("🏎️ لوحة تحكم تحليل سوق السيارات", className="text-center text-primary fw-bold my-3"),
                html.P("استعراض تفاعلي لأسعار ومواصفات السيارات من واقع البيانات", className="text-center text-muted fs-5")
            ]),
            width=12
        )
    ]),

    # فلاتر التصفية (Sidebar / Filter Panel)
    dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("الماركة (Brand):", className="fw-bold"),
                    dcc.Dropdown(id='brand-filter', options=brand_options, value='ALL', clearable=False)
                ], md=3),
                dbc.Col([
                    html.Label("الحالة (Condition):", className="fw-bold"),
                    dcc.Dropdown(id='condition-filter', options=condition_options, value='ALL', clearable=False)
                ], md=3),
                dbc.Col([
                    html.Label("نوع الوقود (Fuel Type):", className="fw-bold"),
                    dcc.Dropdown(id='fuel-filter', options=fuel_options, value='ALL', clearable=False)
                ], md=3),
                dbc.Col([
                    html.Label("سنة الصنع (Year Range):", className="fw-bold"),
                    dcc.RangeSlider(
                        id='year-slider',
                        min=min_year,
                        max=max_year,
                        step=1,
                        value=[2000, max_year],
                        marks={y: str(y) for y in range(min_year, max_year + 1, 5)},
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
                html.H6("إجمالي عدد السيارات", className="card-subtitle text-muted"),
                html.H3(id="kpi-total-cars", className="text-primary fw-bold mt-2")
            ])
        ], className="text-center shadow-sm"), md=3),

        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("متوسط السعر (جنيه)", className="card-subtitle text-muted"),
                html.H3(id="kpi-avg-price", className="text-success fw-bold mt-2")
            ])
        ], className="text-center shadow-sm"), md=3),

        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("متوسط المسافة (KM)", className="card-subtitle text-muted"),
                html.H3(id="kpi-avg-km", className="text-warning fw-bold mt-2")
            ])
        ], className="text-center shadow-sm"), md=3),

        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H6("الماركة الأكثر انتشاراً", className="card-subtitle text-muted"),
                html.H3(id="kpi-top-brand", className="text-info fw-bold mt-2")
            ])
        ], className="text-center shadow-sm"), md=3),
    ], className="mb-4"),

    # الرسوم البيانية - الصف الأول (الماركات والتغير مع السنين)
    dbc.Row([
        dbc.Col(dcc.Graph(id='top-brands-chart'), md=6),
        dbc.Col(dcc.Graph(id='price-by-year-chart'), md=6),
    ], className="mb-4"),

    # الرسوم البيانية - الصف الثاني (الحالة والوقود والمواقع)
    dbc.Row([
        dbc.Col(dcc.Graph(id='condition-pie-chart'), md=4),
        dbc.Col(dcc.Graph(id='fuel-pie-chart'), md=4),
        dbc.Col(dcc.Graph(id='top-locations-chart'), md=4),
    ], className="mb-4"),

    # الرسوم البيانية الإضافية - الصف الثالث (Histogram + Heatmap)
    dbc.Row([
        dbc.Col(dcc.Graph(id='price-histogram-chart'), md=6),
        dbc.Col(dcc.Graph(id='correlation-heatmap-chart'), md=6),
    ], className="mb-4"),

    # الرسوم البيانية الإضافية - الصف الرابع (Box Plot)
    dbc.Row([
        dbc.Col(dcc.Graph(id='price-boxplot-chart'), md=12),
    ], className="mb-4"),

], fluid=True, style={'backgroundColor': '#f8f9fa', 'padding': '20px'})


# ==========================================
# 4. التفاعل والتحديث التلقائي (Callbacks)
# ==========================================
@app.callback(
    [
        Output('kpi-total-cars', 'children'),
        Output('kpi-avg-price', 'children'),
        Output('kpi-avg-km', 'children'),
        Output('kpi-top-brand', 'children'),
        Output('top-brands-chart', 'figure'),
        Output('price-by-year-chart', 'figure'),
        Output('condition-pie-chart', 'figure'),
        Output('fuel-pie-chart', 'figure'),
        Output('top-locations-chart', 'figure'),
        Output('price-histogram-chart', 'figure'),
        Output('correlation-heatmap-chart', 'figure'),
        Output('price-boxplot-chart', 'figure'),
    ],
    [
        Input('brand-filter', 'value'),
        Input('condition-filter', 'value'),
        Input('fuel-filter', 'value'),
        Input('year-slider', 'value')
    ]
)
def update_dashboard(selected_brand, selected_condition, selected_fuel, year_range):
    # تصفية البيانات
    dff = df_clean[
        (df_clean['year_clean'] >= year_range[0]) & 
        (df_clean['year_clean'] <= year_range[1])
    ]

    if selected_brand != 'ALL':
        dff = dff[dff['brand'] == selected_brand]
    if selected_condition != 'ALL':
        dff = dff[dff['condition'] == selected_condition]
    if selected_fuel != 'ALL':
        dff = dff[dff['fuel_type'] == selected_fuel]

    # حساب المؤشرات
    total_cars = len(dff)
    avg_price = f"{dff['price_clean'].mean():,.0f}" if total_cars > 0 else "0"
    avg_km = f"{dff['km_clean'].mean():,.0f}" if total_cars > 0 else "0"
    top_brand = dff['brand'].mode()[0] if total_cars > 0 and not dff['brand'].empty else "N/A"

    # 1. أكثر الماركات تكراراً
    brand_counts = dff['brand'].value_counts().head(10).reset_index()
    brand_counts.columns = ['brand', 'count']
    fig_brands = px.bar(
        brand_counts, x='count', y='brand', orientation='h',
        title="أكثر 10 ماركات تكراراً",
        labels={'count': 'عدد السيارات', 'brand': 'الماركة'},
        color='count', color_continuous_scale='Viridis'
    )
    fig_brands.update_layout(yaxis={'categoryorder': 'total ascending'}, template='plotly_white')

    # 2. متوسط السعر حسب سنة الصنع
    price_year = dff.groupby('year_clean')['price_clean'].mean().reset_index()
    fig_price_year = px.line(
        price_year, x='year_clean', y='price_clean', markers=True,
        title="متوسط السعر حسب سنة الصنع",
        labels={'year_clean': 'سنة الصنع', 'price_clean': 'متوسط السعر (EGP)'}
    )
    fig_price_year.update_layout(template='plotly_white')

    # 3. توزيع حالة السيارات (مستعمل / جديد)
    fig_condition = px.pie(
        dff, names='condition', title="توزيع حالة السيارات",
        hole=0.4, color_discrete_sequence=px.colors.qualitative.Set2
    )

    # 4. توزيع نوع الوقود
    fig_fuel = px.pie(
        dff, names='fuel_type', title="توزيع أنواع الوقود",
        hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel
    )

    # 5. أكثر المواقع/المحافظات انتشاراً
    loc_counts = dff['location'].value_counts().head(8).reset_index()
    loc_counts.columns = ['location', 'count']
    fig_loc = px.bar(
        loc_counts, x='location', y='count',
        title="أكثر المناطق عرضاً للسيارات",
        labels={'location': 'الموقع', 'count': 'العدد'},
        color='count', color_continuous_scale='Blues'
    )
    fig_loc.update_layout(template='plotly_white')

    # 6. مدرج تكراري لتوزيع الأسعار (Histogram)
    fig_hist = px.histogram(
        dff, x='price_clean', nbins=30,
        title="توزيع الأسعار (Histogram)",
        labels={'price_clean': 'السعر (EGP)', 'count': 'عدد السيارات'},
        color_discrete_sequence=['#2ca02c']
    )
    fig_hist.update_layout(template='plotly_white', yaxis_title="عدد السيارات")

    # 7. خريطة حرارية للارتباط (Correlation Heatmap)
    corr_df = dff[['price_clean', 'km_clean', 'year_clean']].corr()
    fig_heatmap = px.imshow(
        corr_df,
        text_auto=".2f",
        aspect="auto",
        title="خريطة الارتباط الحرارية (Correlation Heatmap)",
        labels=dict(x="المتغير", y="المتغير", color="معامل الارتباط"),
        x=['السعر', 'الكيلومترات', 'سنة الصنع'],
        y=['السعر', 'الكيلومترات', 'سنة الصنع'],
        color_continuous_scale='RdBu_r'
    )
    fig_heatmap.update_layout(template='plotly_white')

    # 8. مخطط الصندوق للأسعار حسب الماركات (Box Plot)
    top_10_brands_list = brand_counts['brand'].tolist()
    dff_top_brands = dff[dff['brand'].isin(top_10_brands_list)]
    
    fig_box = px.box(
        dff_top_brands, x='brand', y='price_clean', color='brand',
        title="توزيع الأسعار والقيم الشاذة لأعلى 10 ماركات (Box Plot)",
        labels={'brand': 'الماركة', 'price_clean': 'السعر (EGP)'}
    )
    fig_box.update_layout(template='plotly_white', showlegend=False)

    return (
        f"{total_cars:,}",
        f"{avg_price} EGP",
        f"{avg_km} KM",
        top_brand,
        fig_brands,
        fig_price_year,
        fig_condition,
        fig_fuel,
        fig_loc,
        fig_hist,
        fig_heatmap,
        fig_box
    )

# ==========================================
# 5. تشغيل السيرفر
# ==========================================
if __name__ == '__main__':
    app.run(debug=True, port=8050)