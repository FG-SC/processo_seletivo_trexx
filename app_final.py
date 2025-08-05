import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime

# ===========================================================================
# CONFIGURAÇÃO DA PÁGINA
# ===========================================================================
st.set_page_config(
    page_title="Trexx AI Analytics Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo CSS para melhorar a aparência
st.markdown("""
<style>
    /* Melhora a fonte e o espaçamento */
    .main {
        font-family: 'sans-serif';
    }
    /* Estiliza os títulos das abas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F0F2F6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF;
    }
    /* Adiciona um container com sombra para os gráficos */
    .stPlotlyChart {
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        transition: 0.3s;
    }
    .stPlotlyChart:hover {
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
    }
</style>""", unsafe_allow_html=True)


# ===========================================================================
# CARREGAMENTO DE DADOS (CACHEADO) - CORRIGIDO
# ===========================================================================
# Constante para o diretório de artefatos
ARTIFACTS_DIR = 'artifacts'

# Usar cache para otimizar o carregamento dos dados
@st.cache_data
def load_data(file_name):
    """Carrega um arquivo CSV a partir do nome base, dentro da pasta artifacts."""
    # Mapeamento de nomes amigáveis para nomes de arquivos reais
    file_map = {
        "cluster_analysis": "cluster_analysis.csv",
        "fan_probabilities": "fan_purchase_probabilities.csv",
        "model_comparison": "model_comparison.csv",
        "team_forecasts": "team_revenue_forecasts.csv",
        "revenue_forecast": "revenue_forecast_30d.csv",
        "xgboost_importance": "xgboost_feature_importance.csv",
        "lstm_history": "lstm_training_history.csv"
    }
    
    filename = file_map.get(file_name)
    if filename:
        # Constrói o caminho completo para o arquivo dentro da pasta 'artifacts'
        path = os.path.join(ARTIFACTS_DIR, filename)
        if os.path.exists(path):
            return pd.read_csv(path)
    # Se o arquivo não for encontrado, retorna None para ser tratado nas funções
    return None


# ===========================================================================
# ABA 1: RESUMO EXECUTIVO
# ===========================================================================
def display_home():
    """Exibe a página inicial com o resumo executivo e KPIs."""
    st.title("📊 Resumo Executivo e KPIs")
    st.markdown("Visão geral dos insights e principais indicadores de desempenho (KPIs) extraídos das análises.")

    team_df = load_data("team_forecasts")
    forecast_df = load_data("revenue_forecast")
    fan_df = load_data("fan_probabilities")

    if team_df is not None and forecast_df is not None and fan_df is not None:
        # --- KPIs ---
        total_revenue_forecast = forecast_df['Revenue_Forecast'].sum()
        top_team = team_df.loc[team_df['Expected_Revenue'].idxmax()]
        high_prob_fans = fan_df[fan_df['Purchase_Probability'] > 0.75].shape[0]

        st.markdown("### Principais Indicadores")
        col1, col2, col3 = st.columns(3)
        col1.metric("Receita Prevista (30d)", f"R$ {total_revenue_forecast:,.2f}")
        col2.metric("Time de Maior Potencial", top_team.iloc[0], f"R$ {top_team['Expected_Revenue']:,.2f}")
        col3.metric("Fãs com Alta Prob. de Compra (>75%)", f"{high_prob_fans} fãs")

        st.markdown("---")
        st.markdown(
            """
            A análise aprofundada dos dados de torcedores e transações revelou oportunidades significativas para o crescimento da receita.
            Identificamos dois principais segmentos de fãs (clusters) com comportamentos distintos:

            - **Cluster 2 (Fãs de Alto Valor):** Embora representem uma parcela menor da base, são responsáveis por **74% da receita total**. Eles gastam mais, compram com mais frequência e devem ser o foco principal de campanhas de marketing personalizadas.

            - **Cluster 1 (Fãs Ocasionais):** Um grupo maior com gastos menores. Existem oportunidades para aumentar seu engajamento e valor médio de transação através de ofertas direcionadas.

            O modelo **XGBoost** demonstrou a melhor performance para a previsão de receita diária, sendo a escolha para o forecast de 30 dias. A receita esperada por time indica que **Flamengo** e **Imperial** possuem o maior potencial de faturamento na próxima campanha.
            """
        )
    else:
        st.error("Um ou mais arquivos de dados não foram encontrados na pasta 'artifacts'. Por favor, verifique se os arquivos 'team_revenue_forecasts.csv', 'revenue_forecast_30d.csv' e 'fan_purchase_probabilities.csv' existem.")


# ===========================================================================
# ABA 2: PREVISÃO DE RECEITA
# ===========================================================================
def display_revenue_forecast():
    """Exibe a previsão de receita para os próximos 30 dias."""
    st.title("📈 Previsão de Receita (Próximos 30 Dias)")
    st.markdown("Projeção da receita diária com base no modelo preditivo de melhor performance (XGBoost).")

    forecast_df = load_data("revenue_forecast")
    if forecast_df is not None:
        forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])

        # Gráfico de Linha da Previsão
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'], y=forecast_df['Upper_Bound'],
            fill=None, mode='lines', line_color='rgba(0,176,246,0.2)', name='Limite Superior'
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'], y=forecast_df['Lower_Bound'],
            fill='tonexty', mode='lines', line_color='rgba(0,176,246,0.2)', name='Limite Inferior'
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'], y=forecast_df['Revenue_Forecast'],
            mode='lines+markers', line_color='rgba(0,100,80,0.9)', name='Previsão de Receita'
        ))

        fig.update_layout(
            title='Previsão de Receita Diária para os Próximos 30 Dias',
            xaxis_title='Data',
            yaxis_title='Receita (R$)',
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Análise por Dia da Semana
        st.markdown("### Análise da Receita por Dia da Semana")
        forecast_df['Day_of_Week'] = forecast_df['Date'].dt.day_name()
        revenue_by_day = forecast_df.groupby('Day_of_Week')['Revenue_Forecast'].sum().reindex([
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
        ]).reset_index()

        fig2 = px.bar(
            revenue_by_day,
            x='Day_of_Week',
            y='Revenue_Forecast',
            title='Receita Total Prevista por Dia da Semana',
            labels={'Day_of_Week': 'Dia da Semana', 'Revenue_Forecast': 'Receita Prevista (R$)'},
            color='Revenue_Forecast',
            color_continuous_scale=px.colors.sequential.Viridis_r,
            text_auto='.2f'
        )
        fig2.update_layout(xaxis={'categoryorder':'array', 'categoryarray': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']})
        st.plotly_chart(fig2, use_container_width=True)

    else:
        st.error("Arquivo de previsão de receita ('revenue_forecast_30d.csv') não encontrado na pasta 'artifacts'.")

# ===========================================================================
# ABA 3: RECEITA POR TIME
# ===========================================================================
def display_team_revenue():
    """Exibe a estimativa de receita por time."""
    st.title("🏆 Estimativa de Receita por Time")
    st.markdown("Previsão de receita para cada time na próxima campanha, com base no comportamento dos seus fãs.")

    team_df = load_data("team_forecasts")
    if team_df is not None:
        team_df = team_df.sort_values('Expected_Revenue', ascending=False)
        # A primeira coluna anônima do CSV é lida como 'Unnamed: 0'. Vamos renomeá-la para 'Time'.
        if 'Unnamed: 0' in team_df.columns:
            team_df.rename(columns={'Unnamed: 0': 'Time'}, inplace=True)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### Comparativo de Receita Esperada")
            fig = px.bar(
                team_df,
                x='Time',
                y='Expected_Revenue',
                color='Avg_Purchase_Probability',
                title='Receita Esperada vs. Probabilidade de Compra',
                labels={'Time': 'Time', 'Expected_Revenue': 'Receita Esperada (R$)', 'Avg_Purchase_Probability': 'Prob. Média de Compra'},
                text_auto='.2s',
                color_continuous_scale=px.colors.sequential.Viridis,
                height=500
            )
            fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Distribuição da Receita e Fãs")
            fig2 = px.treemap(
                team_df,
                path=['Time'],
                values='Expected_Revenue',
                color='Fan_Count',
                hover_data=['Historical_Avg'],
                title='Proporção da Receita Esperada por Time',
                color_continuous_scale=px.colors.sequential.Teal,
                height=500
            )
            fig2.update_traces(textinfo='label+percent root')
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("#### Detalhamento dos Dados por Time")
        st.dataframe(team_df.style.format({
            'Expected_Revenue': 'R$ {:,.2f}',
            'Historical_Avg': 'R$ {:,.2f}',
            'Avg_Purchase_Probability': '{:.1%}'
        }), use_container_width=True)

    else:
        st.error("Arquivo de estimativas por time ('team_revenue_forecasts.csv') não encontrado na pasta 'artifacts'.")


# ===========================================================================
# ABA 4: ANÁLISE DE FÃS (CLUSTERIZAÇÃO)
# ===========================================================================
def display_fan_segmentation():
    """Exibe a análise de clusters de fãs."""
    st.title("👥 Análise de Fãs e Segmentação")
    st.markdown("Divisão dos fãs em grupos (clusters) com base em seu comportamento de compra e engajamento.")

    cluster_df = load_data("cluster_analysis")
    fan_df = load_data("fan_probabilities")

    if cluster_df is not None and fan_df is not None:
        cluster_df = cluster_df[cluster_df['Cluster'] != 0] # Remover cluster 0 (inativos)

        st.markdown("### Comparativo entre Clusters")
        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.pie(
                cluster_df,
                names='Cluster',
                values='Revenue_Share',
                title='Fatia da Receita por Cluster (%)',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig1.update_traces(textinfo='percent+label', textfont_size=15)
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig2 = px.bar(
                cluster_df,
                x='Cluster',
                y='Total_Spent_mean',
                title='Gasto Médio por Fã em Cada Cluster',
                labels={'Cluster': 'Cluster', 'Total_Spent_mean': 'Gasto Médio (R$)'},
                color='Cluster',
                text_auto='.2f',
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### Análise Detalhada dos Fãs")
        fig3 = px.scatter(
            fan_df,
            x='Total_Spent',
            y='Purchase_Probability',
            color='Cluster',
            size='Total_Spent',
            hover_data=['fan_id', 'Favorite_Team'],
            title='Relação entre Gasto Total, Probabilidade de Compra e Cluster',
            labels={'Total_Spent': 'Gasto Total (R$)', 'Purchase_Probability': 'Probabilidade de Compra'},
            color_continuous_scale=px.colors.sequential.Viridis,
            size_max=20
        )
        st.plotly_chart(fig3, use_container_width=True)

    else:
        st.error("Arquivos de análise de fãs ('cluster_analysis.csv' ou 'fan_purchase_probabilities.csv') não encontrados na pasta 'artifacts'.")


# ===========================================================================
# ABA 5: DESEMPENHO DOS MODELOS
# ===========================================================================
def display_model_performance():
    """Exibe a performance e métricas dos modelos de ML."""
    st.title("⚙️ Desempenho dos Modelos")
    st.markdown("Comparativo de performance entre os modelos de previsão e análise das features mais importantes.")

    comp_df = load_data("model_comparison")
    importance_df = load_data("xgboost_importance")

    if comp_df is not None:
        st.markdown("### Comparativo de Métricas dos Modelos")
        # Gráfico para RMSE e MAE
        metrics_df = comp_df.melt(id_vars='Model', value_vars=['RMSE', 'MAE'], var_name='Metric', value_name='Value')
        fig1 = px.bar(
            metrics_df,
            x='Metric',
            y='Value',
            color='Model',
            barmode='group',
            title='Comparativo de Erros (RMSE e MAE)',
            labels={'Value': 'Valor do Erro', 'Metric': 'Métrica'},
            text_auto='.2f'
        )
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.warning("Arquivo 'model_comparison.csv' não encontrado na pasta 'artifacts'.")

    if importance_df is not None:
        st.markdown("### Importância das Features (Modelo XGBoost)")
        # Gráfico de importância de features
        importance_df = importance_df.nlargest(15, 'importance')
        fig2 = px.bar(
            importance_df,
            x='importance',
            y='feature',
            orientation='h',
            title='Top 15 Features Mais Importantes para Previsão de Receita',
            labels={'importance': 'Importância', 'feature': 'Feature'},
            color='importance',
            color_continuous_scale=px.colors.sequential.Viridis_r
        )
        fig2.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Arquivo 'xgboost_feature_importance.csv' não encontrado na pasta 'artifacts'.")


# ===========================================================================
# ESTRUTURA PRINCIPAL COM ABAS
# ===========================================================================

# Título Principal do Dashboard
st.sidebar.title("⚽ Trexx AI Analytics")
st.sidebar.markdown("Dashboard para análise de dados de engajamento e previsão de receita de clubes de futebol.")
st.sidebar.info("Navegue pelas abas abaixo para explorar as diferentes análises.")


# Criação das abas de navegação
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Resumo Executivo",
    "📈 Previsão de Receita",
    "🏆 Receita por Time",
    "👥 Análise de Fãs",
    "⚙️ Performance dos Modelos"
])

with tab1:
    display_home()

with tab2:
    display_revenue_forecast()

with tab3:
    display_team_revenue()

with tab4:
    display_fan_segmentation()

with tab5:
    display_model_performance()