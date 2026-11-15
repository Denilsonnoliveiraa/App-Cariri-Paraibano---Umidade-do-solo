
# ============================================================
# ATLAS DA UMIDADE DO SOLO — CARIRI
# STREAMLIT | PYTHON | VS CODE
#
# AUTORES:
# Denilson de Oliveira Silva
# Jader da Silva Jale
# Silvio Fernando Alves Xavier
#
# FUNCIONALIDADES:
# 1. Autores
# 2. Documentação
# 3. Estatística descritiva
# 4. Séries temporais
# 5. Anomalias
# 6. Mapas individuais e recortes
# 7. Conclusão científica
# ============================================================

from pathlib import Path
import re
import unicodedata
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from scipy.stats import gaussian_kde
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# ============================================================
# 1. CONFIGURAÇÕES
# ============================================================

st.set_page_config(
    page_title="Atlas da Umidade do Solo | Cariri",
    page_icon="🌎",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE = Path(__file__).resolve().parent

NOME_EXCEL = "Umidade_Solo_Diaria_Cariri_1991_2026.xlsx"

CANDIDATOS_EXCEL = [
    BASE / "dados" / NOME_EXCEL,
    BASE / NOME_EXCEL,
    BASE.parent / "dados" / NOME_EXCEL
]

ARQUIVO_EXCEL = next(
    (p for p in CANDIDATOS_EXCEL if p.exists()),
    CANDIDATOS_EXCEL[0]
)

CANDIDATOS_ATLAS = [
    BASE / "ATLAS_UMIDADE_CARIRI",
    BASE.parent / "ATLAS_UMIDADE_CARIRI",
    BASE
]

PASTA_ATLAS = next(
    (
        p for p in CANDIDATOS_ATLAS
        if (p / "02_MAPAS_INDIVIDUAIS").is_dir()
    ),
    CANDIDATOS_ATLAS[0]
)

PASTA_INDIVIDUAIS = PASTA_ATLAS / "02_MAPAS_INDIVIDUAIS"
PASTA_RECORTES = PASTA_ATLAS / "03_RECORTES_TEMPORAIS"

ALTURA_GRAFICO = 430

# ============================================================
# 2. IDENTIDADE VISUAL
# ============================================================

FUNDO = "#F6F2EA"
PAPEL = "#FFFEFC"
AREIA = "#E8D8BE"
TERRA = "#A66B45"
MARROM = "#65442F"
CAFE = "#30251E"
VERDE = "#718677"
AZUL = "#406F7A"
TEXTO = "#201C18"

MESES = [
    "Jan", "Fev", "Mar", "Abr",
    "Mai", "Jun", "Jul", "Ago",
    "Set", "Out", "Nov", "Dez"
]


# ============================================================
# 3. CSS — LAYOUT CARTOGRÁFICO
# ============================================================

st.markdown("""
<style>

:root {
    color-scheme: light;
}

html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: #F6F2EA !important;
    color: #201C18 !important;
}

[data-testid="stMainBlockContainer"] {
    max-width: 1500px;
    padding-top: 1.4rem;
    padding-bottom: 3rem;
}

/* FUNDO CLARO: TEXTO ESCURO */

[data-testid="stMain"] h1,
[data-testid="stMain"] h2,
[data-testid="stMain"] h3,
[data-testid="stMain"] h4 {
    color: #30251E !important;
    font-family: Georgia, serif;
}

[data-testid="stMain"] p,
[data-testid="stMain"] label,
[data-testid="stMain"] li {
    color: #201C18;
}

/* BARRA LATERAL */

[data-testid="stSidebar"],
[data-testid="stSidebarContent"] {
    background: #30251E !important;
    color: #FFFFFF !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: #FFFFFF !important;
}

[data-testid="stSidebar"] hr {
    border-color: #756252 !important;
}

/* BOTÕES ESCUROS: TEXTO SEMPRE BRANCO */

.stButton > button,
.stDownloadButton > button,
.stFormSubmitButton > button {
    background: #30251E !important;
    color: #FFFFFF !important;
    border: 1px solid #30251E !important;
    border-radius: 9px !important;
    min-height: 43px;
    font-weight: 650 !important;
}

.stButton > button *,
.stDownloadButton > button *,
.stFormSubmitButton > button * {
    color: #FFFFFF !important;
}

.stButton > button:hover,
.stDownloadButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #65442F !important;
    color: #FFFFFF !important;
    border-color: #65442F !important;
}

.stButton > button:hover *,
.stDownloadButton > button:hover *,
.stFormSubmitButton > button:hover * {
    color: #FFFFFF !important;
}

.stButton > button:active,
.stButton > button:focus,
.stDownloadButton > button:active,
.stDownloadButton > button:focus {
    background: #79543C !important;
    color: #FFFFFF !important;
}

.stButton > button:active *,
.stButton > button:focus *,
.stDownloadButton > button:active *,
.stDownloadButton > button:focus * {
    color: #FFFFFF !important;
}

.stButton > button:disabled {
    background: #75695F !important;
    color: #FFFFFF !important;
    opacity: 0.7;
}

.stButton > button:disabled * {
    color: #FFFFFF !important;
}

/* ABAS */

button[data-baseweb="tab"] {
    background: #EDE4D6 !important;
    border-radius: 8px 8px 0 0;
    padding: 12px 15px !important;
    margin-right: 4px;
}

button[data-baseweb="tab"] * {
    color: #30251E !important;
    font-weight: 600 !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    background: #65442F !important;
}

button[data-baseweb="tab"][aria-selected="true"] * {
    color: #FFFFFF !important;
}

[data-baseweb="tab-highlight"] {
    background: #A66B45 !important;
}

/* INDICADORES */

[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 1px solid #DACAB5;
    border-top: 4px solid #A66B45;
    border-radius: 12px;
    padding: 15px;
}

[data-testid="stMetric"] * {
    color: #201C18 !important;
}

/* CAMPOS DE SELEÇÃO */

[data-baseweb="select"] > div,
[data-baseweb="input"] > div {
    background: #FFFFFF !important;
    color: #201C18 !important;
    border-color: #DCCAB5 !important;
}

[data-testid="stMain"] input {
    color: #201C18 !important;
}

[data-baseweb="popover"],
[data-baseweb="popover"] ul,
[data-baseweb="popover"] li {
    background: #FFFFFF !important;
    color: #201C18 !important;
}

[data-baseweb="popover"] * {
    color: #201C18 !important;
}

/* CARTÕES */

[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: #DCCAB5 !important;
    border-radius: 12px;
}

[data-testid="stExpander"] {
    background: #FFFEFC !important;
    border: 1px solid #DACAB5;
    border-radius: 10px;
}

/* CABEÇALHO */

.atlas-topo {
    background: #FFFDFC;
    border: 1px solid #DCCAB5;
    border-left: 8px solid #A66B45;
    border-radius: 13px;
    padding: 27px 33px;
    margin-bottom: 23px;
}

.atlas-selo {
    color: #A66B45;
    font-size: 11px;
    letter-spacing: 2.5px;
    font-weight: 700;
    text-transform: uppercase;
    margin-bottom: 11px;
}

.atlas-titulo {
    color: #30251E;
    font-family: Georgia, serif;
    font-size: clamp(25px, 3vw, 38px);
    font-weight: bold;
}

.atlas-subtitulo {
    color: #65442F;
    font-size: 15px;
    margin-top: 10px;
}

.atlas-faixa {
    height: 6px;
    margin-top: 22px;
    border-radius: 5px;
    background: linear-gradient(
        90deg,
        #A66B45,
        #E8D8BE,
        #718677,
        #406F7A
    );
}

/* TEXTOS JUSTIFICADOS */

.texto-cientifico {
    text-align: justify;
    color: #201C18;
    font-size: 16px;
    line-height: 1.9;
    margin-bottom: 17px;
}

/* RODAPÉ */

.rodape {
    margin-top: 40px;
    padding: 20px;
    border-top: 1px solid #DACAB5;
    text-align: center;
    font-size: 12px;
    color: #665F56;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 4. FUNÇÕES VISUAIS
# ============================================================

def texto_justificado(conteudo):
    st.markdown(
        '<div class="texto-cientifico">' +
        conteudo +
        '</div>',
        unsafe_allow_html=True
    )


def estilizar_grafico(fig):

    fig.update_layout(
        template="plotly_white",
        height=ALTURA_GRAFICO,
        paper_bgcolor=PAPEL,
        plot_bgcolor=PAPEL,
        font=dict(
            family="Arial",
            color=TEXTO,
            size=12
        ),
        title=dict(
            font=dict(color=MARROM, size=17),
            x=0.03,
            xanchor="left"
        ),
        hoverlabel=dict(
            bgcolor=CAFE,
            font_color="#FFFFFF"
        ),
        margin=dict(
            l=40, r=25, t=65, b=55
        )
    )

    fig.update_xaxes(
        showgrid=False,
        tickfont=dict(color=TEXTO),
        title_font=dict(color=TEXTO)
    )

    fig.update_yaxes(
        gridcolor="#E9E1D6",
        tickfont=dict(color=TEXTO),
        title_font=dict(color=TEXTO)
    )

    return fig


def exibir_grafico(fig):

    st.plotly_chart(
        estilizar_grafico(fig),
        use_container_width=True,
        config={"displaylogo": False}
    )


def baixar_csv(df, nome):

    st.download_button(
        "Baixar tabela CSV",
        data=df.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name=nome,
        mime="text/csv"
    )


def formatar_numero(valor, casas=3):

    if pd.isna(valor):
        return "—"

    return f"{valor:.{casas}f}".replace(".", ",")


# ============================================================
# 5. LEITURA AUTOMÁTICA DO EXCEL
# ============================================================

def normalizar_nome(texto):

    texto = str(texto).strip().lower()

    texto = unicodedata.normalize(
        "NFKD", texto
    )

    texto = "".join(
        c for c in texto
        if not unicodedata.combining(c)
    )

    texto = re.sub(
        r"[^a-z0-9]+", "_", texto
    )

    return texto.strip("_")


@st.cache_data(show_spinner="Lendo dados diários...")
def ler_excel(caminho, modificacao):

    excel = pd.ExcelFile(caminho)
    abas = excel.sheet_names

    nome_aba = next(
        (
            nome for nome in abas
            if normalizar_nome(nome)
            in ["dados_diarios", "dados_diario"]
        ),
        abas[0]
    )

    df = pd.read_excel(
        excel,
        sheet_name=nome_aba
    )

    return df, abas, nome_aba


def converter_numerico(coluna):

    if pd.api.types.is_numeric_dtype(coluna):
        return pd.to_numeric(
            coluna, errors="coerce"
        )

    return pd.to_numeric(
        coluna.astype(str)
        .str.strip()
        .str.replace(",", ".", regex=False),
        errors="coerce"
    )


def preparar_dados(df):

    df = df.copy()

    df.columns = [
        normalizar_nome(c)
        for c in df.columns
    ]

    obrigatorias = [
        "data",
        "umidade_solo_m3_m3"
    ]

    faltantes = [
        c for c in obrigatorias
        if c not in df.columns
    ]

    if faltantes:
        raise ValueError(
            "Colunas não encontradas: " +
            ", ".join(faltantes)
        )

    if pd.api.types.is_numeric_dtype(df["data"]):

        df["data"] = pd.to_datetime(
            df["data"],
            unit="D",
            origin="1899-12-30",
            errors="coerce"
        )

    else:

        df["data"] = pd.to_datetime(
            df["data"],
            errors="coerce"
        )

    df["umidade_m3m3"] = converter_numerico(
        df["umidade_solo_m3_m3"]
    )

    if "umidade_solo_percentual" in df.columns:

        df["umidade_percentual"] = converter_numerico(
            df["umidade_solo_percentual"]
        )

    else:

        df["umidade_percentual"] = (
            df["umidade_m3m3"] * 100
        )

    df = df.dropna(
        subset=["data", "umidade_m3m3"]
    )

    df = df.sort_values("data")

    df = (
        df.groupby("data", as_index=False)
        .agg(
            umidade_m3m3=("umidade_m3m3", "mean"),
            umidade_percentual=(
                "umidade_percentual", "mean"
            )
        )
    )

    df["ano"] = df["data"].dt.year
    df["mes"] = df["data"].dt.month
    df["dia"] = df["data"].dt.day
    df["dia_ano"] = df["data"].dt.dayofyear

    ano_inicio = (
        1991 + 2 * ((df["ano"] - 1991) // 2)
    )

    df["periodo"] = (
        ano_inicio.astype(str)
        + "-"
        + (ano_inicio + 1).astype(str)
    )

    df["mes_nome"] = df["mes"].map(
        dict(enumerate(MESES, start=1))
    )

    return df


dados = None
abas_excel = []
nome_aba_excel = None
erro_excel = None

if ARQUIVO_EXCEL.exists():

    try:

        tabela, abas_excel, nome_aba_excel = (
            ler_excel(
                str(ARQUIVO_EXCEL),
                ARQUIVO_EXCEL.stat().st_mtime_ns
            )
        )

        dados = preparar_dados(tabela)

        if dados.empty:
            erro_excel = "A planilha não contém dados válidos."
            dados = None

    except Exception as erro:
        erro_excel = str(erro)

else:
    erro_excel = (
        "Excel não encontrado: " +
        str(ARQUIVO_EXCEL)
    )


# ============================================================
# 6. BARRA LATERAL
# ============================================================

with st.sidebar:

    st.markdown("## 🧭 ATLAS CARIRI")

    st.divider()

    st.markdown("### BASE DE DADOS")

    if dados is not None:

        st.success("Base Excel conectada")

        st.write(
            f"Registros diários: {len(dados):,}"
        )

        st.write(
            f"Abas Excel: {len(abas_excel)}"
        )

        st.write(
            f"Aba utilizada: {nome_aba_excel}"
        )

        st.write(
            f"Início: {dados['data'].min():%d/%m/%Y}"
        )

        st.write(
            f"Fim: {dados['data'].max():%d/%m/%Y}"
        )

    else:

        st.warning("Aguardando planilha local.")

        if erro_excel:
            st.caption(erro_excel)

    st.divider()

    st.markdown("### UNIDADE DA UMIDADE")

    unidade_opcao = st.radio(
        "Visualização dos valores",
        [
            "Umidade volumétrica (m³/m³)",
            "Umidade percentual (%)"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    
    st.markdown("### AUTORES")

    st.write("Denilson")
    st.write("Jader")
    st.write("Silvio")
    st.write("Débora")



if unidade_opcao.startswith("Umidade volumétrica"):

    COLUNA_UMIDADE = "umidade_m3m3"
    UNIDADE = "m³/m³"

else:

    COLUNA_UMIDADE = "umidade_percentual"
    UNIDADE = "%"


if dados is not None:

    dados = dados.copy()
    dados["umidade"] = dados[COLUNA_UMIDADE]


# ============================================================
# 7. ESTATÍSTICAS E ANÁLISES TEMPORAIS
# ============================================================

def calcular_estatisticas(df):

    return (
        df.groupby("periodo", as_index=False)
        .agg(
            media=("umidade", "mean"),
            mediana=("umidade", "median"),
            desvio=("umidade", "std"),
            minimo=("umidade", "min"),
            maximo=("umidade", "max"),
            n=("umidade", "count")
        )
        .sort_values("periodo")
    )


def criar_serie_diaria(df):

    return (
        df.set_index("data")["umidade"]
        .sort_index()
    )


def criar_serie_mensal(df):

    return (
        criar_serie_diaria(df)
        .resample("MS")
        .mean()
    )


def calcular_anomalias(df):

    mensal = (
        df.set_index("data")["umidade"]
        .resample("MS")
        .mean()
        .rename("umidade")
        .to_frame()
    )

    mensal["mes"] = mensal.index.month

    mensal["climatologia"] = (
        mensal.groupby("mes")["umidade"]
        .transform("mean")
    )

    mensal["anomalia"] = (
        mensal["umidade"]
        - mensal["climatologia"]
    )

    mensal["ano"] = mensal.index.year

    mensal["mes_nome"] = mensal["mes"].map(
        dict(enumerate(MESES, start=1))
    )

    return mensal.reset_index()


@st.cache_data(show_spinner=False)
def calcular_previsao(serie_mensal, horizonte=12):

    serie = serie_mensal.asfreq("MS").copy()

    if len(serie) < 24:
        raise ValueError(
            "São necessários pelo menos 24 meses."
        )

    if serie.isna().mean() > 0.05:
        raise ValueError(
            "Mais de 5% dos meses estão sem dados."
        )

    serie = serie.interpolate(
        method="time",
        limit=2,
        limit_direction="both"
    )

    if serie.isna().any():
        raise ValueError(
            "Existem lacunas mensais não resolvidas."
        )

    with warnings.catch_warnings():

        warnings.simplefilter("ignore")

        modelo = ExponentialSmoothing(
            serie,
            trend="add",
            seasonal="add",
            seasonal_periods=12,
            initialization_method="estimated"
        )

        ajuste = modelo.fit(optimized=True)

    previsao = ajuste.forecast(horizonte)
    previsao = previsao.clip(lower=0)

    return previsao, ajuste.fittedvalues


# ============================================================
# 8. LOCALIZAÇÃO DOS MAPAS
# ============================================================

def listar_mapas(pasta):

    if not pasta.exists():
        return []

    arquivos = [
        p for p in pasta.iterdir()
        if p.is_file()
        and p.suffix.lower() == ".png"
    ]

    def chave(p):

        anos = re.findall(
            r"(?:19|20)\d{2}",
            p.stem
        )

        if anos:
            return (int(anos[0]), p.name)

        return (9999, p.name)

    return sorted(arquivos, key=chave)


def nome_mapa(arquivo):

    anos = re.findall(
        r"(?:19|20)\d{2}",
        arquivo.stem
    )

    if len(anos) >= 2:
        return f"{anos[0]}–{anos[1]}"

    return arquivo.stem.replace("_", " ").title()


mapas_individuais = listar_mapas(PASTA_INDIVIDUAIS)
mapas_recortes = listar_mapas(PASTA_RECORTES)


if "mapa_tela" not in st.session_state:
    st.session_state.mapa_tela = "inicio"

if "mapa_categoria" not in st.session_state:
    st.session_state.mapa_categoria = None

if "mapa_atual" not in st.session_state:
    st.session_state.mapa_atual = None


def abrir_categoria(categoria):

    st.session_state.mapa_tela = "galeria"
    st.session_state.mapa_categoria = categoria


def abrir_mapa(arquivo, categoria):

    st.session_state.mapa_tela = "ampliado"
    st.session_state.mapa_categoria = categoria
    st.session_state.mapa_atual = str(arquivo)


def voltar_inicio():

    st.session_state.mapa_tela = "inicio"
    st.session_state.mapa_categoria = None
    st.session_state.mapa_atual = None


def voltar_galeria():

    st.session_state.mapa_tela = "galeria"
    st.session_state.mapa_atual = None


def galeria_mapas(arquivos, categoria):

    if not arquivos:
        st.warning("Nenhum mapa PNG encontrado.")
        return

    for inicio in range(0, len(arquivos), 3):

        colunas = st.columns(3, gap="medium")

        linha = arquivos[inicio:inicio + 3]

        for i, arquivo in enumerate(linha):

            with colunas[i]:

                with st.container(border=True):

                    st.markdown(
                        f"#### {nome_mapa(arquivo)}"
                    )

                    st.image(
                        str(arquivo),
                        use_container_width=True
                    )

                    st.button(
                        "Visualizar mapa →",
                        key=f"abrir_{categoria}_{arquivo.stem}",
                        use_container_width=True,
                        on_click=abrir_mapa,
                        args=(arquivo, categoria)
                    )


# ============================================================
# 9. CABEÇALHO CORRIGIDO — SEM BLOCO PRETO
# ============================================================

# O HTML está em uma única expressão, sem indentação
# Markdown que poderia ser interpretada como código.

st.markdown(
    '<div class="atlas-topo">'
    '<div class="atlas-selo">'
    'Atlas ambiental • Região do Cariri'
    '</div>'
    '<div class="atlas-titulo">'
    'Dinâmica da Umidade do Solo'
    '</div>'
    '<div class="atlas-subtitulo">'
    'Série histórica 1991–2026 | '
    'Análises estatísticas e cartográficas'
    '</div>'
    '<div class="atlas-faixa"></div>'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# 10. ABAS PRINCIPAIS
# ============================================================

abas = st.tabs([
    "Autores",
    "Documentação",
    "Descritiva",
    "Séries temporais",
    "Anomalias",
    "Mapas",
    "Conclusão"
])


# ============================================================
# ABA 1 — AUTORES COMPLETOS
# ============================================================


# ============================================================
# ABA 1 — AUTORES COMPLETOS
# ============================================================

with abas[0]:

    st.header("Equipe de Autores")

    texto_justificado(
        "O Atlas da Umidade do Solo do Cariri "
        "é desenvolvido por uma equipe vinculada "
        "a instituições públicas de ensino superior, "
        "integrando métodos estatísticos, "
        "análises temporais e visualização "
        "cartográfica de dados ambientais."
    )

    autores = [
        {
            "nome": "Denilson de Oliveira Silva",
            "instituicao": "UFRPE",
            "vinculo": (
                "Programa de Pós-Graduação em "
                "Biometria e Estatística Aplicada"
            ),
            "email": "denilson.oliveira@ufrpe.br",
            "orcid": "0009-0000-4031-7772"
        },
        {
            "nome": "Jader da Silva Jale",
            "instituicao": "UFRPE",
            "vinculo": (
                "Programa de Pós-Graduação em "
                "Biometria e Estatística Aplicada"
            ),
            "email": "jader.silvajale@ufrpe.br",
            "orcid": "0000-0001-7414-1154"
        },
        {
            "nome": "Silvio Fernando Alves Xavier",
            "instituicao": "UEPB",
            "vinculo": "Departamento de Estatística",
            "email": "silvio@servidor.uepb.edu.br",
            "orcid": "0000-0002-4832-0711"
        },
        {
            "nome": "Débora de Sousa Cordeiro",
            "instituicao": "UEPB",
            "vinculo": "Departamento de Estatística",
            "localizacao": "Campina Grande, Paraíba, Brasil",
            "email": "ddesousacordeiro2015@gmail.com",
            "orcid": None,
            "lattes": None
        }
    ]

    # Exibição em duas colunas
    colunas = st.columns(2, gap="medium")

    for indice, autor in enumerate(autores):

        with colunas[indice % 2]:

            with st.container(border=True):

                st.markdown(f"### 👤 {autor['nome']}")

                st.markdown(
                    f"**Instituição:** {autor['instituicao']}"
                )

                st.markdown(
                    f"**Vínculo acadêmico:** {autor['vinculo']}"
                )

                if autor.get("localizacao"):
                    st.markdown(
                        f"**Localização:** {autor['localizacao']}"
                    )

                st.divider()

                st.markdown(
                    f"**E-mail:** "
                    f"[{autor['email']}]"
                    f"(mailto:{autor['email']})"
                )

                if autor.get("orcid"):
                    st.markdown(
                        f"**ORCID:** "
                        f"[{autor['orcid']}]"
                        f"(https://orcid.org/{autor['orcid']})"
                    )
                else:
                    st.markdown(
                        "**ORCID:** Link não informado"
                    )

                if autor.get("lattes"):
                    st.markdown(
                        f"**Currículo Lattes:** "
                        f"[Acessar currículo]({autor['lattes']})"
                    )
                elif autor["nome"] == "Débora Cordeiro de Sousa":
                    st.markdown(
                        "**Currículo Lattes:** Link não informado"
                    )

    st.divider()

    st.subheader("Instituições")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            "**Universidade Federal Rural de Pernambuco (UFRPE)**"
        )
        st.write(
            "Programa de Pós-Graduação em "
            "Biometria e Estatística Aplicada"
        )
        st.write("Recife, Pernambuco, Brasil")

    with c2:
        st.markdown(
            "**Universidade Estadual da Paraíba (UEPB)**"
        )
        st.write("Departamento de Estatística")
        st.write("Campina Grande, Paraíba, Brasil")


# ============================================================
# ABA 2 — DOCUMENTAÇÃO
# ============================================================

with abas[1]:

    st.header("Documentação Científica")

    st.subheader("1. Resumo da aplicação")

    texto_justificado(
        """
        A aplicação tem como objetivo organizar
        e explorar o comportamento temporal da
        umidade do solo na região do Cariri,
        considerando registros diários e produtos
        cartográficos associados ao intervalo
        histórico de 1991 a 2026. A plataforma
        reúne análises descritivas, séries temporais,
        médias móveis, anomalias sazonais e projeções
        exploratórias, permitindo uma visualização
        integrada dos resultados.
        """
    )

    st.subheader("2. Materiais")

    texto_justificado(
        """
        Os materiais compreendem registros diários
        de umidade do solo organizados em planilhas
        Excel, arquivos geoespaciais em formato
        GeoTIFF e produtos cartográficos referentes
        aos períodos bianuais. O desenvolvimento
        do aplicativo utiliza Python, Streamlit,
        Pandas, NumPy, SciPy, Statsmodels e Plotly.
        """
    )

    st.subheader("3. Métodos")

    with st.expander("Estatística descritiva"):

        texto_justificado(
            """
            A caracterização dos dados é realizada
            por meio de medidas de tendência central
            e dispersão, histogramas, boxplots,
            gráficos de violino e funções de
            distribuição acumulada.
            """
        )

    with st.expander("Séries temporais"):

        texto_justificado(
            """
            Os registros diários são organizados
            cronologicamente. Médias móveis são
            calculadas por janelas temporais
            selecionáveis, permitindo observar
            oscilações de curto e médio prazo.
            """
        )

    with st.expander("Anomalias sazonais"):

        texto_justificado(
            """
            As anomalias mensais são calculadas
            pela diferença entre a umidade média
            de cada mês e a média histórica
            correspondente ao mesmo mês do ano.
            """
        )

        st.latex(
            r"A_{m,t}=U_{m,t}-\overline{U}_m"
        )

    with st.expander("Previsão estatística"):

        texto_justificado(
            """
            A previsão exploratória utiliza
            suavização exponencial de Holt-Winters
            aplicada à série mensal. O modelo
            considera tendência e sazonalidade
            anual, produzindo valores extrapolados
            para um horizonte selecionado.
            Esses resultados não representam
            previsão meteorológica operacional.
            """
        )

    st.subheader("4. Considerações")

    texto_justificado(
        """
        Os registros diários representam a
        variável de umidade disponível na base
        fornecida. Sua interpretação depende
        da origem, resolução espacial e método
        de extração dos dados. As previsões e
        tendências devem ser tratadas como
        análises exploratórias até que sejam
        submetidas a procedimentos específicos
        de validação.
        """
    )


# ============================================================
# ABA 3 — ESTATÍSTICA DESCRITIVA
# ============================================================

with abas[2]:

    st.header("Análise Estatística Descritiva")

    if dados is None:

        st.info("Aguardando a planilha Excel local.")

    else:

        anos = sorted(dados["ano"].unique().tolist())

        intervalo = st.slider(
            "Intervalo de anos",
            min_value=int(min(anos)),
            max_value=int(max(anos)),
            value=(int(min(anos)), int(max(anos))),
            key="anos_descritiva"
        )

        df = dados[
            dados["ano"].between(
                intervalo[0],
                intervalo[1]
            )
        ].copy()

        media = df["umidade"].mean()
        mediana = df["umidade"].median()
        desvio = df["umidade"].std()
        minimo = df["umidade"].min()
        maximo = df["umidade"].max()

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Média", formatar_numero(media, 4))
        c2.metric("Mediana", formatar_numero(mediana, 4))
        c3.metric("Desvio-padrão", formatar_numero(desvio, 4))
        c4.metric(
            "Amplitude",
            formatar_numero(maximo - minimo, 4)
        )

        st.divider()

        # HISTOGRAMA E BOXPLOT

        col1, col2 = st.columns(2)

        with col1:

            fig = px.histogram(
                df,
                x="umidade",
                nbins=45,
                title="Histograma da umidade",
                color_discrete_sequence=[TERRA]
            )

            fig.update_layout(
                xaxis_title=UNIDADE,
                yaxis_title="Frequência"
            )

            exibir_grafico(fig)

        with col2:

            fig = px.box(
                df,
                x="periodo",
                y="umidade",
                title="Boxplots por biênio",
                color_discrete_sequence=[VERDE]
            )

            fig.update_layout(
                xaxis_title="Período",
                yaxis_title=UNIDADE
            )

            exibir_grafico(fig)

        # VIOLINO E DENSIDADE

        col3, col4 = st.columns(2)

        with col3:

            fig = px.violin(
                df,
                x="mes_nome",
                y="umidade",
                box=True,
                title="Distribuição mensal",
                category_orders={"mes_nome": MESES},
                color_discrete_sequence=[AZUL]
            )

            fig.update_layout(
                xaxis_title="Mês",
                yaxis_title=UNIDADE
            )

            exibir_grafico(fig)

        with col4:

            valores = df["umidade"].dropna().to_numpy()

            fig = go.Figure()

            if len(valores) >= 3 and np.std(valores) > 0:

                eixo = np.linspace(
                    valores.min(),
                    valores.max(),
                    250
                )

                densidade = gaussian_kde(valores)

                fig.add_trace(
                    go.Scatter(
                        x=eixo,
                        y=densidade(eixo),
                        mode="lines",
                        fill="tozeroy",
                        line=dict(color=MARROM, width=3),
                        fillcolor="rgba(166,107,69,0.25)"
                    )
                )

            fig.update_layout(
                title="Densidade da umidade",
                xaxis_title=UNIDADE,
                yaxis_title="Densidade",
                showlegend=False
            )

            exibir_grafico(fig)

        # MÉDIA MENSAL E DISTRIBUIÇÃO ACUMULADA

        col5, col6 = st.columns(2)

        with col5:

            mensal = (
                df.groupby("mes", as_index=False)
                .agg(
                    media=("umidade", "mean"),
                    desvio=("umidade", "std")
                )
            )

            mensal["mes_nome"] = mensal["mes"].map(
                dict(enumerate(MESES, start=1))
            )

            fig = px.bar(
                mensal,
                x="mes_nome",
                y="media",
                error_y="desvio",
                title="Umidade média mensal",
                color_discrete_sequence=[VERDE]
            )

            fig.update_layout(
                xaxis_title="Mês",
                yaxis_title=UNIDADE
            )

            exibir_grafico(fig)

        with col6:

            valores_ord = np.sort(
                df["umidade"].dropna().to_numpy()
            )

            probabilidades = (
                np.arange(1, len(valores_ord) + 1)
                / len(valores_ord)
            )

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=valores_ord,
                    y=probabilidades,
                    mode="lines",
                    line=dict(color=AZUL, width=3)
                )
            )

            fig.update_layout(
                title="Distribuição acumulada",
                xaxis_title=UNIDADE,
                yaxis_title="Probabilidade acumulada",
                showlegend=False
            )

            exibir_grafico(fig)

        st.subheader("Resumo estatístico")

        resumo = calcular_estatisticas(df)

        st.dataframe(
            resumo,
            use_container_width=True,
            hide_index=True
        )

        baixar_csv(
            resumo,
            "descritiva_umidade_cariri.csv"
        )


# ============================================================
# ABA 4 — SÉRIES TEMPORAIS
# ============================================================

with abas[3]:

    st.header("Análise de Séries Temporais")

    if dados is None:

        st.info("Aguardando os registros diários.")

    else:

        tipo_serie = st.radio(
            "Selecione a análise",
            [
                "Série bruta",
                "Médias móveis",
                "Anomalias",
                "Previsão"
            ],
            horizontal=True,
            key="tipo_serie"
        )

        serie_diaria = criar_serie_diaria(dados)
        serie_mensal = criar_serie_mensal(dados)

        # SÉRIE BRUTA

        if tipo_serie == "Série bruta":

            st.subheader("Série temporal diária original")

            fig = go.Figure()

            fig.add_trace(
                go.Scattergl(
                    x=serie_diaria.index,
                    y=serie_diaria.values,
                    mode="lines",
                    line=dict(color=MARROM, width=1.2),
                    name="Umidade diária"
                )
            )

            fig.update_layout(
                title="Umidade diária do solo",
                xaxis_title="Data",
                yaxis_title=UNIDADE
            )

            exibir_grafico(fig)

            col1, col2 = st.columns(2)

            with col1:

                anual = (
                    dados.groupby("ano", as_index=False)
                    ["umidade"].mean()
                )

                fig2 = px.line(
                    anual,
                    x="ano",
                    y="umidade",
                    markers=True,
                    title="Médias anuais",
                    color_discrete_sequence=[AZUL]
                )

                fig2.update_layout(
                    xaxis_title="Ano",
                    yaxis_title=UNIDADE
                )

                exibir_grafico(fig2)

            with col2:

                bi_anual = calcular_estatisticas(dados)

                fig3 = px.bar(
                    bi_anual,
                    x="periodo",
                    y="media",
                    title="Médias bianuais",
                    color_discrete_sequence=[TERRA]
                )

                fig3.update_layout(
                    xaxis_title="Biênio",
                    yaxis_title=UNIDADE
                )

                exibir_grafico(fig3)

        # MÉDIAS MÓVEIS

        elif tipo_serie == "Médias móveis":

            st.subheader("Médias móveis da umidade")

            janelas = st.multiselect(
                "Janelas de médias móveis (dias)",
                [7, 15, 30, 60, 90, 180, 365],
                default=[30, 90, 365]
            )

            diaria_regular = serie_diaria.asfreq("D")

            fig = go.Figure()

            fig.add_trace(
                go.Scattergl(
                    x=diaria_regular.index,
                    y=diaria_regular.values,
                    mode="lines",
                    name="Série original",
                    line=dict(color="#CFC4B5", width=1)
                )
            )

            paleta_janelas = [
                TERRA, VERDE, AZUL, MARROM,
                "#AB884B", "#9B5F67", "#507E71"
            ]

            for indice, janela in enumerate(janelas):

                media_movel = diaria_regular.rolling(
                    window=janela,
                    min_periods=max(1, janela // 2)
                ).mean()

                fig.add_trace(
                    go.Scatter(
                        x=media_movel.index,
                        y=media_movel.values,
                        mode="lines",
                        name=f"MM {janela} dias",
                        line=dict(
                            color=paleta_janelas[indice],
                            width=2.5
                        )
                    )
                )

            fig.update_layout(
                title="Médias móveis da umidade",
                xaxis_title="Data",
                yaxis_title=UNIDADE,
                legend_orientation="h"
            )

            exibir_grafico(fig)

            col1, col2 = st.columns(2)

            with col1:

                media30 = diaria_regular.rolling(
                    30, min_periods=15
                ).mean()

                fig30 = px.line(
                    x=media30.index,
                    y=media30.values,
                    title="Média móvel de 30 dias"
                )

                fig30.update_traces(line_color=TERRA)

                fig30.update_layout(
                    xaxis_title="Data",
                    yaxis_title=UNIDADE
                )

                exibir_grafico(fig30)

            with col2:

                media365 = diaria_regular.rolling(
                    365, min_periods=183
                ).mean()

                fig365 = px.line(
                    x=media365.index,
                    y=media365.values,
                    title="Média móvel de 365 dias"
                )

                fig365.update_traces(line_color=AZUL)

                fig365.update_layout(
                    xaxis_title="Data",
                    yaxis_title=UNIDADE
                )

                exibir_grafico(fig365)

        # ANOMALIAS

        elif tipo_serie == "Anomalias":

            anomalias = calcular_anomalias(dados)

            fig = px.bar(
                anomalias,
                x="data",
                y="anomalia",
                color="anomalia",
                title="Anomalias mensais da umidade",
                color_continuous_scale=[
                    TERRA, "#F6F2EA", AZUL
                ],
                color_continuous_midpoint=0
            )

            fig.add_hline(y=0, line_color=MARROM)

            fig.update_layout(
                coloraxis_showscale=False,
                xaxis_title="Data",
                yaxis_title=f"Anomalia ({UNIDADE})"
            )

            exibir_grafico(fig)

            col1, col2 = st.columns(2)

            with col1:

                anomalia_anual = (
                    anomalias.groupby(
                        "ano", as_index=False
                    )["anomalia"].mean()
                )

                fig2 = px.bar(
                    anomalia_anual,
                    x="ano",
                    y="anomalia",
                    color="anomalia",
                    title="Anomalias médias anuais",
                    color_continuous_scale=[
                        TERRA, "#F6F2EA", AZUL
                    ],
                    color_continuous_midpoint=0
                )

                fig2.add_hline(y=0, line_color=MARROM)

                fig2.update_layout(
                    coloraxis_showscale=False
                )

                exibir_grafico(fig2)

            with col2:

                matriz = anomalias.pivot(
                    index="ano",
                    columns="mes",
                    values="anomalia"
                ).reindex(columns=range(1, 13))

                fig3 = px.imshow(
                    matriz,
                    aspect="auto",
                    color_continuous_scale=[
                        TERRA, "#F6F2EA", AZUL
                    ],
                    color_continuous_midpoint=0,
                    title="Anomalias por ano e mês"
                )

                fig3.update_layout(
                    xaxis_title="Mês",
                    yaxis_title="Ano"
                )

                exibir_grafico(fig3)

        # PREVISÃO

        elif tipo_serie == "Previsão":

            st.subheader("Previsão estatística exploratória")

            horizonte = st.select_slider(
                "Horizonte da previsão",
                options=[3, 6, 12, 18, 24],
                value=12,
                format_func=lambda x: f"{x} meses"
            )

            try:

                previsao, ajustado = calcular_previsao(
                    serie_mensal, horizonte
                )

                fig = go.Figure()

                fig.add_trace(
                    go.Scatter(
                        x=serie_mensal.index,
                        y=serie_mensal.values,
                        mode="lines",
                        name="Observado",
                        line=dict(color=MARROM, width=2)
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=previsao.index,
                        y=previsao.values,
                        mode="lines+markers",
                        name="Previsto",
                        line=dict(
                            color=AZUL,
                            width=3,
                            dash="dash"
                        )
                    )
                )

                fig.update_layout(
                    title="Previsão mensal da umidade",
                    xaxis_title="Data",
                    yaxis_title=UNIDADE
                )

                exibir_grafico(fig)

                col1, col2 = st.columns(2)

                with col1:

                    historico_recente = serie_mensal.tail(36)

                    fig2 = go.Figure()

                    fig2.add_trace(
                        go.Scatter(
                            x=historico_recente.index,
                            y=historico_recente.values,
                            mode="lines+markers",
                            name="Observado",
                            line=dict(color=MARROM, width=2)
                        )
                    )

                    fig2.add_trace(
                        go.Scatter(
                            x=previsao.index,
                            y=previsao.values,
                            mode="lines+markers",
                            name="Previsto",
                            line=dict(
                                color=AZUL,
                                width=3,
                                dash="dash"
                            )
                        )
                    )

                    fig2.update_layout(
                        title="Últimos 36 meses e projeção",
                        xaxis_title="Data",
                        yaxis_title=UNIDADE
                    )

                    exibir_grafico(fig2)

                with col2:

                    fig3 = px.bar(
                        x=previsao.index,
                        y=previsao.values,
                        title="Valores mensais previstos",
                        color_discrete_sequence=[VERDE]
                    )

                    fig3.update_layout(
                        xaxis_title="Data",
                        yaxis_title=UNIDADE
                    )

                    exibir_grafico(fig3)

                tabela_previsao = pd.DataFrame({
                    "Data": previsao.index,
                    "Umidade_prevista": previsao.values
                })

                st.dataframe(
                    tabela_previsao,
                    use_container_width=True,
                    hide_index=True
                )

                baixar_csv(
                    tabela_previsao,
                    "previsao_umidade_cariri.csv"
                )

                st.caption(
                    "Projeção exploratória por Holt-Winters, "
                    "sem validação preditiva externa."
                )

            except Exception as erro:

                st.warning(
                    f"Não foi possível calcular "
                    f"a previsão: {erro}"
                )


# ============================================================
# ABA 5 — ANOMALIAS
# ============================================================

with abas[4]:

    st.header("Análise das Anomalias")

    if dados is None:

        st.info("Aguardando base de dados.")

    else:

        anomalias = calcular_anomalias(dados)

        col1, col2 = st.columns(2)

        with col1:

            fig = px.bar(
                anomalias,
                x="data",
                y="anomalia",
                title="Anomalias mensais",
                color="anomalia",
                color_continuous_scale=[
                    TERRA, "#F6F2EA", AZUL
                ],
                color_continuous_midpoint=0
            )

            fig.add_hline(y=0, line_color=MARROM)

            fig.update_layout(
                coloraxis_showscale=False
            )

            exibir_grafico(fig)

        with col2:

            matriz = anomalias.pivot(
                index="ano",
                columns="mes",
                values="anomalia"
            ).reindex(columns=range(1, 13))

            fig = px.imshow(
                matriz,
                aspect="auto",
                color_continuous_scale=[
                    TERRA, "#F6F2EA", AZUL
                ],
                color_continuous_midpoint=0,
                title="Anomalias sazonais"
            )

            fig.update_layout(
                xaxis_title="Mês",
                yaxis_title="Ano"
            )

            exibir_grafico(fig)

        col3, col4 = st.columns(2)

        with col3:

            climatologia = (
                anomalias.groupby(
                    "mes", as_index=False
                )["umidade"].mean()
            )

            fig = px.line(
                climatologia,
                x="mes",
                y="umidade",
                markers=True,
                title="Climatologia mensal",
                color_discrete_sequence=[VERDE]
            )

            fig.update_layout(
                xaxis_title="Mês",
                yaxis_title=UNIDADE
            )

            exibir_grafico(fig)

        with col4:

            anomalia_anual = (
                anomalias.groupby(
                    "ano", as_index=False
                )["anomalia"].mean()
            )

            fig = px.bar(
                anomalia_anual,
                x="ano",
                y="anomalia",
                color="anomalia",
                title="Anomalias anuais",
                color_continuous_scale=[
                    TERRA, "#F6F2EA", AZUL
                ],
                color_continuous_midpoint=0
            )

            fig.add_hline(y=0, line_color=MARROM)

            fig.update_layout(
                coloraxis_showscale=False
            )

            exibir_grafico(fig)

        baixar_csv(
            anomalias,
            "anomalias_umidade_cariri.csv"
        )


# ============================================================
# ABA 6 — MAPAS
# ============================================================

with abas[5]:

    st.header("Atlas Cartográfico do Cariri")

    tela = st.session_state.mapa_tela
    categoria = st.session_state.mapa_categoria

    # MENU PRINCIPAL

    if tela == "inicio":

        texto_justificado(
            """
            A visualização cartográfica reúne
            os mapas individuais e os painéis
            de comparação temporal da umidade
            do solo no Cariri, permitindo explorar
            diferentes períodos da série histórica.
            """
        )

        col1, col2 = st.columns(2)

        with col1:

            with st.container(border=True):

                st.subheader("🗺️ Mapas individuais")

                st.write(
                    "Visualização dos períodos bianuais."
                )

                st.metric(
                    "Mapas disponíveis",
                    len(mapas_individuais)
                )

                st.button(
                    "Visualizar mapas →",
                    use_container_width=True,
                    on_click=abrir_categoria,
                    args=("individuais",),
                    key="abrir_individuais"
                )

        with col2:

            with st.container(border=True):

                st.subheader("📅 Recortes temporais")

                st.write(
                    "Comparações cartográficas "
                    "entre períodos selecionados."
                )

                st.metric(
                    "Painéis disponíveis",
                    len(mapas_recortes)
                )

                st.button(
                    "Visualizar recortes →",
                    use_container_width=True,
                    on_click=abrir_categoria,
                    args=("recortes",),
                    key="abrir_recortes"
                )

    # GALERIA

    elif tela == "galeria":

        st.button(
            "← Voltar ao menu",
            key="voltar_menu_galeria",
            on_click=voltar_inicio
        )

        st.divider()

        if categoria == "individuais":

            st.subheader("Mapas individuais")

            galeria_mapas(
                mapas_individuais,
                "individuais"
            )

        elif categoria == "recortes":

            st.subheader("Recortes temporais")

            galeria_mapas(
                mapas_recortes,
                "recortes"
            )

    # MAPA AMPLIADO

    elif tela == "ampliado":

        c1, c2 = st.columns(2)

        with c1:

            st.button(
                "← Voltar à galeria",
                use_container_width=True,
                on_click=voltar_galeria,
                key="voltar_galeria"
            )

        with c2:

            st.button(
                "⌂ Menu de mapas",
                use_container_width=True,
                on_click=voltar_inicio,
                key="voltar_menu"
            )

        caminho = st.session_state.mapa_atual

        arquivo = Path(caminho) if caminho else None

        if arquivo and arquivo.exists():

            st.subheader(nome_mapa(arquivo))

            st.image(
                str(arquivo),
                use_container_width=True
            )

            lista = (
                mapas_individuais
                if categoria == "individuais"
                else mapas_recortes
            )

            caminhos = [str(p) for p in lista]

            if str(arquivo) in caminhos:

                posicao = caminhos.index(str(arquivo))

                c_ant, c_info, c_prox = st.columns(
                    [1, 2, 1]
                )

                with c_ant:

                    if posicao > 0:

                        st.button(
                            "← Anterior",
                            key="mapa_anterior",
                            use_container_width=True,
                            on_click=abrir_mapa,
                            args=(
                                lista[posicao - 1],
                                categoria
                            )
                        )

                with c_info:

                    st.markdown(
                        f"<p style='text-align:center;"
                        f"color:#30251E'>"
                        f"{posicao + 1} de {len(lista)}"
                        f"</p>",
                        unsafe_allow_html=True
                    )

                with c_prox:

                    if posicao < len(lista) - 1:

                        st.button(
                            "Próximo →",
                            key="mapa_proximo",
                            use_container_width=True,
                            on_click=abrir_mapa,
                            args=(
                                lista[posicao + 1],
                                categoria
                            )
                        )

        else:

            st.warning("Mapa não encontrado.")


# ============================================================
# ABA 7 — CONCLUSÃO CIENTÍFICA DINÂMICA
# ============================================================

with abas[6]:

    st.header("Conclusão")

    if dados is None:

        st.info(
            "A conclusão será apresentada "
            "quando os dados estiverem disponíveis."
        )

    else:

        resumo = (
            dados.groupby("periodo", as_index=False)
            .agg(
                media=("umidade", "mean"),
                minimo=("umidade", "min"),
                maximo=("umidade", "max"),
                desvio=("umidade", "std"),
                observacoes=("umidade", "count")
            )
            .sort_values("periodo")
        )

        media_historica = dados["umidade"].mean()

        maior = resumo.loc[
            resumo["media"].idxmax()
        ]

        menor = resumo.loc[
            resumo["media"].idxmin()
        ]

        variacao = (
            maior["media"] - menor["media"]
        )

        # INDICADORES

        st.subheader("Síntese dos resultados")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Média histórica",
            formatar_numero(media_historica, 3)
        )

        c2.metric(
            "Maior média bianual",
            formatar_numero(maior["media"], 3),
            help=str(maior["periodo"])
        )

        c3.metric(
            "Menor média bianual",
            formatar_numero(menor["media"], 3),
            help=str(menor["periodo"])
        )

        st.divider()

        # COMPORTAMENTO TEMPORAL

        st.subheader(
            "Comportamento temporal da umidade"
        )

        texto_justificado(
            "A análise dos registros diários "
            "de umidade do solo na região do "
            "Cariri permitiu caracterizar seu "
            "comportamento ao longo da série "
            "histórica disponível. "
            "A média geral observada foi de "
            f"{media_historica:.4f} {UNIDADE}. "
            "Entre os períodos bianuais, "
            "o maior valor médio foi registrado "
            f"em {maior['periodo']}, atingindo "
            f"{maior['media']:.4f} {UNIDADE}, "
            "enquanto o menor valor médio ocorreu "
            f"em {menor['periodo']}, com "
            f"{menor['media']:.4f} {UNIDADE}. "
            "A diferença entre esses valores "
            f"foi de {variacao:.4f} {UNIDADE}."
        )

        # ESTATÍSTICA DESCRITIVA

        st.subheader(
            "Distribuição e variabilidade"
        )

        texto_justificado(
            "As análises descritivas, realizadas "
            "por meio de histogramas, boxplots, "
            "gráficos de violino e medidas "
            "estatísticas, permitem examinar "
            "a distribuição dos valores, a "
            "dispersão e as diferenças entre "
            "os períodos analisados. "
            "Essas representações contribuem "
            "para identificar características "
            "da variabilidade temporal da "
            "umidade do solo."
        )

        # SÉRIES TEMPORAIS

        st.subheader(
            "Séries temporais e médias móveis"
        )

        texto_justificado(
            "A utilização de séries temporais "
            "diárias e médias móveis possibilita "
            "a identificação de oscilações em "
            "diferentes escalas de tempo. "
            "Enquanto a série original preserva "
            "a variabilidade diária dos registros, "
            "as médias móveis facilitam a "
            "visualização das mudanças graduais "
            "ao longo do período analisado."
        )

        # ANOMALIAS

        st.subheader(
            "Anomalias da umidade do solo"
        )

        texto_justificado(
            "A análise de anomalias mensais "
            "possibilita comparar as observações "
            "com a média histórica de cada "
            "mês do ano. Esse procedimento "
            "permite identificar períodos "
            "relativamente mais úmidos ou "
            "mais secos em relação ao padrão "
            "sazonal da própria série. "
            "As anomalias representam desvios "
            "estatísticos e não constituem, "
            "isoladamente, um índice de seca."
        )

        # PREVISÃO

        st.subheader(
            "Projeção estatística"
        )

        texto_justificado(
            "A modelagem por suavização "
            "exponencial oferece uma abordagem "
            "exploratória para a projeção "
            "dos valores mensais de umidade "
            "do solo. Sua utilização permite "
            "examinar possíveis trajetórias "
            "da série, embora a interpretação "
            "preditiva dependa de validação "
            "estatística e da avaliação "
            "dos erros de previsão."
        )

        # CONSIDERAÇÕES FINAIS

        st.subheader("Considerações finais")

        texto_justificado(
            "A integração entre análise "
            "estatística, séries temporais e "
            "representações cartográficas "
            "constitui uma ferramenta de apoio "
            "à compreensão da dinâmica da "
            "umidade do solo no Cariri. "
            "O aplicativo desenvolvido reúne "
            "essas informações em um ambiente "
            "interativo, facilitando a consulta, "
            "a comparação dos períodos e a "
            "comunicação dos resultados. "
            "A interpretação científica "
            "definitiva deverá considerar "
            "a representatividade espacial "
            "dos dados, a qualidade da série "
            "e a validação dos modelos empregados."
        )

        # OBSERVAÇÃO SOBRE A COBERTURA DOS DADOS

        st.caption(
            "Resultados baseados nos registros "
            "disponíveis na planilha. Períodos "
            "incompletos podem afetar comparações "
            "entre médias bianuais."
        )


# ============================================================
# RODAPÉ
# ============================================================

st.markdown(
    '<div class="rodape">'
    'ATLAS DA UMIDADE DO SOLO — CARIRI'
    '<br>'
    'Denilson • Jader • Silvio • Débora'
    '</div>',
    unsafe_allow_html=True
)
