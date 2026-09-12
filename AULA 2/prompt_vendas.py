from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf


# ============================================================
# CONFIGURAÇÕES
# ============================================================

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

EPOCHS = 100
BATCH_SIZE = 8


# ============================================================
# DATASET DE EXEMPLO
# ============================================================

DATASET: dict[str, list[Any]] = {
    "data": [
        "2024-01-01",
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
        "2024-01-05",
        "2024-01-06",
        "2024-01-07",
        "2024-01-08",
        "2024-01-09",
        "2024-01-10",
        "2024-01-11",
        "2024-01-12",
        "2024-01-13",
        "2024-01-14",
        "2024-01-15",
        "2024-01-16",
        "2024-01-17",
        "2024-01-18",
        "2024-01-19",
        "2024-01-20",
        "2024-01-21",
        "2024-01-22",
        "2024-01-23",
        "2024-01-24",
        "2024-01-25",
        "2024-01-26",
        "2024-01-27",
        "2024-01-28",
        "2024-01-29",
        "2024-01-30",
    ],
    "produto": [
        "Produto A",
        "Produto B",
        "Produto A",
        "Produto C",
        "Produto B",
        "Produto A",
        "Produto C",
        "Produto A",
        "Produto B",
        "Produto C",
        "Produto A",
        "Produto B",
        "Produto C",
        "Produto A",
        "Produto B",
        "Produto C",
        "Produto A",
        "Produto B",
        "Produto C",
        "Produto A",
        "Produto B",
        "Produto C",
        "Produto A",
        "Produto B",
        "Produto C",
        "Produto A",
        "Produto B",
        "Produto C",
        "Produto A",
        "Produto B",
    ],
    "preco": [
        100, 150, 100, 200, 150,
        100, 200, 100, 150, 200,
        100, 150, 200, 100, 150,
        200, 100, 150, 200, 100,
        150, 200, 100, 150, 200,
        100, 150, 200, 100, 150,
    ],
    "quantidade": [
        10, 8, 12, 6, 9,
        15, 5, 13, 10, 7,
        14, 11, 8, 16, 12,
        9, 18, 10, 7, 15,
        13, 8, 17, 11, 9,
        19, 14, 10, 20, 13,
    ],
    "vendas": [
        1000, 1200, 1200, 1200, 1350,
        1500, 1000, 1300, 1500, 1400,
        1400, 1650, 1600, 1600, 1800,
        1800, 1800, 1500, 1400, 1500,
        1950, 1600, 1700, 1650, 1800,
        1900, 2100, 2000, 2000, 1950,
    ],
}


# ============================================================
# FUNÇÕES DE PREPARAÇÃO DOS DADOS
# ============================================================

def carregar_dataset(dataset: dict[str, list[Any]]) -> pd.DataFrame:
    """
    Converte um dicionário Python em DataFrame.

    Args:
        dataset: Dicionário contendo os dados.

    Returns:
        DataFrame validado.

    Raises:
        ValueError: Caso os dados sejam inválidos.
    """
    if not isinstance(dataset, dict):
        raise ValueError("O dataset precisa ser um dicionário Python.")

    if not dataset:
        raise ValueError("O dataset está vazio.")

    try:
        dataframe = pd.DataFrame(dataset)
    except Exception as exc:
        raise ValueError(
            f"Não foi possível criar o DataFrame: {exc}"
        ) from exc

    if dataframe.empty:
        raise ValueError("O DataFrame está vazio.")

    return dataframe


def validar_colunas(dataframe: pd.DataFrame) -> None:
    """
    Verifica se as colunas necessárias existem.
    """
    colunas_obrigatorias = {
        "data",
        "produto",
        "preco",
        "quantidade",
        "vendas",
    }

    faltantes = colunas_obrigatorias - set(dataframe.columns)

    if faltantes:
        raise ValueError(
            "Colunas obrigatórias ausentes: "
            + ", ".join(sorted(faltantes))
        )


def preparar_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Limpa e transforma o DataFrame.

    Cria também variáveis derivadas da data:
        - ano
        - mes
        - dia
        - dia_da_semana
    """
    df = dataframe.copy()

    validar_colunas(df)

    # Conversão da coluna de data.
    df["data"] = pd.to_datetime(
        df["data"],
        errors="coerce",
    )

    # Conversão das variáveis numéricas.
    colunas_numericas = [
        "preco",
        "quantidade",
        "vendas",
    ]

    for coluna in colunas_numericas:
        df[coluna] = pd.to_numeric(
            df[coluna],
            errors="coerce",
        )

    # Remoção de registros inválidos.
    df = df.dropna(
        subset=[
            "data",
            "produto",
            "preco",
            "quantidade",
            "vendas",
        ]
    )

    if df.empty:
        raise ValueError(
            "Não existem registros válidos após a limpeza."
        )

    # Variáveis temporais.
    df["ano"] = df["data"].dt.year
    df["mes"] = df["data"].dt.month
    df["dia"] = df["data"].dt.day
    df["dia_da_semana"] = df["data"].dt.dayofweek

    # Ordenação temporal.
    df = df.sort_values("data").reset_index(drop=True)

    return df


# ============================================================
# ANÁLISE BÁSICA
# ============================================================

def analisar_dataset(dataframe: pd.DataFrame) -> str:
    """
    Gera um relatório textual de análise básica.
    """
    linhas, colunas = dataframe.shape

    vendas_totais = dataframe["vendas"].sum()
    venda_media = dataframe["vendas"].mean()
    venda_minima = dataframe["vendas"].min()
    venda_maxima = dataframe["vendas"].max()

    quantidade_total = dataframe["quantidade"].sum()

    produto_mais_vendido = (
        dataframe.groupby("produto")["quantidade"]
        .sum()
        .sort_values(ascending=False)
        .index[0]
    )

    vendas_por_produto = (
        dataframe.groupby("produto")["vendas"]
        .sum()
        .sort_values(ascending=False)
    )

    relatorio = []

    relatorio.append("=" * 60)
    relatorio.append("ANÁLISE BÁSICA DO DATASET")
    relatorio.append("=" * 60)

    relatorio.append(f"Linhas: {linhas}")
    relatorio.append(f"Colunas: {colunas}")

    relatorio.append("")
    relatorio.append("Colunas:")
    relatorio.append(", ".join(dataframe.columns))

    relatorio.append("")
    relatorio.append("Período:")
    relatorio.append(
        f"{dataframe['data'].min().strftime('%d/%m/%Y')} "
        f"até "
        f"{dataframe['data'].max().strftime('%d/%m/%Y')}"
    )

    relatorio.append("")
    relatorio.append("Indicadores de vendas:")
    relatorio.append(
        f"Faturamento total: R$ {vendas_totais:,.2f}"
    )
    relatorio.append(
        f"Venda média: R$ {venda_media:,.2f}"
    )
    relatorio.append(
        f"Menor venda: R$ {venda_minima:,.2f}"
    )
    relatorio.append(
        f"Maior venda: R$ {venda_maxima:,.2f}"
    )
    relatorio.append(
        f"Quantidade total vendida: {quantidade_total:,.0f}"
    )

    relatorio.append("")
    relatorio.append(
        f"Produto com maior quantidade vendida: "
        f"{produto_mais_vendido}"
    )

    relatorio.append("")
    relatorio.append("Faturamento por produto:")

    for produto, valor in vendas_por_produto.items():
        relatorio.append(
            f"  {produto}: R$ {valor:,.2f}"
        )

    relatorio.append("")
    relatorio.append("Valores ausentes:")
    relatorio.append(
        dataframe.isnull().sum().to_string()
    )

    relatorio.append("")
    relatorio.append("Estatísticas:")
    relatorio.append(
        dataframe[
            ["preco", "quantidade", "vendas"]
        ].describe().round(2).to_string()
    )

    return "\n".join(relatorio)


# ============================================================
# PREPARAÇÃO PARA O TENSORFLOW
# ============================================================

class SalesModel:
    """
    Modelo de Machine Learning para previsão de vendas.

    O modelo recebe:
        - preço
        - quantidade
        - ano
        - mês
        - dia
        - dia da semana

    E tenta prever:
        - vendas
    """

    def __init__(self) -> None:
        self.model: tf.keras.Model | None = None

        self.feature_min: np.ndarray | None = None
        self.feature_max: np.ndarray | None = None

        self.target_min: float = 0.0
        self.target_max: float = 1.0

        self.trained = False

    @staticmethod
    def obter_features(dataframe: pd.DataFrame) -> np.ndarray:
        """
        Extrai as características utilizadas pelo modelo.
        """
        features = dataframe[
            [
                "preco",
                "quantidade",
                "ano",
                "mes",
                "dia",
                "dia_da_semana",
            ]
        ].astype(float)

        return features.to_numpy()

    def normalizar_features(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        """
        Normalização Min-Max das features.
        """
        if self.feature_min is None:
            self.feature_min = features.min(axis=0)

        if self.feature_max is None:
            self.feature_max = features.max(axis=0)

        diferenca = self.feature_max - self.feature_min

        # Evita divisão por zero em colunas constantes.
        diferenca = np.where(
            diferenca == 0,
            1,
            diferenca,
        )

        return (features - self.feature_min) / diferenca

    def normalizar_target(
        self,
        target: np.ndarray,
    ) -> np.ndarray:
        """
        Normalização do valor que será previsto.
        """
        self.target_min = float(target.min())
        self.target_max = float(target.max())

        diferenca = self.target_max - self.target_min

        if diferenca == 0:
            diferenca = 1.0

        return (
            target - self.target_min
        ) / diferenca

    def desnormalizar_target(
        self,
        target: np.ndarray,
    ) -> np.ndarray:
        """
        Converte a previsão para o valor monetário original.
        """
        return (
            target
            * (self.target_max - self.target_min)
            + self.target_min
        )

    def construir_modelo(self) -> None:
        """
        Cria a rede neural usando TensorFlow/Keras.
        """
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(6,)),

                tf.keras.layers.Dense(
                    64,
                    activation="relu",
                ),

                tf.keras.layers.Dense(
                    32,
                    activation="relu",
                ),

                tf.keras.layers.Dense(
                    16,
                    activation="relu",
                ),

                tf.keras.layers.Dense(
                    1,
                    activation="linear",
                ),
            ]
        )

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=0.001
            ),
            loss="mse",
            metrics=["mae"],
        )

    def treinar(
        self,
        dataframe: pd.DataFrame,
        epochs: int = EPOCHS,
    ) -> dict[str, list[float]]:
        """
        Treina o modelo.

        Para datasets pequenos, mantém um conjunto mínimo de validação.
        """
        if len(dataframe) < 5:
            raise ValueError(
                "São necessários pelo menos 5 registros "
                "para treinar o modelo."
            )

        features = self.obter_features(dataframe)

        target = dataframe["vendas"].to_numpy(
            dtype=float
        )

        features_normalizadas = self.normalizar_features(
            features
        )

        target_normalizado = self.normalizar_target(
            target
        )

        # Divide os dados mantendo a ordem temporal.
        tamanho_treino = max(
            int(len(features_normalizadas) * 0.8),
            1,
        )

        if tamanho_treino >= len(features_normalizadas):
            tamanho_treino = len(features_normalizadas) - 1

        x_treino = features_normalizadas[
            :tamanho_treino
        ]

        y_treino = target_normalizado[
            :tamanho_treino
        ]

        x_validacao = features_normalizadas[
            tamanho_treino:
        ]

        y_validacao = target_normalizado[
            tamanho_treino:
        ]

        self.construir_modelo()

        if self.model is None:
            raise RuntimeError(
                "Não foi possível criar o modelo."
            )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=15,
                restore_best_weights=True,
            )
        ]

        historico = self.model.fit(
            x_treino,
            y_treino,
            validation_data=(
                x_validacao,
                y_validacao,
            ),
            epochs=epochs,
            batch_size=min(
                BATCH_SIZE,
                len(x_treino),
            ),
            verbose=0,
            callbacks=callbacks,
        )

        self.trained = True

        return {
            "loss": historico.history["loss"],
            "val_loss": historico.history[
                "val_loss"
            ],
        }

    def prever(
        self,
        preco: float,
        quantidade: float,
        data: str,
    ) -> float:
        """
        Realiza uma previsão de vendas.

        Args:
            preco: preço do produto.
            quantidade: quantidade prevista.
            data: data no formato YYYY-MM-DD.

        Returns:
            Valor estimado das vendas.
        """
        if not self.trained or self.model is None:
            raise RuntimeError(
                "O modelo ainda não foi treinado."
            )

        if self.feature_min is None:
            raise RuntimeError(
                "Os parâmetros de normalização "
                "não foram inicializados."
            )

        try:
            data_convertida = pd.to_datetime(data)
        except Exception as exc:
            raise ValueError(
                "Data inválida. Utilize o formato "
                "YYYY-MM-DD."
            ) from exc

        features = np.array(
            [
                [
                    preco,
                    quantidade,
                    data_convertida.year,
                    data_convertida.month,
                    data_convertida.day,
                    data_convertida.dayofweek,
                ]
            ],
            dtype=float,
        )

        # Usa os mesmos parâmetros de normalização
        # utilizados durante o treinamento.
        diferenca = (
            self.feature_max
            - self.feature_min
        )

        diferenca = np.where(
            diferenca == 0,
            1,
            diferenca,
        )

        features_normalizadas = (
            features - self.feature_min
        ) / diferenca

        previsao_normalizada = self.model.predict(
            features_normalizadas,
            verbose=0,
        )

        previsao = self.desnormalizar_target(
            previsao_normalizada.flatten()
        )[0]

        # Evita uma previsão monetária negativa.
        return max(float(previsao), 0.0)


# ============================================================
# INTERFACE GRÁFICA
# ============================================================

class SalesApplication:
    """
    Aplicação gráfica principal.
    """

    def __init__(
        self,
        root: tk.Tk,
        dataframe: pd.DataFrame,
    ) -> None:
        self.root = root
        self.dataframe = dataframe

        self.model = SalesModel()

        self.root.title(
            "Previsão de Vendas - TensorFlow"
        )

        self.root.geometry("1100x750")
        self.root.minsize(900, 650)

        self.criar_interface()

    def criar_interface(self) -> None:
        """
        Cria todos os componentes da interface.
        """
        titulo = ttk.Label(
            self.root,
            text="Análise e Previsão de Vendas",
            font=("Arial", 20, "bold"),
        )

        titulo.pack(
            pady=(15, 5)
        )

        subtitulo = ttk.Label(
            self.root,
            text=(
                "Python + Pandas + NumPy + TensorFlow"
            ),
            font=("Arial", 11),
        )

        subtitulo.pack(
            pady=(0, 15)
        )

        notebook = ttk.Notebook(
            self.root
        )

        notebook.pack(
            fill="both",
            expand=True,
            padx=15,
            pady=10,
        )

        self.aba_dados = ttk.Frame(notebook)
        self.aba_modelo = ttk.Frame(notebook)
        self.aba_previsao = ttk.Frame(notebook)

        notebook.add(
            self.aba_dados,
            text="Análise dos Dados",
        )

        notebook.add(
            self.aba_modelo,
            text="Modelo TensorFlow",
        )

        notebook.add(
            self.aba_previsao,
            text="Previsão",
        )

        self.criar_aba_dados()
        self.criar_aba_modelo()
        self.criar_aba_previsao()

    def criar_aba_dados(self) -> None:
        """
        Cria a aba de análise dos dados.
        """
        frame_botoes = ttk.Frame(
            self.aba_dados
        )

        frame_botoes.pack(
            fill="x",
            padx=10,
            pady=10,
        )

        botao_atualizar = ttk.Button(
            frame_botoes,
            text="Executar análise",
            command=self.executar_analise,
        )

        botao_atualizar.pack(
            side="left",
            padx=5,
        )

        botao_dataframe = ttk.Button(
            frame_botoes,
            text="Mostrar DataFrame",
            command=self.mostrar_dataframe,
        )

        botao_dataframe.pack(
            side="left",
            padx=5,
        )

        self.texto_analise = tk.Text(
            self.aba_dados,
            wrap="none",
            font=("Consolas", 10),
        )

        self.texto_analise.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=10,
        )

        self.executar_analise()

    def criar_aba_modelo(self) -> None:
        """
        Cria a aba de treinamento do modelo.
        """
        frame = ttk.Frame(
            self.aba_modelo
        )

        frame.pack(
            fill="x",
            padx=20,
            pady=20,
        )

        titulo = ttk.Label(
            frame,
            text="Treinamento da Rede Neural",
            font=("Arial", 15, "bold"),
        )

        titulo.pack(
            pady=(0, 15)
        )

        descricao = ttk.Label(
            frame,
            text=(
                "O TensorFlow utilizará preço, quantidade "
                "e informações temporais para estimar o "
                "valor das vendas."
            ),
            wraplength=800,
            justify="left",
        )

        descricao.pack(
            anchor="w",
            pady=10,
        )

        botao_treinar = ttk.Button(
            frame,
            text="Treinar modelo",
            command=self.treinar_modelo,
        )

        botao_treinar.pack(
            pady=15
        )

        self.status_modelo = ttk.Label(
            frame,
            text="Modelo não treinado.",
            font=("Arial", 11),
        )

        self.status_modelo.pack(
            pady=10
        )

        self.texto_modelo = tk.Text(
            self.aba_modelo,
            height=15,
            font=("Consolas", 10),
        )

        self.texto_modelo.pack(
            fill="both",
            expand=True,
            padx=20,
            pady=10,
        )

    def criar_aba_previsao(self) -> None:
        """
        Cria a aba para realizar novas previsões.
        """
        frame = ttk.Frame(
            self.aba_previsao
        )

        frame.pack(
            padx=30,
            pady=30,
            fill="x",
        )

        titulo = ttk.Label(
            frame,
            text="Prever vendas",
            font=("Arial", 16, "bold"),
        )

        titulo.grid(
            row=0,
            column=0,
            columnspan=2,
            pady=(0, 20),
        )

        ttk.Label(
            frame,
            text="Data (YYYY-MM-DD):",
        ).grid(
            row=1,
            column=0,
            sticky="w",
            pady=8,
        )

        self.entrada_data = ttk.Entry(
            frame,
            width=30,
        )

        self.entrada_data.insert(
            0,
            "2024-01-31",
        )

        self.entrada_data.grid(
            row=1,
            column=1,
            sticky="w",
            pady=8,
        )

        ttk.Label(
            frame,
            text="Preço:",
        ).grid(
            row=2,
            column=0,
            sticky="w",
            pady=8,
        )

        self.entrada_preco = ttk.Entry(
            frame,
            width=30,
        )

        self.entrada_preco.insert(
            0,
            "100",
        )

        self.entrada_preco.grid(
            row=2,
            column=1,
            sticky="w",
            pady=8,
        )

        ttk.Label(
            frame,
            text="Quantidade:",
        ).grid(
            row=3,
            column=0,
            sticky="w",
            pady=8,
        )

        self.entrada_quantidade = ttk.Entry(
            frame,
            width=30,
        )

        self.entrada_quantidade.insert(
            0,
            "10",
        )

        self.entrada_quantidade.grid(
            row=3,
            column=1,
            sticky="w",
            pady=8,
        )

        botao_prever = ttk.Button(
            frame,
            text="Realizar previsão",
            command=self.realizar_previsao,
        )

        botao_prever.grid(
            row=4,
            column=0,
            columnspan=2,
            pady=20,
        )

        self.resultado_previsao = ttk.Label(
            frame,
            text="Aguardando previsão...",
            font=("Arial", 16, "bold"),
        )

        self.resultado_previsao.grid(
            row=5,
            column=0,
            columnspan=2,
            pady=20,
        )

    # ========================================================
    # EVENTOS DA INTERFACE
    # ========================================================

    def executar_analise(self) -> None:
        """
        Executa a análise básica.
        """
        try:
            relatorio = analisar_dataset(
                self.dataframe
            )

            self.texto_analise.delete(
                "1.0",
                tk.END,
            )

            self.texto_analise.insert(
                tk.END,
                relatorio,
            )

        except Exception as exc:
            messagebox.showerror(
                "Erro",
                f"Erro durante a análise:\n{exc}",
            )

    def mostrar_dataframe(self) -> None:
        """
        Mostra os primeiros registros do DataFrame.
        """
        try:
            dados = self.dataframe.to_string(
                index=False
            )

            janela = tk.Toplevel(
                self.root
            )

            janela.title(
                "DataFrame"
            )

            janela.geometry(
                "1000x500"
            )

            texto = tk.Text(
                janela,
                wrap="none",
                font=("Consolas", 9),
            )

            texto.pack(
                fill="both",
                expand=True,
            )

            texto.insert(
                tk.END,
                dados,
            )

        except Exception as exc:
            messagebox.showerror(
                "Erro",
                f"Erro ao mostrar DataFrame:\n{exc}",
            )

    def treinar_modelo(self) -> None:
        """
        Treina o modelo TensorFlow.
        """
        try:
            self.status_modelo.config(
                text="Treinando modelo..."
            )

            self.root.update_idletasks()

            historico = self.model.treinar(
                self.dataframe,
                epochs=EPOCHS,
            )

            ultimo_loss = historico["loss"][-1]
            ultimo_val_loss = historico[
                "val_loss"
            ][-1]

            self.status_modelo.config(
                text="Modelo treinado com sucesso."
            )

            self.texto_modelo.delete(
                "1.0",
                tk.END,
            )

            mensagem = (
                "TREINAMENTO CONCLUÍDO\n"
                "\n"
                f"Épocas executadas: "
                f"{len(historico['loss'])}\n"
                f"Loss final: {ultimo_loss:.6f}\n"
                f"Validation Loss final: "
                f"{ultimo_val_loss:.6f}\n"
                "\n"
                "Arquitetura:\n"
                "Input: 6 características\n"
                "Dense: 64 neurônios - ReLU\n"
                "Dense: 32 neurônios - ReLU\n"
                "Dense: 16 neurônios - ReLU\n"
                "Output: 1 neurônio - Linear\n"
                "\n"
                "O modelo está pronto para realizar "
                "previsões."
            )

            self.texto_modelo.insert(
                tk.END,
                mensagem,
            )

            messagebox.showinfo(
                "Treinamento",
                "Modelo treinado com sucesso!",
            )

        except Exception as exc:
            self.status_modelo.config(
                text="Erro no treinamento."
            )

            messagebox.showerror(
                "Erro no treinamento",
                str(exc),
            )

    def realizar_previsao(self) -> None:
        """
        Realiza uma previsão usando os valores da interface.
        """
        try:
            data = self.entrada_data.get().strip()

            preco_texto = (
                self.entrada_preco.get().strip()
            )

            quantidade_texto = (
                self.entrada_quantidade.get().strip()
            )

            if not data:
                raise ValueError(
                    "Informe uma data."
                )

            preco = float(
                preco_texto.replace(",", ".")
            )

            quantidade = float(
                quantidade_texto.replace(",", ".")
            )

            if preco < 0:
                raise ValueError(
                    "O preço não pode ser negativo."
                )

            if quantidade < 0:
                raise ValueError(
                    "A quantidade não pode ser negativa."
                )

            previsao = self.model.prever(
                preco=preco,
                quantidade=quantidade,
                data=data,
            )

            self.resultado_previsao.config(
                text=(
                    f"Venda estimada: "
                    f"R$ {previsao:,.2f}"
                )
            )

        except ValueError as exc:
            messagebox.showwarning(
                "Dados inválidos",
                str(exc),
            )

        except RuntimeError as exc:
            messagebox.showwarning(
                "Modelo",
                (
                    f"{exc}\n\n"
                    "Acesse a aba 'Modelo TensorFlow' "
                    "e treine o modelo primeiro."
                ),
            )

        except Exception as exc:
            messagebox.showerror(
                "Erro",
                f"Não foi possível realizar a previsão:\n{exc}",
            )


# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================

def main() -> None:
    """
    Ponto de entrada da aplicação.
    """
    try:
        # 1. Carregar o dicionário.
        dataframe = carregar_dataset(
            DATASET
        )

        # 2. Preparar e limpar os dados.
        dataframe = preparar_dataframe(
            dataframe
        )

        # 3. Criar a aplicação.
        root = tk.Tk()

        app = SalesApplication(
            root,
            dataframe,
        )

        # Evita warning de variável não utilizada
        # em algumas ferramentas de análise.
        _ = app

        root.mainloop()

    except Exception as exc:
        # Como ainda não existe uma janela Tkinter,
        # utiliza uma mensagem simples no console.
        print(
            "Erro ao iniciar a aplicação:"
        )
        print(exc)


if __name__ == "__main__":
    main()