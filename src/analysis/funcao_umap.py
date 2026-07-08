"""Visualização 2D (UMAP) do espaço latente, colorida pelo índice temporal."""

from math import ceil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import umap


def plot_umap_latente(
    df_latente_treino,
    df_latente_teste=None,
    title=None,
    save_path=None,
    n_neighbors=15,
    min_dist=0.1,
    random_state=42,
):
    """
    Reduz o espaço latente para 2D via UMAP e plota colorido por índice sequencial.

    A cor representa a posição da amostra na sequência (índice do DataFrame),
    usado como proxy de tempo — NÃO classifica normal/anomalia.
    Treino: círculos. Teste: triângulos. Colormap contínuo plasma.

    Args:
        df_latente_treino: DataFrame com colunas latent_* (saída de processar_autoencoder)
        df_latente_teste:  DataFrame opcional de teste com a mesma estrutura
        title:             Título do gráfico (opcional)
        save_path:         Caminho para salvar a imagem (opcional; exibe se None)
        n_neighbors:       Parâmetro UMAP n_neighbors
        min_dist:          Parâmetro UMAP min_dist
        random_state:      Semente para reprodutibilidade
    """
    latent_cols = [c for c in df_latente_treino.columns if c.startswith("latent_")]
    if not latent_cols:
        raise ValueError("Nenhuma coluna latent_* encontrada no DataFrame.")

    # Usa apenas os dados de teste; índice sequencial como proxy de tempo
    df_plot = df_latente_teste[latent_cols].copy().reset_index(drop=True) if df_latente_teste is not None else df_latente_treino[latent_cols].copy().reset_index(drop=True)
    df_plot["_time_idx"] = np.arange(len(df_plot))

    X = df_plot[latent_cols].values.astype(np.float32)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    embedding = reducer.fit_transform(X)

    df_plot["umap_1"] = embedding[:, 0]
    df_plot["umap_2"] = embedding[:, 1]

    # Normaliza índice para [0, 1] para o colormap
    time_norm = df_plot["_time_idx"] / (len(df_plot) - 1)

    fig, ax = plt.subplots(figsize=(10, 7))

    sc = ax.scatter(
        df_plot["umap_1"],
        df_plot["umap_2"],
        c=time_norm,
        cmap="plasma",
        marker="o",
        s=90,
        alpha=0.9,
        linewidths=0,
        vmin=0,
        vmax=1,
    )

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Índice Temporal (normalizado)", fontsize=10)

    ax.set_xlabel("UMAP 1", fontsize=11)
    ax.set_ylabel("UMAP 2", fontsize=11)
    ax.set_title(title or "Espaço Latente — UMAP 2D", fontsize=13, pad=14)
    ax.grid(True, color="#dddddd", linewidth=0.6)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"UMAP salvo em: {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_umap_latente_por_unidade(
    latentes_por_unidade,
    title=None,
    save_path=None,
    n_neighbors=15,
    min_dist=0.1,
    random_state=42,
    n_cols=3,
):
    """Plota, num grid, o UMAP 2D do espaço latente de cada unidade, colorido por tempo.

    Cada subplot é uma unidade; a cor é o índice sequencial (proxy de tempo), como
    em ``plot_umap_latente``. O UMAP é ajustado por unidade — a leitura é qualitativa
    (a forma/coerência da trajetória de amaciamento), não a posição absoluta entre
    subplots.

    Args:
        latentes_por_unidade: dict {unit_id: DataFrame com colunas ``latent_*``}.
        title, save_path, n_neighbors, min_dist, random_state: como em ``plot_umap_latente``.
        n_cols: número de colunas do grid.
    """
    unidades = list(latentes_por_unidade.keys())
    n = len(unidades)
    if n == 0:
        raise ValueError("Nenhuma unidade fornecida.")

    n_cols = min(n_cols, n)
    n_rows = ceil(n / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 4.2 * n_rows), squeeze=False
    )

    sc = None
    for idx, unit in enumerate(unidades):
        ax = axes[idx // n_cols][idx % n_cols]
        df = latentes_por_unidade[unit]
        latent_cols = [c for c in df.columns if c.startswith("latent_")]
        if not latent_cols:
            raise ValueError(f"Unidade {unit}: nenhuma coluna latent_* encontrada.")

        X = df[latent_cols].values.astype(np.float32)
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(n_neighbors, max(2, len(X) - 1)),
            min_dist=min_dist,
            random_state=random_state,
        )
        embedding = reducer.fit_transform(X)
        tempo = np.arange(len(X)) / max(1, len(X) - 1)

        sc = ax.scatter(
            embedding[:, 0], embedding[:, 1], c=tempo, cmap="plasma",
            s=40, alpha=0.9, vmin=0, vmax=1, linewidths=0,
        )
        ax.set_title(f"Unidade {unit} (n={len(X)})", fontsize=11)
        ax.set_xlabel("UMAP 1", fontsize=10)
        ax.set_ylabel("UMAP 2", fontsize=10)
        ax.grid(True, color="#dddddd", linewidth=0.6)

    for j in range(n, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")

    if sc is not None:
        cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), pad=0.02)
        cbar.set_label("Índice temporal (normalizado)", fontsize=10)

    fig.suptitle(title or "Espaço latente por unidade — UMAP 2D (cor = tempo)", fontsize=13)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"UMAP por unidade salvo em: {save_path}")
    else:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[2]))

    from src.models.autoencoder import processar_autoencoder
    from paths import DATASETS_GER, PLOTS_DIR

    df_treino = __import__("pandas").read_csv(DATASETS_GER / "dataset_balanceado_pronto.csv")
    df_teste  = __import__("pandas").read_csv(DATASETS_GER / "dataset_para_teste_latente.csv")

    params = {"input_dim": 8, "hidden_dim": 64, "latent_dim": 4}

    _, df_lat_treino, df_lat_teste, _, _ = processar_autoencoder(
        df_original=df_treino,
        params_autoencoder=params,
        learning_rate=0.02,
        epochs=200,
        batch_size=32,
        train_size=0.75,
        df_latente_input=df_teste,
        return_both_latent=True,
        return_losses=True,
    )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_umap_latente(
        df_latente_treino=df_lat_treino,
        df_latente_teste=df_lat_teste,
        title="Espaço Latente — hidden=64, latent=4, epochs=200",
        save_path=PLOTS_DIR / "umap_latente.png",
        min_dist=0.01,
    )
