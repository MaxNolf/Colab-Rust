"""
analise_carteira_wrapper.py
===========================
Wrapper Python para o módulo Rust `analise_carteira`.
Fornece interface ergonômica com pandas DataFrames,
idêntica à API de `analise_coint.walk_forward_coint()`.

Uso:
    import colab_rust_bridge as crb
    crb.pull("https://github.com/SEU_USUARIO/analise_carteira", "analise_carteira")
    
    from analise_carteira_wrapper import walk_forward_coint, WalkForwardCointConfig
    
    cfg = WalkForwardCointConfig()
    detalhado, resumo, meta = walk_forward_coint(dados, cfg)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

# O módulo Rust — importado após compilação via colab_rust_bridge
import analise_carteira as _rust


# ═══════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO
# ═══════════════════════════════════════════════════════════════════

@dataclass
class WalkForwardCointConfig:
    """Configuração do Walk-Forward de Cointegração (mesmos defaults do Python original)."""
    janela_treino_dias: int = 504
    janela_teste_dias: int = 63        # para referência (não usado no engine)
    embargo_dias: int = 5
    passo_dias: int = 63
    min_ativos: int = 2
    max_ativos: int = 7
    min_corr: float = 0.90
    top_n_corr: int = 16
    n_sub_janelas: int = 5
    min_aprovacao: float = 0.75
    correcao_multipla: str = 'bh'
    alpha: float = 0.05
    det_order: int = 0
    k_ar_diff: int = 1


# ═══════════════════════════════════════════════════════════════════
# WALK-FORWARD DE COINTEGRAÇÃO
# ═══════════════════════════════════════════════════════════════════

def walk_forward_coint(
    dados: pd.DataFrame,
    cfg: Optional[WalkForwardCointConfig] = None,
) -> tuple:
    """
    Walk-Forward de avaliação de cointegração (via Rust).
    
    Interface idêntica à versão Python original:
    
    Parâmetros:
      dados: pd.DataFrame com séries temporais (cada coluna = um ativo).
             O index deve ser datetime ou numérico.
      cfg:   WalkForwardCointConfig com os parâmetros do pipeline.
    
    Retorna:
      (df_detalhado, df_resumo, meta)
      
      df_detalhado: uma linha por par por fold (todas as métricas)
      df_resumo: agregado por par ao longo dos folds
      meta: dict com informações adicionais (funil, n_folds, etc.)
    """
    if cfg is None:
        cfg = WalkForwardCointConfig()

    # Validar dados
    if not isinstance(dados, pd.DataFrame):
        raise TypeError("dados deve ser um pd.DataFrame")
    
    if dados.shape[0] < cfg.janela_treino_dias + cfg.embargo_dias + cfg.passo_dias:
        raise ValueError(
            f"Dados insuficientes: {dados.shape[0]} linhas, "
            f"mínimo {cfg.janela_treino_dias + cfg.embargo_dias + cfg.passo_dias} para 1 fold"
        )

    # Remover colunas com NaN excessivo
    threshold = 0.5
    valid_cols = dados.columns[dados.isnull().mean() < threshold]
    dados_limpo = dados[valid_cols].ffill().bfill().dropna(axis=1)
    
    if dados_limpo.shape[1] < cfg.min_ativos:
        print(f"Aviso: apenas {dados_limpo.shape[1]} colunas válidas (mín: {cfg.min_ativos})")
        return pd.DataFrame(), pd.DataFrame(), {'funil': pd.DataFrame(), 'n_folds': 0}

    # Converter para formato Rust (flat row-major f64)
    col_names = dados_limpo.columns.tolist()
    data_np = dados_limpo.values.astype(np.float64)
    data_flat = data_np.ravel().tolist()
    n_rows = data_np.shape[0]

    print(f"Dados: {n_rows} observações × {len(col_names)} ativos")
    print(f"Configuração: janela={cfg.janela_treino_dias}, passo={cfg.passo_dias}, "
          f"embargo={cfg.embargo_dias}")
    print(f"Ativos: {cfg.min_ativos}-{cfg.max_ativos}, "
          f"corr≥{cfg.min_corr}, top_n={cfg.top_n_corr}")
    print(f"Correção: {cfg.correcao_multipla.upper()}, α={cfg.alpha}")
    print(f"Compilado em Rust — execução paralela via rayon\n")

    # Chamar o engine Rust
    result = _rust.walk_forward_coint(
        data_flat=data_flat,
        n_rows=n_rows,
        col_names=[str(c) for c in col_names],
        janela_treino_dias=cfg.janela_treino_dias,
        passo_dias=cfg.passo_dias,
        embargo_dias=cfg.embargo_dias,
        min_ativos=cfg.min_ativos,
        max_ativos=cfg.max_ativos,
        min_corr=cfg.min_corr,
        top_n_corr=cfg.top_n_corr,
        n_sub_janelas=cfg.n_sub_janelas,
        min_aprovacao=cfg.min_aprovacao,
        correcao_multipla=cfg.correcao_multipla,
        alpha=cfg.alpha,
        det_order=cfg.det_order,
        k_ar_diff=cfg.k_ar_diff,
    )

    resultados_raw = result['resultados']
    funil_raw = result['funil']
    n_folds = result['n_folds']

    # ── Construir df_detalhado ──
    if not resultados_raw:
        print("\nNenhum par encontrado em nenhum fold.")
        return pd.DataFrame(), pd.DataFrame(), {
            'funil': pd.DataFrame(funil_raw),
            'n_folds': n_folds,
        }

    # Converter ativos de lista para tupla (para consistência com versão Python)
    for r in resultados_raw:
        r['ativos'] = tuple(r['ativos'])
        r['vetor_coint'] = tuple(r['vetor_coint'])

    df_detalhado = pd.DataFrame(resultados_raw)

    # ── Construir df_resumo (agregação por par) ──
    df_resumo = (
        df_detalhado
        .groupby('ativos', sort=False)
        .agg(
            n_folds_apareceu=('n_ativos', 'count'),
            n_ativos=('n_ativos', 'first'),
            # Half-life
            half_life_media=('half_life', 'mean'),
            half_life_std=('half_life', 'std'),
            half_life_min=('half_life', 'min'),
            half_life_max=('half_life', 'max'),
            # ADF
            adf_pvalor_media=('adf_pvalor', 'mean'),
            adf_pvalor_max=('adf_pvalor', 'max'),
            # Hurst
            hurst_media=('hurst', 'mean'),
            hurst_std=('hurst', 'std'),
            hurst_max=('hurst', 'max'),
            # Variance Ratio
            vr_media=('variance_ratio', 'mean'),
            vr_std=('variance_ratio', 'std'),
            # Estabilidade
            estabilidade_media=('estabilidade', 'mean'),
            estabilidade_min=('estabilidade', 'min'),
            # Johansen
            score_johansen_media=('score_johansen', 'mean'),
            score_johansen_min=('score_johansen', 'min'),
        )
        .sort_values('n_folds_apareceu', ascending=False)
        .reset_index()
    )

    # Preencher NaN de std (pares com 1 fold)
    std_cols = [c for c in df_resumo.columns if c.endswith('_std')]
    df_resumo[std_cols] = df_resumo[std_cols].fillna(0)

    print(f"\n{'='*70}")
    print(f"CONSOLIDAÇÃO: {len(df_detalhado)} registros, "
          f"{len(df_resumo)} pares únicos, {n_folds} folds")

    if len(df_resumo) > 0:
        print(f"\nTop 20 pares por consistência:")
        cols_exibir = [
            'ativos', 'n_folds_apareceu', 'n_ativos',
            'half_life_media', 'adf_pvalor_media', 'hurst_media',
            'vr_media', 'estabilidade_media', 'score_johansen_media',
        ]
        cols_disp = [c for c in cols_exibir if c in df_resumo.columns]
        print(df_resumo[cols_disp].head(20).to_string(index=False))

    meta = {
        'funil': pd.DataFrame(funil_raw),
        'n_folds': n_folds,
    }

    return df_detalhado, df_resumo, meta


# ═══════════════════════════════════════════════════════════════════
# FUNÇÕES AUXILIARES EXPOSTAS
# ═══════════════════════════════════════════════════════════════════

def johansen_test(dados: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1) -> dict:
    """
    Executa o teste de Johansen em um grupo de séries.
    
    Parâmetros:
      dados: DataFrame com cada coluna sendo uma série temporal
      det_order: -1, 0 ou 1
      k_ar_diff: lags de diferenças
    
    Retorna: dict com rank, trace_stats, eigenvalues, eigenvector, score, p_value
    """
    data_np = dados.values.astype(np.float64)
    return _rust.johansen_test(
        data_flat=data_np.ravel().tolist(),
        n_rows=data_np.shape[0],
        n_cols=data_np.shape[1],
        det_order=det_order,
        k_ar_diff=k_ar_diff,
    )


def adf_test(series) -> float:
    """Retorna o p-valor do teste ADF (MacKinnon 2010)."""
    if isinstance(series, pd.Series):
        series = series.dropna().values
    return _rust.adf_test(list(np.asarray(series, dtype=np.float64)))


def spread_quality(spread) -> dict:
    """
    Calcula métricas de qualidade de um spread.
    
    Retorna: dict com half_life, adf_pvalor, hurst, variance_ratio
    """
    if isinstance(spread, pd.Series):
        spread = spread.dropna().values
    return _rust.spread_quality(list(np.asarray(spread, dtype=np.float64)))