//! Walk-Forward Cointegration Analysis
//! ====================================
//!
//! Módulo `coint` — integra ao crate `template_rust_module` existente.
//!
//! Pipeline por fold:
//!   1. Pré-filtro correlação (vetorizado)
//!   2. Johansen → vetor, rank, pseudo p-valor
//!   3. Correção de múltiplos testes (BH / Bonferroni)
//!   4. Filtro de estabilidade (sub-janelas)
//!   5. Métricas de qualidade do spread:
//!      - Half-life (Ornstein-Uhlenbeck)
//!      - ADF p-valor (MacKinnon 2010)
//!      - Hurst exponent (R/S)
//!      - Variance Ratio
//!      - Estabilidade (% sub-janelas cointegradas)
//!      - Score Johansen

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyDict, PyList};

use nalgebra::{DMatrix, DVector, SymmetricEigen, Cholesky};
use statrs::distribution::{ContinuousCDF, Normal};
use rayon::prelude::*;

use arrow::array::{Float64Array, Float32Array, Int64Array};
use arrow::datatypes::DataType;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs::File;
use std::io::BufReader;

// ═══════════════════════════════════════════════════════════════════
// TIPOS INTERNOS
// ═══════════════════════════════════════════════════════════════════

#[derive(Clone, Debug)]
struct JohansenResult {
    rank: usize,
    trace_stats: Vec<f64>,
    eigenvalues: Vec<f64>,
    eigenvectors: DMatrix<f64>,
    crit_values: Vec<[f64; 3]>, // [90%, 95%, 99%] por rank
    p_value: f64,
    score: f64,
}

#[derive(Clone, Debug)]
struct PairResult {
    ativos: Vec<String>,
    n_ativos: usize,
    vetor_coint: Vec<f64>,
    score_johansen: f64,
    p_valor: f64,
    estabilidade: f64,
    half_life: f64,
    adf_pvalor: f64,
    hurst: f64,
    variance_ratio: f64,
}

struct WalkForwardConfig {
    janela_treino_dias: usize,
    passo_dias: usize,
    embargo_dias: usize,
    min_ativos: usize,
    max_ativos: usize,
    min_corr: f64,
    top_n_corr: usize,
    n_sub_janelas: usize,
    min_aprovacao: f64,
    correcao_multipla: String,
    alpha: f64,
    det_order: i32,
    k_ar_diff: usize,
}

// ═══════════════════════════════════════════════════════════════════
// VALORES CRÍTICOS — JOHANSEN TRACE (det_order=0, constante restrita)
// Osterwald-Lenum (1992) / MacKinnon-Haug-Michelis (1999)
// Índice = (n - r). Cada entrada: [90%, 95%, 99%]
// ═══════════════════════════════════════════════════════════════════

const TRACE_CRIT: [[f64; 3]; 13] = [
    [0.0, 0.0, 0.0],                    // 0 (não usado)
    [2.7055, 3.8415, 6.6349],           // n-r = 1
    [13.4294, 15.4943, 19.9349],        // 2
    [27.0669, 29.7961, 35.4628],        // 3
    [44.4929, 47.8545, 54.6815],        // 4
    [65.8202, 69.8189, 77.8202],        // 5
    [91.1090, 95.7542, 104.9637],       // 6
    [120.3673, 125.6185, 136.0600],     // 7
    [153.6341, 159.5290, 171.0905],     // 8
    [190.8714, 197.3772, 210.0366],     // 9
    [232.1030, 239.2468, 253.2526],     // 10
    [277.3740, 285.1402, 300.2821],     // 11
    [326.5354, 334.9795, 351.2150],     // 12
];

fn trace_critical_values(n_minus_r: usize) -> [f64; 3] {
    if n_minus_r == 0 || n_minus_r >= TRACE_CRIT.len() {
        let last = TRACE_CRIT[TRACE_CRIT.len() - 1];
        let s = n_minus_r as f64 / 12.0;
        [last[0] * s, last[1] * s, last[2] * s]
    } else {
        TRACE_CRIT[n_minus_r]
    }
}

// ═══════════════════════════════════════════════════════════════════
// ÁLGEBRA LINEAR — HELPERS
// ═══════════════════════════════════════════════════════════════════

fn diff_matrix(data: &DMatrix<f64>) -> DMatrix<f64> {
    let (t, n) = data.shape();
    DMatrix::from_fn(t - 1, n, |i, j| data[(i + 1, j)] - data[(i, j)])
}

fn demean(data: &DMatrix<f64>) -> DMatrix<f64> {
    let (t, n) = data.shape();
    let mut r = data.clone();
    for j in 0..n {
        let mu: f64 = data.column(j).iter().sum::<f64>() / t as f64;
        for i in 0..t {
            r[(i, j)] -= mu;
        }
    }
    r
}

/// OLS: retorna resíduos de Y ~ X.
fn ols_residuals(y: &DMatrix<f64>, x: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    let xtx = x.transpose() * x;
    let beta = xtx.lu().solve(&(x.transpose() * y))?;
    Some(y - x * beta)
}

/// OLS: retorna (coeficientes, resíduos) para vetor y ~ matrix X.
fn ols_fit(y: &DVector<f64>, x: &DMatrix<f64>) -> Option<(DVector<f64>, DVector<f64>)> {
    let beta = (x.transpose() * x).lu().solve(&(x.transpose() * y))?;
    let res = y - x * &beta;
    Some((beta, res))
}

fn select_columns(data: &DMatrix<f64>, cols: &[usize]) -> DMatrix<f64> {
    let t = data.nrows();
    DMatrix::from_fn(t, cols.len(), |i, j| data[(i, cols[j])])
}

// ═══════════════════════════════════════════════════════════════════
// TESTE DE JOHANSEN (TRACE)
// ═══════════════════════════════════════════════════════════════════

fn johansen_trace(
    data: &DMatrix<f64>,
    det_order: i32,
    k_ar_diff: usize,
) -> Option<JohansenResult> {
    let (t, n) = data.shape();
    if t < n + k_ar_diff + 10 {
        return None;
    }

    let dy = diff_matrix(data);
    let y_lag = data.rows(0, t - 1).clone_owned();

    // det_order >= 0 → aumentar Y_lag com coluna de 1s
    let z = if det_order >= 0 {
        let mut aug = DMatrix::zeros(t - 1, n + 1);
        aug.view_mut((0, 0), (t - 1, n)).copy_from(&y_lag);
        for i in 0..t - 1 {
            aug[(i, n)] = 1.0;
        }
        aug
    } else {
        y_lag
    };

    let m = z.ncols();
    if k_ar_diff >= t - 1 {
        return None;
    }

    let eff_t = t - 1 - k_ar_diff;
    if eff_t < m + 5 {
        return None;
    }

    let dy_dep = dy.rows(k_ar_diff, eff_t).clone_owned();
    let z_aligned = z.rows(k_ar_diff, eff_t).clone_owned();

    // Concentrar lags de diferenças
    let (r0, r1) = if k_ar_diff > 0 {
        let mut f = DMatrix::zeros(eff_t, n * k_ar_diff);
        for lag in 0..k_ar_diff {
            let start = k_ar_diff - 1 - lag;
            for row in 0..eff_t {
                for col in 0..n {
                    f[(row, lag * n + col)] = dy[(start + row, col)];
                }
            }
        }
        (ols_residuals(&dy_dep, &f)?, ols_residuals(&z_aligned, &f)?)
    } else {
        (demean(&dy_dep), demean(&z_aligned))
    };

    let tf = eff_t as f64;
    let s00 = (&r0.transpose() * &r0) / tf;
    let s01 = (&r0.transpose() * &r1) / tf;
    let s10 = (&r1.transpose() * &r0) / tf;
    let s11 = (&r1.transpose() * &r1) / tf;

    let s00_inv = s00.clone().try_inverse()?;
    let product = &s10 * &s00_inv * &s01;

    // Regularizar S11
    let s11_reg = &s11 + DMatrix::identity(m, m) * 1e-10;
    let chol = Cholesky::new(s11_reg)?;
    let l = chol.l();
    let l_inv = l.clone().try_inverse()?;
    let lt_inv = l.transpose().try_inverse()?;

    let m_sym = &l_inv * &product * &lt_inv;
    let m_sym = (&m_sym + &m_sym.transpose()) * 0.5;

    let eigen = SymmetricEigen::new(m_sym);
    let eigvecs = &lt_inv * &eigen.eigenvectors;

    // Ordenar autovalores decrescente
    let mut idx: Vec<usize> = (0..m).collect();
    idx.sort_by(|&a, &b| {
        eigen.eigenvalues[b]
            .partial_cmp(&eigen.eigenvalues[a])
            .unwrap()
    });

    let sorted_evals: Vec<f64> = idx
        .iter()
        .map(|&i| eigen.eigenvalues[i].clamp(0.0, 1.0 - 1e-15))
        .collect();

    let mut sorted_evecs = DMatrix::zeros(m, m);
    for (new_col, &old_col) in idx.iter().enumerate() {
        sorted_evecs.set_column(new_col, &eigvecs.column(old_col));
    }

    // Trace statistics & critical values
    let n_tests = n.min(sorted_evals.len());
    let mut trace_stats = Vec::with_capacity(n_tests);
    let mut crit_values = Vec::with_capacity(n_tests);

    for r in 0..n_tests {
        let trace: f64 = (r..n_tests)
            .map(|i| -tf * (1.0 - sorted_evals[i]).max(1e-15).ln())
            .sum();
        trace_stats.push(trace);
        crit_values.push(trace_critical_values(n - r));
    }

    // Rank
    let mut rank = 0;
    for r in 0..n_tests {
        if trace_stats[r] > crit_values[r][1] {
            rank += 1;
        } else {
            break;
        }
    }

    // Pseudo p-valor (interpolação, idêntico ao Python)
    let ts0 = trace_stats[0];
    let [c90, c95, c99] = crit_values[0];

    let p_val = if ts0 >= c99 {
        0.005
    } else if ts0 >= c95 {
        0.05 - 0.04 * (ts0 - c95) / (c99 - c95).max(1e-9)
    } else if ts0 >= c90 {
        0.10 - 0.05 * (ts0 - c90) / (c95 - c90).max(1e-9)
    } else {
        0.50
    };

    Some(JohansenResult {
        rank,
        trace_stats,
        eigenvalues: sorted_evals,
        eigenvectors: sorted_evecs,
        crit_values,
        p_value: p_val.clamp(0.001, 1.0),
        score: ts0 / c95.max(1e-9),
    })
}

// ═══════════════════════════════════════════════════════════════════
// TESTE ADF
// ═══════════════════════════════════════════════════════════════════

fn adf_select_lag(series: &[f64], max_lag: usize) -> usize {
    let n = series.len();
    let dy: Vec<f64> = (1..n).map(|i| series[i] - series[i - 1]).collect();
    let mut best_aic = f64::INFINITY;
    let mut best_lag = 0;

    for lag in 0..=max_lag {
        let eff = dy.len().saturating_sub(lag);
        if eff < lag + 5 {
            break;
        }
        let k = 2 + lag;
        let mut y_vec = DVector::zeros(eff);
        let mut x_mat = DMatrix::zeros(eff, k);

        for i in 0..eff {
            let ti = lag + i;
            y_vec[i] = dy[ti];
            x_mat[(i, 0)] = 1.0;
            x_mat[(i, 1)] = series[ti];
            for j in 0..lag {
                x_mat[(i, 2 + j)] = dy[ti - 1 - j];
            }
        }

        if let Some((_b, res)) = ols_fit(&y_vec, &x_mat) {
            let sse: f64 = res.iter().map(|r| r * r).sum();
            let s2 = sse / eff as f64;
            if s2 > 0.0 {
                let aic = eff as f64 * s2.ln() + 2.0 * k as f64;
                if aic < best_aic {
                    best_aic = aic;
                    best_lag = lag;
                }
            }
        }
    }
    best_lag
}

fn adf_pvalue(series: &[f64]) -> f64 {
    let n = series.len();
    if n < 20 {
        return 1.0;
    }

    let max_lag = ((12.0 * (n as f64 / 100.0).powf(0.25)) as usize).min(n / 4);
    let lag = adf_select_lag(series, max_lag);
    let dy: Vec<f64> = (1..n).map(|i| series[i] - series[i - 1]).collect();
    let eff = dy.len().saturating_sub(lag);
    if eff < lag + 5 {
        return 1.0;
    }

    let k = 2 + lag;
    let mut y_vec = DVector::zeros(eff);
    let mut x_mat = DMatrix::zeros(eff, k);

    for i in 0..eff {
        let ti = lag + i;
        y_vec[i] = dy[ti];
        x_mat[(i, 0)] = 1.0;
        x_mat[(i, 1)] = series[ti];
        for j in 0..lag {
            x_mat[(i, 2 + j)] = dy[ti - 1 - j];
        }
    }

    let (beta, res) = match ols_fit(&y_vec, &x_mat) {
        Some(v) => v,
        None => return 1.0,
    };

    let sse: f64 = res.iter().map(|r| r * r).sum();
    let sigma2 = sse / (eff - k) as f64;
    if sigma2 <= 0.0 {
        return 1.0;
    }

    let xtx_inv = match (x_mat.transpose() * &x_mat).try_inverse() {
        Some(v) => v,
        None => return 1.0,
    };

    let se = (sigma2 * xtx_inv[(1, 1)]).max(0.0).sqrt();
    if se < 1e-15 {
        return 1.0;
    }
    let t_stat = beta[1] / se;

    // MacKinnon (2010) — modelo com constante, sem tendência
    let tf = eff as f64;
    let ti = 1.0 / tf;
    let ti2 = ti * ti;

    let c1 = -3.4336 - 5.999 * ti - 29.25 * ti2;
    let c5 = -2.8621 - 2.738 * ti - 8.36 * ti2;
    let c10 = -2.5671 - 1.438 * ti - 4.48 * ti2;

    if t_stat <= c1 {
        let norm = Normal::new(0.0, 1.0).unwrap();
        (norm.cdf((t_stat - c1) * 0.3) * 0.01).clamp(0.0001, 0.01)
    } else if t_stat <= c5 {
        0.01 + 0.04 * (t_stat - c1) / (c5 - c1).max(1e-9)
    } else if t_stat <= c10 {
        0.05 + 0.05 * (t_stat - c5) / (c10 - c5).max(1e-9)
    } else {
        let norm = Normal::new(0.0, 1.0).unwrap();
        (0.10 + 0.90 * norm.cdf((t_stat - c10) * 0.5)).clamp(0.10, 1.0)
    }
}

// ═══════════════════════════════════════════════════════════════════
// MÉTRICAS DE QUALIDADE DO SPREAD
// ═══════════════════════════════════════════════════════════════════

fn half_life(s: &[f64]) -> f64 {
    let n = s.len();
    if n < 10 {
        return f64::INFINITY;
    }
    let denom: f64 = s[..n - 1].iter().map(|x| x * x).sum();
    if denom < 1e-15 {
        return f64::INFINITY;
    }
    let numer: f64 = s[..n - 1]
        .iter()
        .zip(s[1..].iter())
        .map(|(a, b)| a * (b - a))
        .sum();
    let theta = numer / denom;
    if theta >= 0.0 {
        f64::INFINITY
    } else {
        -(2.0_f64.ln()) / theta
    }
}

fn hurst_exponent(s: &[f64], max_lag: usize) -> f64 {
    let n = s.len();
    if n < 20 {
        return 0.5;
    }
    let max_lag = max_lag.min(n / 2);
    let mut log_lags = Vec::new();
    let mut log_rs = Vec::new();

    for lag in 2..=max_lag {
        let nb = n / lag;
        if nb == 0 {
            continue;
        }
        let mut rs_sum = 0.0;
        let mut rs_cnt = 0u32;

        for b in 0..nb {
            let blk = &s[b * lag..(b + 1) * lag];
            let mu: f64 = blk.iter().sum::<f64>() / lag as f64;
            let mut cum = 0.0;
            let mut mx = f64::NEG_INFINITY;
            let mut mn = f64::INFINITY;
            for &v in blk {
                cum += v - mu;
                mx = mx.max(cum);
                mn = mn.min(cum);
            }
            let var: f64 = blk.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / lag as f64;
            let sd = var.sqrt();
            if sd > 1e-12 {
                rs_sum += (mx - mn) / sd;
                rs_cnt += 1;
            }
        }

        if rs_cnt > 0 {
            let rs_mean = rs_sum / rs_cnt as f64;
            if rs_mean > 0.0 {
                log_lags.push((lag as f64).ln());
                log_rs.push(rs_mean.ln());
            }
        }
    }

    let nv = log_lags.len();
    if nv < 3 {
        return 0.5;
    }
    let sx: f64 = log_lags.iter().sum();
    let sy: f64 = log_rs.iter().sum();
    let sxy: f64 = log_lags.iter().zip(log_rs.iter()).map(|(x, y)| x * y).sum();
    let sx2: f64 = log_lags.iter().map(|x| x * x).sum();
    let d = nv as f64 * sx2 - sx * sx;
    if d.abs() < 1e-12 {
        0.5
    } else {
        ((nv as f64 * sxy - sx * sy) / d).clamp(0.0, 1.0)
    }
}

fn variance_ratio(s: &[f64], lag: usize) -> f64 {
    let n = s.len();
    if n < lag + 10 {
        return 1.0;
    }
    let ret1: Vec<f64> = (1..n).map(|i| s[i] - s[i - 1]).collect();
    let m1: f64 = ret1.iter().sum::<f64>() / ret1.len() as f64;
    let v1: f64 = ret1.iter().map(|x| (x - m1).powi(2)).sum::<f64>() / (ret1.len() - 1) as f64;
    if v1 < 1e-12 {
        return 1.0;
    }
    let retk: Vec<f64> = (lag..n).map(|i| s[i] - s[i - lag]).collect();
    let mk: f64 = retk.iter().sum::<f64>() / retk.len() as f64;
    let vk: f64 = retk.iter().map(|x| (x - mk).powi(2)).sum::<f64>() / (retk.len() - 1) as f64;
    (vk / lag as f64) / v1
}

// ═══════════════════════════════════════════════════════════════════
// PRÉ-FILTRO DE CORRELAÇÃO
// ═══════════════════════════════════════════════════════════════════

fn pre_filtro_correlacao(data: &DMatrix<f64>, min_corr: f64, top_n: usize) -> Vec<usize> {
    let (t, n) = data.shape();
    if t < 3 || n < 2 {
        return (0..n).collect();
    }

    // Retornos percentuais, centralizados, normalizados
    let tr = t - 1;
    let mut ret = DMatrix::zeros(tr, n);
    for i in 0..tr {
        for j in 0..n {
            let p = data[(i, j)];
            ret[(i, j)] = if p.abs() > 1e-15 {
                (data[(i + 1, j)] - p) / p
            } else {
                0.0
            };
        }
    }
    for j in 0..n {
        let mu: f64 = ret.column(j).iter().sum::<f64>() / tr as f64;
        for i in 0..tr {
            ret[(i, j)] -= mu;
        }
    }
    let mut norms = vec![0.0; n];
    for j in 0..n {
        norms[j] = ret.column(j).iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-15);
    }
    for j in 0..n {
        for i in 0..tr {
            ret[(i, j)] /= norms[j];
        }
    }

    let corr = ret.transpose() * &ret;
    let mut contagem = vec![0usize; n];
    for i in 0..n {
        for j in (i + 1)..n {
            if corr[(i, j)].abs() >= min_corr {
                contagem[i] += 1;
                contagem[j] += 1;
            }
        }
    }

    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| contagem[b].cmp(&contagem[a]));
    idx.truncate(top_n);
    idx.sort();
    idx
}

// ═══════════════════════════════════════════════════════════════════
// CORREÇÃO DE MÚLTIPLOS TESTES
// ═══════════════════════════════════════════════════════════════════

fn benjamini_hochberg(p_values: &[f64], alpha: f64) -> Vec<bool> {
    let m = p_values.len();
    if m == 0 {
        return vec![];
    }
    let mut si: Vec<usize> = (0..m).collect();
    si.sort_by(|&a, &b| p_values[a].partial_cmp(&p_values[b]).unwrap());

    let mut max_k: Option<usize> = None;
    for (k, &i) in si.iter().enumerate() {
        if p_values[i] <= ((k + 1) as f64 / m as f64) * alpha {
            max_k = Some(k);
        }
    }
    let mut mask = vec![false; m];
    if let Some(mk) = max_k {
        for &i in &si[..=mk] {
            mask[i] = true;
        }
    }
    mask
}

fn bonferroni(p_values: &[f64], alpha: f64) -> Vec<bool> {
    let th = alpha / p_values.len() as f64;
    p_values.iter().map(|&p| p <= th).collect()
}

// ═══════════════════════════════════════════════════════════════════
// FILTRO DE ESTABILIDADE
// ═══════════════════════════════════════════════════════════════════

fn filtro_estabilidade(
    data: &DMatrix<f64>,
    n_sub: usize,
    min_aprovacao: f64,
    det_order: i32,
    k_ar_diff: usize,
) -> (bool, f64) {
    let tam = data.nrows() / n_sub;
    if tam < 60 {
        return (false, 0.0);
    }
    let mut ok = 0;
    for i in 0..n_sub {
        let sub = data.rows(i * tam, tam).clone_owned();
        if let Some(r) = johansen_trace(&sub, det_order, k_ar_diff) {
            if !r.trace_stats.is_empty()
                && !r.crit_values.is_empty()
                && r.trace_stats[0] > r.crit_values[0][1]
            {
                ok += 1;
            }
        }
    }
    let taxa = ok as f64 / n_sub as f64;
    (taxa >= min_aprovacao, taxa)
}

// ═══════════════════════════════════════════════════════════════════
// COMBINAÇÕES
// ═══════════════════════════════════════════════════════════════════

fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    if k > n {
        return vec![];
    }
    let mut result = Vec::new();
    let mut c: Vec<usize> = (0..k).collect();
    loop {
        result.push(c.clone());
        let mut i = k;
        loop {
            if i == 0 {
                return result;
            }
            i -= 1;
            if c[i] != i + n - k {
                break;
            }
            if i == 0 && c[0] == n - k {
                return result;
            }
        }
        c[i] += 1;
        for j in (i + 1)..k {
            c[j] = c[j - 1] + 1;
        }
    }
}

fn n_combinations(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut r: usize = 1;
    for i in 0..k {
        r = r * (n - i) / (i + 1);
    }
    r
}

// ═══════════════════════════════════════════════════════════════════
// WALK-FORWARD ENGINE
// ═══════════════════════════════════════════════════════════════════

fn walk_forward_engine(
    data: &DMatrix<f64>,
    col_names: &[String],
    cfg: &WalkForwardConfig,
) -> (Vec<PairResult>, Vec<[usize; 6]>, usize) {
    let n_total = data.nrows();
    let mut all_results: Vec<PairResult> = Vec::new();
    let mut funil: Vec<[usize; 6]> = Vec::new();
    let mut n_testes_total: usize = 0;

    let mut janela_inicio = 0usize;
    let mut fold = 0usize;

    while janela_inicio + cfg.janela_treino_dias + cfg.embargo_dias + cfg.passo_dias <= n_total {
        fold += 1;
        let dados_treino = data.rows(janela_inicio, cfg.janela_treino_dias).clone_owned();
        let n_pre = dados_treino.ncols();

        eprintln!(
            "\nFOLD {fold}: linhas {janela_inicio}..{} ({} obs)",
            janela_inicio + cfg.janela_treino_dias,
            cfg.janela_treino_dias
        );

        // ── ETAPA 1: Pré-filtro correlação ──
        let cols_filt = pre_filtro_correlacao(&dados_treino, cfg.min_corr, cfg.top_n_corr);
        let nc = cols_filt.len();

        if nc < cfg.min_ativos {
            eprintln!("  Poucos ativos após correlação ({nc}). Pulando.");
            funil.push([fold, n_pre, nc, 0, 0, 0]);
            janela_inicio += cfg.passo_dias;
            continue;
        }

        let dtf = select_columns(&dados_treino, &cols_filt);
        let nomes_f: Vec<String> = cols_filt.iter().map(|&i| col_names[i].clone()).collect();
        eprintln!("  Pré-filtro: {n_pre} → {nc} ativos");

        // ── ETAPA 2: Johansen (paralelo) ──
        let mut all_combos = Vec::new();
        for k in cfg.min_ativos..=cfg.max_ativos.min(nc) {
            for combo in combinations(nc, k) {
                all_combos.push((k, combo));
            }
        }
        let total_c = all_combos.len();
        n_testes_total += total_c;
        eprintln!("  Testando {total_c} combinações...");

        let found: Vec<_> = all_combos
            .par_iter()
            .filter_map(|(k, combo)| {
                let sub = select_columns(&dtf, combo);
                let r = johansen_trace(&sub, cfg.det_order, cfg.k_ar_diff)?;
                if r.rank == 0 {
                    return None;
                }
                let nv = *k;
                let ev = r.eigenvectors.column(0);
                let mut v: Vec<f64> = (0..nv).map(|i| ev[i]).collect();
                let v0 = v[0];
                if v0.abs() > 1e-15 {
                    v.iter_mut().for_each(|x| *x /= v0);
                }
                Some((combo.clone(), nv, v, r.score, r.p_value))
            })
            .collect();

        let n_joh = found.len();
        eprintln!("  {n_joh} pares cointegrados (antes correção)");

        if found.is_empty() {
            funil.push([fold, n_pre, nc, 0, 0, 0]);
            janela_inicio += cfg.passo_dias;
            continue;
        }

        // ── ETAPA 3: Correção múltiplos testes ──
        let pvs: Vec<f64> = found.iter().map(|f| f.4).collect();
        let mask = if cfg.correcao_multipla == "bh" {
            benjamini_hochberg(&pvs, cfg.alpha)
        } else {
            bonferroni(&pvs, cfg.alpha)
        };

        let corrigidos: Vec<_> = found
            .iter()
            .zip(mask.iter())
            .filter(|(_, &m)| m)
            .map(|(f, _)| f)
            .collect();

        let n_bh = corrigidos.len();
        eprintln!("  Após {}: {n_bh} pares", cfg.correcao_multipla.to_uppercase());

        if corrigidos.is_empty() {
            funil.push([fold, n_pre, nc, n_joh, 0, 0]);
            janela_inicio += cfg.passo_dias;
            continue;
        }

        // ── ETAPA 4: Estabilidade (paralelo) ──
        let estaveis: Vec<_> = corrigidos
            .par_iter()
            .filter_map(|par| {
                let sub = select_columns(&dtf, &par.0);
                let (ok, taxa) = filtro_estabilidade(
                    &sub,
                    cfg.n_sub_janelas,
                    cfg.min_aprovacao,
                    cfg.det_order,
                    cfg.k_ar_diff,
                );
                if ok { Some((*par, taxa)) } else { None }
            })
            .collect();

        let n_est = estaveis.len();
        eprintln!("  Após estabilidade: {n_est} pares");

        if estaveis.is_empty() {
            funil.push([fold, n_pre, nc, n_joh, n_bh, 0]);
            janela_inicio += cfg.passo_dias;
            continue;
        }

        // ── ETAPA 5: Métricas (paralelo) ──
        let fold_results: Vec<PairResult> = estaveis
            .par_iter()
            .filter_map(|(par, taxa)| {
                let (ref aidx, na, ref vetor, score, pv) = *par;
                let sub = select_columns(&dtf, aidx);
                let ts = sub.nrows();
                if ts < 60 {
                    return None;
                }

                let spread: Vec<f64> = (0..ts)
                    .map(|i| (0..na).map(|j| sub[(i, j)] * vetor[j]).sum())
                    .collect();

                let hl = half_life(&spread);
                if hl <= 0.0 || hl.is_infinite() {
                    return None;
                }

                Some(PairResult {
                    ativos: aidx.iter().map(|&i| nomes_f[i].clone()).collect(),
                    n_ativos: na,
                    vetor_coint: vetor.iter().map(|v| (v * 1e6).round() / 1e6).collect(),
                    score_johansen: score,
                    p_valor: pv,
                    estabilidade: *taxa,
                    half_life: hl,
                    adf_pvalor: adf_pvalue(&spread),
                    hurst: hurst_exponent(&spread, 100),
                    variance_ratio: variance_ratio(&spread, 10),
                })
            })
            .collect();

        let nf = fold_results.len();
        funil.push([fold, n_pre, nc, n_joh, n_bh, nf]);
        eprintln!("  ✓ {nf} pares com métricas completas");
        all_results.extend(fold_results);

        janela_inicio += cfg.passo_dias;
    }

    (all_results, funil, fold)
}

// ═══════════════════════════════════════════════════════════════════
// LEITURA DE ARQUIVOS
// ═══════════════════════════════════════════════════════════════════

/// Lê um Parquet onde cada coluna numérica (f64/f32) é uma série temporal.
/// Retorna (data_flat row-major, n_rows, col_names).
fn load_parquet_series(path: &str) -> Result<(Vec<f64>, usize, Vec<String>), String> {
    let file = File::open(path).map_err(|e| format!("Erro ao abrir '{path}': {e}"))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| format!("Erro Parquet '{path}': {e}"))?
        .with_batch_size(8192);

    let schema = builder.schema().clone();

    // Identificar colunas numéricas
    let num_cols: Vec<(usize, String)> = schema
        .fields()
        .iter()
        .enumerate()
        .filter(|(_, f)| matches!(f.data_type(), DataType::Float64 | DataType::Float32 | DataType::Int64))
        .map(|(i, f)| (i, f.name().clone()))
        .collect();

    if num_cols.is_empty() {
        return Err("Nenhuma coluna numérica encontrada no Parquet.".into());
    }

    let reader = builder.build().map_err(|e| format!("Erro reader: {e}"))?;
    let col_names: Vec<String> = num_cols.iter().map(|(_, n)| n.clone()).collect();
    let nc = col_names.len();
    let mut data: Vec<Vec<f64>> = vec![Vec::new(); nc];

    for batch_result in reader {
        let batch = batch_result.map_err(|e| format!("Erro batch: {e}"))?;
        let nr = batch.num_rows();

        for (out_j, (schema_j, _)) in num_cols.iter().enumerate() {
            let col = batch.column(*schema_j);
            let any = col.as_any();

            if let Some(arr) = any.downcast_ref::<Float64Array>() {
                for i in 0..nr {
                    data[out_j].push(if arr.is_null(i) { f64::NAN } else { arr.value(i) });
                }
            } else if let Some(arr) = any.downcast_ref::<Float32Array>() {
                for i in 0..nr {
                    data[out_j].push(if arr.is_null(i) { f64::NAN } else { arr.value(i) as f64 });
                }
            } else if let Some(arr) = any.downcast_ref::<Int64Array>() {
                for i in 0..nr {
                    data[out_j].push(if arr.is_null(i) { f64::NAN } else { arr.value(i) as f64 });
                }
            }
        }
    }

    let n_rows = data[0].len();
    // Flatten row-major
    let mut flat = Vec::with_capacity(n_rows * nc);
    for i in 0..n_rows {
        for j in 0..nc {
            flat.push(data[j][i]);
        }
    }

    Ok((flat, n_rows, col_names))
}

/// Lê um CSV onde cada coluna numérica é uma série temporal.
/// A primeira linha deve conter os headers.
fn load_csv_series(path: &str) -> Result<(Vec<f64>, usize, Vec<String>), String> {
    let file = File::open(path).map_err(|e| format!("Erro ao abrir '{path}': {e}"))?;
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_reader(BufReader::new(file));

    let headers: Vec<String> = rdr
        .headers()
        .map_err(|e| format!("Erro ao ler headers CSV: {e}"))?
        .iter()
        .map(|s| s.to_string())
        .collect();

    let total_cols = headers.len();
    let mut raw_rows: Vec<Vec<String>> = Vec::new();

    for result in rdr.records() {
        let record = result.map_err(|e| format!("Erro CSV: {e}"))?;
        raw_rows.push(record.iter().map(|s| s.to_string()).collect());
    }

    if raw_rows.is_empty() {
        return Err("CSV vazio.".into());
    }

    // Detectar quais colunas são numéricas (tentar parsear a primeira linha não-vazia)
    let mut numeric_cols: Vec<(usize, String)> = Vec::new();
    for j in 0..total_cols {
        let sample = raw_rows.iter().find(|r| r.len() > j && !r[j].is_empty());
        if let Some(row) = sample {
            if row[j].parse::<f64>().is_ok() {
                numeric_cols.push((j, headers[j].clone()));
            }
        }
    }

    if numeric_cols.is_empty() {
        return Err("Nenhuma coluna numérica encontrada no CSV.".into());
    }

    let nc = numeric_cols.len();
    let n_rows = raw_rows.len();
    let col_names: Vec<String> = numeric_cols.iter().map(|(_, n)| n.clone()).collect();

    let mut flat = Vec::with_capacity(n_rows * nc);
    for row in &raw_rows {
        for (j, _) in &numeric_cols {
            let val = if *j < row.len() {
                row[*j].parse::<f64>().unwrap_or(f64::NAN)
            } else {
                f64::NAN
            };
            flat.push(val);
        }
    }

    Ok((flat, n_rows, col_names))
}

/// Carrega de arquivo (detecta .parquet/.csv pela extensão).
fn load_from_file(path: &str) -> Result<(Vec<f64>, usize, Vec<String>), String> {
    let lower = path.to_lowercase();
    if lower.ends_with(".parquet") || lower.ends_with(".pq") {
        load_parquet_series(path)
    } else if lower.ends_with(".csv") || lower.ends_with(".tsv") {
        load_csv_series(path)
    } else {
        Err(format!(
            "Extensão não reconhecida em '{path}'. Use .parquet ou .csv"
        ))
    }
}

/// Forward-fill NaN + drop colunas com NaN restante.
/// Retorna (flat_clean, n_rows, col_names_clean).
fn clean_flat_data(
    flat: &[f64],
    n_rows: usize,
    col_names: &[String],
) -> (Vec<f64>, usize, Vec<String>) {
    let nc = col_names.len();

    // Forward-fill por coluna
    let mut cols: Vec<Vec<f64>> = Vec::with_capacity(nc);
    for j in 0..nc {
        let mut col = Vec::with_capacity(n_rows);
        let mut last = f64::NAN;
        for i in 0..n_rows {
            let v = flat[i * nc + j];
            if v.is_nan() {
                col.push(last);
            } else {
                last = v;
                col.push(v);
            }
        }
        // Backward-fill se começou com NaN
        if col[0].is_nan() {
            let first_valid = col.iter().position(|x| !x.is_nan());
            if let Some(fv) = first_valid {
                for i in 0..fv {
                    col[i] = col[fv];
                }
            }
        }
        cols.push(col);
    }

    // Remover colunas que ainda têm NaN
    let valid: Vec<usize> = (0..nc)
        .filter(|&j| !cols[j].iter().any(|x| x.is_nan()))
        .collect();

    let nc2 = valid.len();
    let names2: Vec<String> = valid.iter().map(|&j| col_names[j].clone()).collect();
    let mut flat2 = Vec::with_capacity(n_rows * nc2);
    for i in 0..n_rows {
        for &j in &valid {
            flat2.push(cols[j][i]);
        }
    }

    (flat2, n_rows, names2)
}

// ═══════════════════════════════════════════════════════════════════
// HELPER: montar resultado Python
// ═══════════════════════════════════════════════════════════════════

fn build_python_result(
    py: Python<'_>,
    results: &[PairResult],
    funil: &[[usize; 6]],
    n_folds: usize,
    n_testes: usize,
) -> PyResult<PyObject> {
    let py_results = PyList::empty_bound(py);
    for r in results {
        let d = PyDict::new_bound(py);
        let ativos_list = PyList::new_bound(py, &r.ativos).map_err(|e| PyValueError::new_err(e.to_string()))?;
        d.set_item("ativos", ativos_list)?;
        d.set_item("n_ativos", r.n_ativos)?;
        let vc_list = PyList::new_bound(py, &r.vetor_coint).map_err(|e| PyValueError::new_err(e.to_string()))?;
        d.set_item("vetor_coint", vc_list)?;
        d.set_item("score_johansen", r.score_johansen)?;
        d.set_item("p_valor", r.p_valor)?;
        d.set_item("estabilidade", r.estabilidade)?;
        d.set_item("half_life", r.half_life)?;
        d.set_item("adf_pvalor", r.adf_pvalor)?;
        d.set_item("hurst", r.hurst)?;
        d.set_item("variance_ratio", r.variance_ratio)?;
        py_results.append(d)?;
    }

    let py_funil = PyList::empty_bound(py);
    for f in funil {
        let d = PyDict::new_bound(py);
        d.set_item("fold", f[0])?;
        d.set_item("ativos_inicial", f[1])?;
        d.set_item("ativos_pos_corr", f[2])?;
        d.set_item("pares_johansen", f[3])?;
        d.set_item("pares_pos_bh", f[4])?;
        d.set_item("pares_final", f[5])?;
        py_funil.append(d)?;
    }

    let out = PyDict::new_bound(py);
    out.set_item("resultados", py_results)?;
    out.set_item("funil", py_funil)?;
    out.set_item("n_folds", n_folds)?;
    out.set_item("n_testes_total", n_testes)?;

    Ok(out.into_any().unbind())
}

fn extract_config(
    janela_treino_dias: usize,
    passo_dias: usize,
    embargo_dias: usize,
    min_ativos: usize,
    max_ativos: usize,
    min_corr: f64,
    top_n_corr: usize,
    n_sub_janelas: usize,
    min_aprovacao: f64,
    correcao_multipla: &str,
    alpha: f64,
    det_order: i32,
    k_ar_diff: usize,
) -> WalkForwardConfig {
    WalkForwardConfig {
        janela_treino_dias,
        passo_dias,
        embargo_dias,
        min_ativos,
        max_ativos,
        min_corr,
        top_n_corr,
        n_sub_janelas,
        min_aprovacao,
        correcao_multipla: correcao_multipla.to_string(),
        alpha,
        det_order,
        k_ar_diff,
    }
}

// ═══════════════════════════════════════════════════════════════════
// PyO3 — FUNÇÕES EXPOSTAS
// ═══════════════════════════════════════════════════════════════════

/// Walk-Forward de Cointegração a partir de dados flat (DataFrame → Python).
///
/// data_flat: lista plana row-major (T × n_cols).
#[pyfunction]
#[pyo3(signature = (
    data_flat, n_rows, col_names,
    janela_treino_dias = 504, passo_dias = 63, embargo_dias = 5,
    min_ativos = 2, max_ativos = 7, min_corr = 0.90, top_n_corr = 16,
    n_sub_janelas = 5, min_aprovacao = 0.75,
    correcao_multipla = "bh", alpha = 0.05,
    det_order = 0, k_ar_diff = 1
))]
pub fn coint_walk_forward(
    py: Python<'_>,
    data_flat: Vec<f64>,
    n_rows: usize,
    col_names: Vec<String>,
    janela_treino_dias: usize,
    passo_dias: usize,
    embargo_dias: usize,
    min_ativos: usize,
    max_ativos: usize,
    min_corr: f64,
    top_n_corr: usize,
    n_sub_janelas: usize,
    min_aprovacao: f64,
    correcao_multipla: &str,
    alpha: f64,
    det_order: i32,
    k_ar_diff: usize,
) -> PyResult<PyObject> {
    let nc = col_names.len();
    if data_flat.len() != n_rows * nc {
        return Err(PyValueError::new_err(format!(
            "data_flat tem {} elementos, esperado {} ({n_rows}×{nc})",
            data_flat.len(), n_rows * nc
        )));
    }

    let (clean, nr, names) = clean_flat_data(&data_flat, n_rows, &col_names);
    let nc2 = names.len();
    if nc2 < min_ativos {
        return Err(PyValueError::new_err(format!(
            "Apenas {nc2} colunas válidas (mínimo: {min_ativos})"
        )));
    }

    let data = DMatrix::from_row_slice(nr, nc2, &clean);
    let cfg = extract_config(
        janela_treino_dias, passo_dias, embargo_dias,
        min_ativos, max_ativos, min_corr, top_n_corr,
        n_sub_janelas, min_aprovacao, correcao_multipla, alpha,
        det_order, k_ar_diff,
    );

    let min_required = cfg.janela_treino_dias + cfg.embargo_dias + cfg.passo_dias;
    if nr < min_required {
        return Err(PyValueError::new_err(format!(
            "Dados insuficientes: {nr} linhas, mínimo {min_required}"
        )));
    }

    let (results, funil, nf) = walk_forward_engine(&data, &names, &cfg);
    let nt: usize = funil.iter().map(|f| {
        // Estimar n_testes a partir das combinações
        let nc_fold = f[2]; // ativos_pos_corr
        (cfg.min_ativos..=cfg.max_ativos.min(nc_fold))
            .map(|k| n_combinations(nc_fold, k))
            .sum::<usize>()
    }).sum();

    build_python_result(py, &results, &funil, nf, nt)
}

/// Walk-Forward a partir de arquivo (.parquet ou .csv).
#[pyfunction]
#[pyo3(signature = (
    file_path,
    janela_treino_dias = 504, passo_dias = 63, embargo_dias = 5,
    min_ativos = 2, max_ativos = 7, min_corr = 0.90, top_n_corr = 16,
    n_sub_janelas = 5, min_aprovacao = 0.75,
    correcao_multipla = "bh", alpha = 0.05,
    det_order = 0, k_ar_diff = 1
))]
pub fn coint_walk_forward_from_file(
    py: Python<'_>,
    file_path: &str,
    janela_treino_dias: usize,
    passo_dias: usize,
    embargo_dias: usize,
    min_ativos: usize,
    max_ativos: usize,
    min_corr: f64,
    top_n_corr: usize,
    n_sub_janelas: usize,
    min_aprovacao: f64,
    correcao_multipla: &str,
    alpha: f64,
    det_order: i32,
    k_ar_diff: usize,
) -> PyResult<PyObject> {
    let (flat, n_rows, col_names) =
        load_from_file(file_path).map_err(|e| PyValueError::new_err(e))?;

    coint_walk_forward(
        py, flat, n_rows, col_names,
        janela_treino_dias, passo_dias, embargo_dias,
        min_ativos, max_ativos, min_corr, top_n_corr,
        n_sub_janelas, min_aprovacao, correcao_multipla, alpha,
        det_order, k_ar_diff,
    )
}

/// Teste de Johansen isolado.
///
/// Retorna dict: rank, trace_stats, eigenvalues, eigenvector, score, p_value, crit_*
#[pyfunction]
#[pyo3(signature = (data_flat, n_rows, n_cols, det_order = 0, k_ar_diff = 1))]
pub fn coint_johansen_test(
    py: Python<'_>,
    data_flat: Vec<f64>,
    n_rows: usize,
    n_cols: usize,
    det_order: i32,
    k_ar_diff: usize,
) -> PyResult<PyObject> {
    if data_flat.len() != n_rows * n_cols {
        return Err(PyValueError::new_err("Dimensões inconsistentes."));
    }
    let data = DMatrix::from_row_slice(n_rows, n_cols, &data_flat);

    match johansen_trace(&data, det_order, k_ar_diff) {
        Some(r) => {
            let d = PyDict::new_bound(py);
            d.set_item("rank", r.rank)?;
            let ts = PyList::new_bound(py, &r.trace_stats)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            d.set_item("trace_stats", ts)?;
            let ev: Vec<f64> = (0..n_cols).map(|i| r.eigenvectors[(i, 0)]).collect();
            let evl = PyList::new_bound(py, &ev)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            d.set_item("eigenvector", evl)?;
            d.set_item("score", r.score)?;
            d.set_item("p_value", r.p_value)?;
            if !r.crit_values.is_empty() {
                let cv = r.crit_values[0];
                d.set_item("crit_90", cv[0])?;
                d.set_item("crit_95", cv[1])?;
                d.set_item("crit_99", cv[2])?;
            }
            Ok(d.into_any().unbind())
        }
        None => Err(PyValueError::new_err(
            "Johansen falhou (dados insuficientes ou singulares).",
        )),
    }
}

/// Teste ADF isolado. Retorna p-valor.
#[pyfunction]
pub fn coint_adf_test(series: Vec<f64>) -> PyResult<f64> {
    if series.len() < 20 {
        return Err(PyValueError::new_err("Série muito curta (mín. 20 obs)."));
    }
    Ok(adf_pvalue(&series))
}

/// Métricas de qualidade de um spread.
///
/// Retorna dict: half_life, adf_pvalor, hurst, variance_ratio
#[pyfunction]
pub fn coint_spread_quality(py: Python<'_>, spread: Vec<f64>) -> PyResult<PyObject> {
    if spread.len() < 20 {
        return Err(PyValueError::new_err("Spread muito curto (mín. 20)."));
    }
    let d = PyDict::new_bound(py);
    d.set_item("half_life", half_life(&spread))?;
    d.set_item("adf_pvalor", adf_pvalue(&spread))?;
    d.set_item("hurst", hurst_exponent(&spread, 100))?;
    d.set_item("variance_ratio", variance_ratio(&spread, 10))?;
    Ok(d.into_any().unbind())
}

// ═══════════════════════════════════════════════════════════════════
// REGISTRO NO MÓDULO PyO3
// ═══════════════════════════════════════════════════════════════════

/// Chamado pelo lib.rs para registrar todas as funções deste módulo.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(coint_walk_forward, m)?)?;
    m.add_function(wrap_pyfunction!(coint_walk_forward_from_file, m)?)?;
    m.add_function(wrap_pyfunction!(coint_johansen_test, m)?)?;
    m.add_function(wrap_pyfunction!(coint_adf_test, m)?)?;
    m.add_function(wrap_pyfunction!(coint_spread_quality, m)?)?;
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════
// TESTES
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // LCG simples para reprodutibilidade
    fn lcg(state: &mut u64) -> f64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((*state >> 33) as f64) / (u32::MAX as f64) - 0.5
    }

    fn gen_cointegrated(n: usize, beta: f64, noise: f64, seed: u64) -> DMatrix<f64> {
        let mut s = seed;
        let mut data = DMatrix::zeros(n, 2);
        let mut y1 = 100.0;
        for i in 0..n {
            y1 += lcg(&mut s) * 0.5;
            data[(i, 0)] = y1;
            data[(i, 1)] = beta * y1 + lcg(&mut s) * noise;
        }
        data
    }

    fn gen_random_walks(n: usize, k: usize, seed: u64) -> DMatrix<f64> {
        let mut s = seed;
        let mut data = DMatrix::zeros(n, k);
        for j in 0..k {
            let mut v = 100.0;
            for i in 0..n {
                v += lcg(&mut s) * 0.5;
                data[(i, j)] = v;
            }
        }
        data
    }

    #[test]
    fn test_johansen_cointegrated() {
        let data = gen_cointegrated(500, 1.5, 0.5, 42);
        let r = johansen_trace(&data, 0, 1);
        assert!(r.is_some());
        let r = r.unwrap();
        assert!(r.rank >= 1, "rank = {}", r.rank);
        assert!(r.score > 1.0, "score = {}", r.score);
    }

    #[test]
    fn test_johansen_independent() {
        let data = gen_random_walks(500, 3, 99);
        if let Some(r) = johansen_trace(&data, 0, 1) {
            assert!(
                r.rank == 0 || r.score < 1.5,
                "rank={}, score={}",
                r.rank,
                r.score
            );
        }
    }

    #[test]
    fn test_half_life_ar1() {
        let mut spread = vec![0.0; 1000];
        let mut s = 42u64;
        for i in 1..1000 {
            spread[i] = 0.9 * spread[i - 1] + lcg(&mut s) * 0.1;
        }
        let hl = half_life(&spread);
        assert!(hl > 2.0 && hl < 20.0, "hl = {hl}");
    }

    #[test]
    fn test_hurst_rw() {
        let mut spread = vec![0.0; 2000];
        let mut s = 123u64;
        for i in 1..2000 {
            spread[i] = spread[i - 1] + lcg(&mut s) * 2.0;
        }
        let h = hurst_exponent(&spread, 100);
        assert!(h > 0.35 && h < 0.65, "hurst = {h}");
    }

    #[test]
    fn test_vr_rw() {
        let mut spread = vec![0.0; 2000];
        let mut s = 456u64;
        for i in 1..2000 {
            spread[i] = spread[i - 1] + lcg(&mut s) * 2.0;
        }
        let vr = variance_ratio(&spread, 10);
        assert!(vr > 0.7 && vr < 1.3, "vr = {vr}");
    }

    #[test]
    fn test_adf_stationary() {
        let mut series = vec![0.0; 500];
        let mut s = 111u64;
        for i in 1..500 {
            series[i] = 0.7 * series[i - 1] + lcg(&mut s) * 0.2;
        }
        let p = adf_pvalue(&series);
        assert!(p < 0.10, "p = {p}");
    }

    #[test]
    fn test_adf_rw() {
        let mut series = vec![100.0; 500];
        let mut s = 222u64;
        for i in 1..500 {
            series[i] = series[i - 1] + lcg(&mut s) * 0.5;
        }
        let p = adf_pvalue(&series);
        assert!(p > 0.05, "p = {p}");
    }

    #[test]
    fn test_bh() {
        let p = vec![0.01, 0.04, 0.03, 0.20, 0.50];
        let m = benjamini_hochberg(&p, 0.05);
        assert!(m[0]);
    }

    #[test]
    fn test_bonferroni() {
        let p = vec![0.001, 0.04, 0.03, 0.20, 0.50];
        let m = bonferroni(&p, 0.05);
        assert!(m[0]);
        assert!(!m[1]);
    }

    #[test]
    fn test_combinations() {
        let c = combinations(4, 2);
        assert_eq!(c.len(), 6);
        assert_eq!(c[0], vec![0, 1]);
        assert_eq!(c[5], vec![2, 3]);
    }

    #[test]
    fn test_pre_filtro() {
        let n = 500;
        let mut data = DMatrix::zeros(n, 3);
        let mut s = 789u64;
        let mut y1 = 100.0;
        for i in 0..n {
            y1 += lcg(&mut s) * 0.5;
            data[(i, 0)] = y1;
            data[(i, 1)] = y1 + lcg(&mut s) * 0.01;
            data[(i, 2)] = 100.0 + lcg(&mut s) * 50.0;
        }
        let sel = pre_filtro_correlacao(&data, 0.95, 2);
        assert!(sel.len() >= 2);
    }
}