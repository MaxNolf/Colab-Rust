use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use arrow::array::{
    Float64Array, Float64Builder, Int64Array, Int64Builder, UInt64Builder,
    TimestampMillisecondArray, TimestampMicrosecondArray, TimestampNanosecondArray, TimestampSecondArray,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

use std::fs::File;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Max bars to buffer before flushing to Parquet.
const FLUSH_EVERY: usize = 10_000;

/// Batch size for reading Parquet row groups.
const READ_BATCH_SIZE: usize = 8_192;

// ---------------------------------------------------------------------------
// Bar accumulator (shared across all bar types)
// ---------------------------------------------------------------------------

struct BarAccumulator {
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
    dollar_volume: f64,
    trade_count: u64,
    timestamp: i64, // bar open time
    has_data: bool,
}

impl BarAccumulator {
    fn new() -> Self {
        Self {
            open: 0.0,
            high: f64::NEG_INFINITY,
            low: f64::INFINITY,
            close: 0.0,
            volume: 0.0,
            dollar_volume: 0.0,
            trade_count: 0,
            timestamp: 0,
            has_data: false,
        }
    }

    fn update(&mut self, price: f64, qty: f64, quote_qty: f64, time: i64) {
        if !self.has_data {
            self.open = price;
            self.timestamp = time;
            self.has_data = true;
        }
        if price > self.high {
            self.high = price;
        }
        if price < self.low {
            self.low = price;
        }
        self.close = price;
        self.volume += qty;
        self.dollar_volume += quote_qty;
        self.trade_count += 1;
    }

    fn reset(&mut self) {
        self.open = 0.0;
        self.high = f64::NEG_INFINITY;
        self.low = f64::INFINITY;
        self.close = 0.0;
        self.volume = 0.0;
        self.dollar_volume = 0.0;
        self.trade_count = 0;
        self.timestamp = 0;
        self.has_data = false;
    }
}

// ---------------------------------------------------------------------------
// Bar buffer — collects completed bars, flushes to ArrowWriter
// ---------------------------------------------------------------------------

struct BarBuffer {
    timestamps: Int64Builder,
    opens: Float64Builder,
    highs: Float64Builder,
    lows: Float64Builder,
    closes: Float64Builder,
    volumes: Float64Builder,
    dollar_volumes: Float64Builder,
    trade_counts: UInt64Builder,
    count: usize,
    total_bars: u64,
}

impl BarBuffer {
    fn new() -> Self {
        Self {
            timestamps: Int64Builder::with_capacity(FLUSH_EVERY),
            opens: Float64Builder::with_capacity(FLUSH_EVERY),
            highs: Float64Builder::with_capacity(FLUSH_EVERY),
            lows: Float64Builder::with_capacity(FLUSH_EVERY),
            closes: Float64Builder::with_capacity(FLUSH_EVERY),
            volumes: Float64Builder::with_capacity(FLUSH_EVERY),
            dollar_volumes: Float64Builder::with_capacity(FLUSH_EVERY),
            trade_counts: UInt64Builder::with_capacity(FLUSH_EVERY),
            count: 0,
            total_bars: 0,
        }
    }

    fn push(&mut self, acc: &BarAccumulator) {
        self.timestamps.append_value(acc.timestamp);
        self.opens.append_value(acc.open);
        self.highs.append_value(acc.high);
        self.lows.append_value(acc.low);
        self.closes.append_value(acc.close);
        self.volumes.append_value(acc.volume);
        self.dollar_volumes.append_value(acc.dollar_volume);
        self.trade_counts.append_value(acc.trade_count);
        self.count += 1;
        self.total_bars += 1;
    }

    fn should_flush(&self) -> bool {
        self.count >= FLUSH_EVERY
    }

    fn flush(&mut self, writer: &mut ArrowWriter<File>) -> Result<(), String> {
        if self.count == 0 {
            return Ok(());
        }

        let batch = RecordBatch::try_new(
            bar_schema(),
            vec![
                Arc::new(self.timestamps.finish()),
                Arc::new(self.opens.finish()),
                Arc::new(self.highs.finish()),
                Arc::new(self.lows.finish()),
                Arc::new(self.closes.finish()),
                Arc::new(self.volumes.finish()),
                Arc::new(self.dollar_volumes.finish()),
                Arc::new(self.trade_counts.finish()),
            ],
        )
        .map_err(|e| format!("Erro ao criar RecordBatch: {e}"))?;

        writer
            .write(&batch)
            .map_err(|e| format!("Erro ao escrever no Parquet: {e}"))?;

        self.count = 0;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Output schema (shared by all bar types)
// ---------------------------------------------------------------------------

fn bar_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("timestamp", DataType::Int64, false),
        Field::new("open", DataType::Float64, false),
        Field::new("high", DataType::Float64, false),
        Field::new("low", DataType::Float64, false),
        Field::new("close", DataType::Float64, false),
        Field::new("volume", DataType::Float64, false),
        Field::new("dollar_volume", DataType::Float64, false),
        Field::new("trade_count", DataType::UInt64, false),
    ]))
}

// ---------------------------------------------------------------------------
// Helper: create ArrowWriter for output
// ---------------------------------------------------------------------------

fn create_writer(output_path: &str) -> Result<ArrowWriter<File>, String> {
    let file = File::create(output_path)
        .map_err(|e| format!("Erro ao criar arquivo de saída '{}': {e}", output_path))?;

    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .set_max_row_group_size(FLUSH_EVERY)
        .build();

    ArrowWriter::try_new(file, bar_schema(), Some(props))
        .map_err(|e| format!("Erro ao inicializar ArrowWriter: {e}"))
}

// ---------------------------------------------------------------------------
// Helper: iterate over trades in a Parquet file, calling a closure per trade
// ---------------------------------------------------------------------------

fn for_each_trade<F>(input_path: &str, mut callback: F) -> Result<(), String>
where
    F: FnMut(i64, f64, f64, f64) -> Result<(), String>, // time, price, qty, quote_qty
{
    let file = File::open(input_path)
        .map_err(|e| format!("Erro ao abrir '{}': {e}", input_path))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| format!("Erro ao ler Parquet '{}': {e}", input_path))?
        .with_batch_size(READ_BATCH_SIZE);

    let reader = builder
        .build()
        .map_err(|e| format!("Erro ao construir reader: {e}"))?;

    for batch_result in reader {
        let batch = batch_result.map_err(|e| format!("Erro ao ler batch: {e}"))?;

        let datetime_col = batch
            .column_by_name("DateTime")
            .ok_or("Coluna 'DateTime' não encontrada")?;

        let num_rows = batch.num_rows();
        let mut time_values: Vec<i64> = Vec::with_capacity(num_rows);
        
        let any_col = datetime_col.as_any();
        if let Some(arr) = any_col.downcast_ref::<Int64Array>() {
            for i in 0..arr.len() { time_values.push(arr.value(i)); }
        } else if let Some(arr) = any_col.downcast_ref::<TimestampMillisecondArray>() {
            for i in 0..arr.len() { time_values.push(arr.value(i)); }
        } else if let Some(arr) = any_col.downcast_ref::<TimestampMicrosecondArray>() {
            for i in 0..arr.len() { time_values.push(arr.value(i) / 1000); }
        } else if let Some(arr) = any_col.downcast_ref::<TimestampNanosecondArray>() {
            for i in 0..arr.len() { time_values.push(arr.value(i) / 1_000_000); }
        } else if let Some(arr) = any_col.downcast_ref::<TimestampSecondArray>() {
            for i in 0..arr.len() { time_values.push(arr.value(i) * 1000); }
        } else {
            return Err(format!("Tipo não suportado na coluna 'DateTime': {:?}", datetime_col.data_type()));
        }

        let price_col = batch
            .column_by_name("Price")
            .ok_or("Coluna 'Price' não encontrada")?
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or("Coluna 'Price' não é Float64")?;

        let qty_col = batch
            .column_by_name("Qty")
            .ok_or("Coluna 'Qty' não encontrada")?
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or("Coluna 'Qty' não é Float64")?;

        for i in 0..num_rows {
            let time = time_values[i];
            let price = price_col.value(i);
            let qty = qty_col.value(i);
            let quote_qty = price * qty;
            callback(time, price, qty, quote_qty)?;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Helper: parse interval string to milliseconds
// ---------------------------------------------------------------------------

fn parse_interval(interval: &str) -> Result<i64, String> {
    let s = interval.trim();
    if s.is_empty() {
        return Err("Intervalo vazio".into());
    }

    // Find where digits end and unit begins
    let digit_end = s
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(s.len());

    if digit_end == 0 {
        return Err(format!("Intervalo inválido: '{s}'. Use ex: '5m', '1h', '1d'"));
    }

    let value: i64 = s[..digit_end]
        .parse()
        .map_err(|_| format!("Número inválido no intervalo: '{s}'"))?;

    let unit = &s[digit_end..];

    let ms_per_unit: i64 = match unit {
        "m" | "min" => 60_000,
        "h" | "hour" => 3_600_000,
        "d" | "day" => 86_400_000,
        "" => {
            return Err(format!(
                "Unidade ausente no intervalo: '{s}'. Use 'm' (minutos), 'h' (horas), 'd' (dias)"
            ));
        }
        _ => {
            return Err(format!(
                "Unidade desconhecida '{unit}' no intervalo: '{s}'. Use 'm', 'h', ou 'd'"
            ));
        }
    };

    Ok(value * ms_per_unit)
}

// ===========================================================================
// PyO3 functions
// ===========================================================================

/// Gera time bars a partir de um arquivo Parquet de trades da Binance.
///
/// - `input_path`: caminho do Parquet de entrada (trades)
/// - `output_path`: caminho do Parquet de saída (barras)
/// - `interval`: intervalo como string, ex: "5m", "1h", "1d"
///
/// Retorna o número de barras geradas.
#[pyfunction]
fn generate_time_bars(input_path: &str, output_path: &str, interval: &str) -> PyResult<u64> {
    let interval_ms =
        parse_interval(interval).map_err(|e| PyValueError::new_err(e))?;

    let mut writer =
        create_writer(output_path).map_err(|e| PyValueError::new_err(e))?;
    let mut buf = BarBuffer::new();
    let mut acc = BarAccumulator::new();
    let mut current_window_end: i64 = 0;

    for_each_trade(input_path, |time, price, qty, quote_qty| {
        // First trade ever — initialise window
        if !acc.has_data {
            current_window_end = time - (time % interval_ms) + interval_ms;
        }

        // While the trade falls beyond the current window, close bars
        while time >= current_window_end {
            if acc.has_data {
                buf.push(&acc);
                acc.reset();
                if buf.should_flush() {
                    buf.flush(&mut writer)?;
                }
            }
            current_window_end += interval_ms;
        }

        acc.update(price, qty, quote_qty, time);
        Ok(())
    })
    .map_err(|e| PyValueError::new_err(e))?;

    // Close final bar
    if acc.has_data {
        buf.push(&acc);
    }
    buf.flush(&mut writer)
        .map_err(|e| PyValueError::new_err(e))?;

    writer
        .close()
        .map_err(|e| PyValueError::new_err(format!("Erro ao finalizar Parquet: {e}")))?;

    Ok(buf.total_bars)
}

/// Gera volume bars a partir de um arquivo Parquet de trades da Binance.
///
/// - `input_path`: caminho do Parquet de entrada (trades)
/// - `output_path`: caminho do Parquet de saída (barras)
/// - `volume_threshold`: volume acumulado necessário para fechar cada barra
///
/// Retorna o número de barras geradas.
#[pyfunction]
fn generate_volume_bars(
    input_path: &str,
    output_path: &str,
    volume_threshold: f64,
) -> PyResult<u64> {
    if volume_threshold <= 0.0 {
        return Err(PyValueError::new_err(
            "volume_threshold deve ser > 0",
        ));
    }

    let mut writer =
        create_writer(output_path).map_err(|e| PyValueError::new_err(e))?;
    let mut buf = BarBuffer::new();
    let mut acc = BarAccumulator::new();

    for_each_trade(input_path, |time, price, qty, quote_qty| {
        acc.update(price, qty, quote_qty, time);

        while acc.volume >= volume_threshold {
            buf.push(&acc);
            acc.reset();
            if buf.should_flush() {
                buf.flush(&mut writer)?;
            }
        }

        Ok(())
    })
    .map_err(|e| PyValueError::new_err(e))?;

    // Close final bar (even if threshold not reached)
    if acc.has_data {
        buf.push(&acc);
    }
    buf.flush(&mut writer)
        .map_err(|e| PyValueError::new_err(e))?;

    writer
        .close()
        .map_err(|e| PyValueError::new_err(format!("Erro ao finalizar Parquet: {e}")))?;

    Ok(buf.total_bars)
}

/// Gera dollar bars a partir de um arquivo Parquet de trades da Binance.
///
/// - `input_path`: caminho do Parquet de entrada (trades)
/// - `output_path`: caminho do Parquet de saída (barras)
/// - `dollar_threshold`: valor em dólares acumulado necessário para fechar cada barra
///
/// Retorna o número de barras geradas.
#[pyfunction]
fn generate_dollar_bars(
    input_path: &str,
    output_path: &str,
    dollar_threshold: f64,
) -> PyResult<u64> {
    if dollar_threshold <= 0.0 {
        return Err(PyValueError::new_err(
            "dollar_threshold deve ser > 0",
        ));
    }

    let mut writer =
        create_writer(output_path).map_err(|e| PyValueError::new_err(e))?;
    let mut buf = BarBuffer::new();
    let mut acc = BarAccumulator::new();

    for_each_trade(input_path, |time, price, qty, quote_qty| {
        acc.update(price, qty, quote_qty, time);

        while acc.dollar_volume >= dollar_threshold {
            buf.push(&acc);
            acc.reset();
            if buf.should_flush() {
                buf.flush(&mut writer)?;
            }
        }

        Ok(())
    })
    .map_err(|e| PyValueError::new_err(e))?;

    // Close final bar
    if acc.has_data {
        buf.push(&acc);
    }
    buf.flush(&mut writer)
        .map_err(|e| PyValueError::new_err(e))?;

    writer
        .close()
        .map_err(|e| PyValueError::new_err(format!("Erro ao finalizar Parquet: {e}")))?;

    Ok(buf.total_bars)
}

/// Calcula os thresholds (time, volume, dollar) para que cada tipo de barra
/// gere aproximadamente `target_num_bars` barras.
///
/// - `input_path`: caminho do Parquet de trades
/// - `target_num_bars`: número desejado de barras
///
/// Retorna uma tupla `(interval_str, volume_threshold, dollar_threshold)` onde
/// `interval_str` é uma string como "5m" que pode ser passada diretamente para
/// `generate_time_bars`.
#[pyfunction]
fn calculate_thresholds(input_path: &str, target_num_bars: u64) -> PyResult<(String, f64, f64)> {
    if target_num_bars == 0 {
        return Err(PyValueError::new_err(
            "target_num_bars deve ser > 0",
        ));
    }

    let mut min_time: i64 = i64::MAX;
    let mut max_time: i64 = i64::MIN;
    let mut total_volume: f64 = 0.0;
    let mut total_dollar: f64 = 0.0;

    for_each_trade(input_path, |time, _price, qty, quote_qty| {
        if time < min_time {
            min_time = time;
        }
        if time > max_time {
            max_time = time;
        }
        total_volume += qty;
        total_dollar += quote_qty;
        Ok(())
    })
    .map_err(|e| PyValueError::new_err(e))?;

    if min_time >= max_time {
        return Err(PyValueError::new_err(
            "Dados insuficientes para calcular thresholds (nenhum trade ou span de tempo zero)",
        ));
    }

    let span_ms = max_time - min_time;
    let interval_ms = span_ms / target_num_bars as i64;

    // Convert interval_ms to a human-readable interval string
    let interval_str = if interval_ms >= 86_400_000 {
        let days = interval_ms / 86_400_000;
        format!("{}d", days.max(1))
    } else if interval_ms >= 3_600_000 {
        let hours = interval_ms / 3_600_000;
        format!("{}h", hours.max(1))
    } else {
        let minutes = interval_ms / 60_000;
        format!("{}m", minutes.max(1))
    };

    let volume_threshold = total_volume / target_num_bars as f64;
    let dollar_threshold = total_dollar / target_num_bars as f64;

    Ok((interval_str, volume_threshold, dollar_threshold))
}

/// Lê um arquivo Parquet e retorna estatísticas básicas sem carregar tudo na memória.
///
/// Retorna uma tupla `(num_rows, num_row_groups, columns)`.
#[pyfunction]
fn parquet_info(input_path: &str) -> PyResult<(i64, usize, Vec<String>)> {
    let file = File::open(input_path)
        .map_err(|e| PyValueError::new_err(format!("Erro ao abrir '{}': {e}", input_path)))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| PyValueError::new_err(format!("Erro ao ler Parquet: {e}")))?;

    let metadata = builder.metadata();
    let num_rows: i64 = metadata.file_metadata().num_rows();
    let num_row_groups = metadata.num_row_groups();

    let schema = builder.schema();
    let columns: Vec<String> = schema
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();

    Ok((num_rows, num_row_groups, columns))
}

// ===========================================================================
// PyO3 module
// ===========================================================================

#[pymodule]
fn template_rust_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_time_bars, m)?)?;
    m.add_function(wrap_pyfunction!(generate_volume_bars, m)?)?;
    m.add_function(wrap_pyfunction!(generate_dollar_bars, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_thresholds, m)?)?;
    m.add_function(wrap_pyfunction!(parquet_info, m)?)?;
    Ok(())
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{BooleanBuilder, Float64Builder, Int64Builder};
    use std::path::Path;

    /// Creates a synthetic Parquet file with `n` trades for testing.
    fn create_test_parquet(path: &str, n: usize) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("Price", DataType::Float64, false),
            Field::new("Qty", DataType::Float64, false),
            Field::new("quoteQty", DataType::Float64, false),
            Field::new("DateTime", DataType::Int64, false),
            Field::new("isBuyerMaker", DataType::Boolean, false),
            Field::new("isBestMatch", DataType::Boolean, false),
        ]));

        let file = File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema.clone(), None).unwrap();

        let mut id_builder = Int64Builder::with_capacity(n);
        let mut price_builder = Float64Builder::with_capacity(n);
        let mut qty_builder = Float64Builder::with_capacity(n);
        let mut quote_qty_builder = Float64Builder::with_capacity(n);
        let mut time_builder = Int64Builder::with_capacity(n);
        let mut buyer_maker_builder = BooleanBuilder::with_capacity(n);
        let mut best_match_builder = BooleanBuilder::with_capacity(n);

        // Simulate 1 trade per second starting at 2024-01-01 00:00:00 UTC
        let base_time: i64 = 1_704_067_200_000; // 2024-01-01 UTC in ms
        let base_price: f64 = 42000.0; // BTC-like price

        for i in 0..n {
            let price = base_price + (i as f64 * 0.1).sin() * 100.0;
            let qty = 0.01 + (i as f64 * 0.03).sin().abs() * 0.05;
            let quote_qty = price * qty;
            let time = base_time + (i as i64 * 1000); // 1 second apart

            id_builder.append_value(i as i64);
            price_builder.append_value(price);
            qty_builder.append_value(qty);
            quote_qty_builder.append_value(quote_qty);
            time_builder.append_value(time);
            buyer_maker_builder.append_value(i % 2 == 0);
            best_match_builder.append_value(true);
        }

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(id_builder.finish()),
                Arc::new(price_builder.finish()),
                Arc::new(qty_builder.finish()),
                Arc::new(quote_qty_builder.finish()),
                Arc::new(time_builder.finish()),
                Arc::new(buyer_maker_builder.finish()),
                Arc::new(best_match_builder.finish()),
            ],
        )
        .unwrap();

        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    #[test]
    fn test_parse_interval() {
        assert_eq!(parse_interval("5m").unwrap(), 300_000);
        assert_eq!(parse_interval("1h").unwrap(), 3_600_000);
        assert_eq!(parse_interval("1d").unwrap(), 86_400_000);
        assert!(parse_interval("").is_err());
        assert!(parse_interval("abc").is_err());
        assert!(parse_interval("5x").is_err());
    }

    #[test]
    fn test_time_bars() {
        let dir = std::env::temp_dir().join("rust_bar_test");
        std::fs::create_dir_all(&dir).unwrap();

        let input = dir.join("trades.parquet");
        let output = dir.join("time_bars.parquet");

        create_test_parquet(input.to_str().unwrap(), 3600); // 1 hour of data (1 trade/sec)

        // 5 minute bars → expect ~12 bars
        let n = generate_time_bars(
            input.to_str().unwrap(),
            output.to_str().unwrap(),
            "5m",
        )
        .unwrap();

        assert!(n >= 11 && n <= 13, "Expected ~12 time bars, got {n}");
        assert!(Path::new(output.to_str().unwrap()).exists());

        // Verify output is valid Parquet
        let (rows, _, cols) = parquet_info(output.to_str().unwrap()).unwrap();
        assert_eq!(rows as u64, n);
        assert_eq!(cols.len(), 8);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_volume_and_dollar_bars() {
        let dir = std::env::temp_dir().join("rust_bar_test_vd");
        std::fs::create_dir_all(&dir).unwrap();

        let input = dir.join("trades.parquet");
        let vol_out = dir.join("volume_bars.parquet");
        let dol_out = dir.join("dollar_bars.parquet");

        create_test_parquet(input.to_str().unwrap(), 10000);

        let n_vol = generate_volume_bars(
            input.to_str().unwrap(),
            vol_out.to_str().unwrap(),
            1.0, // 1 BTC per bar
        )
        .unwrap();

        let n_dol = generate_dollar_bars(
            input.to_str().unwrap(),
            dol_out.to_str().unwrap(),
            50000.0, // ~50k USD per bar
        )
        .unwrap();

        assert!(n_vol > 0, "Expected at least 1 volume bar");
        assert!(n_dol > 0, "Expected at least 1 dollar bar");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_calculate_thresholds_equal_bars() {
        let dir = std::env::temp_dir().join("rust_bar_test_th");
        std::fs::create_dir_all(&dir).unwrap();

        let input = dir.join("trades.parquet");
        let time_out = dir.join("time_bars_th.parquet");
        let vol_out = dir.join("volume_bars_th.parquet");
        let dol_out = dir.join("dollar_bars_th.parquet");

        create_test_parquet(input.to_str().unwrap(), 36000); // 10 hours of data

        let target = 100u64;
        let (interval_str, vol_th, dol_th) =
            calculate_thresholds(input.to_str().unwrap(), target).unwrap();

        let n_time = generate_time_bars(
            input.to_str().unwrap(),
            time_out.to_str().unwrap(),
            &interval_str,
        )
        .unwrap();

        let n_vol = generate_volume_bars(
            input.to_str().unwrap(),
            vol_out.to_str().unwrap(),
            vol_th,
        )
        .unwrap();

        let n_dol = generate_dollar_bars(
            input.to_str().unwrap(),
            dol_out.to_str().unwrap(),
            dol_th,
        )
        .unwrap();

        // Volume and dollar bars should be close to target (±5%)
        let margin = (target as f64 * 0.05).ceil() as u64 + 1;
        assert!(
            n_vol.abs_diff(target) <= margin,
            "Volume bars: expected ~{target}, got {n_vol}"
        );
        assert!(
            n_dol.abs_diff(target) <= margin,
            "Dollar bars: expected ~{target}, got {n_dol}"
        );

        // Time bars may deviate more due to rounding to minutes/hours/days
        // but should be in reasonable range
        assert!(
            n_time > 0 && n_time < target * 3,
            "Time bars: expected reasonable count, got {n_time}"
        );

        std::fs::remove_dir_all(&dir).ok();
    }
}
