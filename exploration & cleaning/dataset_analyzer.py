import pandas as pd
import numpy as np
import unicodedata
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# Layout helpers

WIDTH = 80

def _hr():
    print("─" * WIDTH)

def _section(text):
    print()
    print(text)
    _hr()
    print()

def _kv(key, value, key_width=30):
    print(f"  {key:<{key_width}} {value}")


# Core helpers

def _fmt(x, decimals=4):
    if isinstance(x, float):
        if abs(x) >= 1_000_000:
            return f"{x:,.0f}"
        if abs(x) >= 1_000:
            return f"{x:,.2f}"
        return f"{x:.{decimals}f}"
    if isinstance(x, (int, np.integer)):
        return f"{x:,}"
    return str(x)

def _is_numeric(series):
    return pd.api.types.is_numeric_dtype(series)

def _is_string(series):
    return pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)

def _outlier_bounds_iqr(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr

def _display(text: str) -> str:
    return text.replace('\r\n', '↵ ').replace('\r', '↵ ').replace('\n', '↵ ')

def _visual_len(s: str) -> int:
    """Lunghezza visiva di una stringa: i caratteri wide (es. kanji) contano 2."""
    w = 0
    for ch in s:
        eaw = unicodedata.east_asian_width(ch)
        w += 2 if eaw in ('W', 'F') else 1
    return w

def _ljust_visual(s: str, width: int) -> str:
    return s + ' ' * max(0, width - _visual_len(s))

def _rjust_visual(s: str, width: int) -> str:
    return ' ' * max(0, width - _visual_len(s)) + s

def _row_str(df, idx, skip_col=None, max_str_len=40):
    """Restituisce una stringa compatta con tutte le colonne della riga all'indice idx."""
    if df is None:
        return ""
    try:
        row = df.loc[idx]
    except KeyError:
        return ""
    parts = []
    for col, val in row.items():
        if col == skip_col:
            continue
        try:
            if pd.isna(val):
                continue
        except (TypeError, ValueError):
            pass
        if isinstance(val, str):
            disp = _display(val)
            if len(disp) > max_str_len:
                disp = disp[:max_str_len] + '…'
            parts.append(f"{col}: {disp}")
        elif isinstance(val, float):
            parts.append(f"{col}: {_fmt(val)}")
        else:
            parts.append(f"{col}: {val}")
    return ("  ↳  " + " | ".join(parts)) if parts else ""

def _example_idx(sv, val):
    """Restituisce il primo indice dove sv == val."""
    matches = sv[sv == val].index
    return matches[0] if len(matches) else None

def _example_idx_containing(sv, word):
    """Restituisce il primo indice dove sv contiene la parola word (come token)."""
    pattern = r'(?<!\w)' + word + r'(?!\w)'
    matches = sv[sv.str.contains(pattern, regex=True, na=False)].index
    return matches[0] if len(matches) else None

def _print_table(headers, rows, alignments=None, max_str_width=50):
    """Stampa una tabella testuale con intestazioni e righe di dati."""
    if alignments is None:
        alignments = ['l'] * len(headers)

    def fmt_cell(val):
        if val is None:
            return ""
        try:
            if pd.isna(val):
                return ""
        except (TypeError, ValueError):
            pass
        if isinstance(val, str):
            s = _display(val)
            return (s[:max_str_width] + "…") if _visual_len(s) > max_str_width else s
        if isinstance(val, (int, np.integer)):
            return _fmt(val)
        if isinstance(val, float):
            return _fmt(val)
        return str(val)

    str_rows = [[fmt_cell(v) for v in row] for row in rows]
    widths = [max(_visual_len(h), max((_visual_len(r[i]) for r in str_rows), default=0))
              for i, h in enumerate(headers)]

    def fmt_row(cells):
        parts = []
        for i, cell in enumerate(cells):
            w = widths[i]
            parts.append(_rjust_visual(cell, w) if alignments[i] == 'r' else _ljust_visual(cell, w))
        return "  " + "   ".join(parts)

    print(fmt_row(headers))
    print("  " + "   ".join("─" * w for w in widths))
    for row in str_rows:
        print(fmt_row(row))


# ANALISI NUMERICA

def _analyze_numeric(s: pd.Series, df=None):

    _section("Conteggi di base")
    total  = len(s)
    n_null = s.isna().sum()
    n_valid = s.notna().sum()
    n_zero = (s == 0).sum()
    n_neg  = (s < 0).sum()
    n_pos  = (s > 0).sum()
    n_uniq = s.nunique()

    _kv("Righe totali",       _fmt(total))
    _kv("Valori non nulli",   f"{_fmt(n_valid)}  ({n_valid/total*100:.2f}%)")
    _kv("Null / NaN",         f"{_fmt(n_null)}  ({n_null/total*100:.2f}%)")
    _kv("Zeri",               f"{_fmt(n_zero)}  ({n_zero/total*100:.2f}%)")
    _kv("Positivi",           f"{_fmt(n_pos)}   ({n_pos/total*100:.2f}%)")
    _kv("Negativi",           f"{_fmt(n_neg)}   ({n_neg/total*100:.2f}%)")
    _kv("Valori unici",       f"{_fmt(n_uniq)}  ({n_uniq/total*100:.2f}%)")

    sv = s.dropna()
    if sv.empty:
        print("  Nessun valore valido da analizzare.")
        return

    if s.dtype == np.float64 or s.dtype == np.float32:
        n_float = int((sv % 1 != 0).sum())
        _kv("Valori float", _fmt(n_float))


    # Statistiche descrittive

    _section("Statistiche descrittive")
    mean   = sv.mean()
    median = sv.median()
    mode_v = sv.mode()
    std    = sv.std()
    var    = sv.var()
    sem    = sv.sem()
    mn     = sv.min()
    mx     = sv.max()
    rng    = mx - mn
    q1     = sv.quantile(0.25)
    q3     = sv.quantile(0.75)
    iqr    = q3 - q1
    p5     = sv.quantile(0.05)
    p95    = sv.quantile(0.95)

    _kv("Min",                      _fmt(mn))
    _kv("Max",                      _fmt(mx))
    _kv("Range",                    _fmt(rng))
    _kv("Media",                    _fmt(mean))
    _kv("Mediana",                  _fmt(median))
    _kv("Moda/e",                   ", ".join(_fmt(v) for v in mode_v.head(3).values))
    _kv("Dev. standard",            _fmt(std))
    _kv("Varianza",                 _fmt(var))
    _kv("Errore std (SEM)",         _fmt(sem))
    _kv("IQR  (Q3 − Q1)",           _fmt(iqr))
    _kv("Q1  (25° pct)",            _fmt(q1))
    _kv("Q3  (75° pct)",            _fmt(q3))
    _kv("P5  ( 5° pct)",            _fmt(p5))
    _kv("P95 (95° pct)",            _fmt(p95))
    _kv("Coefficiente di variazione", f"{(std/mean*100) if mean else 0:.2f}%")


    # Analisi outlier

    _section("Analisi outlier")
    lo_iqr, hi_iqr = _outlier_bounds_iqr(sv)
    out_iqr = sv[(sv < lo_iqr) | (sv > hi_iqr)]

    _kv("Soglia bassa  (Q1 − 1.5×IQR)", _fmt(lo_iqr))
    _kv("Soglia alta   (Q3 + 1.5×IQR)", _fmt(hi_iqr))
    _kv("Outlier rilevati",              f"{_fmt(len(out_iqr))}  ({len(out_iqr)/total*100:.2f}%)")

    if len(out_iqr):
        top_out = out_iqr.abs().nlargest(5)
        print(f"\n  Top 5 outlier estremi (per valore assoluto):")
        for idx, val in top_out.items():
            print(f"    [{idx}]  {_fmt(val)}")
            row = _row_str(df, idx, skip_col=s.name)
            if row:
                print(f"    {row}")

    # Distribuzione

    _section("Distribuzione")
    bins = min(20, n_uniq)
    counts, edges = np.histogram(sv.values, bins=bins)
    total_valid = counts.sum()
    print(f"  {'Intervallo':<30} {'Conteggio':>10}   {'%':>6}")
    print(f"  {'─'*30} {'─'*10}   {'─'*6}")
    for i, count in enumerate(counts):
        lo = int(edges[i])
        hi = int(edges[i + 1])
        pct = count / total_valid * 100
        range_str = f"{lo:,} – {hi:,}"
        print(f"  {range_str:<30} {count:>10,}   {pct:>5.1f}%")


    # Valori estremi

    _section("Valori estremi")
    top10    = sv.nlargest(10)
    bottom10 = sv.nsmallest(10)

    print(f"  I 10 più grandi:")
    for i, (idx, val) in enumerate(top10.items(), 1):
        print(f"    #{i:<3} {_fmt(val)}")
        row = _row_str(df, idx, skip_col=s.name)
        if row:
            print(f"    {row}")

    print(f"\n  I 10 più piccoli:")
    for i, (idx, val) in enumerate(bottom10.items(), 1):
        print(f"    #{i:<3} {_fmt(val)}")
        row = _row_str(df, idx, skip_col=s.name)
        if row:
            print(f"    {row}")


    # Valori più e meno frequenti

    _section("Valori più e meno frequenti")

    vc = sv.value_counts()

    print(f"  I 10 più frequenti:")
    print(f"  {'Valore':>20}   {'Conteggio':>10}   {'%':>6}")
    print(f"  {'─'*20}   {'─'*10}   {'─'*6}")
    for val, cnt in vc.head(10).items():
        v = f"{_fmt(val):>20}"
        print(f"  {v}   {_fmt(cnt):>10}   {cnt/total*100:>5.1f}%")
        ex_idx = _example_idx(sv, val)
        if ex_idx is not None:
            row = _row_str(df, ex_idx, skip_col=s.name)
            if row:
                print(f"  {row}")

    print(f"\n  I 10 meno frequenti:")
    print(f"  {'Valore':>20}   {'Conteggio':>10}   {'%':>6}")
    print(f"  {'─'*20}   {'─'*10}   {'─'*6}")
    for val, cnt in vc.tail(10)[::-1].items():
        v = f"{_fmt(val):>20}"
        print(f"  {v}   {_fmt(cnt):>10}   {cnt/total*100:>5.1f}%")
        ex_idx = _example_idx(sv, val)
        if ex_idx is not None:
            row = _row_str(df, ex_idx, skip_col=s.name)
            if row:
                print(f"  {row}")


# ANALISI TESTUALE

def _analyze_string(s: pd.Series, df=None):

    _section("Conteggi di base")
    total   = len(s)
    n_empty = (s == "").sum()
    if n_empty > 0:
        s = s.replace("", np.nan)
    n_null  = s.isna().sum()
    n_valid = s.notna().sum()
    n_uniq  = s.nunique()
    dup_cnt = total - n_uniq - n_null

    _kv("Righe totali",              _fmt(total))
    _kv("Valori non nulli",          f"{_fmt(n_valid)}  ({n_valid/total*100:.2f}%)")
    _kv("Null / NaN",                f"{_fmt(n_null)}  ({n_null/total*100:.2f}%)")
    _kv("  di cui stringhe vuote",   f"{_fmt(n_empty)}  (convertite a NULL)")
    _kv("Valori unici",              f"{_fmt(n_uniq)}  ({n_uniq/total*100:.2f}%)")
    _kv("Valori duplicati",          f"{_fmt(dup_cnt)}  ({dup_cnt/total*100:.2f}%)")

    sv = s.dropna().astype(str)
    if sv.empty:
        print("  Nessun valore valido da analizzare.")
        return


    # Analisi lunghezza stringhe

    _section("Analisi lunghezza stringhe")
    lengths = sv.str.len()
    ln_min  = lengths.min()
    ln_max  = lengths.max()
    ln_mean = lengths.mean()
    ln_med  = lengths.median()
    ln_std  = lengths.std()
    ln_q1   = lengths.quantile(0.25)
    ln_q3   = lengths.quantile(0.75)

    _kv("Lunghezza min",            f"{_fmt(ln_min)}  caratteri")
    _kv("Lunghezza max",            f"{_fmt(ln_max)}  caratteri")
    _kv("Lunghezza media",          f"{_fmt(ln_mean, 2)}  caratteri")
    _kv("Lunghezza mediana",        f"{_fmt(ln_med)}  caratteri")
    _kv("Dev. standard lunghezza",  f"{_fmt(ln_std, 2)}")
    _kv("IQR lunghezza",            f"{_fmt(ln_q3 - ln_q1)}")


    # Stringhe più lunghe e più corte

    _section("Valori di lunghezza estrema")
    idx_max = lengths.nlargest(10).index
    idx_min = lengths.nsmallest(10).index

    headers_len = ["#", "Lunghezza", "Valore"]
    aligns_len  = ["r", "r", "l"]

    def _len_rows(indices):
        rows = []
        for rank, idx in enumerate(indices, 1):
            val = sv[idx]
            rows.append([f"#{rank}", len(val), val])
        return rows

    print(f"  Le 10 stringhe più lunghe:")
    _print_table(headers_len, _len_rows(idx_max), alignments=aligns_len)

    print(f"\n  Le 10 stringhe più corte:")
    _print_table(headers_len, _len_rows(idx_min), alignments=aligns_len)


    # Distribuzione lunghezze

    _section("Distribuzione lunghezze")
    bins = min(20, ln_max - ln_min + 1)
    if bins > 1:
        counts, edges = np.histogram(lengths.values, bins=int(bins))
        total_valid = counts.sum()
        print(f"  {'Intervallo':<30} {'Conteggio':>10}   {'%':>6}")
        print(f"  {'─'*30} {'─'*10}   {'─'*6}")
        for i, count in enumerate(counts):
            lo = int(edges[i])
            hi = int(edges[i + 1])
            range_str = f"{lo:,} – {hi:,}"
            print(f"  {range_str:<30} {count:>10,}   {count/total_valid*100:>5.1f}%")
    else:
        print(f"  Tutte le stringhe hanno la stessa lunghezza. ({ln_min} caratteri)")


    # Valori più e meno frequenti

    _section("Valori più e meno frequenti")
    vc = sv.value_counts()

    top10 = list(vc.head(10).items())
    bot10 = list(vc.tail(10)[::-1].items())

    headers_freq = ["Valore", "Conteggio", "%"]
    aligns_freq  = ["l", "r", "r"]

    def _freq_rows(items):
        return [[_display(val), cnt, f"{cnt/n_valid*100:.1f}%"] for val, cnt in items]

    print(f"  I 10 più frequenti:")
    _print_table(headers_freq, _freq_rows(top10), alignments=aligns_freq)

    print(f"\n  I 10 meno frequenti (più rari):")
    _print_table(headers_freq, _freq_rows(bot10), alignments=aligns_freq)


    # Analisi pattern / contenuto

    _section("Analisi pattern / contenuto")
    has_digits   = sv.str.contains(r'\d', regex=True).sum()
    has_upper    = sv.str.contains(r'[A-Z]', regex=True).sum()
    has_lower    = sv.str.contains(r'[a-z]', regex=True).sum()
    has_space    = sv.str.contains(r'\s', regex=True).sum()
    has_special  = sv.str.contains(r'[^a-zA-Z0-9\s]', regex=True).sum()
    has_url      = sv.str.contains(r'https?://[a-zA-Z0-9\-._~:/?#\[\]@!$&\'()*+,;=%]+', regex=True).sum()
    has_email    = sv.str.contains(r'@', regex=True).sum()
    all_numeric  = sv.str.fullmatch(r'\d+').sum()
    all_alpha    = sv.str.fullmatch(r'[a-zA-Z]+').sum()
    all_alphanum = sv.str.fullmatch(r'[a-zA-Z0-9]+').sum()

    _kv("Contiene cifre",             f"{_fmt(has_digits)}  ({has_digits/n_valid*100:.1f}%)")
    _kv("Contiene maiuscole",         f"{_fmt(has_upper)}  ({has_upper/n_valid*100:.1f}%)")
    _kv("Contiene minuscole",         f"{_fmt(has_lower)}  ({has_lower/n_valid*100:.1f}%)")
    _kv("Contiene spazi",             f"{_fmt(has_space)}  ({has_space/n_valid*100:.1f}%)")
    _kv("Contiene caratteri speciali",f"{_fmt(has_special)}  ({has_special/n_valid*100:.1f}%)")
    _kv("Contiene URL",               f"{_fmt(has_url)}  ({has_url/n_valid*100:.1f}%)")
    _kv("Contiene @",                 f"{_fmt(has_email)}  ({has_email/n_valid*100:.1f}%)")
    _kv("Solo numerico",              f"{_fmt(all_numeric)}")
    _kv("Solo alfabetico",            f"{_fmt(all_alpha)}")
    _kv("Solo alfanumerico",          f"{_fmt(all_alphanum)}")


    # Statistiche a livello di carattere

    _section("Statistiche a livello di carattere")
    all_chars  = "".join(sv.values)
    n_chars    = len(all_chars)
    uniq_chars = len(set(all_chars))

    _kv("Caratteri totali",         _fmt(n_chars))
    _kv("Caratteri unici",          _fmt(uniq_chars))
    _kv("Caratteri medi per valore",f"{n_chars/n_valid:.2f}")


    # Analisi parole / token (solo se multi-parola)

    avg_words = sv.str.split().str.len().mean()
    if avg_words > 1.5:
        _section("Analisi parole / token")
        word_counts = sv.str.split().str.len()
        _kv("Parole medie per valore", f"{word_counts.mean():.2f}")
        _kv("Max parole",              _fmt(word_counts.max()))
        _kv("Min parole",              _fmt(word_counts.min()))

        all_words = " ".join(sv.values).split()
        word_freq = Counter(all_words).most_common(10)
        print(f"\n  Le 10 parole più comuni:")
        print(f"  {'Parola':<20}   {'Conteggio':>10}")
        print(f"  {'─'*20}   {'─'*10}")
        for word, cnt in word_freq:
            print(f"  {word:<20}   {_fmt(cnt):>10}")


# API PUBBLICA

def analyze(series: pd.Series, name: str | None = None, df: pd.DataFrame | None = None) -> None:
    
    if not isinstance(series, pd.Series):
        raise TypeError(f"Atteso un pandas Series, ricevuto {type(series).__name__}")

    label = name or (str(series.name) if series.name is not None else "Series")
    dtype = str(series.dtype)

    print()
    _kv("Nome serie",   label)
    _kv("dtype",        dtype)
    _kv("Dimensione",   _fmt(len(series)))
    _kv("Range indice", f"{series.index[0]} … {series.index[-1]}" if len(series) else "vuoto")

    if _is_numeric(series):
        _kv("Tipo", "NUMERICO")
        _analyze_numeric(series, df=df)
    elif _is_string(series):
        _kv("Tipo", "STRINGA / OGGETTO")
        _analyze_string(series, df=df)
    else:
        _kv("Tipo", f"ALTRO ({dtype})")
        print("\n  Analisi automatica non disponibile per questo dtype.")
        print(f"  Tentativo di conversione a stringa…\n")
        _analyze_string(series.astype(str), df=df)

    print()
