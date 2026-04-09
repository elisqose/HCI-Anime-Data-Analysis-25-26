import pandas as pd
import warnings
warnings.filterwarnings("ignore")


#  Layout helpers (stesso stile di dataset_analyzer) 

WIDTH = 80

def _hr():
    print("─" * WIDTH)

def _section(text):
    print()
    print(text)
    _hr()
    print()

def _kv(key, value, key_width=35):
    print(f"  {key:<{key_width}} {value}")

def _fmt(x):
    if isinstance(x, float):
        return f"{x:,.4f}"
    if isinstance(x, int):
        return f"{x:,}"
    return str(x)


#  API pubblica 

def check_fk(
    child: pd.Series,
    parent: pd.Series,
    child_df: pd.DataFrame = None,
    sample_rows: int = 10,
) -> pd.Series:

    child_name  = child.name  or "FK"
    parent_name = parent.name or "PK"

    # ── Intestazione ────────────────────────────────────────────────────────
    print()
    _kv("Colonna FK  (tabella figlia)",  str(child_name))
    _kv("Colonna PK  (tabella padre)",   str(parent_name))
    _hr()

    # ── Preparazione ────────────────────────────────────────────────────────
    valid_pks   = set(parent.dropna().unique())
    total       = len(child)
    n_null_fk   = child.isna().sum()
    n_valid_fk  = child.notna().sum()

    mask_orphan = child.notna() & ~child.isin(valid_pks)
    n_orphan    = mask_orphan.sum()
    n_ok        = n_valid_fk - n_orphan

    orphan_ids  = sorted(child[mask_orphan].unique())
    n_uniq_orphan = len(orphan_ids)

    #  Riepilogo conteggi 
    _section("Riepilogo conteggi")
    _kv("Righe totali (tabella figlia)",      _fmt(total))
    _kv("Valori null nella FK",               f"{_fmt(n_null_fk)}  ({n_null_fk/total*100:.2f}%)")
    _kv("Valori non-null nella FK",           f"{_fmt(n_valid_fk)}  ({n_valid_fk/total*100:.2f}%)")
    _kv("Valori unici nella PK padre",        _fmt(len(valid_pks)))
    print()
    _kv("✓  Righe con FK valida",             f"{_fmt(n_ok)}  ({n_ok/total*100:.2f}%)")
    _kv("✗  Righe orfane (FK non in PK)",     f"{_fmt(n_orphan)}  ({n_orphan/total*100:.2f}%)")
    _kv("   → ID orfani unici",               _fmt(n_uniq_orphan))

    #  Valutazione 
    _section("Valutazione integrità referenziale")
    if n_orphan == 0:
        print("Nessuna violazione. Tutti i valori FK esistono nella tabella padre.")
    else:
        print(f"{n_orphan:,} riga/e ({n_orphan/total*100:.2f}%) violano l'integrità referenziale.")

    #  ID orfani 
    if orphan_ids:
        _section("ID orfani")
        preview = orphan_ids[:20]
        suffix  = f"  … (+{n_uniq_orphan - 20} altri)" if n_uniq_orphan > 20 else ""
        print(f"  {preview}{suffix}")

    #  Campione righe orfane 
    if child_df is not None and n_orphan > 0:
        _section(f"Campione righe orfane (prime {min(sample_rows, n_orphan)})")
        sample = child_df[mask_orphan].head(sample_rows)
        print(sample.to_string(index=True))

    print()
    return mask_orphan
