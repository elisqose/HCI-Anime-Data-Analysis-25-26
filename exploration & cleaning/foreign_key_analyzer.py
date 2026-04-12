import pandas as pd
import numpy as np
import textwrap
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
    if isinstance(x, (int, np.integer)):
        return f"{x:,}"
    if isinstance(x, float):
        return f"{x:,.4f}"
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
        joined  = ", ".join(str(v) for v in preview)
        for line in textwrap.wrap(joined, width=WIDTH - 2):
            print(f"  {line}")
        if n_uniq_orphan > 20:
            print(f"  … (+{n_uniq_orphan - 20} altri)")

    #  Campione righe orfane 
    if child_df is not None and n_orphan > 0:
        _section(f"Campione righe orfane (prime {min(sample_rows, n_orphan)})")
        sample = child_df[mask_orphan].head(sample_rows)
        print(sample.to_string(index=True))

    print()
    return mask_orphan


def check_pk_referenced(
    parent: pd.Series,
    children: dict,
    parent_df: pd.DataFrame = None,
    sample_rows: int = 10,
) -> pd.Series:
    """
    Controllo inverso rispetto a check_fk.

    Verifica quali valori della PK (tabella padre) sono referenziati
    da almeno una delle tabelle figlie fornite.

    Parametri
    ---------
    parent   : Series con i valori PK da controllare (es. gli ID anomali)
    children : dict { nome_tabella: Series_FK } — le colonne FK delle tabelle figlie
    parent_df: DataFrame padre opzionale, usato per stampare un campione
               delle righe non referenziate
    sample_rows: numero di righe nel campione

    Ritorna
    -------
    mask_unreferenced : maschera booleana True sulle righe di parent_df
                        i cui ID non compaiono in nessuna tabella figlia
    """

    parent_name = parent.name or "PK"

    # ── Intestazione ────────────────────────────────────────────────────────
    print()
    _kv("Colonna PK  (tabella padre)", str(parent_name))
    _kv("Tabelle figlie verificate",   ", ".join(children.keys()))
    _hr()

    pk_values = set(parent.dropna().unique())
    total_pk  = len(pk_values)

    # ── Conteggio per tabella figlia ─────────────────────────────────────────
    _section("Referenze per tabella figlia")
    referenced_union = set()

    for name, fk_series in children.items():
        fk_vals      = set(fk_series.dropna().unique())
        found        = pk_values & fk_vals
        n_found      = len(found)
        referenced_union |= found
        _kv(f"  {name}", f"{_fmt(n_found)} / {_fmt(total_pk)} ID trovati  ({n_found/total_pk*100:.1f}%)")

    # ── Riepilogo globale ────────────────────────────────────────────────────
    _section("Riepilogo globale")
    n_referenced   = len(referenced_union)
    n_unreferenced = total_pk - n_referenced
    _kv("PK totali analizzati",               _fmt(total_pk))
    _kv("✓  Referenziati in almeno 1 figlia", f"{_fmt(n_referenced)}  ({n_referenced/total_pk*100:.1f}%)")
    _kv("✗  Non referenziati in nessuna",     f"{_fmt(n_unreferenced)}  ({n_unreferenced/total_pk*100:.1f}%)")

    # ── ID non referenziati ──────────────────────────────────────────────────
    unreferenced_ids = sorted(pk_values - referenced_union)
    if unreferenced_ids:
        _section("ID non referenziati (rimovibili in sicurezza)")
        preview = unreferenced_ids[:20]
        joined  = ", ".join(str(v) for v in preview)
        for line in textwrap.wrap(joined, width=WIDTH - 2):
            print(f"  {line}")
        if n_unreferenced > 20:
            print(f"  … (+{n_unreferenced - 20} altri)")
    else:
        _section("ID non referenziati")
        print("  Tutti gli ID sono referenziati in almeno una tabella figlia.")

    # ── Campione righe non referenziate ─────────────────────────────────────
    mask_unreferenced = parent.isin(unreferenced_ids)
    if parent_df is not None and mask_unreferenced.any():
        _section(f"Campione righe non referenziate (prime {min(sample_rows, mask_unreferenced.sum())})")
        sample_df = parent_df[parent_df[parent_name].isin(unreferenced_ids)]
        print(sample_df.head(sample_rows).to_string(index=True))

    print()
    return mask_unreferenced
