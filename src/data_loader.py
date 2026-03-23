from pathlib import Path
import pandas as pd

VALID_TYPES = {"factoid", "list", "yesno", "summary"}
QUESTION_COLUMNS = ("question", "body")
TYPE_COLUMN = "type"
ID_COLUMN = "id"


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    # Prefer 'question', fallback to 'body'
    for col in QUESTION_COLUMNS:
        if col in df.columns:
            if "question" not in df.columns and col == "body":
                df = df.rename(columns={"body": "question"})
            break
    if "question" not in df.columns:
        raise ValueError(f"CSV must have one of {QUESTION_COLUMNS} for question text")
    if TYPE_COLUMN not in df.columns:
        raise ValueError(f"CSV must have column '{TYPE_COLUMN}'")
    df = df[df[TYPE_COLUMN].astype(str).str.lower().isin(VALID_TYPES)].copy()
    df[TYPE_COLUMN] = df[TYPE_COLUMN].astype(str).str.lower()
    return df


def load_and_sample_test_set(
    csv_path: str | Path,
    n_total: int = 300,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Load test CSV and sample n_total questions with proportional representation by type.
    Expected columns: question (or body), type (factoid|list|yesno|summary), optional id.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Test CSV not found: {path}")
    df = pd.read_csv(path)
    df = _normalize_df(df)

    # 🔥 Filter only factoid + summary
    df = df[df[TYPE_COLUMN].isin(["factoid", "summary"])].copy()
    
    total = len(df)
    if total == 0:
        raise ValueError("No rows with valid question types (factoid, list, yesno, summary)")
    if total <= n_total:
        return df[["question", TYPE_COLUMN] + ([ID_COLUMN] if ID_COLUMN in df.columns else [])].reset_index(drop=True)

    type_counts = df.groupby(TYPE_COLUMN).size()
    proportions = type_counts / type_counts.sum()
    n_per_type = (proportions * n_total).round().astype(int)

    largest_type = type_counts.idxmax()
    diff = n_total - n_per_type.sum()
    if diff != 0:
        n_per_type[largest_type] = n_per_type[largest_type] + diff
    n_per_type = n_per_type.clip(lower=0)
    if n_per_type.sum() != n_total:
        n_per_type[largest_type] = n_per_type[largest_type] + (n_total - n_per_type.sum())

    sampled_list = []
    for qtype, group in df.groupby(TYPE_COLUMN, sort=False):
        n = int(n_per_type.get(qtype, 0))
        if n <= 0:
            continue
        n = min(n, len(group))
        sampled_list.append(group.sample(n=n, random_state=random_state))
    sampled = pd.concat(sampled_list, ignore_index=True)

    if len(sampled) < n_total:
        remaining = df.drop(sampled.index)
        need = n_total - len(sampled)
        extra = remaining.sample(n=min(need, len(remaining)), random_state=random_state)
        sampled = pd.concat([sampled, extra], ignore_index=True)
    elif len(sampled) > n_total:
        sampled = sampled.sample(n=n_total, random_state=random_state)

    out_cols = ["question", TYPE_COLUMN, ID_COLUMN]
    if ID_COLUMN in sampled.columns:
        out_cols.append(ID_COLUMN)
    return sampled[out_cols].reset_index(drop=True)