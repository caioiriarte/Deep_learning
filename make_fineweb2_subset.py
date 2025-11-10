from datasets import load_dataset, Dataset
from pathlib import Path


def main():
    # 1) Choose a language subset; "eng_Latn" is English in Latin script.
    #    You can swap this string for another language-script pair later.
    stream = load_dataset(
        "HuggingFaceFW/fineweb-2",
        "ita_Latn",          # subset name
        split="train",
        streaming=True,      # don't load everything into RAM
    )

    # 2) Take a *very* small subset, e.g. first 1,000 documents
    small_samples = []
    for i, row in enumerate(stream):
        # Here is where you would apply *your* filtering if you have one
        # e.g. if len(row["text"]) > 200: ...
        small_samples.append(
            {
                "text": row["text"],
                "url": row.get("url", None),
                "language": row.get("language", None),
            }
        )
        if len(small_samples) >= 1000:   # <- change size here
            break

    # 3) Turn the list into a regular Dataset
    small_ds = Dataset.from_list(small_samples)

    # 4) Save to a Datasets/ folder inside your repo
    out_dir = Path("Datasets/fineweb2_subset")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "fineweb2_1000.jsonl"
    small_ds.to_json(out_path.as_posix(), lines=True)

    print(f"Saved subset with {len(small_ds)} rows to {out_path}")


if __name__ == "__main__":
    main()
    # print(Path.cwd())
