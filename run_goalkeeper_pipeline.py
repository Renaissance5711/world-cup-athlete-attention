from pathlib import Path

from ijsms.goalkeeper_workflow import run_goalkeeper_workflow


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    paths = run_goalkeeper_workflow(
        workspace=root,
        news_csv_paths=[
            root / "data" / "raw" / "news" / "google_news_01_20_articles.csv",
            root / "data" / "raw" / "news" / "google_news_21_40_articles.csv",
        ],
        output_dir=root / "outputs" / "goalkeeper",
    )
    for name, path in paths.items():
        print(f"{name}: {path}")
