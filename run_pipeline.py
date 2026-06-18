import sys
import json
from pipeline import VilagPipeline, VilagConfig
from utils import load_posts_from_csv, filter_potential_hate

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_pipeline.py <your_data.csv>")
        sys.exit(1)
    file_path = sys.argv[1]
    config = VilagConfig()
    pipeline = VilagPipeline(config)

    try:
        posts = load_posts_from_csv(file_path)
        print(f"Loaded {len(posts)} posts from {file_path}.")
    except ValueError as e:
        print(f"Error loading CSV: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        sys.exit(1)

    # The filter_potential_hate is currently a placeholder, returning all posts.
    filtered_posts = filter_potential_hate(posts)

    clusters = pipeline.detect_coordination(filtered_posts)

    report = {
        "source_file": file_path,
        "total_posts": len(posts),
        "clusters_detected": len(clusters),
        "clusters": clusters,
        "methodology": "PROJECT VILAG – coordination by meaning and time, zero author data stored."
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    with open("custom_report.json", 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("Report saved to custom_report.json")
