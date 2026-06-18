from flask import Flask, render_template, jsonify, request, Response
import json
from datetime import datetime, timezone
from pipeline import VilagPipeline, VilagConfig
from utils import generate_organic_posts, generate_attack_posts, load_posts_from_csv
import tempfile
import os

app = Flask(__name__)
posts_db = []
clusters_db = []
pipeline = VilagPipeline(VilagConfig())


def _analysis_message(post_count, cluster_count):
    if cluster_count:
        return f"Loaded {post_count} posts and detected {cluster_count} coordination cluster(s)."
    return (
        f"Loaded {post_count} posts, but no coordination clusters matched the current "
        "thresholds. Check that similar toxic posts happen close together in time."
    )


@app.route('/')
def dashboard():
    return render_template("dashboard.html")


@app.route('/status')
def status():
    return jsonify({"total_posts": len(posts_db), "clusters_detected": len(clusters_db)})


@app.route('/init_organic', methods=['POST'])
def init_organic():
    global posts_db, clusters_db
    posts_db = generate_organic_posts(200)
    clusters_db = []
    return jsonify({"status": "ok", "count": len(posts_db)})


@app.route('/simulate_attack', methods=['POST'])
def simulate_attack():
    global clusters_db, posts_db
    target = request.json.get("target", "Anna Candidate")
    attack_posts = generate_attack_posts(target)
    posts_db.extend(attack_posts)
    clusters = pipeline.detect_coordination(posts_db)
    clusters_db = clusters
    return jsonify({
        "status": "attack_simulated",
        "attack_posts": len(attack_posts),
        "clusters_found": len(clusters),
        "target": target
    })


@app.route('/analyze', methods=['POST'])
def analyze():
    global clusters_db
    clusters_db = pipeline.detect_coordination(posts_db)
    return jsonify({"clusters": clusters_db})


@app.route('/clusters')
def get_clusters():
    return jsonify(clusters_db)


# ============================================================================
# ROUTES THAT MATCH THE DASHBOARD (dashboard.html calls these)
# ============================================================================

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """CSV upload endpoint — matches dashboard.html's fetch('/api/analyze')"""
    global posts_db, clusters_db

    if 'file' not in request.files:
        return jsonify({"error": "No file provided. Send a CSV with key 'file'."}), 400

    file = request.files['file']
    if not file or not file.filename:
        return jsonify({"error": "Empty file"}), 400

    if not file.filename.lower().endswith('.csv'):
        return jsonify({"error": "File must be a .csv"}), 400

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    tmp_path = tmp.name
    file.save(tmp_path)
    tmp.close()

    try:
        custom = load_posts_from_csv(tmp_path)
        if not custom:
            return jsonify({"error": "No valid posts found in CSV. Ensure it has a 'text' column."}), 400

        posts_db = custom
        clusters = pipeline.detect_coordination(posts_db)
        clusters_db = clusters
        return jsonify({
            "total_posts": len(custom),
            "clusters_detected": len(clusters),
            "clusters": clusters,
            "message": _analysis_message(len(custom), len(clusters)),
            "generated": datetime.now(timezone.utc).isoformat(),
            "methodology": "PROJECT VILAG – coordination by meaning and time, zero author data stored."
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.route('/api/demo', methods=['GET'])
def api_demo():
    """Demo endpoint — matches dashboard.html's fetch('/api/demo')"""
    global posts_db, clusters_db

    # Generate organic background + coordinated attack
    posts_db = generate_organic_posts(200)
    attack_posts = generate_attack_posts("Anna Candidate")
    posts_db.extend(attack_posts)

    # Run pipeline
    clusters = pipeline.detect_coordination(posts_db)
    clusters_db = clusters

    return jsonify({
        "total_posts": len(posts_db),
        "clusters_detected": len(clusters),
        "clusters": clusters,
        "message": _analysis_message(len(posts_db), len(clusters)),
        "generated": datetime.now(timezone.utc).isoformat(),
        "methodology": "PROJECT VILAG – coordination by meaning and time, zero author data stored.",
        "demo": True
    })


@app.route('/api/export', methods=['POST'])
def api_export():
    """Export endpoint — matches dashboard.html's fetch('/api/export')"""
    data = request.get_json() if request.is_json else {}

    report = {
        "report_generated": datetime.now(timezone.utc).isoformat(),
        "total_posts_analyzed": data.get("total_posts", len(posts_db)),
        "clusters_detected": len(data.get("clusters", clusters_db)),
        "clusters": data.get("clusters", clusters_db),
        "methodology": "PROJECT VILAG – coordination by meaning and time, zero author data stored.",
        "odihr_compatible": True
    }

    return Response(
        json.dumps(report, indent=2, ensure_ascii=False),
        mimetype='application/json',
        headers={"Content-Disposition": "attachment;filename=vilag_report.json"}
    )


# Keep old routes for backward compatibility
@app.route('/upload_data', methods=['POST'])
def upload_data():
    """Legacy upload route (kept for backward compatibility)"""
    global posts_db, clusters_db
    file = request.files.get('file')
    if not file or not file.filename.endswith('.csv'):
        return jsonify({"error": "CSV required"}), 400
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    tmp_path = tmp.name
    file.save(tmp_path)
    tmp.close()
    try:
        custom = load_posts_from_csv(tmp_path)
        posts_db = custom
        clusters = pipeline.detect_coordination(posts_db)
        clusters_db = clusters
        return jsonify({
            "total_posts": len(custom),
            "clusters_found": len(clusters),
            "message": _analysis_message(len(custom), len(clusters))
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.route('/export_json')
def export_json():
    """Legacy export route (kept for backward compatibility)"""
    report = {
        "report_time": datetime.now(timezone.utc).isoformat(),
        "total_posts_analyzed": len(posts_db),
        "clusters_detected": len(clusters_db),
        "clusters": clusters_db
    }
    return Response(json.dumps(report, indent=2, ensure_ascii=False),
                    mimetype='application/json',
                    headers={"Content-Disposition": "attachment;filename=vilag_report.json"})


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)
