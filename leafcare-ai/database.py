"""
LeafCare AI - SQLite Database Handler
Provides persistent storage for leaf disease prediction history and analytics.
"""

import sqlite3
import os
import json
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), 'data', 'leafcare.db')


def get_connection():
    """Create a database connection with dictionary-like row access."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the SQLite database schema if not already created."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            image_url TEXT NOT NULL,
            plant_name TEXT NOT NULL,
            disease_name TEXT NOT NULL,
            confidence REAL NOT NULL,
            status TEXT NOT NULL,
            severity TEXT DEFAULT 'Unknown',
            recommendations TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()


def insert_prediction(filename, image_url, plant_name, disease_name, confidence, status, severity, recommendations):
    """
    Insert a new leaf disease prediction record into the database.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # Serialize recommendations if it's a dict or list
    if isinstance(recommendations, (dict, list)):
        recs_str = json.dumps(recommendations)
    else:
        recs_str = str(recommendations)

    cursor.execute('''
        INSERT INTO predictions (
            filename, image_url, plant_name, disease_name,
            confidence, status, severity, recommendations, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        filename,
        image_url,
        plant_name,
        disease_name,
        float(confidence),
        status,
        severity,
        recs_str,
        datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ))
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return new_id


def get_all_predictions(limit=100):
    """
    Fetch all historical predictions sorted by most recent first.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, filename, image_url, plant_name, disease_name,
               confidence, status, severity, recommendations, created_at
        FROM predictions
        ORDER BY id DESC
        LIMIT ?
    ''', (limit,))
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for row in rows:
        rec = dict(row)
        try:
            rec['recommendations'] = json.loads(rec['recommendations'])
        except Exception:
            pass
        results.append(rec)
    return results


def get_prediction_by_id(pred_id):
    """
    Retrieve a single prediction by its primary key ID.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM predictions WHERE id = ?', (pred_id,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return None
    rec = dict(row)
    try:
        rec['recommendations'] = json.loads(rec['recommendations'])
    except Exception:
        pass
    return rec


def delete_prediction(pred_id):
    """
    Delete a specific prediction record from history.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM predictions WHERE id = ?', (pred_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


def clear_all_history():
    """
    Clear all prediction records (useful for demo resets).
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM predictions')
    conn.commit()
    conn.close()
    return True


def get_statistics():
    """
    Compute aggregate stats for dashboard metrics.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) FROM predictions')
    total_scans = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM predictions WHERE status = 'Healthy'")
    healthy_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM predictions WHERE status != 'Healthy'")
    diseased_count = cursor.fetchone()[0]
    
    cursor.execute('''
        SELECT disease_name, COUNT(*) as cnt 
        FROM predictions 
        WHERE status != 'Healthy' 
        GROUP BY disease_name 
        ORDER BY cnt DESC 
        LIMIT 1
    ''')
    top_disease_row = cursor.fetchone()
    top_disease = top_disease_row[0] if top_disease_row else "None"
    
    cursor.execute('SELECT AVG(confidence) FROM predictions')
    avg_conf_row = cursor.fetchone()[0]
    avg_confidence = round(avg_conf_row, 2) if avg_conf_row is not None else 0.0

    conn.close()
    return {
        "total_scans": total_scans,
        "healthy_count": healthy_count,
        "diseased_count": diseased_count,
        "top_disease": top_disease,
        "avg_confidence": avg_confidence
    }


if __name__ == '__main__':
    init_db()
    print("LeafCare AI SQLite database initialized successfully at:", DB_PATH)
