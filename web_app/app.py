import os, json, uuid
import numpy as np
import pandas as pd
import tensorflow as tf
import neurokit2 as nk
from scipy.signal import resample as scipy_resample
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from pathlib import Path
from collections import Counter, deque
from datetime import datetime

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024   # 20 MB
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs('uploads/', exist_ok=True)

MODEL_DIR = Path(__file__).parent.parent / 'model'
print("Loading model...")
MODEL = tf.keras.models.load_model(MODEL_DIR / 'ecg_arrhythmia_model.h5')
with open(MODEL_DIR / 'model_metadata.json') as f:
    META = json.load(f)
CLASS_NAMES = {int(k): v for k, v in META['class_names'].items()}
TARGET_LENGTH = META['window_size']   # 187
TARGET_FS = 360                        # Hz that MIT-BIH was recorded at
print("Model ready.")


# ─── Feature 3: Alert History (in-memory, resets on restart) ──────────────────
ALERT_HISTORY = []
ALERT_COUNTER = 0

# ─── Feature 1: Confirmation buffer ──────────────────────────────────────────
CONFIRMATION_BUFFER = deque(maxlen=5)

# ─── Feature 4: Clinical explanations ────────────────────────────────────────
CLINICAL_INFO = {
    0: {
        "title": "Normal Sinus Rhythm",
        "explanation": "Your ECG shows a normal sinus rhythm. The heart's electrical pattern appears regular with no signs of arrhythmia detected across the analyzed beats.",
        "waveform_note": "The QRS complex is narrow and upright, the P wave precedes each QRS at a consistent interval, and the T wave follows normally. This is the expected pattern of a healthy heart.",
        "action": "No immediate action required. Continue routine cardiac monitoring as advised by your physician."
    },
    1: {
        "title": "Supraventricular Ectopic Beat (SVEB)",
        "explanation": "A Supraventricular Ectopic Beat was detected. This means an abnormal electrical impulse originated above the ventricles — in the atria or AV node — causing a premature or irregular heartbeat. The beat arrives earlier than expected and may feel like a skipped or extra beat.",
        "waveform_note": "The QRS complex is typically narrow (similar to normal) but is preceded by an abnormal or absent P wave, or a P wave with unusual morphology. The beat appears earlier than the expected rhythm.",
        "action": "Often benign, especially if infrequent. However, frequent SVEBs can trigger supraventricular tachycardia (SVT). Medical evaluation is recommended if symptoms like palpitations, dizziness, or shortness of breath occur."
    },
    2: {
        "title": "Ventricular Ectopic Beat (VEB)",
        "explanation": "A Ventricular Ectopic Beat was detected. This is an abnormal heartbeat that originates in the lower chambers (ventricles) rather than the heart's natural pacemaker (SA node). The electrical pathway is abnormal, causing the ventricles to contract inefficiently.",
        "waveform_note": "The QRS complex is abnormally wide and bizarrely shaped. The T wave points in the opposite direction to the QRS complex — a hallmark of ventricular origin. There is no preceding P wave. The beat is followed by a compensatory pause.",
        "action": "Isolated VEBs are common and often harmless in healthy individuals. Frequent VEBs (more than 10% of beats), runs of VEBs, or VEBs occurring in the context of heart disease require urgent medical evaluation as they can indicate increased risk of ventricular tachycardia or fibrillation."
    },
    3: {
        "title": "Fusion Beat",
        "explanation": "A Fusion Beat was detected. This occurs when a normal supraventricular impulse and a ventricular ectopic impulse activate the ventricles simultaneously, producing a hybrid waveform that is intermediate between a normal beat and a ventricular ectopic beat.",
        "waveform_note": "The QRS complex has intermediate morphology — wider than normal but not as wide or bizarre as a pure ventricular ectopic beat. The shape is a blend of the normal QRS and a ventricular beat. This pattern is often seen in accelerated idioventricular rhythm or ventricular tachycardia.",
        "action": "Fusion beats themselves are not dangerous but indicate the presence of an ectopic ventricular focus competing with the normal pacemaker. Medical evaluation is recommended to identify the underlying cause."
    },
    4: {
        "title": "Unclassifiable Beat (Unknown)",
        "explanation": "An unclassifiable beat pattern was detected. This may represent a paced rhythm (from an artificial pacemaker), severe signal noise or artifact, or an uncommon morphology not well-represented in the MIT-BIH training data.",
        "waveform_note": "The beat morphology does not clearly match any of the four standard categories. This can occur with pacemaker spikes, lead misplacement, patient movement artifact, or rare arrhythmia patterns.",
        "action": "If the patient does not have a pacemaker, this result warrants clinical review. Ensure good electrode contact and repeat the recording. If the pattern persists, consult a cardiologist."
    }
}

# Synthetic normal QRS template (187 samples)
SYNTHETIC_NORMAL_TEMPLATE = (
    [0.0]*40 +
    [0.1, 0.2, 0.35, 0.55, 0.75, 0.95, 1.0, 0.95, 0.75, 0.55, 0.35, 0.2, 0.1] +
    [0.05]*20 +
    [0.15, 0.25, 0.3, 0.25, 0.15] +
    [0.0]*109
)


# ─── Signal loading ───────────────────────────────────────────────────────────

def load_csv(filepath):
    """
    Load any CSV/TXT file flexibly.
    Handles: header or no header, multiple columns, single column.
    Always returns a 1D numpy array of float32.
    """
    # Try reading with header first
    df = pd.read_csv(filepath)

    # If only one column, just use it
    if df.shape[1] == 1:
        return df.iloc[:, 0].values.astype(np.float32)

    # Multiple columns — try to find the signal column
    # Common column names in ECG exports
    ecg_col_hints = ['ecg', 'signal', 'mlii', 'lead', 'voltage',
                     'value', 'amplitude', 'ch1', 'channel']
    for col in df.columns:
        if any(hint in col.lower() for hint in ecg_col_hints):
            return df[col].values.astype(np.float32)

    # No obvious column name — use the first numeric column
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        return df[numeric_cols[0]].values.astype(np.float32)

    # Last resort: try reading without header
    df2 = pd.read_csv(filepath, header=None)
    return df2.iloc[:, 0].values.astype(np.float32)


# ─── Two processing modes ─────────────────────────────────────────────────────

def process_already_segmented(signal):
    """
    File looks like MIT-BIH format:
    - Short (≤ 250 samples) → treat as single beat
    - Multiple rows each 187 values → batch of beats
    Normalizes each beat to [0, 1].
    """
    # Normalize to [0, 1]
    def norm(x):
        mn, mx = x.min(), x.max()
        if mx - mn < 1e-8:
            return x - mn
        return (x - mn) / (mx - mn)

    # If signal is exactly 187 or close, treat as one beat
    if len(signal) <= 250:
        beat = scipy_resample(signal, TARGET_LENGTH).astype(np.float32)
        beat = norm(beat)
        return beat.reshape(1, TARGET_LENGTH, 1), signal

    # Otherwise treat as multiple concatenated beats each of 187 samples
    n = len(signal) // TARGET_LENGTH
    if n >= 2:
        trimmed = signal[:n * TARGET_LENGTH]
        beats = trimmed.reshape(n, TARGET_LENGTH)
        beats = np.array([norm(b) for b in beats], dtype=np.float32)
        return beats.reshape(-1, TARGET_LENGTH, 1), signal

    return None, signal


def process_raw_ecg(signal, input_fs):
    """
    File is a raw continuous ECG recording.
    Steps:
      1. Resample to TARGET_FS (360 Hz)
      2. Clean signal
      3. Detect R-peaks
      4. Segment each beat (100 before, 260 after R-peak)
      5. Resize each beat to TARGET_LENGTH (187)
      6. Normalize each beat to [0, 1]
    """
    # 1. Resample to 360 Hz
    n_resampled = int(len(signal) * TARGET_FS / input_fs)
    resampled = scipy_resample(signal, n_resampled).astype(np.float32)

    # 2. Clean
    try:
        cleaned = nk.ecg_clean(resampled, sampling_rate=TARGET_FS)
    except Exception:
        cleaned = resampled   # fallback if cleaning fails

    # 3. R-peak detection
    try:
        _, info = nk.ecg_peaks(cleaned, sampling_rate=TARGET_FS)
        r_peaks = info['ECG_R_Peaks']
    except Exception as e:
        raise ValueError(
            f"Could not detect heartbeats in this signal. "
            f"Make sure it is a real ECG voltage signal. Details: {e}"
        )

    if len(r_peaks) < 2:
        raise ValueError(
            "Found fewer than 2 heartbeats. "
            "Signal may be too short or not a valid ECG."
        )

    # 4 & 5 & 6. Extract, resize, normalize
    before, after = 100, 260
    beats = []
    for peak in r_peaks:
        if peak - before < 0 or peak + after > len(cleaned):
            continue
        beat = cleaned[peak - before: peak + after]
        beat_resized = scipy_resample(beat, TARGET_LENGTH).astype(np.float32)
        mn, mx = beat_resized.min(), beat_resized.max()
        if mx - mn > 1e-8:
            beat_resized = (beat_resized - mn) / (mx - mn)
        beats.append(beat_resized)

    if len(beats) == 0:
        raise ValueError("No complete beats could be extracted from the signal.")

    beats_arr = np.array(beats, dtype=np.float32).reshape(-1, TARGET_LENGTH, 1)
    return beats_arr, resampled, r_peaks


def detect_mode_and_process(signal, input_fs=None):
    """
    Automatically decide which processing path to take.

    Logic:
    - Very short signal (≤ 250 samples) → single pre-segmented beat
    - Short-ish (≤ 5000 samples) and length divisible by 187 → batch of beats
    - Long signal (> 1000 samples) → raw ECG, run R-peak detection

    Returns (windows, cleaned_signal, mode_description, beats_found)
    """
    if input_fs is None:
        input_fs = TARGET_FS

    n = len(signal)

    # Remove NaN/Inf
    signal = signal[np.isfinite(signal)]
    if len(signal) == 0:
        raise ValueError("File contains no valid numeric values.")

    # Single pre-segmented beat
    if n <= 250:
        windows, raw = process_already_segmented(signal)
        return windows, raw, "single beat", len(windows), None

    # Batch of pre-segmented beats (e.g. rows from mitbih_test.csv flattened)
    if n <= 5000 and n % TARGET_LENGTH == 0:
        windows, raw = process_already_segmented(signal)
        if windows is not None:
            return windows, raw, "pre-segmented beats", len(windows), None

    # Raw continuous ECG — use the provided sample rate
    windows, cleaned, r_peaks = process_raw_ecg(signal, input_fs)
    return windows, cleaned, "raw ECG recording", len(windows), r_peaks


# ─── Feature 2: Attention / saliency computation ─────────────────────────────

def compute_attention(beat, predicted_class):
    """
    Compute gradient-based saliency for a single beat.
    Returns a 1D array of length 187 with values in [0, 1].
    """
    beat_tensor = tf.constant(beat.reshape(1, TARGET_LENGTH, 1), dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(beat_tensor)
        predictions = MODEL(beat_tensor, training=False)
        target_class_score = predictions[0][predicted_class]
    grads = tape.gradient(target_class_score, beat_tensor)
    attention = tf.abs(grads).numpy().reshape(-1)
    att_min = attention.min()
    att_max = attention.max()
    if att_max - att_min > 1e-8:
        attention = (attention - att_min) / (att_max - att_min)
    else:
        attention = np.zeros_like(attention)
    # Smooth with rolling window of 5
    window = 5
    attention_smooth = np.convolve(attention, np.ones(window) / window, mode='same')
    # Re-normalize after smoothing
    att_min2 = attention_smooth.min()
    att_max2 = attention_smooth.max()
    if att_max2 - att_min2 > 1e-8:
        attention_smooth = (attention_smooth - att_min2) / (att_max2 - att_min2)
    return attention_smooth


# ─── Feature 1: Confirmation buffer logic ────────────────────────────────────

def compute_rr_features(windows, r_peaks=None, sampling_rate=360):
    """
    Compute RR interval features.
    If actual r_peaks are provided (from raw ECG), use real positions.
    Otherwise estimate from pre-segmented beat count.
    Returns RR intervals in milliseconds.
    """
    # Use actual R-peak positions if available
    if r_peaks is not None and len(r_peaks) >= 2:
        rr_samples = np.diff(np.array(r_peaks, dtype=float))
        rr_ms = (rr_samples / sampling_rate) * 1000  # Convert to ms
        return {
            'rr_mean': round(float(np.mean(rr_ms)), 1),
            'rr_std': round(float(np.std(rr_ms)), 1),
            'rr_min': round(float(np.min(rr_ms)), 1),
            'rr_max': round(float(np.max(rr_ms)), 1)
        }

    # Fallback for pre-segmented beats
    n_beats = windows.shape[0]
    if n_beats < 2:
        # Single beat — estimate ~830ms (72 BPM typical)
        return {'rr_mean': 830.0, 'rr_std': 0.0, 'rr_min': 830.0, 'rr_max': 830.0}

    # For pre-segmented MIT-BIH: each beat is 187 samples at ~360Hz ≈ 519ms
    estimated_rr = 1000.0 * TARGET_LENGTH / sampling_rate
    return {
        'rr_mean': round(estimated_rr, 1),
        'rr_std': 0.0,
        'rr_min': round(estimated_rr, 1),
        'rr_max': round(estimated_rr, 1)
    }


def update_confirmation_buffer(per_beat_classes):
    """
    Feed beat predictions into the rolling confirmation buffer.
    Returns the buffer state and rhythm label.
    """
    global CONFIRMATION_BUFFER

    for cls in per_beat_classes:
        CONFIRMATION_BUFFER.append(cls)

    buffer_list = list(CONFIRMATION_BUFFER)
    arr_count = sum(1 for c in buffer_list if c != 0)

    return buffer_list, arr_count


def compute_rhythm_label(arr_count, rr_std):
    """
    Generate rhythm label based on buffer state and RR variability.
    """
    if arr_count == 0:
        if rr_std >= 25:
            return "Irregular rhythm — possible AFib"
        elif rr_std >= 10:
            return "Mild RR variability"
        else:
            return "Regular sinus rhythm"
    elif arr_count < 3:
        return f"Occasional ectopic beats ({arr_count}/5)"
    else:
        return f"Sustained arrhythmia ({arr_count}/5 beats)"


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    global ALERT_COUNTER

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if not file.filename:
        return jsonify({'error': 'Empty filename'}), 400

    original_filename = file.filename

    # Optional: user can tell us the sample rate of their device
    input_fs = int(request.form.get('sample_rate', TARGET_FS))

    fname = secure_filename(f"{uuid.uuid4()}_{file.filename}")
    fpath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    file.save(fpath)

    try:
        # Load signal
        signal = load_csv(fpath)

        if len(signal) < 10:
            return jsonify({'error': 'File has too few data points.'}), 400

        # ── Feature 6: Improved auto-detect with sample rate handling ─────
        n = len(signal)
        is_presegmented = False
        actual_r_peaks = None

        if n <= 250:
            # Single beat
            windows, cleaned, mode, n_beats, actual_r_peaks = detect_mode_and_process(signal, input_fs)
            is_presegmented = True
        elif n <= 5000 and n % TARGET_LENGTH == 0:
            # Pre-segmented beats — route directly, ignore user sample rate
            windows, cleaned, mode, n_beats, actual_r_peaks = detect_mode_and_process(signal, input_fs)
            is_presegmented = True
        elif n > 1000:
            # Raw ECG
            if input_fs != TARGET_FS:
                windows, cleaned, actual_r_peaks = process_raw_ecg(signal, input_fs)
                n_beats = len(windows)
                mode = f"raw ECG @ {input_fs} Hz"
            else:
                windows, cleaned, mode, n_beats, actual_r_peaks = detect_mode_and_process(signal, input_fs)
        else:
            windows, cleaned, mode, n_beats, actual_r_peaks = detect_mode_and_process(signal, input_fs)

        # ── Predict ───────────────────────────────────────────────────────
        preds = MODEL.predict(windows, batch_size=128, verbose=0)
        per_beat_classes = np.argmax(preds, axis=1).tolist()
        avg_probs = preds.mean(axis=0)
        votes = Counter(per_beat_classes)
        final_class = votes.most_common(1)[0][0]

        # Build per-class vote breakdown
        vote_breakdown = {
            CLASS_NAMES[i]: int(votes.get(i, 0))
            for i in range(len(CLASS_NAMES))
        }

        # ── Feature 5: Beat-by-beat breakdown ─────────────────────────────
        beat_details = []
        for i in range(len(preds)):
            cls_idx = per_beat_classes[i]
            beat_details.append({
                "beat_num": i + 1,
                "class_id": int(cls_idx),
                "class_name": CLASS_NAMES[cls_idx],
                "confidence": round(float(preds[i][cls_idx]) * 100, 1),
                "is_arrhythmia": bool(cls_idx != 0)
            })

        arrhythmia_beat_count = sum(1 for b in beat_details if b['is_arrhythmia'])
        arrhythmia_percentage = round(
            (arrhythmia_beat_count / len(beat_details)) * 100, 1
        ) if len(beat_details) > 0 else 0.0

        # ── Feature 1: Confirmation buffer ────────────────────────────────
        buffer_list, arr_count = update_confirmation_buffer(per_beat_classes)

        # Compute RR features (use actual R-peaks if available)
        rr_features = compute_rr_features(windows, r_peaks=actual_r_peaks, sampling_rate=TARGET_FS)
        rr_std = rr_features['rr_std']

        # Rhythm label
        rhythm_label = compute_rhythm_label(arr_count, rr_std)

        # Confirmed arrhythmia: 3+ of last 5 beats are non-Normal
        confirmed_arrhythmia = arr_count >= 3

        # Buffer visualization: list of class IDs for last 5 beats
        buffer_classes = list(CONFIRMATION_BUFFER)

        # ── Feature 2: Attention heatmap ──────────────────────────────────
        # Find the most abnormal beat
        most_abnormal_idx = 0
        most_abnormal_conf = 0.0
        for i in range(len(preds)):
            cls = per_beat_classes[i]
            if cls != 0:
                conf = float(preds[i][cls])
                if conf > most_abnormal_conf:
                    most_abnormal_conf = conf
                    most_abnormal_idx = i

        # If all normal, use the first beat
        abnormal_beat = windows[most_abnormal_idx, :, 0]
        abnormal_class = per_beat_classes[most_abnormal_idx]

        # Compute attention
        attention_187 = compute_attention(
            windows[most_abnormal_idx],
            abnormal_class
        )
        # Resample attention to 500 points to match waveform_preview
        attention_500 = scipy_resample(attention_187, 500).astype(float)
        # Clamp to [0, 1]
        attention_500 = np.clip(attention_500, 0, 1)

        # Compute normal reference beat
        normal_indices = [i for i, c in enumerate(per_beat_classes) if c == 0]
        if len(normal_indices) > 0:
            normal_beats = windows[normal_indices, :, 0]
            mean_normal = normal_beats.mean(axis=0)
            normal_reference = scipy_resample(mean_normal, 500).astype(float).tolist()
        else:
            # Use synthetic template
            synth = np.array(SYNTHETIC_NORMAL_TEMPLATE, dtype=np.float32)
            normal_reference = scipy_resample(synth, 500).astype(float).tolist()

        # Compute abnormal beat index in waveform_preview space
        if n_beats > 0:
            abnormal_beat_index = int(most_abnormal_idx * (500 // max(n_beats, 1)))
            abnormal_beat_index = max(0, min(abnormal_beat_index, 450))
        else:
            abnormal_beat_index = 0

        # ── Feature 4: Clinical explanation ───────────────────────────────
        clinical = dict(CLINICAL_INFO[final_class])
        # Interpolate beats info
        clinical['explanation'] = clinical['explanation'].replace(
            'across the analyzed beats',
            f'across {n_beats} analyzed beat{"s" if n_beats != 1 else ""}'
        )
        if final_class == 2:
            clinical['explanation'] += (
                f" {arrhythmia_beat_count} out of {n_beats} beats showed this pattern."
            )
        elif final_class == 1:
            clinical['explanation'] += (
                f" {arrhythmia_beat_count} out of {n_beats} beats showed this pattern."
            )

        # ── Feature 3: Alert history ──────────────────────────────────────
        severity = (
            'HIGH RISK' if final_class == 2
            else 'MODERATE' if final_class in [1, 3]
            else 'NORMAL'
        )

        if confirmed_arrhythmia:
            ALERT_COUNTER += 1
            alert_entry = {
                'alert_id': ALERT_COUNTER,
                'timestamp': datetime.now().strftime("%H:%M:%S"),
                'filename': original_filename,
                'classification': CLASS_NAMES[final_class],
                'confidence': round(float(avg_probs[final_class]) * 100, 1),
                'severity': severity,
                'rhythm_label': rhythm_label,
                'beats_analyzed': int(n_beats),
                'arr_in_buffer': arr_count,
                'acknowledged': False
            }
            ALERT_HISTORY.append(alert_entry)

        # ── Waveform preview ──────────────────────────────────────────────
        waveform_preview = cleaned[:500].tolist()

        # ── Unified Interpretation Logic ──────────────────────────────────
        model_confidence = round(float(avg_probs[final_class]) * 100, 1)

        if model_confidence < 70.0:
            final_status_level = 'borderline'
            final_status_text = 'Borderline / Low Confidence Result'
        elif final_class != 0:
            # Model predicts a non-Normal class with high confidence
            final_status_level = 'abnormal'
            final_status_text = 'Irregular Rhythm Detected'
        elif arrhythmia_beat_count > 0:
            # Model predicts Normal overall, but some beats were irregular
            final_status_level = 'normal'
            final_status_text = 'Predominantly Normal Rhythm'
        else:
            final_status_level = 'normal'
            final_status_text = 'Predominantly Normal Rhythm'

        if final_class == 0:
            if arrhythmia_beat_count > 0:
                final_interpretation = "Mixed rhythm with intermittent irregularities."
            elif model_confidence < 70.0:
                final_interpretation = "Predominantly normal rhythm, but low model confidence requires clinical validation."
            else:
                final_interpretation = "High confidence normal sinus rhythm."
        else:
            if model_confidence < 70.0:
                final_interpretation = f"Possible {CLASS_NAMES[final_class]} detected, but low model confidence requires clinical validation."
            else:
                final_interpretation = f"High confidence detection of {CLASS_NAMES[final_class]}."

        # ── Reconcile flags with unified status ──────────────────────────
        # Arrhythmia label follows the MODEL's final class prediction:
        #   - Class 0 (Normal) → always "No", even if a few beats were irregular
        #   - Class != 0 + high confidence → "Yes"
        #   - Class != 0 + low confidence → "Inconclusive"
        # Confirmed label requires either:
        #   - High confidence non-Normal prediction (model is sure), OR
        #   - Buffer-based confirmation (3+ of last 5 beats abnormal)
        unified_is_arrhythmia = bool(final_class != 0)

        if final_class == 0:
            # Model says Normal — arrhythmia is No regardless of stray beats
            unified_is_arrhythmia_label = 'No'
            unified_confirmed_label = 'No'
        elif model_confidence < 70.0:
            # Model says non-Normal but isn't confident
            unified_is_arrhythmia_label = 'Inconclusive'
            unified_confirmed_label = 'Inconclusive'
        else:
            # Model says non-Normal AND is confident
            unified_is_arrhythmia_label = 'Yes'
            # Confirmed if EITHER model is confident OR buffer agrees
            unified_confirmed_label = 'Yes'

        # ── Build response ────────────────────────────────────────────────
        return jsonify({
            'prediction': CLASS_NAMES[final_class],
            'class_id': int(final_class),
            'is_arrhythmia': unified_is_arrhythmia,
            'is_arrhythmia_label': unified_is_arrhythmia_label,
            'severity': severity,
            'confidence': model_confidence,
            'final_status_level': final_status_level,
            'final_status_text': final_status_text,
            'final_interpretation': final_interpretation,
            'class_probabilities': {
                CLASS_NAMES[i]: round(float(avg_probs[i]) * 100, 1)
                for i in range(len(CLASS_NAMES))
            },
            'vote_breakdown': vote_breakdown,
            'beats_analyzed': int(n_beats),
            'signal_samples': int(len(signal)),
            'processing_mode': mode,
            'waveform_preview': waveform_preview,
            # Feature 1: Confirmation buffer
            'confirmed_arrhythmia': confirmed_arrhythmia,
            'confirmed_arrhythmia_label': unified_confirmed_label,
            'rhythm_label': rhythm_label,
            'arr_in_buffer': arr_count,
            'buffer_classes': buffer_classes,
            'rr_std': rr_features['rr_std'],
            'rr_mean': rr_features['rr_mean'],
            # Feature 2: Attention heatmap
            'attention_scores': attention_500.tolist(),
            'normal_reference': normal_reference,
            'abnormal_beat_index': abnormal_beat_index,
            # Attention explanation
            'attention_explanation': {
                'beat_index': most_abnormal_idx + 1,
                'beat_class': CLASS_NAMES[abnormal_class],
                'beat_confidence': round(float(preds[most_abnormal_idx][abnormal_class]) * 100, 1),
                'high_attention_pct': round(float(np.mean(attention_187 > 0.6)) * 100, 1),
                'peak_region': 'QRS complex' if np.argmax(attention_187) > 30 and np.argmax(attention_187) < 120 else ('P-wave region' if np.argmax(attention_187) <= 30 else 'T-wave region'),
                # Separate, clear explanation fields
                'red_region_explanation': "The red/orange shaded areas on the graph show where the AI focused most when making its decision. Darker red = stronger focus. These regions contain the signal features that most influenced the classification.",
                'beat_summary': f"The AI concentrated on beat #{most_abnormal_idx + 1}, which it classified as {CLASS_NAMES[abnormal_class]} with {round(float(preds[most_abnormal_idx][abnormal_class]) * 100, 1)}% confidence.",
                'waveform_note': CLINICAL_INFO[abnormal_class].get('waveform_note', ''),
                'clinical_note': CLINICAL_INFO[abnormal_class].get('title', ''),
            },
            # Feature 4: Clinical explanation
            'clinical_explanation': clinical,
            # Feature 5: Beat-by-beat
            'beat_details': beat_details,
            'arrhythmia_beat_count': arrhythmia_beat_count,
            'arrhythmia_percentage': arrhythmia_percentage,
        })

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500
    finally:
        if os.path.exists(fpath):
            os.remove(fpath)


# ─── Feature 3: Alert history endpoints ──────────────────────────────────────

@app.route('/alerts', methods=['GET'])
def get_alerts():
    return jsonify({'alerts': ALERT_HISTORY})


@app.route('/alerts', methods=['DELETE'])
def clear_alerts():
    global ALERT_COUNTER
    ALERT_HISTORY.clear()
    ALERT_COUNTER = 0
    return jsonify({'message': 'cleared'})


@app.route('/alerts/<int:alert_id>/ack', methods=['PUT'])
def acknowledge_alert(alert_id):
    for alert in ALERT_HISTORY:
        if alert['alert_id'] == alert_id:
            alert['acknowledged'] = True
            return jsonify({'message': 'ok'})
    return jsonify({'error': 'Alert not found'}), 404


@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
