"""
Streamlit Frontend - Heart Disease Prediction System
Clean, minimal, medical-grade UI for healthcare AI
"""

import streamlit as st
import requests
import time
from typing import Dict, Any, Optional
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# API Configuration - Configurable via environment variable or config
import os
try:
    from config import API_BASE_URL, API_TIMEOUT, get_health_endpoint, get_predict_endpoint
    API_PREDICT_ENDPOINT = get_predict_endpoint()
    API_HEALTH_ENDPOINT = get_health_endpoint()
except ImportError:
    # Fallback if config.py is not available
    API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
    API_PREDICT_ENDPOINT = f"{API_BASE_URL}/predict"
    API_HEALTH_ENDPOINT = f"{API_BASE_URL}/health"
    API_TIMEOUT = int(os.getenv("API_TIMEOUT", "10"))  # seconds


# --- Design System (centralized tokens) ---
DESIGN_TOKENS = {
    "colors": {
        # Primary design palette (healthcare-friendly)
        "primary": "#0EA5E9",       # sky blue
        "primary_dark": "#0284C7",  # darker blue for hover
        "secondary": "#14B8A6",     # teal
        "background": "#F5F7FB",
        "surface": "#FFFFFF",
        "border_subtle": "#E2E8F0",
        "text_main": "#0F172A",
        "text_muted": "#64748B",
        # Semantic colors
        "success_bg": "#DCFCE7",
        "success_border": "#22C55E",
        "warning_bg": "#FEF9C3",
        "warning_border": "#FACC15",
        "danger_bg": "#FEE2E2",
        "danger_border": "#EF4444",
    },
    "radius": {
        "sm": "4px",
        "md": "8px",
        "lg": "12px",
        "pill": "999px",
    },
    "shadow": {
        "soft": "0 4px 10px rgba(15, 23, 42, 0.06)",
        "medium": "0 6px 18px rgba(15, 23, 42, 0.10)",
    },
    "spacing": {
        "xs": "0.5rem",
        "sm": "0.75rem",
        "md": "1rem",
        "lg": "1.5rem",
        "xl": "2rem",
    },
    "typography": {
        "font_family": "'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        "heading_weight": "600",
        "body_weight": "400",
        "h1_size": "2.4rem",
        "h2_size": "1.6rem",
        "body_size": "1rem",
    },
}


# Custom CSS for medical-grade styling using design tokens
def load_custom_css():
    """Load custom CSS for medical-grade, clinical UI"""
    c = DESIGN_TOKENS["colors"]
    r = DESIGN_TOKENS["radius"]
    s = DESIGN_TOKENS["shadow"]
    sp = DESIGN_TOKENS["spacing"]
    t = DESIGN_TOKENS["typography"]

    st.markdown(
        f"""
    <style>
        :root {{
            --color-primary: {c["primary"]};
            --color-primary-dark: {c["primary_dark"]};
            --color-secondary: {c["secondary"]};
            --color-bg: {c["background"]};
            --color-surface: {c["surface"]};
            --color-border-subtle: {c["border_subtle"]};
            --color-text-main: {c["text_main"]};
            --color-text-muted: {c["text_muted"]};
            --color-success-bg: {c["success_bg"]};
            --color-success-border: {c["success_border"]};
            --color-warning-bg: {c["warning_bg"]};
            --color-warning-border: {c["warning_border"]};
            --color-danger-bg: {c["danger_bg"]};
            --color-danger-border: {c["danger_border"]};

            --radius-sm: {r["sm"]};
            --radius-md: {r["md"]};
            --radius-lg: {r["lg"]};
            --radius-pill: {r["pill"]};

            --shadow-soft: {s["soft"]};
            --shadow-medium: {s["medium"]};

            --space-xs: {sp["xs"]};
            --space-sm: {sp["sm"]};
            --space-md: {sp["md"]};
            --space-lg: {sp["lg"]};
            --space-xl: {sp["xl"]};

            --font-family: {t["font_family"]};
            --font-weight-heading: {t["heading_weight"]};
            --font-weight-body: {t["body_weight"]};
            --font-size-h1: {t["h1_size"]};
            --font-size-h2: {t["h2_size"]};
            --font-size-body: {t["body_size"]};
        }}

        html, body, [class^="css"]  {{
            font-family: var(--font-family);
            background-color: var(--color-bg);
            color: var(--color-text-main);
        }}

        /* Main container styling */
        .main {{
            padding: var(--space-xl);
        }}
        
        /* Hero section */
        .hero-wrapper {{
            border-radius: var(--radius-lg);
            padding: var(--space-lg);
            background: linear-gradient(135deg, #0EA5E9, #14B8A6);
            color: #FFFFFF;
            margin-bottom: var(--space-lg);
            box-shadow: var(--shadow-medium);
        }}

        .hero-inner {{
            max-width: 1000px;
            margin: 0 auto;
        }}

        .main-title {{
            font-size: var(--font-size-h1);
            font-weight: var(--font-weight-heading);
            margin-bottom: 0.25rem;
            letter-spacing: -0.02em;
        }}
        
        .subtitle {{
            font-size: 1.05rem;
            opacity: 0.95;
            margin-bottom: 1rem;
            font-weight: 400;
        }}

        /* Generic card */
        .card {{
            background-color: var(--color-surface);
            border-radius: var(--radius-lg);
            padding: var(--space-lg);
            box-shadow: var(--shadow-soft);
            border: 1px solid var(--color-border-subtle);
        }}
        
        /* Form typography */
        .stNumberInput label, .stSelectbox label {{
            font-weight: 500;
            color: var(--color-text-main);
        }}

        .stNumberInput small, .stSelectbox small {{
            color: var(--color-text-muted);
        }}
        
        /* Primary & secondary Buttons */
        .stButton > button {{
            width: 100%;
            height: 3.1rem;
            font-size: 1.1rem;
            font-weight: 600;
            border-radius: var(--radius-md);
            border: 1px solid transparent;
            transition: all 0.2s ease;
        }}

        .btn-primary .stButton > button {{
            background-color: var(--color-primary);
            color: #ffffff;
        }}

        .btn-primary .stButton > button:hover {{
            background-color: var(--color-primary-dark);
            box-shadow: var(--shadow-medium);
            transform: translateY(-1px);
        }}

        .btn-secondary .stButton > button {{
            background-color: #ffffff;
            color: var(--color-primary);
            border-color: var(--color-border-subtle);
        }}

        .btn-secondary .stButton > button:hover {{
            background-color: #F3F4F6;
            box-shadow: var(--shadow-soft);
        }}
        
        /* Result card styling */
        .result-card {{
            padding: var(--space-lg);
            border-radius: var(--radius-lg);
            margin: var(--space-lg) 0;
            box-shadow: var(--shadow-soft);
        }}
        
        .result-card.low-risk {{
            background-color: var(--color-success-bg);
            border-left: 4px solid var(--color-success-border);
        }}
        
        .result-card.medium-risk {{
            background-color: var(--color-warning-bg);
            border-left: 4px solid var(--color-warning-border);
        }}
        
        .result-card.high-risk {{
            background-color: var(--color-danger-bg);
            border-left: 4px solid var(--color-danger-border);
        }}
        
        /* Prediction text styling */
        .prediction-text {{
            font-size: 1.6rem;
            font-weight: 600;
            margin: 0.5rem 0;
        }}
        
        .probability-text {{
            font-size: 1.2rem;
            font-weight: 500;
            margin: 0.25rem 0;
        }}
        
        /* Risk level badge */
        .risk-badge {{
            display: inline-block;
            padding: 0.35rem 1.25rem;
            border-radius: var(--radius-pill);
            font-size: 0.9rem;
            font-weight: 600;
            margin-top: 0.75rem;
        }}
        
        .risk-badge.low {{
            background-color: var(--color-success-border);
            color: #ffffff;
        }}
        
        .risk-badge.medium {{
            background-color: var(--color-warning-border);
            color: #1F2933;
        }}
        
        .risk-badge.high {{
            background-color: var(--color-danger-border);
            color: #ffffff;
        }}
        
        /* Alerts */
        .alert {{
            padding: var(--space-md);
            border-radius: var(--radius-md);
            margin: var(--space-sm) 0;
            border-left-width: 4px;
            border-left-style: solid;
            background-color: #ffffff;
        }}

        .alert-success {{
            background-color: var(--color-success-bg);
            border-left-color: var(--color-success-border);
        }}

        .alert-warning {{
            background-color: var(--color-warning-bg);
            border-left-color: var(--color-warning-border);
        }}

        .alert-danger {{
            background-color: var(--color-danger-bg);
            border-left-color: var(--color-danger-border);
        }}
        
        /* Form section headers */
        .section-header {{
            font-size: var(--font-size-h2);
            font-weight: var(--font-weight-heading);
            color: var(--color-text-main);
            margin: 2rem 0 1rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--color-border-subtle);
        }}
        
        /* Hide Streamlit default elements */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}

        /* Hero + highlights layout */
        .hero-badge {{
            display: inline-block;
            padding: 0.15rem 0.7rem;
            border-radius: var(--radius-pill);
            background-color: rgba(15, 23, 42, 0.18);
            color: #E5E7EB;
            font-size: 0.78rem;
            font-weight: 500;
            margin-right: 0.35rem;
            margin-bottom: 0.35rem;
        }}

        .hero-description {{
            font-size: 0.95rem;
            color: #E5E7EB;
            margin-bottom: var(--space-md);
        }}

        .highlights-row {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: var(--space-md);
            margin-top: var(--space-md);
        }}

        @media (max-width: 900px) {{
            .highlights-row {{
                grid-template-columns: 1fr;
            }}
        }}

        .highlight-card {{
            background-color: #F9FAFB;
            border-radius: var(--radius-md);
            padding: var(--space-md);
            border: 1px solid var(--color-border-subtle);
        }}

        .highlight-title {{
            font-weight: 600;
            margin-bottom: 0.25rem;
            font-size: 0.98rem;
        }}

        .highlight-text {{
            font-size: 0.9rem;
            color: var(--color-text-muted);
        }}

        .cta-row {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: var(--space-sm);
            margin-top: var(--space-md);
            flex-wrap: wrap;
        }}

        .cta-note {{
            font-size: 0.85rem;
            color: var(--color-text-muted);
        }}

        /* Explainability card */
        .xai-card {{
            margin-top: var(--space-md);
            padding: var(--space-lg);
            border-radius: var(--radius-lg);
            background-color: var(--color-surface);
            border: 1px solid var(--color-border-subtle);
            box-shadow: var(--shadow-soft);
        }}

        .xai-title {{
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: var(--space-sm);
        }}

        .xai-subtitle {{
            font-size: 0.9rem;
            color: var(--color-text-muted);
            margin-bottom: var(--space-md);
        }}

        .xai-feature-row {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0.4rem;
            gap: var(--space-sm);
        }}

        .xai-feature-label {{
            font-size: 0.9rem;
            color: #1e40af; /* dark blue for feature labels */
            flex: 0 0 45%;
        }}

        .xai-feature-bar-wrapper {{
            flex: 1;
            background-color: #E5E7EB;
            border-radius: var(--radius-pill);
            height: 0.5rem;
            overflow: hidden;
        }}

        .xai-feature-bar {{
            height: 100%;
            border-radius: var(--radius-pill);
            background: linear-gradient(90deg, #FDE68A, #DC2626);
        }}

        .xai-feature-strength {{
            min-width: 3rem;
            text-align: right;
            font-size: 0.8rem;
            color: var(--color-text-muted);
        }}
    </style>
    """,
        unsafe_allow_html=True,
    )


def check_api_health():
    """
    Check if backend API is running and return detailed status.
    
    Returns:
        (is_reachable, health_payload_or_none)
    """
    try:
        response = requests.get(API_HEALTH_ENDPOINT, timeout=2)
        if response.status_code == 200:
            data = response.json()
            return True, data
        # API responded but not with 200 – treat as reachable but unhealthy
        return True, {"status": "error", "message": f"Unexpected status code: {response.status_code}"}
    except Exception:
        # Completely unreachable
        return False, None


def make_prediction(patient_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Make prediction request to backend API
    
    Args:
        patient_data: Dictionary with patient medical features
        
    Returns:
        Prediction result dictionary or None if error
    """
    try:
        response = requests.post(
            API_PREDICT_ENDPOINT,
            json=patient_data,
            headers={"Content-Type": "application/json"},
            timeout=API_TIMEOUT
        )
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 422:
            # Validation error - show user-friendly message
            error_data = response.json()
            error_detail = error_data.get('detail', 'Validation error')
            if isinstance(error_detail, list):
                errors = [err.get('msg', str(err)) for err in error_detail]
                st.error("⚠️ **Validation Error:**")
                for err in errors:
                    st.error(f"  - {err}")
            else:
                st.error(f"⚠️ **Validation Error:** {error_detail}")
            return None
        elif response.status_code == 503:
            st.error("⚠️ **Service Unavailable:** Model is not loaded. Please check backend logs.")
            return None
        else:
            st.error(f"❌ **API Error {response.status_code}:**")
            try:
                error_data = response.json()
                st.error(f"  {error_data.get('detail', response.text)}")
            except:
                st.error(f"  {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        st.error("""
        ❌ **Cannot connect to API server**
        
        Please ensure the backend is running:
        ```bash
        python run_api.py
        ```
        
        Or check if the API URL is correct:
        - Current API URL: `{}`
        - Set custom URL: `export API_BASE_URL=http://your-url:8000`
        """.format(API_BASE_URL))
        return None
    except requests.exceptions.Timeout:
        st.error(f"⏱️ **Request timed out** (>{API_TIMEOUT}s). The server may be slow or overloaded. Please try again.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"❌ **Network Error:** {str(e)}")
        return None
    except Exception as e:
        st.error(f"❌ **Unexpected Error:** {str(e)}")
        return None


def get_risk_color_class(risk_level: str) -> str:
    """Get CSS class for risk level"""
    risk_map = {
        "Low": "low-risk",
        "Medium": "medium-risk",
        "High": "high-risk"
    }
    return risk_map.get(risk_level, "low-risk")


def get_risk_badge_class(risk_level: str) -> str:
    """Get CSS class for risk badge"""
    risk_map = {
        "Low": "low",
        "Medium": "medium",
        "High": "high"
    }
    return risk_map.get(risk_level, "low")


def display_result(result: Dict[str, Any]):
    """Display prediction result with styling"""
    prediction = result['prediction']
    probability = result['probability']
    risk_level = result['risk_level']
    prediction_text = result['prediction_text']
    
    # Determine color class
    risk_class = get_risk_color_class(risk_level)
    badge_class = get_risk_badge_class(risk_level)
    
    # Display result card
    st.markdown(f"""
    <div class="result-card {risk_class}">
        <div class="prediction-text">
            {'❤️ Heart Disease Detected' if prediction == 1 else '✅ No Heart Disease'}
        </div>
        <div class="probability-text">
            Probability: {probability:.2%}
        </div>
        <div>
            {prediction_text}
        </div>
        <div class="risk-badge {badge_class}">
            Risk Level: {risk_level}
        </div>
    </div>
    """, unsafe_allow_html=True)


def build_feature_contributions(patient_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Simple heuristic feature "contribution" scores based on known risk patterns.
    This is not a true model explanation, but a faculty-friendly indication
    of which entered values are most concerning.
    """
    scores: Dict[str, float] = {}

    age = patient_data.get("age", 0)
    scores["Age"] = min(max((age - 40) / 40.0, 0.0), 1.0)  # higher after ~40

    trestbps = patient_data.get("trestbps", 0)
    scores["Resting blood pressure (mmHg)"] = min(max((trestbps - 120) / 60.0, 0.0), 1.0)

    chol = patient_data.get("chol", 0)
    scores["Cholesterol (mg/dL)"] = min(max((chol - 200) / 150.0, 0.0), 1.0)

    thalach = patient_data.get("thalach", 0)
    # Lower max heart rate under stress may indicate higher risk
    scores["Max heart rate (bpm)"] = min(max((170 - thalach) / 60.0, 0.0), 1.0)

    oldpeak = patient_data.get("oldpeak", 0.0)
    scores["ST depression (oldpeak)"] = min(max(oldpeak / 3.0, 0.0), 1.0)

    exang = patient_data.get("exang", 0)
    scores["Exercise-induced angina"] = 1.0 if exang == 1 else 0.0

    fbs = patient_data.get("fbs", 0)
    scores["High fasting blood sugar"] = 0.7 if fbs == 1 else 0.0

    return scores


def render_explainability(result: Dict[str, Any], patient_data: Dict[str, Any]):
    """Render a simple, faculty-friendly explainability section beneath the result."""
    scores = build_feature_contributions(patient_data)
    # Sort by score descending and take top 4
    top_features = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:4]

    risk_level = result.get("risk_level", "Low")
    prob = result.get("probability", 0.0)
    pred = result.get("prediction", 0)

    summary_text = ""
    if pred == 1:
        summary_text = (
            "The model flagged an increased risk because several of your measurements "
            "are in ranges that are commonly associated with heart strain or blocked arteries."
        )
    else:
        summary_text = (
            "The model did not detect strong signs of heart disease based on the values you entered. "
            "Most of your measurements fall in ranges usually seen in lower-risk patients."
        )

    st.markdown(
        """
        <div class="xai-card">
            <div class="xai-title">Why this result?</div>
            <div class="xai-subtitle">
                This explanation shows which of your entered values most influenced the risk assessment,
                using simple medical rules of thumb. It is meant for teaching and should not replace a full clinical workup.
            </div>
        """,
        unsafe_allow_html=True,
    )

    for label, score in top_features:
        width_pct = int(score * 100)
        # Color-code contribution strength
        if width_pct >= 30:
            contrib_color = "#dc2626"  # red - high
        elif width_pct >= 10:
            contrib_color = "#ca8a04"  # yellow - medium
        else:
            contrib_color = "#15803d"  # green - low

        st.markdown(
            f"""
            <div class="xai-feature-row">
                <div class="xai-feature-label" style="color:#1e40af;">{label}</div>
                <div class="xai-feature-bar-wrapper">
                    <div class="xai-feature-bar" style="width: {width_pct}%;"></div>
                </div>
                <div class="xai-feature-strength" style="color:{contrib_color};">{width_pct}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div style="
            margin-top: 0.75rem;
            padding: 0.75rem 1rem;
            border-left: 4px solid #0284c7;
            background-color: #f0f9ff;
            border-radius: 8px;
            color: #0f172a;
            font-size: 0.9rem;
        ">
            <strong>In plain language:</strong> {summary_text}
            <br/>
            <span style="font-size: 0.85rem;">
                Current risk category: <strong>{risk_level}</strong>, probability {prob:.0%}.
            </span>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_section_header(title: str):
    """Reusable section header component"""
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def render_step_progress(step: int, total_steps: int = 3):
    """Render simple step progress indicator"""
    st.write(f"**Step {step} of {total_steps}**")
    st.progress(float(step) / float(total_steps))


def primary_button(label: str, key: Optional[str] = None):
    """
    Reusable primary button component.
    Wrapped in a container so CSS can target `.btn-primary .stButton > button`.
    """
    st.markdown('<div class="btn-primary">', unsafe_allow_html=True)
    clicked = st.form_submit_button(label, use_container_width=True, key=key)
    st.markdown("</div>", unsafe_allow_html=True)
    return clicked


def main():
    """Main Streamlit application"""
    # Load custom CSS
    load_custom_css()

    # Ensure session state for CTA and multi-step flow
    if "show_form" not in st.session_state:
        st.session_state["show_form"] = False
    if "assessment_step" not in st.session_state:
        st.session_state["assessment_step"] = 1
    if "backend_status" not in st.session_state:
        st.session_state["backend_status"] = "Unknown"
    if "backend_message" not in st.session_state:
        st.session_state["backend_message"] = "Status not checked yet."

    # Landing page hero (no API calls)
    st.markdown(
        """
    <div class="hero-wrapper">
        <div class="hero-inner">
            <div class="main-title">
                ❤️ Heart Disease Prediction System
            </div>
            <div class="subtitle">
                Clinical-grade risk assessment using privacy-preserving federated learning.
            </div>
            <div>
                <span class="hero-badge">Healthcare AI</span>
                <span class="hero-badge">Federated Learning</span>
            </div>
            <div class="hero-description">
                This dashboard demonstrates how a deep learning model can estimate heart disease risk while 
                keeping raw patient data local to each site. Only model updates are shared, not identities.
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # System status strip (manual refresh, no API on initial load)
    status_col1, status_col2, status_col3 = st.columns([1.4, 1.4, 2])
    with status_col1:
        st.markdown("**System status**")
        current = st.session_state["backend_status"]
        if current == "healthy":
            st.success("Backend: Running")
        elif current == "degraded":
            st.warning("Backend: Degraded")
        elif current == "error":
            st.error("Backend: Unreachable")
        else:
            st.info("Backend: Unknown")
    with status_col2:
        if st.button("Refresh status"):
            reachable, health = check_api_health()
            if not reachable:
                st.session_state["backend_status"] = "error"
                st.session_state["backend_message"] = "API is not reachable. Please ensure the backend is running."
            else:
                st.session_state["backend_status"] = (health or {}).get("status", "degraded")
                st.session_state["backend_message"] = (health or {}).get(
                    "message", "API responded but status is not fully healthy."
                )
            st.rerun()
    with status_col3:
        st.caption(st.session_state["backend_message"])

    st.markdown("---")

    # CTA button (after hero & status)
    cta_col1, cta_col2, _ = st.columns([1.1, 2, 1])
    with cta_col1:
        st.markdown('<div class="btn-primary">', unsafe_allow_html=True)
        cta_clicked = st.button("Start Risk Assessment", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with cta_col2:
        st.markdown(
            '<p class="cta-note">Begin by entering basic demographic and test results. '
            "No real patient identifiers are required for this demo.</p>",
            unsafe_allow_html=True,
        )

    if cta_clicked:
        st.session_state["show_form"] = True
    
    # Check API health
    with st.spinner("Checking API connection..."):
        api_reachable, health = check_api_health()
    
    if not api_reachable:
        st.error("""
        ⚠️ **Backend API is not reachable**
        
        Please start the API server first:
        ```bash
        python run_api.py
        ```
        """)
        st.stop()
    
    # If reachable but degraded, surface the backend message clearly
    backend_status = (health or {}).get("status", "unknown")
    backend_message = (health or {}).get("message", "")
    if backend_status != "healthy":
        st.warning(f"Backend status: **{backend_status}** – {backend_message or 'Model or database may not be fully initialized.'}")
    else:
        st.success("✅ Backend API is connected and ready!")
    
    # Show assessment form only after CTA, keeping landing concise
    if not st.session_state["show_form"]:
        return

    # Multi-step assessment flow
    step = st.session_state["assessment_step"]
    total_steps = 3

    render_section_header("Risk Assessment")
    render_step_progress(step, total_steps=total_steps)

    # Step 1: Basic Information
    if step == 1:
        render_section_header("Step 1 · Basic Information")
        age = st.number_input(
            "Age (years)",
            min_value=1,
            max_value=120,
            value=st.session_state.get("age", 50),
            step=1,
            help="Most clinical heart risk tools are calibrated for adults between 18–90 years."
        )
        sex = st.selectbox(
            "Gender",
            options=[0, 1],
            index=st.session_state.get("sex", 1),
            format_func=lambda x: "Female" if x == 0 else "Male",
            help="Biological sex is a key factor in baseline cardiovascular risk."
        )
        st.caption("These values set your baseline risk profile before adding detailed clinical measurements.")

        st.markdown('<div class="btn-primary">', unsafe_allow_html=True)
        next_step_1 = st.button("Next: Clinical Measurements", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if next_step_1:
            # Inline validation: keep within realistic adult range
            if age < 18 or age > 90:
                st.error("For this prototype, please enter an age between 18 and 90 years.")
            else:
                st.session_state["age"] = int(age)
                st.session_state["sex"] = int(sex)
                st.session_state["assessment_step"] = 2
                st.rerun()
        return

    # Step 2: Clinical Measurements
    if step == 2:
        render_section_header("Step 2 · Clinical Measurements")
        trestbps = st.number_input(
            "Resting Blood Pressure (mmHg)",
            min_value=50,
            max_value=250,
            value=st.session_state.get("trestbps", 120),
            step=1,
            help="Typical normal resting systolic pressure is around 120 mmHg; values above 140 mmHg suggest hypertension."
        )
        chol = st.number_input(
            "Serum Cholesterol (mg/dL)",
            min_value=100,
            max_value=600,
            value=st.session_state.get("chol", 200),
            step=1,
            help="Desirable total cholesterol is usually below 200 mg/dL."
        )
        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dL",
            options=[0, 1],
            index=st.session_state.get("fbs", 0),
            format_func=lambda x: "No (≤ 120 mg/dL)" if x == 0 else "Yes (> 120 mg/dL)",
            help="This is a proxy for elevated fasting glucose; 'Yes' may indicate diabetes or pre-diabetes."
        )
        restecg = st.selectbox(
            "Resting ECG Results",
            options=[0, 1, 2],
            index=st.session_state.get("restecg", 0),
            format_func=lambda x: {
                0: "Normal",
                1: "ST-T wave abnormality",
                2: "Left ventricular hypertrophy"
            }.get(x, "Unknown"),
            help="Baseline electrical activity of the heart from a 12-lead ECG."
        )
        thalach = st.number_input(
            "Maximum Heart Rate Achieved (bpm)",
            min_value=50,
            max_value=250,
            value=st.session_state.get("thalach", 150),
            step=1,
            help="Typically approximated as 220 − age; much lower values under stress testing may indicate ischemia."
        )
        st.caption("These measurements describe your cardiovascular status at rest and under exercise.")

        col_back, col_next = st.columns(2)
        with col_back:
            back_1 = st.button("Back to Basic Information", use_container_width=True)
        with col_next:
            st.markdown('<div class="btn-primary">', unsafe_allow_html=True)
            next_step_2 = st.button("Next: Lifestyle & History", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        if back_1:
            st.session_state["assessment_step"] = 1
            st.rerun()

        if next_step_2:
            # Inline validation with simple clinical bounds
            errors = []
            if trestbps < 70 or trestbps > 220:
                errors.append("Resting blood pressure should typically be between 70 and 220 mmHg for this tool.")
            if chol < 100 or chol > 600:
                errors.append("Cholesterol should be between 100 and 600 mg/dL for valid assessment.")

            if errors:
                for e in errors:
                    st.error(e)
            else:
                st.session_state["trestbps"] = int(trestbps)
                st.session_state["chol"] = int(chol)
                st.session_state["fbs"] = int(fbs)
                st.session_state["restecg"] = int(restecg)
                st.session_state["thalach"] = int(thalach)
                st.session_state["assessment_step"] = 3
                st.rerun()
        return

    # Step 3: Lifestyle & History (plus exercise findings)
    if step == 3:
        render_section_header("Step 3 · Lifestyle & History")

        # Lifestyle/history fields (not yet used by model, but important for UX)
        smoking = st.selectbox(
            "Current smoking status",
            options=["Never smoked", "Former smoker", "Current smoker"],
            index=["Never smoked", "Former smoker", "Current smoker"].index(
                st.session_state.get("smoking", "Never smoked")
            ),
            help="Smoking significantly increases long-term cardiovascular risk."
        )
        diabetes_history = st.selectbox(
            "History of diabetes",
            options=["No", "Yes"],
            index=["No", "Yes"].index(st.session_state.get("diabetes_history", "No")),
            help="Diabetes is a major risk factor for coronary artery disease."
        )
        family_history = st.selectbox(
            "Family history of early heart disease",
            options=["No", "Yes"],
            index=["No", "Yes"].index(st.session_state.get("family_history", "No")),
            help="First-degree relatives with early heart disease increase inherited risk."
        )

        st.markdown("### Exercise & Imaging Findings")
        cp = st.selectbox(
            "Chest Pain Type",
            options=[0, 1, 2, 3],
            index=st.session_state.get("cp", 0),
            format_func=lambda x: {
                0: "Typical angina",
                1: "Atypical angina",
                2: "Non-anginal pain",
                3: "Asymptomatic"
            }.get(x, "Unknown"),
            help="Symptom pattern during exertion helps differentiate stable angina from other causes of chest pain."
        )
        exang = st.selectbox(
            "Exercise Induced Angina",
            options=[0, 1],
            index=st.session_state.get("exang", 0),
            format_func=lambda x: "No" if x == 0 else "Yes",
            help="Chest pain provoked by exercise suggests possible ischemia."
        )
        oldpeak = st.number_input(
            "ST Depression (Oldpeak)",
            min_value=0.0,
            max_value=10.0,
            value=st.session_state.get("oldpeak", 0.0),
            step=0.1,
            format="%.1f",
            help="ST segment depression measured in mm; higher values may indicate more severe ischemia."
        )
        slope = st.selectbox(
            "Slope of Peak Exercise ST Segment",
            options=[0, 1, 2],
            index=st.session_state.get("slope", 1),
            format_func=lambda x: {
                0: "Upsloping",
                1: "Flat",
                2: "Downsloping"
            }.get(x, "Unknown"),
            help="Downsloping ST segments are more strongly associated with ischemia."
        )
        ca = st.selectbox(
            "Number of Major Vessels (0–3)",
            options=[0, 1, 2, 3],
            index=st.session_state.get("ca", 0),
            help="Number of major coronary vessels colored by fluoroscopy."
        )
        thal = st.selectbox(
            "Thalassemia (Thal) status",
            options=[0, 1, 2, 3],
            index=st.session_state.get("thal", 0),
            format_func=lambda x: {
                0: "Normal",
                1: "Fixed defect",
                2: "Reversible defect",
                3: "Unknown"
            }.get(x, "Unknown"),
            help="Nuclear imaging findings related to perfusion defects."
        )

        st.caption("Lifestyle and history are captured for context; the current model focuses on the classic clinical features above.")

        col_back2, col_submit = st.columns(2)
        with col_back2:
            back_2 = st.button("Back to Clinical Measurements", use_container_width=True)
        with col_submit:
            st.markdown('<div class="btn-primary">', unsafe_allow_html=True)
            do_predict = st.button("🔍 Run Risk Assessment", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        if back_2:
            st.session_state["assessment_step"] = 2
            st.rerun()

        if do_predict:
            # Persist contextual fields (not yet used in model)
            st.session_state["smoking"] = smoking
            st.session_state["diabetes_history"] = diabetes_history
            st.session_state["family_history"] = family_history
            st.session_state["cp"] = int(cp)
            st.session_state["exang"] = int(exang)
            st.session_state["oldpeak"] = float(oldpeak)
            st.session_state["slope"] = int(slope)
            st.session_state["ca"] = int(ca)
            st.session_state["thal"] = int(thal)

            # Final inline validation for numeric fields already constrained above
            patient_data = {
                "age": int(st.session_state["age"]),
                "sex": int(st.session_state["sex"]),
                "cp": int(st.session_state["cp"]),
                "trestbps": int(st.session_state["trestbps"]),
                "chol": int(st.session_state["chol"]),
                "fbs": int(st.session_state["fbs"]),
                "restecg": int(st.session_state["restecg"]),
                "thalach": int(st.session_state["thalach"]),
                "exang": int(st.session_state["exang"]),
                "oldpeak": float(st.session_state["oldpeak"]),
                "slope": int(st.session_state["slope"]),
                "ca": int(st.session_state["ca"]),
                "thal": int(st.session_state["thal"]),
            }

            with st.spinner("🔄 Analyzing patient data and making prediction..."):
                time.sleep(0.5)
                result = make_prediction(patient_data)

            if result:
                render_section_header("Prediction Result")
                display_result(result)
                with st.expander("📊 View Detailed Information"):
                    st.json(result)
                    st.info(f"Record ID: {result.get('record_id', 'N/A')}")

                # Explainability section (faculty-friendly, non-technical)
                render_explainability(result, patient_data)


if __name__ == "__main__":
    main()

