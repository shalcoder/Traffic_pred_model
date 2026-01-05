"""
Premium UI Design System for Vehicle Collision Analysis Engine
Includes: Glassmorphism, Gradients, Animations, Custom Components
"""

def get_base_css():
    return """
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Reset & Base Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1e35 50%, #0f1419 100%) !important;
        background-attachment: fixed;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Enhanced Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(26, 29, 36, 0.95) 0%, rgba(14, 17, 23, 0.98) 100%);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #e0e0e0;
    }
    
    /* Glass-morphic Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 24px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .glass-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 48px rgba(0, 0, 0, 0.5);
        border-color: rgba(255, 75, 75, 0.3);
    }
    
    /* Premium Metric Cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(255, 75, 75, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        transition: all 0.3s ease;
    }
    
    [data-testid="stMetric"]:hover {
        border-color: rgba(255, 75, 75, 0.4);
        box-shadow: 0 4px 24px rgba(255, 75, 75, 0.2);
    }
    
    [data-testid="stMetric"] label {
        color: #a0a0a0 !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    
    [data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #ff4b4b 0%, #ff8a8a 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Gradient Headers */
    h1, h2, h3 {
        background: linear-gradient(135deg, #ffffff 0%, #b0b0b0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }
    
    /* Premium Button Styles */
    .stButton>button {
        background: linear-gradient(135deg, #ff4b4b 0%, #ff6b6b 100%);
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 12px 32px;
        font-weight: 600;
        font-size: 1rem;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 16px rgba(255, 75, 75, 0.4);
        transition: all 0.3s ease;
        text-transform: uppercase;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(255, 75, 75, 0.6);
        background: linear-gradient(135deg, #ff6b6b 0%, #ff8a8a 100%);
    }
    
    .stButton>button:active {
        transform: translateY(0);
    }
    
    /* Enhanced Input Fields */
    .stSelectbox, .stMultiSelect, .stTextInput {
        background: rgba(255, 255, 255, 0.03) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.02);
        padding: 8px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #a0a0a0;
        font-weight: 600;
        padding: 12px 24px;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(255, 75, 75, 0.2) 0%, rgba(139, 92, 246, 0.2) 100%);
        color: #ffffff !important;
        box-shadow: 0 4px 12px rgba(255, 75, 75, 0.3);
    }
    
    /* Divider Enhancement */
    hr {
        margin: 2rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, rgba(255, 255, 255, 0.1) 50%, transparent 100%);
    }
    
    /* Plotly Chart Container Enhancement */
    .js-plotly-plot {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* Animated Gradient Background Elements */
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .animated-gradient {
        background: linear-gradient(-45deg, #ff4b4b, #8b5cf6, #3b82f6, #10b981);
        background-size: 400% 400%;
        animation: gradient-shift 15s ease infinite;
    }
    
    /* Info/Warning/Success Box Enhancement */
    .stAlert {
        border-radius: 12px;
        border-left: 4px solid;
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
    }
    
    /* Spinner Enhancement */
    .stSpinner > div {
        border-top-color: #ff4b4b !important;
    }
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.02);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #ff4b4b 0%, #ff6b6b 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #ff6b6b 0%, #ff8a8a 100%);
    }
    </style>
    """

def create_hero_section(title, subtitle, icon="🚦"):
    return f"""
    <div style='text-align: center; padding: 40px 20px; background: linear-gradient(135deg, rgba(255, 75, 75, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%); border-radius: 20px; margin-bottom: 32px; border: 1px solid rgba(255, 255, 255, 0.05);'>
        <div style='font-size: 3rem; margin-bottom: 16px;'>{icon}</div>
        <h1 style='font-size: 2rem; margin: 0; background: linear-gradient(135deg, #ffffff 0%, #ff4b4b 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>{title}</h1>
        <p style='font-size: 1rem; color: #b0b0b0; margin-top: 12px; font-weight: 300;'>{subtitle}</p>
    </div>
    """

def create_stat_card(label, value, delta="", icon="📊"):
    delta_color = "#10b981" if not delta.startswith("-") else "#ef4444"
    return f"""
    <div class='glass-card' style='text-align: center;'>
        <div style='font-size: 2rem; margin-bottom: 10px;'>{icon}</div>
        <div style='font-size: 2rem; font-weight: 800; background: linear-gradient(135deg, #ff4b4b 0%, #ff8a8a 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>{value}</div>
        <div style='color: #a0a0a0; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-top: 8px;'>{label}</div>
        {f"<div style='color: {delta_color}; font-size: 0.9rem; font-weight: 600; margin-top: 6px;'>{delta}</div>" if delta else ""}
    </div>
    """

def create_feature_card(title, description, icon, link_text="Explore"):
    return f"""
    <div class='glass-card' style='height: 100%; display: flex; flex-direction: column; justify-content: space-between;'>
        <div>
            <div style='font-size: 2.5rem; margin-bottom: 14px;'>{icon}</div>
            <h3 style='margin: 0 0 10px 0; font-size: 1.3rem;'>{title}</h3>
            <p style='color: #b0b0b0; line-height: 1.6; font-size: 0.95rem;'>{description}</p>
        </div>
        <div style='margin-top: 16px; padding-top: 16px; border-top: 1px solid rgba(255, 255, 255, 0.1);'>
            <span style='color: #ff4b4b; font-weight: 600; font-size: 0.9rem;'>{link_text} →</span>
        </div>
    </div>
    """

def create_nav_bar(current_page):
    """Create a navigation bar with home button"""
    return f"""
    <div style='background: rgba(255, 255, 255, 0.02); padding: 12px 24px; 
                border-radius: 12px; margin-bottom: 24px; display: flex; 
                justify-content: space-between; align-items: center;
                border: 1px solid rgba(255, 255, 255, 0.05);'>
        <div style='display: flex; align-items: center; gap: 12px;'>
            <span style='font-size: 1.2rem;'>🏠</span>
            <span style='color: #666;'>/</span>
            <span style='color: #ff4b4b; font-weight: 600;'>{current_page}</span>
        </div>
        <div style='color: #888; font-size: 0.9rem;'>
            Vehicle Collision Analysis Engine
        </div>
    </div>
    """

def create_back_button():
    """Create a styled back to home button"""
    return """
    <style>
    .back-btn {
        display: inline-block;
        background: linear-gradient(135deg, rgba(255, 75, 75, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        color: #ff4b4b;
        padding: 10px 20px;
        border-radius: 8px;
        text-decoration: none;
        font-weight: 600;
        border: 1px solid rgba(255, 75, 75, 0.2);
        transition: all 0.3s ease;
        margin-bottom: 20px;
        font-size: 0.95rem;
    }
    .back-btn:hover {
        background: linear-gradient(135deg, rgba(255, 75, 75, 0.2) 0%, rgba(139, 92, 246, 0.2) 100%);
        border-color: rgba(255, 75, 75, 0.4);
        transform: translateX(-4px);
    }
    </style>
    """
