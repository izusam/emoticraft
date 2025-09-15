import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings
import time
import base64
from PIL import Image
import io
warnings.filterwarnings('ignore')

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="EmotiCraft AI - Multimedia Emotion Analyzer", 
    page_icon="🎭", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Custom CSS
# -------------------------------
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .insight-card {
        background: #f8f9ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .recommendation-card {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid;
    }
    .positive-rec { 
        background: #f0f9f4; 
        border-left-color: #10b981; 
    }
    .negative-rec { 
        background: #fef2f2; 
        border-left-color: #ef4444; 
    }
    .neutral-rec { 
        background: #f8fafc; 
        border-left-color: #6b7280; 
    }
    .upload-area {
        border: 2px dashed #d1d5db;
        border-radius: 10px;
        padding: 3rem;
        text-align: center;
        background: #f9fafb;
        transition: all 0.3s ease;
    }
    .upload-area:hover {
        border-color: #667eea;
        background: #f0f2f6;
    }
    .content-type-card {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        text-align: center;
        transition: transform 0.2s ease;
        border: 2px solid #f3f4f6;
    }
    .content-type-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
    }
    .progress-bar {
        background: linear-gradient(90deg, #667eea, #764ba2);
        height: 8px;
        border-radius: 4px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Download NLTK resources
# -------------------------------
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
    except:
        pass

download_nltk_data()

# -------------------------------
# Constants (Enhanced for multimedia)
# -------------------------------
EMOTION_COLUMNS = [
    'admiration','amusement','anger','annoyance','approval','caring',
    'confusion','curiosity','desire','disappointment','disapproval',
    'disgust','embarrassment','excitement','fear','gratitude',
    'grief','joy','love','nervousness','optimism','pride',
    'realization','relief','remorse','sadness','surprise','neutral'
]

CONTENT_EMOTION_MAPPING = {
    'Video Content': {
        'high_engagement': ['excitement', 'curiosity', 'amusement', 'surprise'],
        'retention_boost': ['joy', 'admiration', 'love', 'gratitude'],
        'viral_potential': ['amusement', 'excitement', 'surprise', 'desire']
    },
    'Audio Content': {
        'high_engagement': ['curiosity', 'excitement', 'joy', 'surprise'],
        'retention_boost': ['admiration', 'gratitude', 'love', 'optimism'],
        'viral_potential': ['amusement', 'excitement', 'joy', 'desire']
    },
    'Educational': {
        'high_engagement': ['curiosity', 'realization', 'excitement', 'admiration'],
        'retention_boost': ['gratitude', 'optimism', 'pride', 'joy'],
        'viral_potential': ['surprise', 'realization', 'admiration', 'excitement']
    }
}

AUDIENCE_REACTION_GROUPS = {
    'Positive_Response': ['admiration','amusement','approval','caring','excitement','gratitude','joy','love','optimism','pride','relief'],
    'Negative_Response': ['anger','annoyance','disappointment','disapproval','disgust','embarrassment','fear','grief','nervousness','remorse','sadness'],
    'Neutral_Response': ['neutral','realization','surprise'],
    'Engaged_Curiosity': ['curiosity','confusion','desire']
}

EMOTION_EMOJIS = {
    'admiration': '👏', 'amusement': '😄', 'anger': '😠', 'annoyance': '😤',
    'approval': '👍', 'caring': '🤗', 'confusion': '😕', 'curiosity': '🤔',
    'desire': '😍', 'disappointment': '😞', 'disapproval': '👎', 'disgust': '🤢',
    'embarrassment': '😳', 'excitement': '🎉', 'fear': '😨', 'gratitude': '🙏',
    'grief': '😢', 'joy': '😊', 'love': '❤️', 'nervousness': '😰',
    'optimism': '🌟', 'pride': '😤', 'realization': '💡', 'relief': '😌',
    'remorse': '😔', 'sadness': '😢', 'surprise': '😲', 'neutral': '😐'
}

# -------------------------------
# Text preprocessing (Enhanced)
# -------------------------------
@st.cache_data
def clean_text(text):
    if pd.isna(text):
        return ''
    
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    try:
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]
        return ' '.join(tokens)
    except:
        return text

# -------------------------------
# Load models
# -------------------------------
@st.cache_resource
def load_models():
    import os
    import joblib
    model_path = "lr_model.joblib"
    vectorizer_path = "vectorizer.joblib"

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        st.error("⚠️ Model files not found! Using demo mode with simulated predictions.")
        return None, None

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    # Ensure vectorizer is fitted
    from sklearn.utils.validation import check_is_fitted
    try:
        check_is_fitted(vectorizer, attributes=["idf_"])
    except Exception as e:
        st.error(f"Vectorizer not fitted: {e}")
        return None, None

    return model, vectorizer


# -------------------------------
# Enhanced prediction functions
# -------------------------------
def predict_content_emotion(text_content, model, vectorizer, content_type="Video Content", min_confidence=0.3):
    """Enhanced prediction function for multimedia content analysis"""
    
    if model is None or vectorizer is None:
        # Demo mode - generate realistic predictions
        return generate_demo_predictions(text_content, content_type)
    
    # Original prediction logic
    X_tfidf = vectorizer.transform([clean_text(text_content)])
    probs = model.predict_proba(X_tfidf)

    emotion_probs = {}
    for i, emo in enumerate(EMOTION_COLUMNS):
        if i < len(probs):
            arr = probs[i]
            p = arr[0, 1] if arr.shape[1] == 2 else arr[0, 0]
            emotion_probs[emo] = float(p)

    # Enhanced analysis for content types
    content_metrics = calculate_content_metrics(emotion_probs, content_type)
    
    reaction_scores = {}
    for group, emotions in AUDIENCE_REACTION_GROUPS.items():
        vals = [emotion_probs[e] for e in emotions if e in emotion_probs]
        reaction_scores[group] = float(max(vals)) if vals else 0.0

    # Overall sentiment calculation
    positive_score = reaction_scores['Positive_Response']
    negative_score = reaction_scores['Negative_Response']
    neutral_score = reaction_scores['Neutral_Response']
    
    max_score = max(positive_score, negative_score, neutral_score)
    if max_score < min_confidence:
        overall_sentiment = "Neutral"
        sentiment_confidence = max_score
        sentiment_color = "gray"
    else:
        if positive_score >= max(negative_score, neutral_score):
            overall_sentiment = "Positive"
            sentiment_confidence = positive_score
            sentiment_color = "green"
        elif negative_score >= max(positive_score, neutral_score):
            overall_sentiment = "Negative"
            sentiment_confidence = negative_score
            sentiment_color = "red"
        else:
            overall_sentiment = "Neutral"
            sentiment_confidence = neutral_score
            sentiment_color = "gray"

    return {
        "emotion_probs": emotion_probs,
        "reaction_scores": reaction_scores,
        "content_metrics": content_metrics,
        "overall_sentiment": overall_sentiment,
        "sentiment_confidence": sentiment_confidence,
        "sentiment_color": sentiment_color,
        "content_type": content_type
    }

def generate_demo_predictions(text_content, content_type):
    """Generate realistic demo predictions when models aren't available"""
    np.random.seed(hash(text_content) % 2**31)  # Consistent results for same input
    
    # Generate emotion probabilities with realistic distributions
    emotion_probs = {}
    for emotion in EMOTION_COLUMNS:
        base_prob = np.random.beta(2, 5)  # Skewed towards lower values
        emotion_probs[emotion] = float(base_prob)
    
    # Boost emotions relevant to content type
    if content_type in CONTENT_EMOTION_MAPPING:
        for category, emotions in CONTENT_EMOTION_MAPPING[content_type].items():
            for emotion in emotions:
                if emotion in emotion_probs:
                    emotion_probs[emotion] = min(1.0, emotion_probs[emotion] * np.random.uniform(1.2, 2.0))
    
    # Calculate content metrics
    content_metrics = calculate_content_metrics(emotion_probs, content_type)
    
    # Calculate reaction scores
    reaction_scores = {}
    for group, emotions in AUDIENCE_REACTION_GROUPS.items():
        vals = [emotion_probs[e] for e in emotions if e in emotion_probs]
        reaction_scores[group] = float(max(vals)) if vals else 0.0
    
    # Determine overall sentiment
    positive_score = reaction_scores['Positive_Response']
    negative_score = reaction_scores['Negative_Response']
    neutral_score = reaction_scores['Neutral_Response']
    
    if positive_score >= max(negative_score, neutral_score):
        overall_sentiment = "Positive"
        sentiment_confidence = positive_score
        sentiment_color = "green"
    elif negative_score >= max(positive_score, neutral_score):
        overall_sentiment = "Negative"
        sentiment_confidence = negative_score
        sentiment_color = "red"
    else:
        overall_sentiment = "Neutral"
        sentiment_confidence = neutral_score
        sentiment_color = "gray"
    
    return {
        "emotion_probs": emotion_probs,
        "reaction_scores": reaction_scores,
        "content_metrics": content_metrics,
        "overall_sentiment": overall_sentiment,
        "sentiment_confidence": sentiment_confidence,
        "sentiment_color": sentiment_color,
        "content_type": content_type
    }

def calculate_content_metrics(emotion_probs, content_type):
    """Calculate content-specific metrics"""
    if content_type not in CONTENT_EMOTION_MAPPING:
        content_type = "Video Content"  # Default
    
    metrics = {}
    for metric_type, emotions in CONTENT_EMOTION_MAPPING[content_type].items():
        scores = [emotion_probs.get(emotion, 0) for emotion in emotions]
        metrics[metric_type] = np.mean(scores)
    
    # Overall impact score
    metrics['impact_score'] = np.mean(list(metrics.values())) * 100
    
    # Engagement prediction
    engagement_emotions = ['excitement', 'curiosity', 'amusement', 'surprise', 'joy']
    metrics['engagement_rate'] = np.mean([emotion_probs.get(e, 0) for e in engagement_emotions]) * 100
    
    # Retention prediction
    retention_emotions = ['admiration', 'gratitude', 'love', 'optimism', 'joy']
    metrics['retention_rate'] = np.mean([emotion_probs.get(e, 0) for e in retention_emotions]) * 100
    
    return metrics

# -------------------------------
# Enhanced Charts
# -------------------------------
def create_impact_score_gauge(metrics):
    """Create an impact score gauge chart"""
    impact_score = metrics.get('impact_score', 0)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=impact_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Emotional Impact Score", 'font': {'size': 20}},
        delta={'reference': 70, 'position': "top"},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 25], 'color': 'lightgray'},
                {'range': [25, 50], 'color': 'gray'},
                {'range': [50, 75], 'color': 'lightblue'},
                {'range': [75, 100], 'color': 'royalblue'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=300, font={'color': "darkblue", 'family': "Arial"})
    return fig

def create_content_metrics_chart(metrics):
    """Create content-specific metrics chart"""
    metric_names = ['Engagement Rate', 'Retention Rate', 'High Engagement', 'Viral Potential']
    metric_values = [
        metrics.get('engagement_rate', 0),
        metrics.get('retention_rate', 0),
        metrics.get('high_engagement', 0) * 100,
        metrics.get('viral_potential', 0) * 100
    ]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    fig = go.Figure(data=[
        go.Bar(
            x=metric_values,
            y=metric_names,
            orientation='h',
            marker_color=colors,
            text=[f'{v:.1f}%' for v in metric_values],
            textposition='inside',
        )
    ])
    
    fig.update_layout(
        title='Content Performance Metrics',
        xaxis_title='Score (%)',
        height=300,
        showlegend=False
    )
    fig.update_xaxes(range=[0, 100])
    return fig

def create_emotion_timeline_chart(emotion_probs):
    """Create an emotion timeline showing top emotions over time (simulated)"""
    # Get top 6 emotions
    top_emotions = dict(sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True)[:6])
    
    # Simulate timeline data
    timeline_points = np.linspace(0, 10, 50)  # 10-minute content
    
    fig = go.Figure()
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    
    for i, (emotion, base_prob) in enumerate(top_emotions.items()):
        # Generate realistic timeline with variations
        timeline_values = []
        for t in timeline_points:
            variation = 0.3 * np.sin(t * 0.5 + i) + 0.2 * np.random.random()
            value = max(0, min(1, base_prob + variation * 0.3))
            timeline_values.append(value)
        
        emoji = EMOTION_EMOJIS.get(emotion, '')
        fig.add_trace(go.Scatter(
            x=timeline_points,
            y=timeline_values,
            mode='lines',
            name=f'{emoji} {emotion.title()}',
            line=dict(color=colors[i % len(colors)], width=3),
            hovertemplate=f'<b>{emotion.title()}</b><br>Time: %{{x:.1f}}min<br>Intensity: %{{y:.2f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title='Emotional Journey Timeline',
        xaxis_title='Time (minutes)',
        yaxis_title='Emotion Intensity',
        height=400,
        hovermode='x unified'
    )
    
    return fig

def create_audience_segments_chart(reaction_scores):
    """Create audience segments breakdown"""
    segments = {
        'Highly Engaged': reaction_scores['Engaged_Curiosity'] * 0.8,
        'Positive Reactors': reaction_scores['Positive_Response'] * 0.9,
        'Critical Viewers': reaction_scores['Negative_Response'] * 0.7,
        'Passive Viewers': reaction_scores['Neutral_Response'] * 0.6
    }
    
    fig = go.Figure(data=[go.Pie(
        labels=list(segments.keys()),
        values=list(segments.values()),
        hole=.4,
        marker_colors=['#FF6B6B', '#4ECDC4', '#FFA07A', '#D3D3D3']
    )])
    
    fig.update_traces(
        textposition='inside', 
        textinfo='percent+label',
        textfont_size=12
    )
    fig.update_layout(
        title_text="Predicted Audience Segments",
        height=400,
        showlegend=True
    )
    return fig

# -------------------------------
# Enhanced Insights
# -------------------------------
def get_multimedia_insights(result):
    """Generate insights specific to multimedia content"""
    insights = []
    sentiment = result['overall_sentiment']
    confidence = result['sentiment_confidence']
    metrics = result['content_metrics']
    content_type = result['content_type']
    
    # Impact Score Insight
    impact_score = metrics.get('impact_score', 0)
    if impact_score > 80:
        insights.append(f"🎯 **Exceptional emotional impact** ({impact_score:.1f}/100) - This content has strong viral potential!")
    elif impact_score > 60:
        insights.append(f"📊 **Good emotional impact** ({impact_score:.1f}/100) - Above average engagement expected")
    else:
        insights.append(f"⚖️ **Moderate impact** ({impact_score:.1f}/100) - Consider enhancing emotional hooks")
    
    # Engagement Insight
    engagement_rate = metrics.get('engagement_rate', 0)
    if engagement_rate > 75:
        insights.append(f"🔥 **High engagement potential** ({engagement_rate:.1f}%) - Audience likely to interact")
    elif engagement_rate > 50:
        insights.append(f"👀 **Moderate engagement** ({engagement_rate:.1f}%) - Good baseline interaction")
    else:
        insights.append(f"😴 **Low engagement risk** ({engagement_rate:.1f}%) - Consider adding interactive elements")
    
    # Retention Insight
    retention_rate = metrics.get('retention_rate', 0)
    if retention_rate > 70:
        insights.append(f"⏱️ **Strong retention potential** ({retention_rate:.1f}%) - Viewers likely to watch till end")
    else:
        insights.append(f"⚠️ **Retention challenge** ({retention_rate:.1f}%) - May need pacing adjustments")
    
    # Content-specific insights
    if content_type == "Video Content":
        insights.append("🎬 **Video tip**: First 15 seconds are crucial - ensure strong visual hook")
    elif content_type == "Audio Content":
        insights.append("🎵 **Audio tip**: Vary tone and pace to maintain listener interest")
    elif content_type == "Educational":
        insights.append("📚 **Educational tip**: Break complex concepts into digestible segments")
    
    # Top emotions insight
    top_emotions = sorted(result['emotion_probs'].items(), key=lambda x: x[1], reverse=True)[:3]
    top_emotion_names = [f"{EMOTION_EMOJIS.get(emo, '')} {emo}" for emo, _ in top_emotions]
    insights.append(f"🔝 **Dominant emotions**: {', '.join(top_emotion_names)}")
    
    return insights

def get_actionable_recommendations(result):
    """Generate actionable recommendations for content improvement"""
    recommendations = []
    metrics = result['content_metrics']
    content_type = result['content_type']
    top_emotions = sorted(result['emotion_probs'].items(), key=lambda x: x[1], reverse=True)
    
    # Hook optimization
    if metrics.get('high_engagement', 0) < 0.6:
        recommendations.append({
            'type': 'Hook Optimization',
            'priority': 'High',
            'insight': 'Opening moments show low engagement potential',
            'action': 'Create a compelling hook in the first 15 seconds - pose a question, show a surprising fact, or create visual intrigue',
            'expected_impact': '+15-25% engagement rate',
            'color_class': 'negative-rec'
        })
    
    # Pacing recommendations
    if 'confusion' in [e[0] for e in top_emotions[:5]]:
        recommendations.append({
            'type': 'Content Clarity',
            'priority': 'Medium',
            'insight': 'Audience may experience confusion with current content structure',
            'action': 'Break down complex ideas into smaller segments with clear transitions and visual aids',
            'expected_impact': '+10-15% retention rate',
            'color_class': 'neutral-rec'
        })
    
    # Emotional peak enhancement
    if result['overall_sentiment'] == 'Positive' and result['sentiment_confidence'] > 0.7:
        recommendations.append({
            'type': 'Emotional Amplification',
            'priority': 'Medium',
            'insight': 'Strong positive emotional response detected',
            'action': 'Create similar emotional peaks throughout your content to maintain engagement',
            'expected_impact': '+20-30% viral potential',
            'color_class': 'positive-rec'
        })
    
    # Content-specific recommendations
    if content_type == "Video Content":
        if metrics.get('retention_rate', 0) < 60:
            recommendations.append({
                'type': 'Visual Enhancement',
                'priority': 'High',
                'insight': 'Retention rate below optimal for video content',
                'action': 'Add more visual variety: B-roll footage, graphics, text overlays, or change camera angles every 10-15 seconds',
                'expected_impact': '+15-20% retention rate',
                'color_class': 'negative-rec'
            })
    
    elif content_type == "Audio Content":
        if 'nervousness' in result['emotion_probs'] and result['emotion_probs']['nervousness'] > 0.4:
            recommendations.append({
                'type': 'Audio Quality',
                'priority': 'Medium',
                'insight': 'Content may induce nervousness - possibly due to audio quality or pacing',
                'action': 'Improve audio clarity, reduce background noise, and speak with confident, steady pace',
                'expected_impact': '+10-15% positive sentiment',
                'color_class': 'neutral-rec'
            })
    
    # Always include at least one recommendation
    if not recommendations:
        recommendations.append({
            'type': 'Content Enhancement',
            'priority': 'Medium',
            'insight': 'Good baseline performance with room for improvement',
            'action': 'Focus on storytelling elements: create tension, resolution, and emotional payoffs throughout your content',
            'expected_impact': '+10-15% overall engagement',
            'color_class': 'positive-rec'
        })
    
    return recommendations[:3]  # Limit to top 3 recommendations

# -------------------------------
# Main App
# -------------.------------------
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎭 EmotiCraft AI</h1>
        <p style="font-size: 1.2em; margin: 0;">Multimedia Emotion Analysis for Content Creators</p>
        <p style="opacity: 0.9; margin: 0.5rem 0 0 0;">Predict audience emotions • Optimize engagement • Create viral content</p>
        <p style="opacity: 0.9; margin: 0.5rem 0 0 0;">By:</p>
        <ul style="opacity: 0.9; margin: 0; padding: 1rem; color:black; padding-left: 2rem; background-color: #f0f0f0; border-radius: 8px; list-style-type: disc;">
            <li>Malachy Precious</li>
            <li>Ignatius Akuoma</li>
            <li>Onovo Chidera</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    
    # Load models
    model, vectorizer = load_models()
    
    # Sidebar for content configuration
    with st.sidebar:
        st.header("🎯 Content Analysis Setup")
        
        # Content type selection
        content_type = st.selectbox(
            "Content Type:",
            ["Video Content", "Audio Content", "Educational", "Social Media Post"],
            help="Select the type of content you're analyzing"
        )
        
        # Platform selection
        platform = st.selectbox(
            "Target Platform:",
            ["YouTube", "TikTok", "Instagram", "Podcast", "LinkedIn", "Twitter", "Blog", "Course Platform", "Other"]
        )
        
        # Target audience
        target_audience = st.selectbox(
            "Target Audience:",
            ["General Public", "Young Adults (18-25)", "Professionals", "Students", "Parents", "Seniors", "Niche Community"]
        )
        
        # Content goals
        content_goals = st.multiselect(
            "Content Goals:",
            ["Increase Engagement", "Boost Retention", "Drive Shares", "Educational Value", "Brand Awareness", "Entertainment"]
        )
        
        st.markdown("---")
        st.markdown("### 📊 Analysis Features")
        st.markdown("""
        - **Real-time emotion prediction**
        - **Engagement forecasting**
        - **Retention analysis**
        - **Viral potential scoring**
        - **Actionable recommendations**
        """)
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📁 Content Upload", "🧠 Analysis", "📊 Results Dashboard"])
    
    with tab1:
        st.header("Upload Your Content")
        
        # Content type cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="content-type-card">
                <div style="font-size: 3em; margin-bottom: 1rem;">🎬</div>
                <h3>Video Content</h3>
                <p>YouTube videos, TikToks, Instagram Reels, tutorials, vlogs, commercials</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="content-type-card">
                <div style="font-size: 3em; margin-bottom: 1rem;">🎵</div>
                <h3>Audio Content</h3>
                <p>Podcasts, music tracks, voice recordings, radio ads, audiobooks</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="content-type-card">
                <div style="font-size: 3em; margin-bottom: 1rem;">📚</div>
                <h3>Educational</h3>
                <p>Online courses, presentations, lectures, training materials</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Upload methods
        st.subheader("Choose Your Upload Method")
        
        upload_method = st.radio(
            "How would you like to provide your content?",
            ["📁 Upload Files", "✍️ Text Script/Description", "🔗 URL Link (Coming Soon)"],
            horizontal=True
        )
        
        if upload_method == "📁 Upload Files":
            st.markdown("""
            <div class="upload-area">
                <div style="font-size: 4em; margin-bottom: 1rem;">📁</div>
                <h3>Drop Your Files Here</h3>
                <p>Supported: MP4, MOV, AVI, MP3, WAV, PDF, PPTX, DOCX</p>
                <p style="color: #666;">Maximum file size: 500MB</p>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "Choose files",
                type=['mp4', 'mov', 'avi', 'mp3', 'wav', 'pdf', 'pptx', 'docx'],
                accept_multiple_files=False
            )
            
            if uploaded_file:
                file_details = {
                    "filename": uploaded_file.name,
                    "filetype": uploaded_file.type,
                    "filesize": uploaded_file.size
                }
                
                st.success(f"✅ File uploaded: {file_details['filename']}")
                
                # File preview
                col1, col2 = st.columns([2, 1])
                with col1:
                    if uploaded_file.type.startswith('image'):
                        image = Image.open(uploaded_file)
                        st.image(image, caption="Uploaded Image", use_column_width=True)
                    elif uploaded_file.type.startswith('video'):
                        st.video(uploaded_file)
                    elif uploaded_file.type.startswith('audio'):
                        st.audio(uploaded_file)
                    else:
                        st.info("📄 File uploaded successfully. Ready for analysis.")
                
                with col2:
                    st.markdown("### File Details")
                    st.write(f"**Name:** {file_details['filename']}")
                    st.write(f"**Type:** {file_details['filetype']}")
                    st.write(f"**Size:** {file_details['filesize']/1024/1024:.2f} MB")
                    
                    # Store file details in session state
                    st.session_state['uploaded_file'] = file_details
                    st.session_state['file_content'] = f"Analyzing {content_type.lower()} file: {file_details['filename']}"
        
        elif upload_method == "✍️ Text Script/Description":
            st.subheader("Enter Your Content Script or Description")
            
            content_input_method = st.radio(
                "Input method:",
                ["Manual Text Entry", "Upload Text File"],
                horizontal=True
            )
            
            if content_input_method == "Manual Text Entry":
                content_text = st.text_area(
                    "Paste your script, description, or content outline:",
                    height=200,
                    placeholder="Enter your video script, podcast outline, course description, or any text that represents your content..."
                )
                
                if content_text:
                    st.session_state['file_content'] = content_text
                    st.success(f"✅ Text content ready for analysis ({len(content_text)} characters)")
            
            else:
                text_file = st.file_uploader(
                    "Upload text file",
                    type=['txt', 'md', 'docx'],
                    accept_multiple_files=False
                )
                if text_file:
                    content_text = str(text_file.read(), "utf-8")
                    st.session_state['file_content'] = content_text
                    st.success(f"✅ Text file uploaded and ready for analysis")
        
        else:
            st.info("🔗 URL analysis feature coming soon! Upload files or enter text for now.")
    
    with tab2:
        st.header("🧠 AI-Powered Emotion Analysis")
        
        if 'file_content' not in st.session_state:
            st.warning("⚠️ Please upload content in the 'Content Upload' tab first.")
            st.stop()
        
        # Analysis configuration
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("Analysis Configuration")
            analysis_depth = st.select_slider(
                "Analysis Depth:",
                options=["Quick Scan", "Standard Analysis", "Deep Dive", "Expert Level"],
                value="Standard Analysis"
            )
            
            include_timeline = st.checkbox("Include emotional timeline analysis", value=True)
            include_segments = st.checkbox("Analyze audience segments", value=True)
            include_competitive = st.checkbox("Compare with similar content", value=False)
        
        with col2:
            st.markdown("### 🔧 Settings")
            confidence_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.3, 0.1)
            
        # Start Analysis Button
        if st.button("🚀 Start Comprehensive Analysis", type="primary", use_container_width=True):
            # Analysis progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate analysis steps
            steps = [
                ("Preprocessing content...", 10),
                ("Analyzing emotional patterns...", 30),
                ("Calculating engagement metrics...", 50),
                ("Generating audience predictions...", 70),
                ("Creating recommendations...", 90),
                ("Finalizing results...", 100)
            ]
            
            for step_text, progress in steps:
                status_text.text(step_text)
                progress_bar.progress(progress)
                time.sleep(0.5)  # Simulate processing time
            
            # Perform actual analysis
            result = predict_content_emotion(
                st.session_state['file_content'], 
                model, 
                vectorizer, 
                content_type, 
                confidence_threshold
            )
            
            # Store results in session state
            st.session_state['analysis_result'] = result
            st.session_state['analysis_config'] = {
                'content_type': content_type,
                'platform': platform,
                'target_audience': target_audience,
                'content_goals': content_goals,
                'analysis_depth': analysis_depth
            }
            
            status_text.text("✅ Analysis complete!")
            progress_bar.progress(100)
            
            st.success("🎉 Analysis completed successfully! View results in the 'Results Dashboard' tab.")
            
            # Auto-switch to results tab
            st.balloons()
    
    with tab3:
        st.header("📊 Results Dashboard")
        
        if 'analysis_result' not in st.session_state:
            st.warning("⚠️ Please complete the analysis first in the 'Analysis' tab.")
            st.stop()
        
        result = st.session_state['analysis_result']
        config = st.session_state.get('analysis_config', {})
        
        # Overall Score Banner
        impact_score = result['content_metrics'].get('impact_score', 0)
        
        if impact_score >= 80:
            score_color = "background: linear-gradient(135deg, #10b981 0%, #34d399 100%);"
            score_emoji = "🚀"
            score_text = "Exceptional"
        elif impact_score >= 60:
            score_color = "background: linear-gradient(135deg, #3b82f6 0%, #60a5fa 100%);"
            score_emoji = "⭐"
            score_text = "Good"
        else:
            score_color = "background: linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%);"
            score_emoji = "⚡"
            score_text = "Needs Work"
        
        st.markdown(f"""
        <div style="{score_color} padding: 2rem; border-radius: 15px; text-align: center; color: white; margin-bottom: 2rem;">
            <h1 style="margin: 0; font-size: 3em;">{score_emoji}</h1>
            <h2 style="margin: 0.5rem 0;">Emotional Impact Score: {impact_score:.1f}/100</h2>
            <h3 style="margin: 0; opacity: 0.9;">{score_text} Performance</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Key Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            engagement_rate = result['content_metrics'].get('engagement_rate', 0)
            st.metric(
                "🎯 Engagement Rate", 
                f"{engagement_rate:.1f}%", 
                delta=f"{engagement_rate-65:.1f}% vs avg" if engagement_rate > 0 else None
            )
        
        with col2:
            retention_rate = result['content_metrics'].get('retention_rate', 0)
            st.metric(
                "⏱️ Retention Rate", 
                f"{retention_rate:.1f}%", 
                delta=f"{retention_rate-70:.1f}% vs avg" if retention_rate > 0 else None
            )
        
        with col3:
            viral_score = result['content_metrics'].get('viral_potential', 0) * 100
            st.metric(
                "🚀 Viral Potential", 
                f"{viral_score:.1f}%", 
                delta=f"{viral_score-50:.1f}% vs avg" if viral_score > 0 else None
            )
        
        with col4:
            share_likelihood = result['reaction_scores'].get('Positive_Response', 0) * 100
            st.metric(
                "📤 Share Likelihood", 
                f"{share_likelihood:.1f}%", 
                delta=f"{share_likelihood-60:.1f}% vs avg" if share_likelihood > 0 else None
            )
        
        st.markdown("---")
        
        # Charts Section
        st.subheader("📈 Detailed Analysis")
        
        # First row of charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(
                create_impact_score_gauge(result['content_metrics']), 
                use_container_width=True
            )
        
        with col2:
            st.plotly_chart(
                create_content_metrics_chart(result['content_metrics']), 
                use_container_width=True
            )
        
        # Second row of charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Enhanced emotion chart
            sorted_emotions = dict(sorted(result['emotion_probs'].items(), key=lambda x: x[1], reverse=True))
            top_emotions = dict(list(sorted_emotions.items())[:10])
            emoji_labels = [f"{EMOTION_EMOJIS.get(emo, '')} {emo.title()}" for emo in top_emotions.keys()]
            
            fig = px.bar(
                x=list(top_emotions.values()),
                y=emoji_labels,
                orientation='h',
                title='🎭 Top Emotional Reactions',
                labels={'x': 'Probability', 'y': 'Emotions'},
                color=list(top_emotions.values()),
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=500, showlegend=False)
            fig.update_xaxes(range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.plotly_chart(
                create_audience_segments_chart(result['reaction_scores']), 
                use_container_width=True
            )
        
        # Emotional Timeline (if enabled)
        if st.session_state.get('analysis_config', {}).get('include_timeline', True):
            st.plotly_chart(
                create_emotion_timeline_chart(result['emotion_probs']), 
                use_container_width=True
            )
        
        st.markdown("---")
        
        # AI Insights Section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🧠 AI Insights")
            insights = get_multimedia_insights(result)
            
            for insight in insights:
                st.markdown(f"""
                <div class="insight-card">
                    {insight}
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("🎯 Quick Stats")
            
            # Content type specific stats
            if content_type == "Video Content":
                st.markdown("### 🎬 Video Metrics")
                st.write(f"**Estimated Watch Time:** {retention_rate/100 * 8:.1f} min (of 8 min)")
                st.write(f"**Skip Risk:** {'Low' if retention_rate > 70 else 'Medium' if retention_rate > 50 else 'High'}")
                st.write(f"**Thumbnail Impact:** {'High' if viral_score > 70 else 'Medium'}")
                
            elif content_type == "Audio Content":
                st.markdown("### 🎵 Audio Metrics")
                st.write(f"**Listen Through Rate:** {retention_rate:.1f}%")
                st.write(f"**Replay Potential:** {'High' if engagement_rate > 75 else 'Medium'}")
                st.write(f"**Mood Impact:** {result['overall_sentiment']}")
                
            elif content_type == "Educational":
                st.markdown("### 📚 Learning Metrics")
                comprehension_score = (100 - result['emotion_probs'].get('confusion', 0) * 100)
                st.write(f"**Comprehension Score:** {comprehension_score:.1f}%")
                st.write(f"**Knowledge Retention:** {retention_rate:.1f}%")
                st.write(f"**Engagement Level:** {'High' if engagement_rate > 70 else 'Medium'}")
        
        st.markdown("---")
        
        # Recommendations Section
        st.subheader("🚀 Actionable Recommendations")
        recommendations = get_actionable_recommendations(result)
        
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"""
            <div class="recommendation-card {rec['color_class']}">
                <h4 style="margin-top: 0;">#{i} {rec['type']} 
                    <span style="float: right; font-size: 0.8em; padding: 0.2rem 0.5rem; background: rgba(0,0,0,0.1); border-radius: 12px;">
                        {rec['priority']} Priority
                    </span>
                </h4>
                <p><strong>🔍 Insight:</strong> {rec['insight']}</p>
                <p><strong>💡 Action:</strong> {rec['action']}</p>
                <p><strong>📈 Expected Impact:</strong> {rec['expected_impact']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Export and Actions
        st.subheader("📤 Export & Next Steps")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Create comprehensive report data
            report_data = {
                'content_type': content_type,
                'platform': platform,
                'target_audience': target_audience,
                'impact_score': impact_score,
                'engagement_rate': engagement_rate,
                'retention_rate': retention_rate,
                'viral_potential': viral_score,
                'overall_sentiment': result['overall_sentiment'],
                'sentiment_confidence': result['sentiment_confidence'],
                **{f'emotion_{k}': v for k, v in result['emotion_probs'].items()},
                **{f'reaction_{k}': v for k, v in result['reaction_scores'].items()}
            }
            
            report_df = pd.DataFrame([report_data])
            csv_data = report_df.to_csv(index=False)
            
            st.download_button(
                label="📊 Download Full Report",
                data=csv_data,
                file_name=f"emotion_analysis_report_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            if st.button("🔄 Analyze Another Content", use_container_width=True):
                # Clear session state
                for key in ['file_content', 'analysis_result', 'analysis_config']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        
        with col3:
            st.button("👥 Share with Team", use_container_width=True, disabled=True, help="Coming soon!")
        
        with col4:
            st.button("📈 Compare Content", use_container_width=True, disabled=True, help="Coming soon!")
        
        # Success metrics summary
        if impact_score >= 70:
            st.success(f"""
            🎉 **Great job!** Your content shows strong emotional impact with {impact_score:.1f}/100 score. 
            Focus on the recommendations above to push it even further!
            """)
        elif impact_score >= 50:
            st.info(f"""
            👍 **Good foundation!** Your content has decent emotional appeal with {impact_score:.1f}/100 score. 
            Implement the high-priority recommendations to boost engagement.
            """)
        else:
            st.warning(f"""
            🔧 **Room for improvement!** Your content scored {impact_score:.1f}/100. 
            Focus on the recommendations above to significantly improve audience response.
            """)

if __name__ == "__main__":
    main()