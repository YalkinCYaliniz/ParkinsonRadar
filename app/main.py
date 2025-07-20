from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import numpy as np
import pandas as pd
import json
from werkzeug.utils import secure_filename
from app.core.audio_feature_extraction import extract_features_from_audio
from app.models.advanced_models import ParkinsonModelTrainer
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'parkinson_detection_2024'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables
model_trainer = None
healthy_stats = None
parkinson_stats = None

def initialize_models():
    """Initialize the model trainer and load statistics"""
    global model_trainer, healthy_stats, parkinson_stats
    
    try:
        # Try to load existing models
        model_trainer = ParkinsonModelTrainer()
        model_trainer.load_models('app/models/trained/parkinson_models.joblib')
        print("Loaded existing models successfully!")
    except:
        print("No existing models found. Training new models...")
        model_trainer = ParkinsonModelTrainer()
        model_trainer.train_all_models()
        model_trainer.save_models('app/models/trained/parkinson_models.joblib')
        print("New models trained and saved!")
    
    # Load dataset statistics
    df = pd.read_csv('data/raw/parkinsons.data')
    healthy_data = df[df['status'] == 0].drop(['name', 'status'], axis=1)
    parkinson_data = df[df['status'] == 1].drop(['name', 'status'], axis=1)
    
    healthy_stats = healthy_data.describe()
    parkinson_stats = parkinson_data.describe()

# Initialize models on startup
initialize_models()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    """Handle audio file upload and analysis"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Extract features
        features = extract_features_from_audio(file_path)
        
        if features is None:
            return jsonify({'error': 'Failed to extract features from audio'}), 400
        
        # Make prediction
        prediction_result = model_trainer.predict_parkinson(features)
        
        # Create visualizations
        feature_comparison_plot = create_feature_comparison_plot(features)
        radar_plot = create_radar_plot(features)
        
        # Clean up uploaded file
        os.remove(file_path)
        
        return jsonify({
            'prediction': prediction_result,
            'features': features,
            'feature_comparison_plot': feature_comparison_plot,
            'radar_plot': radar_plot,
            'healthy_averages': healthy_stats.loc['mean'].to_dict(),
            'parkinson_averages': parkinson_stats.loc['mean'].to_dict()
        })
        
    except Exception as e:
        print(f"Error in upload_audio: {e}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

def create_feature_comparison_plot(user_features):
    """Create comparison plot with adaptive scaling for better visibility"""
    # Select key features for comparison
    key_features = [
        'MDVP:Fo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Shimmer', 
        'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'PPE'
    ]
    
    healthy_means = [healthy_stats.loc['mean', feature] for feature in key_features]
    parkinson_means = [parkinson_stats.loc['mean', feature] for feature in key_features]
    user_values = [user_features.get(feature, 0) for feature in key_features]
    
    # Simple multiplication for visibility - much more aggressive!
    def enhance_visibility(values, feature_names):
        enhanced = []
        multipliers = []
        
        for value, feature_name in zip(values, feature_names):
            # ULTRA AGGRESSIVE multipliers for visibility
            if value < 5.0:  # Anything under 5 gets boosted
                if feature_name == 'MDVP:Jitter(%)':
                    multiplier = 2000  # 0.006 to 12.0 - VISIBLE!
                    enhanced.append(value * multiplier)
                    multipliers.append(f"x{multiplier}")
                elif feature_name in ['MDVP:Shimmer', 'NHR']:
                    multiplier = 200   # 0.042 to 8.4, 0.095 to 19.0 - VISIBLE!
                    enhanced.append(value * multiplier)
                    multipliers.append(f"x{multiplier}")
                elif feature_name in ['RPDE', 'DFA', 'PPE']:
                    multiplier = 50    # 0.386 to 19.3 - VISIBLE!
                    enhanced.append(value * multiplier)
                    multipliers.append(f"x{multiplier}")
                elif feature_name == 'spread1' and value < 0:
                    # Handle negative spread1
                    multiplier = 10
                    enhanced.append(abs(value) * multiplier)  # Make positive and boost
                    multipliers.append(f"x{multiplier} (abs)")
                else:
                    enhanced.append(value)
                    multipliers.append("")
            else:
                enhanced.append(value)
                multipliers.append("")
        
        return enhanced, multipliers
    
    # Apply adaptive visibility to ALL values for consistent scale
    enhanced_user_values, multipliers = enhance_visibility(user_values, key_features)
    enhanced_healthy_means, _ = enhance_visibility(healthy_means, key_features)
    enhanced_parkinson_means, _ = enhance_visibility(parkinson_means, key_features)
    
    # Create subplot
    fig = go.Figure()
    
    # Add bars for averages (same enhanced scale)
    fig.add_trace(go.Bar(
        name='Healthy Average (Enhanced)',
        x=key_features,
        y=enhanced_healthy_means,
        marker_color='green',
        opacity=0.7,
        showlegend=True
    ))
    
    fig.add_trace(go.Bar(
        name="Parkinson's Average (Enhanced)",
        x=key_features,
        y=enhanced_parkinson_means,
        marker_color='red',
        opacity=0.7,
        showlegend=True
    ))
    
    # Add user values with enhanced visibility
    # Create hover text with multiplier info
    hover_texts = []
    for i, (original, enhanced, mult, feature) in enumerate(zip(user_values, enhanced_user_values, multipliers, key_features)):
        if mult:
            hover_text = f"<b>{feature}</b><br>Gerçek: {original:.6f}<br>Görsel: {enhanced:.2f} {mult}<br><i>Tüm değerler aynı oranda çarpılmıştır</i>"
        else:
            hover_text = f"<b>{feature}</b><br>Değer: {original:.3f}"
        hover_texts.append(hover_text)
    
    fig.add_trace(go.Scatter(
        name='Your Voice (Enhanced)',
        x=key_features,
        y=enhanced_user_values,
        mode='markers+lines',
        marker=dict(size=12, color='blue', symbol='diamond'),
        line=dict(width=4, color='blue'),
        showlegend=True,
        hovertemplate='%{hovertext}<extra></extra>',
        hovertext=hover_texts
    ))
    

    
    # Smart y-axis range calculation
    all_visible_values = enhanced_healthy_means + enhanced_parkinson_means + enhanced_user_values
    y_max = max(all_visible_values)
    y_min = min([v for v in all_visible_values if v > 0])
    
    # Set smart y-axis range - accommodate ultra-enhanced values
    if y_max > 100:  # Large scale features or ultra-enhanced
        y_range = [0, y_max * 1.1]
    elif y_max > 20:  # Medium enhanced values
        y_range = [0, y_max * 1.2]
    else:  # Small values
        y_range = [0, max(25, y_max * 1.3)]  # Ensure good visibility
    
    fig.update_layout(
        title='Voice Feature Comparison (Küçük Değerler Çarpılmıştır)',
        xaxis_title='Voice Features',
        yaxis_title='Feature Values',
        barmode='group',
        height=500,
        xaxis_tickangle=-45,
        yaxis=dict(range=y_range),
        annotations=[
            dict(
                text="TÜM DEĞERLER görsellik için çarpılmıştır: Jitter x2000, Shimmer/NHR x200, RPDE/DFA/PPE x50",
                xref="paper", yref="paper",
                x=0.5, y=1.1, xanchor='center', yanchor='bottom',
                showarrow=False,
                font=dict(size=10, color="blue")
            )
        ]
    )
    
    return json.dumps(fig, cls=PlotlyJSONEncoder)

def create_radar_plot(user_features):
    """Create radar plot for voice features"""
    # Select features for radar plot
    radar_features = [
        'MDVP:Fo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Shimmer', 
        'NHR', 'HNR', 'RPDE', 'DFA', 'PPE'
    ]
    
    # Real dataset ranges for proper normalization
    dataset_ranges = {
        'MDVP:Fo(Hz)': (88.33, 260.11),
        'MDVP:Jitter(%)': (0.168, 3.316),
        'MDVP:Shimmer': (0.0095, 0.119),
        'NHR': (0.001, 0.315),
        'HNR': (8.44, 33.05),
        'RPDE': (0.257, 0.686),
        'DFA': (0.575, 0.825),
        'PPE': (0.045, 0.527)
    }
    
    # Adaptive normalization with minimum visibility for radar chart
    def normalize_to_radar_scale(value, feature_name):
        if feature_name not in dataset_ranges:
            return 50  # Default to middle
        
        min_val, max_val = dataset_ranges[feature_name]
        range_val = max_val - min_val
        
        if range_val == 0:
            return 50
        
        # Standard normalization
        normalized = ((value - min_val) / range_val) * 100
        clamped = max(0, min(100, normalized))
        
        # Adaptive visibility - ensure ALL features are visible with smart thresholds
        feature_visibility = {
            'MDVP:Jitter(%)': {'min_visible': 8, 'boost_factor': 2.0},   # Jitter needs high visibility
            'MDVP:Shimmer': {'min_visible': 6, 'boost_factor': 1.5},     # Shimmer moderate visibility  
            'NHR': {'min_visible': 10, 'boost_factor': 1.8},             # NHR very important
            'RPDE': {'min_visible': 8, 'boost_factor': 1.6},             # RPDE important
            'DFA': {'min_visible': 8, 'boost_factor': 1.6},              # DFA important
            'PPE': {'min_visible': 8, 'boost_factor': 1.6},              # PPE important
            'MDVP:Fo(Hz)': {'min_visible': 5, 'boost_factor': 1.0},      # F0 usually visible
            'HNR': {'min_visible': 5, 'boost_factor': 1.0}               # HNR usually visible
        }
        
        if feature_name in feature_visibility:
            config = feature_visibility[feature_name]
            min_visible = config['min_visible']
            boost_factor = config['boost_factor']
            
            if clamped < min_visible and value > 0:
                # Smart boost: preserve relative differences but ensure visibility
                if normalized < 0:  # Below dataset minimum
                    clamped = min_visible + abs(normalized) * 0.1
                else:
                    clamped = max(min_visible, clamped * boost_factor)
            
        return min(clamped, 100)  # Cap at 100
    
    user_normalized = []
    healthy_normalized = []
    parkinson_normalized = []
    
    for feature in radar_features:
        healthy_mean = healthy_stats.loc['mean', feature]
        parkinson_mean = parkinson_stats.loc['mean', feature]
        user_value = user_features.get(feature, 0)
        
        # Now each value gets properly normalized using real dataset ranges
        user_normalized.append(normalize_to_radar_scale(user_value, feature))
        healthy_normalized.append(normalize_to_radar_scale(healthy_mean, feature))
        parkinson_normalized.append(normalize_to_radar_scale(parkinson_mean, feature))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=healthy_normalized + [healthy_normalized[0]],
        theta=radar_features + [radar_features[0]],
        fill='toself',
        name='Healthy Average',
        line_color='green',
        opacity=0.6
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=parkinson_normalized + [parkinson_normalized[0]],
        theta=radar_features + [radar_features[0]],
        fill='toself',
        name="Parkinson's Average",
        line_color='red',
        opacity=0.6
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=user_normalized + [user_normalized[0]],
        theta=radar_features + [radar_features[0]],
        fill='toself',
        name='Your Voice',
        line_color='blue',
        opacity=0.8
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="Voice Features Radar Chart",
        height=500
    )
    
    return json.dumps(fig, cls=PlotlyJSONEncoder)

@app.route('/get_statistics')
def get_statistics():
    """Get statistical information about the dataset"""
    df = pd.read_csv('parkinsons.data')
    
    stats = {
        'total_samples': len(df),
        'parkinson_samples': df['status'].sum(),
        'healthy_samples': (df['status'] == 0).sum(),
        'parkinson_percentage': (df['status'].mean() * 100),
        'healthy_percentage': ((1 - df['status'].mean()) * 100),
        'feature_count': len(df.columns) - 2  # Exclude 'name' and 'status'
    }
    
    return jsonify(stats)

@app.route('/about')
def about():
    """About page with information about the project"""
    return render_template('about.html')

if __name__ == '__main__':
    print("Initializing Parkinson's Detection System...")
    initialize_models()
    print("System ready!")
    app.run(debug=True, host='0.0.0.0', port=5001) 