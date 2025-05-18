from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import os

app = Flask(__name__)
CORS(app)

# Define the questions and their corresponding learning style categories
questions = [
    # Multiple choice questions (first 10 questions)
    {"question": "When studying for an exam, what method do you find most effective?", "styles": ["Visual", "Auditory", "Read/Write", "Kinesthetic"]},
    {"question": "When you need to remember a phone number or address, how do you do it?", "styles": ["Visual", "Auditory", "Read/Write", "Kinesthetic"]},
    {"question": "If you were to learn how to assemble a piece of furniture, what would you do?", "styles": ["Visual", "Auditory", "Read/Write", "Kinesthetic"]},
    {"question": "In a classroom setting, what type of teaching style helps you learn best?", "styles": ["Visual", "Auditory", "Read/Write", "Kinesthetic"]},
    {"question": "How do you prefer to express your ideas and thoughts?", "styles": ["Visual", "Auditory", "Read/Write", "Kinesthetic"]},
    {"question": "When learning a new language, what method works best for you?", "styles": ["Visual", "Auditory", "Read/Write", "Kinesthetic"]},
    {"question": "How do you prefer to organize your work or study materials?", "styles": ["Visual", "Auditory", "Read/Write", "Kinesthetic"]},
    {"question": "What kind of environment helps you focus the best when working or studying?", "styles": ["Visual", "Auditory", "Read/Write", "Kinesthetic"]},
    {"question": "When you need to remember a key point or quote from a book, how do you do it?", "styles": ["Visual", "Auditory", "Read/Write", "Kinesthetic"]},
    {"question": "How do you prefer to solve a problem?", "styles": ["Visual", "Auditory", "Read/Write", "Kinesthetic"]},
    # Likert scale questions (last 8 questions)
    {"question": "I like to make lists, summaries, or take detailed notes when studying new material.", "style": "Read/Write"},
    {"question": "I find it easier to understand and remember information when it is presented in diagrams, charts, or pictures.", "style": "Visual"},
    {"question": "I find that hands-on practice or physical activity helps me understand new concepts.", "style": "Kinesthetic"},
    {"question": "I prefer to learn by engaging in discussions or listening to explanations rather than reading.", "style": "Auditory"},
    {"question": "When learning new material, I find that mind maps or visual summaries help me remember better.", "style": "Visual"},
    {"question": "I often read text out loud to help myself understand or memorize it.", "style": "Auditory"},
    {"question": "I learn best when I can work through material at my own pace, using written resources.", "style": "Read/Write"},
    {"question": "I prefer interactive or experiential learning environments over traditional lectures or reading assignments.", "style": "Kinesthetic"}
]

def preprocess_data():
    try:
        # Create a dummy dataset based on the questions
        num_samples = 1000
        data = []
        for _ in range(num_samples):
            sample = []
            for q in questions:
                if 'styles' in q:  # Multiple choice questions
                    sample.append(np.random.choice(len(q['styles'])) + 1)
                else:  # Likert scale questions
                    sample.append(np.random.randint(1, 6))
            data.append(sample)
        
        df = pd.DataFrame(data, columns=[f"Q{i+1}" for i in range(len(questions))])
        
        # Calculate scores for each learning style
        style_scores = {style: np.zeros(num_samples) for style in ['Visual', 'Auditory', 'Read/Write', 'Kinesthetic']}
        
        for i, q in enumerate(questions):
            if 'styles' in q:  # Multiple choice questions
                for j, style in enumerate(q['styles']):
                    style_scores[style] += (df[f'Q{i+1}'] == j+1).astype(int)
            else:  # Likert scale questions
                style_scores[q['style']] += df[f'Q{i+1}']
        
        # Determine the dominant learning style
        learning_style = pd.DataFrame(style_scores).idxmax(axis=1)
        
        X = df
        y = learning_style
        
        # Split data for training
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        print("Model training complete.")
        return model, X.columns.tolist()
    
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None, None

# Train the model once when the server starts
model, feature_columns = preprocess_data()

def generate_plot(user_scores):
    try:
        # Create a scatter plot of the user scores for each learning style
        styles = ['Visual', 'Auditory', 'Read/Write', 'Kinesthetic']
        scores = [user_scores[style] for style in styles]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(styles, scores, color='blue')
        plt.title('Learning Style Scores')
        plt.xlabel('Learning Styles')
        plt.ylabel('Scores')
        plt.ylim(0, max(scores) + 1)  # Set y-axis limit
        plt.grid(True)

        # Annotate the scores on the plot
        for i, score in enumerate(scores):
            plt.annotate(score, (styles[i], score), textcoords="offset points", xytext=(0,10), ha='center')

        # Save the plot to a file
        plot_filename = 'static/user_learning_style_plot.png'
        plt.savefig(plot_filename)
        plt.close()
        
        return plot_filename
    except Exception as e:
        print(f"Error during plot generation: {e}")
        return None
def generate_bar_plot(user_scores):
    try:
        # Create a bar plot of the user scores for each learning style
        styles = ['Visual', 'Auditory', 'Read/Write', 'Kinesthetic']
        scores = [user_scores[style] for style in styles]
        
        plt.figure(figsize=(10, 6))
        plt.bar(styles, scores, color='skyblue')
        plt.title('Learning Style Scores')
        plt.xlabel('Learning Styles')
        plt.ylabel('Scores')
        plt.ylim(0, max(scores) + 1)  # Set y-axis limit
        plt.grid(axis='y')

        # Save the plot to a file
        plot_filename = 'static/user_learning_style_bar_plot.png'
        plt.savefig(plot_filename)
        plt.close()
        
        return plot_filename
    except Exception as e:
        print(f"Error during bar plot generation: {e}")
        return None
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user responses from the request
        data = request.get_json()
        user_responses = data.get('responses')
        
        if user_responses is None or len(user_responses) != len(feature_columns):
            return jsonify({"error": "Invalid input data"}), 400
        
        # Prepare data for prediction
        user_data = pd.DataFrame([user_responses], columns=feature_columns)
        
        # Predict learning style
        prediction = model.predict(user_data)
        
        # Calculate user scores for each learning style
        style_scores = {style: 0 for style in ['Visual', 'Auditory', 'Read/Write', 'Kinesthetic']}
        for i, q in enumerate(questions):
            if 'styles' in q:  # Multiple choice questions
                style_scores[q['styles'][user_responses[i] - 1]] += 1
            else:  # Likert scale questions
                style_scores[q['style']] += user_responses[i]
        
        # Generate plots
        scatter_plot_file = generate_plot(style_scores)
        bar_plot_file = generate_bar_plot(style_scores)

        if scatter_plot_file is None or bar_plot_file is None:
            return jsonify({'error': 'Plot generation failed'}), 500
        
        return jsonify({
            'learning_style': prediction[0],
            'scatter_plot_url': scatter_plot_file,
            'bar_plot_url': bar_plot_file
        })
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'Prediction failed'}), 500
   


if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)
