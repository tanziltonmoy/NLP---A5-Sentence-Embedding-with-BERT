import dash
from dash import html, dcc, Input, Output
import torch
import torchtext
from bert import BERT, calculate_similarity

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server

# Setup device for model calculations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load tokenizer, vocabulary, and model
tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
vocab = torch.load('model/vocab.pth')

save_path = './model/bert.pt'
params, state_dict = torch.load(save_path, map_location=device)
model = BERT(**params, device=device).to(device)
model.load_state_dict(state_dict)
model.eval()

# App layout
app.layout = html.Div([
    html.H1("Text Similarity Calculator", style={'textAlign': 'center', 'marginBottom': '50px'}),
    html.P("Enter two sentences to calculate their semantic similarity.", style={'textAlign': 'center'}),
    html.Div([
        dcc.Textarea(id='sentence_a', placeholder='Enter Sentence A', style={'width': '90%', 'height': '100px', 'margin': '10px'}),
        dcc.Textarea(id='sentence_b', placeholder='Enter Sentence B', style={'width': '90%', 'height': '100px', 'margin': '10px'}),
        html.Button('Calculate Similarity', id='calculate-button', n_clicks=0, style={'margin': '20px', 'padding': '10px 20px', 'fontSize': '20px'}),
        html.Div(id='output-div', style={'fontSize': '24px', 'marginTop': '20px', 'fontWeight': 'bold'})
    ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'width': '50%', 'margin': '0 auto'}),
], style={'fontFamily': 'Arial, sans-serif', 'padding': '20px'})

# Callback for updating similarity score
@app.callback(
    Output('output-div', 'children'),
    [Input('calculate-button', 'n_clicks')],
    [Input('sentence_a', 'value'), Input('sentence_b', 'value')]
)
def prediction(n_clicks, sentence_a, sentence_b):
    if n_clicks > 0:
        score = calculate_similarity(model, tokenizer, vocab, params['max_len'], sentence_a, sentence_b, device)
        return f"Similarity Score: {round(score, 4)}"
    return ""

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
