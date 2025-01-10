from flask import Flask, request, render_template_string, send_file
import numpy as np
import torch
from joblib import load
from pykeen.models import TransE
import os

# Paths to saved models
TRANSE_MODEL_PATH = "datasets/trainedModel/trained_model_with_factory.pth"
MLP_MODEL_PATH = "datasets/trainedModel/mlp_model.joblib"
OUTPUT_DIR = "output"

# Load TransE model and triples_factory
checkpoint = torch.load(TRANSE_MODEL_PATH)
triples_factory = checkpoint["triples_factory"]

#TransE model
model = TransE(
    triples_factory=triples_factory,
    embedding_dim=200,  # Same dimension as used during training
    scoring_fct_norm=1,
    loss=None,  # The loss is not needed for inference
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Loading MLP model
mlp = load(MLP_MODEL_PATH)

# Extract entity and relation ID mappings
entity_to_id = triples_factory.entity_to_id
relation_to_id = triples_factory.relation_to_id

# Flask app
app = Flask(__name__)

os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    file_path = None

    if request.method == 'POST':
        subj = request.form.get('subj')
        pred = request.form.get('pred')
        obj = request.form.get('obj')

        if subj not in entity_to_id or obj not in entity_to_id or pred not in relation_to_id:
            emb_dim = model.entity_representations[0]._embeddings.weight.shape[-1]
            embedding = np.zeros(3 * emb_dim)
        else:
            s_id = entity_to_id[subj]
            p_id = relation_to_id[pred]
            o_id = entity_to_id[obj]

            s_emb = model.entity_representations[0](indices=torch.tensor([s_id]))
            p_emb = model.relation_representations[0](indices=torch.tensor([p_id]))
            o_emb = model.entity_representations[0](indices=torch.tensor([o_id]))

            embedding = torch.cat([s_emb[0], p_emb[0], o_emb[0]], dim=0).detach().cpu().numpy()

        if np.all(embedding == 0):
            result = f"Invalid triple: ({subj}, {pred}, {obj})."
        else:
            score = mlp.predict_proba([embedding])[:, 1][0]
            result = f"The predicted truth value score for ({subj}, {pred}, {obj}) is {score:.4f}."

            file_name = "result.ttl"
            file_path = os.path.join(OUTPUT_DIR, file_name)
            with open(file_path, "w", encoding="utf-8") as f:
                turtle_line = f'<{subj}> <{pred}> <{obj}> .\n'
                truth_line = f'<{subj}> <http://swc2017.aksw.org/hasTruthValue> "{score}"^^<http://www.w3.org/2001/XMLSchema#double> .\n'
                f.write(turtle_line)
                f.write(truth_line)

    html_form = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <title>Triple Accuracy Prediction</title>
    </head>
    <body>
    <center>
        <h1>Enter a Triple (s, p, o) without braces to Get Accuracy Score</h1>
        <form method="post">
            <label for="subj">Subject:</label>
            <input type="text" id="subj" name="subj" required><br><br>
            <label for="pred">Predicate:</label>
            <input type="text" id="pred" name="pred" required><br><br>
            <label for="obj">Object:</label>
            <input type="text" id="obj" name="obj" required><br><br>
            <button type="submit">Get Score</button>
        </form>
        <br>
        <h2>Result:</h2>
        <p>{{ result }}</p>
        {% if file_path %}
            <a href="/download?file={{ file_path }}" download>Download Turtle File</a>
        {% endif %}
    </center>
    </body>
    </html>
    '''
    return render_template_string(html_form, result=result, file_path=file_path)

@app.route('/download')
def download_file():
    file = request.args.get('file')
    if file and os.path.exists(file):
        return send_file(file, as_attachment=True)
    return "File not found.", 404

if __name__ == '__main__':
    app.run(debug=True)
