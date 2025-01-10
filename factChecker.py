import rdflib
import pandas as pd
from rdflib import Graph, URIRef, RDF
import numpy as np
import os
import torch
from joblib import dump

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score

############################################
# PyKEEN: MANUAL APIS
############################################
from pykeen.triples import TriplesFactory
from pykeen.models import TransE
from pykeen.training import SLCWATrainingLoop  # or LCWATrainingLoop
from pykeen.losses import MarginRankingLoss

############################################
# 1) Paths
############################################

REFERENCE_DATASET = "datasets/reference-kg.nt"
TRAIN_FILE = "datasets/fokg-sw-train-2024.nt"
TEST_FILE  = "datasets/fokg-sw-test-2024.nt"
RESULT_FILE = "resultFile/result.ttl"
TRUTH_VALUE_PRED = URIRef("http://swc2017.aksw.org/hasTruthValue")

valid_triples = []
with open(REFERENCE_DATASET, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line and line.endswith('.'):
            try:
                # Attempt to parse the triple
                s, p, o = line.split(' ', 2)
                valid_triples.append(line)
            except ValueError:
                print(f"Skipping invalid line: {line}")
        else:
            print(f"Skipping invalid line: {line}")

# Write the filtered triples to a new file
with open("datasets/filtered-reference-kg.nt", "w", encoding="utf-8") as f:
    f.write("\n".join(valid_triples))

# Update the reference dataset path
REFERENCE_DATASET = "datasets/filtered-reference-kg.nt"


############################################
# 2) Load Reference KG as (s, p, o)
############################################

print("Loading reference dataset:", REFERENCE_DATASET)
ref_graph = Graph()
ref_graph.parse(REFERENCE_DATASET, format="nt")

all_triples = []
for s, p, o in ref_graph.triples((None, None, None)):
    if isinstance(s, rdflib.URIRef) and isinstance(o, rdflib.URIRef):
        all_triples.append((str(s), str(p), str(o)))

print("Reference dataset triple count:", len(all_triples))
all_triples_array = np.array(all_triples, dtype=object)

# Create TriplesFactory
reference_factory = TriplesFactory.from_labeled_triples(all_triples_array)
print("reference_factory.num_triples =", reference_factory.num_triples)

############################################
# 3) Manually Create a TransE Model
############################################

embedding_dim = 200
margin = 1.0

model = TransE(
    triples_factory=reference_factory,
    embedding_dim=embedding_dim,
    scoring_fct_norm=1,  # L1 distance
    loss=MarginRankingLoss(margin=margin),  # margin-based ranking
    # You can set random_seed=..., but we'll do that globally or outside
)

print("Created TransE model with embedding_dim =", embedding_dim)

############################################
# 4) TRAINING LOOP (No pipeline)
############################################

# We'll use SLCWATrainingLoop for standard negative sampling.
# TRAINING LOOP
training_loop = SLCWATrainingLoop(
    model=model,
    triples_factory=reference_factory,
    optimizer="adam",
    optimizer_kwargs={"lr": 1e-3},
    negative_sampler="basic",
    negative_sampler_kwargs={"num_negs_per_pos": 10},
)

num_epochs = 5
batch_size = 256
print(f"Training TransE for {num_epochs} epochs, batch_size={batch_size} ...")
_ = training_loop.train(
    triples_factory=reference_factory,
    num_epochs=num_epochs,
    batch_size=batch_size,
    use_tqdm=True,
)
print("Done training TransE manually.")


# After training, we can retrieve the final entity/relation embeddings via model.

# ID mappings
entity_to_id = reference_factory.entity_to_id
relation_to_id = reference_factory.relation_to_id

############################################
# 5) Access the Embedding Modules
############################################

entity_representation = model.entity_representations[0]
relation_representation = model.relation_representations[0]

############################################
# 6) Parse Reified Train/Test
############################################

def load_reified_facts(nt_file, load_truth=True):
    g = rdflib.Graph()
    g.parse(nt_file, format="nt")
    facts = []
    for fact_iri in g.subjects(predicate=RDF.type, object=RDF.Statement):
        subj = g.value(subject=fact_iri, predicate=RDF.subject)
        pred = g.value(subject=fact_iri, predicate=RDF.predicate)
        obj = g.value(subject=fact_iri, predicate=RDF.object)
        if subj is None or pred is None or obj is None:
            continue
        if load_truth:
            truth_val = g.value(subject=fact_iri, predicate=TRUTH_VALUE_PRED)
            if truth_val is None:
                continue
            truth_val = float(truth_val.toPython())
            facts.append((str(fact_iri), str(subj), str(pred), str(obj), truth_val))
        else:
            facts.append((str(fact_iri), str(subj), str(pred), str(obj), None))
    return facts

train_facts = load_reified_facts(TRAIN_FILE, load_truth=True)
test_facts  = load_reified_facts(TEST_FILE, load_truth=False)

print(f"Loaded {len(train_facts)} reified train facts, {len(test_facts)} reified test facts.")

############################################
# 7) Embedding Lookup for (s, p, o)
############################################

# Updated embedding lookup function
def get_embedding_for_fact(subj, pred, obj):
    if subj not in entity_to_id or obj not in entity_to_id or pred not in relation_to_id:
        emb_dim = model.entity_representations[0]._embeddings.weight.shape[-1]
        return np.zeros(3 * emb_dim)

    s_id = entity_to_id[subj]
    p_id = relation_to_id[pred]
    o_id = entity_to_id[obj]

    s_emb = model.entity_representations[0](indices=torch.tensor([s_id]))  # shape [1, dim]
    p_emb = model.relation_representations[0](indices=torch.tensor([p_id]))
    o_emb = model.entity_representations[0](indices=torch.tensor([o_id]))

    cat = torch.cat([s_emb[0], p_emb[0], o_emb[0]], dim=0)
    return cat.detach().cpu().numpy()


############################################
# 8) Build X_train, X_test
############################################

train_triples, train_labels = [], []
for fact_iri, s, p, o, tv in train_facts:
    train_triples.append((s, p, o))
    train_labels.append(tv)

test_triples, test_fact_iris = [], []
for fact_iri, s, p, o, tv in test_facts:
    test_triples.append((s, p, o))
    test_fact_iris.append(fact_iri)

X_train = [get_embedding_for_fact(s, p, o) for (s, p, o) in train_triples]
y_train = np.array(train_labels)
X_train = np.array(X_train)

X_test = [get_embedding_for_fact(s, p, o) for (s, p, o) in test_triples]
X_test = np.array(X_test)

############################################
# 9) Classify with MLP
############################################

mlp = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", solver="adam", max_iter=50, random_state=42)
mlp.fit(X_train, y_train)

train_probs = mlp.predict_proba(X_train)[:, 1]
train_auc = roc_auc_score(y_train, train_probs)
print(f"Train AUC: {train_auc:.4f}")

test_probs = mlp.predict_proba(X_test)[:, 1]

############################################
# 10) Write TTL file
############################################

os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)
with open(RESULT_FILE, "w", encoding="utf-8") as f:
    for fact_iri, score in zip(test_fact_iris, test_probs):
        line = f'<{fact_iri}> <http://swc2017.aksw.org/hasTruthValue> "{score}"^^<http://www.w3.org/2001/XMLSchema#double> .\n'
        f.write(line)

print("Wrote predictions to", RESULT_FILE)

# torch.save(model, "datasets/trainedModel/trained_model.pkl")
# print("Model saved successfully.")

# Save the model and triples_factory together
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "triples_factory": reference_factory,
    },
    "datasets/trainedModel/trained_model_with_factory.pth"
)
print("Model and triples_factory saved successfully.")

dump(mlp, "datasets/trainedModel/mlp_model.joblib")
print("MLP model saved successfully.")