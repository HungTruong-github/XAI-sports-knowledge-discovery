# XAI-Football — Project Context
 
## 1. Project Identity
 
**General research topic:**  
Explainable Artificial Intelligence (XAI) for sports evaluation and knowledge discovery.
 
**Current specific research direction:**  
**Football Passing Networks and Tactical Analysis using Graph Neural Networks (GNN) combined with XAI.**
 
The current direction should be treated as the primary research direction unless a newer progress report explicitly changes it.
 
---
 
## 2. Research Direction and Motivation
 
The project exploits a natural relational structure in football: the passing network between players.
 
Instead of representing football data only as independent tabular features, the project models an attacking possession sequence as a directed graph:
 
- **Nodes:** players
- **Edges:** passes between players
- **Edge/node features:** information such as pass location, pass type, and time
- **Graph label:** whether the possession sequence leads to a shot or goal
The system is intended not only to make predictions, but also to explain those predictions.
 
The intended explanations should help answer questions such as:
 
- Which players are central to the attacking sequence?
- Which passes are important for the predicted outcome?
- Which groups of players form an important subgraph?
- Where are tactical bottlenecks in a team's passing structure?
- How can passing-network structure be compared quantitatively between teams or matches?
---
 
## 4. Research Gap
 
The progress report identifies three main gaps.
 
### 4.1 Lack of graph-level interpretability
 
Existing GNN-based approaches may optimize predictive performance, but there is limited work showing which subgraphs — including groups of players and sequences of passes — actually drive a prediction.
 
The project therefore aims to use graph-specific XAI to identify important components of a possession graph.
 
### 4.2 Lack of model-validated centrality
 
Traditional network centrality measures can identify important or central players, but the report highlights a lack of connection between these network measures and a specifically trained predictive machine-learning model.
 
The project proposes combining network centrality with XAI-derived importance.
 
### 4.3 Lack of a quantitative and reproducible tactical comparison process
 
Traditional passing-network analysis can remain qualitative or visualization-oriented.
 
The project aims to develop a quantitative and repeatable process for comparing tactics between teams and/or matches using model-based explanations.
 
---
 
## 5. Main Proposed System
 
The proposed system follows this conceptual pipeline:
 
**Raw football event data**
→ **Preprocessing**
→ **Possession sequences**
→ **Directed passing graphs**
→ **GNN prediction**
→ **Graph XAI**
→ **Hybrid Centrality**
→ **Tactical case-study analysis / knowledge discovery**
 
Each stage is described below.
 
---
 
## 6. Data Collection and Preprocessing
 
### Input
 
The report specifies:
 
- StatsBomb Open Data
- Raw event data in JSON format
- Bundesliga 2023/2024
- UEFA Champions League
### Processing
 
The raw event data is processed to create:
 
**Possession sequences**
 
These sequences are grouped and cleaned before graph construction.
 
### Output
 
A collection of cleaned attacking possession sequences suitable for graph construction.
 
---
 
## 7. Graph Construction
 
Each possession sequence is converted into a directed graph.
 
### Graph definition
 
**Nodes = players**
 
Each player participating in the possession sequence is represented as a graph node.
 
**Edges = passes**
 
A directed edge represents a pass from one player to another.
 
### Edge information
 
The report specifies that passing edges can include features such as:
 
- Location
- Pass type
- Timestamp
### Graph label
 
The graph/possession sequence is labeled according to whether it leads to:
 
- A shot
- A goal
The exact labeling rule and prediction target should be finalized during implementation and documented explicitly.
 
---
 
## 8. GNN Modeling
 
The report proposes using a Graph Neural Network to predict the outcome of an attacking possession sequence.
 
Candidate GNN architectures explicitly mentioned in the report:
 
- **GCN — Graph Convolutional Network**
- **MPNN — Message Passing Neural Network**
### Input
 
A dataset of possession graphs split into:
 
- Training set
- Validation set
- Test set
### Output
 
The GNN predicts the probability that a possession sequence will lead to a shot or goal.
 
### Research principle
 
Model performance should be evaluated experimentally rather than assumed.
 
Any final choice between GCN, MPNN, or another graph architecture must be justified by the actual experimental design and available evidence.
 
---
 
## 9. Graph XAI Layer
 
After training the GNN, the system applies graph-specific XAI.
 
The report explicitly identifies:
 
- **GNNExplainer**
- **PGExplainer**
### Input
 
- A trained GNN model
- A specific possession graph
### Output
 
An explanation identifying the most important parts of the graph, including:
 
- Important players
- Important passing edges
- Important subgraphs
- Explanation weights/scores
The purpose is to determine which graph structures contribute most strongly to the model's prediction.
 
---
 
## 10. Proposed Contribution: Hybrid Centrality
 
The report identifies **Hybrid Centrality** as the main proposed contribution.
 
### Concept
 
Hybrid Centrality combines:
 
1. Traditional network centrality measures
2. XAI-derived importance scores aggregated across multiple possession sequences
The resulting score is intended to rank the "central" or strategically important players of a team.
 
### Purpose
 
Unlike a purely traditional centrality metric, the proposed hybrid score is intended to be connected to the predictive behavior of the trained model.
 
The report describes this as a way to obtain a model-validated measure of player centrality.
 
### Important research requirement
 
The exact mathematical formulation of Hybrid Centrality is not fully specified in the progress report. It must therefore be formally defined later, including:
 
- Which traditional centrality measures are used
- How XAI scores are aggregated
- How different scales are normalized
- How multiple possession sequences are combined
- How the final player score is calculated
- How the score is validated
Do not assume a final formula unless it is explicitly defined in a later research document or experiment.
 
---
 
## 11. Tactical Case Study
 
The proposed case-study stage compares at least two teams and/or matches.
 
### Input
 
- Hybrid Centrality scores
- XAI-important subgraphs
- Passing-network information
### Output
 
A quantitative tactical analysis and visualization.
 
The report proposes visualizing the passing network while highlighting the important components identified by XAI.
 
Potential analytical questions include:
 
- Which players act as important connectors?
- Which passing routes are emphasized by the model?
- Are there identifiable bottlenecks?
- How does the important passing structure differ between teams or matches?
These questions are research objectives/examples from the current project direction, not already-established experimental findings.
 
---
 
## 12. Intended Users and Applications
 
The report identifies several intended application groups.
 
### Coaches / Assistant Analysts
 
Use the system to review matches, identify attacking combinations that create opportunities, and identify players who genuinely contribute to play organization.
 
### Opposition Scouting
 
Use the analysis to identify bottlenecks in an opponent's structure and potentially support pressing strategy design.
 
### Recruitment / Transfer Scouting
 
Use the system to evaluate a player's connecting role within a tactical system.
 
### Data-driven Sports Journalism / Commentary
 
Use visual explanations to communicate tactical insights to audiences.
 
### Sports Analytics Research Community
 
Use the methodology, particularly the proposed Hybrid Centrality concept, as a basis for further research.
 
---
 
## 13. Expected System Outputs
 
The system should ultimately produce more than a prediction score.
 
Expected outputs include:
 
1. Possession-level prediction
2. Important players
3. Important passes
4. Important subgraphs
5. XAI importance scores
6. Hybrid Centrality scores
7. Player rankings based on the proposed hybrid measure
8. Passing-network visualizations
9. Quantitative tactical comparisons
10. Interpretable football knowledge extracted from model explanations
---
 
## 14. Current End-to-End Architecture
 
The current conceptual architecture is:
 
### Stage 1 — Data
StatsBomb Open Data
→ raw event JSON
 
### Stage 2 — Preprocessing
Raw events
→ cleaned possession sequences
 
### Stage 3 — Graph Construction
Possession sequence
→ directed player-passing graph
 
### Stage 4 — Prediction
Possession graph
→ GCN / MPNN
→ probability of shot/goal
 
### Stage 5 — Explanation
Trained GNN + possession graph
→ GNNExplainer / PGExplainer
→ important nodes, edges and subgraphs
 
### Stage 6 — Hybrid Centrality
Traditional centrality + XAI importance
→ hybrid player-centrality score
 
### Stage 7 — Tactical Analysis
Hybrid Centrality + important subgraphs
→ quantitative team/match comparison
→ network visualization
→ interpretable tactical knowledge
 
---
 
## 15. Research Terminology
 
Use the following terminology consistently:
 
- **Possession sequence:** a sequence of events belonging to an attacking possession.
- **Passing network:** a graph representing passing relationships between players.
- **Node:** a player in the passing graph.
- **Edge:** a directed pass between players.
- **Graph-level prediction:** prediction made for an entire possession graph.
- **GNN:** Graph Neural Network.
- **GCN:** Graph Convolutional Network.
- **MPNN:** Message Passing Neural Network.
- **GNNExplainer:** graph-specific XAI method for identifying important graph components.
- **PGExplainer:** parametric graph explanation method.
- **Centrality:** network-based measure of structural importance.
- **Hybrid Centrality:** the proposed combination of traditional network centrality and XAI-derived importance.
- **Bottleneck:** a structurally important or constrained part of a team's passing network, to be operationally defined by the research.
- **Knowledge discovery:** extracting meaningful and interpretable tactical insights from the analyzed data and explanations.
---
 
## 16. Research Workflow Principles
 
When working on this project:
 
### Data
 
- Clearly document the source and structure of the football event data.
- Keep preprocessing reproducible.
- Clearly define how possession sequences are created.
- Clearly define the graph construction rules.
- Prevent information leakage between training, validation, and test data.
### Modeling
 
- Train and evaluate models experimentally.
- Use the same evaluation framework when comparing models.
- Clearly distinguish baseline models from the proposed method.
- Do not claim that a model is better without experimental evidence.
### XAI
 
- Explain what each XAI method is actually measuring.
- Distinguish model explanation from causal explanation.
- Do not interpret an XAI importance score as automatically proving that a player or pass causally caused an outcome.
- Validate explanations where possible.
### Hybrid Centrality
 
- Define the formula explicitly before implementation.
- Explain normalization and aggregation.
- Test whether the proposed score provides useful information beyond traditional centrality.
- Avoid presenting the proposed metric as established knowledge; it is a project contribution that must be experimentally validated.
### Tactical Analysis
 
- Prefer quantitative and reproducible comparisons.
- Combine numerical results with interpretable visualizations.
- Clearly distinguish observed data patterns from tactical interpretations.
---
 
## 17. Experimental Questions to Keep in Mind
 
The following questions should guide the research and experiments:
 
1. Can a GNN effectively predict whether a possession sequence leads to a shot or goal?
2. Which graph representation provides useful predictive information?
3. Which GNN architecture performs best under the chosen experimental setup?
4. Which players and passes are identified as important by graph XAI?
5. Are XAI-important components consistent across possessions?
6. Does Hybrid Centrality provide information that traditional centrality measures do not?
7. Can the proposed method distinguish meaningful tactical differences between teams or matches?
8. Can the resulting explanations be converted into reproducible tactical knowledge?
These are research questions to investigate, not conclusions.
 
---
 
## 18. What the Project Must NOT Assume
 
Do not automatically assume:
 
- That GNN will outperform non-graph models.
- That GCN is the final model.
- That MPNN is the final model.
- That GNNExplainer is better than PGExplainer.
- That the proposed Hybrid Centrality is valid before validation.
- That a highly important XAI feature is causally responsible for the outcome.
- That the reported research gap is proven to be globally exhaustive.
- That a particular team or player is tactically superior without experimental evidence.
- That a specific accuracy, F1, AUC, or other metric has already been achieved.
Any such claim must be supported by later experiments or reliable literature.
 
---
 
## 19. Current Known Scope From the Progress Report
 
The current scope is:
 
**Sport:** Football
 
**Primary structure:** Passing networks
 
**Data type:** Event data
 
**Primary data source named:** StatsBomb Open Data
 
**Competitions named:** Bundesliga 2023/2024 and UEFA Champions League
 
**Graph:** Directed passing graph
 
**Primary ML family:** Graph Neural Networks
 
**Candidate GNN models:** GCN / MPNN
 
**Primary XAI family:** Graph-specific XAI
 
**Candidate XAI methods:** GNNExplainer / PGExplainer
 
**Proposed analytical contribution:** Hybrid Centrality
 
**Final application:** Tactical analysis and football knowledge discovery
 
---
 
## 20. Relationship to the Original General Topic
 
The general registered topic remains:
 
**"Hệ thống trí tuệ nhân tạo có thể giải thích (XAI) trong bài toán đánh giá và khám phá tri thức thể thao."**
 
The current specific research direction is:
 
**"Mạng chuyền bóng và phân tích chiến thuật bóng đá bằng Graph Neural Network (GNN) kết hợp XAI."**
 
The current direction therefore keeps:
 
- XAI
- Sports analytics
- Knowledge discovery
while specializing the technical problem in:
 
- Football
- Passing networks
- Graph Neural Networks
- Graph-specific explainability
- Tactical analysis
---
 
## 21. Guidance for Future Project Conversations
 
When assisting with this project:
 
1. Treat the current specific direction as the primary research direction.
2. Use the progress report as the current project baseline until a newer project document supersedes it.
3. Preserve the distinction between:
   - what has already been completed,
   - what has been proposed,
   - what is still experimental,
   - and what is only a hypothesis or research question.
4. Do not invent experimental results.
5. Do not invent literature findings or citations.
6. If a claim requires external verification, explicitly identify it as something that needs literature verification.
7. When proposing a methodology, explain its role in the overall pipeline.
8. Keep the project centered on graph-based football analysis rather than drifting back to generic tabular player evaluation.
9. When discussing XAI, focus on explanations at the graph level where appropriate.
10. When discussing the proposed Hybrid Centrality, treat it as a research contribution that requires formal definition and empirical validation.
---
 
## 22. One-Sentence Project Summary
 
**XAI-Football develops an interpretable GNN-based framework that models football attacking possessions as directed passing graphs, predicts whether possessions lead to shots/goals, explains important players, passes and subgraphs using graph-specific XAI, and combines network centrality with explanation scores into a proposed Hybrid Centrality measure for quantitative tactical analysis and football knowledge discovery.**
 
---
 
## Source Note
 
This context is based primarily on the project's progress report dated **11/08/2026**. It records the research direction, motivation, identified research gaps, proposed system pipeline, datasets named in the report, candidate GNN/XAI methods, Hybrid Centrality concept, and intended applications.
 
Where the progress report does not specify an implementation detail or final mathematical definition, this context intentionally leaves it open rather than inventing a decision.