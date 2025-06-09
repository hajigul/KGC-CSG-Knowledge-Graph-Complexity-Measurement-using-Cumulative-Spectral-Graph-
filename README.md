# KGC-CSG-Knowledge-Graph-Complexity-Measurement-using-Cumulative-Spectral-Graph-

KGC-CSG is a research framework designed to measure the structural complexity of knowledge graph datasets using Cumulative Spectral Graph (CSG) analysis. The method leverages spectral graph theory and semantic embedding to quantify how complex and diverse different knowledge graphs are, with a focus on understanding their suitability for machine learning tasks such as link prediction and knowledge graph completion.  


Key Features:  
•	Converts knowledge graph triples into class-based similarity graphs using tail-entity grouping.  
•	Embeds (head, relation) pairs using pre-trained BERT or Sentence-BERT models.  
•	Computes pairwise distances to generate a similarity matrix.  
•	Constructs graphs from similarity matrices and applies graph Laplacian decomposition.  
•	Measures dataset complexity via Cumulative Spectral Gap (CSG), providing interpretable spectral signatures.  


Applications:  
•	Benchmarking and comparing knowledge graph datasets  
•	Complexity-based selection of datasets for link prediction  
•	Understanding training difficulty and diversity in graph-structured data  


Datasets Supported:  
•	FB15k-237  
•	WN18RR  
•	Codex-S  
•	Nations  

