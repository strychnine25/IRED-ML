# IRED-ML

# Introduction
Imine reductases (IREDs) are a class of NAD(P)H-dependent oxidoreductase enzymes which catalyse the reduction of imine substrates to the corresponding amine.<sup>1</sup> Cyclic imines are the archetypal substrate class these enzymes accept,<sup>2</sup> with a range of ring sizes being tolerated. Imines/iminium substrates formed _in situ_ (_via_ a condensation reaction between an amine and ketone) can also be reduced by IREDs, but suffer from lower efficiency and require large excesses of the amine component.<sup>3</sup> Reduction of  prochiral C=N bonds using IREDs can be highly enantioselective, with access to both enantiomers of a substrate possible using different IRED enzymes. The ability to affect enantioselective 
C–N bond formation in a mild and environmentally-begnin manner has attracted attention from the chemical industries, as chiral amines are a priveleged motif in societally important commodities such as pharmaceuticals and agrochemicals - over 40% of drugs contain a chiral amine.<sup>4</sup> Highlighting a recent high-profile example, a genetically engineered IRED was employed in the synthesis of an intermediate on preparative scale (1.4 kg yield (84.4%, 99.7% _e.e._) across three batches) _en route_ to LSD1-inhibitor GSK2879552.<sup>5</sup>
# IRED-ML

# Introduction
Imine reductases (IREDs) are a class of NAD(P)H-dependent oxidoreductase enzymes which catalyse the reduction of imine substrates to the corresponding amine.<sup>1</sup> Cyclic imines are the archetypal substrate class these enzymes accept,<sup>2</sup> with a range of ring sizes being tolerated. Imines/iminium substrates formed _in situ_ (_via_ a condensation reaction between an amine and ketone) can also be reduced by IREDs, but suffer from lower efficiency and require large excesses of the amine component.<sup>3</sup> Reduction of  prochiral C=N bonds using IREDs can be highly enantioselective, with access to both enantiomers of a substrate possible using different IRED enzymes. The ability to affect enantioselective 
C–N bond formation in a mild and environmentally-begnin manner has attracted attention from the chemical industries, as chiral amines are a priveleged motif in societally important commodities such as pharmaceuticals and agrochemicals - over 40% of drugs contain a chiral amine.<sup>4</sup> Highlighting a recent high-profile example, a genetically engineered IRED was employed in the synthesis of an intermediate on preparative scale (1.4 kg yield (84.4%, 99.7% _e.e._) across three batches) _en route_ to LSD1-inhibitor GSK2879552.<sup>5</sup>

Although IREDs are now relatively well understood, owing to the experimental research efforts by structural biologists and protein engineers, they have not yet been the subject of machine learning pursuits. Combining existing experimental results/observations with machine learning models could increase the scientific knowledge around IRED enzymes that may not be possible by human inference. The high-throughput fashion in which biocatalysis research is routinely performed sets the stage perfectly for implementing machine learning techniques, as large amounts of data are already available in the literature for IRED-catalysed reactions inspite of this area of biocatalysis only reaching maturity.<sup>6</sup>

# Objectives
During a 12-week rotation project, the aim was to develop machine learning (ML) models (supervised learning) which can confidently predict reaction outcomes of IRED-catalysed reactions. 

Initially, the reaction outcome of interest is stereoselectivity - which enantiomer will be formed? This important, albeit simple question provides the first opportunity to build and subsequently develop machine learning models to shed light on if IRED protein sequence can be used as a predictor variable. Here, ML models will be trained on protein sequence data that has been transformed in some way and reaction outcomes. 

# Results Summary
First, k-nearest neighbours (kNN) and logistic regression (logreg) models within sci-kit learn (sklearn) were [built](https://github.com/strychnine25/IRED-ML/blob/main/Embedder%20exploration/SeqVec/SeqVec.ipynb) using the embedded (Bio Embeddings, SeqVec) sequences of IRED enzymes employed in the IREDy-to-go panel that showed some level of stereoselectivity (208 data points) toward a given substrate.<sup>7</sup> The performance of these two models was evaluated using three metrics: accuracy, 10-fold cross-validation, and ROC-AUC score (receiver operating characteristic-area under curve). Accuracy scores were good, 73% and 80% for kNN and logreg respectively, but ROC-AUC scores were poor: kNN, 0.60; logreg, 0.69.

To investigate whether different embedding models provided by Bio Embeddings could improve model performance, two models ([ProtTransBert_BFD](https://github.com/strychnine25/IRED-ML/blob/main/Embedder%20exploration/ProtTrans/prottrans_bert_bfd.ipynb) and [UniRep](https://github.com/strychnine25/IRED-ML/blob/main/Embedder%20exploration/UniRep/UniRep.ipynb)) were tested and compared to SeqVec. For this investigation, the number of ML models was increased to 6 (soft & hard voting classifier ensembles, decision tree bagging classifier, and XGBoost - all from sklearn) and the number of evaluation metrics increased to 5 (f-1 score and logloss). SeqVec remained the embedding model of choice based on overall model performance, but widening the ML model scope revealed XGBoost to be a superbly performing model for predicting the stereochemical outcome. An accuracy of 85% was achieved in addition to a logloss score of 5.31 - for comparison, kNN and logreg models both scored 7.31 on logloss.

Using a computational modelling workflow, whereby a substrate was docked in an IRED enzyme from the IREDy-to-go panel and a residue search performed within a given number of Angstroms, the IRED active site sequences could be predicted. Using these putative active site sequences, ML models were [built](https://github.com/strychnine25/IRED-ML/blob/main/Active%20site%20data%20exploration/SeqVecEmbedder/SeqVec%20active%20site%20sequences.ipynb) as described using SeqVec embedding above and evaluated across the same metrics. Model performance increased significantly, notably for logreg: accuracy, 89%; f-1 score, 0.93; logloss, 3.99. A proposed explanation for this observed improvement in model performance is that the active site sequence dataset contains less noise, i.e. fewer insignificant residues, than the full sequence dataset. Note, models using active site sequence data were also built for ProtTrans and UniRep so to perform a thorough evaluation, but there was no observed improvement in model performance.

With competantly performing models built, attention was turned to trying to ascertain which amino acid residue(s) were important, and possibily deterministic in the prediction of stereoselectivity for a given sequence. SHAP (**SH**apley **A**dditive ex**P**lanations) is a tool<sup>8</sup> that can be used to explain the output of machine learning models, extracting important features from the model by virtue of a game theoretic approach. Important features are determined by the magnitude of the effect they impose on model performance if they are peturbed. However, a different method for transforming the protein sequence string into some numerical form was required, as an embedded protein sequence cannot be worked back to individual amino acids from important features in the data. An amino acid level featurisation method was thus required, and so ZScales was employed. ZScales is a 5-component vector (Figure 1) consisting of measured physicochemical properties,<sup>9</sup> with each amino acid having a distinct vector. 

![image](https://user-images.githubusercontent.com/78608646/112637404-08167600-8e36-11eb-9be3-54081935cdde.png)

**Figure 1: Examples of the 5-component descriptors used to featurise IRED sequences at the amino acid level**

The 384 metagenomic IRED sequences were aligned, then the ZScales vectors applied to each residue in the sequences. Note, an additional component was added to each vector to account for whether the residue was a gap or an amino acid - as gaps are artefacts of the aligning process. Sequence length post-alignment was 1000, and so a 6-component vector applied to each residue produced a total length of 6000, i.e. 6000 individual features. [SHAP was used on an XGBoost model](https://github.com/strychnine25/IRED-ML/blob/main/Explaining%20the%20models/Full%20sequence/SHAP%20-%20XGBoost%20model.ipynb), 9 of the most important features were output and the corresponding amino acid traced back simply dividing by 6 and rounding down.

The output of the using SHAP is shown in figure 2. The amino acid residues at each important position were obtained for every protein sequence which displayed activity/stereoselectivity and compared between (_R_)- and (_S_)-selective enzymes. Unfortunately, there was no stark difference in AA frequency between IREDs of opposing stereoselectivity at any of the positions suggested by SHAP.

![image](https://user-images.githubusercontent.com/78608646/112637502-23818100-8e36-11eb-87e9-bb05f6877f07.png)

**Figure 2: Output for SHAP analysis on XGBoost model**

The same process was applied to the [active site sequences](https://github.com/strychnine25/IRED-ML/blob/main/Explaining%20the%20models/Active%20site%20sequences/SHAP%20-%20XGBoost%20model.ipynb) (_vide supra_), seeking to suppress noise in the data. But again no meaningful relationshpis were found between (_R_)- and (_S_)-selective IREDs in the features proposed by SHAP. In fact, the most frequent residues in sequence positions proposed by SHAP were gaps.

## Software and tools employed
Python (3.8.5)
<br>
Jupyter Notebook (6.2.0)
<br>
[Bio Embeddings](https://docs.bioembeddings.com/v0.1.6/#) (0.1.5)
<br>
[SciKit-Learn](https://scikit-learn.org/stable/index.html) (0.24.1)
<br>
[Pandas](https://pandas.pydata.org/) (1.2.3)
<br>
[Seaborn](https://seaborn.pydata.org/index.html) (0.11.1)
<br>
[Biopython](biopython.org) (1.78)
<br>
[Clustal Omega](https://www.ebi.ac.uk/Tools/msa/clustalo/)

## References
1. N. J. Turner _et al._, _Curr. Opin. Chem. Biol._, 2017, **37**, 19-25
2. T. Nagasawa _et al._, _Org. Biomol. Chem._, 2010, **8**, 4533-4535
3. (a) B. Nestl _et al._, _ChemCatChem_, 2015, **7**, 3239-3242; (b) D. Wetzl _et al._, _ChemCatChem_, 2016, **8**, 2023-2026
4. N. J. Turner _et al._, Top. Catal., 2014, **57**, 284-300
5. G.-D. Roiban _et al._, _Nat. Catal._, 2019, **2**, 909-915
6. The first IRED-catalysed imine reduction was disclosed only 10 years ago.
7. N. J. Turner _et al._, _Nat. Chem._, 2021, **13**, 140-148
8. arXiv preprint arXiv:1705.07874, 2017
9. J. Wold _et al._, _J. Med. Chem._, 1998, **41**, 2481-2491
