# Metadata to KG Track Round 1 Datasets

In this round, a JSONL file is provided with each line representing a column in a table, along with table name, column name, and other columns in the same table. The goal is to map each such column to one DBpedia ontology property (which we also refer to as a "glossary" item). We have also provided the metadata and the glossary in the form of an OWL ontology, to facilitate the mapping using ontology matching tools.

Sample data:
- [Sample Metadata File in JSONL](r1_sample_metadata.jsonl)
- [Sample Metadata Ontology in OWL](r1_sample_metadata.owl)
- [Sample Ground Truth](r1_sample_GT.csv)
- [Sample Output of Mapping](r1_sample_output.jsonl)
  - Note: The mappings array will be sorted in descending order by score.

Test data (what we expect you to map to the glossary):
- [Metadata File in JSONL](r1_test_metadata.jsonl)
- [Metadata Ontology in OWL](r1_test_metadata.owl)

Glossary:
- [Glossary in JSONL](r1_glossary.jsonl)
- [Glossary in OWL](r1_glossary.owl)
  - Note that this is a representation of all the DBpedia properties as classes with one label and one description only. You can also use the full DBpedia OWL ontology that can be found [here](https://github.com/dbpedia/dbpedia/tree/master/ontology) but you need to ensure that you only map to properties.

Evaluation script:
- To try the evaluation script over the provided sample input/output, go to the data folder and run:
```
python evaluate.py -m r1_sample_output.jsonl -g r1_sample_metadata_GT.csv
```

## Citations

Our paper introducing the problem:
```
@inproceedings{LoboHPMSS23,
  author       = {Elita A. Lobo and
                  Oktie Hassanzadeh and
                  Nhan Pham and
                  Nandana Mihindukulasooriya and
                  Dharmashankar Subramanian and
                  Horst Samulowitz},
  title        = {Matching table metadata with business glossaries using large language
                  models},
  booktitle    = {Proceedings of the 18th International Workshop on Ontology Matching
                  co-located with the 22nd International Semantic Web Conference {(ISWC}
                  2023), Athens, Greece, November 7, 2023},
  series       = {{CEUR} Workshop Proceedings},
  volume       = {3591},
  pages        = {25--36},
  publisher    = {CEUR-WS.org},
  year         = {2023},
  url          = {https://ceur-ws.org/Vol-3591/om2023\_LTpaper3.pdf},
}
```

SemTab 2024 summary paper:

```
@inproceedings{semtab2024results,
  author       = {Oktie Hassanzadeh and
                  Nora Abdelmageed and
                  Marco Cremaschi and
                  Vincenzo Cutrona and
                  Fabio D’Adda and
                  Vasilis Efthymiou and
                  Benno Kruit and
                  Elita Lobo and
                  Nandana Mihindukulasooriya and
                  Nhan H. Pham},
  title        = {Results of SemTab 2024},
  booktitle    = {Proceedings of the Semantic Web Challenge on Tabular Data to Knowledge
                  Graph Matching, SemTab 2024, co-located with the 23rd International
                  Semantic Web Conference, {ISWC} 2024, Baltimore, US, November 11-15,
                  2024},
  series       = {{CEUR} Workshop Proceedings},
  publisher    = {CEUR-WS.org},
  year         = {2024},
}
```