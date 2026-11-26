# Earnings call analysis

This project parses earnings calls into speaker turns, summarizes management commentary and retrieves transcript passages for investor questions.

- [Analysis notebook in Colab](https://colab.research.google.com/drive/14vZUU0VbiyV78cjWfgHkPi4-oK0GqWp7)
- [Project report](report.pdf)
- [Failure journal](failure-journal.pdf)

## Retrieval evaluation

Results on 20 evaluation questions. Hit@k is the percentage of questions for which a relevant passage appeared among the first k results.

| Retrieval | Chunk size (words) | Hit@1 | Hit@3 | Mean reciprocal rank |
|---|---:|---:|---:|---:|
| Dense | 500 | 60% | 65% | 0.625 |
| Dense | 220 | 75% | 85% | 0.800 |
| Hybrid | 220 | 80% | 95% | 0.867 |

## Examples from development

The roster parser identified “Goldman Sachs” as a person. The revised parser first finds speaker labels in the body of the call and then looks for those speakers in the roster.

The chatbot initially produced bullet points. After an adjustment intended to elicit connected prose, it struggled to answer even simple questions. Retrieval metrics above measure passage ranking, so they cannot establish whether an answer is correct.
