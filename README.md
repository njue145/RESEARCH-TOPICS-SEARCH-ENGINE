# RESEARCH-TOPICS-SEARCH-ENGINE
# Research Topic Similarity Detection System

## Overview

The Research Topic Similarity Detection System is a web-based application developed to assist faculty members, research coordinators, and students in identifying potential similarities between proposed research titles and previously approved research projects.

The system utilizes Natural Language Processing (NLP) techniques and semantic similarity analysis to compare a submitted research topic against a repository of existing research titles. The objective is to reduce topic duplication, encourage originality, and support the research approval process.

---

## Key Features

* Research topic similarity detection
* Semantic comparison using NLP techniques
* Similarity score calculation in percentage form
* Risk classification of submitted topics
* Top matching research topics display
* Downloadable similarity results
* Interactive visualization of similarity rankings
* User-friendly web interface built with Streamlit

---

## Similarity Risk Classification

The current version of the system uses the following similarity thresholds:

| Similarity Score | Risk Level                             | Recommendation                                                                                          |
| ---------------- | -------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| 80% and above    | High Risk                              | Topic is highly similar to an existing research study. Revision or replacement is strongly recommended. |
| 60% – 79%        | Medium Risk (Major Revisions Required) | Topic contains substantial similarities and should undergo significant refinement before approval.      |
| Below 60%        | Low Risk                               | Topic appears sufficiently distinct and may proceed for further evaluation.                             |

### Recent Update

The classification thresholds were revised to better align with faculty evaluation requirements:

* Previous threshold values have been adjusted.
* High Risk classification now begins at **80% similarity**.
* Medium Risk classification now covers **60% to 79% similarity** and is labeled **"Major Revisions Required."**
* Similarity scores below 60% are considered **Low Risk**.

This adjustment provides a more practical interpretation of topic similarity and reduces the likelihood of false-positive duplication assessments.

---

## Intended Use

The system is designed as a decision-support tool and should not be used as the sole basis for research approval decisions. Faculty members are encouraged to review similarity results alongside research objectives, methodologies, and scope before making final evaluations.

---

## Limitations

* Similarity scores are based primarily on research titles and available metadata.
* The system does not evaluate full research manuscripts.
* Semantic similarity may identify conceptually related topics even when titles differ significantly.
* Human judgment remains necessary during the final approval process.

---

## Developer Information

Developed by: Moses Njue

Version 1.0

© 2026 All Rights Reserved
