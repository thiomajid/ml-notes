## Known isssues
- Can't reproduce the $u_{mass}$ figure
- Using 10 topics produces weird results
- When rerunning the same code on the database, we get different results. Apologetic answers count goes from 558 to 447 even with the same SQL query (same for non apologetic ones).

## To Do
- Use up to 30 keywords to measure the overlap of topics (the set intersection thing from last time)
- Rework on the data processing section by explaining the RegEx, contraction, etc... parts to give more details about the process that lead us to a different dataset while working on the experiments.
  Write a convincing section explaining why the dataset changed leading us to different results.
- Use the picture showing keywords distribution for prompts and answers just before the LDA part to show that the dataset is composed of relevant tokens.
- Use the CV pictures for each analysis.
- Remove the u_mass part and replace it with c_v.
- Propose new titles for each topic using ChatGPT + manual validation
- Add screenshots of the PCA (LDA viz), explain the thing and add citations for salience and relevance.
- Non apologetic answers need more investigation to find a suitable number of topics.
- Create a CSV file containing each document's answer's max topic and its prompt's max topic.