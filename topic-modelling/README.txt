This portion of the project work is done by mrh9

Datasets 
Downloaded from https://zenodo.org/records/394138
-15 specific mental health support groups (r/EDAnonymous, r/addiction, r/alcoholism, r/adhd, r/anxiety, r/autism, r/bipolarreddit, r/bpd, r/depression, r/healthanxiety, r/lonely, r/ptsd, r/schizophrenia, r/socialanxiety, and r/suicidewatch)
-scraped using pushshift.io


For the purpose of this project, data extracted by the original authors were excluded. 
We only use the posts,data,subreddit columns for our analysis. 
Citation
Low, D. M., Rumker, L., Torous, J., Cecchi, G., Ghosh, S. S., & Talkar, T. (2020). Natural Language Processing Reveals Vulnerable Mental Health Support Groups and Heightened Health Anxiety on Reddit During COVID-19: Observational Study. Journal of medical Internet research, 22(10), e22635.

@article{low2020natural,
  title={Natural Language Processing Reveals Vulnerable Mental Health Support Groups and Heightened Health Anxiety on Reddit During COVID-19: Observational Study},
  author={Low, Daniel M and Rumker, Laurie and Torous, John and Cecchi, Guillermo and Ghosh, Satrajit S and Talkar, Tanya},
  journal={Journal of medical Internet research},
  volume={22},
  number={10},
  pages={e22635},
  year={2020},
  publisher={JMIR Publications Inc., Toronto, Canada}
}

License
This dataset is made available under the Public Domain Dedication and License v1.0 whose full text can be found at: http://www.opendatacommons.org/licenses/pddl/1.0/

It was downloaded using pushshift API. Re-use of this data is subject to Reddit API terms.


To test this part of the project, run the python notebooks on google colab. 
1. ExploratoryDataAnalysis.ipynb - data exploration and intitial descriptive and inferential analytics while also testing data visualizations
2. TopicModelling.ipynb - contains notebook with experimentation and hyperparameter tuning of LDA model and Bisecting K means using pysparkMLlib for topic modeling. 
3. ETL.ipynb - contains data transformations for analysis and visualization purposes. 
