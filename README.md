# Summarizing legal documents 

In this project, I experimented with different sumamrizing models (both extractive & abstractive) to summarize legal documents. From that summary, I then rank the importance of each sentence using pagerank.
Some models I tried in this project:
* Bart & T5 from Hugging Face
* Bert from Bert-distill-summarizer library
* Pegasus (the new SOTA) from Google [Fine-tuned on a small legal summarization data set]

I ended up chosing Bert as it performs better (based on Rouge-L scores) out-of-the-box compared Bart & T5.
Pegasus, on the other hand, performs the best, but it's very slow (about 10 minutes on 2070 Ti). Moreover, the current implementation limits input to only 1024 tokens. 
So, Bert seems to be the way to go!

I've also built a knowledge graph, using Spacy to extract the triples. 

In this project, I was trying to study how campanies changed their terms and conditions as the years progressed.
Specifically, I wanted to study how companies responded to COPRA that was implemented earlier this year. 
Essentially, consumers now have the right to access their data ala (almost) GDPR-style.

Some companies I studied:
1) Facebook (2015, 2018, 2019 -- each time their terms are revised). I'm especially curious how Facebook changed their term after the Cambridge-Analytica scandal.
2) Patreon


For now, in this project,  we can eye-ball the important sentences extarcted and see how the wordings about data privacy changed to become more stringent after 2016.
We can see this even clearer in the knowledge graph, we 

## Files:
1) Summarizer.ipynb

