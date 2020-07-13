# Summarizing legal documents 

In this project, I was trying to study how campanies changed their terms and conditions as the years progressed.
Specifically, I wanted to study how companies responded to CCPA that was implemented earlier this year. 
Essentially, consumers now have the right to access their data ala (almost) GDPR-style.
Also, prepared they will be in lieu of a bill (COPRA) that is essentially an country-wide extension of CCPA.

Some companies' T&C I studied:
1) Facebook (2015, Pre-Apr 2018, After-Apr 2018, After-Jul 2019 -- each time their terms are revised). I'm especially curious how Facebook changed their term after the Cambridge-Analytica scandal.
2) Patreon (2020)
3) Robinhood (2020)


I experimented with a couple different summarizing models (both extractive & abstractive) to summarize legal documents. From that summary, I then rank the importance of each sentence using pagerank.
Models I've tried in this project:
* Bart & T5 from Hugging Face
* Bert from Bert-distill-summarizer library
* Pegasus (the new SOTA) from Google [Fine-tuned on a small legal summarization data set]

I ended up chosing Bert as it performs better (based on Rouge-L scores) out-of-the-box compared Bart & T5.
Pegasus, on the other hand, performs the best, but it's very slow (about 10 minutes on 2070 Ti). Moreover, the current implementation limits input to only 1024 tokens. 
So, Bert seems to be the way to go!

I've also built a knowledge graph, using Spacy to extract the triples and use networkX to draw the knowledge graph. 

For now, in this project,  we can eye-ball the important sentences extracted and see how the wordings about data privacy changed over the years. The two companies studied did overall become more stringent in regards to data privacy. Patreon, in particular, is taking a very pro-active approach, making sure their terms and conditions are in a digestible format even way before CCPA. Facebook, on the other hand, needs some improvements. Last they update their terms and condition was in Jul 2019. Given that CCPA was implemented Jan this year, we're not sure how CCPA-ready Facebook is.

Another interesting thing, although Facebook still leaves much to be desired, Facebook did noticably become more stringent about data privacy after the Cambridge Analytica Scandal in 2018. This is definitely most noticable in the knowledge graph drawn for pre-apr 2018 vs after-apr 2018.

#### On reproducing Pegasus
I can't upload the models and chkpoints on Github as they're huge!
But, steps to re-create:
1) Clone library on github and install requirements.

```
git clone https://github.com/google-research/pegasus
cd pegasus
export PYTHONPATH=.
pip3 install -r requirements.txt
```
2) Install [gsutil](https://cloud.google.com/storage/docs/gsutil_install)
3) Download vocab, pretrained and fine-tuned checkpoints of all experiments.
```
mkdir ckpt
gsutil cp -r gs://pegasus_ckpt/ ckpt/
```
4) Move fine-tune files into pegasus (ie all_v1.json, prep_fineT_data.py, summarize.ipynb)
5) Fine-tune setups
```
vim <your_path>/pegasus/params/public_params.py
```
And put this in the file
```
save_path = "<your_path>/pegasus/data/testdata/fine_tune_data.tfrecords"

@registry.register("fine_tune")
def test_transformer(param_overrides):
  return transformer_params(
      {
          "train_pattern": save_path,
          "dev_pattern": save_path,
          "test_pattern": save_path,
          "max_input_len": 1024,
          "max_output_len": 256,
          "train_steps": 180000,
          "learning_rate": 0.0001,
          "batch_size": 8,
      }, param_overrides)
```
6) Generate fine-tune data
```
cd <your_path>/pegasus
python3 prep_fineT_data.py
```

7) Fine tune the model 
WARNING: Don't use CPU, this already takes a very long time on a 2070 Ti
```
cd <your_path>/pegasus
python3 pegasus/bin/evaluate.py --params=fine_tune \
--param_overrides=vocab_filename=ckpt/pegasus_ckpt/c4.unigram.newline.10pct.96000.model,batch_size=1,beam_size=5,beam_alpha=0.6 \
--model_dir=ckpt/pegasus_ckpt/
```
8) Run Jupyter Notebook for summarize.ipynb and observe its summary (best performance by far when I evaluated on rouge-L score)

Despite Pegasus's amazing performance, I decided not to use it. Pegasus is slow and it take very long to summarize a document. Moreover, the current implementation
limits input to only 1024 tokens, which in my point defeats the purpose of document-level summarization. Thus, at least, until an quick interface is built for this model, Pegasus isn't the best choice.

## Files:
1) Summarizer.ipynb: Main notebook and analysis
2) Alt_summarizer.ipynb: Notebook to test out BART and T5
3) pegasus (cloned from https://github.com/google-research/pegasus): only pushed own notebook, fine-tune dataset and fine-tune data prep code. Model too big...
    *  all_v1.json: legal summarization (in plain english) data set
    *  prep_fineT_data.py: python file to create TFRecord for fine tuning tasks
    *  summarize.ipynb: notebook to 
4) etc: legal text to summarize


## References
```
@misc{zhang2019pegasus,
    title={PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization},
    author={Jingqing Zhang and Yao Zhao and Mohammad Saleh and Peter J. Liu},
    year={2019},
    eprint={1912.08777},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}

@inproceedings{manor-li-2019-plain,
    title = "Plain {E}nglish Summarization of Contracts",
    author = "Manor, Laura  and
      Li, Junyi Jessy",
    booktitle = "Proceedings of the Natural Legal Language Processing Workshop 2019",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/W19-2201",
    pages = "1--11",
}
```




