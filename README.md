# lyrical_clustering

## Overview

The code in this repo can be used to scrape lyrics from genius, embed the lyrics with an S-BERT model, reduce the dimensionality of the embeddings then plot the embeddings on a 2d plot. The embeddings of each song will be marked on the plot using the song's album cover. An image for each artist will be plotted at the center point of all the artist's songs. 

<p align="center">
<img src="https://i.imgur.com/qdaxIMw.png"  width="800px"></img>
</p>

## How to run

### Local Run

1. Edit the artist list in `scrape_genius.py` so it includes your artists. 
2. Run `scripts.scrape_genius.main()` to scrape the lyrics.
    - Be aware, this may take a long time as each song from each artist needs to be queried individually.
    - Running on the cloud can be difficult bc the scraping makes so many requests. 
    - This job isn't too heavy so you should be able to run it locally. 
3. In `scripts.generate_embeddings.main()`, modify the file locations the embeddings are loaded from then run the script. 
    - This step is MUCH faster / may only be possible with a GPU or a computer with a ton of RAM. I would not recommend the latter as it'll still be very slow on a CPU.
    - I have included code to submit jobs to Google Cloud's AI Platform. I'd highly recommend using it for this as its fairly cheap and not too difficult. 
4. Download an image for each artist in your query. Add the locations to the `scripts.make_plot.main()` function.
5. Run `scripts.make_plot.main()` to generate a plot. 

### Running on GCP

Before you start,
  - Make sure you have a GCP project with storage and AI platform activated. 
  - I authenticated my job requests using a local service account file so to use this code exactly, you'll need to do the same. 
    - To do this, create a service account and then download its credentials as a JSON. 
  - Create a storage bucket for your files to go to and maybe a folder for your compressed code files. 

#### How to run on GCP

0. Edit `setup.py` if you want to modify the package name for version control or other reasons. 
1. Use `python3 setup.py sdist` to create a `.tar.gz` archive of the code.
2. In the newly created directory, `dist`, find the file that will be named `lyrical_clustering-0.1.tar.gz`.
3. Upload that file to GCS. 
4. Add that file location to the `packageUris` parameter in `submit_jobs/submit_jobs.py`. 
5. You will also need to modify the `project_id` variable and the `jobDir` request parameter to refer to your own project and bucket/folder locations. 
6. With this all done, one can run one of the two functions on the in `submit_jobs/submit_jobs` to run a job.
    - `scrape_lyrics_Task()` will submit a job to scrape lyrics from Genius.
    - `gen_embeddings_Task()` will submit a job to generate embeddings of scraped lyrics.
```
python3 -c "from submit_jobs.submit_jobs import gen_embeddings_Task; gen_embeddings_Task()"
```

If you have [gsutil](https://cloud.google.com/storage/docs/gsutil) installed and configured, you can update the tar.gz archive and  the file in GCS with one command. 

```
python3 setup.py sdist && \
gsutil cp dist/lyrical_clustering-0.1.tar.gz gs://BUCKET/CODE_FOLDER
```

The location `gs://BUCKET/CODE_FOLDER` will be used as the beginning of the `jobDir` parameter in `submit_jobs/submit_jobs.py`.

---

## What's happening

### High-level

The goal of this repo was to generate 2d representations of songs' lyrics so I could plot songs and see which was similar. My goal  was to generate the image displayed above but there is a lot more that could be done with latent representations of song lyrics. 

### Details

I took each song's lyrics, tokenized them and then converted the tokenized text to latent vectors using a pre-trained S-BERT model from HuggingFace. Specifically, I used the [sentence-transformers/bert-base-nli-mean-tokens](https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens) model and its documentation. 

[S-BERT](https://arxiv.org/abs/1908.10084) is a BERT model trained to generate latent vectors that can be used for calculatinng semantic similarity scores and text clustering. After running inference with this model on tokenized text, S-BERT will return a 768d vector for each token in the text. This series of vectors can be reduced to a single 768d vector with mean pooling. 

Once each song is represented by a single vector, I used PCA to reduce each vectors dimensionality down to 2d. These values were used as coordinates in a plot to visualize the content of the lyrics in 2 dimensions. 

I will explain the process in more depth in a future blog post (will add here when its ready).
