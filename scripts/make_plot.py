import requests
import json
import os
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from PIL import Image, ImageEnhance
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def load_data():
	"""
	Load embeddings generated with `generate_embeddings.py`

	Create list of all embeddings, transform with 2 component PCA

	Add PCA embeddings to songs, export all_songs
	"""
	
	all_songs = []
	all_embs = []
	
	with open('data/artist_albums_lyrics_embs_0608.json', 'r') as f:
		data = json.load(f)

	### Flatten nested data
	for artist, albums in data.items():
		for album, songs in albums.items():
			for song in songs:
				song_copy = copy.deepcopy(song)
				song_copy['artist'] = artist
				song_copy['embedding'] = song_copy['embedding'][0]
				all_embs.append(song_copy['embedding'])
				all_songs.append(song_copy)

	pca = PCA(n_components=2)
	pca_embs = pca.fit(all_embs).transform(all_embs)

	for song, pca_emb in zip(all_songs, pca_embs):
		song.update({'pca_emb': pca_emb.tolist()})
	
	all_songs = [i for i in all_songs if len(i['lyrics']) > 275]
	return all_songs


def getImage(path, size=(100,100), fade=False):
	"""
	Takes an image path and returns an `OffsetImage` object.

	If `fade`, reduce alpha value. 
		Used to fade artist pictures so they don't block albums
	"""
	try:
		img = Image.open(path)
	except:
		return None

	if img.mode != 'RGBA':
		img = img.convert('RGBA')
	else:
		img = img.copy()
	
	img.thumbnail(size)
	
	if fade:
		alpha = img.split()[3]
		alpha = ImageEnhance.Brightness(alpha).enhance(.7)
		img.putalpha(alpha)
	
	a = np.asarray(img)
	return OffsetImage(a)


def make_plot(all_songs):
	"""
	Plot all songs using the PCA coordinates derived from the pooled embeddings
		for each song.

	Plot artist images at the center of the coordinates of the artist's songs
	"""
	all_songs = [i for i in all_songs if i['lyrics']]
	artist_names = set([song['artist'] for song in all_songs])

	fig = plt.figure(figsize=(50,40),)
	ax = fig.subplots()

	plt.axis('off')

	sns.despine()
	
	artists_images = [('BROCKHAMPTON', 'brock.png'), 
			  ('Frank Ocean', 'frank.png'), 
			  ('The Front Bottoms', 'frontbottoms.png'), 
			  ('Rich Brian', 'brian.png'), 
			  ('Action Bronson', 'action.png'),
			  ('Joji', 'joji.png'),
			  ('Rex Orange County', 'rex.png'),
			  ('Injury Reserve', 'ir.png'),
			  ('Smino', 'smino.png'),
			  ('Justin Bieber', 'jb.png'),
			  ('A Tribe Called Quest', 'atcq.png'),
			  ('Vince Staples', 'vs.png'),
			  ('Earl Sweatshirt', 'earl.png'),
			  ('MF DOOM', 'MF.png'),
			  ('The Notorious B.I.G.', 'big.png'),
			  ('Flatbush Zombies', 'flatbush.png'),
			  ('Jack Johnson', 'jack.png'), 
			  ('Hobo Johnson', 'hobo.png'),
			  ('100 Gecs', '100gecs.jpg'),
			  ('Denzel Curry', 'denzel.png'), 
			  ('JPEGMAFIA', 'jpegmafia.png'), 
			  ('John Mayer', 'mayer.png'), 
			  ('Tyler, the Creator', 'tyler.png'),
			  ('Amine', 'amine.png'),
			  ('Kanye West', 'kw.png'),
			  ('Mac Miller', 'mac.png')]


	### Plot songs
	for artist, image_path in artists_images:
		artist_pca_1 = [i['pca_emb'][0] for i in all_songs if artist == i['artist']]
		artist_pca_2 = [i['pca_emb'][1] for i in all_songs if artist == i['artist']]
		artist_albums = [i['album'] for i in all_songs if artist == i['artist']]

		### Download all the artist's album images if they are not found locally
		for album in set(artist_albums):
			if not os.path.exists(f'images/{album.split("/")[-1]}'):
				with open(f'images/{album.split("/")[-1]}', "wb") as f:
					f.write(requests.get(album).content)

		### Create mapping of album links to pixel arrays
		artist_albums_mapping = {album: getImage(f"images/{album.split('/')[-1]}", 
												size=(50,50))
									for album in set(artist_albums)}

		### For each song, plot album cover
		for x,y, album in zip(artist_pca_1, artist_pca_2, artist_albums):
			if artist_albums_mapping[album]:
				ab = AnnotationBbox(artist_albums_mapping[album], (y, x))
				ax.add_artist(ab)

	### Plot artists after songs so artists sit on top of album images
	for artist, image_path in artists_images:
		artist_pca_1 = [i['pca_emb'][0] for i in all_songs if artist == i['artist']]
		artist_pca_2 = [i['pca_emb'][1] for i in all_songs if artist == i['artist']]
		try:
			kmeans_model = KMeans(n_clusters=1).fit(pd.DataFrame(list(zip(artist_pca_1, artist_pca_2))))
		except:
			print(f'Failed to compute kMeans for {artist}')
			continue
		else:
			artist_center = kmeans_model.cluster_centers_[0]

			try:
				ab = AnnotationBbox(getImage(f"images/{image_path}", 
											size=(250,250), fade=True), 
									(artist_center[1], artist_center[0]), 
									frameon=False)
				ax.add_artist(ab)
			except: ### Should not be any failures but catch them here
				print(f"Failed to plot {artist}")
				print(__repr__(e))
	
	### Set axis limits with min/max PCA values
	artist_pca_1 = [i['pca_emb'][0] for i in all_songs]
	artist_pca_2 = [i['pca_emb'][1] for i in all_songs]
	min_y, max_y = min(artist_pca_1), max(artist_pca_1)
	min_x, max_x = min(artist_pca_2), max(artist_pca_2)

	ax.set_ylim((min_y, max_y))
	ax.set_xlim((min_x, max_x))

	plt.savefig('plot.png', transparent=True)


def main():
	all_songs = load_data()
	print('Loaded data, making plot')
	make_plot(all_songs)