import imutils
import cv2
import json
import requests
import os
import re
import logging
import time
import numpy as np
import undetected_chromedriver as uc
from lxml import html
from selenium import webdriver
from tqdm import tqdm
from skimage.measure import compare_ssim

### May differ between AI platform runs and local runs
try:
    from scripts.utils import download_blob, upload_blob
except:
    from utils import download_blob, upload_blob

from config.local_settings import GENIUS_API_TOKEN

def get_page_of_songs(artist_name, page):
    """
    Given an artist name and a page, gather the returned songs from the genius API
    """
    base_url = 'https://api.genius.com'
    headers = {'Authorization': 'Bearer ' + GENIUS_API_TOKEN}
    search_url = base_url + '/search?per_page=10&page=' + str(page)
    data = {'q': artist_name}
    response = requests.get(search_url, data=data, headers=headers)
    return response

def get_artist_songs(artist_name, song_cap=1000):
    """
    Get all the songs for a given artist

    For each song, only grab the relevant fields
    """
    page = 1
    songs = []
    
    while True:
        response = get_page_of_songs(artist_name, page)
        json = response.json()
        song_info = []
        if not json['response']['hits']:
            break
        ### Add songs with artist to list
        for hit in json['response']['hits']:
            if artist_name.lower() in hit['result']['primary_artist']['name'].lower():
                song_info.append(hit)
    
        ### Collect song data from song objects
        for song in song_info:
            if (len(songs) < song_cap):
                url = song['result']['url']
                title = song['result']['title']
                album = song['result']['header_image_thumbnail_url']
                color = song['result']['song_art_primary_color']
                songs.append({'url': url, 
                                'color': color, 
                                'album': album, 
                                'title': title})
            else:
                break
        if (len(songs) >= song_cap):
            break
        else:
            page += 1
    
    logging.warning(f'Found {len(songs)} songs by {artist_name}')
    return songs

def get_artists_links(artist_list):
    """
    Get songs for each artist in list
    """
    artist_links = {}
    for artist in artist_list:
        logging.warning(f'Querying {artist}...')
        artist_links[artist] = get_artist_songs(artist)
    return artist_links

def get_song_lyrics(data, driver):
    """
    Use selenium driver to open song page and scrape lyrics
    """
    lyrics_dict = {}
    
    for song in data:
        link = song['url']
        driver.get(link)
        etree = html.fromstring(driver.page_source)
        lyrics = etree.xpath('//div[@class="Lyrics__Container-sc-1ynbvzw-6 krDVEH"]')
        lyrics = "\n".join(["\n".join([j.strip() for j in i.itertext() if j.strip()]) for i in lyrics])
        lyrics = re.sub('\[((.)|(\n))*?\]', '', lyrics)
        lyrics = re.sub('\n\n', '\n', lyrics).strip()
        song.update({'lyrics': lyrics})
    return data

def compute_sim_score(urlA, urlB):
    """
    Returns similarity score of images from two URLs
    """
    respA = requests.get(urlA, stream=True).raw
    imageA = np.asarray(bytearray(respA.read()), dtype="uint8")
    imageA = cv2.imdecode(imageA, cv2.IMREAD_COLOR)
    
    respB = requests.get(urlB, stream=True).raw
    imageB = np.asarray(bytearray(respB.read()), dtype="uint8")
    imageB = cv2.imdecode(imageB, cv2.IMREAD_COLOR)
    
    if imageA is not None and imageB is not None:
        imageA = cv2.resize(imageA, (300,300), interpolation = cv2.INTER_AREA)
        imageB = cv2.resize(imageB, (300,300), interpolation = cv2.INTER_AREA)
        # convert the images to grayscale
        grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

        (score, diff) = compare_ssim(grayA, grayB, full=True)
        diff = (diff * 255).astype("uint8")
        return score
    else:
        return 0

def group_similar_albums(artists_links, start_time):
    """
    Create album groups for all songs for each artist

    Album names are not in genius API, there is an album image but
        there are sometimes multiple different links to the same album cover.

    To work around this, we'll compare images within an artist, figure out which are the same then remove
        duplicate links, using the remaining link as the "name" of the album.

    This function will loop through all the unique album artwork links for each artist. 
        Each artist loop, album_groups[artist] is created and then the artist's albums are looped through.
        For each album not already in the dictionary, it is compared to the keys of the dictionary.
        If it matches an album, its added to that album's list
        If it does not, it a new entry in the artist dict is created for this album.

    At the end, all songs' albums are compared to the artist's album_groups[artist] dictionary,
        if the album is in one of the values in the dictionary, the song's artwork is changed
        the to key corresponding to that value.
    """

    artist_albums = {}
    album_groups = {}

    for artist, artist_songs in tqdm(artists_links.items(), desc='Grouping artists songs into albums'):
        ### Create this for later
        artist_albums[artist] = {}
        album_groups[artist] = {}

        all_albums = set([i.get('album') for i in artist_songs])
        ### Compares all albums from artist, creating groups of albums that are nearly the same
        for album in all_albums:
            ### If changed to True, will skip rest of album checks in loop
            skip_rest = False
            for key in album_groups[artist]:
                if skip_rest or album == key:
                    continue
                try:
                    sim_score = compute_sim_score(album, key)
                except:
                    sim_score = 0
                ### If sim score > 0.95, assume albums are same, add to list and skip
                ### rest of loop
                if sim_score > .95:
                    album_groups[artist][key].append(album)
                    skip_rest = True
            ### if cover doesn't match any existing options, make a new group
            if not skip_rest:
                album_groups[artist][album] = [album]      

        for album, albums in album_groups[artist].items():
            artist_albums[artist][album] = [s for s in artist_songs if s['album'] in albums]
        logging.warning(f'Finished {artist} after {time.time() - start_time} seconds')
    return artist_albums


def main():
    artist_list = ['Mac Miller', 
                   'Tyler, the Creator', 
                   'Action Bronson',
                   'Joji',
                   'Kid Cudi',
                   'Injury Reserve',
                   'Amine',
                   'Smino',
                   'Justin Bieber',
                   'A Tribe Called Quest',
                   'Kanye West',
                   'Vince Staples',
                   'Earl Sweatshirt',
                   'MF DOOM',
                   'A Day to Remember',
                   'Neck Deep',
                   'Knuckle Puck',
                   'Childish Gambino',
                   'Eminem',
                   'Slipknot',
                   'The Wonder Years',
                   'Tenacious D',
                   'ScHoolboy Q',
                   'A$AP Rocky',
                   'The Notorious B.I.G.',
                   'Flatbush Zombies',
                   'Jack Johnson', 
                   'Hobo Johnson',
                   '100 Gecs',
                   'Rex Orange County',
                   'The Front Bottoms',
                   'Rich Brian',
                   'John Mayer',
                   'BROCKHAMPTON',
                   'Denzel Curry',
                   'JPEGMAFIA',
                   'Frank Ocean']
    start_time = time.time()
    ### Get links to all songs by each artist in list - Up to 1000 songs per artist
    artists_links = get_artists_links(artist_list, n_songs=1000)
    
    ### Initiate selenium
    options = webdriver.ChromeOptions()
    options.add_argument("headless")
    driver = uc.Chrome(options=options)

    ### For each artist, grab the lyrics for each of their songs
    for artist, data in artists_links.items():
        logging.warning(f'Querying lyrics for {artist}')
        ### Get lyrics for each artist's songs
        artists_links[artist] = get_song_lyrics(data, driver)

    driver.quit()

    ### Save the lyrics to a file
    with open('artist_lyrics_0607.json', 'w') as f:
        json.dump(artists_links, f, indent=4, default=str)

    upload_blob('artist_lyrics_0607.json', folder_name='data')

    ### Compare album artworks to group songs by album
    logging.warning(f'Grouping albums after {time.time() - start_time} seconds')
    artists_albums = group_similar_albums(artists_links, start_time)

    with open('artist_albums_lyrics_0607.json', 'w') as f:
        json.dump(artists_albums, f, indent=4, default=str)

    upload_blob('artist_albums_lyrics_0607.json', folder_name='data')

if __name__ == "__main__":
    main()