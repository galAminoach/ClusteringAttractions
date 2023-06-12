import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote

def download_image(url, destination):
        headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()
        with open(destination, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                                file.write(chunk)

attractions_set = {
    'Sheki', 'Ruta de las Flores', 'San José', 'Leon', 'Limbe', 'Meroe Pyramids', 'Nui', 'Sihanoukville',
    'Hungarian National Parks', 'Yasawa Islands', 'Everest Base Camp', 'Lake Assal', 'Chuuk', 'Gizo', 'Cartagena',
    'Ayutthaya', 'Nis', 'Bocas del Toro', 'Hill of Crosses', 'Finnish Lakeland', 'Kokoda Track', 'St. George\'s',
    'Golden Circle', 'Mayan Ruins', 'Nassau', 'Historic Center of San Marino', 'Dogon Country', 'Tripoli',
    'Underwater Sculpture Park', 'Copán Ruins', 'Kuang Si Falls', 'Harrison\'s Cave', 'Makasutu Cultural Forest',
    'Tobago Main Ridge Forest Reserve', 'Boa Vista', 'Ari Atoll', 'Georgetown', 'Blue Lagoon', 'Larvotto Beach',
    'Grand Anse Beach', 'Khor Al Adaid', 'Queenstown', 'Paro', 'Fogo', 'Mohéli Marine Park', 'Yaren', 'Sossusvlei',
    'Bijagós Archipelago', 'Red Sea coast', 'Djado Plateau', 'Fouta Djallon', 'Prince\'s Way', 'Demilitarized Zone (DMZ)',
    'The Pearl', 'Vaiaku Lagoon'
}


# save_directory = '/Users/galaminoach/Desktop/clusteringProject'
# not_found = []
# for key in attractions_set:
#     search_query = quote(key)
#
#     search_url = f'https://www.flickr.com/search/?text={search_query}'
#     print(search_url)
#     response = requests.get(search_url)
#     try:
#         response.raise_for_status()
#     except:
#         print(f"############ No page found for key: {key} ###############")
#         not_found.append(key)
#         continue
#     soup = BeautifulSoup(response.content, 'html.parser')
#
#     # Find the infobox element if available
#     infobox = soup.find('table', class_='infobox')
#     if infobox:
#             # Find the first image within the infobox
#             image_element = infobox.find('img')
#             if image_element:
#                     image_url = image_element['src']
#                     if image_url.startswith('//'):
#                             image_url = 'https:' + image_url
#                     image_destination = os.path.join(save_directory, f'{key}.jpg')
#                     download_image(image_url, image_destination)
#                     print(f"Downloaded image for key: {key}")
#             else:
#                     print(f"############ No image found for key: {key} ###############")
#                     not_found.append(key)
#
#     else:
#             print(f"############  No infobox found for key: {key} ############ ")
#             not_found.append(key)

#
# for key in not_found:
#         print(key)
from bing_image_downloader import downloader
for key in attractions_set:
    search_query = quote(key)
    downloader.download(search_query, limit=1,  output_dir='dataset', adult_filter_off=True, force_replace=False, timeout=60, verbose=True)