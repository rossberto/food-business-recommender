from dotenv import load_dotenv
load_dotenv()

import os
from flask import Flask, request
import flask

import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
from pytrends.request import TrendReq
from sklearn.linear_model import LinearRegression
import requests
from bs4 import BeautifulSoup as bs
import json
from pandas.io.json import json_normalize
import foursquare
import folium

app = Flask(__name__)

'''
    VARIABLES
'''
foods = ['Cocina Económica', 'Antojitos', 'Pizza', 'Hamburguesas', 'Hot Dogs', 'Sushi', 'Tamales', 'Mariscos',
         'Pescado', 'Tacos', 'Carne', 'Asada', 'Panuchos', 'Cochinita', 'Pollo', 'Desayunos', 'Tortas', 'Mondongo',
         'Menudo', 'Memelas', 'Empanadas', 'Chicharrón', 'Gorditas', 'Costillas', 'Carnitas', 'Doraditas',
         'Baguettes', 'Parrilla', 'Huaraches', 'Rosticería', 'China', 'Yucateca', 'Tabasqueña', 'Arrachera',
         'Pastor', 'Birria', 'Barbacoa', 'Carnero', 'Pozole', 'Enchiladas', 'Chilaquiles']

'''
    INEGI Methods
'''
def prepareInegi():
    inegi = pd.read_csv('inegi.csv')
    inegi.drop(columns=['nom_estab', 'per_ocu', 'fecha_alta', 'nom_processed', 'tipoUniEco', 'localidad'], inplace=True)
    return inegi

def getInegiByState(estado, inegi):
    return inegi[inegi['entidad']==estado]

def inegiFilterByMunicipio(municipio, df):
    return df.drop(df[df['municipio']!=municipio].index)

def getInegiResults(df):
    results = {}

    for food in foods:
        results[food] = df[food].sum()

    return results

def inegiTask(estado, municipio):
    inegi = prepareInegi()
    inegi = getInegiByState(estado, inegi)
    inegi = inegiFilterByMunicipio(municipio, inegi)
    return getInegiResults(inegi)

'''
    DF INEGI
'''
def getInegiDfResult(inegi_results):
    return pd.DataFrame(inegi_results.items(), columns = ['Index', 'InegiCount']).set_index('Index')

'''
    USER LOCATION - GEOPY
'''
def userLocationGeocoding(string):
    geolocator = Nominatim(user_agent="food-business-recommender")
    return geolocator.geocode(string)

'''
    GOOGLE TRENDS
'''
def getGoogleTrends():
    pytrends = TrendReq(hl='es-MX', tz=360)

    resultados = []
    for comida in foods:
        print(comida)
        pytrends.build_payload([comida], cat=71, geo='MX', gprop='')
        resultados.append(pytrends.interest_by_region(resolution='REGION', inc_low_vol=True, inc_geo_code=False))
    return resultados

def getTrends(google_trends, state):
    requests = google_trends[0].join(google_trends[1:]).T
    return requests['Yucatán'].sort_values(ascending=False)

'''
    INTEREST OVER TIME
'''
def getGoogleIOT():
    pytrends = TrendReq(hl='es-MX', tz=360)

    iot = {}
    for comida in foods:
        print(comida)
        pytrends.build_payload([comida], cat=71, geo='MX', gprop='')
        iot[comida] = pytrends.interest_over_time()

    return iot

def getTrendSlope(iot):
    x = [i+1 for i in range(len(iot['Desayunos']['Desayunos']))]
    linreg = LinearRegression()

    slope = {}
    for food in foods:
        linreg.fit(np.array(x).reshape(-1,1), iot[food][food].values)
        slope[food] = linreg.intercept_

    return slope

'''
    DF GOOGLE IOT
'''
def addSlopeToDf(slope, df):
    df['slope'] = ''
    for key in slope:
        df['slope'].loc[key] = slope[key]
    df.slope = df.slope.astype('float64')

    return df

'''
    YELP
'''
def searchYelp(location):
    api_key = os.getenv("YELP_API_KEY")
    endpoint = 'https://api.yelp.com/v3/businesses/search?'

    yelp_search = []
    for comida in foods:
        term = 'term={}&'.format(comida)
        print(term)
        latitude = 'latitude='+str(location.latitude)+'&'
        longitude = 'longitude='+str(location.longitude)+'&'
        locale = 'locale=es_MX&'
        radius = 'radius=3000&'
        limit = 'limit=50'
        res = requests.get(endpoint+term+latitude+longitude+locale+radius+limit, headers={'Authorization':'Bearer '+api_key})
        soup = bs(res.content)
        yelp_json = json.loads(soup.findAll('p')[0].text)
        ydf = pd.DataFrame(yelp_json['businesses'])
        ydf['tipo'] = comida
        yelp_search.append(ydf)

    return yelp_search

def getYelpDf(yelp_search):
    lugares = pd.concat(yelp_search, axis=0, sort=False)
    lugares.reset_index(inplace=True)

    ratings = dict(lugares.tipo.value_counts())
    print(lugares.columns)
    for comida in foods:
        ratings[comida] = lugares[lugares.tipo == comida]['rating'].mean()

    yf = pd.concat(yelp_search, sort=False)
    yf.drop(columns = ['alias', 'display_phone', 'location', 'id', 'image_url', 'is_closed', 'phone', 'transactions', 'url'], inplace=True)
    yf.reset_index(inplace=True)
    return yf

'''
    SCORE
'''
def calculateScore(df):
    #df['score'] = df.GoogleTrend/100 - df.YelpRating/5 - df.InegiCount/df.InegiCount.max()
    #df['score'] = 0.05*df.GoogleTrend/df.YelpRating - df.InegiCount/df.InegiCount.max() + 0.5*df.slope/df.slope.max()
    df['score'] = df.GoogleTrend/100 + df.slope/df.slope.max() - df.YelpRating/5 - df.InegiCount/df.InegiCount.max()
    return df.sort_values(by='score', ascending=False).head()

'''
    YELP LOCATIONS DATAFRAME
'''
def flatLatLong(yf):
    yf[['latitude', 'longitude']] = json_normalize(yf.coordinates)
    return yf

'''
def getTopYfLocs(df, yf):
    top_types = df.sort_values(by='score', ascending=False).head().index

    top_yf_locs = {}
    for top in top_types:
        top_yf_locs[top] = yf[yf.tipo==top][['latitude', 'longitude']].head().values

    return top_yf_locs
'''

def getTopYf(df, yf):
    top_yf = pd.DataFrame(columns = yf.columns)
    top_types = df.sort_values(by='score', ascending=False).head().index

    for top in top_types:
        top_yf = pd.concat([top_yf, yf[yf.tipo==top].sort_values(by='rating', ascending=False).head(10)])

    return top_yf

'''
    FOURSQUARE
'''
def getFsResults(top_yf, location):
    client_id = os.getenv("FS_CLIENT_ID")
    client_secret = os.getenv("FS_CLIENT_SECRET")
    client = foursquare.Foursquare(client_id=client_id, client_secret=client_secret)

    fs_results = {}
    for comida in top_yf.tipo.unique():
        print(comida)
        fs_results[comida] = client.venues.search(params={'query': comida, 'intent':'checkin', 'll': str(location.latitude)+', '+str(location.longitude), 'radius':3000, 'limit':10})

    return fs_results

def getFsLocs(fs_results):
    fs_locs = {}
    for result in fs_results:
        result_locs = []
        for i in range(len(fs_results[result]['venues'])):
            loc = {}
            loc[fs_results[result]['venues'][i]['name']] = []
            loc[fs_results[result]['venues'][i]['name']].append(fs_results[result]['venues'][i]['location']['lat'])
            loc[fs_results[result]['venues'][i]['name']].append(fs_results[result]['venues'][i]['location']['lng'])
            result_locs.append(loc)
        fs_locs[result] = result_locs

    return fs_locs

@app.route('/')
def start():
    # Inegi
    print('Getting Inegi results...')
    inegi_results = inegiTask('YUCATAN', 'Mérida')
    print('Inegi results done.')

    # Google trends
    print('Getting Google trends...')
    google_trends = getGoogleTrends()
    trends =  getTrends(google_trends, 'Yucatán')
    print('Google trends done.')

    # Google interest over time
    print('Getting Google interest over time...')
    iot = getGoogleIOT()
    slope = getTrendSlope(iot)
    print('Google IOT done.')

    # Geocode location
    print('Getting location geocode...')
    colonia = request.args.get('colonia')
    location = userLocationGeocoding('pensiones merida yucatan')
    print('Location geocode done.')

    # Yelp
    print('Getting Yelp searches...')
    yelp_search = searchYelp(location)
    yf = getYelpDf(yelp_search)
    yf = flatLatLong(yf)
    print('Yelp searches done.')

    # DF Score
    print('Calculating results score...')
    df = getInegiDfResult(inegi_results)
    df['GoogleTrend'] = trends
    df = addSlopeToDf(slope, df)
    df['YelpRating'] = yf.groupby('tipo').mean().rating
    df = calculateScore(df)
    print('Score done.')

    # Top Yelp locations
    top_yf = getTopYf(df, yf)

    # Foursquare locations
    fs_results = getFsResults(top_yf, location)
    fs_locs = getFsLocs(fs_results)

    return flask.jsonify(fs_locs=fs_locs)
