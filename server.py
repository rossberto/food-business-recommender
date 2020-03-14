from dotenv import load_dotenv
load_dotenv()

import os
from flask import Flask, request
from flask_cors import CORS
import flask

from removeaccents import remove_accents
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
CORS(app)

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
    return requests[state].sort_values(ascending=False)

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
        slope[food] = linreg.coef_[0]

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
    print(location.latitude, location.longitude)

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
    return df.sort_values(by='score', ascending=False).head(3)

'''
    YELP LOCATIONS DATAFRAME
'''
def flatLatLong(yf):
    yf[['latitude', 'longitude']] = json_normalize(yf.coordinates)
    return yf

def getTopYf(df, yf):
    top_yf = pd.DataFrame(columns = yf.columns)
    top_types = df.sort_values(by='score', ascending=False).head().index

    for top in top_types:
        top_yf = pd.concat([top_yf, yf[yf.tipo==top].sort_values(by='rating', ascending=False).head(10)])

    return top_yf

def getTopCompetitors(top_yf):
    top_yf_list = []
    for tipo in top_yf.tipo.unique():
        for index in top_yf[top_yf.tipo==tipo].index:
            toadd = {'tipo': tipo}
            toadd['name'] = top_yf.name.loc[index]
            toadd['coords'] = [top_yf.latitude.loc[index], top_yf.longitude.loc[index]]
            top_yf_list.append(toadd)
    return top_yf_list

@app.route('/')
def start():
    estado = request.args.get('estado')
    ciudad = request.args.get('ciudad')
    colonia = request.args.get('colonia')

    print(estado, ciudad, colonia)

    # Inegi
    print('Getting Inegi results...')
    inegi_results = inegiTask(remove_accents(estado.upper()), ciudad)
    print('Inegi results done.')
    print(' ')

    # Google trends
    print('Getting Google trends...')
    google_trends = getGoogleTrends()
    trends =  getTrends(google_trends, estado)
    print('Google trends done.')
    print(' ')

    # Google interest over time
    print('Getting Google interest over time...')
    iot = getGoogleIOT()
    slope = getTrendSlope(iot)
    print('Google IOT done.')
    print(' ')

    # Geocode location
    print('Getting location geocode...')
    user_loc = colonia + ' ' + ciudad + ' ' + estado
    print('Getting coords from: ', user_loc)
    location = userLocationGeocoding(user_loc)
    print(location)
    print('Location geocode done.')
    print(' ')

    # Yelp
    print('Getting Yelp searches...')
    yelp_search = searchYelp(location)
    yf = getYelpDf(yelp_search)
    print('Yelp searches done.')
    print(' ')

    # DF Score
    print('Calculating results score...')
    df = getInegiDfResult(inegi_results)
    df['GoogleTrend'] = trends
    df = addSlopeToDf(slope, df)
    df['YelpRating'] = yf.groupby('tipo').mean().rating
    df = calculateScore(df)
    print('Score done.')
    print(' ')

    # Top Yelp locations
    yf = flatLatLong(yf)
    top_yf = getTopYf(df, yf)
    top_competitors = getTopCompetitors(top_yf)


    return flask.jsonify(top_competitors=top_competitors, recomend=df.index[0])
