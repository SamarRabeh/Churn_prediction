import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import chi2,mutual_info_classif,mutual_info_regression
import plotly.graph_objs as go
import plotly.offline as py
from scipy.stats.contingency import chi2_contingency
from plotly.offline import plot
import geopandas as gpd
import contextily as ctx
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtGui import QPixmap


""" I. distribution des contrats résiliés """

""" 1. distibution geographique (carte) """
def map_churn2(df):
    # Group by Latitude', 'Longitude' and 'Churn Label' and count the occurrences
    grouped = df.groupby(['Latitude','Longitude', 'Churn Label'])['CustomerID'].count()
    total_counts = grouped.groupby(['Latitude','Longitude']).transform('sum')
    churn_rate = round(grouped / total_counts, 2)
    loc_ChurnLabel = pd.DataFrame({'churn_rate': churn_rate}).reset_index()

    # Create a DataFrame with the total count
    loc_counts = df.groupby(['Latitude','Longitude'])['CustomerID'].count().reset_index()
    loc_counts.rename(columns={'CustomerID': 'count'}, inplace=True)

    # Merge the City_ChurnLabel and city_counts DataFrames
    loc_ChurnLabel = loc_ChurnLabel.merge(loc_counts, on=['Latitude','Longitude'])

    # Filter out the rows where 'Churn Label' is 'No' and sort dataframe by count
    loc_ChurnLabel = loc_ChurnLabel[loc_ChurnLabel['Churn Label'] == 'Yes']
    loc_ChurnLabel = loc_ChurnLabel.sort_values(by='count', ascending=False)
    loc_ChurnLabel.drop(columns='Churn Label', inplace=True)

    loc_ChurnLabel.head()
    
    # Create GeoDataFrame
    geometry = gpd.points_from_xy(loc_ChurnLabel['Longitude'], loc_ChurnLabel['Latitude'])
    gdf = gpd.GeoDataFrame(loc_ChurnLabel, geometry=geometry)

    # Plot using GeoPandas
    fig, ax = plt.subplots(figsize=(12, 8))
    gdf.plot(ax=ax, markersize=gdf['count'], cmap='viridis', legend=True)
    ax.set_title('Churn Locations')
    plt.legend()
    plt.show()

def map_churn1(df):
    # Group by Latitude', 'Longitude' and 'Churn Label' and count the occurrences
    grouped = df.groupby(['Latitude','Longitude', 'Churn Label'])['CustomerID'].count()
    total_counts = grouped.groupby(['Latitude','Longitude']).transform('sum')
    churn_rate = round(grouped / total_counts, 2)
    loc_ChurnLabel = pd.DataFrame({'churn_rate': churn_rate}).reset_index()

    # Create a DataFrame with the total count
    loc_counts = df.groupby(['Latitude','Longitude'])['CustomerID'].count().reset_index()
    loc_counts.rename(columns={'CustomerID': 'count'}, inplace=True)

    # Merge the City_ChurnLabel and city_counts DataFrames
    loc_ChurnLabel = loc_ChurnLabel.merge(loc_counts, on=['Latitude','Longitude'])

    # Filter out the rows where 'Churn Label' is 'No' and sort dataframe by count
    loc_ChurnLabel = loc_ChurnLabel[loc_ChurnLabel['Churn Label'] == 'Yes']
    loc_ChurnLabel = loc_ChurnLabel.sort_values(by='count', ascending=False)
    loc_ChurnLabel.drop(columns='Churn Label', inplace=True)

    loc_ChurnLabel.head()

    fig = px.scatter_mapbox(loc_ChurnLabel,
                            lat="Latitude", lon='Longitude',
                            hover_data= ['count'], mapbox_style="carto-positron",
                            color="count", color_continuous_scale='Viridis'
            )
    plot(fig)

""" 2. distribution par ville """
def city_churncount(df):
  #ax.clear()
  # Group by 'City' and 'Churn Label' and count the occurrences
  grouped = df.groupby(['City', 'Churn Label'])['CustomerID'].count()
  total_counts = grouped.groupby('City').transform('sum')
  churn_rate = round(grouped / total_counts, 2)
  City_ChurnLabel = pd.DataFrame({'churn_rate': churn_rate}).reset_index()

  # Create a DataFrame with the total count per city
  city_counts = df.groupby('City')['CustomerID'].count().reset_index()
  city_counts.rename(columns={'CustomerID': 'count'}, inplace=True)

  # Merge the City_ChurnLabel and city_counts DataFrames
  City_ChurnLabel = City_ChurnLabel.merge(city_counts, on='City')

  # Filter out the rows where 'Churn Label' is 'No' and sort dataframe by count
  City_ChurnLabel = City_ChurnLabel[City_ChurnLabel['Churn Label'] == 'Yes']
  City_ChurnLabel = City_ChurnLabel.sort_values(by='count', ascending=False)
  City_ChurnLabel.drop(columns='Churn Label', inplace=True)
  
  plt.ioff()
  fig = plt.figure() 
  ax = fig.add_subplot()
  
  ax.bar(City_ChurnLabel.head(20)['City'], City_ChurnLabel.head(20)['count'], color='#23238E')
  ax.set_xlabel('Ville')
  ax.set_ylabel('Nombre de contrats résiliés')    
  ax.set_title('Distribution géographique des contrats résiliés')
  ax.set_xticks(ax.get_xticks())
  ax.set_xticklabels(City_ChurnLabel.head(20)['City'], rotation=45, ha='right')

  for i, value in enumerate(City_ChurnLabel.head(20)['count']):
      ax.text(i, value, str(value), ha='center', va='bottom')
      
  return fig

""" 3. distribution par genre """
def Churn_gender_distribution(df):
    #ax.clear()
    Churn_male_counts = df.loc[df.Gender=="Male", 'Churn Label'].value_counts()[0]
    Churn_female_counts = df.loc[df.Gender=="Female", 'Churn Label'].value_counts()[0]
    plt.ioff()
    fig = plt.figure() 
    ax = fig.add_subplot()
    ax.pie([Churn_female_counts, Churn_male_counts], labels= ["Female", "Male"], autopct='%1.1f%%', colors=['#23238E', '#32CD99'])
    ax.set_title('Distribution des contrats résiliés par genre')
    return fig

""" 4. distribution par age """
def Churn_age_distribution(df):
    #ax.clear()
    Churn_senior_counts = df.loc[df["Senior Citizen"]=="Yes", 'Churn Label'].value_counts()[0]
    Churn_nonsenior_counts = df.loc[df["Senior Citizen"]=="No", 'Churn Label'].value_counts()[0]
    fig = plt.figure() 
    ax = fig.add_subplot()
    ax.pie([Churn_senior_counts, Churn_nonsenior_counts], labels=["Senior", "Genior"], autopct='%1.1f%%', colors=['#23238E', '#32CD99'])
    ax.set_title('Distribution des contrats résiliés par age')
    return fig

""" 5. distribution par durée d'engagement """
def dist_tenure(df):
    churn_data = df[df['Churn Label'] == 'Yes']["Tenure Months"]
    non_churn_data = df[df['Churn Label'] == 'No']["Tenure Months"]    
    plt.ioff()
    fig = plt.figure() 
    ax = fig.add_subplot()
    ax.hist([churn_data, non_churn_data], color=['#32CD99', '#23238E'], label=['contrat résilié', 'contrat non résilié'])
    ax.legend()
    ax.set_xlabel('Durée en mois')
    ax.set_ylabel('Nombre de personne')
    ax.set_title('Distribution de la durée de détention des contrats')
    return fig

""" 6. distibution par charges mensuels """
def dist_monthlycharges(df):
    #ax.clear()
    churn_data = df[df['Churn Label'] == 'Yes']['Monthly Charges']
    non_churn_data = df[df['Churn Label'] == 'No']['Monthly Charges']
    plt.ioff()
    fig = plt.figure() 
    ax = fig.add_subplot()
    ax.hist([churn_data, non_churn_data], color=['#32CD99', '#23238E'], label=['contrat résilié', 'contrat non résilié'])
    ax.legend()
    ax.set_xlabel('Charges mensuelles en $')
    ax.set_ylabel('Nombre de personne')
    ax.set_title('Distribution des charges mensuelles des contrats résiliés et non résiliés')
    return fig

""" 7. distribution par métohode de paiement """
def paiementmethod_churncount(df):
  #ax.clear()
  grouped = df.groupby(['Payment Method', 'Churn Label'])['CustomerID'].count()
  total_counts = grouped.groupby('Payment Method').transform('sum')
  churn_rate = round(grouped / total_counts, 2)
  method_ChurnLabel = pd.DataFrame({'churn_rate': churn_rate}).reset_index()

  method_counts = df.groupby('Payment Method')['CustomerID'].count().reset_index()
  method_counts.rename(columns={'CustomerID': 'count'}, inplace=True)

  method_ChurnLabel = method_ChurnLabel.merge(method_counts, on='Payment Method')

  # Filter out the rows where 'Churn Label' is 'No' and sort dataframe by count
  method_ChurnLabel = method_ChurnLabel[method_ChurnLabel['Churn Label'] == 'Yes']
  method_ChurnLabel = method_ChurnLabel.sort_values(by='count', ascending=False)
  method_ChurnLabel.drop(columns='Churn Label', inplace=True)
  
  plt.ioff()
  fig = plt.figure() 
  ax = fig.add_subplot()
  ax.bar(method_ChurnLabel.head(20)['Payment Method'], method_ChurnLabel.head(20)['count'], color='#23238E')
  ax.set_xlabel('Méthode de paiement')
  ax.set_ylabel('Nombre de contrats résiliés')    
  ax.set_title('distribution des méthodes de paiement des contrats résiliés')
  ax.set_xticks(ax.get_xticks())
  #ax.set_xticklabels(method_ChurnLabel.head(20)['Payment Method'], rotation=45, ha='right')

  for i, value in enumerate(method_ChurnLabel.head(20)['count']):
      ax.text(i, value, str(value), ha='center', va='bottom')
  return fig

""" II. Raisons de résiliation """
""" 8. raisons déclarées """
def churnreason_count(df):
    values = df['Churn Reason'].value_counts(ascending=False)  
    plt.ioff()
    fig = plt.figure() 
    ax = fig.add_subplot()
    bars = ax.bar(values.index, values.values, color='skyblue') 
    ax.set_xlabel('Raison de résiliation')
    ax.set_ylabel('Nombre de personne')
    ax.set_title('Raisons de résiliation de contrat')
    ax.set_xticklabels([])

    for i,bar in enumerate(bars):
        xval = bar.get_x() + bar.get_width()/2
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')
        ax.text(xval, 0, values.index[i], ha='center', va='bottom', rotation=90)
    return fig

""" 9. correlation avec les attributs """
def correlation(df):
    corr = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    plt.ioff()
    fig = plt.figure() 
    ax = fig.add_subplot()
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="viridis", ax=ax)    
    ax.set_title("Matrice de correlation")
    return fig

""" III. Prediction """


""" Autre """
def figure_to_pixmap(figure):
    # Convertir une figure Matplotlib en QPixmap
    figure.tight_layout()
    figure.canvas.draw()
    img = figure.canvas.tostring_rgb()
    img_width, img_height = figure.canvas.get_width_height()
    qimg = PyQt5.QtGui.QImage(img, img_width, img_height, PyQt5.QtGui.QImage.Format_RGB32)
    pixmap = PyQt5.QtGui.QPixmap.fromImage(qimg)
    return pixmap


""" TEST d'AFFICHAGE """
if __name__ == '$__main__':
    df = pd.read_csv(r"C:\Users\33755\OneDrive\Documents\Mes Etudes\Semestre 9\projet_python\projet_telco\telco_cusomer_churn.txt", delimiter='\t', decimal=",")
    map_churn1(df)
    city_churncount(df)
    Churn_gender_distribution(df)
    Churn_age_distribution(df)
    dist_tenure(df)
    dist_monthlycharges(df)
    paiementmethod_churncount(df)
    correlation(df)
