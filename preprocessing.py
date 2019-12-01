import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper, CategoricalImputer
from sklearn.impute import SimpleImputer
from sklearn.base import TransformerMixin


def leer_csv(ruta):
    return pd.read_csv(ruta, dtype={
    'tipodepropiedad':'category', 'ciudad':'category',\
    'provincia':'category'}, parse_dates=[16])

# funcion para setear los superficies
def set_metros(row):
    total = row.metrostotales
    covered = row.metroscubiertos
    if np.isnan(total):
        row.metrostotales = covered
        return row
    if np.isnan(covered):
        row.metroscubiertos = total
        return row
    return row

def set_importancia_palabra(row):
    titulo = row.titulo
    descripcion = row.descripcion
    palabra_importante = ['roof garden', 'doble altura', 'sala tv', \
                          'lugares estacionamiento','cajones estacionamiento',\
                            'family room','double altura','salón juego','ideal para',\
                          'amplios espacio', 'independiente', 'casa condominio',\
                          'salón juego', 'pisos madera','exclusivo']
    for palabra in palabra_importante:
        if palabra in titulo:
            row.importancia_palabra += 1
        if palabra in descripcion:
            row.importancia_palabra += 1
    return row

# define numerical imputer
num_imputer = SimpleImputer(strategy='mean')

class SeriesImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        If the Series is of dtype Category, then impute with the most frequent object.
        """
    def fit(self, X, y=None):
        self.fill = X.value_counts().index[0]
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

# define categorical imputer
cat_imputer = SeriesImputer()


# imputer para variables numéricas

def num_imputer_columnas(df_train, df_test, cols):
    
    df_conjunto = pd.concat([df_train[cols], df_test[cols]], ignore_index=True)
    
    num_imputer.fit(df_conjunto[cols])
    df_train[cols] = num_imputer.transform(df_train[cols])
    df_test[cols] = num_imputer.transform(df_test[cols])
    
    return df_train, df_test

# imputer para variables categóricas

def cat_imputer_columnas(df_train, df_test, cols):
    
    df_conjunto = pd.concat([df_train[cols], df_test[cols]], ignore_index=True)
    
    for col in cols:
        cat_imputer.fit(df_conjunto[col])
        df_train[col] = cat_imputer.transform(df_train[col])
        df_test[col] = cat_imputer.transform(df_test[col])
    
    return df_train, df_test
        


def one_hot_encoder(df, columnas):
    for col in columnas:
        df = pd.get_dummies(df, prefix = ['cat'], columns = [col])
    return df

def processing_fechas(df, anio, mes, dia):
    df['fecha'] = pd.to_datetime(df['fecha'])
        
    if anio:
        df['anio'] = df['fecha'].dt.year
    if mes:
        df['mes'] = df['fecha'].dt.month
    if dia:
        df['dia'] = df['fecha'].dt.day
    return df

def suma_columnas(df, cols):
    nombre = ''
    for col in cols:
        nombre = nombre + col
    df['suma_'+nombre] = sum([df[col] for col in cols])
    return df

def procesar_nulos_lat_lon(df_train, df_test):
    
    df_conjunto = pd.concat([df_train[['idzona', 'lat', 'lng']], df_test[['idzona', 'lat', 'lng']]])

    df_conjunto = df_conjunto.groupby('idzona')[['lat', 'lng']].apply(np.mean)

    dict_lat = df_conjunto['lat'].to_dict()
    dict_lng = df_conjunto['lng'].to_dict()

    df_train['lat'] = df_train['lat'].fillna(value=df_train['idzona'].map(dict_lat))
    df_train['lng'] = df_train['lng'].fillna(value=df_train['idzona'].map(dict_lng))
    df_test['lat'] = df_test['lat'].fillna(value=df_test['idzona'].map(dict_lat))
    df_test['lng'] = df_test['lng'].fillna(value=df_test['idzona'].map(dict_lng))
    
    return df_train, df_test

#procesar los textos importantes
def encoder_texto_importante(df_train, df_test):
    df_train['descripcion'] = df_train['descripcion'].map(str)
    df_train['titulo'] = df_train['titulo'].map(str)
    df_train['importancia_palabra'] = 0
    df_train[['titulo','descripcion','importancia_palabra']] = \
    df_train[['titulo','descripcion','importancia_palabra']].apply(set_importancia_palabra,axis = 1)
    
    df_test['descripcion'] = df_test['descripcion'].map(str)
    df_test['titulo'] = df_test['titulo'].map(str)
    df_test['importancia_palabra'] = 0
    df_test[['titulo','descripcion','importancia_palabra']] = \
    df_test[['titulo','descripcion','importancia_palabra']].apply(set_importancia_palabra,axis = 1)
    
    return df_train, df_test

def cambiar_valores(x):
    i = 0
    if x.name == 'precioporm2':
        i+=1
        return i
    return x

def encoder_ordinal_cols(df_train, df_test, cols):
    
    for col in cols:
        precio_m2_por_col = df_train.groupby(col).agg({'precioporm2' : 'mean'})
        precio_m2_por_col.reset_index(inplace = True)
        precio_m2_por_col = precio_m2_por_col.sort_values(by = 'precioporm2')

        precio_m2_por_col = precio_m2_por_col.apply(cambiar_valores)
        precio_m2_por_col['precioporm2'] = precio_m2_por_col['precioporm2'].cumsum()

        orden_col = precio_m2_por_col.set_index(col).to_dict()
        orden_col = orden_col['precioporm2']

        df_train[col+'_ordinal'] = df_train[col].map(orden_col)
        df_test[col+'_ordinal'] = df_test[col].map(orden_col)
        
    df_train, df_test = cat_imputer_columnas(df_train, df_test, [col+'_ordinal' for col in cols])
    
    return df_train, df_test

    
def encoder_idzona_ordinal(df_train, df_test): 
    
    df_train['precioporm2'] = df_train['precio'] / df_train['metrostotales']

    precio_m2_por_idzona = df_train.groupby('idzona').agg({'precioporm2' : 'mean'})
    precio_m2_por_idzona.reset_index(inplace = True)
    precio_m2_por_idzona = precio_m2_por_idzona.sort_values(by = 'precioporm2')

    precio_m2_por_idzona = precio_m2_por_idzona.apply(cambiar_valores)
    precio_m2_por_idzona['precioporm2'] = precio_m2_por_idzona['precioporm2'].cumsum()

    orden_idzona = precio_m2_por_idzona.set_index('idzona').to_dict()
    orden_idzona = orden_idzona['precioporm2']

    df_train['idzona_ordinal'] = cat_imputer.fit_transform(df_train.idzona.map(orden_idzona))
    df_test['idzona_ordinal'] = cat_imputer.fit_transform(df_test.idzona.map(orden_idzona))
    
    return df_train, df_test

def encoder_ciudad_ordinal(df_train, df_test): 
    
    df_train['precioporm2'] = df_train['precio'] / df_train['metrostotales']

    precio_m2_por_ciudad = df_train.groupby('ciudad').agg({'precioporm2' : 'mean'})
    precio_m2_por_ciudad.reset_index(inplace = True)
    precio_m2_por_ciudad = precio_m2_por_ciudad.sort_values(by = 'precioporm2')

    precio_m2_por_ciudad = precio_m2_por_ciudad.apply(cambiar_valores)
    precio_m2_por_ciudad['precioporm2'] = precio_m2_por_ciudad['precioporm2'].cumsum()

    orden_ciudad = precio_m2_por_ciudad.set_index('ciudad').to_dict()
    orden_ciudad = orden_ciudad['precioporm2']

    df_train['ciudad_ordinal'] = df_train.ciudad.map(orden_ciudad)
    df_test['ciudad_ordinal'] = df_test.ciudad.map(orden_ciudad)
    
    df_train, df_test = cat_imputer_columnas(df_train, df_test, ['ciudad_ordinal'])
    
    return df_train, df_test

def encoder_provincia_ordinal(df_train, df_test): 
    
    df_train['precioporm2'] = df_train['precio'] / df_train['metrostotales']

    precio_m2_por_provincia = df_train.groupby('provincia').agg({'precioporm2' : 'mean'})
    precio_m2_por_provincia.reset_index(inplace = True)
    precio_m2_por_provincia = precio_m2_por_provincia.sort_values(by = 'precioporm2')

    precio_m2_por_provincia = precio_m2_por_provincia.apply(cambiar_valores)
    precio_m2_por_provincia['precioporm2'] = precio_m2_por_provincia['precioporm2'].cumsum()

    orden_provincia = precio_m2_por_provincia.set_index('provincia').to_dict()
    orden_provincia = orden_provincia['precioporm2']
    
    df_train['provincia_ordinal'] = df_train.provincia.map(orden_provincia)
    df_test['provincia_ordinal'] = df_test.provincia.map(orden_provincia)
    
    df_train, df_test = cat_imputer_columnas(df_train, df_test, ['provincia_ordinal'])
    
    return df_train, df_test

def encoder_por_precio_m2_cols(df_train, df_test, cols):
    
    for col in cols:
        df_train_col = df_train.groupby(col).agg({'precioporm2':'mean'}).sort_values('precioporm2').reset_index()
        #df_train_col['precioporm2'] = np.log(df_train_col['precioporm2'])
        col_log_precio = df_train_col.set_index(col).to_dict()['precioporm2']

        df_train[col+'_promedio_m2'] = df_train[col].map(col_log_precio)
        df_test[col+'_promedio_m2'] = df_test[col].map(col_log_precio)

    df_train, df_test = cat_imputer_columnas(df_train, df_test, [col+'_promedio_m2' for col in cols])
    
    return df_train, df_test

def encoder_ciudad_por_precio_m2(df_train, df_test):
    #filtrando los precios
    #hay que eliminar los nulls primero
    df_train['precioporm2'] = df_train['precio'] / df_train['metrostotales']
    df_train_ciudad = df_train.groupby('ciudad').agg({'precioporm2':'mean'}).sort_values('precioporm2').reset_index()
    #df_train_ciudad['precioporm2'] = np.log(df_train_ciudad['precioporm2'])
    ciudad_log_precio = df_train_ciudad.set_index('ciudad').to_dict()['precioporm2']

    df_train['ciudad_promedio_m2'] = df_train.ciudad.map(ciudad_log_precio)
    df_test['ciudad_promedio_m2'] = df_test.ciudad.map(ciudad_log_precio)
    
    df_train, df_test = cat_imputer_columnas(df_train, df_test, ['ciudad_promedio_m2'])
    
    return df_train, df_test

def encoder_idzona_por_precio_m2(df_train, df_test):
    #filtrando los precios
    #hay que eliminar los nulls primero
    df_train['precioporm2'] = df_train['precio'] / df_train['metrostotales']
    df_train_idzona = df_train.groupby('idzona').agg({'precioporm2':'mean'}).sort_values('precioporm2').reset_index()
    #df_train_idzona['precioporm2'] = np.log(df_train_idzona['precioporm2'])
    idzona_log_precio = df_train_idzona.set_index('idzona').to_dict()['precioporm2']

    df_train['idzona_promedio_m2'] = df_train.idzona.map(idzona_log_precio)
    df_test['idzona_promedio_m2'] = df_test.idzona.map(idzona_log_precio)
    
    df_train, df_test = cat_imputer_columnas(df_train, df_test, ['idzona_promedio_m2'])
    
    return df_train, df_test

def encoder_posicion_por_precio_m2(df_train, df_test):
    #modificar las lat y lon en una colunma
    df_train['precioporm2'] = df_train['precio'] / df_train['metrostotales']
    #modificar lat y lon
    df_train['lat'] = df_train['lat'].apply(lambda x: round(x,2))    
    df_train['lng'] = df_train['lng'].apply(lambda x: round(x,2))
    df_test['lat'] = df_test['lat'].apply(lambda x: round(x,2))    
    df_test['lng'] = df_test['lng'].apply(lambda x: round(x,2))
    
    #unir lat y lon como posicion
    df_train['posicion'] = df_train['lat'].map(str) + df_train['lng'].map(str)
    df_test['posicion'] = df_test['lat'].map(str) + df_test['lng'].map(str)
    
    df_train_posicion = df_train.groupby('posicion').agg({'precioporm2':'mean'}).sort_values('precioporm2').reset_index()
    df_train_posicion = df_train_posicion[df_train_posicion.posicion != 'nannan']
    #df_train_idzona['precioporm2'] = np.log(df_train_idzona['precioporm2'])
    posicion_log_precio = df_train_posicion.set_index('posicion').to_dict()['precioporm2']

    df_test['posicion_promedio_m2'] = df_test.posicion.map(posicion_log_precio)
    df_train['posicion_promedio_m2'] = df_train.posicion.map(posicion_log_precio)
    
    return df_train, df_test

def agregar_tiene_sup_descubierta(df_train, df_test):
    df_train['tiene_sup_descubierta'] = (df_train['metrostotales'] > df_train['metroscubiertos']).astype('int')
    df_test['tiene_sup_descubierta'] = (df_test['metrostotales'] > df_test['metroscubiertos']).astype('int')
    return df_train, df_test

def agregar_diferencia_metros_totales_y_cubiertos(df_train, df_test):
    df_train['diff_metros_cubiertos_y_totales'] = abs(df_train['metrostotales'] - df_train['metroscubiertos'])
    df_test['diff_metros_cubiertos_y_totales'] = abs(df_test['metrostotales'] - df_test['metroscubiertos'])
    return df_train, df_test

def preprocessing(guardar_csv=False, procesar_nulos=True, cols_eliminar=['fecha', 'id', 'titulo', 'descripcion', 'direccion',\
                          'lat', 'lng','posicion','provincia', 'ciudad','gimnasio','usosmultiples',\
                          'escuelascercanas','centroscomercialescercanos',\
                          'idzona_promedio_m2', 'ciudad_promedio_m2',\
                          'posicion_promedio_m2', 'tiene_sup_descubierta', \
                          'diff_metros_cubiertos_y_totales']):
    df_train = leer_csv('data/train.csv')
    df_test  = leer_csv('data/test.csv')
    
    df_train = df_train[(df_train['tipodepropiedad'] != 'Hospedaje')&\
                        (df_train['tipodepropiedad'] != 'Garage')]
    
    ## Fechas
    
    df_train = processing_fechas(df_train, True, True, True)
    df_test = processing_fechas(df_test, True, True, True)
    
    ## Nulos
    
    # Metros cubiertos y totales
    
    df_train[['metrostotales', 'metroscubiertos']] = \
    df_train[['metrostotales', 'metroscubiertos']].apply(set_metros, axis = 1)
    df_test[['metrostotales', 'metroscubiertos']] = \
    df_test[['metrostotales', 'metroscubiertos']].apply(set_metros, axis = 1)
    
    # Imputing de variables categoricas y numericas
    
    if procesar_nulos:
        num_cols = ['antiguedad', 'habitaciones', 'banos', 'garages']
        df_train, df_test = num_imputer_columnas(df_train, df_test, num_cols)

        cat_cols = ['ciudad', 'tipodepropiedad']
        df_train, df_test = cat_imputer_columnas(df_train, df_test, cat_cols)
    
    # Nulos en id_zona
    
    if procesar_nulos:
        df_train.idzona.fillna(inplace=True, value=-1)
        df_test.idzona.fillna(inplace=True, value=-1)
    
    # Latitud y longitud
    
    if procesar_nulos:
        df_train, df_test = procesar_nulos_lat_lon(df_train, df_test)
    ## Agregamos precio por m2 en train
    
    df_train['precioporm2'] = df_train['precio'] / df_train['metrostotales']
    
    ## Encoding de variables categóricas
    
    # One Hot encoding para 'tipo de propiedad'
    df_train['tipodepropiedad'].cat.remove_unused_categories()
    
    antiguedad_train = df_train['antiguedad'] #Los guardo para agregarlos nuevamente luego del one hot encoding.
    antiguedad_test = df_test['antiguedad']
    df_test['antiguedad'] = df_test['antiguedad'].astype('category').cat.add_categories(new_categories = [63.0,64.0,66.0,71.0,74.0,75.0,77.0])
    
    df_train = one_hot_encoder(df_train, ['tipodepropiedad', 'antiguedad'])
    df_test = one_hot_encoder(df_test, ['tipodepropiedad', 'antiguedad'])
        
    df_train['antiguedad'] = antiguedad_train
    df_test['antiguedad'] = antiguedad_test
    
    #Encoding por textos
    df_train, df_test = encoder_texto_importante(df_train, df_test)
    
    # Encoding por precio promedio por m2 de las ciudades
    
    df_train, df_test = encoder_por_precio_m2_cols(df_train, df_test, ['idzona', 'ciudad'])
    df_train, df_test = encoder_posicion_por_precio_m2(df_train, df_test)

    # Encoding ordinal 
    
    df_train, df_test = encoder_ordinal_cols(df_train, df_test, ['idzona', 'ciudad', 'provincia'])
    
    ## Otras Features
    
    # 'cant_extras' = 'gimnasio'+'usosmultiples'+'piscina'
    extras = ['gimnasio', 'usosmultiples', 'piscina']
    df_train = suma_columnas(df_train, extras)
    df_test = suma_columnas(df_test, extras)
    
    # 'cant_cercanos' = 'escuelascercanas'+'centroscomercialescercanos'
    cercanos = ['escuelascercanas', 'centroscomercialescercanos']
    df_train = suma_columnas(df_train, cercanos)
    df_test = suma_columnas(df_test, cercanos)
    
    # 'tiene_sup_descubierta' = 1{'(metrostotales > metroscubiertos)'}
    df_train, df_test = agregar_tiene_sup_descubierta(df_train, df_test)
    
    # 'diff_metros_totales_y_cubiertos' = abs(metrostotales - metroscubiertos)
    df_train, df_test = agregar_diferencia_metros_totales_y_cubiertos(df_train, df_test)
    
    ## Eliminamos columnas no utilizadas
    df_train.drop(columns=cols_eliminar+['precioporm2', 'cat_Garage', 'cat_Hospedaje'], inplace=True)
    df_test.drop(columns=cols_eliminar, inplace=True)
    
    ## Agregamos columna con log(precio)
    df_train['log_precio'] = np.log(df_train['precio'])
    
    if guardar_csv:
        df_train.to_csv('data/train_preproc.csv', index = False)
        df_test.to_csv('data/test_preproc.csv', index = False)
    return df_train, df_test