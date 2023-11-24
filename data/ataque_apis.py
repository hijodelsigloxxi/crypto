import ccxt
import pandas as pd
from datetime import datetime

#convertir a milisegundos desde la epoca Unix
def fecha_a_milisegundos(fecha):
    return int(datetime.strtime(fecha,'%d-%m-%Y'.timestamp()*1000))

#Esta función sirve para adquirir datos historicos a traves de la libreria ccxt que conecta con una api
def descargar_datos_historicos(exchange,coin_id, desde, hasta):
    datos_historicos=[]
    limite=1000

    while desde<hasta:
        try:
            ohlcv= exchange.fetch_ohlcv(coin_id, '1d', desde , limite)
            datos_historicos.extend(ohlcv)
            if len(ohlcv)==0:
                break
            desde= ohlcv[-1][0]+1
        except Exception as e:
            print(f'Error al descargar datos para {coin_id}:{e}')
            break
    return datos_historicos

#Aqui creamos el objeto de exchance para solicitar los metodos y atributos necesatios para trabajar con la api especifica
exchange= ccxt.binance()

# Lista de las 10 principales criptomonedas
criptos_interesantes = [
    'BTC/USDT',  # Bitcoin
    'ETH/USDT',  # Ethereum
    'BNB/USDT',  # Binance Coin
    'XRP/USDT',  # XRP
    'SOL/USDT',  # Solana
    'ADA/USDT',  # Cardano
    'DOT/USDT',  # Polkadot
    'DOGE/USDT', # Dogecoin
    'AVAX/USDT', # Avalanche
    'MATIC/USDT' # Polygon
]

#Crear las variables de milisigundos de los datos que queremos adquirir
desde= fecha_a_milisegundos('01-01-2010')
hasta= exchange.milliseconds()

#crear un dataframe vacio preparado para llenarlo con los datos recopilados
df_consolidado= pd.DataFrame()

#Esta parte del codigo obtiene los datos requeridos para cada columna para cada cryptomoneda
for coin_symbol in criptos_interesantes:
    try:
        print(f'descargando datos para {coin_symbol}...')
        datos=descargar_datos_historicos(exchange,coin_symbol, desde, hasta)

        if datos:
            df=pd.DataFrame(datos, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp']=pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol']=coin_symbol

            #SMA - Media movil simple
            df['sma_50']= df['close'].rolling(window=50).mean()

            #EMA - Media movil Exponencial
            df['ema_20']= df['close'].ewm(span=20, adjust=False).mean()

             # RSI - Índice de Fuerza Relativa
            delta = df['close'].diff(1)
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))
             # MACD - Convergencia y Divergencia de Medias Móviles
            df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = df['ema_12'] - df['ema_26']
            df['signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()

            # Banda de Bollinger
            df['bollinger_upper'] = df['sma_50'] + 2 * df['close'].rolling(window=50).std()
            df['bollinger_lower'] = df['sma_50'] - 2 * df['close'].rolling(window=50).std()

            # Tasa de Cambio (ROC)
            df['roc'] = df['close'].pct_change(periods=9) * 100

            # Identificar picos históricos
            df['top_3_highs'] = df['high'].nlargest(3)
            df['bottom_3_lows'] = df['low'].nsmallest(3)

            # Concatenar con el DataFrame consolidado
            df_consolidado = pd.concat([df_consolidado, df])
        else:
            print(f"No se encontraron datos para {coin_symbol} o hubo un error")
    except Exception as e:
        print(f'error general al procesar {coin_symbol}:{e}')

nombre_archivo_consolidado= 'criptomonedas_historico_consolidado.csv'
df_consolidado.to_csv(nombre_archivo_consolidado, index=False)
print(f"Todos los datos consolidados guardados en '{nombre_archivo_consolidado}'")
print('proceso terminado')