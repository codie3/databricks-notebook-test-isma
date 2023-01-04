# Databricks notebook source
# MAGIC %pip install PyHive
# MAGIC %pip install thrift
# MAGIC %pip install cx-Oracle
# MAGIC %pip install sqlalchemy
# MAGIC %pip install openpyxl
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC dbutils.widgets.text("dt", "")

# COMMAND ----------



# COMMAND ----------

import os
import sys
from datetime import datetime

import cx_Oracle
import numpy as np
import pandas as pd
from pyhive import hive
from sqlalchemy import create_engine

# COMMAND ----------

# MAGIC %md
# MAGIC # Extract from BBDD

# COMMAND ----------

# MAGIC %md
# MAGIC ## STS

# COMMAND ----------

# tarda 1min 22s
# dt = '20220413'
#dt = sys.argv[1].strip("{").strip("}").strip("'")
dt = dbutils.widgets.get("dt")

print(f'dt: {dt}')

# COMMAND ----------




query = f"select ID_ENTERPRISE,ID_SALES_ACCOUNT,DATE_VALIDATE,TEAM_NAME,NUM_EMPLOYEE_PA,NUM_EMPLOYEE_DC_PUBLIC,NUM_EMPLOYEE_DC,INDUSTRY,REAL_ENGAGEMENT,THEORETICAL_ENGAGEMENT,NUM_PUBLI_L3Y,NUM_PUBLI_COMPETITORS,FACT_AMOUNT_L3Y,FACT_AMOUNT_L1Y,FACT_RECURRENCY_L3Y,RATING_SCORE from products_ci_score_segmentation.infojobs_account where dt='{dt}'"

df = spark.sql(query).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data preparation

# COMMAND ----------

#### change data types if needed
# NUM_EMPLOYEES
# df['NUM_EMPLOYEE_PA'] = df['NUM_EMPLOYEE_PA'].astype(int)
# df['NUM_EMPLOYEE_DC_PUBLIC'] = df['NUM_EMPLOYEE_DC_PUBLIC'].astype('Int64')
df['NUM_EMPLOYEE_DC'] = df['NUM_EMPLOYEE_DC'].astype(float)
# df['NUM_EMPLOYEE_DC'] = df['NUM_EMPLOYEE_DC'].astype('Int64')
df['NEW_NUM_EMPLEADOS'] = df[['NUM_EMPLOYEE_PA', 'NUM_EMPLOYEE_DC_PUBLIC', 'NUM_EMPLOYEE_DC']].max(axis=1)

# OTHER TYPES
df['TEAM_NAME'] = df['TEAM_NAME'].astype('category')
df['INDUSTRY'] = df['INDUSTRY'].astype('category')
df['REAL_ENGAGEMENT'] = df['REAL_ENGAGEMENT'].astype('category')
df['THEORETICAL_ENGAGEMENT'] = df['THEORETICAL_ENGAGEMENT'].astype('category')

# df['ID_ENTERPRISE'] = df['ID_ENTERPRISE'].astype(int)
# df['NUM_PUBLI_L3Y'] = df['NUM_PUBLI_L3Y'].astype('Int64')
# df['NUM_PUBLI_COMPETITORS'] = df['NUM_PUBLI_COMPETITORS'].astype('Int64')
# df['FACT_RECURRENCY_L3Y'] = df['FACT_RECURRENCY_L3Y'].astype('Int64')

df.info()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Comprobamos que no haya duplicados en ID_ENTERPRISE

# COMMAND ----------

# afegim dos rows pertenir dupes! TEMPORAL!

# print(df.shape)
# df = df.append(df.loc[1], ignore_index=True)
# df = df.append(df.loc[398], ignore_index=True)
# print(df.shape)
# df = df.reset_index(drop=True)

# COMMAND ----------

print(len(df.ID_ENTERPRISE.unique()), "should be equal to", df.shape[0])

# COMMAND ----------

# cada ID_ENTERPRISE tenga solo un ID_SALES_ACCOUNT
clidss = df.ID_ENTERPRISE
unique_clids = clidss[clidss.duplicated(keep=False)].unique()
unique_clids = list(unique_clids)

# COMMAND ----------

# NOTIFICACIO!! AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
if len(unique_clids) > 0:
    print("ERROR! CLIDS DUPLICADAS!!!")
    print("Clids duplicadas:", unique_clids)
else:
    print("no duplicates found")

# COMMAND ----------

# drop ids dupes dinámicamente
for idx, clid in enumerate(df.ID_ENTERPRISE):
    if clid in unique_clids:
        df.loc[idx]['ID_SALES_ACCOUNT'] = np.nan
df = df.drop_duplicates()

# COMMAND ----------

# MAGIC %md
# MAGIC # Compute Model

# COMMAND ----------

# Auxiliar function
def None_sum(*args):
    args = [a for a in args if not a is None]
    return sum(args) if args else None


# COMMAND ----------


def score_and_segment(data):

    # Desc Industria --> 0.08
    dict_industria = {}
    dict_industria['A'] = [
        'Servicios de RRHH',
        'Programación, Consultoria y otras Activ. informaticas',
        'Servicios y tecnología de la información',
    ]
    dict_industria['B'] = [
        'Servicios financieros',
        'Hostelería y restaurantes',
        'Atención sanitaria y hospitalaria',
        'Servicios de asesoría y auditoría',
        'Gran consumo y alimentación',
        'Industria textil, moda  y calzado',
        'Construcción',
        'Venta al por mayor',
        'Telecomunicaciones',
    ]
    # C-->todo lo que no esta en la A o B

    # Engagement Real --> 0.08
    dict_engagement = {}
    dict_engagement['A'] = ['Clientes Contrato']
    dict_engagement['B'] = ['Compradora', 'PI Customers']
    # C-->todo lo que no esta en la A o B

    # encontrar valor de corte A,B,C con quantiles (valor más pequeño de % de clientes)

    # SCC-Tuve que convertirlo a Float

    # Crawlers LY --> 0.2
    dict_crawlers = {}
    dict_crawlers['A'] = data['NUM_PUBLI_COMPETITORS'].astype('float').quantile(0.73)  # 0.73 =7, like excel --> A:>=7
    dict_crawlers['B'] = data['NUM_PUBLI_COMPETITORS'].astype('float').quantile(0.45)  # 0.45 =2, like excel --> B:>=2 pero <7

    # Facturacion3Y --> 0.08
    dict_fact = {}
    dict_fact['A'] = data['FACT_AMOUNT_L3Y'].astype('float').quantile(0.95)  # 0.95= 4231 más o menos like excel 4203
    dict_fact['B'] = data['FACT_AMOUNT_L3Y'].astype('float').quantile(0.52)  # 0.52 = 189 like excel

    # Num Empleados --> 0.2
    dict_emp = {}
    dict_emp['A'] = data['NEW_NUM_EMPLEADOS'].astype('float').quantile(0.915)  # 0.915 = 55, like excel(53)
    dict_emp['B'] = data['NEW_NUM_EMPLEADOS'].astype('float').quantile(0.52)  # 0.52 = 5, like excel

    # Publicacion3Y --> 0.2
    dict_pub = {}
    dict_pub['A'] = data['NUM_PUBLI_L3Y'].astype('float').quantile(0.9)  # 0.9 = 3, like excel
    dict_pub['B'] = data['NUM_PUBLI_L3Y'].astype('float').quantile(0.7)  # 0.7 = 1, like excel

    # RATING_SCORE --> 0.08
    dict_rat = {}
    dict_rat['A'] = data['RATING_SCORE'].astype('float').quantile(0.955)  # 0.955 = 4.9, like excel
    dict_rat['B'] = data['RATING_SCORE'].astype('float').quantile(0.5)  # 0.5 = 4.1, like excel

    # Recurrencia --> 0.08
    dict_rec = {}
    dict_rec['A'] = data['FACT_RECURRENCY_L3Y'].astype('float').quantile(0.83)  # 0.83 = 3, like excel
    dict_rec['B'] = data['FACT_RECURRENCY_L3Y'].astype('float').quantile(0.5)  # 0.5 = 1, like excel

    # por cada cliente, asignar A,B,C en cada variable
    for index, row in data.iterrows():
        # SCC-Tuve que meter el control de None

        if row['NUM_PUBLI_COMPETITORS'] is not None and row['NUM_PUBLI_COMPETITORS'] >= dict_crawlers['A']:
            data.at[index, 'Abc_Crawlers'] = 'A'
        elif row['NUM_PUBLI_COMPETITORS'] is not None and row['NUM_PUBLI_COMPETITORS'] >= dict_crawlers['B']:
            data.at[index, 'Abc_Crawlers'] = 'B'
        else:
            data.at[index, 'Abc_Crawlers'] = 'C'

        if row['NEW_NUM_EMPLEADOS'] >= dict_emp['A']:
            data.at[index, 'Abc_Empleados'] = 'A'
        elif row['NEW_NUM_EMPLEADOS'] >= dict_emp['B']:
            data.at[index, 'Abc_Empleados'] = 'B'
        else:
            data.at[index, 'Abc_Empleados'] = 'C'

        if row['REAL_ENGAGEMENT'] is not None and row['REAL_ENGAGEMENT'] in dict_engagement['A']:
            data.at[index, 'Abc_Eng'] = 'A'
        elif row['REAL_ENGAGEMENT'] is not None and row['REAL_ENGAGEMENT'] in dict_engagement['B']:
            data.at[index, 'Abc_Eng'] = 'B'
        else:
            data.at[index, 'Abc_Eng'] = 'C'

        if row['FACT_AMOUNT_L3Y'] is not None and row['FACT_AMOUNT_L3Y'] >= dict_fact['A']:
            data.at[index, 'Abc_Fact'] = 'A'
        elif row['FACT_AMOUNT_L3Y'] is not None and row['FACT_AMOUNT_L3Y'] >= dict_fact['B']:
            data.at[index, 'Abc_Fact'] = 'B'
        else:
            data.at[index, 'Abc_Fact'] = 'C'

        if row['INDUSTRY'] is not None and row['INDUSTRY'] in dict_industria['A']:
            data.at[index, 'Abc_Industria'] = 'A'
        elif row['INDUSTRY'] is not None and row['INDUSTRY'] in dict_industria['B']:
            data.at[index, 'Abc_Industria'] = 'B'
        else:
            data.at[index, 'Abc_Industria'] = 'C'

        if row['NUM_PUBLI_L3Y'] is not None and row['NUM_PUBLI_L3Y'] >= dict_pub['A']:
            data.at[index, 'Abc_Pub'] = 'A'
        elif row['NUM_PUBLI_L3Y'] is not None and row['NUM_PUBLI_L3Y'] >= dict_pub['B']:
            data.at[index, 'Abc_Pub'] = 'B'
        else:
            data.at[index, 'Abc_Pub'] = 'C'

        if row['FACT_RECURRENCY_L3Y'] is not None and row['FACT_RECURRENCY_L3Y'] >= dict_rec['A']:
            data.at[index, 'Abc_Rec'] = 'A'
        elif row['FACT_RECURRENCY_L3Y'] is not None and row['FACT_RECURRENCY_L3Y'] >= dict_rec['B']:
            data.at[index, 'Abc_Rec'] = 'B'
        else:
            data.at[index, 'Abc_Rec'] = 'C'

        if row['RATING_SCORE'] is not None and row['RATING_SCORE'] >= dict_rat['A']:
            data.at[index, 'Abc_RAT'] = 'A'
        elif row['RATING_SCORE'] is not None and row['RATING_SCORE'] >= dict_rat['B']:
            data.at[index, 'Abc_RAT'] = 'B'
        else:
            data.at[index, 'Abc_RAT'] = 'C'

    #####SCORE######
    columns = ['Abc_Crawlers', 'Abc_Empleados', 'Abc_Eng', 'Abc_Fact', 'Abc_Industria', 'Abc_Pub', 'Abc_Rec', 'Abc_RAT']
    # transformar A:3, B:2, C:1
    for c in columns:
        data[str(c + "_disc")] = data[c].replace({'A': 3, 'B': 2, 'C': 1})

    data['SCORE'] = (
        data['Abc_Crawlers_disc'] * 0.2
        + data['Abc_Empleados_disc'] * 0.2
        + data['Abc_Eng_disc'] * 0.08
        + data['Abc_Fact_disc'] * 0.08
        + data['Abc_Industria_disc'] * 0.08
        + data['Abc_Pub_disc'] * 0.2
        + data['Abc_Rec_disc'] * 0.08
        + data['Abc_RAT_disc'] * 0.08
    )
    data['SCORE'] = data['SCORE'].round(decimals=1)  # solo 1 decimal

    data = data.drop(
        columns=[
            'Abc_Crawlers_disc',
            'Abc_Empleados_disc',
            'Abc_Eng_disc',
            'Abc_Fact_disc',
            'Abc_Industria_disc',
            'Abc_Pub_disc',
            'Abc_Rec_disc',
            'Abc_RAT_disc',
        ]
    )

    ######SEG########
    conditions = [
        (data['SCORE'] >= 2.8),
        (data['SCORE'] < 2.8) & (data['SCORE'] >= 2.4),
        (data['SCORE'] < 2.4) & (data['SCORE'] >= 2),
        (data['SCORE'] < 2) & (data['SCORE'] >= 1.8),
        (data['SCORE'] < 1.8),
    ]
    values = [1, 2, 3, 4, 5]
    data['SEGMENTO'] = np.select(conditions, values)

    ###### calculo aproximacion de valor medio del segmento ######
    data = data.sort_values(['SEGMENTO', 'SCORE'], ascending=(True, False))
    data = data.reset_index(drop=True)
    size_1 = data[data['SEGMENTO'] == 1].index[-1]
    size_2 = data[data['SEGMENTO'] == 2].index[-1]
    size_3 = round(len(data) * 0.037) + size_2
    size_4 = round(len(data) * 0.05) + size_3
    score2 = ((data.iloc[size_1 + 1]['SCORE'] + data.iloc[size_2]['SCORE']) / 2).round(decimals=1)
    score3 = ((data.iloc[size_2 + 1]['SCORE'] + data.iloc[size_3]['SCORE']) / 2).round(decimals=1)
    score4 = ((data.iloc[size_3 + 1]['SCORE'] + data.iloc[size_4]['SCORE']) / 2).round(decimals=1)

    ############# REGLAS DE NEGOCIO #############
    data['FINAL_SEGMENTO'] = data['SEGMENTO'].values  # segmentación sin tener en cuenta las reglas de negocio
    data['SEGMENTO_NEGOCIO'] = data['SEGMENTO'].values  # segmentación CON las reglas de negocio
    data['SCORE_NEGOCIO'] = data['SCORE'].values
    data['Modified'] = 0  # si tiene un valor será el del segmento segun la regla de negocio
    for index, row in data.iterrows():
        # si el cliente tiene Eng_Real=Contrato o potencial --> seg_2 mínimo, score 2.5 mínimo
        addition = None_sum(row['NUM_PUBLI_L3Y'], row['NUM_PUBLI_COMPETITORS'])
        if (row['REAL_ENGAGEMENT'] is not None and row['REAL_ENGAGEMENT'] == 'Clientes Contrato') | (
            addition is not None and addition >= 450
        ):
            if row['SEGMENTO'] > 2:
                data.at[index, 'Modified'] = 2
                data.at[index, 'SEGMENTO_NEGOCIO'] = 2
                data.at[index, 'SCORE_NEGOCIO'] = score2
        # si el cliente tiene FACT_AMOUNT_L1Y>=1000 --> seg_3 mínimo, score 2.3 mínimo
        elif row['FACT_AMOUNT_L1Y'] is not None and row['FACT_AMOUNT_L1Y'] >= 1000:
            if row['SEGMENTO'] is not None and row['SEGMENTO'] > 3:
                data.at[index, 'Modified'] = 3
                data.at[index, 'SEGMENTO_NEGOCIO'] = 3
                data.at[index, 'SCORE_NEGOCIO'] = score3
        # si el cliente está carterizado y es un cliente farming --> seg_3 mínimo, score 2.3 mínimo
        # COMENTAR CUANDO SE PREPAREN CARTERAS
        elif (row['TEAM_NAME'] is not None and row['TEAM_NAME'] != 'SIN ESPECIFICAR') and (
            row['THEORETICAL_ENGAGEMENT'] in ['Compradora', 'PI Customers', 'Clientes Contrato']
        ):
            if row['SEGMENTO'] is not None and row['SEGMENTO'] > 3:
                data.at[index, 'Modified'] = 3
                data.at[index, 'SEGMENTO_NEGOCIO'] = 3
                data.at[index, 'SCORE_NEGOCIO'] = score3

    ############### FIX SEGMENTS' SIZES ##################
    # ordenar clientes por segmento, y dentro del segmento por score
    # SIN NEGOCIO
    data = data.sort_values(['FINAL_SEGMENTO', 'SCORE'], ascending=(True, False))
    data = data.reset_index(drop=True)

    size_1 = data[data['FINAL_SEGMENTO'] == 1].index[-1]  # último index de cliente del segmento 1
    size_2 = data[data['FINAL_SEGMENTO'] == 2].index[-1]  # último index de cliente del segmento 2
    size_3 = round(len(data) * 0.037) + size_2  # 3.7% de los clientes más altos que no esten en seg1 y seg2
    size_4 = round(len(data) * 0.05) + size_3  # 5% de los cleintes más altos que no esten en seg1, seg2, seg3
    # seg5 --> todo lo que queda

    for index, row in data.iterrows():
        # si el indice del cliente es mayor que seg2 però menor que seg3 --> tiene que estar en seg3
        if (index > size_2) & (index <= size_3):
            data.at[index, 'FINAL_SEGMENTO'] = 3
        # si el indice del cliente es mayor que seg3 però menor que seg4 --> tiene que estar en seg4
        elif (index > size_3) & (index <= size_4):
            data.at[index, 'FINAL_SEGMENTO'] = 4
        # si el indice del cliente es mayor que seg4 --> tiene que estar en seg5
        elif index > size_4:
            data.at[index, 'FINAL_SEGMENTO'] = 5

    # si no tiene tres años de antigüedad no se puede segmentar
    data['FINAL_SEGMENTO_3Y'] = data['FINAL_SEGMENTO'].values
    for index, row in data.iterrows():
        if (datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0) - row['DATE_VALIDATE']).days < 1095:
            data.at[index, 'FINAL_SEGMENTO_3Y'] = None

    # CON NEGOCIO
    data = data.sort_values(['SEGMENTO_NEGOCIO', 'SCORE_NEGOCIO'], ascending=(True, False))
    data = data.reset_index(drop=True)
    size_1 = data[data['SEGMENTO_NEGOCIO'] == 1].index[-1]  # último index de cliente del segmento 1
    size_2 = data[data['SEGMENTO_NEGOCIO'] == 2].index[-1]  # último index de cliente del segmento 2
    size_3 = round(len(data) * 0.037) + size_2  # 3.7% de los clientes más altos que no esten en seg1 y seg2
    size_4 = round(len(data) * 0.05) + size_3  # 5% de los cleintes más altos que no esten en seg1, seg2, seg3
    # seg5 --> todo lo que queda

    for index, row in data.iterrows():
        if row['Modified'] == 0:
            # si el indice del cliente es mayor que seg2 però menor que seg3 --> tiene que estar en seg3
            if (index > size_2) & (index <= size_3):
                data.at[index, 'SEGMENTO_NEGOCIO'] = 3
            # si el indice del cliente es mayor que seg3 però menor que seg4 --> tiene que estar en seg4
            elif (index > size_3) & (index <= size_4):
                data.at[index, 'SEGMENTO_NEGOCIO'] = 4
            # si el indice del cliente es mayor que seg4 --> tiene que estar en seg5
            elif index > size_4:
                data.at[index, 'SEGMENTO_NEGOCIO'] = 5

    # ÚLTIMA REGLA DE NEGOCI
    for index, row in data[data['Modified'] == 0].iterrows():
        # si el cliente es nuevo (DATE_VALIDATE < 12 mesos) --> min segmento 4
        if (datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0) - row['DATE_VALIDATE']).days <= 365:
            if row['SEGMENTO'] > 4:
                data.at[index, 'Modified'] = 4
                data.at[index, 'SEGMENTO_NEGOCIO'] = 4
                data.at[index, 'SCORE_NEGOCIO'] = score4

    # segmentos modificados (Up, Down, same)
    conditions = [
        (data['SEGMENTO'] > data['SEGMENTO_NEGOCIO']),  # si el segmento anterior era mayor que el nuevo --> ha subido
        (data['SEGMENTO'] < data['SEGMENTO_NEGOCIO']),  # si el segmento anterior era menor que el nuevo --> ha bajado
        (data['SEGMENTO'] == data['SEGMENTO_NEGOCIO']),  # si el segmento anterior es el mismo que el nuevo --> esta igual
    ]
    values = ['Up', 'Down', 'same']
    data['Changes'] = np.select(conditions, values)

    # scoring o regla de negocio
    conditions = [
        (data['SEGMENTO'] == data['SEGMENTO_NEGOCIO']),  # si el segmento anterior es el mismo que el nuevo --> scoring
        (data['SEGMENTO'] != data['SEGMENTO_NEGOCIO']),  # si ha cambiado --> regla de negocio
    ]
    values = ['Score', 'Negocio']
    data['SCORE_REASON'] = np.select(conditions, values)

    ######NOMS SEG########
    data['FINAL_SEGMENTO_NAME'] = data['SEGMENTO_NEGOCIO'].replace(
        {1: '1. Diamond', 2: '2. Platinum', 3: '3. Gold', 4: '4. Silver', 5: '5. Bronze'}
    )

    return data


# COMMAND ----------

# runear el modelo
df_copy = df.copy()
df_copy = score_and_segment(df_copy)

# COMMAND ----------

# MAGIC %md
# MAGIC # Exportar

# COMMAND ----------

#### change data types for the output

# NUM_EMPLOYEES
df_copy['NEW_NUM_EMPLEADOS'] = df_copy['NEW_NUM_EMPLEADOS'].astype(int)

# OTHER TYPES
df_copy['ID_ENTERPRISE'] = df_copy['ID_ENTERPRISE'].astype(int)
# SCC-Esto no tira
# df_copy['NUM_PUBLI_L3Y'] = df_copy['NUM_PUBLI_L3Y'].astype('Int64')
# df_copy['NUM_PUBLI_COMPETITORS'] = df_copy['NUM_PUBLI_COMPETITORS'].astype('Int64')
# df_copy['FACT_RECURRENCY_L3Y'] = df_copy['FACT_RECURRENCY_L3Y'].astype('Int64')
# df_copy['FINAL_SEGMENTO_3Y'] = df_copy['FINAL_SEGMENTO_3Y'].astype('Int64')

# SCORES
# df_copy['SCORE'] = df_copy['SCORE'].astype(int)
# df_copy['SCORE_NEGOCIO'] = df_copy['SCORE_NEGOCIO'].astype(int)

# COMMAND ----------

df_copy['Segmento_New'] = df_copy['FINAL_SEGMENTO_NAME']

# COMMAND ----------

# MAGIC %md
# MAGIC ## Histórico

# COMMAND ----------

df_output = df_copy.copy(deep=True)
df_output.rename(
    columns={
        'COD_EMPRESA': 'ID_ENTERPRISE',
        'SEGMENTO': 'SEGMENTO_BASE',
        'FINAL_SEGMENTO_3Y': 'SEGMENT',
        'SCORE_REASON': 'BUSINESS_SCORE_TYPE',
        'SCORE_NEGOCIO': 'BUSINESS_SCORE',
        'SEGMENTO_NEGOCIO': 'BUSINESS_SEGMENT',
    },
    inplace=True,
)

# COMMAND ----------

# NO SE COMO LO TENGO QUE SACAR, SUPONGO QUE TENGO QUE SUBIRLO PERO EL DATAFRAME TIENEN LOS SIGUIENTES CAMPOS
# cols = ['ID_ENTERPRISE', 'ID_SALES_ACCOUNT', 'SCORE', 'SEGMENT', 'BUSINESS_SCORE_TYPE', 'BUSINESS_SCORE', 'BUSINESS_SEGMENT']
# writer = pd.ExcelWriter('PEAK_PRO_Client_Segmentation_Historico@DD.MM.22.xlsx')
# df_output[cols].to_excel(writer)
# SCC esto no se si pega en un script orquestado, entiendo que no
# writer.save()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Escribir Output

# COMMAND ----------

# añadimos dt
df_output['dt'] = dt

# COMMAND ----------

# creamos dataframe a exportar
cols = ['ID_ENTERPRISE', 'ID_SALES_ACCOUNT', 'SCORE', 'SEGMENT', 'BUSINESS_SCORE_TYPE', 'BUSINESS_SCORE', 'BUSINESS_SEGMENT', 'dt']
df_toHive = df_output[cols]

# COMMAND ----------

spark_df = spark.createDataFrame(df_toHive)
partition = f"dt = '{dt}'"

output_database = "products_ci_score_segmentation"
output_table = "test_couto_infojobs_account_segmentation"
output_path = f"s3a://data-sch-products-dev/ci/{output_table}"
spark_df.write.mode("overwrite").partitionBy('dt').option("replaceWhere", partition).saveAsTable(
    f"{output_database}.{output_table}", path=output_path
)

