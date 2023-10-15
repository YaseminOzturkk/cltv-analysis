##############################################################
# BG-NBD ve Gamma-Gamma ile CLTV Prediction
##############################################################

###############################################################
# İş Problemi (Business Problem)
###############################################################
# FLO satış ve pazarlama faaliyetleri için roadmap belirlemek istemektedir.
# Şirketin orta uzun vadeli plan yapabilmesi için var olan müşterilerin gelecekte şirkete sağlayacakları potansiyel değerin tahmin edilmesi gerekmektedir.


###############################################################
# Veri Seti Hikayesi
###############################################################

# Veri seti son alışverişlerini 2020 - 2021 yıllarında OmniChannel(hem online hem offline alışveriş yapan) olarak yapan müşterilerin geçmiş alışveriş davranışlarından
# elde edilen bilgilerden oluşmaktadır.

# master_id: Eşsiz müşteri numarası
# order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : En son alışverişin yapıldığı kanal
# first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date : Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online : Muşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline : Muşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi

################## Gerekli Kütüphaneler ####################
# !pip install lifetimes
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)



###############################################################
# GÖREVLER
###############################################################
# GÖREV 1: Veriyi Hazırlama

# 1. flo_data_20K.csv verisini okuyunuz.Dataframe’in kopyasını oluşturunuz.

df_ = pd.read_csv('C:/Users/yasmi/Desktop/FLOCLTVPrediction/flo_data_20k.csv')
df = df_.copy()


# 2. Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayınız.
# Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız.
def outlier_thresholds(dataframe, variables):
    quartile1 = dataframe[variables].quantile(0.001)
    quartile3 = dataframe[variables].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit,0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit,0)


# 3. "order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline",
# "customer_value_total_ever_online" değişkenlerinin aykırı değerleri varsa baskılayanız.

outlier_thresholds_column = ["order_num_total_ever_online", "order_num_total_ever_offline",
                             "customer_value_total_ever_offline", "customer_value_total_ever_online"]
for col in outlier_thresholds_column:
    replace_with_thresholds(df, col)

# 4. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir. Herbir
# müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.

df['total_order'] = df['order_num_total_ever_online'] + df['order_num_total_ever_offline']
df['total_spend'] = df['customer_value_total_ever_offline'] + df['customer_value_total_ever_offline']

# 5. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
df.dtypes

for i in df.columns:
    if "date" in i:
       df[i] = df[i].astype("datetime64[ns]")



# GÖREV 2: CLTV Veri Yapısının Oluşturulması

# 1.Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.

df["last_order_date"].max() # Timestamp('2021-05-30 00:00:00')
today_date = dt.datetime(2021, 6, 1)

# 2.customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı
# yeni bir cltv dataframe'i oluşturunuz. Monetary değeri satın alma başına ortalama değer olarak, recency ve
# tenure değerleri ise haftalık cinsten ifade edilecek.

# Recency: KULLANICI ÖZELİNDE, son satın alma üzerinden geçen zaman.(Haftalık)
# T: Müşterinin yaşı. (Haftalık) Analiz tarihinden ne kadar süre önce ilk satın alma yapılmış.
# frequency: Tekrar eden toplam sayın alma sayısı (frequency < 1)
# Monetary: Satın alma başına ORTALAMA kazanç

cltv_df = pd.DataFrame({"customer_id": df["master_id"],
             "recency_cltv_weekly": ((df["last_order_date"] - df["first_order_date"]).dt.days)/7,
             "T_weekly": ((today_date - df["first_order_date"]).astype('timedelta64[D]'))/7,
             "frequency": df["total_order"],
             "monetary_cltv_avg": df["total_spend"] / df["total_order"]})

cltv_df = cltv_df[(cltv_df['frequency'] > 1)]

# GÖREV 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması, CLTV'nin hesaplanması

# 1. BG/NBD modelini fit ediniz.

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency_cltv_weekly'],
        cltv_df['T_weekly'])


# a. 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak
# cltv dataframe'ine ekleyiniz.

cltv_df['exp_sales_3_month'] = bgf.predict(4 * 3,
            cltv_df['frequency'],
            cltv_df['recency_cltv_weekly'],
            cltv_df['T_weekly'])


# b. 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak
# cltv dataframe'ine ekleyiniz.

cltv_df['exp_sales_6_month'] = bgf.predict(4 * 6,
            cltv_df['frequency'],
            cltv_df['recency_cltv_weekly'],
            cltv_df['T_weekly'])

# 2. Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip
# exp_average_value olarak cltv dataframe'ine ekleyiniz.

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])
cltv_df['exp_average_value'] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                          cltv_df['monetary_cltv_avg'])
cltv_df.head()

# 3. 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency_cltv_weekly'],
                                   cltv_df['T_weekly'],
                                   cltv_df['monetary_cltv_avg'],
                                   time=6, # 6 aylık
                                   freq="W", # T'nin frekans bilgisi
                                   discount_rate=0.01)

cltv_df["cltv"] = cltv

# b. Cltv değeri en yüksek 20 kişiyi gözlemleyiniz.
cltv_df.sort_values(by="cltv", ascending=False).head(20)

# GÖREV 4: CLTV'ye Göre Segmentlerin Oluşturulması
# 1. 6 aylık tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz.
# cltv_segment ismi ile dataframe'e ekleyiniz.

cltv_df["cltv_segment"] = pd.cut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])

# 2. 4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon önerilerinde bulununuz.

cltv_df.groupby("cltv_segment").agg({"count", "mean", "sum"})
