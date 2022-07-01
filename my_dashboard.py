import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import graphviz
from sklearn import tree
from PIL import Image
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Lasso
import operator

#Wszystkie metody sa ozdobione @st.experimental_memo() aby wartosci zapisywaly sie w cache
#st.experimental_memo() jest eksperymentalna metoda st.cache()


#Wczytywanie
@st.experimental_memo()
def ReadData():
    df = pd.read_csv('data_cleaned.csv', index_col=0)
    return df
df = ReadData()

#Outliners
@st.experimental_memo()
def ReadDataOutliners():
    df_outliners = pd.read_csv('data_outliners.csv', index_col=0)
    return df_outliners
df_outliners = ReadDataOutliners()

#Orginał
@st.experimental_memo()
def OriginalDf():
    df_original = pd.read_csv('data.csv', index_col=0)
    return(df_original)
df_original = OriginalDf()

#Wykres Outliners
@st.experimental_memo()
def FigOutliners():
    fig = px.box(df_outliners, y='hotel_price', title='Wykres pudełkowy ceny')
    return fig
fig_outliners = FigOutliners()

#Obliczanie ilości kategorii
@st.experimental_memo()
def CountCat():
    cattegory = []
    for col in df_original.columns:
        if(col.split(':')[0] not in cattegory):
            cattegory.append(col.split(':')[0])
    return cattegory
cattegory = CountCat()

#Wykres groupby
@st.experimental_memo()
def FigGroupBy():
    df_group_by_price = df.groupby('hotel_city')['hotel_price'].mean()
    fig = px.bar(df_group_by_price, title='Wykres średniej ceny w miejscowości')
    fig.update_layout(
        xaxis_title="Miejscowość",
        yaxis_title="Cena",
        legend_title="Atrybut",
    )
    return fig
fig_group_by_city = FigGroupBy()

#Train test podzial
@st.experimental_memo()
def TestTrainSplit():
    df_without_name = df.drop(columns=['hotel_name','hotel_city'])
    y = df_without_name['hotel_price']
    X = df_without_name.drop(columns=['hotel_price'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test, y, X
X_train, X_test, y_train, y_test, y, X  = TestTrainSplit()

#Model drzewa regresyjnego
@st.experimental_memo()
def ModelTree():
    DTR = DecisionTreeRegressor(random_state=0,max_depth=4)
    DTR.fit(X_train, y_train)
    predictons = DTR.predict(X_test)
    r2 = r2_score(y_test, predictons)
    MSE = mean_squared_error(y_test, predictons)
    return r2, MSE, DTR
r2_tree, MSE_tree, DTR = ModelTree()

#Obraz drzewa i zapis
@st.experimental_memo()
def TreeGraph():
    dot_data = tree.export_graphviz(DTR, out_file=None,feature_names=X.columns, max_depth=2)
    graph = graphviz.Source(dot_data, format='png')
    graph.render(directory='doctest-output').replace('\\', '/')
TreeGraph()

#Pobieranie zdjęcia 
@st.experimental_memo()
def ReadImgTree():
    img = Image.open('doctest-output/Source.gv.png')
    return img
img_graph = ReadImgTree()

#Model regresyjnych lasów losowych
@st.experimental_memo()
def ModelRandomForest():
    RFR = RandomForestRegressor(random_state=0)
    RFR.fit(X_train, y_train)
    predictons = RFR.predict(X_test)
    r2 = r2_score(y_test, predictons)
    MSE = mean_squared_error(y_test, predictons)
    return r2, MSE, RFR, predictons
r2_forest, MSE_forest, RFR, predictions_forest = ModelRandomForest()

#Tworzenie wykresu dla lasów
@st.experimental_memo()
def MakePlotPredForest():
    zipped = zip(np.array(y_test),predictions_forest)
    sorted_pairs = sorted(zipped)
    tuples = zip(*sorted_pairs)
    sorted_y_test, sorted_pred = [ list(tuple) for tuple in  tuples]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y = sorted_y_test,
        name = 'Prawdziwe wartości'
    ))
    fig.add_trace(go.Scatter(
        y = sorted_pred,
        name = 'Predykcja modelu'
    ))
    fig.update_layout(
        title="Wykres porównujący prawdziwą wartość od wartości predykcji",
        xaxis_title="Wystąpienie",
        yaxis_title="Cena",
        #legend_title="Wartości",
    )
    return fig
fig_forest = MakePlotPredForest()

#Tworzenie wykresu ważności atrybutów dla lasów
@st.experimental_memo()
def MakePlotFeatForest():
    sort = RFR.feature_importances_.argsort()
    fig = px.bar(y=X.columns[sort], x=RFR.feature_importances_[sort], height=600, title='Wykres ważności atrybutów dla lasów losowych')
    fig.update_layout(xaxis_title='Znaczenie',yaxis_title='Atrybut')
    return fig
fig_feat_forest = MakePlotFeatForest()

#Wyznaczanie najlepszej wartości alpha(lambda sie nazywa, alpha w metodzie)
# - hiperparametru odpowiadajacego za wage "kary" w funkcji straty
@st.experimental_memo()
def CalculationAlpha():
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    LCV = LassoCV(alphas=np.arange(0,1, 0.01), cv=cv, n_jobs=-1)
    LCV.fit(X_train,y_train)
    alpha_best = LCV.alpha_
    return alpha_best
alpha_best = CalculationAlpha() # można go też tu ustawić i zmieni model obliczany w kolejnym kroku, zakres(0,1)

#Model Lasso
@st.experimental_memo()
def ModelLASSO():
    LS = Lasso(alpha=alpha_best)
    LS.fit(X_train,y_train)
    predictons = LS.predict(X_test)
    r2 = r2_score(y_test, predictons)
    MSE = mean_squared_error(y_test, predictons)
    return r2 , MSE , LS
r2_lasso , MSE_lasso, LS = ModelLASSO()

#Wykres współczynników Lasso
@st.experimental_memo()
def MakePlotCoefLasso():
    coef_list = list(zip(LS.feature_names_in_ , LS.coef_))
    coef_list_sorted = sorted(coef_list, key = operator.itemgetter(1), reverse=False)
    feature_name , feature_value = zip(*coef_list_sorted)
    fig = px.bar(y=feature_name, x=feature_value, height=600, title='Wykres wartości współczynników')
    fig.update_layout(xaxis_title='Znaczenie',yaxis_title='Atrybut')
    return fig
fig_lasso_coef_importance = MakePlotCoefLasso()

with st.sidebar:
    selected = option_menu(
        menu_title = "Projekt PAD",
        options=['Cel','Dane', 'Wstępna analiza danych', 'Przygotowywanie danych', 'Modelowanie i ewaluacja', 'Wnioski'],
        icons=['question', 'clipboard-data', 'clipboard-check', 'bar-chart', 'cpu', 'file-diff'],
        menu_icon='pin',
        default_index=0
) 
if selected == 'Cel':
    st.title('Cel')
    st.subheader('Celem projektu jest predykcja ceny wynajmu na podstawie danych z serwisu booking.com')

if selected == 'Dane':
    st.title('Dane')
    st.write('Dane zostały pobrane z serwisu booking.com za pomocą biblioteki selenium. Dane obejmują siedem miejscowości. Rezerwacja dotyczny tygodnia dla dwóch osób. Każdy z rekordów poza paroma podstawowymi atrybutami takimi jak: miejscowość, nazwa obiektu, cena oraz opinie posiadał różne udogodnienia (ponad 700). Wykorzystałem pliki jsona do zapisu danych, które później zagregowałem w Tabele. Tabela po procesie pobierania danych posiadała: 1852 wierszy i 776 kolumn. ')
    st.write(df_original)
    st.write('Scrapowanie był to długi proces. W godzine pobierało około 250-300 rekordów.')

if selected == 'Wstępna analiza danych':
    st.title('Wstępna analiza danych')
    st.write('W zbiorze występuje dużo brakujących wartości. Wynika to ilości udogodnień jakie mają zawarte poszczególne obiekty na stronie. Każdy z obiektów posiada od około 20 do 50 cech dotyczących udogodnień. Posiadanie udogodnienia przez obiekt zostało opisane jako 1. W dalszej częsci braki zostaną zamienione na 0.')
    st.write(f'Ilość kolumn: {len(df_original.columns)}')
    st.write(f'Ilość kategorii: {len(cattegory)}')
    st.write('W danych pojawiły się duplikaty. Przypuszczam, że podczas pobierania zmieniała się kolejność wyświetlania na stronie booking.')

if selected == 'Przygotowywanie danych':
    st.title('Przygotowywanie danych')
    st.write(f'Duplikaty zostały usunięte.')
    st.write(f'Usunięte zostały rekordy z brakującymi opiniami.')
    st.write(f'Rekordy z ceną i opiniami powinny być odpowiednio typem: {int} i {float}. Opinie zostały pobrane z przecinkami zamiast kropek. Ceny posiadały czasem dopisek "zł" i miały przerwy. Dane zostały przetransformowane do odpowienich typów.')
    st.plotly_chart(fig_outliners, use_container_width=True)
    st.write(f'Zostały usunięte rekordy odstające pod względem ceny (0.5% największych wartości).')
    st.write(f'Tabela posiadała wiele kolumn. Wiele z nich dotyczyła tylko kilku rekordów (występowanie 1). Kolumny, które posiadało tylko 0.5% rekordów zostały usunięte.')
    st.write(f'Pozostało {df.shape[1]} kolumn i {df.shape[0]} wierszy.')
    st.write(f'Wartości NaN zostały zastąpione przez 0.')
    st.subheader(f'Tabela po oczyszczeniu:')
    st.write(df)
    st.plotly_chart(fig_group_by_city, use_container_width=True)
    
if selected == 'Modelowanie i ewaluacja':
    st.title('Modelowanie i ewaluacja')
    st.write(f'Z tabeli zostały usunięte kolumny z nazwami obiektów i miejscowości. Atrybut decyzyjny to cena wynajmu (hotel_price). Dane zostały podzielone na testowe i treningowe.')
    st.header('Drzewo regresyjne')
    st.write(f'Pierwszy wykorzystany model to drzewo regresyjne. Głębokość drzewa została ustalona na 4.')
    st.write(f'Wynik R2 dla drzewa to: {round(r2_tree,2)}, średni błąd kwadratowy dla drzewa: {round(MSE_tree,2)}')
    st.write(f'Tak prezentuje się drzewo stworzone przez model. Ograniczyłem głębokość drzewa na rysunku do 2, aby poprawić przejrzystość.')
    st.image(img_graph, use_column_width=True)
    st.header('Regresyjne lasy losowe')
    st.write(f'Kolejny wykorzystany model to regresyjne lasy losowe.')
    st.write(f'Wynik R2 dla lasów losowych to: {round(r2_forest,2)}, średni błąd kwadratowy dla lasów losowych: {round(MSE_forest,2)}')
    st.plotly_chart(fig_forest, use_container_width=True)
    st.subheader(f'Polecam dwukrotnie przybliżyć na największe wartości')
    st.plotly_chart(fig_feat_forest, use_container_width=True)
    st.header('Wnioski')
    st.write("Lasy losowe miały większy wynik R2, dane bardziej dopasowały się do modelu regresji. Ciekawie prezentują się informacje zawarte na wykresie ważności atrybutów. Z wykresu porównującego prawdziwą wartość od predykcji możemy zauważyć, że model nie jest zbyt skuteczny. Można zauważyć pewną zależność. Dla najmniejszych i największych watości model rzeczywiście wyznaczył zauważalnie 'bliższe' wartości niż dla pozostałych wartości (środkowych).")
    st.header('Dodatkowo, z ciekawości nieomawiana metoda LASSO')
    st.write(f'Obliczony najbardziej dopasowany paramatr lambda o wartości: {alpha_best}')
    st.write(f'Wynik R2 dla LASSO to: {round(r2_lasso,2)}, średni błąd kwadratowy dla LASSO: {round(MSE_lasso,2)}')
    st.plotly_chart(fig_lasso_coef_importance, use_container_width=True)

if selected == 'Wnioski':
    st.title('Wnioski')
    st.write(f'Po pobraniu danych szukałem gotowych rozwiązań scrapera. W trzech, które znalazłem, była wykorzystana biblioteka Beautiful Soup. Kolejna implementacja takiego rozwiązania będzie wykonana w tej bibliotece. Powinna być znacząco szybsza.')
    st.write(f'Każdy z obiektów noclegowych posiada inne udogodnienia. Nawet jeśli obiekt oferuje podobną usługe, może mieć zaznaczone inne udogodnienia. Warto również pobrać inne dane takie jak: ilość miejsc, klasa obiektu, jeśli hotel to ile gwiazdek oraz przefiltrować udogodnienia. Można również pobrać dane o odległościach np. od plaży czy centrum.')
    st.write(f'Próbowałem również stworzyć nową tabele na podstawie kategorii (45 zamiast ponad 700) i na jej podstawie dokonać predykcji. Efekt predykcji dla różnych random_state był porównywalny lub gorszy, niż dla wszystkich atrybutów. Zauważalnie spadła przewaga lasów losowych nad drzewami.')