import copy
import sys
import datetime

from GoogleNews import GoogleNews
import pandas as pd
from newspaper import Config
import yfinance as yf
from newspaper import Article
from os.path import exists
import nltk
import re


#  Primeiro criamos uma base de treino usando noticias aleatorias -> Como avaliar elas como boas ou ruins? Usamos
#  a classificação pronta do SentiLex-lem-PT02.txt
# A partir disso temos uma lista de noticias classificadas como boas ou ruins.


def main():
    # Arguments:
    args = sys.argv[1:]
    update_google = args[0]  # True or False
    initial_date = args[1]  # mm/dd/yyyy
    final_date = args[2]  # mm/dd/yyyy
    pages = int(args[3])  # int (ex: 5)
    period = args[4]  # str (ex: '5d')
    first_time = args[5]  # True or False

    # Quando for rodar pela primeira vez
    if first_time == "True":
        nltk.download('wordnet')
        nltk.download('punkt')
        nltk.download('stopwords')

    if update_google == "True":
        update_google = True
    else:
        update_google = False

    if update_google:
        print("Atualizando arquivos por opção")
        updateData(update_google, starting_date=initial_date, ending_date=final_date, pages=pages)
    elif not exists('export_dataframe.csv'):
        print("Dataframe final não encontrado. Atualizando arquivos")
        updateData(update_google, starting_date=initial_date, ending_date=final_date, pages=pages)

    final_df = pd.read_csv('export_dataframe.csv', '&')         # Read DATAFRAME from file
    final_df.drop('Unnamed: 0', axis='columns', inplace=True)   # Remove extra column from read process
    preprocess(final_df)                                        # Format DATAFRAME
    update_sentiment_score(final_df)
    # print_data_frame(final_df)

    get_news(string_to_date(final_df['Date'][180]), string_to_date(final_df['Date'][10]), final_df)
    quit(1)
    # FOR
    # variation = get_variation_from_date(final_df['Date'][130])
    variation = get_variation_from_date_range(string_to_date(final_df['Date'][220]), string_to_date(final_df['Date'][10]))


    # print_news_score(final_df)

    # print(finance_data)

def get_variation_from_date_range(first_day, last_day):
    print("First " + str(first_day))
    print("last " + str(last_day))
    first_date = datetime.datetime.strptime(first_day[:10], '%Y-%m-%d')
    last_date = datetime.datetime.strptime(last_day[:10], '%Y-%m-%d')
    one_day = datetime.timedelta(1)
    days = [first_date - one_day - one_day, last_date + one_day + one_day]
    finance_data = yf.download(tickers='PBR', period='2y', interval='1d')
    period_closes = finance_data.loc[days[0]:days[1]]['Close']
    standardized_closes = []
    for i in range(len(period_closes) - 1):
        standardized_closes.append(round(period_closes[i + 1] - period_closes[i], 3))

    # print(period_closes)
    #
    print(standardized_closes)
    return standardized_closes


def get_variation_from_date(news_date):
    news_date = datetime.datetime.strptime(news_date[:10], '%Y-%m-%d')
    one_day = datetime.timedelta(1)
    days = [news_date - one_day - one_day, news_date + one_day + one_day]

    finance_data = yf.download(tickers='PBR', period='2y', interval='1d')
    period_closes = finance_data.loc[days[0]:days[1]]['Close']
    standardized_closes = []
    for i in range(len(period_closes) - 1):
        standardized_closes.append(round(period_closes[i + 1] - period_closes[i], 3))

    # print(period_closes)
    #
    # print(standardized_closes)
    return standardized_closes


def get_news(initial_date, final_date, dataframe):
    print("DATAS: inicial: " + str(initial_date) + "\nfinal: " + str(final_date))
    dates = [initial_date]
    news = [get_news_by_date(dataframe, str(initial_date)[:10])]
    current_date = initial_date + datetime.timedelta(1)
    while current_date != final_date + datetime.timedelta(1):
        news.append(get_news_by_date(dataframe, str(current_date)[:10]))
        current_date = current_date + datetime.timedelta(1)
        print(str(current_date))
    return news                                           # Return a list of dataframes


def get_news_by_date(dataframe, date):
    news = dataframe.loc[dataframe['Date'] == date]
    return news.head(1)


def transform_date(date_string):
    return str(datetime.datetime.strptime(date_string[:10], '%Y-%m-%d'))[:10]


def string_to_date(date_string):
    return datetime.datetime.strptime(date_string[:10], '%Y-%m-%d')


def update_sentiment_score(data_base):
    database_sentiment = set_database_sentiment()

    score_list = []
    for i in data_base['Article']:
        score_list.append(evaluate_sentiment(i, database_sentiment))
    max_value = 10
    min_value = -10

    x = 0
    for j in data_base['Article']:
        data_base['Score'][x] = scale(score_list[x], (min_value, max_value), (-10, 10))
        x = x + 1


def print_news_score(news_df):
    counter = 0
    for j in news_df['Score']:
        print(str(j) + "  Noticia: " + news_df['Title'][counter])
        counter = counter + 1


def print_data_frame(dataframe):
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    print(dataframe)


def scale(value, current_limits, desired_range):
    return ((value - current_limits[0]) / (current_limits[1] - current_limits[0])) * (desired_range[1] - desired_range[0]) + desired_range[0]


def updateData(update_google, starting_date, ending_date, pages):
    print("Atualizando dados!")
    if update_google:
        print("Atualizando google por escolha")
        updateGoogle(starting_date, ending_date, pages)
    elif not exists('google_news_df.csv'):
        print("Arquivo com noticias do google nao existe, criando novo arquivo")
        updateGoogle(starting_date, ending_date, pages)

    google_news_dataframe = pd.read_csv(filepath_or_buffer="google_news_df.csv",
                                        sep='&')  # Lê o arquivo com as notícias do google
    google_news_dataframe.drop('Unnamed: 0', axis='columns', inplace=True)

    dicts = []

    # Configure settings to download articles
    user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) ' \
                 'Chrome/91.0.4472.124  Safari/537.36 '
    config = Config()
    config.browser_user_agent = user_agent

    print("Iniciando download das noticias completas: ")
    for ind in google_news_dataframe.index:
        print("Noticia " + str(ind))
        dict = {}
        article = Article(google_news_dataframe['link'][ind])
        article.download()
        try:
            article.parse()

        except:
            continue

        article.nlp()
        dict['Date'] = str(google_news_dataframe['datetime'][ind])[:10]
        dict['Media'] = google_news_dataframe['media'][ind]
        dict['Title'] = article.title
        dict['Article'] = article.text
        dict['Summary'] = article.summary
        dict['Score'] = 0
        dicts.append(dict)
    news_df = pd.DataFrame(dicts)
    news_df.to_csv(path_or_buf='export_dataframe.csv', sep='&')  # Exporta para um arquivo com as noticias completas


def updateGoogle(starting_date, ending_date, pages):
    google_news = GoogleNews(start=starting_date, end=ending_date)
    google_news.setlang('pt')
    google_news.search('Petrobras')
    print("Recuperando notícias atraves da API do Google News...")
    for i in range(2, pages):
        google_news.getpage(i)
        result = google_news.result()
        df = pd.DataFrame(result)
        print("Noticias da pagina " + str(i) + " extraidas.")

    pd.set_option("display.max_rows", None, "display.max_columns", None)  # Altera a visualização do dataframe
    df.to_csv(path_or_buf='google_news_df.csv', sep='&')  # Cria um arquivo com as noticias do google


def preprocess(final_df):
    language = 'portuguese'
    final_df['Article'] = final_df.apply(lambda row: nltk.word_tokenize(str(row['Article'])), axis=1)
    stopwords = nltk.corpus.stopwords.words(language)
    stopwords = list(set(stopwords))
    final_df['Article'] = final_df.apply(lambda row: normalize(row['Article'], stopwords), axis=1)

    # print(final_df['Article'][0])


def normalize(words, stopwords):
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = remove_stopwords(words, stopwords)

    return ' '.join(words)


def remove_stopwords(words, stopwords):
    """Remover as Stopwords das palavras tokenizadas"""
    new_words = []
    for word in words:
        if word not in stopwords:
            new_words.append(word)
    return new_words


def to_lowercase(words):
    """converter todos os caracteres para lowercase"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words


def remove_punctuation(words):
    """remover pontuação"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)

    return new_words


def set_database_sentiment():
    sentilexpt = open('SentiLex-lem-PT02.txt')

    dic_palavra_polaridade = {}
    for i in sentilexpt.readlines():
        pos_ponto = i.find('.')
        palavra = (i[:pos_ponto])
        pol_pos = i.find('POL')
        polaridade = (i[pol_pos + 7:pol_pos + 9]).replace(';', '')
        dic_palavra_polaridade[palavra] = polaridade

    return dic_palavra_polaridade


def evaluate_sentiment(text, database_sentiment):
    score = 0
    text = str(text)
    num_palavras = 0
    num_iterado = 0

    for word in text.split():
        num_palavras = num_palavras + 1
        if word in database_sentiment:
            score = score + int(database_sentiment[word])
            num_iterado = num_iterado + 1

    # print( "Numero de palavras: " + str(num_palavras) + "  Numero iterado: " + str(num_iterado))
    return score


main()
