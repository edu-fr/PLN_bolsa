import sys
import datetime
import scipy.stats
from GoogleNews import GoogleNews
import pandas as pd
from newspaper import Config
import yfinance as yf
from newspaper import Article
from os.path import exists
import nltk
import re


def main():
    # Arguments:
    args = sys.argv[1:]
    update_google = args[0]  # True or False
    initial_date = args[1]  # mm/dd/yyyy
    amount_of_news = int(args[2])
    download_dependencies = args[3]  # True or False

    # Se precisar baixar as dependências
    if download_dependencies == "True":
        nltk.download('wordnet')
        nltk.download('punkt')
        nltk.download('stopwords')

    if update_google == "True":
        update_google = True
    else:
        update_google = False

    if update_google:
        print("Atualizando arquivos")
        updateData(update_google, starting_date=initial_date, amount_of_news=amount_of_news)
    elif not exists('export_dataframe.csv'):
        print("Dataframe não encontrado. Atualizando arquivos")
        updateData(update_google, starting_date=initial_date, amount_of_news=amount_of_news)

    final_df = pd.read_csv('export_dataframe.csv', '&')  # Read DATAFRAME from file
    final_df.drop('Unnamed: 0', axis='columns', inplace=True)  # Remove extra column from read process
    preprocess(final_df)  # Format DATAFRAME
    update_sentiment_score(final_df)

    final_news = get_news(string_to_date(final_df['Date'][0]), string_to_date(final_df['Date'][final_df.__len__() - 1]),
                          final_df)

    sentiments = []
    preco_2_dias_atras = []
    preco_1_dia_atras = []
    preco_dia_atual = []
    preco_1_dia_depois = []
    preco_2_dias_depois = []

    for news in final_news:
        sentiments.append(news['Score'].values[0])
        variation = get_variation_from_date(news['Date'].values[0])
        preco_2_dias_atras.append(variation[0])
        preco_1_dia_atras.append(variation[1])
        preco_dia_atual.append(variation[2])
        preco_1_dia_depois.append(variation[3])
        preco_2_dias_depois.append(variation[4])

    # print("Sentiments")
    # print(sentiments)

    print("== Correlacao == ")
    print("Dois dias antes: " + "-")
    print("Um dia antes: " + str(scipy.stats.pearsonr(sentiments, preco_1_dia_atras)[1]))
    print("Dia da notícia: " + str(scipy.stats.pearsonr(sentiments, preco_dia_atual)[1]))
    print("Um dia depois: " + str(scipy.stats.pearsonr(sentiments, preco_1_dia_depois)[1]))
    print("Dois dias depois: " + str(scipy.stats.pearsonr(sentiments, preco_2_dias_depois)[1]))


def get_variation_from_date(news_date):
    news_date = datetime.datetime.strptime(news_date[:10], '%Y-%m-%d')
    one_day = datetime.timedelta(1)
    days = [news_date - one_day - one_day, news_date + one_day + one_day]
    finance_data = yf.download(tickers='PBR', period='2y', interval='1d', progress=False)
    period_closes = finance_data.loc[days[0]:days[1]]['Close']
    current_date = news_date - one_day - one_day
    new_period_closes = []
    for i in range(0, 5):
        if period_closes.get(key=str(current_date)[:10]) is not None:
            new_period_closes.append(period_closes.get(key=str(current_date)[:10]))
        else:
            if i == 0:
                aux_current_date = current_date
                counter = 0
                while 1:
                    if counter > 4:
                        break
                    if period_closes.get(key=str(aux_current_date)[:10]) is not None:
                        new_period_closes.append(period_closes.get(key=str(aux_current_date)[:10]))
                        break
                    else:
                        aux_current_date += one_day
                    counter += 1
            else:
                new_period_closes.append(new_period_closes[i - 1])
        current_date += one_day
    standardized_closes = []
    for i in range(len(new_period_closes)):
        if i == 0:
            standardized_closes.append(0)
        else:
            standardized_closes.append(round(new_period_closes[i] - new_period_closes[i - 1], 3))

    return standardized_closes


def get_news(initial_date, final_date, dataframe):
    news = [get_news_by_date(dataframe, str(initial_date)[:10])]
    current_date = initial_date + datetime.timedelta(1)
    while current_date != final_date + datetime.timedelta(1):
        news.append(get_news_by_date(dataframe, str(current_date)[:10]))
        current_date = current_date + datetime.timedelta(1)
    return news  # Return a list of dataframes


def get_news_by_date(dataframe, date):
    news = dataframe.loc[dataframe['Date'] == date]
    return news.head(1)


def transform_date(date_string):
    return str(datetime.datetime.strptime(date_string[:10], '%Y-%m-%d'))[:10]


def string_to_date(date_string):
    return datetime.datetime.strptime(date_string[:10], '%Y-%m-%d')


def string_to_date_google(date_string):
    replaced = date_string.replace('/', '-')
    return datetime.datetime.strptime(replaced, '%m-%d-%Y')


def date_to_google_news_string(date):
    new_date = str(date)[5:7] + '/' + str(date)[8:10] + '/' + str(date)[0:4]
    return str(new_date)[:10]


def update_sentiment_score(data_base):
    database_sentiment = set_database_sentiment()

    max_value = 10
    min_value = -10
    for i in range(data_base['Article'].__len__()):
        data_base['Score'][i] = scale(evaluate_sentiment(data_base['Article'][i], database_sentiment), (-30, 30), (min_value, max_value))


def scale(value, current_limits, desired_range):
    return ((value - current_limits[0]) / (current_limits[1] - current_limits[0])) * (
                desired_range[1] - desired_range[0]) + desired_range[0]


def updateData(update_google, starting_date, amount_of_news):
    if update_google:
        print("Atualizando arquivo de noticias do google...")
        get_daily_google_news_by_period(starting_date, amount_of_news)
    elif not exists('google_news_df.csv'):
        print("Arquivo com noticias do google nao existe, criando novo arquivo...")
        get_daily_google_news_by_period(starting_date, amount_of_news)
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
        print("Baixando noticia " + str(ind) + "...")
        dict = {}
        article = Article(google_news_dataframe['link'][ind])
        article.download()
        try:
            article.parse()
        except:
            print("Falha em baixar noticia " + str(ind) + ". Prosseguindo...")
            continue
        article.nlp()
        dict['Date'] = str(google_news_dataframe['datetime'][ind])[:10]
        dict['Media'] = google_news_dataframe['media'][ind]
        dict['Title'] = article.title
        dict['Article'] = article.text
        dict['Summary'] = article.summary
        dict['Score'] = 0
        dicts.append(dict)
        print("Concluido!")
    news_df = pd.DataFrame(dicts)
    news_df.to_csv(path_or_buf='export_dataframe.csv', sep='&')  # Exporta para um arquivo com as noticias completas
    print("Dataframe atualizado!")


def get_daily_google_news_by_period(starting_date, amount_of_news):
    current_date = starting_date
    results = []
    while amount_of_news > 0:
        google_news = GoogleNews(start=current_date, end=current_date)
        google_news.setlang('pt')
        google_news.search('Petrobras')
        google_news.getpage(0)
        result = google_news.result()
        results = results + result
        print("Noticia do dia " + str(current_date) + " extraida!")
        current_date = string_to_date_google(current_date) + datetime.timedelta(1)
        current_date = date_to_google_news_string(current_date)
        amount_of_news -= 1
    df = pd.DataFrame(results)
    df.to_csv(path_or_buf='google_news_df.csv', sep='&')  # Cria um arquivo com as noticias do google


def preprocess(final_df):
    language = 'portuguese'
    final_df['Article'] = final_df.apply(lambda row: nltk.word_tokenize(str(row['Article'])), axis=1)
    stopwords = nltk.corpus.stopwords.words(language)
    stopwords = list(set(stopwords))
    final_df['Article'] = final_df.apply(lambda row: normalize(row['Article'], stopwords), axis=1)


def normalize(words, stopwords):
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = remove_stopwords(words, stopwords)
    return ' '.join(words)


def remove_stopwords(words, stopwords):
    new_words = []
    for word in words:
        if word not in stopwords:
            new_words.append(word)
    return new_words


def to_lowercase(words):
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words


def remove_punctuation(words):
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
    text = str(text)
    score = 0
    num_words = 0
    num_iterated = 0
    for word in text.split():
        num_words = num_words + 1
        if word in database_sentiment:
            score = score + int(database_sentiment[word])
            num_iterated = num_iterated + 1
    return score


main()
